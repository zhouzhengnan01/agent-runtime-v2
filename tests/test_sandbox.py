from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient
import pytest

from app.core.sandbox.config import SandboxConfig, load_sandbox_config
from app.core.sandbox import config as sandbox_config_module
from app.core.sandbox.env_cache import SkillEnvironmentCache, normalized_requirements, requirements_hash_for_text
from app.core.sandbox.policy import load_sandbox_policy
from app.core.sandbox.status import get_sandbox_status
from app.core.skills import SkillRegistry
from app.main import create_app
from app.api import skills as skills_api


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def test_sandbox_status_defaults_to_lightweight_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SANDBOX_PROVIDER", raising=False)
    monkeypatch.delenv("SANDBOX_SKILLS", raising=False)

    status = asyncio.run(get_sandbox_status())

    assert status.provider == "local"
    assert status.configured is True
    assert status.server_url is None
    assert status.routing_mode == "selective"
    assert status.sandboxed_skills == ["data-auto-annotation", "drawio-generation", "excel-generation", "pptx-generation"]
    assert status.skill_profiles["data-auto-annotation"] == "python-skill"
    assert status.skill_profiles["drawio-generation"] == "drawio"
    assert status.profiles["python-skill"].image == "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.2"
    assert status.profiles["drawio"].image == "jetlinks/opensandbox-drawio:0.1.0"
    assert status.executor_enabled is False


def test_sandbox_config_reads_opensandbox_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_PROVIDER", "opensandbox")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "127.0.0.1:19090")
    monkeypatch.setenv("OPENSANDBOX_PROTOCOL", "http")
    monkeypatch.setenv("OPENSANDBOX_IMAGE", "example/opensandbox:test")
    monkeypatch.setenv("SANDBOX_SKILLS", "drawio-generation,pptx-generation")
    monkeypatch.setenv("SKILL_ENV_CACHE_ENABLED", "true")

    config = load_sandbox_config()

    assert config.provider == "opensandbox"
    assert config.opensandbox_server_url == "http://127.0.0.1:19090"
    assert config.opensandbox_image == "example/opensandbox:test"
    assert config.sandboxed_skills == ("drawio-generation", "pptx-generation")
    assert config.should_use_sandbox("drawio-generation") is True
    assert config.should_use_sandbox("behavior-detection") is False
    assert config.skill_env_cache_enabled is True


def test_sandbox_config_reads_local_subprocess_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_PROVIDER", "local_subprocess")
    monkeypatch.setenv("SANDBOX_EXECUTOR_ENABLED", "true")

    config = load_sandbox_config()

    assert config.provider == "local_subprocess"
    assert config.executor_enabled is True


def test_sandbox_config_reads_runtime_json_and_local_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_dir = tmp_path / "config" / "sandbox"
    sandbox_dir.mkdir(parents=True)
    (sandbox_dir / "runtime.json").write_text(
        json_dumps(
            {
                "provider": "opensandbox",
                "opensandbox_domain": "127.0.0.1:19090",
                "executor_enabled": False,
                "skill_env_cache": {
                    "enabled": True,
                    "image_prefix": "default-prefix",
                    "max_images": 10,
                    "torch_base_image": "torch-default",
                },
            }
        ),
        encoding="utf-8",
    )
    (sandbox_dir / "runtime.local.json").write_text(
        json_dumps(
            {
                "executor_enabled": True,
                "skill_env_cache": {
                    "image_prefix": "local-prefix",
                    "max_images": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sandbox_config_module.Path, "resolve", lambda self: tmp_path / "app/core/sandbox/config.py")

    config = load_sandbox_config()

    assert config.provider == "opensandbox"
    assert config.opensandbox_domain == "127.0.0.1:19090"
    assert config.executor_enabled is True
    assert config.skill_env_cache_enabled is True
    assert config.skill_env_cache_image_prefix == "local-prefix"
    assert config.skill_env_cache_max_images == 5
    assert config.skill_env_cache_torch_base_image == "torch-default"


def test_sandbox_config_environment_overrides_runtime_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_dir = tmp_path / "config" / "sandbox"
    sandbox_dir.mkdir(parents=True)
    (sandbox_dir / "runtime.json").write_text(
        json_dumps({"skill_env_cache": {"enabled": False, "image_prefix": "file-prefix"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sandbox_config_module.Path, "resolve", lambda self: tmp_path / "app/core/sandbox/config.py")
    monkeypatch.setenv("SKILL_ENV_CACHE_ENABLED", "true")
    monkeypatch.setenv("SKILL_ENV_CACHE_IMAGE_PREFIX", "env-prefix")

    config = load_sandbox_config()

    assert config.skill_env_cache_enabled is True
    assert config.skill_env_cache_image_prefix == "env-prefix"


def test_requirements_normalization_ignores_comments_and_order() -> None:
    first = """
requests==2.32.3
openpyxl==3.1.5
"""
    second = """
# excel skill
openpyxl==3.1.5

requests==2.32.3  # http client
"""

    assert normalized_requirements(first) == "openpyxl==3.1.5\nrequests==2.32.3"
    assert requirements_hash_for_text(first) == requirements_hash_for_text(second)


def test_skill_environment_cache_uses_base_image_without_requirements(tmp_path: Path) -> None:
    cache = SkillEnvironmentCache(tmp_path)
    package_root = tmp_path / "plugin" / "skills" / "empty"
    package_root.mkdir(parents=True)

    env = cache.prepare(
        skill_name="empty",
        package_root=package_root,
        profile=load_sandbox_policy(SandboxConfig()).profiles["office"],
        config=SandboxConfig(skill_env_cache_enabled=True),
    )

    assert env.status == "base"
    assert env.requirements_hash is None
    assert env.image == "jetlinks/opensandbox-office:0.1.0"


def test_skill_environment_cache_reuses_ready_dependency_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SkillEnvironmentCache(tmp_path)
    package_root = tmp_path / "plugin" / "skills" / "deps"
    package_root.mkdir(parents=True)
    (package_root / "requirements.txt").write_text("requests==2.32.3\n", encoding="utf-8")
    req_hash = requirements_hash_for_text("requests==2.32.3\n")
    image = f"jetlinks-python-skill-deps:{req_hash[:16]}"
    cache.cache_dir.mkdir(parents=True)
    cache.index_path.write_text(
        json_dumps(
            {
                req_hash: {
                    "status": "ready",
                    "image": image,
                    "base_image": "jetlinks/opensandbox-office:0.1.0",
                    "skills": ["old"],
                    "last_used_at": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cache, "_image_exists", lambda _image: True)

    env = cache.prepare(
        skill_name="new",
        package_root=package_root,
        profile=load_sandbox_policy(SandboxConfig()).profiles["office"],
        config=SandboxConfig(skill_env_cache_enabled=True),
    )

    assert env.status == "ready"
    assert env.image == image
    index = cache._read_index()
    assert index[req_hash]["skills"] == ["new", "old"]


def test_sandbox_policy_resolves_skill_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SANDBOX_SKILLS", raising=False)
    config = SandboxConfig(provider="opensandbox", executor_enabled=True)
    policy = load_sandbox_policy(config)

    drawio = policy.resolve("drawio-generation", config)
    pptx = policy.resolve("pptx-generation", config)
    behavior = policy.resolve("behavior-detection", config)

    assert drawio.eligible is True
    assert drawio.use_sandbox is True
    assert drawio.profile_name == "drawio"
    assert drawio.profile is not None
    assert drawio.profile.python_requirements == ["pillow>=11.0.0", "lxml>=5.3.0"]
    assert pptx.profile_name == "office"
    assert pptx.profile is not None
    assert "libreoffice" in pptx.profile.system_packages
    assert behavior.eligible is False
    assert behavior.use_sandbox is False


def test_skill_manifest_declares_schema_and_sandbox_profile() -> None:
    registry = SkillRegistry()

    drawio = registry.get("drawio-generation")
    behavior = registry.get("behavior-detection")

    assert drawio.input_schema is not None
    assert drawio.input_schema["type"] == "object"
    assert drawio.sandbox.enabled is True
    assert drawio.sandbox.profile == "drawio"
    assert drawio.sandbox.adapter_command is not None
    assert behavior.sandbox.enabled is False


def test_sandbox_policy_allowlist_filters_profiles() -> None:
    config = SandboxConfig(provider="opensandbox", executor_enabled=True, sandboxed_skills=("drawio-generation",))
    policy = load_sandbox_policy(config)

    drawio = policy.resolve("drawio-generation", config)
    pptx = policy.resolve("pptx-generation", config)

    assert drawio.use_sandbox is True
    assert pptx.eligible is False
    assert "SANDBOX_SKILLS" in pptx.reason


def test_sandbox_policy_keeps_local_until_executor_enabled() -> None:
    config = SandboxConfig(provider="opensandbox", executor_enabled=False)
    policy = load_sandbox_policy(config)

    decision = policy.resolve("drawio-generation", config)

    assert decision.eligible is True
    assert decision.use_sandbox is False
    assert decision.execution_mode == "local"
    assert "executor is not enabled" in decision.reason


def test_sandbox_policy_uses_local_subprocess_when_enabled() -> None:
    config = SandboxConfig(provider="local_subprocess", executor_enabled=True)
    policy = load_sandbox_policy(config)

    decision = policy.resolve("drawio-generation", config)

    assert decision.eligible is True
    assert decision.use_sandbox is True
    assert decision.execution_mode == "sandbox"
    assert decision.provider == "local_subprocess"
    assert "local subprocess" in decision.reason


def test_sandbox_global_fallback_false_overrides_skill_default() -> None:
    config = SandboxConfig(provider="opensandbox", executor_enabled=True, fallback_to_local=False)
    policy = load_sandbox_policy(config)

    decision = policy.resolve("drawio-generation", config)

    assert decision.use_sandbox is True
    assert decision.fallback_to_local is False


def test_opensandbox_status_reports_healthy_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.timeout = kwargs["timeout"]

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr("app.core.sandbox.status.is_opensandbox_sdk_installed", lambda: True)
    monkeypatch.setattr("app.core.sandbox.status.httpx.AsyncClient", FakeAsyncClient)

    status = asyncio.run(
        get_sandbox_status(
            SandboxConfig(
                provider="opensandbox",
                opensandbox_domain="127.0.0.1:19090",
                status_timeout_seconds=0.2,
            )
        )
    )

    assert status.provider == "opensandbox"
    assert status.configured is True
    assert status.available is True
    assert status.server_url == "http://127.0.0.1:19090"
    assert calls == ["http://127.0.0.1:19090/health"]


def test_opensandbox_status_reports_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            self.timeout = kwargs["timeout"]

        async def __aenter__(self) -> "FailingAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.core.sandbox.status.is_opensandbox_sdk_installed", lambda: True)
    monkeypatch.setattr("app.core.sandbox.status.httpx.AsyncClient", FailingAsyncClient)

    status = asyncio.run(get_sandbox_status(SandboxConfig(provider="opensandbox")))

    assert status.provider == "opensandbox"
    assert status.configured is True
    assert status.available is False
    assert status.sdk_installed is True
    assert "not reachable" in status.message


def test_sandbox_status_api_defaults_to_lightweight_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SANDBOX_PROVIDER", raising=False)
    client = TestClient(create_app())

    response = client.get("/api/sandbox/status")

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "local"
    assert data["routing_mode"] == "selective"
    assert data["sandboxed_skills"] == ["data-auto-annotation", "drawio-generation", "excel-generation", "pptx-generation"]
    assert data["skill_profiles"]["data-auto-annotation"] == "python-skill"
    assert data["skill_profiles"]["pptx-generation"] == "office"
    assert data["profiles"]["office"]["image"] == "jetlinks/opensandbox-office:0.1.0"
    assert data["executor_enabled"] is False


def test_skills_api_exposes_manifest_for_platform() -> None:
    client = TestClient(create_app())

    response = client.get("/api/skills/drawio-generation")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "drawio-generation"
    assert data["input_schema"]["type"] == "object"
    assert data["output_schema"]["type"] == "object"
    assert data["sandbox"]["enabled"] is True
    assert data["sandbox"]["profile"] == "drawio"
    assert data["editable"] is True
    assert data["manifest"]["name"] == "drawio-generation"
    assert Path(data["source_path"]).parts[-2:] == ("skills", "drawio-generation.json")
    assert '"name"' in data["source_text"]


def test_skills_api_updates_manifest_without_touching_real_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config" / "skills"
    config_dir.mkdir(parents=True)
    manifest_path = config_dir / "demo-skill.json"
    manifest_path.write_text(
        """
{
  "name": "demo-skill",
  "description": "Original skill",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": ["draft"],
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {
    "enabled": false,
    "profile": null,
    "request_schema_version": "skill-run.v1",
    "fallback_to_local": true
  }
}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    client = TestClient(create_app())

    response = client.put(
        "/api/skills/demo-skill",
        json={
            "name": "ignored-name",
            "description": "Edited skill",
            "output_kind": "presentation",
            "generation": False,
            "quality_template": ["review", "publish"],
            "input_schema": {"type": "object", "additionalProperties": True},
            "output_schema": {"type": "object"},
            "sandbox": {
                "enabled": True,
                "profile": "office",
                "request_schema_version": "skill-run.v1",
                "fallback_to_local": False,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "demo-skill"
    assert data["description"] == "Edited skill"
    assert data["output_kind"] == "presentation"
    assert data["generation"] is False
    assert data["quality_template"] == ["review", "publish"]
    assert data["sandbox"]["enabled"] is True
    assert data["sandbox"]["profile"] == "office"
    assert data["sandbox"]["fallback_to_local"] is False

    saved = manifest_path.read_text(encoding="utf-8")
    assert '"name": "demo-skill"' in saved
    assert '"description": "Edited skill"' in saved


def test_skills_api_update_requires_admin_token_when_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNTIME_API_TOKEN", "admin-secret")
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    client = TestClient(create_app())
    payload = {
        "description": "Protected skill",
        "output_kind": "markdown",
        "generation": True,
        "quality_template": ["summary"],
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "sandbox": {"enabled": False, "profile": None, "request_schema_version": "skill-run.v1"},
    }

    denied = client.put("/api/skills/protected-skill", json=payload)
    allowed = client.put(
        "/api/skills/protected-skill",
        headers={"Authorization": "Bearer admin-secret"},
        json=payload,
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json()["name"] == "protected-skill"


def test_skills_api_creates_new_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    client = TestClient(create_app())

    response = client.put(
        "/api/skills/custom-summary",
        json={
            "description": "Custom summary skill",
            "output_kind": "markdown",
            "generation": True,
            "quality_template": ["summary"],
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "sandbox": {"enabled": False, "profile": None, "request_schema_version": "skill-run.v1"},
        },
    )

    assert response.status_code == 200
    assert response.json()["name"] == "custom-summary"
    assert (tmp_path / "config" / "skills" / "custom-summary.json").is_file()
