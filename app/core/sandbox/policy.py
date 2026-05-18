from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.sandbox.config import (
    DEFAULT_OPENSANDBOX_IMAGE,
    SandboxConfig,
    SandboxProvider,
    load_sandbox_config,
)
from app.core.skills import SkillRegistry


ExecutionMode = Literal["local", "sandbox"]

DEFAULT_POLICY_DATA: dict[str, object] = {
    "profiles": {
        "drawio": {
            "description": "Draw.io XML and PNG rendering profile.",
            "image": "jetlinks/opensandbox-drawio:0.1.0",
            "python_requirements": ["pillow>=11.0.0", "lxml>=5.3.0"],
            "system_packages": ["fonts-noto-cjk"],
            "requirements_file": "skills/drawio/requirements.txt",
            "command": "python /opt/jetlinks/skills/run_skill.py --request {request_path} --outputs {outputs_dir}",
            "timeout_seconds": 300,
            "runtime_install": False,
        },
        "office": {
            "description": "Office document generation and preview conversion profile.",
            "image": "jetlinks/opensandbox-office:0.1.0",
            "python_requirements": [
                "python-pptx>=1.0.2",
                "openpyxl>=3.1.5",
                "xlsxwriter>=3.2.0",
                "pillow>=11.0.0",
            ],
            "system_packages": ["libreoffice", "fonts-noto-cjk"],
            "requirements_file": "skills/office/requirements.txt",
            "command": "python /opt/jetlinks/skills/run_skill.py --request {request_path} --outputs {outputs_dir}",
            "timeout_seconds": 600,
            "runtime_install": False,
        },
        "data-code": {
            "description": "Python data analysis and code execution profile.",
            "image": "jetlinks/opensandbox-code:0.1.0",
            "python_requirements": ["pandas>=2.2.0", "matplotlib>=3.9.0"],
            "system_packages": ["fonts-noto-cjk"],
            "requirements_file": "skills/code/requirements.txt",
            "command": "python /opt/jetlinks/skills/run_skill.py --request {request_path} --outputs {outputs_dir}",
            "timeout_seconds": 600,
            "runtime_install": False,
        },
    },
}


class SandboxProfile(BaseModel):
    name: str = ""
    description: str = ""
    image: str = DEFAULT_OPENSANDBOX_IMAGE
    python_requirements: list[str] = Field(default_factory=list)
    requirements_file: str | None = None
    system_packages: list[str] = Field(default_factory=list)
    command: str = "python /opt/jetlinks/skills/run_skill.py --request {request_path} --outputs {outputs_dir}"
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=300, gt=0)
    runtime_install: bool = False
    fallback_to_local: bool | None = None


class SandboxDecision(BaseModel):
    skill_name: str
    eligible: bool
    use_sandbox: bool
    execution_mode: ExecutionMode
    provider: SandboxProvider
    profile_name: str | None = None
    profile: SandboxProfile | None = None
    request_schema_version: str = "skill-run.v1"
    fallback_to_local: bool
    reason: str


class SandboxPolicy(BaseModel):
    profiles: dict[str, SandboxProfile] = Field(default_factory=dict)
    skill_profiles: dict[str, str] = Field(default_factory=dict)
    skill_sandbox: dict[str, dict[str, object]] = Field(default_factory=dict)
    source: str = "defaults"
    config_error: str | None = None

    @property
    def sandboxed_skills(self) -> list[str]:
        return sorted(self.skill_profiles)

    def active_skill_profiles(self, config: SandboxConfig) -> dict[str, str]:
        if config.sandboxed_skills is None:
            return dict(self.skill_profiles)
        allowed = set(config.sandboxed_skills)
        return {
            skill_name: profile_name
            for skill_name, profile_name in self.skill_profiles.items()
            if skill_name in allowed
        }

    def resolve(self, skill_name: str, config: SandboxConfig | None = None) -> SandboxDecision:
        sandbox_config = config or load_sandbox_config()
        active_mappings = self.active_skill_profiles(sandbox_config)
        profile_name = active_mappings.get(skill_name)
        if profile_name is None:
            has_mapping = skill_name in self.skill_profiles
            reason = (
                "Skill is not enabled by SANDBOX_SKILLS; use local execution."
                if has_mapping
                else "No sandbox profile configured for skill; use local execution."
            )
            return SandboxDecision(
                skill_name=skill_name,
                eligible=False,
                use_sandbox=False,
                execution_mode="local",
                provider=sandbox_config.provider,
                fallback_to_local=sandbox_config.fallback_to_local,
                reason=reason,
            )

        profile = self.profiles.get(profile_name)
        if profile is None:
            skill_sandbox = self.skill_sandbox.get(skill_name, {})
            return SandboxDecision(
                skill_name=skill_name,
                eligible=False,
                use_sandbox=False,
                execution_mode="local",
                provider=sandbox_config.provider,
                profile_name=profile_name,
                request_schema_version=_request_schema_version(skill_sandbox),
                fallback_to_local=sandbox_config.fallback_to_local,
                reason=f"Sandbox profile {profile_name!r} is not defined; use local execution.",
            )

        skill_sandbox = self.skill_sandbox.get(skill_name, {})
        request_schema_version = _request_schema_version(skill_sandbox)
        effective_profile = _effective_profile(profile, skill_sandbox)
        fallback_to_local = _fallback_to_local(sandbox_config, profile, skill_sandbox)

        if sandbox_config.provider == "local":
            return SandboxDecision(
                skill_name=skill_name,
                eligible=True,
                use_sandbox=False,
                execution_mode="local",
                provider=sandbox_config.provider,
                profile_name=profile_name,
                profile=effective_profile,
                request_schema_version=request_schema_version,
                fallback_to_local=fallback_to_local,
                reason="Sandbox profile matched, but SANDBOX_PROVIDER is local; use local execution.",
            )

        if not sandbox_config.executor_enabled:
            return SandboxDecision(
                skill_name=skill_name,
                eligible=True,
                use_sandbox=False,
                execution_mode="local",
                provider=sandbox_config.provider,
                profile_name=profile_name,
                profile=effective_profile,
                request_schema_version=request_schema_version,
                fallback_to_local=fallback_to_local,
                reason="Sandbox profile matched, but sandbox executor is not enabled; use local execution.",
            )

        if sandbox_config.provider == "local_subprocess":
            return SandboxDecision(
                skill_name=skill_name,
                eligible=True,
                use_sandbox=True,
                execution_mode="sandbox",
                provider=sandbox_config.provider,
                profile_name=profile_name,
                profile=effective_profile,
                request_schema_version=request_schema_version,
                fallback_to_local=fallback_to_local,
                reason="Sandbox profile matched; use local subprocess execution.",
            )

        return SandboxDecision(
            skill_name=skill_name,
            eligible=True,
            use_sandbox=True,
            execution_mode="sandbox",
            provider=sandbox_config.provider,
            profile_name=profile_name,
            profile=effective_profile,
            request_schema_version=request_schema_version,
            fallback_to_local=fallback_to_local,
            reason="Sandbox profile matched; use OpenSandbox execution.",
        )


def load_sandbox_policy(
    config: SandboxConfig | None = None,
    root_dir: Path | None = None,
    skill_registry: SkillRegistry | None = None,
) -> SandboxPolicy:
    sandbox_config = config or load_sandbox_config()
    root = root_dir or Path(__file__).resolve().parents[3]
    if sandbox_config.profile_config_path is None:
        policy = _policy_from_data(DEFAULT_POLICY_DATA, source="defaults")
        return _with_skill_manifests(policy, skill_registry or SkillRegistry(root), sandbox_config)

    path = Path(sandbox_config.profile_config_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        policy = _policy_from_data(DEFAULT_POLICY_DATA, source=str(path), config_error=f"Profile config not found: {path}")
        return _with_skill_manifests(policy, skill_registry or SkillRegistry(root), sandbox_config)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        policy = _policy_from_data(data, source=str(path))
        return _with_skill_manifests(policy, skill_registry or SkillRegistry(root), sandbox_config)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        policy = _policy_from_data(DEFAULT_POLICY_DATA, source=str(path), config_error=str(exc))
        return _with_skill_manifests(policy, skill_registry or SkillRegistry(root), sandbox_config)


def _policy_from_data(data: object, *, source: str, config_error: str | None = None) -> SandboxPolicy:
    if not isinstance(data, dict):
        raise ValueError("Sandbox profile config must be a JSON object.")

    raw_profiles = data.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError("Sandbox profile config field 'profiles' must be an object.")

    profiles: dict[str, SandboxProfile] = {}
    for name, raw_profile in raw_profiles.items():
        if not isinstance(name, str) or not isinstance(raw_profile, dict):
            continue
        profile_data: dict[str, Any] = dict(raw_profile)
        profile_data.setdefault("name", name)
        profiles[name] = SandboxProfile.model_validate(profile_data)

    return SandboxPolicy(
        profiles=profiles,
        skill_profiles={},
        source=source,
        config_error=config_error,
    )


def _with_skill_manifests(
    policy: SandboxPolicy,
    skill_registry: SkillRegistry,
    config: SandboxConfig,
) -> SandboxPolicy:
    skill_profiles: dict[str, str] = {}
    skill_sandbox: dict[str, dict[str, object]] = {}
    for skill in skill_registry.list(executable_only=True):
        payload = skill.sandbox.to_payload()
        profile_name = skill.sandbox.profile
        if not skill.sandbox.enabled or profile_name is None:
            continue
        skill_profiles[skill.name] = profile_name
        skill_sandbox[skill.name] = payload
    return policy.model_copy(update={"skill_profiles": skill_profiles, "skill_sandbox": skill_sandbox})


def _effective_profile(profile: SandboxProfile, skill_sandbox: dict[str, object]) -> SandboxProfile:
    command = _optional_string(skill_sandbox.get("adapter_command"))
    if command is None:
        return profile
    return profile.model_copy(update={"command": command})


def _request_schema_version(skill_sandbox: dict[str, object]) -> str:
    return _optional_string(skill_sandbox.get("request_schema_version")) or "skill-run.v1"


def _fallback_to_local(
    config: SandboxConfig,
    profile: SandboxProfile,
    skill_sandbox: dict[str, object],
) -> bool:
    if not config.fallback_to_local:
        return False
    fallback_to_local = profile.fallback_to_local
    fallback_override = _optional_bool(skill_sandbox.get("fallback_to_local"))
    if fallback_override is not None:
        fallback_to_local = fallback_override
    if fallback_to_local is None:
        return config.fallback_to_local
    return fallback_to_local


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None
