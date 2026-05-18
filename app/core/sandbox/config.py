from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


SandboxProvider = Literal["local", "local_subprocess", "opensandbox"]
SandboxProtocol = Literal["http", "https"]
SandboxRoutingMode = Literal["selective"]

DEFAULT_OPENSANDBOX_IMAGE = (
    "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.2"
)
DEFAULT_PROFILE_CONFIG_PATH = "config/sandbox/profiles.json"
DEFAULT_RUNTIME_CONFIG_PATH = "config/sandbox/runtime.json"


class SandboxConfig(BaseModel):
    provider: SandboxProvider = "local"
    raw_provider: str = "local"
    opensandbox_domain: str = "127.0.0.1:8080"
    opensandbox_protocol: SandboxProtocol = "http"
    opensandbox_api_key: str | None = None
    opensandbox_timeout_seconds: int = 1800
    opensandbox_image: str = DEFAULT_OPENSANDBOX_IMAGE
    status_timeout_seconds: float = Field(default=2.0, gt=0)
    provider_valid: bool = True
    routing_mode: SandboxRoutingMode = "selective"
    sandboxed_skills: tuple[str, ...] | None = None
    fallback_to_local: bool = True
    profile_config_path: str | None = DEFAULT_PROFILE_CONFIG_PATH
    executor_enabled: bool = False
    skill_env_cache_enabled: bool = False
    skill_env_cache_image_prefix: str = "jetlinks-python-skill-deps"
    skill_env_cache_max_images: int = 20
    skill_env_cache_max_bytes: int = 20 * 1024 * 1024 * 1024
    skill_env_cache_build_timeout_seconds: int = 1800
    skill_env_cache_torch_base_image: str = ""

    @property
    def opensandbox_server_url(self) -> str:
        domain = self.opensandbox_domain.removeprefix("http://").removeprefix("https://").rstrip("/")
        return f"{self.opensandbox_protocol}://{domain}"

    def should_use_sandbox(self, skill_name: str) -> bool:
        return self.sandboxed_skills is None or skill_name in self.sandboxed_skills


def load_sandbox_config(env: Mapping[str, str] | None = None) -> SandboxConfig:
    values = env if env is not None else os.environ
    file_config = _load_runtime_config() if env is None else {}
    raw_cache_config = file_config.get("skill_env_cache")
    cache_config: Mapping[str, object] = raw_cache_config if isinstance(raw_cache_config, dict) else {}
    raw_provider = _env_or_config(values, "SANDBOX_PROVIDER", file_config, "provider", "local").strip().lower() or "local"
    provider: SandboxProvider
    if raw_provider == "opensandbox":
        provider = "opensandbox"
    elif raw_provider == "local_subprocess":
        provider = "local_subprocess"
    else:
        provider = "local"
    protocol = _protocol(_env_or_config(values, "OPENSANDBOX_PROTOCOL", file_config, "opensandbox_protocol", "http"))
    return SandboxConfig(
        provider=provider,
        raw_provider=raw_provider,
        provider_valid=raw_provider in {"local", "local_subprocess", "opensandbox"},
        opensandbox_domain=_env_or_config(values, "OPENSANDBOX_DOMAIN", file_config, "opensandbox_domain", "127.0.0.1:8080").strip()
        or "127.0.0.1:8080",
        opensandbox_protocol=protocol,
        opensandbox_api_key=_optional(values.get("OPENSANDBOX_API_KEY")),
        opensandbox_timeout_seconds=_int_value(
            values.get("OPENSANDBOX_TIMEOUT_SECONDS"),
            default=_int_config(file_config.get("opensandbox_timeout_seconds"), 1800),
        ),
        opensandbox_image=_env_or_config(values, "OPENSANDBOX_IMAGE", file_config, "opensandbox_image", DEFAULT_OPENSANDBOX_IMAGE).strip()
        or DEFAULT_OPENSANDBOX_IMAGE,
        status_timeout_seconds=_float_value(
            values.get("OPENSANDBOX_STATUS_TIMEOUT_SECONDS"),
            default=_float_config(file_config.get("status_timeout_seconds"), 2.0),
        ),
        sandboxed_skills=_csv_values(values.get("SANDBOX_SKILLS")) if "SANDBOX_SKILLS" in values else _string_tuple_config(file_config.get("sandboxed_skills")),
        fallback_to_local=_bool_value(values.get("SANDBOX_FALLBACK_TO_LOCAL"), default=_bool_config(file_config.get("fallback_to_local"), True)),
        profile_config_path=_optional(values.get("SANDBOX_PROFILE_CONFIG"))
        or _optional_string(file_config.get("profile_config_path"))
        or DEFAULT_PROFILE_CONFIG_PATH,
        executor_enabled=_bool_value(values.get("SANDBOX_EXECUTOR_ENABLED"), default=_bool_config(file_config.get("executor_enabled"), False)),
        skill_env_cache_enabled=_bool_value(
            values.get("SKILL_ENV_CACHE_ENABLED"),
            default=_bool_config(cache_config.get("enabled"), False),
        ),
        skill_env_cache_image_prefix=_env_or_config(
            values,
            "SKILL_ENV_CACHE_IMAGE_PREFIX",
            cache_config,
            "image_prefix",
            "jetlinks-python-skill-deps",
        ),
        skill_env_cache_max_images=_int_value(
            values.get("SKILL_ENV_CACHE_MAX_IMAGES"),
            default=_int_config(cache_config.get("max_images"), 20),
        ),
        skill_env_cache_max_bytes=_int_value(
            values.get("SKILL_ENV_CACHE_MAX_BYTES"),
            default=_int_config(cache_config.get("max_bytes"), 20 * 1024 * 1024 * 1024),
        ),
        skill_env_cache_build_timeout_seconds=_int_value(
            values.get("SKILL_ENV_CACHE_BUILD_TIMEOUT_SECONDS"),
            default=_int_config(cache_config.get("build_timeout_seconds"), 1800),
        ),
        skill_env_cache_torch_base_image=_env_or_config(
            values,
            "SKILL_ENV_CACHE_TORCH_BASE_IMAGE",
            cache_config,
            "torch_base_image",
            "",
        ).strip(),
    )


def _load_runtime_config(root_dir: Path | None = None) -> dict[str, object]:
    root = root_dir or Path(__file__).resolve().parents[3]
    config_path = root / DEFAULT_RUNTIME_CONFIG_PATH
    data: dict[str, object] = {}
    if config_path.is_file():
        data = _read_json_object(config_path)
    local_path = config_path.with_name("runtime.local.json")
    if local_path.is_file():
        data = _deep_merge(data, _read_json_object(local_path))
    return data


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(base_value, value)
        else:
            merged[key] = value
    return merged


def _env_or_config(
    env: Mapping[str, str],
    env_name: str,
    config: Mapping[str, object],
    config_name: str,
    default: str,
) -> str:
    if env_name in env:
        return str(env.get(env_name) or "")
    value = config.get(config_name)
    return str(value) if value is not None else default


def _protocol(value: str | None) -> SandboxProtocol:
    normalized = (value or "http").strip().lower()
    return "https" if normalized == "https" else "http"


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _int_value(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _int_config(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
    return default


def _float_value(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _float_config(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float | str):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default
    return default


def _csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed


def _string_tuple_config(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, list):
        parsed = tuple(str(item).strip() for item in value if str(item).strip())
        return parsed
    if isinstance(value, str):
        return _csv_values(value)
    return None


def _bool_value(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _bool_config(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default
