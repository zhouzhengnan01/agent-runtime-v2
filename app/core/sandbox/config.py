from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, Field


SandboxProvider = Literal["local", "opensandbox"]
SandboxProtocol = Literal["http", "https"]
SandboxRoutingMode = Literal["selective"]

DEFAULT_OPENSANDBOX_IMAGE = (
    "sandbox-registry.cn-zhangjiakou.cr.aliyuncs.com/opensandbox/code-interpreter:v1.0.2"
)
DEFAULT_PROFILE_CONFIG_PATH = "config/sandbox/profiles.json"


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

    @property
    def opensandbox_server_url(self) -> str:
        domain = self.opensandbox_domain.removeprefix("http://").removeprefix("https://").rstrip("/")
        return f"{self.opensandbox_protocol}://{domain}"

    def should_use_sandbox(self, skill_name: str) -> bool:
        return self.sandboxed_skills is None or skill_name in self.sandboxed_skills


def load_sandbox_config(env: Mapping[str, str] | None = None) -> SandboxConfig:
    values = env if env is not None else os.environ
    raw_provider = values.get("SANDBOX_PROVIDER", "local").strip().lower() or "local"
    provider: SandboxProvider = "opensandbox" if raw_provider == "opensandbox" else "local"
    protocol = _protocol(values.get("OPENSANDBOX_PROTOCOL", "http"))
    return SandboxConfig(
        provider=provider,
        raw_provider=raw_provider,
        provider_valid=raw_provider in {"local", "opensandbox"},
        opensandbox_domain=values.get("OPENSANDBOX_DOMAIN", "127.0.0.1:8080").strip() or "127.0.0.1:8080",
        opensandbox_protocol=protocol,
        opensandbox_api_key=_optional(values.get("OPENSANDBOX_API_KEY")),
        opensandbox_timeout_seconds=_int_value(values.get("OPENSANDBOX_TIMEOUT_SECONDS"), default=1800),
        opensandbox_image=values.get("OPENSANDBOX_IMAGE", DEFAULT_OPENSANDBOX_IMAGE).strip()
        or DEFAULT_OPENSANDBOX_IMAGE,
        status_timeout_seconds=_float_value(values.get("OPENSANDBOX_STATUS_TIMEOUT_SECONDS"), default=2.0),
        sandboxed_skills=_csv_values(values.get("SANDBOX_SKILLS")),
        fallback_to_local=_bool_value(values.get("SANDBOX_FALLBACK_TO_LOCAL"), default=True),
        profile_config_path=_optional(values.get("SANDBOX_PROFILE_CONFIG")) or DEFAULT_PROFILE_CONFIG_PATH,
        executor_enabled=_bool_value(values.get("SANDBOX_EXECUTOR_ENABLED"), default=False),
    )


def _protocol(value: str | None) -> SandboxProtocol:
    normalized = (value or "http").strip().lower()
    return "https" if normalized == "https" else "http"


def _optional(value: str | None) -> str | None:
    if value is None:
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


def _float_value(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed


def _bool_value(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
