from __future__ import annotations

import importlib.util

import httpx
from pydantic import BaseModel

from app.core.sandbox.config import (
    SandboxConfig,
    SandboxProvider,
    SandboxProtocol,
    SandboxRoutingMode,
    load_sandbox_config,
)
from app.core.sandbox.policy import SandboxProfile, load_sandbox_policy


class SandboxStatus(BaseModel):
    provider: SandboxProvider
    configured: bool
    available: bool
    sdk_installed: bool
    domain: str | None
    protocol: SandboxProtocol
    server_url: str | None
    image: str | None
    routing_mode: SandboxRoutingMode
    policy_source: str
    policy_error: str | None
    profiles: dict[str, SandboxProfile]
    skill_profiles: dict[str, str]
    sandboxed_skills: list[str]
    fallback_to_local: bool
    executor_enabled: bool
    message: str


async def get_sandbox_status(config: SandboxConfig | None = None) -> SandboxStatus:
    sandbox_config = config or load_sandbox_config()
    sandbox_policy = load_sandbox_policy(sandbox_config)
    active_skill_profiles = sandbox_policy.active_skill_profiles(sandbox_config)
    sdk_installed = is_opensandbox_sdk_installed()

    if sandbox_config.provider == "local":
        message = "Using local thread workspace."
        if not sandbox_config.provider_valid:
            message = f"Unknown SANDBOX_PROVIDER={sandbox_config.raw_provider!r}; using local thread workspace."
        return SandboxStatus(
            provider="local",
            configured=sandbox_config.provider_valid,
            available=True,
            sdk_installed=sdk_installed,
            domain=None,
            protocol=sandbox_config.opensandbox_protocol,
            server_url=None,
            image=None,
            routing_mode=sandbox_config.routing_mode,
            policy_source=sandbox_policy.source,
            policy_error=sandbox_policy.config_error,
            profiles=sandbox_policy.profiles,
            skill_profiles=active_skill_profiles,
            sandboxed_skills=sorted(active_skill_profiles),
            fallback_to_local=sandbox_config.fallback_to_local,
            executor_enabled=sandbox_config.executor_enabled,
            message=message,
        )

    server_url = sandbox_config.opensandbox_server_url
    if not sdk_installed:
        return SandboxStatus(
            provider="opensandbox",
            configured=False,
            available=False,
            sdk_installed=False,
            domain=sandbox_config.opensandbox_domain,
            protocol=sandbox_config.opensandbox_protocol,
            server_url=server_url,
            image=sandbox_config.opensandbox_image,
            routing_mode=sandbox_config.routing_mode,
            policy_source=sandbox_policy.source,
            policy_error=sandbox_policy.config_error,
            profiles=sandbox_policy.profiles,
            skill_profiles=active_skill_profiles,
            sandboxed_skills=sorted(active_skill_profiles),
            fallback_to_local=sandbox_config.fallback_to_local,
            executor_enabled=sandbox_config.executor_enabled,
            message="OpenSandbox SDK is not installed.",
        )

    available, message = await _probe_opensandbox_health(sandbox_config)
    return SandboxStatus(
        provider="opensandbox",
        configured=True,
        available=available,
        sdk_installed=True,
        domain=sandbox_config.opensandbox_domain,
        protocol=sandbox_config.opensandbox_protocol,
        server_url=server_url,
        image=sandbox_config.opensandbox_image,
        routing_mode=sandbox_config.routing_mode,
        policy_source=sandbox_policy.source,
        policy_error=sandbox_policy.config_error,
        profiles=sandbox_policy.profiles,
        skill_profiles=active_skill_profiles,
        sandboxed_skills=sorted(active_skill_profiles),
        fallback_to_local=sandbox_config.fallback_to_local,
        executor_enabled=sandbox_config.executor_enabled,
        message=message,
    )


def is_opensandbox_sdk_installed() -> bool:
    return importlib.util.find_spec("opensandbox") is not None


async def _probe_opensandbox_health(config: SandboxConfig) -> tuple[bool, str]:
    url = f"{config.opensandbox_server_url}/health"
    try:
        async with httpx.AsyncClient(timeout=config.status_timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return False, f"OpenSandbox server health check failed: HTTP {exc.response.status_code}."
    except httpx.HTTPError as exc:
        return False, f"OpenSandbox server is not reachable: {exc}."
    return True, f"OpenSandbox server is reachable at {config.opensandbox_server_url}."
