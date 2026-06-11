from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.parse import urlparse

from app.core.artifacts import ArtifactStore
from app.core.memory import MarkdownMemoryStore, MemoryStore
from app.core.skills import SkillRunner
from app.core.tools.providers import (
    ArtifactWorkspaceToolProvider,
    BigscreenToolset,
    DelegateToolProvider,
    LocalToolProvider,
    ManualToolProvider,
    MarkdownMemoryToolProvider,
    MemoryToolProvider,
    McpStreamableHttpToolProvider,
    McpStdioToolProvider,
    SkillToolProvider,
)
from app.core.tools.registry import ToolRegistry
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class ToolProvider(Protocol):
    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult: ...


@dataclass
class RuntimeMcpDiscoveryFailure:
    server_name: str
    server_type: str
    endpoint: str
    reason: str
    error: str = ""

    def to_event_data(self) -> dict[str, str]:
        payload = {
            "server_name": self.server_name,
            "server_type": self.server_type,
            "endpoint": self.endpoint,
            "reason": self.reason,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class RuntimeMcpDiscoveryResult:
    tools: list[ToolDefinition] = field(default_factory=list)
    failures: list[RuntimeMcpDiscoveryFailure] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0


class ToolInvocationService:
    """Protocol-neutral tool discovery and invocation service."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        artifact_store: ArtifactStore | None = None,
        skill_runner: SkillRunner | None = None,
        memory_store: MemoryStore | None = None,
        markdown_memory_store: MarkdownMemoryStore | None = None,
        root_dir: Path | None = None,
    ) -> None:
        registry_root = getattr(registry, "root_dir", None)
        self.root_dir = root_dir or (
            registry_root if isinstance(registry_root, Path) else Path(__file__).resolve().parents[3]
        )
        self.artifact_store = artifact_store or ArtifactStore()
        self.memory_store = memory_store or MemoryStore()
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()
        self.registry = registry or ToolRegistry(
            root_dir=self.root_dir,
            artifact_store=self.artifact_store,
            skill_runner=skill_runner,
        )
        self.providers: dict[str, ToolProvider] = {
            LocalToolProvider.source_type: LocalToolProvider(artifact_store=self.artifact_store),
            ArtifactWorkspaceToolProvider.source_type: ArtifactWorkspaceToolProvider(
                artifact_store=self.artifact_store,
                markdown_memory_store=self.markdown_memory_store,
            ),
            DelegateToolProvider.source_type: DelegateToolProvider(root_dir=self.root_dir),
            ManualToolProvider.source_type: ManualToolProvider(),
            BigscreenToolset.source_type: BigscreenToolset(artifact_store=self.artifact_store),
            McpStreamableHttpToolProvider.source_type: McpStreamableHttpToolProvider(),
            McpStdioToolProvider.source_type: McpStdioToolProvider(artifact_store=self.artifact_store),
            MemoryToolProvider.source_type: MemoryToolProvider(memory_store=self.memory_store),
            MarkdownMemoryToolProvider.source_type: MarkdownMemoryToolProvider(
                markdown_memory_store=self.markdown_memory_store
            ),
            SkillToolProvider.source_type: SkillToolProvider(
                artifact_store=self.artifact_store,
                skill_runner=skill_runner,
            ),
        }
        self._runtime_mcp_tool_cache: dict[str, tuple[float, list[ToolDefinition]]] = {}

    def list_tools(self, *, include_disabled: bool = False, include_skill_tools: bool = True) -> list[ToolDefinition]:
        return self.registry.list(include_disabled=include_disabled, include_skill_tools=include_skill_tools)

    def list_runtime_mcp_tools(self, mcp_servers: list[dict[str, Any]]) -> list[ToolDefinition]:
        return self.discover_runtime_mcp_tools(mcp_servers).tools

    def discover_runtime_mcp_tools(self, mcp_servers: list[dict[str, Any]]) -> RuntimeMcpDiscoveryResult:
        http_provider = self.providers.get(McpStreamableHttpToolProvider.source_type)
        stdio_provider = self.providers.get(McpStdioToolProvider.source_type)
        tools: list[ToolDefinition] = []
        failures: list[RuntimeMcpDiscoveryFailure] = []
        seen: set[str] = set()
        for server in mcp_servers:
            if not isinstance(server, dict):
                continue
            allowed, reason = _runtime_mcp_server_allowed(server)
            if not allowed:
                failures.append(_discovery_failure(server, reason=reason))
                continue
            provider = http_provider if str(server.get("type") or "").lower() in {"http", "sse"} else stdio_provider
            if not isinstance(provider, (McpStreamableHttpToolProvider, McpStdioToolProvider)):
                failures.append(_discovery_failure(server, reason="provider_unavailable"))
                continue
            cache_key = _runtime_mcp_server_cache_key(server)
            cached = self._runtime_mcp_cached_tools(cache_key)
            if cached is None:
                discovered, error = provider.list_remote_tools_with_error(server)
                if error:
                    failures.append(_discovery_failure(server, reason="discovery_failed", error=error))
                    continue
                self._runtime_mcp_tool_cache[cache_key] = (time.monotonic() + _runtime_mcp_cache_ttl(), discovered)
            else:
                discovered = cached
            for tool in discovered:
                if tool.name in seen:
                    continue
                seen.add(tool.name)
                tools.append(tool)
        return RuntimeMcpDiscoveryResult(tools=tools, failures=failures)

    def get_tool(self, name: str) -> ToolDefinition:
        return self.registry.get(name)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvocationResult:
        tool = self._runtime_mcp_tool(name, arguments) or self.registry.get(name)
        if not tool.enabled:
            raise ValueError(f"Tool is disabled: {name}")
        source_type = str(tool.source.get("type") or "manual")
        provider = self.providers.get(source_type)
        if provider is None:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"Unsupported tool source type: {source_type}"}],
                structured_content={"tool_name": name, "source_type": source_type},
                is_error=True,
            )
        return provider.call(tool, arguments)

    @staticmethod
    def _runtime_mcp_tool(name: str, arguments: dict[str, Any]) -> ToolDefinition | None:
        raw_tools = arguments.get("_runtime_mcp_tools")
        if not isinstance(raw_tools, list):
            return None
        for item in raw_tools:
            if not isinstance(item, dict) or item.get("name") != name:
                continue
            return ToolDefinition(
                name=str(item["name"]),
                title=str(item.get("title") or item["name"]),
                description=str(item.get("description") or item["name"]),
                input_schema=dict(item.get("input_schema") or {}),
                output_schema=dict(item.get("output_schema") or {}),
                enabled=bool(item.get("enabled", True)),
                source=dict(item.get("source") or {}),
                editable=False,
            )
        return None

    def _runtime_mcp_cached_tools(self, cache_key: str) -> list[ToolDefinition] | None:
        cached = self._runtime_mcp_tool_cache.get(cache_key)
        if cached is None:
            return None
        expires_at, tools = cached
        if expires_at <= time.monotonic():
            self._runtime_mcp_tool_cache.pop(cache_key, None)
            return None
        return tools


def _runtime_mcp_server_allowed(server: dict[str, Any]) -> tuple[bool, str]:
    server_type = str(server.get("type") or "").strip().lower()
    if server_type not in {"http", "sse", "stdio"}:
        return False, "unsupported_server_type"
    if server_type == "stdio":
        if os.getenv("RUNTIME_MCP_STDIO_ENABLED", "").lower() not in {"1", "true", "yes", "on"}:
            return False, "stdio_disabled"
        command = str(server.get("command") or "").strip()
        if not command:
            return False, "missing_command"
        allowed_commands = _host_patterns(os.getenv("RUNTIME_MCP_STDIO_ALLOWED_COMMANDS", ""))
        if allowed_commands and command not in allowed_commands:
            return False, "command_not_allowed"
        return True, ""
    parsed = urlparse(str(server.get("url") or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False, "invalid_url"
    host = parsed.hostname.lower()
    if _host_matches(host, _blocked_runtime_mcp_hosts()):
        return False, "blocked_host"
    allowed_hosts = _allowed_runtime_mcp_hosts()
    if allowed_hosts and not _host_matches(host, allowed_hosts):
        return False, "host_not_allowed"
    return True, ""


def _runtime_mcp_server_cache_key(server: dict[str, Any]) -> str:
    rendered = json.dumps(server, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _runtime_mcp_cache_ttl() -> float:
    raw = os.getenv("RUNTIME_MCP_TOOLS_CACHE_SECONDS", "60")
    try:
        return max(0.0, min(float(raw), 3600.0))
    except ValueError:
        return 60.0


def _allowed_runtime_mcp_hosts() -> list[str]:
    return _host_patterns(os.getenv("RUNTIME_MCP_ALLOWED_HOSTS", ""))


def _blocked_runtime_mcp_hosts() -> list[str]:
    configured = _host_patterns(os.getenv("RUNTIME_MCP_BLOCKED_HOSTS", ""))
    return [*configured, "169.254.169.254", "metadata.google.internal"]


def _host_patterns(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _host_matches(host: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("*.") and host.endswith(pattern[1:]):
            return True
        if host == pattern:
            return True
    return False


def _discovery_failure(
    server: dict[str, Any],
    *,
    reason: str,
    error: str = "",
) -> RuntimeMcpDiscoveryFailure:
    parsed = urlparse(str(server.get("url") or ""))
    endpoint = ""
    if parsed.scheme and parsed.netloc:
        endpoint = f"{parsed.scheme}://{parsed.netloc}"
    return RuntimeMcpDiscoveryFailure(
        server_name=str(server.get("name") or ""),
        server_type=str(server.get("type") or ""),
        endpoint=endpoint,
        reason=reason,
        error=error[:1000],
    )
