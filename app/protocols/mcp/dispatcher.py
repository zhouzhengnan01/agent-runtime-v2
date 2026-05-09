from __future__ import annotations

from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner
from app.core.tools import ToolInvocationService, ToolRegistry
from app.protocols.mcp.registry import to_mcp_tool
from app.protocols.mcp.schemas import MCP_PROTOCOL_VERSION


class McpDispatcher:
    """Dispatch MCP JSON-RPC methods to the protocol-neutral tool service."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        artifact_store: ArtifactStore | None = None,
        skill_runner: SkillRunner | None = None,
        tool_service: ToolInvocationService | None = None,
    ) -> None:
        self.tool_service = tool_service or ToolInvocationService(
            registry=registry,
            artifact_store=artifact_store,
            skill_runner=skill_runner,
        )

    def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            client_version = str(params.get("protocolVersion") or MCP_PROTOCOL_VERSION)
            return {
                "protocolVersion": client_version if client_version <= MCP_PROTOCOL_VERSION else MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "jetlinks-agent-runtime-v2", "version": "0.1.0"},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": [to_mcp_tool(tool) for tool in self.tool_service.list_tools()]}
        if method == "tools/call":
            return self.call_tool(params)
        raise ValueError(f"Unsupported MCP method: {method}")

    def call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tools/call params.name is required.")
        raw_arguments = params.get("arguments")
        arguments: dict[str, Any] = raw_arguments if isinstance(raw_arguments, dict) else {}
        return self.tool_service.call_tool(name, arguments).to_mcp_result()
