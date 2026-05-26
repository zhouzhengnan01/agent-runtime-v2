from __future__ import annotations

from typing import Any

from app.core.mcp import McpToolDefinition, McpToolRegistry
from app.core.tools import ToolDefinition


def to_mcp_tool(tool: ToolDefinition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "inputSchema": tool.input_schema or {"type": "object", "additionalProperties": True},
    }
    if tool.output_schema:
        payload["outputSchema"] = tool.output_schema
    return payload


__all__ = ["McpToolDefinition", "McpToolRegistry", "to_mcp_tool"]
