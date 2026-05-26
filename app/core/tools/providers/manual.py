from __future__ import annotations

from typing import Any

from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class ManualToolProvider:
    source_type = "manual"

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        response = str(tool.source.get("response_template") or f"Tool {tool.name} is registered.")
        return ToolInvocationResult(
            content=[{"type": "text", "text": response}],
            structured_content={"arguments": arguments},
            is_error=False,
        )

