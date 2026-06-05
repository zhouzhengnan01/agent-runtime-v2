from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    title: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    source: dict[str, Any] = field(default_factory=dict)
    editable: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "enabled": self.enabled,
            "source": self.source,
            "editable": self.editable,
        }


@dataclass(frozen=True)
class ToolInvocationResult:
    content: list[dict[str, Any]]
    structured_content: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False

    def to_mcp_result(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "structuredContent": self.structured_content,
            "isError": self.is_error,
        }

