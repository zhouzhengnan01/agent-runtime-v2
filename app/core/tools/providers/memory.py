from __future__ import annotations

from typing import Any, Literal

from app.core.memory import MemoryStore
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult

MemoryToolOperation = Literal["remember", "search", "forget", "clear"]


class MemoryToolProvider:
    """Local persistent memory tools backed by MemoryStore."""

    source_type = "memory"

    def __init__(self, memory_store: MemoryStore | None = None) -> None:
        self.memory_store = memory_store or MemoryStore()

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        agent_name = str(arguments.get("_agent_name") or "default")
        operation = str(tool.source.get("operation") or "")
        scope = self._scope(arguments.get("scope") or arguments.get("_memory_scope"))
        try:
            if operation == "remember":
                return self._remember(agent_name, arguments, scope)
            if operation == "search":
                return self._search(agent_name, arguments, scope)
            if operation == "forget":
                return self._forget(agent_name, arguments, scope)
            if operation == "clear":
                return self._clear(agent_name, scope)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Unsupported memory operation: {operation}"}],
            structured_content={"tool_name": tool.name, "operation": operation},
            is_error=True,
        )

    def _remember(self, agent_name: str, arguments: dict[str, Any], scope: str) -> ToolInvocationResult:
        item = self.memory_store.remember(
            agent_name,
            str(arguments.get("text") or ""),
            tags=self._tags(arguments.get("tags")),
            scope=self._scope(scope),
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Saved memory {item.id}."}],
            structured_content={"memory": item.to_payload(), "scope": scope},
            is_error=False,
        )

    def _search(self, agent_name: str, arguments: dict[str, Any], scope: str) -> ToolInvocationResult:
        limit = self._bounded_int(arguments.get("limit"), default=20, minimum=1, maximum=100)
        items = self.memory_store.list(
            agent_name,
            query=str(arguments.get("query") or "").strip() or None,
            tags=self._tags(arguments.get("tags")),
            limit=limit,
            scope=self._scope(scope),
        )
        lines = [f"{item.id}: {item.text}" for item in items]
        return ToolInvocationResult(
            content=[{"type": "text", "text": "\n".join(lines) or "No memories found."}],
            structured_content={"memories": [item.to_payload() for item in items], "scope": scope},
            is_error=False,
        )

    def _forget(self, agent_name: str, arguments: dict[str, Any], scope: str) -> ToolInvocationResult:
        memory_id = str(arguments.get("id") or "").strip()
        removed = self.memory_store.forget(agent_name, memory_id, scope=self._scope(scope))
        text = f"Forgot memory {memory_id}." if removed else f"Memory not found: {memory_id}"
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"id": memory_id, "removed": removed, "scope": scope},
            is_error=False,
        )

    def _clear(self, agent_name: str, scope: str) -> ToolInvocationResult:
        count = self.memory_store.clear(agent_name, scope=self._scope(scope))
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Cleared {count} memories."}],
            structured_content={"cleared": count, "scope": scope},
            is_error=False,
        )

    @staticmethod
    def _tags(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    @staticmethod
    def _scope(value: object) -> Literal["agent", "global"]:
        return "global" if str(value or "").strip().lower() == "global" else "agent"

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))


def memory_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="memory_remember",
            title="Remember",
            description="Persist a useful long-term memory for this agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "scope": {"type": "string", "enum": ["agent", "global"], "default": "agent"},
                },
                "required": ["text"],
            },
            source={"type": MemoryToolProvider.source_type, "operation": "remember"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_search",
            title="Search Memories",
            description="Search persisted memories available to this agent.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    "scope": {"type": "string", "enum": ["agent", "global"], "default": "agent"},
                },
            },
            source={"type": MemoryToolProvider.source_type, "operation": "search"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_forget",
            title="Forget Memory",
            description="Remove one persisted memory by id.",
            input_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "scope": {"type": "string", "enum": ["agent", "global"], "default": "agent"},
                },
                "required": ["id"],
            },
            source={"type": MemoryToolProvider.source_type, "operation": "forget"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_clear",
            title="Clear Memories",
            description="Remove all persisted memories in the selected memory scope.",
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["agent", "global"], "default": "agent"},
                },
            },
            source={"type": MemoryToolProvider.source_type, "operation": "clear"},
            editable=False,
        ),
    ]
