from __future__ import annotations

from typing import Any

from app.core.memory import MarkdownMemoryContext, MarkdownMemoryScope, MarkdownMemoryStore
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class MarkdownMemoryToolProvider:
    """Markdown folder memory tools backed by MarkdownMemoryStore."""

    source_type = "markdown_memory"

    def __init__(self, markdown_memory_store: MarkdownMemoryStore | None = None) -> None:
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        operation = str(tool.source.get("operation") or "")
        context = self._context(arguments)
        try:
            if operation == "list":
                return self._list(context, arguments)
            if operation == "read":
                return self._read(context, arguments)
            if operation == "append":
                return self._append(context, arguments)
            if operation == "search":
                return self._search(context, arguments)
            if operation == "compress":
                return self._compress(context, arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Unsupported markdown memory operation: {operation}"}],
            structured_content={"tool_name": tool.name, "operation": operation},
            is_error=True,
        )

    def _list(self, context: MarkdownMemoryContext, arguments: dict[str, Any]) -> ToolInvocationResult:
        scope = self._scope(arguments.get("scope"))
        files = self.markdown_memory_store.list_files(
            context,
            scope,
            path=str(arguments.get("path") or "."),
            max_results=self._bounded_int(arguments.get("max_results"), default=100, minimum=1, maximum=500),
        )
        payloads = [file.to_payload() for file in files]
        text = "\n".join(f"- {item['scope']}/{item['path']} ({item['size']} bytes)" for item in payloads) or "No markdown memories."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"scope": scope, "files": payloads},
            is_error=False,
        )

    def _read(self, context: MarkdownMemoryContext, arguments: dict[str, Any]) -> ToolInvocationResult:
        scope = self._scope(arguments.get("scope"))
        path = str(arguments.get("path") or "")
        max_chars_default = self._bounded_int(
            arguments.get("_markdown_max_chars"),
            default=12000,
            minimum=1000,
            maximum=100000,
        )
        text, truncated = self.markdown_memory_store.read(
            context,
            scope,
            path,
            max_chars=self._bounded_int(arguments.get("max_chars"), default=max_chars_default, minimum=100, maximum=500000),
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"scope": scope, "path": path, "chars": len(text), "truncated": truncated},
            is_error=False,
        )

    def _append(self, context: MarkdownMemoryContext, arguments: dict[str, Any]) -> ToolInvocationResult:
        scope = self._scope(arguments.get("scope"))
        self._ensure_writable(scope, arguments)
        path = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        if not content.strip():
            raise ValueError("content is required")
        file = self.markdown_memory_store.append(
            context,
            scope,
            path,
            content,
            heading=str(arguments.get("heading") or "").strip() or None,
        )
        payload = file.to_payload()
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Appended markdown memory {scope}/{payload['path']}."}],
            structured_content={"file": payload},
            is_error=False,
        )

    def _search(self, context: MarkdownMemoryContext, arguments: dict[str, Any]) -> ToolInvocationResult:
        scopes = self._scopes(arguments.get("scopes"))
        query = str(arguments.get("query") or "")
        matches = self.markdown_memory_store.search(
            context,
            scopes,
            query,
            max_results=self._bounded_int(arguments.get("max_results"), default=50, minimum=1, maximum=200),
        )
        payloads = [match.to_payload() for match in matches]
        text = "\n".join(
            f"{item['scope']}/{item['path']}:{item['line']}: {item['text']}" for item in payloads
        ) or "No markdown memory matches."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"matches": payloads, "scopes": scopes, "query": query},
            is_error=False,
        )

    def _compress(self, context: MarkdownMemoryContext, arguments: dict[str, Any]) -> ToolInvocationResult:
        scope = self._scope(arguments.get("scope"))
        self._ensure_writable(scope, arguments)
        compression = self.markdown_memory_store.compress(
            context,
            scope,
            str(arguments.get("source_path") or arguments.get("path") or ""),
            target_path=str(arguments.get("target_path") or "").strip() or None,
            max_chars=self._bounded_int(arguments.get("max_chars"), default=4000, minimum=500, maximum=100000),
            keywords=self._string_list(arguments.get("keywords")),
        )
        payload = compression.to_payload()
        return ToolInvocationResult(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"Compressed {scope}/{payload['source_path']} to {scope}/{payload['target_path']} "
                        f"({payload['original_chars']} -> {payload['compressed_chars']} chars)."
                    ),
                }
            ],
            structured_content={"compression": payload},
            is_error=False,
        )

    @staticmethod
    def _context(arguments: dict[str, Any]) -> MarkdownMemoryContext:
        return MarkdownMemoryContext(
            agent_name=str(arguments.get("_agent_name") or "default"),
            user_id=str(arguments.get("_user_id") or "").strip() or None,
            project_id=str(arguments.get("_project_id") or "").strip() or None,
            thread_id=str(arguments.get("_thread_id") or "").strip() or None,
        )

    @staticmethod
    def _scope(value: object) -> MarkdownMemoryScope:
        raw = str(value or "session").strip().lower()
        if raw in {"global", "user", "project", "session"}:
            return raw  # type: ignore[return-value]
        return "session"

    @classmethod
    def _ensure_writable(cls, scope: MarkdownMemoryScope, arguments: dict[str, Any]) -> None:
        writable_scopes = cls._writable_scopes(arguments.get("_markdown_writable_scopes"))
        if scope not in writable_scopes:
            allowed = ", ".join(writable_scopes) or "none"
            raise PermissionError(f"Markdown memory scope is read-only: {scope}. Writable scopes: {allowed}.")

    @classmethod
    def _writable_scopes(cls, value: object) -> list[MarkdownMemoryScope]:
        if not isinstance(value, list):
            return ["session"]
        scopes: list[MarkdownMemoryScope] = []
        for item in value:
            raw = str(item or "").strip().lower()
            if raw in {"global", "user", "project", "session"} and raw not in scopes:
                scopes.append(raw)  # type: ignore[arg-type]
        return scopes or ["session"]

    @classmethod
    def _scopes(cls, value: object) -> list[MarkdownMemoryScope]:
        if not isinstance(value, list) or not value:
            return ["session", "project", "user", "global"]
        scopes: list[MarkdownMemoryScope] = []
        for item in value:
            scope = cls._scope(item)
            if scope not in scopes:
                scopes.append(scope)
        return scopes

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))


def markdown_memory_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="memory_md_list",
            title="List Markdown Memories",
            description="List Markdown memory files in a hierarchical global/user/project/session scope.",
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["global", "user", "project", "session"], "default": "session"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
                },
            },
            source={"type": MarkdownMemoryToolProvider.source_type, "operation": "list"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_md_read",
            title="Read Markdown Memory",
            description="Read a Markdown memory file from the selected hierarchical scope.",
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["global", "user", "project", "session"], "default": "session"},
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 500000},
                },
                "required": ["path"],
            },
            source={"type": MarkdownMemoryToolProvider.source_type, "operation": "read"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_md_append",
            title="Append Markdown Memory",
            description="Append notes to a Markdown memory file in the selected hierarchical scope.",
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["global", "user", "project", "session"], "default": "session"},
                    "path": {"type": "string"},
                    "heading": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            source={"type": MarkdownMemoryToolProvider.source_type, "operation": "append"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_md_search",
            title="Search Markdown Memories",
            description="Search Markdown memory files across global/user/project/session scopes.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scopes": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["global", "user", "project", "session"]},
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["query"],
            },
            source={"type": MarkdownMemoryToolProvider.source_type, "operation": "search"},
            editable=False,
        ),
        ToolDefinition(
            name="memory_md_compress",
            title="Compress Markdown Memory",
            description="Create an extractive compressed summary Markdown file from a larger memory file.",
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["global", "user", "project", "session"], "default": "session"},
                    "source_path": {"type": "string"},
                    "target_path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 100000},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["source_path"],
            },
            source={"type": MarkdownMemoryToolProvider.source_type, "operation": "compress"},
            editable=False,
        ),
    ]
