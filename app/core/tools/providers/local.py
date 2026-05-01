from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


LOCAL_TOOL_NAMES = {
    "local_read_file",
    "local_write_file",
    "local_search_text",
    "local_todo",
    "local_shell_command",
}


class LocalToolProvider:
    """Thread-workspace local tools inspired by Hermes' file/terminal/todo tools."""

    source_type = "local"

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        thread_id = str(arguments.get("_thread_id") or "mcp-local")
        paths = self.artifact_store.prepare_thread(thread_id)
        operation = str(tool.source.get("operation") or "")
        try:
            if operation == "read_file":
                return self._read_file(paths, arguments)
            if operation == "write_file":
                return self._write_file(paths, arguments)
            if operation == "search_text":
                return self._search_text(paths, arguments)
            if operation == "todo":
                return self._todo(paths, arguments)
            if operation == "shell_command":
                return self._shell_command(paths, arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Unsupported local operation: {operation}"}],
            structured_content={"tool_name": tool.name, "operation": operation},
            is_error=True,
        )

    def _read_file(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._workspace_path(paths, str(arguments.get("path") or ""))
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self._display_path(paths, target)}")
        max_chars = self._bounded_int(arguments.get("max_chars"), default=12000, minimum=100, maximum=50000)
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        content = text[:max_chars]
        return ToolInvocationResult(
            content=[{"type": "text", "text": content}],
            structured_content={
                "path": self._display_path(paths, target),
                "chars": len(content),
                "truncated": truncated,
            },
            is_error=False,
        )

    def _write_file(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._workspace_path(paths, str(arguments.get("path") or ""))
        content = str(arguments.get("content") or "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        display_path = self._display_path(paths, target)
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Wrote {len(content)} characters to {display_path}"}],
            structured_content={
                "path": display_path,
                "chars": len(content),
            },
            is_error=False,
        )

    def _search_text(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            raise ValueError("pattern is required")
        root = self._workspace_path(paths, str(arguments.get("path") or "."))
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {self._display_path(paths, root)}")
        max_results = self._bounded_int(arguments.get("max_results"), default=50, minimum=1, maximum=200)
        case_sensitive = bool(arguments.get("case_sensitive", False))
        needle = pattern if case_sensitive else pattern.lower()
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        results: list[dict[str, Any]] = []
        for file_path in files:
            if len(results) >= max_results:
                break
            if self._skip_file(file_path):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.lower()
                if needle in haystack:
                    results.append(
                        {
                            "path": self._display_path(paths, file_path),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(results) >= max_results:
                        break
        text = "\n".join(f"{item['path']}:{item['line']}: {item['text']}" for item in results) or "No matches."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"pattern": pattern, "matches": results, "truncated": len(results) >= max_results},
            is_error=False,
        )

    def _todo(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        action = str(arguments.get("action") or "list").strip().lower()
        todo_path = paths.workspace / ".todos.json"
        todos = self._read_todos(todo_path)
        if action == "add":
            text = str(arguments.get("text") or "").strip()
            if not text:
                raise ValueError("text is required for add")
            next_id = max((int(item["id"]) for item in todos), default=0) + 1
            todos.append({"id": next_id, "text": text, "done": False})
            self._write_todos(todo_path, todos)
        elif action == "complete":
            todo_id = self._bounded_int(arguments.get("id"), default=0, minimum=1, maximum=1_000_000)
            matched = False
            for item in todos:
                if int(item["id"]) == todo_id:
                    item["done"] = True
                    matched = True
                    break
            if not matched:
                raise ValueError(f"todo id not found: {todo_id}")
            self._write_todos(todo_path, todos)
        elif action == "clear":
            todos = []
            self._write_todos(todo_path, todos)
        elif action != "list":
            raise ValueError("action must be one of: list, add, complete, clear")
        lines = [f"{item['id']}. [{'x' if item['done'] else ' '}] {item['text']}" for item in todos]
        return ToolInvocationResult(
            content=[{"type": "text", "text": "\n".join(lines) or "No todos."}],
            structured_content={"todos": todos},
            is_error=False,
        )

    def _shell_command(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        if os.getenv("LOCAL_SHELL_TOOL_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
            raise PermissionError("local_shell_command is disabled. Set LOCAL_SHELL_TOOL_ENABLED=true to enable it.")
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise ValueError("command is required")
        cwd = self._workspace_path(paths, str(arguments.get("cwd") or "."))
        if not cwd.is_dir():
            raise NotADirectoryError(f"cwd is not a directory: {self._display_path(paths, cwd)}")
        timeout = self._bounded_int(arguments.get("timeout_seconds"), default=20, minimum=1, maximum=60)
        max_chars = self._bounded_int(arguments.get("max_chars"), default=12000, minimum=100, maximum=50000)
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout[:max_chars]
        stderr = completed.stderr[:max_chars]
        text = "\n".join(
            part
            for part in [
                f"exit_code: {completed.returncode}",
                f"stdout:\n{stdout}" if stdout else "",
                f"stderr:\n{stderr}" if stderr else "",
            ]
            if part
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "command": command,
                "cwd": self._display_path(paths, cwd),
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            is_error=completed.returncode != 0,
        )

    @staticmethod
    def _read_todos(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_items = data.get("todos") if isinstance(data, dict) else []
        if not isinstance(raw_items, list):
            return []
        todos: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, dict) and "id" in item and "text" in item:
                todos.append({"id": int(item["id"]), "text": str(item["text"]), "done": bool(item.get("done"))})
        return todos

    @staticmethod
    def _write_todos(path: Path, todos: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps({"todos": todos}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _skip_file(path: Path) -> bool:
        try:
            return path.stat().st_size > 1_000_000
        except OSError:
            return True

    @staticmethod
    def _display_path(paths: ThreadPaths, path: Path) -> str:
        try:
            return path.resolve().relative_to(paths.workspace.resolve()).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _workspace_path(paths: ThreadPaths, raw_path: str) -> Path:
        if not raw_path.strip():
            raise ValueError("path is required")
        normalized = raw_path.replace("\\", "/").lstrip("/")
        candidate = (paths.workspace / normalized).resolve()
        workspace = paths.workspace.resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("Workspace path traversal blocked") from exc
        return candidate


def local_tool_definitions() -> list[ToolDefinition]:
    shell_enabled = os.getenv("LOCAL_SHELL_TOOL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    return [
        ToolDefinition(
            name="local_read_file",
            title="Read Workspace File",
            description="Read a UTF-8 text file from the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000},
                },
                "required": ["path"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "read_file"},
            editable=False,
        ),
        ToolDefinition(
            name="local_write_file",
            title="Write Workspace File",
            description="Write UTF-8 text into the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "write_file"},
            editable=False,
        ),
        ToolDefinition(
            name="local_search_text",
            title="Search Workspace Text",
            description="Search text files in the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["pattern"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "search_text"},
            editable=False,
        ),
        ToolDefinition(
            name="local_todo",
            title="Thread Todo",
            description="Manage a simple todo list stored in the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "add", "complete", "clear"]},
                    "text": {"type": "string"},
                    "id": {"type": "integer"},
                },
                "required": ["action"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "todo"},
            editable=False,
        ),
        ToolDefinition(
            name="local_shell_command",
            title="Run Workspace Shell Command",
            description="Run a shell command inside the current thread workspace. Disabled unless LOCAL_SHELL_TOOL_ENABLED=true.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 60},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000},
                },
                "required": ["command"],
            },
            enabled=shell_enabled,
            source={"type": LocalToolProvider.source_type, "operation": "shell_command"},
            editable=False,
        ),
    ]
