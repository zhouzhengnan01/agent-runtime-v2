from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.memory import MarkdownMemoryContext, MarkdownMemoryStore
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class ArtifactWorkspaceToolProvider:
    """Thread artifact workspace tools backed by ArtifactStore."""

    source_type = "artifact_workspace"

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        markdown_memory_store: MarkdownMemoryStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        thread_id = str(arguments.get("_thread_id") or "artifact-workspace")
        paths = self.artifact_store.prepare_thread(thread_id)
        operation = str(tool.source.get("operation") or "")
        try:
            if operation == "manifest_read":
                return self._manifest_read(paths)
            if operation == "manifest_update":
                return self._manifest_update(paths, arguments)
            if operation == "list":
                return self._list(paths)
            if operation == "read":
                return self._read(paths, arguments)
            if operation == "write":
                result = self._write(paths, arguments)
                self._sync_artifacts_memory(paths, arguments)
                return result
            if operation == "patch":
                result = self._patch(paths, arguments)
                self._sync_artifacts_memory(paths, arguments)
                return result
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Unsupported artifact operation: {operation}"}],
            structured_content={"tool_name": tool.name, "operation": operation},
            is_error=True,
        )

    def _manifest_read(self, paths: ThreadPaths) -> ToolInvocationResult:
        manifest = self.artifact_store.read_manifest(paths.thread_id)
        return ToolInvocationResult(
            content=[{"type": "text", "text": self._manifest_text(manifest)}],
            structured_content={"manifest": manifest},
            is_error=False,
        )

    def _manifest_update(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        metadata = arguments.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        manifest = self.artifact_store.update_manifest(
            paths.thread_id,
            primary_artifact=str(arguments.get("primary_artifact") or "").strip() or None,
            metadata=metadata,
        )
        self._sync_artifacts_memory(paths, arguments)
        return ToolInvocationResult(
            content=[{"type": "text", "text": "Updated artifact manifest."}],
            structured_content={"manifest": manifest},
            is_error=False,
        )

    def _list(self, paths: ThreadPaths) -> ToolInvocationResult:
        refs = [artifact.model_dump() for artifact in self.artifact_store.list_artifacts(paths.thread_id)]
        manifest = self.artifact_store.read_manifest(paths.thread_id)
        text = "\n".join(f"- {item['path']} ({item['size']} bytes)" for item in refs) or "No artifacts."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"thread_id": paths.thread_id, "artifacts": refs, "manifest": manifest},
            is_error=False,
        )

    def _read(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self.artifact_store.output_path(paths, str(arguments.get("path") or ""))
        if not target.is_file():
            raise FileNotFoundError(f"Artifact not found: {self._display_output_path(paths, target)}")
        max_chars = self._bounded_int(arguments.get("max_chars"), default=12000, minimum=100, maximum=500000)
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        content = text[:max_chars]
        return ToolInvocationResult(
            content=[{"type": "text", "text": content}],
            structured_content={
                "path": self._display_output_path(paths, target),
                "chars": len(content),
                "truncated": truncated,
            },
            is_error=False,
        )

    def _write(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        path = str(arguments.get("path") or "")
        content = str(arguments.get("content") or "")
        title = str(arguments.get("title") or "").strip() or None
        ref = self.artifact_store.write_text_artifact(paths, path, content)
        target = self.artifact_store.output_path(paths, path)
        entry = self.artifact_store.upsert_artifact(paths, target, title=title)
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Wrote artifact {entry['path']} ({len(content)} chars)."}],
            structured_content={"artifact": ref.model_dump(), "manifest_entry": entry},
            is_error=False,
        )

    def _patch(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self.artifact_store.output_path(paths, str(arguments.get("path") or ""))
        if not target.is_file():
            raise FileNotFoundError(f"Artifact not found: {self._display_output_path(paths, target)}")
        old_text = str(arguments.get("old_text") or "")
        new_text = str(arguments.get("new_text") or "")
        if not old_text:
            raise ValueError("old_text is required")
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_text)
        if count == 0:
            raise ValueError("old_text was not found in the artifact")
        expected = arguments.get("expected_replacements")
        if expected is not None and count != self._bounded_int(expected, default=count, minimum=1, maximum=100000):
            raise ValueError(f"expected_replacements did not match: expected {expected}, found {count}")
        limit = self._bounded_int(arguments.get("max_replacements"), default=1, minimum=1, maximum=100000)
        updated = text.replace(old_text, new_text, limit)
        target.write_text(updated, encoding="utf-8")
        entry = self.artifact_store.upsert_artifact(paths, target)
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Patched artifact {entry['path']} ({min(count, limit)} replacement(s))."}],
            structured_content={
                "path": entry["path"],
                "replacements": min(count, limit),
                "available_matches": count,
                "manifest_entry": entry,
            },
            is_error=False,
        )

    def _sync_artifacts_memory(self, paths: ThreadPaths, arguments: dict[str, Any]) -> None:
        manifest = self.artifact_store.read_manifest(paths.thread_id)
        context = MarkdownMemoryContext(
            agent_name=str(arguments.get("_agent_name") or "default"),
            user_id=str(arguments.get("_user_id") or "").strip() or None,
            project_id=str(arguments.get("_project_id") or "").strip() or None,
            thread_id=paths.thread_id,
        )
        self.markdown_memory_store.write(context, "session", "artifacts.md", self._artifacts_memory_text(manifest))

    @staticmethod
    def _manifest_text(manifest: dict[str, Any]) -> str:
        artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
        lines = [
            f"thread_id: {manifest.get('thread_id')}",
            f"primary_artifact: {manifest.get('primary_artifact') or ''}",
            f"artifact_count: {len(artifacts)}",
        ]
        for item in artifacts:
            if isinstance(item, dict):
                lines.append(f"- {item.get('path')} ({item.get('kind')}, {item.get('size')} bytes)")
        return "\n".join(lines)

    @staticmethod
    def _artifacts_memory_text(manifest: dict[str, Any]) -> str:
        artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
        lines = [
            "# Artifacts",
            "",
            "## Current",
            "",
        ]
        if artifacts:
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("name") or item.get("path"))
                suffix = " primary" if item.get("path") == manifest.get("primary_artifact") else ""
                lines.append(f"- `{item.get('path')}` - {title}, {item.get('kind')}, {item.get('status')}{suffix}.")
        else:
            lines.append("- No current artifacts.")
        lines.extend(["", "## Manifest", "", f"- Updated at: {manifest.get('updated_at') or ''}"])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _display_output_path(paths: ThreadPaths, path: Path) -> str:
        return f"outputs/{path.resolve().relative_to(paths.outputs.resolve()).as_posix()}"

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))


def artifact_workspace_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="artifact_manifest_read",
            title="Read Artifact Manifest",
            description="Read the current thread artifact manifest.",
            input_schema={"type": "object", "properties": {}},
            source={"type": ArtifactWorkspaceToolProvider.source_type, "operation": "manifest_read"},
            editable=False,
        ),
        ToolDefinition(
            name="artifact_manifest_update",
            title="Update Artifact Manifest",
            description="Update the current thread artifact manifest metadata or primary artifact.",
            input_schema={
                "type": "object",
                "properties": {
                    "primary_artifact": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
            source={"type": ArtifactWorkspaceToolProvider.source_type, "operation": "manifest_update"},
            editable=False,
        ),
        ToolDefinition(
            name="artifact_list",
            title="List Artifacts",
            description="List current thread output artifacts and manifest entries.",
            input_schema={"type": "object", "properties": {}},
            source={"type": ArtifactWorkspaceToolProvider.source_type, "operation": "list"},
            editable=False,
        ),
        ToolDefinition(
            name="artifact_read",
            title="Read Artifact",
            description="Read a UTF-8 text artifact from current thread outputs.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 500000},
                },
                "required": ["path"],
            },
            source={"type": ArtifactWorkspaceToolProvider.source_type, "operation": "read"},
            editable=False,
        ),
        ToolDefinition(
            name="artifact_write",
            title="Write Artifact",
            description="Write a UTF-8 text artifact into current thread outputs and update the manifest.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            source={"type": ArtifactWorkspaceToolProvider.source_type, "operation": "write"},
            editable=False,
        ),
        ToolDefinition(
            name="artifact_patch",
            title="Patch Artifact",
            description="Patch a text artifact by replacing old_text with new_text, then update the manifest.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "max_replacements": {"type": "integer", "minimum": 1},
                    "expected_replacements": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "old_text", "new_text"],
            },
            source={"type": ArtifactWorkspaceToolProvider.source_type, "operation": "patch"},
            editable=False,
        ),
    ]
