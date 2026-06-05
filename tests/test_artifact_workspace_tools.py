from __future__ import annotations

from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.memory import MarkdownMemoryStore
from app.core.tools import ToolInvocationService


def test_artifact_workspace_tools_write_list_read_patch_and_sync_memory(tmp_path: Path) -> None:
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    memory_store = MarkdownMemoryStore(root_dir=tmp_path / "memory", threads_root_dir=tmp_path / "threads")
    service = ToolInvocationService(artifact_store=artifact_store, markdown_memory_store=memory_store)
    args = {"_thread_id": "artifact-session", "_agent_name": "default", "_user_id": "u1"}

    write_result = service.call_tool(
        "artifact_write",
        {
            **args,
            "path": "index.html",
            "title": "架构页面",
            "content": "<h1>Old Title</h1>",
        },
    )
    assert write_result.is_error is False
    assert write_result.structured_content["manifest_entry"]["path"] == "outputs/index.html"
    assert (tmp_path / "threads" / "artifact-session" / "outputs" / "index.html").is_file()

    manifest_result = service.call_tool("artifact_manifest_read", args)
    assert manifest_result.is_error is False
    assert manifest_result.structured_content["manifest"]["primary_artifact"] == "outputs/index.html"

    list_result = service.call_tool("artifact_list", args)
    assert list_result.is_error is False
    assert list_result.structured_content["artifacts"][0]["name"] == "index.html"
    assert list_result.structured_content["artifacts"][0]["kind"] == "html"

    read_result = service.call_tool("artifact_read", {**args, "path": "outputs/index.html"})
    assert read_result.is_error is False
    assert "Old Title" in read_result.content[0]["text"]

    virtual_read_result = service.call_tool("artifact_read", {**args, "path": "/mnt/user-data/outputs/index.html"})
    assert virtual_read_result.is_error is False
    assert virtual_read_result.structured_content["path"] == "outputs/index.html"
    assert "Old Title" in virtual_read_result.content[0]["text"]

    patch_result = service.call_tool(
        "artifact_patch",
        {
            **args,
            "path": "/mnt/user-data/outputs/index.html",
            "old_text": "Old Title",
            "new_text": "New Title",
            "expected_replacements": 1,
        },
    )
    assert patch_result.is_error is False
    assert "New Title" in (tmp_path / "threads" / "artifact-session" / "outputs" / "index.html").read_text()

    artifacts_memory = (tmp_path / "threads" / "artifact-session" / "memory" / "artifacts.md").read_text()
    assert "`outputs/index.html`" in artifacts_memory
    assert "primary" in artifacts_memory


def test_artifact_workspace_tool_blocks_output_traversal(tmp_path: Path) -> None:
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = service.call_tool(
        "artifact_write",
        {"_thread_id": "artifact-session", "path": "../escape.html", "content": "<h1>bad</h1>"},
    )

    assert result.is_error is True
    assert "traversal" in result.content[0]["text"].lower()
