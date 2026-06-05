from __future__ import annotations

from pathlib import Path

from app.core.memory import MarkdownMemoryStore
from app.core.tools import ToolInvocationService


def test_markdown_memory_tools_append_list_read_search_and_compress(tmp_path: Path) -> None:
    service = ToolInvocationService(markdown_memory_store=MarkdownMemoryStore(root_dir=tmp_path))
    base_args = {
        "_agent_name": "default",
        "_user_id": "u1",
        "_project_id": "runtime-v2",
        "_thread_id": "session-a",
    }

    append_result = service.call_tool(
        "memory_md_append",
        {
            **base_args,
            "scope": "session",
            "path": "notes/working.md",
            "heading": "需求",
            "content": "用户需要 Markdown 目录层级记忆系统，并且需要压缩逻辑。" * 20,
        },
    )
    assert append_result.is_error is False
    assert append_result.structured_content["file"]["path"] == "notes/working.md"

    list_result = service.call_tool("memory_md_list", {**base_args, "scope": "session", "path": "notes"})
    assert list_result.is_error is False
    assert list_result.structured_content["files"][0]["path"] == "notes/working.md"

    read_result = service.call_tool(
        "memory_md_read",
        {**base_args, "scope": "session", "path": "notes/working.md", "max_chars": 120},
    )
    assert read_result.is_error is False
    assert read_result.structured_content["truncated"] is True
    assert "目录层级" in read_result.content[0]["text"]

    search_result = service.call_tool(
        "memory_md_search",
        {**base_args, "query": "压缩逻辑", "scopes": ["session"], "max_results": 10},
    )
    assert search_result.is_error is False
    assert search_result.structured_content["matches"][0]["path"] == "notes/working.md"

    compress_result = service.call_tool(
        "memory_md_compress",
        {
            **base_args,
            "scope": "session",
            "source_path": "notes/working.md",
            "target_path": "notes/working.summary.md",
            "max_chars": 700,
            "keywords": ["目录层级", "压缩"],
        },
    )
    assert compress_result.is_error is False
    compression = compress_result.structured_content["compression"]
    assert compression["target_path"] == "notes/working.summary.md"
    assert compression["compressed_chars"] <= 700


def test_markdown_memory_tool_session_scope_is_owned_by_thread(tmp_path: Path) -> None:
    service = ToolInvocationService(markdown_memory_store=MarkdownMemoryStore(root_dir=tmp_path))
    service.call_tool(
        "memory_md_append",
        {
            "_agent_name": "default",
            "_user_id": "u1",
            "_thread_id": "session-a",
            "scope": "session",
            "path": "summary.md",
            "content": "session-a 的隔离记忆",
        },
    )

    same_thread_other_user = service.call_tool(
        "memory_md_search",
        {
            "_agent_name": "default",
            "_user_id": "u2",
            "_thread_id": "session-a",
            "query": "隔离记忆",
            "scopes": ["session"],
        },
    )
    same_user_other_thread = service.call_tool(
        "memory_md_search",
        {
            "_agent_name": "default",
            "_user_id": "u1",
            "_thread_id": "session-b",
            "query": "隔离记忆",
            "scopes": ["session"],
        },
    )

    assert same_thread_other_user.is_error is False
    assert same_thread_other_user.structured_content["matches"][0]["path"] == "summary.md"
    assert same_user_other_thread.is_error is False
    assert same_user_other_thread.structured_content["matches"] == []


def test_markdown_memory_tools_list_default_thread_memory_files(tmp_path: Path) -> None:
    service = ToolInvocationService(markdown_memory_store=MarkdownMemoryStore(root_dir=tmp_path))

    list_result = service.call_tool(
        "memory_md_list",
        {
            "_agent_name": "default",
            "_user_id": "u1",
            "_thread_id": "session-a",
            "scope": "session",
        },
    )

    assert list_result.is_error is False
    paths = {file["path"] for file in list_result.structured_content["files"]}
    assert {
        "conversation.md",
        "summary.md",
        "decisions.md",
        "todo.md",
        "artifacts.md",
    }.issubset(paths)
    assert (tmp_path / "threads" / "session-a" / "memory" / "artifacts.md").is_file()


def test_markdown_memory_tools_default_to_session_only_writes(tmp_path: Path) -> None:
    service = ToolInvocationService(markdown_memory_store=MarkdownMemoryStore(root_dir=tmp_path))

    denied = service.call_tool(
        "memory_md_append",
        {
            "_agent_name": "default",
            "_user_id": "u1",
            "_project_id": "runtime-v2",
            "_thread_id": "session-a",
            "scope": "project",
            "path": "decisions.md",
            "content": "不应默认写入项目级记忆。",
        },
    )
    allowed = service.call_tool(
        "memory_md_append",
        {
            "_agent_name": "default",
            "_user_id": "u1",
            "_project_id": "runtime-v2",
            "_thread_id": "session-a",
            "_markdown_writable_scopes": ["session", "project"],
            "scope": "project",
            "path": "decisions.md",
            "content": "允许写入项目级记忆。",
        },
    )

    assert denied.is_error is True
    assert "read-only: project" in denied.content[0]["text"]
    assert allowed.is_error is False
    assert allowed.structured_content["file"]["scope"] == "project"
