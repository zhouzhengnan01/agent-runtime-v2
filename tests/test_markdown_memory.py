from __future__ import annotations

from pathlib import Path

import pytest

from app.core.memory import MarkdownMemoryContext, MarkdownMemoryStore


def test_markdown_memory_store_uses_thread_memory_for_session_scope(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root_dir=tmp_path)
    first = MarkdownMemoryContext(
        agent_name="default",
        user_id="user/a",
        project_id="jetlinks/runtime",
        thread_id="thread/one",
    )
    second = MarkdownMemoryContext(
        agent_name="default",
        user_id="user/b",
        project_id="jetlinks/runtime",
        thread_id="thread/one",
    )

    session_file = store.append(first, "session", "notes/summary.md", "会话一：讨论 Markdown 记忆。")
    project_file = store.append(first, "project", "architecture/overview.md", "项目记忆：目录层级和压缩。")
    user_file = store.append(first, "user", "preferences.md", "用户偏好：中文回复。")

    assert session_file.path == "notes/summary.md"
    assert project_file.path == "architecture/overview.md"
    assert user_file.path == "preferences.md"
    assert (tmp_path / "threads" / "thread-one" / "memory" / "notes" / "summary.md").is_file()
    assert (
        tmp_path
        / "users"
        / "user-a"
        / "projects"
        / "jetlinks-runtime"
        / "architecture"
        / "overview.md"
    ).is_file()
    assert store.search(first, ["session", "project", "user"], "记忆", max_results=10)
    second_matches = store.search(second, ["session", "project", "user"], "记忆", max_results=10)
    assert [match.path for match in second_matches] == ["notes/summary.md"]


def test_markdown_memory_store_creates_default_thread_memory_files(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root_dir=tmp_path)
    context = MarkdownMemoryContext(agent_name="default", user_id="u1", thread_id="thread/one")

    files = store.list_files(context, "session")
    paths = {file.path for file in files}

    assert {
        "conversation.md",
        "summary.md",
        "decisions.md",
        "todo.md",
        "artifacts.md",
    }.issubset(paths)
    assert (tmp_path / "threads" / "thread-one" / "memory" / "conversation.md").is_file()
    summary, truncated = store.read(context, "session", "summary.md")
    assert truncated is False
    assert summary == "# Summary\n\n"


def test_markdown_memory_store_blocks_path_traversal_and_non_markdown(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root_dir=tmp_path)
    context = MarkdownMemoryContext(agent_name="default", user_id="u1", thread_id="t1")

    with pytest.raises(ValueError, match="path traversal|invalid markdown memory path"):
        store.append(context, "session", "../outside.md", "bad")

    with pytest.raises(ValueError, match="only .md"):
        store.append(context, "session", "notes.txt", "bad")

    with pytest.raises(ValueError, match="home-relative"):
        store.append(context, "session", "~/notes.md", "bad")


def test_markdown_memory_store_compresses_large_memory_with_keywords(tmp_path: Path) -> None:
    store = MarkdownMemoryStore(root_dir=tmp_path)
    context = MarkdownMemoryContext(agent_name="default", user_id="u1", project_id="p1", thread_id="t1")
    content = "\n".join(
        [
            "# 会话工作记录",
            "这是一段普通背景。" * 20,
            "## 需求",
            "- 用户需要 Markdown 文件夹记忆系统。",
            "- 记忆必须有目录层级，方便找到对应记忆。",
            "- 需要压缩逻辑，避免上下文过长。",
            "## 噪音",
            *[f"- 普通日志 {index}: {'x' * 80}" for index in range(80)],
            "## 决策",
            "- 决定先实现本地抽取式压缩。",
            "- TODO: 后续可以接 LLM 摘要。",
        ]
    )
    store.write(context, "session", "working-notes.md", content)

    compression = store.compress(
        context,
        "session",
        "working-notes.md",
        target_path="summaries/working-notes.summary.md",
        max_chars=900,
        keywords=["目录层级", "压缩"],
    )
    compressed, truncated = store.read(context, "session", "summaries/working-notes.summary.md", max_chars=2000)

    assert compression.original_chars > compression.compressed_chars
    assert compression.compressed_chars <= 900
    assert compression.target_path == "summaries/working-notes.summary.md"
    assert compression.truncated is True
    assert truncated is False
    assert "compressed: true" in compressed
    assert "目录层级" in compressed
    assert "压缩" in compressed
