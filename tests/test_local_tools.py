from __future__ import annotations

import asyncio
import json
from typing import Any

from pytest import MonkeyPatch

from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.tools import ToolInvocationService
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_local_tools_write_read_search_and_todo(tmp_path) -> None:
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))
    thread_args = {"_thread_id": "local-tools"}

    write_result = service.call_tool(
        "local_write_file",
        {**thread_args, "path": "notes/plan.md", "content": "JetLinks agent runtime\n工具闭环\n"},
    )
    assert write_result.is_error is False
    assert write_result.structured_content["path"] == "notes/plan.md"

    read_result = service.call_tool("local_read_file", {**thread_args, "path": "notes/plan.md"})
    assert read_result.is_error is False
    assert "工具闭环" in read_result.content[0]["text"]

    search_result = service.call_tool("local_search_text", {**thread_args, "pattern": "agent", "path": "."})
    assert search_result.is_error is False
    assert search_result.structured_content["matches"][0]["path"] == "notes/plan.md"

    add_result = service.call_tool("local_todo", {**thread_args, "action": "add", "text": "补齐本地工具"})
    assert add_result.is_error is False
    assert add_result.structured_content["todos"][0]["text"] == "补齐本地工具"

    complete_result = service.call_tool("local_todo", {**thread_args, "action": "complete", "id": 1})
    assert complete_result.is_error is False
    assert complete_result.structured_content["todos"][0]["done"] is True


def test_present_files_lists_thread_outputs_and_optional_workspace(tmp_path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("present-files")
    store.write_text_artifact(paths, "result.md", "# Result")
    service = ToolInvocationService(artifact_store=store)

    outputs_only = service.call_tool("present_files", {"_thread_id": "present-files"})
    with_workspace = service.call_tool(
        "present_files",
        {"_thread_id": "present-files", "include_workspace": True},
    )

    assert outputs_only.is_error is False
    assert outputs_only.structured_content["files"][0]["scope"] == "outputs"
    assert outputs_only.structured_content["files"][0]["path"] == "result.md"
    assert with_workspace.is_error is False


def test_local_tools_block_path_traversal(tmp_path) -> None:
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = service.call_tool(
        "local_write_file",
        {"_thread_id": "local-tools", "path": "../escape.txt", "content": "blocked"},
    )

    assert result.is_error is True
    assert "path traversal" in result.content[0]["text"].lower()


def test_agent_loop_can_use_local_workspace_tools(tmp_path, monkeypatch: MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        calls.append([tool["function"]["name"] for tool in tools])
        if len(calls) == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_write",
                        name="local_write_file",
                        arguments='{"path":"result.txt","content":"hello from local tool"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_message = next(message for message in messages if message.get("role") == "tool")
        payload = json.loads(tool_message["content"])
        assert payload["isError"] is False
        assert payload["structuredContent"]["path"] == "result.txt"
        return LlmChatResponse(content="已写入。", finish_reason="stop")

    monkeypatch.delenv("LOCAL_SHELL_TOOL_ENABLED", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="local-tool-agent",
        display_name="Local Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["local_write_file", "local_shell_command"],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="写入文件")],
                runtime_options=RuntimeOptions(thread_id="agent-local-tools"),
            ),
        )
    )

    assert result.status == "completed"
    assert result.reply == "已写入。"
    assert calls[0] == ["local_write_file"]
    assert (tmp_path / "agent-local-tools" / "user-data" / "workspace" / "result.txt").read_text() == (
        "hello from local tool"
    )
