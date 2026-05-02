from __future__ import annotations

import asyncio
import json
from typing import Any

from pytest import MonkeyPatch

from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import MemoryConfig, ModelConfig, PromptConfig
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.memory import MemoryStore
from app.core.tools import ToolInvocationService
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_memory_store_crud_and_query(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    first = store.remember("default", "用户喜欢中文回答", tags=["preference", "语言"])
    second = store.remember("default", "项目优先实现 memory 功能", tags=["roadmap"])

    assert (tmp_path / "default.json").is_file()
    assert [item.id for item in store.list("default", limit=10)] == [second.id, first.id]
    assert store.list("default", query="中文", limit=10)[0].id == first.id
    assert store.list("default", tags=["roadmap"], limit=10)[0].id == second.id
    assert store.forget("default", first.id) is True
    assert [item.id for item in store.list("default", limit=10)] == [second.id]
    assert store.clear("default") == 1
    assert store.list("default", limit=10) == []


def test_memory_store_global_scope_is_shared(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    item = store.remember("agent-a", "全局偏好", scope="global")

    assert (tmp_path / "global.json").is_file()
    assert store.list("agent-b", scope="global", limit=10)[0].id == item.id
    assert store.list("agent-b", scope="agent", limit=10) == []


def test_memory_tools_through_unified_tool_service(tmp_path) -> None:
    service = ToolInvocationService(memory_store=MemoryStore(root_dir=tmp_path))

    remember_result = service.call_tool(
        "memory_remember",
        {"_agent_name": "default", "text": "JetLinks 使用 agent-v2 分支", "tags": ["repo"]},
    )
    assert remember_result.is_error is False
    memory_id = remember_result.structured_content["memory"]["id"]

    search_result = service.call_tool("memory_search", {"_agent_name": "default", "query": "agent-v2"})
    assert search_result.is_error is False
    assert search_result.structured_content["memories"][0]["id"] == memory_id

    forget_result = service.call_tool("memory_forget", {"_agent_name": "default", "id": memory_id})
    assert forget_result.is_error is False
    assert forget_result.structured_content["removed"] is True


def test_agent_loop_exposes_memory_tools_only_when_enabled(tmp_path, monkeypatch: MonkeyPatch) -> None:
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="ok", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="memory-disabled",
        display_name="Memory Disabled",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["memory_remember", "memory_search", "memory_md_search", "local_todo"],
        memory=MemoryConfig(enabled=False),
    )
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        memory_store=MemoryStore(root_dir=tmp_path / "memory"),
    )

    result = asyncio.run(runtime.run(agent, ChatRequest(messages=[Message(role="user", content="test")])))

    assert result.status == "completed"
    assert seen_tools == ["local_todo"]


def test_agent_loop_exposes_markdown_memory_tools_only_when_markdown_enabled(
    tmp_path,
    monkeypatch: MonkeyPatch,
) -> None:
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="ok", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="markdown-memory-disabled",
        display_name="Markdown Memory Disabled",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["memory_search", "memory_md_search"],
        memory=MemoryConfig(enabled=True, markdown_enabled=False),
    )
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        memory_store=MemoryStore(root_dir=tmp_path / "memory"),
    )

    result = asyncio.run(runtime.run(agent, ChatRequest(messages=[Message(role="user", content="test")])))

    assert result.status == "completed"
    assert seen_tools == ["memory_search"]


def test_agent_loop_can_remember_and_search(tmp_path, monkeypatch: MonkeyPatch) -> None:
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
                        id="call_remember",
                        name="memory_remember",
                        arguments='{"text":"用户希望 README 使用中文","tags":["preference"]}',
                    )
                ],
                finish_reason="tool_calls",
            )
        if len(calls) == 2:
            remember_message = next(message for message in messages if message.get("role") == "tool")
            remember_payload = json.loads(remember_message["content"])
            assert remember_payload["isError"] is False
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_search",
                        name="memory_search",
                        arguments='{"query":"README"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        search_payload = json.loads(tool_messages[-1]["content"])
        assert search_payload["structuredContent"]["memories"][0]["text"] == "用户希望 README 使用中文"
        return LlmChatResponse(content="已记住。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="memory-agent",
        display_name="Memory Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["memory_remember", "memory_search"],
        memory=MemoryConfig(enabled=True, inject_context=False),
    )
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        memory_store=MemoryStore(root_dir=tmp_path / "memory"),
    )

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="记住我的偏好")],
                runtime_options=RuntimeOptions(thread_id="memory-loop"),
            ),
        )
    )

    assert result.status == "completed"
    assert result.reply == "已记住。"
    assert calls[0] == ["memory_remember", "memory_search"]
    assert MemoryStore(root_dir=tmp_path / "memory").list("memory-agent", query="中文", limit=10)


def test_agent_loop_injects_memory_context_when_enabled(tmp_path, monkeypatch: MonkeyPatch) -> None:
    store = MemoryStore(root_dir=tmp_path / "memory")
    store.remember("memory-context-agent", "用户偏好：所有 README 用中文。", tags=["preference"])
    seen_prompt = ""

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        nonlocal seen_prompt
        seen_prompt = system_prompt
        return LlmChatResponse(content="ok", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="memory-context-agent",
        display_name="Memory Context Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["memory_search"],
        memory=MemoryConfig(enabled=True, inject_context=True, max_items=5),
        prompts=PromptConfig(system="基础系统提示。"),
    )
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        memory_store=store,
    )

    result, events = asyncio.run(
        runtime.run_with_events(agent, ChatRequest(messages=[Message(role="user", content="README 怎么写")]))
    )

    assert result.status == "completed"
    assert "基础系统提示。" in seen_prompt
    assert "用户偏好：所有 README 用中文。" in seen_prompt
    memory_events = [event for event in events if event.type == "memory.context.loaded"]
    assert memory_events[0].data["count"] == 1
