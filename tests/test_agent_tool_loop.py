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
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_agent_loop_executes_llm_tool_calls_through_unified_tool_service(
    tmp_path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        calls.append({"messages": list(messages), "tools": tools})
        if len(calls) == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_status",
                        name="jetlinks_runtime_status",
                        arguments='{"probe": true}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_message = next(message for message in messages if message.get("role") == "tool")
        tool_payload = json.loads(tool_message["content"])
        assert tool_payload["isError"] is False
        assert "JetLinks Agent Runtime v2 MCP endpoint is reachable." in tool_payload["content"][0]["text"]
        return LlmChatResponse(content="运行时状态正常。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="tool-agent",
        display_name="Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["jetlinks_runtime_status"],
        skills=[],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    request = ChatRequest(
        messages=[Message(role="user", content="检查运行时状态")],
        runtime_options=RuntimeOptions(thread_id="tool-loop"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.status == "completed"
    assert result.reply == "运行时状态正常。"
    assert result.metadata["workflow"] == "agent_loop"
    assert result.metadata["tool_rounds"] == 2
    assert result.metadata["tool_call_count"] == 1
    assert len(calls) == 2
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["jetlinks_runtime_status"]
    event_types = [event.type for event in events]
    assert "tool.started" in event_types
    assert "tool.completed" in event_types
    assert events[-1].type == "run.completed"


def test_agent_loop_exposes_only_agent_declared_tools(
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
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="没有调用工具。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="restricted-agent",
        display_name="Restricted Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["jetlinks_runtime_status"],
        skills=[],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(messages=[Message(role="user", content="普通聊天")]),
        )
    )

    assert result.status == "completed"
    assert seen_tools == ["jetlinks_runtime_status"]


def test_agent_loop_skips_tools_when_model_tool_choice_is_none(
    tmp_path,
    monkeypatch: MonkeyPatch,
) -> None:
    complete_calls: list[str] = []
    tool_calls: list[str] = []

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        complete_calls.append(system_prompt)
        return "普通聊天已完成。"

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        tool_calls.append(system_prompt)
        return LlmChatResponse(content="不应调用。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="tool-choice-none-agent",
        display_name="Tool Choice None Agent",
        model=ModelConfig(
            base_url="http://llm.local/v1",
            api_key="key",
            model="tool-model",
            tool_choice="none",
        ),
        tools=["jetlinks_runtime_status"],
        skills=[],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result, events = asyncio.run(
        runtime.run_with_events(agent, ChatRequest(messages=[Message(role="user", content="普通聊天")]))
    )

    assert result.status == "completed"
    assert result.reply == "普通聊天已完成。"
    assert len(complete_calls) == 1
    assert tool_calls == []
    tools_event = next(event for event in events if event.type == "tools.available")
    assert tools_event.data["tool_count"] == 0
    assert tools_event.data["tool_choice"] == "none"


def test_agent_loop_can_execute_skill_backed_tools(
    tmp_path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        calls.append({"messages": list(messages), "tools": tools})
        if len(calls) == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_markdown",
                        name="markdown-rendering",
                        arguments='{"title":"统一工具层","summary":"Skill 也通过 ToolInvocationService 执行"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_message = next(message for message in messages if message.get("role") == "tool")
        payload = json.loads(tool_message["content"])
        assert payload["isError"] is False
        assert payload["structuredContent"]["skill_name"] == "markdown-rendering"
        assert payload["structuredContent"]["thread_id"] == "skill-loop"
        assert payload["structuredContent"]["artifacts"][0]["name"] == "result.md"
        return LlmChatResponse(content="Markdown 已生成。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="skill-tool-agent",
        display_name="Skill Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=["markdown-rendering"],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="生成 markdown")],
                runtime_options=RuntimeOptions(thread_id="skill-loop"),
            ),
        )
    )

    assert result.status == "completed"
    assert result.reply == "Markdown 已生成。"
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["markdown-rendering"]
