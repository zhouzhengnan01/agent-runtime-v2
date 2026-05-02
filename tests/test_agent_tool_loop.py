from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from app.core.agent import AgentRuntime
from app.core.agent.tool_loop import ToolCallingAgentLoop
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig, RuntimeConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.skills import SkillRunner
from app.core.tools import ToolInvocationService, ToolRegistry
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_agent_loop_executes_llm_tool_calls_through_unified_tool_service(
    tmp_path: Path,
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
    tmp_path: Path,
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
    tmp_path: Path,
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
    tmp_path: Path,
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


def test_agent_loop_treats_selected_skill_as_tool_when_no_workflow_mapping(
    tmp_path: Path,
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
        return LlmChatResponse(content="已暴露选中的 Skill 工具。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="direct-skill-agent",
        display_name="Direct Skill Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=["drawio-generation"],
        workflows={"default": "agent_loop"},
    )
    artifact_store = ArtifactStore(root_dir=tmp_path)
    loop = ToolCallingAgentLoop(ToolInvocationService(artifact_store=artifact_store))
    recorder = EventRecorder(agent=agent.name, thread_id="direct-drawio")

    loop_result = asyncio.run(
        loop.run(
            agent_config=agent,
            messages=[Message(role="user", content="帮我画一个工作台原型图")],
            thread_id="direct-drawio",
            recorder=recorder,
            runtime_options=RuntimeOptions(
                thread_id="direct-drawio",
                selected_skills=["drawio-generation"],
            ),
        )
    )
    result = loop_result.result

    assert result.status == "completed"
    assert result.reply == "已暴露选中的 Skill 工具。"
    assert result.metadata["workflow"] == "agent_loop"
    assert seen_tools == ["drawio-generation"]
    event_types = [event.type for event in recorder.events]
    assert "direct_skill.started" not in event_types
    assert "artifact.created" not in event_types
    tools_event = next(event for event in recorder.events if event.type == "tools.available")
    assert tools_event.data["tools"] == ["drawio-generation"]


def test_agent_loop_exposes_runtime_selected_mcp_tools(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config" / "mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "tools.json").write_text(
        """
{
  "tools": [
    {
      "name": "selected_status_tool",
      "title": "Selected Status",
      "description": "Runtime-selected custom MCP status tool.",
      "enabled": true,
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"},
      "source": {"type": "manual", "response_template": "selected ok"}
    }
  ]
}
""",
        encoding="utf-8",
    )
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="已看到本轮选择的 MCP 工具。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    skill_runner = SkillRunner(store)
    runtime = AgentRuntime(artifact_store=store)
    runtime.agent_loop.tool_service.registry = ToolRegistry(
        tmp_path,
        artifact_store=store,
        skill_runner=skill_runner,
    )
    agent = AgentConfig(
        name="runtime-selected-tool-agent",
        display_name="Runtime Selected Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=[],
        workflows={"default": "agent_loop"},
    )

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="检查本轮选择工具")],
                runtime_options=RuntimeOptions(
                    thread_id="selected-mcp-tool",
                    selected_mcp_tools=["selected_status_tool"],
                ),
            ),
        )
    )

    assert result.status == "completed"
    assert seen_tools == ["selected_status_tool"]
    tools_event = next(event for event in events if event.type == "tools.available")
    assert tools_event.data["tools"] == ["selected_status_tool"]
    assert tools_event.data["tool_choice"] == "auto"


def test_agent_loop_ignores_unknown_runtime_selected_mcp_tools(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    complete_calls: list[str] = []
    tool_call_attempts: list[str] = []

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        del self, messages
        complete_calls.append(system_prompt)
        return "没有可用工具，按普通聊天处理。"

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        tool_call_attempts.append("called")
        return LlmChatResponse(content="不应调用。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    skill_runner = SkillRunner(store)
    runtime = AgentRuntime(artifact_store=store)
    runtime.agent_loop.tool_service.registry = ToolRegistry(
        tmp_path,
        artifact_store=store,
        skill_runner=skill_runner,
    )
    agent = AgentConfig(
        name="unknown-selected-tool-agent",
        display_name="Unknown Selected Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=[],
        workflows={"default": "agent_loop"},
    )

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="选择了不存在的 MCP 工具")],
                runtime_options=RuntimeOptions(
                    thread_id="unknown-selected-mcp-tool",
                    selected_mcp_tools=["missing_tool"],
                ),
            ),
        )
    )

    assert result.status == "completed"
    assert result.reply == "没有可用工具，按普通聊天处理。"
    assert len(complete_calls) == 1
    assert tool_call_attempts == []
    tools_event = next(event for event in events if event.type == "tools.available")
    assert tools_event.data["tools"] == []


def test_agent_loop_runtime_selected_mcp_tools_respect_registered_tool_safety(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("LOCAL_SHELL_TOOL_ENABLED", raising=False)
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="本轮安全工具已暴露。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="runtime-selected-local-tool-agent",
        display_name="Runtime Selected Local Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=[],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="本轮选择本地工具")],
                runtime_options=RuntimeOptions(
                    thread_id="selected-safe-local-tools",
                    selected_mcp_tools=["local_write_file", "local_shell_command"],
                ),
            ),
        )
    )

    assert result.status == "completed"
    assert seen_tools == ["local_write_file"]
    tools_event = next(event for event in events if event.type == "tools.available")
    assert tools_event.data["tools"] == ["local_write_file"]


def test_agent_loop_truncates_large_tool_results_before_returning_to_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("large-tool-result")
    large_text = "alpha\n" + ("0123456789" * 800)
    (paths.workspace / "large.txt").write_text(large_text, encoding="utf-8")
    calls: list[list[dict[str, Any]]] = []
    tool_messages: list[dict[str, Any]] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt, tools
        calls.append(list(messages))
        if len(calls) == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_large_read",
                        name="local_read_file",
                        arguments='{"path": "large.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_messages.extend(message for message in messages if message.get("role") == "tool")
        return LlmChatResponse(content="已读取裁剪后的工具结果。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="truncate-tool-agent",
        display_name="Truncate Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        runtime=RuntimeConfig(max_tool_rounds=2, max_tool_result_chars=1000),
        tools=["local_read_file"],
        skills=[],
        workflows={"default": "agent_loop"},
    )
    loop = ToolCallingAgentLoop(ToolInvocationService(artifact_store=store))
    recorder = EventRecorder(agent=agent.name, thread_id=paths.thread_id)

    loop_result = asyncio.run(
        loop.run(
            agent_config=agent,
            messages=[Message(role="user", content="读取大文件")],
            thread_id=paths.thread_id,
            recorder=recorder,
        )
    )

    sent_tool_content = tool_messages[0]["content"]
    payload = json.loads(sent_tool_content)

    assert loop_result.result.status == "completed"
    assert loop_result.result.reply == "已读取裁剪后的工具结果。"
    assert len(sent_tool_content) <= 1000
    assert payload["isError"] is False
    assert payload["structuredContent"]["_truncated"] is True
    assert payload["structuredContent"]["original_json_chars"] > 1000
    assert "alpha" in payload["content"][0]["text"]
    assert "tool result truncated" in payload["content"][0]["text"]
