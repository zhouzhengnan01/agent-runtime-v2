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
                        id="call_todo",
                        name="local_todo",
                        arguments='{"action": "add", "text": "检查运行时状态"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_message = next(message for message in messages if message.get("role") == "tool")
        tool_payload = json.loads(tool_message["content"])
        assert tool_payload["isError"] is False
        assert tool_payload["structuredContent"]["todos"][0]["text"] == "检查运行时状态"
        return LlmChatResponse(content="运行时状态正常。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="tool-agent",
        display_name="Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["local_todo"],
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
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["local_todo"]
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
        tools=["local_todo"],
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
    assert seen_tools == ["local_todo"]


def test_agent_loop_exposes_explicit_selected_mcp_tools(
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
        return LlmChatResponse(content="已看到显式 MCP 工具。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    config_dir = tmp_path / "config" / "mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "tools.json").write_text(
        """
{
  "tools": [
    {
      "name": "mcp_demo_status",
      "title": "MCP Demo Status",
      "description": "A selected MCP demo tool.",
      "enabled": true,
      "input_schema": {"type": "object"},
      "source": {"type": "manual", "response_template": "ok"}
    }
  ]
}
""",
        encoding="utf-8",
    )
    artifact_store = ArtifactStore(root_dir=tmp_path / "runtime")
    runtime = AgentRuntime(artifact_store=artifact_store)
    runtime.agent_loop.tool_service.registry.root_dir = tmp_path
    runtime.agent_loop.tool_service.registry.config_path = config_dir / "tools.json"
    agent = AgentConfig(
        name="selected-mcp-agent",
        display_name="Selected MCP Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=[],
        workflows={"default": "agent_loop"},
    )

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="使用我显式选择的 MCP 工具")],
                runtime_options=RuntimeOptions(
                    thread_id="selected-mcp",
                    selected_mcp_tools=["mcp_demo_status"],
                ),
            ),
        )
    )

    assert result.status == "completed"
    assert seen_tools == ["mcp_demo_status"]


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
        tools=["local_todo"],
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


def test_agent_loop_directly_executes_explicit_selected_generation_skill(
    tmp_path,
    monkeypatch: MonkeyPatch,
) -> None:
    tool_call_attempts: list[str] = []
    planner_prompts: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        tool_call_attempts.append("called")
        return LlmChatResponse(content="不应调用模型工具。", finish_reason="stop")

    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        del self, system_prompt
        planner_prompts.append(messages[0].content)
        return """
        {
          "diagram_type": "prototype_wireframe",
          "visual_style": "polished",
          "swimlanes": ["导航与入口", "任务编排", "输出预览", "质量反馈"],
          "nodes": ["左侧导航", "Agent 列表", "聊天主区", "技能选择", "任务输入", "运行事件", "文件输出", "PNG 预览", "Spec 面板", "校验面板"],
          "lane_nodes": {
            "导航与入口": ["左侧导航", "Agent 列表"],
            "任务编排": ["聊天主区", "技能选择", "任务输入", "运行事件"],
            "输出预览": ["文件输出", "PNG 预览", "Spec 面板"],
            "质量反馈": ["校验面板"]
          },
          "edges": [
            ["左侧导航", "Agent 列表"],
            ["Agent 列表", "聊天主区"],
            ["聊天主区", "技能选择"],
            ["技能选择", "任务输入"],
            ["任务输入", "运行事件"],
            ["运行事件", "文件输出"],
            ["文件输出", "PNG 预览"],
            ["运行事件", "Spec 面板"],
            ["Spec 面板", "校验面板"]
          ]
        }
        """

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    monkeypatch.setenv("LLM_SPEC_PLANNER", "1")
    agent = AgentConfig(
        name="direct-skill-agent",
        display_name="Direct Skill Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=["drawio-generation"],
        workflows={"default": "agent_loop"},
    )
    artifact_store = ArtifactStore(root_dir=tmp_path)
    runtime = AgentRuntime(artifact_store=artifact_store)

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="帮我画一个工作台原型图")],
                runtime_options=RuntimeOptions(
                    thread_id="direct-drawio",
                    selected_skills=["drawio-generation"],
                ),
            ),
        )
    )

    assert result.status == "completed"
    assert result.metadata["workflow"] == "agent_loop"
    assert result.metadata["direct_skill"] is True
    assert result.metadata["skill_name"] == "drawio-generation"
    assert tool_call_attempts == []
    assert planner_prompts
    assert result.spec is not None
    assert result.spec["planner"]["mode"] == "llm"
    assert result.spec["visual_style"] == "polished"
    assert result.spec["swimlanes"] == ["导航与入口", "任务编排", "输出预览", "质量反馈"]
    artifact_names = {artifact.name for artifact in result.artifacts}
    assert "prototype.drawio" in artifact_names
    assert "prototype.png" in artifact_names
    assert (tmp_path / "direct-drawio" / "user-data" / "outputs" / "prototype.drawio").is_file()
    assert (tmp_path / "direct-drawio" / "user-data" / "outputs" / "prototype.png").is_file()
    event_types = [event.type for event in events]
    assert "direct_skill.started" in event_types
    assert "spec.planner.started" in event_types
    assert "spec.planner.completed" in event_types
    assert "artifact.created" in event_types


def test_stream_agent_loop_runs_explicit_skill_when_llm_is_not_configured(
    tmp_path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_SPEC_PLANNER", "1")
    agent = AgentConfig(
        name="direct-skill-agent",
        display_name="Direct Skill Agent",
        model=ModelConfig(base_url=None, api_key=None, model="tool-model"),
        tools=[],
        skills=["drawio-generation"],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    async def collect() -> list[Any]:
        return [
            event
            async for event in runtime.iter_events(
                agent,
                ChatRequest(
                    messages=[Message(role="user", content="帮我画一个工作台原型图")],
                    runtime_options=RuntimeOptions(
                        thread_id="direct-drawio-stream",
                        selected_skills=["drawio-generation"],
                    ),
                ),
            )
        ]

    events = asyncio.run(collect())
    event_types = [event.type for event in events]
    final = events[-1].data["result"]

    assert event_types[:3] == ["run.started", "llm.started", "direct_skill.started"]
    assert "spec.planner.completed" in event_types
    assert "artifact.created" in event_types
    assert final["metadata"]["direct_skill"] is True
    assert final["metadata"]["llm_configured"] is False
    assert final["spec"]["planner"]["reason"] == "llm_not_configured"
    names = {artifact["name"] for artifact in final["artifacts"]}
    assert "prototype.drawio" in names
    assert "prototype.png" in names
