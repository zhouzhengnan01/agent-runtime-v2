from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from app.core.agent.tool_loop import ToolCallingAgentLoop
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.tools import ToolInvocationService
from app.schemas import Message, RuntimeOptions


def test_delegate_task_runs_child_model_call(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _write_agent_config(tmp_path)
    captured: dict[str, Any] = {}

    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
    ) -> str:
        captured["model"] = self.model
        captured["system_prompt"] = system_prompt
        captured["messages"] = messages
        return "子任务结论：配置需要补齐超时边界。"

    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    service = ToolInvocationService(
        root_dir=tmp_path,
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
    )

    result = service.call_tool(
        "delegate_task",
        {
            "_agent_name": "default",
            "goal": "审查 agent loop 的失败恢复策略",
            "context": "父智能体已发现工具调用闭环和本地 memory。",
        },
    )

    assert result.is_error is False
    assert "子任务结论" in result.content[0]["text"]
    assert result.structured_content["agent_name"] == "default"
    assert result.structured_content["model"] == "delegate-model"
    assert "delegated subagent" in captured["system_prompt"]
    assert "审查 agent loop" in captured["messages"][0].content
    assert "父智能体已发现" in captured["messages"][0].content


def test_agent_loop_can_call_delegate_task_tool(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    _write_agent_config(tmp_path)
    seen_tool_lists: list[list[str]] = []

    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
    ) -> str:
        del self, system_prompt, messages
        return "delegate summary"

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        del self, system_prompt
        seen_tool_lists.append([tool["function"]["name"] for tool in tools])
        if len(seen_tool_lists) == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_delegate",
                        name="delegate_task",
                        arguments='{"goal":"并行检查 README 缺口","context":"只需要总结"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_message = next(message for message in messages if message.get("role") == "tool")
        payload = json.loads(tool_message["content"])
        assert payload["isError"] is False
        assert payload["structuredContent"]["tool_name"] == "delegate_task"
        assert payload["content"][0]["text"] == "delegate summary"
        return LlmChatResponse(content="已合并子任务结论。", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="default",
        display_name="Default",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="parent-model"),
        tools=["delegate_task"],
        workflows={"default": "agent_loop"},
    )
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    loop = ToolCallingAgentLoop(ToolInvocationService(root_dir=tmp_path, artifact_store=artifact_store))
    recorder = EventRecorder(agent=agent.name, thread_id="delegate-loop")

    result = asyncio.run(
        loop.run(
            agent_config=agent,
            messages=[Message(role="user", content="拆分检查 runtime 能力")],
            thread_id="delegate-loop",
            recorder=recorder,
            runtime_options=RuntimeOptions(thread_id="delegate-loop"),
        )
    ).result

    assert result.status == "completed"
    assert result.reply == "已合并子任务结论。"
    assert seen_tool_lists[0] == ["delegate_task"]
    assert "tool.completed" in [event.type for event in recorder.events]


def _write_agent_config(root_dir: Path) -> None:
    config_dir = root_dir / "config" / "agents"
    config_dir.mkdir(parents=True)
    (config_dir / "default.json").write_text(
        json.dumps(
            {
                "name": "default",
                "display_name": "Delegate Child",
                "model": {
                    "provider": "openai_compatible",
                    "model": "delegate-model",
                    "base_url": "http://llm.local/v1",
                    "api_key": "key",
                    "tool_choice": "none",
                },
                "runtime": {"stateless": True},
                "tools": [],
                "skills": [],
                "workflows": {"default": "agent_loop"},
                "prompts": {"system": "Base child prompt."},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
