from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from app.core.agent import AgentRuntime
from app.core.agent.context import ConversationContextManager
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig, RuntimeConfig
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_context_compaction_preserves_tail_tool_pair() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "head request"},
        {"role": "assistant", "content": "old analysis " * 120},
        {"role": "user", "content": "old follow-up " * 120},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "local_read_file", "arguments": '{"path":"a.txt"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_read", "content": "file result"},
        {"role": "assistant", "content": "tool result handled"},
        {"role": "user", "content": "latest task"},
    ]
    manager = ConversationContextManager(max_chars=1000, keep_first_messages=1, keep_last_messages=3)

    result = manager.compact(messages)

    assert result.compacted is True
    assert result.summarized_message_count == 2
    assert result.messages[0]["content"] == "head request"
    assert "CONTEXT COMPACTION" in result.messages[1]["content"]
    assert result.messages[2]["tool_calls"][0]["function"]["name"] == "local_read_file"
    assert result.messages[3]["role"] == "tool"
    assert result.messages[-1]["content"] == "latest task"


def test_agent_loop_compacts_large_context_before_model_call(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    captured_messages: list[dict[str, Any]] = []

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
    ) -> str:
        del self, system_prompt
        captured_messages.extend(dict(message) for message in messages)
        return "上下文已压缩。"

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    agent = AgentConfig(
        name="context-agent",
        display_name="Context Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="context-model", tool_choice="none"),
        runtime=RuntimeConfig(
            context_compression_enabled=True,
            context_max_chars=1000,
            context_keep_first_messages=1,
            context_keep_last_messages=2,
        ),
        tools=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[
            Message(role="user", content="first requirement"),
            Message(role="assistant", content="old answer " * 200),
            Message(role="user", content="old question " * 200),
            Message(role="assistant", content="another old answer " * 200),
            Message(role="user", content="current task"),
        ],
        runtime_options=RuntimeOptions(thread_id="context-compaction"),
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.reply == "上下文已压缩。"
    assert result.metadata["context_compaction_count"] == 1
    assert [event.type for event in events].count("context.compacted") == 1
    assert captured_messages[0]["content"] == "first requirement"
    assert "CONTEXT COMPACTION" in captured_messages[1]["content"]
    assert captured_messages[-1]["content"] == "current task"
