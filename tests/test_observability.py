from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest
from pytest import MonkeyPatch

from app.api import agents as agents_api
from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.events import EventRecorder, RunEventStore
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.main import create_app
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_agent_run_persists_observable_timeline(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
    ) -> str:
        del self, system_prompt, messages
        return "observed"

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    store = RunEventStore(tmp_path / "runs")
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"), run_event_store=store)
    agent = AgentConfig(
        name="observable-agent",
        display_name="Observable Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=[],
        skills=[],
    )

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="hi")],
                runtime_options=RuntimeOptions(thread_id="observable-thread"),
            ),
        )
    )

    run_id = result.metadata["run_id"]
    saved = store.get(run_id)
    assert saved["run_id"] == run_id
    assert saved["agent"] == "observable-agent"
    assert saved["thread_id"] == "observable-thread"
    assert saved["status"] == "completed"
    assert saved["workflow"] == "agent_loop"
    assert saved["event_count"] == len(events)
    assert saved["events"][0]["type"] == "run.started"
    assert saved["events"][0]["data"]["run_id"] == run_id
    assert isinstance(saved["events"][0]["data"]["elapsed_ms"], float)
    assert saved["events"][-1]["type"] == "run.completed"
    assert saved["agent_snapshot"]["model"]["api_key"] == "********"
    assert saved["request_snapshot"]["runtime_options"]["thread_id"] == "observable-thread"
    llm_completed = next(event for event in saved["events"] if event["type"] == "llm.request.completed")
    assert llm_completed["data"]["mode"] == "chat"
    assert llm_completed["data"]["content_chars"] == len("observed")
    assert store.list()[0]["run_id"] == run_id


def test_tool_events_include_duration_and_arguments(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = 0

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        nonlocal calls
        del self, system_prompt, messages, tools
        calls += 1
        if calls == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_status",
                        name="jetlinks_runtime_status",
                        arguments='{"probe": true}',
                    )
                ],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            )
        return LlmChatResponse(content="done", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    monkeypatch.setattr(OpenAICompatibleClient, "configured", property(lambda self: True))
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        run_event_store=RunEventStore(tmp_path / "runs"),
    )
    agent = AgentConfig(
        name="tool-observable-agent",
        display_name="Tool Observable Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["jetlinks_runtime_status"],
        skills=[],
    )

    _result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="检查")],
                runtime_options=RuntimeOptions(thread_id="tool-observable-thread"),
            ),
        )
    )

    completed = next(event for event in events if event.type == "tool.completed")
    assert completed.data["duration_ms"] >= 0
    assert completed.data["arguments"] == {"probe": True}
    assert completed.data["structured_content"]["arguments"]["probe"] is True
    llm_started = next(event for event in events if event.type == "llm.request.started")
    llm_completed = next(event for event in events if event.type == "llm.request.completed")
    assert llm_started.data["mode"] == "tool_calling"
    assert llm_started.data["tool_count"] >= 1
    assert "jetlinks_runtime_status" in llm_started.data["tools"]
    assert llm_completed.data["finish_reason"] == "tool_calls"
    assert llm_completed.data["tool_call_count"] == 1
    assert llm_completed.data["usage"]["total_tokens"] == 14


def test_runtime_event_detail_logs_full_redacted_payload(
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import app.core.events as events_module

    monkeypatch.setattr(events_module, "RUNTIME_EVENT_TRACE_PAYLOADS", True)
    monkeypatch.setattr(events_module, "RUNTIME_EVENT_TRACE_MAX_CHARS", 0)
    recorder = EventRecorder(agent="observable-agent", thread_id="observable-detail-thread")

    with caplog.at_level("INFO", logger="uvicorn.error"):
        recorder.emit(
            "run.completed",
            {
                "result": {
                    "agent": "observable-agent",
                    "thread_id": "observable-detail-thread",
                    "reply": "日志详情测试",
                    "metadata": {
                        "api_key": "runtime-secret-key",
                        "max_tokens": 128,
                    },
                }
            },
        )

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "runtime event detail type=run.completed" in logs
    assert "observable-detail-thread" in logs
    assert "日志详情测试" in logs
    assert '"api_key": "********"' in logs
    assert '"max_tokens": 128' in logs
    assert "runtime-secret-key" not in logs


def test_tool_failures_include_error_code_and_debug_diagnostics(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        nonlocal calls
        del self, system_prompt, messages, tools
        calls += 1
        if calls == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_bad_args",
                        name="jetlinks_runtime_status",
                        arguments="{bad json",
                    )
                ],
                finish_reason="tool_calls",
            )
        return LlmChatResponse(content="handled", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    event_store = RunEventStore(tmp_path / "runs")
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        run_event_store=event_store,
    )
    agent = AgentConfig(
        name="tool-failure-agent",
        display_name="Tool Failure Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["jetlinks_runtime_status"],
        skills=[],
    )

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="检查")],
                runtime_options=RuntimeOptions(thread_id="tool-failure-thread"),
            ),
        )
    )

    failed = next(event for event in events if event.type == "tool.failed")
    assert failed.data["error_code"] == "TOOL_ARGUMENTS_INVALID"
    assert failed.data["structured_content"]["recoverable"] is True
    bundle = event_store.debug_bundle(result.metadata["run_id"])
    assert bundle["schema"] == "jetlinks-agent-run-debug-bundle.v1"
    assert bundle["diagnostics"]["failure_count"] == 1
    assert bundle["diagnostics"]["failures"][0]["error_code"] == "TOOL_ARGUMENTS_INVALID"


def test_run_observability_api_returns_saved_timeline(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DISABLED", "1")
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        run_event_store=RunEventStore(tmp_path / "runs"),
    )
    monkeypatch.setattr(agents_api, "runtime", runtime)
    client = TestClient(create_app())

    response = client.post(
        "/api/agents/default/runs",
        json={
            "messages": [{"role": "user", "content": "你好"}],
            "runtime_options": {"thread_id": "api-observable-thread"},
        },
    )

    assert response.status_code == 200
    run_id = response.json()["metadata"]["run_id"]
    recent = client.get("/api/agents/runs/recent")
    detail = client.get(f"/api/agents/runs/{run_id}")
    events = client.get(f"/api/agents/runs/{run_id}/events")
    bundle = client.get(f"/api/agents/runs/{run_id}/debug-bundle")

    assert recent.status_code == 200
    assert recent.json()["runs"][0]["run_id"] == run_id
    assert detail.status_code == 200
    assert detail.json()["run_id"] == run_id
    assert detail.json()["result"]["thread_id"] == "api-observable-thread"
    assert detail.json()["agent_snapshot"]["model"]["api_key"] in {None, "********"}
    assert events.status_code == 200
    assert events.json()["events"][0]["data"]["run_id"] == run_id
    assert bundle.status_code == 200
    assert bundle.json()["schema"] == "jetlinks-agent-run-debug-bundle.v1"
    assert bundle.json()["run"]["run_id"] == run_id


def test_agent_stream_api_emits_tool_loop_events_as_sse(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = 0

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        nonlocal calls
        del self, system_prompt, messages, tools
        calls += 1
        if calls == 1:
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
        return LlmChatResponse(content="streamed done", finish_reason="stop")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    monkeypatch.setattr(OpenAICompatibleClient, "configured", property(lambda self: True))
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        run_event_store=RunEventStore(tmp_path / "runs"),
    )
    monkeypatch.setattr(agents_api, "runtime", runtime)
    client = TestClient(create_app())

    response = client.post(
        "/api/agents/default/runs/stream",
        json={
            "messages": [{"role": "user", "content": "检查运行时状态"}],
            "runtime_options": {
                "thread_id": "sse-tool-loop-thread",
                "selected_mcp_tools": ["jetlinks_runtime_status"],
                "mode": "autonomous",
                "config_options": {"max_tool_rounds": 4},
            },
        },
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_types = [event["type"] for event in events]
    assert event_types[0] == "run.started"
    assert "llm.request.started" in event_types
    assert "llm.request.completed" in event_types
    assert "tool.calls.started" in event_types
    assert "tool.started" in event_types
    assert "tool.completed" in event_types
    assert "agent.message.delta" in event_types
    assert event_types[-1] == "run.completed"

    started = next(event for event in events if event["type"] == "llm.request.started")
    completed = next(event for event in events if event["type"] == "tool.completed")
    final = events[-1]["data"]["result"]
    assert started["data"]["mode"] == "tool_calling"
    assert "jetlinks_runtime_status" in started["data"]["tools"]
    assert completed["data"]["tool_name"] == "jetlinks_runtime_status"
    assert final["reply"] == "streamed done"
    assert final["metadata"]["mode"] == "autonomous"


def _parse_sse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame in text.strip().split("\n\n"):
        data_lines = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        payload = json.loads("\n".join(data_lines))
        events.append(payload)
    return events
