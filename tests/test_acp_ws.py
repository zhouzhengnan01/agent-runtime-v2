from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from acp.schema import PromptResponse, SessionNotification
from fastapi.testclient import TestClient

from app.api import acp as acp_api
from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfigLoader
from app.core.llm.openai_compatible import LlmChatResponse, OpenAICompatibleClient
from app.core.runtime import ModelManager
from app.schemas import AgentRunResult, ArtifactRef, ChatEvent, ChatRequest
from app.schemas import Message
from app.main import create_app
from app.protocols.acp.adapter import _event_to_update
from app.protocols.acp import transport_ws as acp_transport_ws


class CapturingAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore, *, block: bool = False) -> None:
        self.artifact_store = artifact_store
        self.block = block
        self.requests: list[ChatRequest] = []

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        self.requests.append(request)
        yield ChatEvent(type="run.started", data={"agent": agent_config.name})
        if self.block:
            await asyncio.sleep(60)
            return
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "acp-test-thread",
            reply="ok",
        )
        yield ChatEvent(type="run.completed", data={"result": result.model_dump()})


class FailingAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore, message: str) -> None:
        self.artifact_store = artifact_store
        self.message = message

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        if False:
            yield ChatEvent(type="run.started", data={})
        raise RuntimeError(self.message)


class HttpStatusFailingAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        if False:
            yield ChatEvent(type="run.started", data={})
        http_request = httpx.Request("POST", "http://model.example/v1/chat/completions")
        http_response = httpx.Response(
            500,
            request=http_request,
            json={"error": {"message": "", "type": "InternalServerError", "param": None, "code": 500}},
        )
        raise httpx.HTTPStatusError(
            "Server error '500 Internal Server Error'",
            request=http_request,
            response=http_response,
        )


class ArtifactAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "acp-artifact-thread",
            reply="生成完成。",
            artifacts=[
                ArtifactRef(
                    thread_id=request.runtime_options.thread_id or "acp-artifact-thread",
                    path="/mnt/user-data/outputs/reports/result.md",
                    name="result.md",
                    mime_type="text/markdown",
                    kind="markdown",
                    size=12,
                    preview_url="/api/artifacts/acp-artifact-thread/mnt/user-data/outputs/reports/result.md",
                    download_url="/api/artifacts/acp-artifact-thread/mnt/user-data/outputs/reports/result.md?download=true",
                )
            ],
        )
        yield ChatEvent(type="run.completed", data={"result": result.model_dump()})


class ToolEventAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "acp-tool-events",
            reply="ok",
        )
        yield ChatEvent(type="run.started", data={"agent": agent_config.name, "sequence": 0})
        yield ChatEvent(
            type="tool.started",
            data={"tool_name": "artifact_write", "tool_call_id": "call_write", "sequence": 1},
        )
        yield ChatEvent(
            type="tool.completed",
            data={
                "tool_name": "artifact_write",
                "tool_call_id": "call_write",
                "sequence": 2,
                "is_error": False,
                "structured_content": {"path": "outputs/demo.md"},
            },
        )
        yield ChatEvent(
            type="tool.failed",
            data={
                "tool_name": "local_read_file",
                "tool_call_id": "call_read",
                "sequence": 3,
                "is_error": True,
                "structured_content": {"error": "not found"},
            },
        )
        yield ChatEvent(type="run.completed", data={"result": result.model_dump(), "sequence": 4})


class PlanDiffEventAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "acp-plan-diff",
            reply="ok",
        )
        yield ChatEvent(type="run.started", data={"workflow": "agent_loop"})
        yield ChatEvent(type="tool.started", data={"tool_name": "local_write_file", "tool_call_id": "call-write"})
        yield ChatEvent(
            type="tool.completed",
            data={
                "tool_name": "local_write_file",
                "tool_call_id": "call-write",
                "structured_content": {"path": "/mnt/user-data/workspace/demo.txt"},
            },
        )
        yield ChatEvent(type="run.completed", data={"result": result.model_dump()})


class InputRequiredAcpRuntime(AgentRuntime):
    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        reply: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.reply = reply
        self.metadata = metadata or {}

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "acp-input-required",
            reply=self.reply,
            metadata=dict(self.metadata),
        )
        yield ChatEvent(type="run.completed", data={"result": result.model_dump()})


class FencedJsonAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        reply = (
            "```json\n"
            "[\n"
            "  {\n"
            '    "reviewSourceId": "source-1",\n'
            '    "reviewEventId": "event-1",\n'
            '    "hit": 0,\n'
            '    "result": "未发现杂物堆积。"\n'
            "  }\n"
            "]\n"
            "```"
        )
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "acp-fenced-json",
            reply=reply,
        )
        yield ChatEvent(type="run.completed", data={"result": result.model_dump()})


class StreamingFencedJsonAcpRuntime(AgentRuntime):
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        del request
        interim = "```json\n[{\"reviewSourceId\":\"source-1\",\"hit\":0}]\n```"
        final = (
            "```json\n"
            "[\n"
            "  {\n"
            '    "reviewSourceId": "source-1",\n'
            '    "reviewEventId": "event-final",\n'
            '    "hit": 0,\n'
            '    "result": "最终复判结果。"\n'
            "  }\n"
            "]\n"
            "```"
        )
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id="acp-streaming-fenced-json",
            reply=final,
        )
        yield ChatEvent(type="agent.message.delta", data={"text": interim})
        yield ChatEvent(type="agent.message", data={"text": interim})
        yield ChatEvent(type="run.completed", data={"result": result.model_dump()})


def test_acp_websocket_prompt_streams_runtime_events() -> None:
    client = TestClient(create_app())
    thread_id = f"acp-ws-test-{uuid.uuid4().hex}"

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        initialized = websocket.receive_json()
        assert initialized["result"]["agentInfo"]["name"] == "jetlinks-agent-runtime-v2"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                    "params": _acp_params(thread_id=thread_id, cwd="/tmp"),
            }
        )
        created = websocket.receive_json()
        session_id = created["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                    "params": _acp_params(
                        sessionId=session_id,
                        thread_id=thread_id,
                        prompt=[{"type": "text", "text": "人员翻越围栏进入禁区"}],
                        runtime_options={"workflow": "evidence_first_detection"},
                    ),
            }
        )

        updates: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(30):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet)
                continue
            if packet.get("id") == 3:
                final = packet
                break

        assert final is not None
        PromptResponse.model_validate(final["result"])
        assert final["result"]["stopReason"] == "end_turn"
        assert final["result"]["_meta"]["jetlinks"]["stopReason"] == "input_required"
        assert "待上传" in final["result"]["result"]["reply"]
        assert final["result"]["result"]["metadata"]["requires_input"] is True
        assert final["result"]["result"]["metadata"]["required_inputs"] == [
            {
                "type": "image",
                "accept": "image/*",
                "required": True,
                "reason": "The agent requires an uploaded image.",
            },
            {
                "type": "video",
                "accept": "video/*",
                "required": True,
                "reason": "The agent requires an uploaded video.",
            },
        ]
        event_types = [
            update["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"]
            for update in updates
            if "_meta" in update["params"]["update"]
        ]
        _assert_valid_acp_updates(updates)
        session_update_types = {update["params"]["update"]["sessionUpdate"] for update in updates}
        assert "run.started" in event_types
        assert "agent.message" in event_types
        assert "agent_thought_chunk" in session_update_types
        assert "runtime_event" not in session_update_types
        assert "agent_message" not in session_update_types


def test_acp_websocket_prompt_sends_keepalive_during_long_runtime(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"), block=True)
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_transport_ws, "ACP_PROMPT_KEEPALIVE_SECONDS", 0.01)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-keepalive", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "long task"}]},
            }
        )

        updates: list[dict[str, Any]] = []
        for _ in range(10):
            packet = websocket.receive_json()
            if packet.get("method") != "session/update":
                continue
            updates.append(packet["params"]["update"])
            runtime_event = packet["params"]["update"].get("_meta", {}).get("jetlinksRuntimeEvent", {})
            if runtime_event.get("type") == "acp.prompt.keepalive":
                break

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/cancel",
                "params": {"sessionId": session_id},
            }
        )
        for _ in range(5):
            packet = websocket.receive_json()
            if packet.get("method") != "session/update":
                continue
            updates.append(packet["params"]["update"])
            runtime_event = packet["params"]["update"].get("_meta", {}).get("jetlinksRuntimeEvent", {})
            if runtime_event.get("type") == "acp.prompt.keepalive.completed":
                break

    keepalive_updates = [
        update
        for update in updates
        if update.get("_meta", {}).get("jetlinksRuntimeEvent", {}).get("type") == "acp.prompt.keepalive"
    ]
    progress_updates = [
        update
        for update in updates
        if update.get("_meta", {}).get("jetlinksRuntimeEvent", {}).get("type") == "acp.prompt.progress"
    ]
    wait_message_updates = [
        update
        for update in updates
        if update.get("_meta", {}).get("jetlinksRuntimeEvent", {}).get("type") == "acp.prompt.wait_message"
    ]
    assert progress_updates
    assert progress_updates[0]["sessionUpdate"] == "agent_thought_chunk"
    assert progress_updates[0]["content"] == {"type": "text", "text": "正在处理，请等待..."}
    assert wait_message_updates
    assert wait_message_updates[0]["sessionUpdate"] == "agent_thought_chunk"
    assert wait_message_updates[0]["content"] == {"type": "text", "text": "已收到请求，正在处理，请等待..."}
    assert not [
        update
        for update in wait_message_updates
        if update.get("sessionUpdate") == "agent_message_chunk"
    ]
    assert keepalive_updates
    assert keepalive_updates[0]["sessionUpdate"] == "tool_call"
    assert keepalive_updates[0]["toolCallId"] == "acp-prompt-keepalive"
    assert keepalive_updates[0]["status"] == "in_progress"
    keepalive_completed = [
        update
        for update in updates
        if update.get("_meta", {}).get("jetlinksRuntimeEvent", {}).get("type") == "acp.prompt.keepalive.completed"
    ]
    assert keepalive_completed
    assert keepalive_completed[0]["sessionUpdate"] == "tool_call_update"
    assert keepalive_completed[0]["toolCallId"] == "acp-prompt-keepalive"
    assert keepalive_completed[0]["status"] == "completed"


def test_acp_websocket_prompt_emits_visible_model_error_chunk(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = HttpStatusFailingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-visible-model-error", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]},
            }
        )

        visible_error = ""
        final_reply = ""
        for _ in range(20):
            packet = websocket.receive_json()
            if packet.get("id") == 2 and isinstance(packet.get("result"), dict):
                final_reply = packet["result"].get("result", {}).get("reply") or ""
            if packet.get("method") != "session/update":
                if final_reply:
                    break
                continue
            update = packet.get("params", {}).get("update", {})
            if update.get("_meta", {}).get("jetlinksRuntimeEvent", {}).get("type") != "acp.prompt.error":
                continue
            visible_error = update.get("content", {}).get("text") or ""
            if final_reply:
                break

    assert "请求处理失败：模型连接/调用失败：模型服务返回 500 Internal Server Error" in visible_error
    assert "http://model.example/v1/chat/completions" in visible_error
    assert '{"error":{"message":"","type":"InternalServerError","param":null,"code":500}}' in visible_error
    assert "请求处理失败：模型连接/调用失败：模型服务返回 500 Internal Server Error" in final_reply
    assert "http://model.example/v1/chat/completions" in final_reply


def test_acp_websocket_new_session_defaults_session_id_to_thread_id() -> None:
    client = TestClient(create_app())
    thread_id = f"acp-same-id-{uuid.uuid4().hex}"

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "new_session",
                "params": _acp_params(thread_id=thread_id, cwd="/tmp"),
            }
        )
        created = websocket.receive_json()["result"]

    assert created["threadId"] == thread_id
    assert created["sessionId"] == thread_id


def test_acp_websocket_prompt_returns_content_resource_links(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ArtifactAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(thread_id="acp-artifact-thread"),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "生成报告"}],
                },
            }
        )
        updates: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(10):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet["params"]["update"])
                continue
            if packet.get("id") == 3:
                final = packet
                break

    assert final is not None
    content = final["result"]["content"]
    assert content == final["result"]["result"]["content"]
    assert content[0] == {"type": "text", "text": "生成完成。"}
    assert content[1]["type"] == "resource_link"
    assert content[1]["uri"] == "outputs/reports/result.md"
    assert content[1]["path"] == "outputs/reports/result.md"
    assert content[1]["name"] == "result.md"
    assert content[1]["mimeType"] == "text/markdown"
    resource_update = next(
        update
        for update in updates
        if update.get("sessionUpdate") == "agent_message_chunk"
        and update.get("content", {}).get("type") == "resource_link"
    )
    assert resource_update["content"] == content[1]


def test_acp_websocket_skips_empty_agent_message_updates() -> None:
    assert _event_to_update(ChatEvent(type="agent.message", data={"text": ""})) is None
    assert _event_to_update(ChatEvent(type="agent.message.delta", data={"text": ""})) is None
    completed = _event_to_update(ChatEvent(type="run.completed", data={"result": {"status": "completed"}}))
    assert completed is not None
    assert completed["sessionUpdate"] == "agent_thought_chunk"


def test_acp_websocket_prompt_returns_input_required_from_result_metadata(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = InputRequiredAcpRuntime(
        ArtifactStore(root_dir=tmp_path / "threads"),
        reply="请上传模型配置后继续。",
        metadata={
            "requires_input": True,
            "required_inputs": [
                {
                    "type": "model_config",
                    "reason": "model configuration is required",
                }
            ],
        },
    )
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(thread_id="acp-explicit-input-required"),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "继续训练"}],
                },
            }
        )

        final = _receive_final_packet(websocket, 3)

    PromptResponse.model_validate(final["result"])
    assert final["result"]["stopReason"] == "end_turn"
    assert final["result"]["_meta"]["jetlinks"]["stopReason"] == "input_required"
    result = final["result"]["result"]
    assert result["status"] == "completed"
    assert result["metadata"]["requires_input"] is True
    assert result["metadata"]["required_inputs"] == [
        {
            "type": "model_config",
            "reason": "model configuration is required",
            "accept": ".json,.yaml,.yml,.toml",
            "required": True,
        }
    ]


def test_acp_websocket_prompt_infers_input_required_image(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = InputRequiredAcpRuntime(
        ArtifactStore(root_dir=tmp_path / "threads"),
        reply="当前没有检测到这次上传图片的可用附件路径，请重新上传图片。",
    )
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(
                    thread_id="acp-inferred-input-required",
                    runtime_options={"selectedSkills": ["data-auto-annotation"]},
                ),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "请把我上传的图片自动标注成 COCO JSON"}],
                },
            }
        )

        final = _receive_final_packet(websocket, 3)

    PromptResponse.model_validate(final["result"])
    assert final["result"]["stopReason"] == "end_turn"
    assert final["result"]["_meta"]["jetlinks"]["stopReason"] == "input_required"
    required_inputs = final["result"]["result"]["metadata"]["required_inputs"]
    assert required_inputs == [
        {
            "type": "image",
            "accept": "image/*",
            "required": True,
            "reason": "The agent requires an uploaded image.",
        }
    ]


def test_acp_websocket_parking_review_without_image_returns_final_json(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        raise AssertionError("LLM should not be called when parking review has no image attachment.")

    def fail_sync_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del self, system_prompt, messages
        raise AssertionError("LLM should not be called when parking review has no image attachment.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fail_if_called)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fail_sync_if_called)
    monkeypatch.setattr(acp_api, "runtime", AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads")))
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(
                    thread_id="acp-parking-review-no-image",
                    app_template_name="ParkingAbnormalEventMonitoring",
                ),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "复判这张图里是否有杂物堆积"}],
                },
            }
        )

        final = _receive_final_packet(websocket, 3)

    PromptResponse.model_validate(final["result"])
    assert final["result"]["stopReason"] == "end_turn"
    result = final["result"]["result"]
    parsed = json.loads(result["reply"])
    assert parsed[0]["hit"] == 0
    assert "未提供可访问的图片" in parsed[0]["result"]
    assert result["metadata"]["workflow"] == "parking_abnormal_review"


def test_acp_websocket_parking_review_strips_fenced_json_reply(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FencedJsonAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(
                    thread_id="acp-parking-json-cleanup",
                    app_template_name="ParkingAbnormalEventMonitoring",
                ),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "复判"}],
                },
            }
        )

        final = _receive_final_packet(websocket, 3)

    PromptResponse.model_validate(final["result"])
    result = final["result"]["result"]
    assert result["reply"].startswith("[")
    assert "```" not in result["reply"]
    assert json.loads(result["reply"]) == [
        {
            "reviewSourceId": "source-1",
            "reviewEventId": "event-1",
            "hit": 0,
            "result": "未发现杂物堆积。",
        }
    ]
    assert final["result"]["content"][0]["text"] == result["reply"]
    assert result["content"][0]["text"] == result["reply"]


def test_acp_websocket_parking_review_update_contains_final_json_only(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = StreamingFencedJsonAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    thread_id="acp-parking-final-update",
                    app_template_name="ParkingAbnormalEventMonitoring",
                ),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "复判"}],
                },
            }
        )

        updates: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(20):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet["params"]["update"])
                continue
            if packet.get("id") == 2:
                final = packet
                break

    assert final is not None
    result = final["result"]["result"]
    chunks = [
        update["content"]["text"]
        for update in updates
        if update.get("sessionUpdate") == "agent_message_chunk"
        and update.get("content", {}).get("type") == "text"
    ]
    assert chunks == [result["reply"]]
    assert "```" not in chunks[0]
    assert json.loads(chunks[0]) == [
        {
            "reviewSourceId": "source-1",
            "reviewEventId": "event-final",
            "hit": 0,
            "result": "最终复判结果。",
        }
    ]
    assert updates[-1]["_meta"]["jetlinksRuntimeEvent"]["data"]["final"] is True


def test_acp_websocket_response_format_json_strips_fenced_json_reply(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FencedJsonAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    thread_id="acp-response-format-json-cleanup",
                    runtime_options={"responseFormat": "json"},
                ),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "返回 JSON"}],
                },
            }
        )

        final = _receive_final_packet(websocket, 2)

    result = final["result"]["result"]
    assert result["reply"].startswith("[")
    assert "```" not in result["reply"]
    assert final["result"]["content"][0]["text"] == result["reply"]


def test_acp_websocket_default_agent_streams_delta_before_prompt_result(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = "我可以处理 JetLinks 对话、文件生成和证据优先的行为识别。"

    monkeypatch.setattr(OpenAICompatibleClient, "configured", property(lambda self: True))

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        return reply

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        assert tools
        return LlmChatResponse(content=reply)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(thread_id="acp-ws-default-stream", cwd="/tmp"),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                "params": _acp_params(
                    sessionId=session_id,
                    thread_id="acp-ws-default-stream",
                    prompt=[{"type": "text", "text": "你能干啥呢"}],
                ),
            }
        )

        updates: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(30):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet)
                continue
            if packet.get("id") == 3:
                final = packet
                break

        assert final is not None
        assert final["result"]["stopReason"] == "end_turn"
        assert final["result"]["result"]["reply"] == reply

        event_types = [
            update["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"]
            for update in updates
            if "_meta" in update["params"]["update"]
        ]
        chunks = [
            update["params"]["update"]["content"]["text"]
            for update in updates
            if update["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        ]

        _assert_valid_acp_updates(updates)
        assert "agent.message.delta" in event_types
        assert "run.completed" in event_types
        assert chunks == [reply]


def test_acp_websocket_maps_tool_events_to_acp_tool_call_updates(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ToolEventAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-tool-events", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "use tools"}]},
            }
        )

        updates: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(10):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet["params"]["update"])
                continue
            if packet.get("id") == 3:
                final = packet
                break

    assert final is not None
    _assert_valid_acp_updates([{"params": {"sessionId": session_id, "update": update}} for update in updates])
    started = next(update for update in updates if update["_meta"]["jetlinksRuntimeEvent"]["type"] == "tool.started")
    completed = next(update for update in updates if update["_meta"]["jetlinksRuntimeEvent"]["type"] == "tool.completed")
    failed = next(update for update in updates if update["_meta"]["jetlinksRuntimeEvent"]["type"] == "tool.failed")

    assert started["sessionUpdate"] == "tool_call"
    assert started["toolCallId"] == "call_write"
    assert started["title"] == "artifact_write"
    assert started["status"] == "in_progress"

    assert completed["sessionUpdate"] == "tool_call_update"
    assert completed["toolCallId"] == "call_write"
    assert completed["status"] == "completed"
    assert completed["rawOutput"] == {"path": "outputs/demo.md"}

    assert failed["sessionUpdate"] == "tool_call_update"
    assert failed["toolCallId"] == "call_read"
    assert failed["status"] == "failed"
    assert failed["rawOutput"] == {"error": "not found"}


def test_acp_websocket_routes_explicit_workflow_and_selected_skills() -> None:
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": _acp_params(thread_id="acp-direct-drawio", cwd="/tmp"),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                "params": _acp_params(
                    sessionId=session_id,
                    thread_id="acp-direct-drawio",
                    prompt=[{"type": "text", "text": "画一个工作台原型图"}],
                    runtime_options={
                        "workflow": "artifact_workflow",
                        "selectedSkills": ["drawio-generation"],
                    },
                ),
            }
        )

        final: dict[str, Any] | None = None
        for _ in range(30):
            packet = websocket.receive_json()
            if packet.get("id") == 3:
                final = packet
                break

        assert final is not None
        result = final["result"]["result"]
        assert result["metadata"]["workflow"] == "artifact_workflow"
        assert result["metadata"]["skill_name"] == "drawio-generation"
        assert result["verification"]["passed"] is True
        assert result["spec"]["skill_name"] == "drawio-generation"
        names = {artifact["name"] for artifact in result["artifacts"]}
        assert any(name.startswith("prototype") and name.endswith(".drawio") for name in names)
        assert any(name.startswith("prototype") and name.endswith(".png") for name in names)


def test_acp_websocket_routes_multimodal_prompt_and_session_model(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        initialized = websocket.receive_json()
        capabilities = initialized["result"]["agentCapabilities"]
        assert capabilities["promptCapabilities"]["image"] is True
        assert capabilities["promptCapabilities"]["audio"] is True
        assert capabilities["promptCapabilities"]["embeddedContext"] is True
        assert capabilities["sessionCapabilities"]["list"] is True
        assert capabilities["sessionCapabilities"]["close"] is True
        assert capabilities["sessionCapabilities"]["cancel"] is True

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-ws-multimodal", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/set_model",
                "params": {"sessionId": session_id, "modelId": "test-model"},
            }
        )
        assert websocket.receive_json()["result"]["modelName"] == "test-model"

        websocket.send_json({"jsonrpc": "2.0", "id": 4, "method": "session/list", "params": {}})
        listed = websocket.receive_json()
        assert any(item["sessionId"] == session_id for item in listed["result"]["sessions"])

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [
                        {"type": "text", "text": "分析图片和上下文"},
                        {
                            "type": "image",
                            "data": "aW1hZ2U=",
                            "mimeType": "image/png",
                            "uri": "file:///camera.png",
                        },
                        {
                            "type": "resource_link",
                            "name": "log.txt",
                            "uri": "/mnt/user-data/uploads/log.txt",
                            "mimeType": "text/plain",
                        },
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///ctx.txt",
                                "mimeType": "text/plain",
                                "text": "zone=restricted",
                            },
                        },
                    ],
                },
            }
        )

        final: dict[str, Any] | None = None
        for _ in range(10):
            packet = websocket.receive_json()
            if packet.get("id") == 5:
                final = packet
                break

        assert final is not None
        assert final["result"]["stopReason"] == "end_turn"

    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.runtime_options.model_name == "test-model"
    assert "分析图片和上下文" in request.messages[0].content
    assert "Embedded resource (file:///ctx.txt):" in request.messages[0].content
    assert "zone=restricted" in request.messages[0].content
    attachment_types = {attachment.metadata["acp_type"] for attachment in request.attachments}
    assert attachment_types == {"image", "resource_link", "embedded_text_resource"}


def test_acp_websocket_session_new_applies_app_template_and_runtime_options(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    cwd=str(tmp_path),
                    app_template_name="iot-architecture-diagram",
                    thread_id="acp-ws-app-template",
                    runtime_options={
                        "modelName": "session-model",
                        "temperature": 0.2,
                        "topP": 0.9,
                        "maxTokens": 1234,
                        "requestTimeoutSeconds": 9,
                    },
                ),
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]

        assert created["agentName"] == "default"
        assert created["appTemplateName"] == "iot-architecture-diagram"
        assert "appTemplateName" not in created["runtimeOptions"]
        assert created["models"]["currentModelId"] == "session-model"
        assert "workflow" not in created["runtimeOptions"]
        assert created["runtimeOptions"]["selectedSkills"] == ["drawio-generation"]
        assert created["runtimeOptions"]["modelName"] == "session-model"
        assert created["runtimeOptions"]["temperature"] == 0.2
        assert created["runtimeOptions"]["maxTokens"] == 1234

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "画一个 IoT 架构图"}],
                },
            }
        )
        _receive_final_packet(websocket, 2)

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": _acp_params(
                    agent_name=None,
                    sessionId=session_id,
                    prompt=[{"type": "text", "text": "这轮改成 Markdown"}],
                    runtime_options={
                        "selectedSkills": ["markdown-rendering"],
                        "modelName": "prompt-model",
                        "temperature": 0.7,
                        "maxTokens": 2000,
                    },
                ),
            }
        )
        _receive_final_packet(websocket, 3)

    assert len(runtime.requests) == 2
    first = runtime.requests[0].runtime_options
    assert first.thread_id == "acp-ws-app-template"
    assert first.workflow is None
    assert first.selected_skills == ["drawio-generation"]
    assert first.model_name == "session-model"
    assert first.temperature == 0.2
    assert first.top_p == 0.9
    assert first.max_tokens == 1234
    assert first.request_timeout_seconds == 9

    second = runtime.requests[1].runtime_options
    assert second.thread_id == "acp-ws-app-template"
    assert second.workflow is None
    assert second.selected_skills == ["markdown-rendering"]
    assert second.model_name == "prompt-model"
    assert second.temperature == 0.7
    assert second.max_tokens == 2000


def test_acp_websocket_top_level_app_template_preserves_template_config_options(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "threadId": "acp-screen-template",
                    "appTemplateName": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                },
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]

        assert created["appTemplateName"] == "70aaee52-99c2-49f5-a9c7-fb746821d3df"
        assert created["runtimeOptions"]["selectedSkills"] == ["generate-screen-skill"]
        assert created["runtimeOptions"]["configOptions"]["auto_execute_primary_skill"] is True

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "生成一个可视化大屏"}],
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.thread_id == "acp-screen-template"
    assert options.selected_skills == ["generate-screen-skill"]
    assert options.config_options["auto_execute_primary_skill"] is True
    assert options.config_options["max_tool_rounds"] == 3


def test_acp_websocket_prompt_expands_platform_skill_alias_before_runtime(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-platform-skill-alias", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": _acp_params(
                    sessionId=session_id,
                    prompt=[{"type": "text", "text": "生成一个园区本月用电的echarts的柱状图组件"}],
                    runtime_options={
                        "selectedSkills": ["178054047680069qndudx"],
                        "configOptions": {"auto_execute_primary_skill": True},
                    },
                ),
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.selected_skills == ["jetlinks-ai-component"]
    assert options.config_options["auto_execute_primary_skill"] is True


def test_acp_websocket_expands_plugin_skill_alias_from_loader_root_dir(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "config" / "agents"
    plugin_root = tmp_path / "plugins" / "skills" / "workspace-platform-id"
    agents_dir.mkdir(parents=True)
    plugin_root.mkdir(parents=True)
    (agents_dir / "default.json").write_text(
        json.dumps({"name": "default", "display_name": "Default"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (plugin_root / "plugin.json").write_text(
        json.dumps(
            {
                "id": "workspace-platform-id",
                "name": "Workspace Platform Skill",
                "version": "1.0.0",
                "skills": ["manifest.json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (plugin_root / "manifest.json").write_text(
        json.dumps(
            {
                "name": "workspace-runtime-skill",
                "description": "Workspace runtime skill.",
                "output_kind": "markdown",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "loader", AgentConfigLoader(tmp_path))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", ModelManager())
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    thread_id="acp-root-dir-alias",
                    cwd=str(tmp_path),
                    runtime_options={"selectedSkills": ["workspace-platform-id"]},
                ),
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]
        assert created["runtimeOptions"]["selectedSkills"] == ["workspace-runtime-skill"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": _acp_params(
                    sessionId=session_id,
                    prompt=[{"type": "text", "text": "hello"}],
                ),
            }
        )
        _receive_final_packet(websocket, 2)

    assert runtime.requests[0].runtime_options.selected_skills == ["workspace-runtime-skill"]


def test_acp_websocket_accepts_platform_session_init_and_agent_command(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session.init",
                "params": {
                    "agentId": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                    "parameters": {
                        "appTemplateName": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                    },
                    "tools": [],
                    "expands": {},
                    "sessionName": "生成一个可视化大屏",
                },
            }
        )
        created = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "agent.command",
                "params": {
                    "command": "Chat",
                    "arguments": {
                        "type": "generation",
                        "content": "生成一个可视化大屏",
                    },
                },
            }
        )
        accepted = websocket.receive_json()
        assert accepted["id"] == 2
        assert accepted["result"]["accepted"] is True
        _receive_platform_end(websocket)

    assert created["appTemplateName"] == "70aaee52-99c2-49f5-a9c7-fb746821d3df"
    assert len(runtime.requests) == 1
    request = runtime.requests[0]
    assert request.messages[-1].content == "生成一个可视化大屏"
    assert request.runtime_options.app_template_name == "70aaee52-99c2-49f5-a9c7-fb746821d3df"
    assert request.runtime_options.selected_skills == ["generate-screen-skill"]
    assert request.runtime_options.config_options["auto_execute_primary_skill"] is True


def test_acp_websocket_platform_agent_command_emits_platform_stream_events(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session.init",
                "params": {
                    "agentId": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                    "parameters": {},
                    "tools": [],
                    "expands": {},
                    "sessionName": "生成一个可视化大屏",
                },
            }
        )
        websocket.receive_json()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "agent.command",
                "params": {
                    "command": "Chat",
                    "arguments": {
                        "type": "generation",
                        "content": "生成一个可视化大屏",
                    },
                },
            }
        )

        platform_types: list[str] = []
        platform_chunks: list[str] = []
        platform_end_params: dict[str, Any] | None = None
        agent_message_types: list[str] = []
        agent_message_response_ids: set[str] = set()
        accepted: dict[str, Any] | None = None
        session_event_ended = False
        agent_message_ended = False
        for _ in range(60):
            packet = websocket.receive_json()
            if packet.get("id") == 2:
                accepted = packet
                continue
            params = packet.get("params")
            if packet.get("method") == "session.event" and isinstance(params, dict):
                event_type = params.get("type")
                if isinstance(event_type, str):
                    platform_types.append(event_type)
                    if event_type == "session.response_end":
                        session_event_ended = True
                event_params = params.get("params")
                if isinstance(event_params, dict):
                    if event_type == "session.response_end":
                        platform_end_params = event_params
                    chunk = event_params.get("chunk")
                    if isinstance(chunk, dict) and isinstance(chunk.get("content"), str):
                        platform_chunks.append(chunk["content"])
            if packet.get("method") == "agent.message" and isinstance(params, dict):
                event_type = params.get("type")
                if isinstance(event_type, str):
                    agent_message_types.append(event_type)
                    if event_type == "session.response_end":
                        agent_message_ended = True
                headers = params.get("headers")
                if isinstance(headers, dict) and isinstance(headers.get("responseId"), str):
                    agent_message_response_ids.add(headers["responseId"])
            if session_event_ended and agent_message_ended:
                break

    assert accepted is not None
    assert accepted["result"]["accepted"] is True
    assert session_event_ended is True
    assert agent_message_ended is True
    assert "session.response_start" in platform_types
    assert "session.response_chunk" in platform_types
    assert platform_types[-1] == "session.response_end"
    assert "session.response_start" in agent_message_types
    assert "session.response_chunk" in agent_message_types
    assert agent_message_types[-1] == "session.response_end"
    assert len(agent_message_response_ids) == 1
    assert platform_chunks == ["ok"]
    assert platform_end_params is not None
    assert platform_end_params["stopReason"] == "end_turn"
    assert platform_end_params["status"] == "completed"
    assert platform_end_params["reply"] == "ok"
    assert platform_end_params["content"] == [{"type": "text", "text": "ok"}]


def test_acp_websocket_platform_agent_command_emits_error_chunk(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = FailingAcpRuntime(
        ArtifactStore(root_dir=tmp_path / "threads"),
        "Server error '500 Internal Server Error'; response body: {\"error\":{\"code\":500}}",
    )
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session.init",
                "params": {
                    "agentId": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                    "parameters": {},
                    "tools": [],
                    "expands": {},
                    "sessionName": "生成一个可视化大屏",
                },
            }
        )
        websocket.receive_json()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "agent.command",
                "params": {
                    "command": "Chat",
                    "arguments": {
                        "type": "generation",
                        "content": "生成一个可视化大屏",
                    },
                },
            }
        )

        chunks: list[str] = []
        end_params: dict[str, Any] | None = None
        accepted: dict[str, Any] | None = None
        for _ in range(80):
            packet = websocket.receive_json()
            if packet.get("id") == 2:
                accepted = packet
                continue
            if packet.get("method") != "session.event":
                continue
            params = packet.get("params")
            if not isinstance(params, dict):
                continue
            event_params = params.get("params")
            if not isinstance(event_params, dict):
                continue
            if params.get("type") == "session.response_chunk":
                chunk = event_params.get("chunk")
                if isinstance(chunk, dict) and isinstance(chunk.get("content"), str):
                    chunks.append(chunk["content"])
            if params.get("type") == "session.response_end":
                end_params = event_params
                break

    assert accepted is not None
    assert accepted["result"]["accepted"] is True
    assert chunks == [
        "请求处理失败：Server error '500 Internal Server Error'; response body: {\"error\":{\"code\":500}}"
    ]
    assert end_params is not None
    assert end_params["stopReason"] == "error"
    assert end_params["error"] == "Server error '500 Internal Server Error'; response body: {\"error\":{\"code\":500}}"


def test_acp_websocket_trace_logs_full_conversation_payloads(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_transport_ws, "ACP_WS_TRACE_PAYLOADS", True)
    monkeypatch.setattr(acp_transport_ws, "ACP_WS_TRACE_MAX_CHARS", 0)
    client = TestClient(create_app())

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "session.init",
                    "params": {
                        "agentId": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                        "runtimeOptions": {"api_key": "incoming-secret-key"},
                    },
                }
            )
            websocket.receive_json()

            websocket.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "agent.command",
                    "params": {
                        "command": "Chat",
                        "arguments": {
                            "type": "generation",
                            "content": "完整日志测试",
                        },
                    },
                }
            )
            _receive_platform_end(websocket)

    trace_logs = "\n".join(record.getMessage() for record in caplog.records if "acp ws trace" in record.getMessage())

    assert "direction=recv" in trace_logs
    assert "direction=send" in trace_logs
    assert '"method":"agent.command"' in trace_logs
    assert '"method":"agent.message"' in trace_logs
    assert "完整日志测试" in trace_logs
    assert '"api_key":"********"' in trace_logs
    assert "incoming-secret-key" not in trace_logs
    assert "abc@123" not in trace_logs


def test_acp_websocket_logs_connection_lifecycle(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with caplog.at_level("INFO", logger="uvicorn.error"):
        with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
            websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            websocket.receive_json()

    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "acp ws connected connection_id=acp-ws-" in logs
    assert "path=/api/acp/ws" in logs
    assert "requested_subprotocols=acp.v1" in logs
    assert "accepted_subprotocol=acp.v1" in logs
    assert "acp ws received connection_id=acp-ws-" in logs
    assert "method=initialize" in logs
    assert "acp ws cleanup complete connection_id=acp-ws-" in logs
    assert "duration_ms=" in logs


def test_acp_websocket_accepts_bridge_payload_containers_for_app_template(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "threadId": "acp-bridge-template",
                    "parameters": {
                        "appTemplateName": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                    },
                },
            }
        )
        created = websocket.receive_json()["result"]
        assert created["appTemplateName"] == "70aaee52-99c2-49f5-a9c7-fb746821d3df"
        assert created["runtimeOptions"]["selectedSkills"] == ["generate-screen-skill"]
        assert created["runtimeOptions"]["configOptions"]["auto_execute_primary_skill"] is True

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": created["sessionId"],
                    "prompt": [{"type": "text", "text": "生成一个智慧园区大屏"}],
                    "arguments": {
                        "type": "generation",
                        "content": "生成一个智慧园区大屏",
                    },
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.thread_id == "acp-bridge-template"
    assert options.selected_skills == ["generate-screen-skill"]
    assert options.config_options["auto_execute_primary_skill"] is True


def test_acp_websocket_prompt_bridge_container_can_switch_app_template(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "threadId": "acp-bridge-prompt-template",
                },
            }
        )
        created = websocket.receive_json()["result"]
        assert created["appTemplateName"] is None

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": created["sessionId"],
                    "prompt": [{"type": "text", "text": "生成一个智慧园区大屏"}],
                    "arguments": {
                        "type": "generation",
                        "content": "生成一个智慧园区大屏",
                        "appTemplateName": "70aaee52-99c2-49f5-a9c7-fb746821d3df",
                    },
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.thread_id == "acp-bridge-prompt-template"
    assert options.app_template_name == "70aaee52-99c2-49f5-a9c7-fb746821d3df"
    assert options.selected_skills == ["generate-screen-skill"]
    assert options.config_options["auto_execute_primary_skill"] is True


def test_acp_websocket_session_new_accepts_standard_meta_extensions(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "_meta": {
                        "appTemplateName": "iot-architecture-diagram",
                        "runtimeOptions": {
                            "threadId": "acp-meta-app-template",
                            "selectedSkills": ["markdown-rendering"],
                            "modelName": "meta-model",
                            "temperature": 0.31,
                        },
                    },
                },
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]

        assert created["threadId"] == "acp-meta-app-template"
        assert created["appTemplateName"] == "iot-architecture-diagram"
        assert "appTemplateName" not in created["runtimeOptions"]
        assert created["runtimeOptions"]["selectedSkills"] == ["markdown-rendering"]
        assert created["runtimeOptions"]["modelName"] == "meta-model"
        assert created["runtimeOptions"]["temperature"] == 0.31

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "按 meta 创建模板会话"}],
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.thread_id == "acp-meta-app-template"
    assert options.selected_skills == ["markdown-rendering"]
    assert options.model_name == "meta-model"
    assert options.temperature == 0.31


def test_acp_websocket_session_new_accepts_standard_mcp_servers(
    tmp_path: Any,
) -> None:
    client = TestClient(create_app())
    mcp_servers = [
        {
            "name": "jetlinks-session",
            "url": "http://127.0.0.1:9100/api/ai/agent/mcp/session-token",
            "headers": [
                {"name": "Authorization", "value": "Bearer session-token"},
                {"name": "X-Trace-Id", "value": "trace-1"},
            ],
            "type": "http",
        }
    ]

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {
                    "cwd": str(tmp_path),
                    "mcpServers": mcp_servers,
                    "_meta": {"appTemplateName": "general-jetlinks-assistant"},
                },
            }
        )
        created = websocket.receive_json()["result"]

        assert created["appTemplateName"] == "general-jetlinks-assistant"
        assert created["mcpServers"][0]["name"] == "jetlinks-session"
        assert created["mcpServers"][0]["type"] == "http"
        assert created["mcpServers"][0]["url"] == "http://127.0.0.1:9100/api/ai/agent/mcp/session-token"
        assert created["mcpServers"][0]["headers"] == [
            {"name": "Authorization", "value": "********"},
            {"name": "X-Trace-Id", "value": "trace-1"},
        ]
        assert "mcpServers" not in created["runtimeOptions"]

        websocket.send_json({"jsonrpc": "2.0", "id": 2, "method": "session/list", "params": {}})
        listed = websocket.receive_json()["result"]["sessions"]

    assert any(session["mcpServers"] == created["mcpServers"] for session in listed)


def test_acp_websocket_prompt_deduplicates_current_prompt_from_messages(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-dedup-current-prompt", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "messages": [
                        {"role": "user", "content": "你好呀"},
                        {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
                        {"role": "user", "content": "你有哪些工具呢"},
                    ],
                    "prompt": [{"type": "text", "text": "你有哪些工具呢"}],
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    assert [(message.role, message.content) for message in runtime.requests[0].messages] == [
        ("user", "你好呀"),
        ("assistant", "你好！有什么我可以帮你的吗？"),
        ("user", "你有哪些工具呢"),
    ]


def test_acp_websocket_prompt_appends_distinct_prompt_after_history(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-append-current-prompt", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "messages": [
                        {"role": "user", "content": "你好呀"},
                        {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
                    ],
                    "prompt": [{"type": "text", "text": "你有哪些工具呢"}],
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    assert [(message.role, message.content) for message in runtime.requests[0].messages] == [
        ("user", "你好呀"),
        ("assistant", "你好！有什么我可以帮你的吗？"),
        ("user", "你有哪些工具呢"),
    ]


def test_acp_websocket_session_new_uses_app_model_from_config(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "config" / "agents"
    apps_dir = tmp_path / "config" / "apps"
    agents_dir.mkdir(parents=True)
    apps_dir.mkdir(parents=True)
    (agents_dir / "default.json").write_text(
        json.dumps({"name": "default", "display_name": "Default"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (apps_dir / "demo-gpt.json").write_text(
        json.dumps(
            {
                "name": "demo-gpt",
                "title": "Demo GPT",
                "agent_name": "default",
                "models": [
                    {
                        "name": "gpt-5.5",
                        "model": "gpt-5.5",
                        "default_model": "gpt-5.5",
                        "base_url": "http://model.local/v1",
                        "api_key": "app-key",
                        "temperature": 0.4,
                        "max_tokens": 128,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "loader", AgentConfigLoader(tmp_path))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", ModelManager())
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    app_template_name="demo-gpt",
                    thread_id="acp-app-model",
                    cwd=str(tmp_path),
                ),
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]

        assert created["models"]["currentModelId"] == "gpt-5.5"
        assert created["models"]["currentModelName"] == "gpt-5.5"
        assert created["runtimeOptions"]["modelName"] == "gpt-5.5"
        assert created["runtimeOptions"]["baseUrl"] == "http://model.local/v1"
        assert created["runtimeOptions"]["temperature"] == 0.4
        assert created["runtimeOptions"]["maxTokens"] == 128
        assert "apiKey" not in created["runtimeOptions"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "hello"}],
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.model_name == "gpt-5.5"
    assert options.base_url == "http://model.local/v1"
    assert options.api_key == "app-key"
    assert options.temperature == 0.4
    assert options.max_tokens == 128


def test_acp_websocket_prompt_infers_upload_app_model_from_review_source_id(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "config" / "agents"
    apps_dir = tmp_path / "config" / "upload" / "apps"
    agents_dir.mkdir(parents=True)
    apps_dir.mkdir(parents=True)
    (agents_dir / "default.json").write_text(
        json.dumps(
            {
                "name": "default",
                "display_name": "Default",
                "model": {
                    "model": "agent-model",
                    "base_url": "http://agent.local/v1",
                    "api_key": "agent-key",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (apps_dir / "uploaded-review-app.json").write_text(
        json.dumps(
            {
                "name": "uploaded-review-app",
                "title": "Uploaded Review App",
                "agent_name": "default",
                "runtime_options": {"config_options": {"force_model_config": True}},
                "models": [
                    {
                        "name": "uploaded-model",
                        "model": "uploaded-model",
                        "default_model": "uploaded-model",
                        "base_url": "http://uploaded.local/v1",
                        "api_key": "uploaded-key",
                        "temperature": 0.2,
                        "max_tokens": 321,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "loader", AgentConfigLoader(tmp_path))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", ModelManager())
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-upload-app-inferred", cwd=str(tmp_path)),
            }
        )
        created = websocket.receive_json()["result"]
        assert created["appTemplateName"] is None

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": created["sessionId"],
                    "prompt": [
                        {
                            "type": "text",
                            "text": (
                                "当前复判事件来源reviewSourceId为"
                                "[uploaded-review-app_source-123]。请执行复判。"
                            ),
                        }
                    ],
                },
            }
        )
        _receive_final_packet(websocket, 2)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.app_template_name == "uploaded-review-app"
    assert options.model_name == "uploaded-model"
    assert options.base_url == "http://uploaded.local/v1"
    assert options.api_key == "uploaded-key"
    assert options.temperature == 0.2
    assert options.max_tokens == 321
    assert options.config_options["force_model_config"] is True


def test_acp_websocket_session_update_applies_app_model_from_config(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "config" / "agents"
    apps_dir = tmp_path / "config" / "apps"
    agents_dir.mkdir(parents=True)
    apps_dir.mkdir(parents=True)
    (agents_dir / "default.json").write_text(
        json.dumps({"name": "default", "display_name": "Default"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (apps_dir / "demo-gpt.json").write_text(
        json.dumps(
            {
                "name": "demo-gpt",
                "title": "Demo GPT",
                "agent_name": "default",
                "selected_skills": ["algorithm-engineer"],
                "models": [
                    {
                        "name": "gpt-5.5",
                        "model": "gpt-5.5",
                        "default_model": "gpt-5.5",
                        "base_url": "http://model.local/v1",
                        "api_key": "app-key",
                        "temperature": 0.4,
                        "max_tokens": 128,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "loader", AgentConfigLoader(tmp_path))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", ModelManager())
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-update-app-model", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/update",
                "params": {
                    "sessionId": session_id,
                    "_meta": {
                        "appTemplateName": "demo-gpt",
                        "runtimeOptions": {
                            "selectedSkills": [],
                            "maxTokens": 256,
                            "requestTimeoutSeconds": 11,
                        },
                    },
                },
            }
        )
        updated = websocket.receive_json()["result"]
        assert updated["appTemplateName"] == "demo-gpt"
        assert "appTemplateName" not in updated["runtimeOptions"]
        assert updated["models"]["currentModelId"] == "gpt-5.5"
        assert updated["runtimeOptions"]["modelName"] == "gpt-5.5"
        assert updated["runtimeOptions"]["baseUrl"] == "http://model.local/v1"
        assert updated["runtimeOptions"]["selectedSkills"] == []
        assert updated["runtimeOptions"]["maxTokens"] == 256
        assert updated["runtimeOptions"]["requestTimeoutSeconds"] == 11.0
        assert "apiKey" not in updated["runtimeOptions"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]},
            }
        )
        _receive_final_packet(websocket, 3)

    assert len(runtime.requests) == 1
    options = runtime.requests[0].runtime_options
    assert options.model_name == "gpt-5.5"
    assert options.base_url == "http://model.local/v1"
    assert options.api_key == "app-key"
    assert options.selected_skills == []
    assert options.max_tokens == 256
    assert options.request_timeout_seconds == 11


def test_acp_websocket_supports_session_scoped_fs_methods(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    host_file = cwd / "notes.txt"
    host_file.write_text("one\ntwo\nthree\n", encoding="utf-8")

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-fs", cwd=str(cwd)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "fs/read_text_file",
                "params": {"sessionId": session_id, "path": str(host_file), "line": 2, "limit": 1},
            }
        )
        read_host = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "fs/write_text_file",
                "params": {
                    "sessionId": session_id,
                    "path": "/mnt/user-data/workspace/generated.txt",
                    "content": "generated",
                },
            }
        )
        websocket.receive_json()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "fs/read_text_file",
                "params": {"sessionId": session_id, "path": "/mnt/user-data/workspace/generated.txt"},
            }
        )
        read_workspace = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "fs/read_text_file",
                "params": {"sessionId": session_id, "path": str(tmp_path / "outside.txt")},
            }
        )
        outside = websocket.receive_json()

    assert read_host["content"] == "two"
    assert read_workspace["content"] == "generated"
    assert outside["error"]["code"] == -32602
    assert "must stay within session cwd" in outside["error"]["message"]


def test_acp_websocket_supports_terminal_methods(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-terminal", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "terminal/create",
                "params": {
                    "sessionId": session_id,
                    "command": "/bin/sh",
                    "args": ["-c", "printf terminal-ok"],
                },
            }
        )
        created = websocket.receive_json()["result"]
        terminal_id = created["terminalId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "terminal/wait_for_exit",
                "params": {"sessionId": session_id, "terminalId": terminal_id},
            }
        )
        exited = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "terminal/output",
                "params": {"sessionId": session_id, "terminalId": terminal_id},
            }
        )
        output = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "terminal/release",
                "params": {"sessionId": session_id, "terminalId": terminal_id},
            }
        )
        released = websocket.receive_json()["result"]

    assert exited["exitCode"] == 0
    assert output["output"] == "terminal-ok"
    assert output["exitStatus"]["exitCode"] == 0
    assert released == {}


def test_acp_websocket_session_mode_and_config_options_affect_runtime_options(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-mode-config", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/set_mode",
                "params": {"sessionId": session_id, "modeId": "safe"},
            }
        )
        mode_result = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/set_config_option",
                "params": {"sessionId": session_id, "configId": "temperature", "value": 0.1},
            }
        )
        config_result = websocket.receive_json()["result"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "hello"}]},
            }
        )
        _receive_final_packet(websocket, 4)

    assert mode_result["modeId"] == "safe"
    assert config_result["values"]["temperature"] == 0.1
    assert runtime.requests[0].runtime_options.mode == "safe"
    assert runtime.requests[0].runtime_options.temperature == 0.1


def test_acp_websocket_permission_request_emits_update_then_returns_cancelled(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-permission", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/request_permission",
                "params": {
                    "sessionId": session_id,
                    "title": "Run command",
                    "description": "Allow shell command",
                    "options": [{"id": "allow"}],
                },
            }
        )
        update = websocket.receive_json()
        final = websocket.receive_json()

    assert update["method"] == "session/update"
    assert update["params"]["update"]["sessionUpdate"] == "permission_request"
    assert update["params"]["update"]["permissionRequest"]["title"] == "Run command"
    assert final["result"]["outcome"]["outcome"] == "cancelled"


def test_acp_websocket_yolo_mode_auto_approves_permission_request(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    thread_id="acp-yolo-permission",
                    cwd=str(tmp_path),
                    runtime_options={"mode": "yolo"},
                ),
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/request_permission",
                "params": {
                    "sessionId": session_id,
                    "title": "Run command",
                    "description": "Allow shell command",
                    "options": [{"id": "allow", "kind": "allow_once", "name": "Allow once"}],
                },
            }
        )
        final = websocket.receive_json()

    assert created["modeId"] == "yolo"
    assert final["result"]["outcome"] == {"outcome": "selected", "optionId": "allow"}


def test_acp_websocket_updates_include_plan_and_diff_metadata(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = PlanDiffEventAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-plan-diff", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "write"}]},
            }
        )
        updates: list[dict[str, Any]] = []
        for _ in range(10):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet)
            if packet.get("id") == 2:
                break

    metas = [packet["params"]["update"].get("_meta", {}) for packet in updates]
    assert any(meta.get("jetlinksPlan", {}).get("type") == "tool" for meta in metas)
    assert any((meta.get("jetlinksDiff") or {}).get("path") == "/mnt/user-data/workspace/demo.txt" for meta in metas)


def test_acp_websocket_resolves_server_managed_model_id(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    model_manager = ModelManager(
        [
            {
                "id": "managed-qwen",
                "display_name": "Managed Qwen",
                "config": {
                    "model": "qwen-runtime-model",
                    "base_url": "http://model-manager.local/v1",
                    "api_key": "managed-key",
                    "temperature": 0.25,
                    "max_tokens": 321,
                },
                "capabilities": ["tool_calling"],
            }
        ],
        default_model_id="managed-qwen",
    )
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", model_manager)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        initialized = websocket.receive_json()["result"]
        assert initialized["_meta"]["jetlinks"]["modelManagement"] == "server"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-managed-model", cwd=str(tmp_path)),
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]
        assert created["models"]["currentModelId"] == "managed-qwen"
        assert created["models"]["availableModels"][0]["id"] == "managed-qwen"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/set_model",
                "params": {"sessionId": session_id, "modelId": "managed-qwen"},
            }
        )
        selected = websocket.receive_json()["result"]
        assert selected["modelId"] == "managed-qwen"
        assert selected["modelName"] == "qwen-runtime-model"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "你好"}],
                },
            }
        )
        _receive_final_packet(websocket, 4)

    request_options = runtime.requests[0].runtime_options
    assert request_options.model_name == "qwen-runtime-model"
    assert request_options.base_url == "http://model-manager.local/v1"
    assert request_options.api_key == "managed-key"
    assert request_options.temperature == 0.25
    assert request_options.max_tokens == 321


def test_acp_websocket_force_app_model_config_over_default_agent_model(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apps_dir = tmp_path / "config" / "apps"
    agents_dir = tmp_path / "config" / "agents"
    apps_dir.mkdir(parents=True)
    agents_dir.mkdir(parents=True)
    (agents_dir / "default.json").write_text(
        json.dumps(
            {
                "name": "default",
                "display_name": "Default",
                "model": {
                    "model": "old-model",
                    "default_model": "old-model",
                    "base_url": "http://218.67.242.10:59202/v1",
                    "api_key": "old-key",
                    "temperature": 0.9,
                    "max_tokens": 64,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (apps_dir / "parking-review.json").write_text(
        json.dumps(
            {
                "name": "parking-review",
                "title": "Parking Review",
                "agent_name": "default",
                "runtime_options": {"config_options": {"force_model_config": True}},
                "models": [
                    {
                        "name": "Qwen3.6-35B-A3B",
                        "model": "Qwen3.6-35B-A3B",
                        "default_model": "Qwen3.6-35B-A3B",
                        "base_url": "http://192.168.35.140:9100/api/llm/openai/v1/providers/builtin-openai-compatible/",
                        "api_key": "new-key",
                        "temperature": 0.4,
                        "max_tokens": 2048,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loader = AgentConfigLoader(tmp_path)
    agent_default = loader.load("default")
    model_manager = ModelManager()
    model_manager.configure_from_agent_default(agent_default)
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "loader", loader)
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", model_manager)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(
                    app_template_name="parking-review",
                    thread_id="acp-force-template-model",
                    cwd=str(tmp_path),
                    runtime_options={
                        "modelName": "old-model",
                        "baseUrl": "http://218.67.242.10:59202/v1",
                        "apiKey": "old-key",
                        "temperature": 0.9,
                        "maxTokens": 64,
                        "configOptions": {"mcpServers": []},
                    },
                ),
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]
        assert created["models"]["currentModelId"] == "Qwen3.6-35B-A3B"
        assert created["runtimeOptions"]["modelName"] == "Qwen3.6-35B-A3B"
        assert created["runtimeOptions"]["baseUrl"] == (
            "http://192.168.35.140:9100/api/llm/openai/v1/providers/builtin-openai-compatible/"
        )
        assert created["runtimeOptions"]["temperature"] == 0.4
        assert created["runtimeOptions"]["maxTokens"] == 2048

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "复判"}],
                    "runtimeOptions": {
                        "modelName": "old-model",
                        "baseUrl": "http://218.67.242.10:59202/v1",
                        "apiKey": "old-key",
                    },
                },
            }
        )
        _receive_final_packet(websocket, 2)

    request_options = runtime.requests[0].runtime_options
    assert request_options.model_name == "Qwen3.6-35B-A3B"
    assert request_options.base_url == "http://192.168.35.140:9100/api/llm/openai/v1/providers/builtin-openai-compatible/"
    assert request_options.api_key == "new-key"
    assert request_options.temperature == 0.4
    assert request_options.max_tokens == 2048
    assert request_options.config_options["force_model_config"] is True
    assert request_options.config_options["session_cwd"] == str(tmp_path)


def test_acp_websocket_does_not_expose_default_agent_model_when_no_model_configured(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    model_manager = ModelManager()
    agent_default = acp_api.loader.load("default")
    model_manager.configure_from_agent_default(agent_default)
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", model_manager)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-default-model", cwd=str(tmp_path)),
            }
        )
        created = websocket.receive_json()["result"]

    assert agent_default.model.model == "Qwen3.6-35B-A3B"
    assert agent_default.model.default_model == "Qwen3.6-35B-A3B"
    assert created["models"]["currentModelId"] == "Qwen3.6-35B-A3B"
    assert created["models"]["currentModelName"] == "Qwen3.6-35B-A3B"
    assert created["runtimeOptions"]["modelName"] == "Qwen3.6-35B-A3B"
    assert created["runtimeOptions"]["baseUrl"] == "http://218.67.242.10:59202/v1"
    assert created["models"]["availableModels"]


def test_default_agent_model_registration_ignores_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "env-qwen")
    monkeypatch.setenv("LLM_BASE_URL", "http://env-ollama.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    model_manager = ModelManager()

    agent = acp_api.loader.load("default")
    registered = model_manager.configure_from_agent_default(agent)

    assert agent.model.model == "Qwen3.6-35B-A3B"
    assert agent.model.default_model == "Qwen3.6-35B-A3B"
    assert registered is not None
    assert registered.id == "Qwen3.6-35B-A3B"
    assert model_manager.list()


def test_acp_websocket_cancel_interrupts_active_prompt(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"), block=True)
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-ws-cancel", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "hold"}]},
            }
        )
        started = websocket.receive_json()
        assert started["method"] == "session/update"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/cancel",
                "params": {"sessionId": session_id},
            }
        )

        responses: dict[int, dict[str, Any]] = {}
        for _ in range(6):
            packet = websocket.receive_json()
            packet_id = packet.get("id")
            if packet_id in {2, 3}:
                responses[packet_id] = packet
            if 2 in responses and 3 in responses:
                break

        assert responses[2]["result"]["stopReason"] == "cancelled"
        assert responses[3]["result"]["stopReason"] == "cancelled"


def test_acp_websocket_delete_session_files_extension(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": _acp_params(thread_id="acp-ws-delete", cwd=str(tmp_path)),
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        paths = runtime.artifact_store.prepare_thread("acp-ws-delete")
        workspace_file = paths.workspace / "notes.txt"
        uploads_file = paths.uploads / "photo.png"
        outputs_file = paths.outputs / "nested" / "report.md"
        outputs_file.parent.mkdir(parents=True, exist_ok=True)
        workspace_file.write_text("workspace-data", encoding="utf-8")
        uploads_file.write_bytes(b"upload-data")
        outputs_file.write_text("outputs-data", encoding="utf-8")

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "jetlinks/session/delete_files",
                "params": {
                    "sessionId": session_id,
                    "scopes": ["workspace", "outputs"],
                    "dryRun": True,
                },
            }
        )
        dry_run = websocket.receive_json()["result"]
        assert dry_run["dryRun"] is True
        assert dry_run["deletedFileCount"] == 2
        assert workspace_file.exists()
        assert outputs_file.exists()
        assert uploads_file.exists()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "jetlinks/session/delete_files",
                "params": {
                    "sessionId": session_id,
                    "scopes": ["workspace", "outputs"],
                },
            }
        )
        deleted = websocket.receive_json()["result"]

    deleted_paths = {item["path"] for item in deleted["deleted"]}
    assert deleted["sessionId"] == session_id
    assert deleted["threadId"] == "acp-ws-delete"
    assert deleted["dryRun"] is False
    assert deleted["scopes"] == ["workspace", "outputs"]
    assert deleted["deletedFileCount"] == 2
    assert deleted["deletedBytes"] == len("workspace-data") + len("outputs-data")
    assert "/mnt/user-data/workspace/notes.txt" in deleted_paths
    assert "/mnt/user-data/outputs/nested/report.md" in deleted_paths
    assert not workspace_file.exists()
    assert not outputs_file.exists()
    assert paths.workspace.exists()
    assert paths.outputs.exists()
    assert uploads_file.exists()


def _receive_final_packet(websocket: Any, request_id: int) -> dict[str, Any]:
    for _ in range(10):
        packet = websocket.receive_json()
        if packet.get("id") == request_id:
            return packet
    raise AssertionError(f"ACP WebSocket response not received: {request_id}")


def _receive_platform_end(websocket: Any) -> dict[str, Any]:
    for _ in range(30):
        packet = websocket.receive_json()
        params = packet.get("params")
        if packet.get("method") == "session.event" and isinstance(params, dict):
            if params.get("type") == "session.response_end":
                return packet
    raise AssertionError("Platform response end event not received")


def _acp_params(
    *,
    cwd: str | None = None,
    agent_name: str | None = "default",
    thread_id: str | None = None,
    app_template_name: str | None = None,
    runtime_options: dict[str, Any] | None = None,
    **params: Any,
) -> dict[str, Any]:
    if cwd is not None:
        params["cwd"] = cwd
    meta: dict[str, Any] = {}
    if agent_name is not None:
        meta["agentName"] = agent_name
    if app_template_name is not None:
        meta["appTemplateName"] = app_template_name
    options = dict(runtime_options or {})
    if thread_id is not None:
        options["threadId"] = thread_id
    if options:
        meta["runtimeOptions"] = options
    if meta:
        params["_meta"] = meta
    return params


def _assert_valid_acp_updates(updates: list[dict[str, Any]]) -> None:
    for packet in updates:
        params = packet["params"]
        SessionNotification.model_validate(
            {
                "sessionId": params["sessionId"],
                "update": params["update"],
            }
        )
