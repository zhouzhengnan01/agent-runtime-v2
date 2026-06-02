from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

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
    assert wait_message_updates[0]["sessionUpdate"] == "agent_message_chunk"
    assert wait_message_updates[0]["content"] == {"type": "text", "text": "已收到请求，正在处理，请等待..."}
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

    assert agent_default.model.model is None
    assert agent_default.model.default_model is None
    assert created["models"]["currentModelId"] is None
    assert created["models"]["currentModelName"] is None
    assert "modelName" not in created["runtimeOptions"]
    assert "baseUrl" not in created["runtimeOptions"]
    assert created["models"]["availableModels"] == []


def test_default_agent_model_registration_ignores_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "env-qwen")
    monkeypatch.setenv("LLM_BASE_URL", "http://env-ollama.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    model_manager = ModelManager()

    agent = acp_api.loader.load("default")
    registered = model_manager.configure_from_agent_default(agent)

    assert agent.model.model is None
    assert agent.model.default_model is None
    assert registered is None
    assert model_manager.list() == []


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
