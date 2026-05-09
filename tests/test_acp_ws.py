from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from acp.schema import SessionNotification
from fastapi.testclient import TestClient

from app.api import acp as acp_api
from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.llm.openai_compatible import LlmChatResponse, OpenAICompatibleClient
from app.core.runtime import ModelManager
from app.schemas import AgentRunResult, ChatEvent, ChatRequest
from app.schemas import Message
from app.main import create_app


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


def test_acp_websocket_prompt_streams_runtime_events() -> None:
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        initialized = websocket.receive_json()
        assert initialized["result"]["agentInfo"]["name"] == "jetlinks-agent-runtime-v2"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": {
                    "agentName": "default",
                    "threadId": "acp-ws-test",
                    "cwd": "/tmp",
                },
            }
        )
        created = websocket.receive_json()
        session_id = created["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                "params": {
                    "sessionId": session_id,
                    "agentName": "default",
                    "threadId": "acp-ws-test",
                    "prompt": [{"type": "text", "text": "人员翻越围栏进入禁区"}],
                    "runtimeOptions": {"workflow": "evidence_first_detection"},
                },
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
        assert "待上传" in final["result"]["result"]["reply"]
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


def test_acp_websocket_default_agent_streams_delta_before_prompt_result(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = "我可以处理 JetLinks 对话、文件生成和证据优先的行为识别。"

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
                "params": {
                    "agentName": "default",
                    "threadId": "acp-ws-default-stream",
                    "cwd": "/tmp",
                },
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                "params": {
                    "sessionId": session_id,
                    "agentName": "default",
                    "threadId": "acp-ws-default-stream",
                    "prompt": [{"type": "text", "text": "你能干啥呢"}],
                },
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
                "params": {"agentName": "default", "threadId": "acp-tool-events", "cwd": str(tmp_path)},
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
                "params": {
                    "agentName": "default",
                    "threadId": "acp-direct-drawio",
                    "cwd": "/tmp",
                },
            }
        )
        session_id = websocket.receive_json()["result"]["sessionId"]

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                "params": {
                    "sessionId": session_id,
                    "agentName": "default",
                    "threadId": "acp-direct-drawio",
                    "prompt": [{"type": "text", "text": "画一个工作台原型图"}],
                    "runtimeOptions": {
                        "workflow": "artifact_workflow",
                        "selectedSkills": ["drawio-generation"],
                    },
                },
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
                "params": {
                    "agentName": "default",
                    "threadId": "acp-ws-multimodal",
                    "cwd": str(tmp_path),
                },
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
                "params": {
                    "appTemplateName": "iot-architecture-diagram",
                    "threadId": "acp-ws-app-template",
                    "cwd": str(tmp_path),
                    "runtimeOptions": {
                        "modelName": "session-model",
                        "temperature": 0.2,
                        "topP": 0.9,
                        "maxTokens": 1234,
                        "requestTimeoutSeconds": 9,
                    },
                },
            }
        )
        created = websocket.receive_json()["result"]
        session_id = created["sessionId"]

        assert created["agentName"] == "default"
        assert created["appTemplateName"] == "iot-architecture-diagram"
        assert created["models"]["currentModelId"] == "session-model"
        assert created["runtimeOptions"]["workflow"] == "artifact_workflow"
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
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "这轮改成 Markdown"}],
                    "runtimeOptions": {
                        "selectedSkills": ["markdown-rendering"],
                        "modelName": "prompt-model",
                        "temperature": 0.7,
                        "maxTokens": 2000,
                    },
                },
            }
        )
        _receive_final_packet(websocket, 3)

    assert len(runtime.requests) == 2
    first = runtime.requests[0].runtime_options
    assert first.thread_id == "acp-ws-app-template"
    assert first.workflow == "artifact_workflow"
    assert first.selected_skills == ["drawio-generation"]
    assert first.model_name == "session-model"
    assert first.temperature == 0.2
    assert first.top_p == 0.9
    assert first.max_tokens == 1234
    assert first.request_timeout_seconds == 9

    second = runtime.requests[1].runtime_options
    assert second.thread_id == "acp-ws-app-template"
    assert second.workflow == "artifact_workflow"
    assert second.selected_skills == ["markdown-rendering"]
    assert second.model_name == "prompt-model"
    assert second.temperature == 0.7
    assert second.max_tokens == 2000


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
                "params": {
                    "agentName": "default",
                    "threadId": "acp-managed-model",
                    "cwd": str(tmp_path),
                },
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


def test_acp_websocket_exposes_default_agent_model_when_model_manager_is_empty(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = CapturingAcpRuntime(ArtifactStore(root_dir=tmp_path / "threads"))
    model_manager = ModelManager()
    model_manager.configure_from_agent_default(acp_api.loader.load("default"))
    monkeypatch.setattr(acp_api, "runtime", runtime)
    monkeypatch.setattr(acp_api, "model_manager", model_manager)
    client = TestClient(create_app())

    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"agentName": "default", "threadId": "acp-default-model", "cwd": str(tmp_path)},
            }
        )
        created = websocket.receive_json()["result"]

    assert created["models"]["currentModelId"] == "qwen3.6-27b"
    assert created["models"]["currentModelName"] == "qwen3.6-27b"
    assert created["runtimeOptions"]["modelName"] == "qwen3.6-27b"
    assert created["runtimeOptions"]["baseUrl"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert created["models"]["availableModels"][0]["id"] == "qwen3.6-27b"
    assert created["models"]["availableModels"][0]["model"] == "qwen3.6-27b"


def test_default_agent_model_registration_uses_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "env-qwen")
    monkeypatch.setenv("LLM_BASE_URL", "http://env-ollama.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "")
    model_manager = ModelManager()

    model_manager.configure_from_agent_default(acp_api.loader.load("default"))
    options = model_manager.runtime_options_for("env-qwen")

    assert options is not None
    assert options.model_name == "env-qwen"
    assert options.base_url == "http://env-ollama.local/v1"
    assert options.api_key == ""


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
                "params": {"agentName": "default", "threadId": "acp-ws-cancel", "cwd": str(tmp_path)},
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
                "params": {"agentName": "default", "threadId": "acp-ws-delete", "cwd": str(tmp_path)},
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


def _assert_valid_acp_updates(updates: list[dict[str, Any]]) -> None:
    for packet in updates:
        params = packet["params"]
        SessionNotification.model_validate(
            {
                "sessionId": params["sessionId"],
                "update": params["update"],
            }
        )
