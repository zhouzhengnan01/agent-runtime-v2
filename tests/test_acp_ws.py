from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.core.llm.openai_compatible import LlmChatResponse, OpenAICompatibleClient
from app.schemas import Message
from app.main import create_app


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
                    "agentName": "behavior-detector",
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
                    "agentName": "behavior-detector",
                    "threadId": "acp-ws-test",
                    "prompt": [{"type": "text", "text": "人员翻越围栏进入禁区"}],
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
        assert "run.started" in event_types
        assert "agent.message" in event_types


def test_acp_websocket_default_agent_streams_delta_before_prompt_result(monkeypatch) -> None:
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
            update["params"]["update"]["text"]
            for update in updates
            if update["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        ]

        assert "agent.message.delta" in event_types
        assert "run.completed" in event_types
        assert chunks == [reply]
