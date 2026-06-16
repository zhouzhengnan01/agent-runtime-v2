from __future__ import annotations

import asyncio
import json

from app.api.acp_http_stream import AcpHttpSubscriptionRequest, _stream_acp_subscription
from app.protocols.acp.event_broker import acp_event_broker


async def _collect_subscription() -> list[str]:
    request = AcpHttpSubscriptionRequest(
        sessionId="postman-yolo-test-002",
        threadId="postman-yolo-test-002",
    )
    chunks: list[str] = []

    async def collect() -> None:
        async for chunk in _stream_acp_subscription(request):
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    await acp_event_broker.publish(
        {"postman-yolo-test-002"},
        "session/update",
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "postman-yolo-test-002",
                "update": {
                    "_meta": {"jetlinksRuntimeEvent": {"type": "tool.started", "data": {}}},
                    "sessionUpdate": "tool_call",
                },
            },
        },
    )
    await acp_event_broker.publish(
        {"postman-yolo-test-002"},
        "session/update",
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "postman-yolo-test-002",
                "update": {
                    "_meta": {
                        "jetlinksRuntimeEvent": {
                            "type": "artifact.created",
                            "data": {"artifact": {"name": "best.pt"}},
                        }
                    },
                    "sessionUpdate": "agent_thought_chunk",
                },
            },
        },
    )
    await acp_event_broker.publish(
        {"postman-yolo-test-002"},
        "result",
        {"jsonrpc": "2.0", "id": 7, "result": {"stopReason": "end_turn"}},
    )
    await task
    return chunks


def test_http_subscription_receives_websocket_published_events() -> None:
    chunks = asyncio.run(_collect_subscription())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]

    assert chunks[0] == ": connected\n\n"
    assert payloads[0]["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"] == "tool.started"
    assert payloads[1]["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"] == "artifact.created"
    assert payloads[2]["result"]["stopReason"] == "end_turn"


def test_http_subscription_requires_session_or_thread() -> None:
    async def collect() -> list[str]:
        return [chunk async for chunk in _stream_acp_subscription(AcpHttpSubscriptionRequest())]

    chunks = asyncio.run(collect())

    assert "sessionId or threadId is required" in chunks[0]
