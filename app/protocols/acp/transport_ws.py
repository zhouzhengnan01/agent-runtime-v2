from __future__ import annotations

from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.protocols.acp.dispatcher import AcpDispatcher
from app.protocols.acp.schemas import AcpWebSocketSession, JsonRpcId


async def handle_acp_websocket(websocket: WebSocket, dispatcher: AcpDispatcher | None = None) -> None:
    """Serve ACP-shaped JSON-RPC over WebSocket.

    ACP's common editor transport is stdio. This transport keeps the ACP
    lifecycle and JSON-RPC envelope, then uses WebSocket for browser clients.
    """

    await websocket.accept(subprotocol="acp.v1")
    sessions: dict[str, AcpWebSocketSession] = {}
    active_dispatcher = dispatcher or AcpDispatcher()

    async def send_update(session_id: str, update: dict[str, Any]) -> None:
        await _send_session_update(websocket, session_id, update)

    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await _send_error(websocket, None, -32600, "JSON-RPC message must be an object")
                continue

            request_id = _request_id(message)
            method = message.get("method")
            params = _params(message.get("params"))

            if not isinstance(method, str) or not method:
                await _send_error(websocket, request_id, -32600, "JSON-RPC method is required")
                continue

            try:
                result = await active_dispatcher.dispatch(sessions, method, params, send_update)
            except FileNotFoundError as exc:
                await _send_error(websocket, request_id, -32004, str(exc))
                continue
            except ValidationError as exc:
                await _send_error(websocket, request_id, -32602, exc.errors()[0]["msg"])
                continue
            except ValueError as exc:
                await _send_error(websocket, request_id, -32602, str(exc))
                continue
            except Exception as exc:
                await _send_error(websocket, request_id, -32000, str(exc))
                continue

            if request_id is not None:
                await _send_result(websocket, request_id, result)
    except WebSocketDisconnect:
        return
    finally:
        await active_dispatcher.close(sessions)


async def _send_session_update(websocket: WebSocket, session_id: str, update: dict[str, Any]) -> None:
    await websocket.send_json(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": update,
            },
        }
    )


async def _send_result(websocket: WebSocket, request_id: JsonRpcId, result: dict[str, Any]) -> None:
    await websocket.send_json({"jsonrpc": "2.0", "id": request_id, "result": result})


async def _send_error(websocket: WebSocket, request_id: JsonRpcId, code: int, message: str) -> None:
    await websocket.send_json(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
    )


def _request_id(message: dict[str, Any]) -> JsonRpcId:
    raw_id = message.get("id")
    if isinstance(raw_id, str | int) or raw_id is None:
        return raw_id
    return str(raw_id)


def _params(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
