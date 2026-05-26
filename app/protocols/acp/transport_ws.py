from __future__ import annotations

import asyncio
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
    prompt_tasks: dict[str, asyncio.Task[None]] = {}
    send_lock = asyncio.Lock()
    active_dispatcher = dispatcher or AcpDispatcher()

    async def send_update(session_id: str, update: dict[str, Any]) -> None:
        async with send_lock:
            await _send_session_update(websocket, session_id, update)

    async def send_result(request_id: JsonRpcId, result: dict[str, Any]) -> None:
        async with send_lock:
            await _send_result(websocket, request_id, result)

    async def send_error(request_id: JsonRpcId, code: int, message: str) -> None:
        async with send_lock:
            await _send_error(websocket, request_id, code, message)

    async def run_prompt(request_id: JsonRpcId, params: dict[str, Any], task_session_id: str | None) -> None:
        try:
            result = await active_dispatcher.dispatch(sessions, "prompt", params, send_update)
        except asyncio.CancelledError:
            if request_id is not None:
                await send_result(request_id, {"stopReason": "cancelled"})
        except FileNotFoundError as exc:
            if request_id is not None:
                await send_error(request_id, -32004, str(exc))
        except ValidationError as exc:
            if request_id is not None:
                await send_error(request_id, -32602, exc.errors()[0]["msg"])
        except ValueError as exc:
            if request_id is not None:
                await send_error(request_id, -32602, str(exc))
        except Exception as exc:
            if request_id is not None:
                await send_error(request_id, -32000, str(exc))
        else:
            if request_id is not None:
                await send_result(request_id, result)
        finally:
            if task_session_id is not None and prompt_tasks.get(task_session_id) is asyncio.current_task():
                prompt_tasks.pop(task_session_id, None)

    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await send_error(None, -32600, "JSON-RPC message must be an object")
                continue

            request_id = _request_id(message)
            method = message.get("method")
            params = _params(message.get("params"))

            if not isinstance(method, str) or not method:
                await send_error(request_id, -32600, "JSON-RPC method is required")
                continue

            normalized_method = method.replace("-", "_")
            if normalized_method in {"prompt", "session/prompt"}:
                task_session_id = _session_id(params)
                task = asyncio.create_task(run_prompt(request_id, params, task_session_id))
                if task_session_id is not None:
                    previous = prompt_tasks.get(task_session_id)
                    if previous is not None and not previous.done():
                        previous.cancel()
                    prompt_tasks[task_session_id] = task
                continue

            if normalized_method in {"cancel", "session/cancel"}:
                session_id = _session_id(params)
                if session_id is not None:
                    cancel_task = prompt_tasks.pop(session_id) if session_id in prompt_tasks else None
                    if cancel_task is not None and not cancel_task.done():
                        cancel_task.cancel()

            if normalized_method in {"close_session", "session/close"}:
                session_id = _session_id(params)
                if session_id is not None:
                    close_task = prompt_tasks.pop(session_id) if session_id in prompt_tasks else None
                    if close_task is not None and not close_task.done():
                        close_task.cancel()

            try:
                result = await active_dispatcher.dispatch(sessions, method, params, send_update)
            except FileNotFoundError as exc:
                await send_error(request_id, -32004, str(exc))
                continue
            except ValidationError as exc:
                await send_error(request_id, -32602, exc.errors()[0]["msg"])
                continue
            except ValueError as exc:
                await send_error(request_id, -32602, str(exc))
                continue
            except Exception as exc:
                await send_error(request_id, -32000, str(exc))
                continue

            if request_id is not None:
                await send_result(request_id, result)
    except WebSocketDisconnect:
        return
    finally:
        for task in prompt_tasks.values():
            if not task.done():
                task.cancel()
        if prompt_tasks:
            await asyncio.gather(*prompt_tasks.values(), return_exceptions=True)
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


def _session_id(params: dict[str, Any]) -> str | None:
    value = params.get("sessionId") or params.get("session_id")
    return value if isinstance(value, str) and value.strip() else None
