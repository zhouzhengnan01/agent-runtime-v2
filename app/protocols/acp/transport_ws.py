from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.diagnostics import diagnostic_json, env_flag, env_int
from app.protocols.acp.dispatcher import AcpDispatcher
from app.protocols.acp.schemas import AcpWebSocketSession, JsonRpcId


ACP_PROMPT_KEEPALIVE_SECONDS = max(
    1.0,
    float(os.getenv("ACP_PROMPT_KEEPALIVE_SECONDS", "10") or "10"),
)
ACP_PROMPT_KEEPALIVE_TOOL_CALL_ID = "acp-prompt-keepalive"
ACP_PROMPT_PROGRESS_TEXT = "正在处理，请等待..."
ACP_PROMPT_WAITING_TEXT = "已收到请求，正在处理，请等待..."
ACP_PROMPT_STILL_WAITING_TEXT = "仍在处理，请等待..."
ACP_WS_TRACE_PAYLOADS = env_flag("ACP_WS_TRACE_PAYLOADS", "1")
ACP_WS_TRACE_MAX_CHARS = env_int("ACP_WS_TRACE_MAX_CHARS", 0)

logger = logging.getLogger("uvicorn.error")


async def handle_acp_websocket(websocket: WebSocket, dispatcher: AcpDispatcher | None = None) -> None:
    """Serve ACP-shaped JSON-RPC over WebSocket.

    ACP's common editor transport is stdio. This transport keeps the ACP
    lifecycle and JSON-RPC envelope, then uses WebSocket for browser clients.
    """

    await websocket.accept(subprotocol="acp.v1")
    client_label = _client_label(websocket)
    logger.info("acp ws connected client=%s", client_label)
    sessions: dict[str, AcpWebSocketSession] = {}
    prompt_tasks: dict[str, asyncio.Task[None]] = {}
    keepalive_sessions: set[str] = set()
    platform_sessions: set[str] = set()
    platform_response_ids: dict[str, str] = {}
    send_lock = asyncio.Lock()
    active_dispatcher = dispatcher or AcpDispatcher()

    async def send_update(session_id: str, update: dict[str, Any]) -> None:
        async with send_lock:
            await _send_session_update(websocket, session_id, update)
            if session_id in platform_sessions:
                await _send_platform_update(
                    websocket,
                    session_id,
                    update,
                    response_id=platform_response_ids.get(session_id),
                )

    async def send_result(request_id: JsonRpcId, result: dict[str, Any]) -> None:
        async with send_lock:
            await _send_result(websocket, request_id, result)

    async def send_error(request_id: JsonRpcId, code: int, message: str) -> None:
        async with send_lock:
            await _send_error(websocket, request_id, code, message)

    async def run_prompt(request_id: JsonRpcId, params: dict[str, Any], task_session_id: str | None) -> None:
        keepalive_task: asyncio.Task[None] | None = None
        platform_stream_started = False
        if task_session_id is not None:
            if task_session_id in platform_sessions:
                platform_response_ids[task_session_id] = _platform_response_id_for_prompt(
                    task_session_id,
                    request_id,
                    params,
                )
                await send_platform_response_start(task_session_id, request_id)
                platform_stream_started = True
            await send_prompt_progress(task_session_id, sequence=0)
            await send_prompt_wait_message(task_session_id, sequence=0)
            keepalive_task = asyncio.create_task(send_prompt_keepalive(task_session_id))
        logger.info("acp prompt started session_id=%s request_id=%s", task_session_id, request_id)
        try:
            result = await active_dispatcher.dispatch(sessions, "prompt", params, send_update)
        except asyncio.CancelledError:
            logger.info("acp prompt cancelled session_id=%s request_id=%s", task_session_id, request_id)
            if request_id is not None:
                await send_result(request_id, {"stopReason": "cancelled"})
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_response_end(
                    task_session_id,
                    request_id,
                    {"stopReason": "cancelled", "result": {"reply": "请求已取消。"}},
                )
        except FileNotFoundError as exc:
            logger.warning(
                "acp prompt file not found session_id=%s request_id=%s error=%s",
                task_session_id,
                request_id,
                exc,
            )
            if request_id is not None:
                await send_error(request_id, -32004, str(exc))
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_error_end(task_session_id, request_id, str(exc))
        except ValidationError as exc:
            error_message = exc.errors()[0]["msg"]
            logger.warning(
                "acp prompt validation failed session_id=%s request_id=%s error=%s",
                task_session_id,
                request_id,
                error_message,
            )
            if request_id is not None:
                await send_error(request_id, -32602, error_message)
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_error_end(task_session_id, request_id, error_message)
        except ValueError as exc:
            logger.warning(
                "acp prompt value error session_id=%s request_id=%s error=%s",
                task_session_id,
                request_id,
                exc,
            )
            if request_id is not None:
                await send_error(request_id, -32602, str(exc))
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_error_end(task_session_id, request_id, str(exc))
        except Exception as exc:
            logger.exception(
                "acp prompt failed session_id=%s request_id=%s",
                task_session_id,
                request_id,
            )
            if request_id is not None:
                await send_error(request_id, -32000, str(exc))
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_error_end(task_session_id, request_id, str(exc))
        else:
            logger.info("acp prompt completed session_id=%s request_id=%s", task_session_id, request_id)
            if task_session_id is not None and task_session_id in platform_sessions:
                if not platform_stream_started:
                    await send_platform_response_start(task_session_id, request_id)
                await send_platform_result_chunks(task_session_id, result)
                await send_platform_response_end(task_session_id, request_id, result)
            if request_id is not None:
                await send_result(request_id, result)
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
                if (
                    task_session_id is not None
                    and task_session_id in sessions
                    and task_session_id in keepalive_sessions
                ):
                    await send_prompt_keepalive_completed(task_session_id)
                    keepalive_sessions.discard(task_session_id)
            if task_session_id is not None and prompt_tasks.get(task_session_id) is asyncio.current_task():
                prompt_tasks.pop(task_session_id, None)
                platform_response_ids.pop(task_session_id, None)

    async def send_prompt_keepalive(session_id: str) -> None:
        sequence = 0
        while True:
            await asyncio.sleep(ACP_PROMPT_KEEPALIVE_SECONDS)
            if session_id not in sessions:
                return
            sequence += 1
            keepalive_sessions.add(session_id)
            logger.info("acp prompt keepalive session_id=%s sequence=%s", session_id, sequence)
            await send_prompt_progress(session_id, sequence=sequence)
            await send_prompt_wait_message(session_id, sequence=sequence)
            await send_update(
                session_id,
                {
                    "sessionUpdate": "tool_call" if sequence == 1 else "tool_call_update",
                    "toolCallId": ACP_PROMPT_KEEPALIVE_TOOL_CALL_ID,
                    "title": "processing",
                    "kind": "other",
                    "status": "in_progress",
                    "_meta": {
                        "jetlinksRuntimeEvent": {
                            "type": "acp.prompt.keepalive",
                            "data": {"sequence": sequence},
                        }
                    },
                },
            )

    async def send_prompt_progress(session_id: str, *, sequence: int) -> None:
        if session_id not in sessions:
            return
        await send_update(
            session_id,
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": ACP_PROMPT_PROGRESS_TEXT},
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "acp.prompt.progress",
                        "data": {"sequence": sequence},
                    }
                },
            },
        )

    async def send_prompt_wait_message(session_id: str, *, sequence: int) -> None:
        if session_id not in sessions:
            return
        text = ACP_PROMPT_WAITING_TEXT if sequence <= 0 else ACP_PROMPT_STILL_WAITING_TEXT
        await send_update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "acp.prompt.wait_message",
                        "data": {"sequence": sequence, "text": text},
                    }
                },
            },
        )

    async def send_prompt_keepalive_completed(session_id: str) -> None:
        await send_update(
            session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": ACP_PROMPT_KEEPALIVE_TOOL_CALL_ID,
                "title": "processing",
                "kind": "other",
                "status": "completed",
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "acp.prompt.keepalive.completed",
                        "data": {},
                    }
                },
            },
        )

    async def send_platform_response_start(session_id: str, request_id: JsonRpcId) -> None:
        async with send_lock:
            await _send_platform_event(
                websocket,
                session_id,
                "session.response_start",
                {"requestId": request_id},
                response_id=platform_response_ids.get(session_id),
            )

    async def send_platform_result_chunks(session_id: str, result: dict[str, Any]) -> None:
        for chunk in _platform_chunks_from_result(result):
            async with send_lock:
                await _send_platform_event(
                    websocket,
                    session_id,
                    "session.response_chunk",
                    {"chunk": {"content": chunk}},
                    response_id=platform_response_ids.get(session_id),
                )

    async def send_platform_response_end(session_id: str, request_id: JsonRpcId, result: dict[str, Any]) -> None:
        async with send_lock:
            await _send_platform_event(
                websocket,
                session_id,
                "session.response_end",
                {
                    "requestId": request_id,
                    "stopReason": result.get("stopReason"),
                    "threadId": result.get("threadId"),
                    "result": result.get("result"),
                },
                response_id=platform_response_ids.get(session_id),
            )

    async def send_platform_error_end(session_id: str, request_id: JsonRpcId, message: str) -> None:
        async with send_lock:
            await _send_platform_event(
                websocket,
                session_id,
                "session.response_chunk",
                {"chunk": {"content": f"请求处理失败：{message}"}},
                response_id=platform_response_ids.get(session_id),
            )
            await _send_platform_event(
                websocket,
                session_id,
                "session.response_end",
                {
                    "requestId": request_id,
                    "stopReason": "error",
                    "error": message,
                },
                response_id=platform_response_ids.get(session_id),
            )

    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await send_error(None, -32600, "JSON-RPC message must be an object")
                continue

            request_id = _request_id(message)
            method = message.get("method")
            params = _params(message.get("params"))
            logger.info(
                "acp ws received client=%s method=%s request_id=%s session_id=%s param_keys=%s bridge_container_keys=%s active_sessions=%s",
                client_label,
                method,
                request_id,
                _session_id(params),
                sorted(params.keys()),
                _bridge_container_key_summary(params),
                sorted(sessions.keys()),
            )
            _log_ws_trace("recv", message, client=client_label, method=method, request_id=request_id)

            if not isinstance(method, str) or not method:
                await send_error(request_id, -32600, "JSON-RPC method is required")
                continue

            normalized_method = method.replace("-", "_")
            if normalized_method in {"prompt", "session/prompt", "agent.command"}:
                task_session_id = _session_id(params) or _single_active_session_id(sessions)
                if task_session_id is not None and _session_id(params) is None:
                    params = {**params, "sessionId": task_session_id}
                platform_agent_command = normalized_method == "agent.command" and task_session_id in platform_sessions
                task_request_id = None if platform_agent_command else request_id
                if platform_agent_command:
                    params = {
                        **params,
                        "_platformResponseId": _platform_response_id(task_session_id, request_id),
                    }
                if platform_agent_command and request_id is not None:
                    await send_result(
                        request_id,
                        {
                            "sessionId": task_session_id,
                            "threadId": sessions[task_session_id].thread_id if task_session_id in sessions else None,
                            "accepted": True,
                        },
                    )
                task = asyncio.create_task(run_prompt(task_request_id, params, task_session_id))
                if task_session_id is not None:
                    previous = prompt_tasks.get(task_session_id)
                    if previous is not None and not previous.done():
                        previous.cancel()
                    prompt_tasks[task_session_id] = task
                continue

            if normalized_method == "session.init":
                result = await active_dispatcher.dispatch(sessions, method, params, send_update)
                session_id = result.get("sessionId")
                if isinstance(session_id, str) and session_id:
                    platform_sessions.add(session_id)
                if request_id is not None:
                    await send_result(request_id, result)
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
        logger.info("acp ws disconnected client=%s active_sessions=%s", client_label, sorted(sessions.keys()))
        return
    finally:
        for task in prompt_tasks.values():
            if not task.done():
                task.cancel()
        if prompt_tasks:
            await asyncio.gather(*prompt_tasks.values(), return_exceptions=True)
        await active_dispatcher.close(sessions)


async def _send_session_update(websocket: WebSocket, session_id: str, update: dict[str, Any]) -> None:
    logger.info(
        "acp ws send update session_id=%s session_update=%s runtime_event=%s tool_call_id=%s status=%s content_text_chars=%s",
        session_id,
        update.get("sessionUpdate"),
        _runtime_event_type(update),
        update.get("toolCallId"),
        update.get("status"),
        _content_text_chars(update),
    )
    payload = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": update,
        },
    }
    _log_ws_trace("send", payload, method="session/update", session_id=session_id)
    await websocket.send_json(payload)


async def _send_platform_update(
    websocket: WebSocket,
    session_id: str,
    update: dict[str, Any],
    *,
    response_id: str | None = None,
) -> None:
    event_type = _platform_type_for_update(update)
    if event_type is None:
        return
    await _send_platform_event(
        websocket,
        session_id,
        event_type,
        {"chunk": {"content": _platform_update_content(update)}},
        response_id=response_id,
    )


async def _send_platform_event(
    websocket: WebSocket,
    session_id: str,
    event_type: str,
    params: dict[str, Any] | None = None,
    *,
    response_id: str | None = None,
) -> None:
    event_params = params or {}
    resolved_response_id = response_id or _platform_response_id(session_id, None)
    headers = {
        "responseId": resolved_response_id,
        "messageId": f"{resolved_response_id}:{event_type}",
        "origin": "assistant",
        "provider": "jetlinks-agent-runtime-v2",
    }
    logger.info(
        "acp ws send platform event session_id=%s type=%s response_id=%s content_text_chars=%s",
        session_id,
        event_type,
        resolved_response_id,
        _platform_event_content_chars(event_params),
    )
    session_event_payload = {
        "jsonrpc": "2.0",
        "method": "session.event",
        "params": {
            "sessionId": session_id,
            "type": event_type,
            "params": event_params,
            "headers": headers,
        },
    }
    agent_message_payload = {
        "jsonrpc": "2.0",
        "method": "agent.message",
        "params": {
            "sessionId": session_id,
            "type": event_type,
            "params": event_params,
            "headers": headers,
        },
    }
    _log_ws_trace("send", session_event_payload, method="session.event", session_id=session_id)
    await websocket.send_json(session_event_payload)
    _log_ws_trace("send", agent_message_payload, method="agent.message", session_id=session_id)
    await websocket.send_json(agent_message_payload)


async def _send_result(websocket: WebSocket, request_id: JsonRpcId, result: dict[str, Any]) -> None:
    logger.info(
        "acp ws send result request_id=%s stop_reason=%s thread_id=%s content_items=%s",
        request_id,
        result.get("stopReason"),
        result.get("threadId"),
        len(result.get("content") or []) if isinstance(result.get("content"), list) else 0,
    )
    payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
    _log_ws_trace("send", payload, method="result", request_id=request_id)
    await websocket.send_json(payload)


async def _send_error(websocket: WebSocket, request_id: JsonRpcId, code: int, message: str) -> None:
    logger.info("acp ws send error request_id=%s code=%s message=%s", request_id, code, message)
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
    _log_ws_trace("send", payload, method="error", request_id=request_id)
    await websocket.send_json(payload)


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


def _single_active_session_id(sessions: dict[str, AcpWebSocketSession]) -> str | None:
    if len(sessions) != 1:
        return None
    return next(iter(sessions))


def _platform_response_id(session_id: str, request_id: JsonRpcId) -> str:
    if request_id is None:
        return f"{session_id}:platform-response"
    return f"{session_id}:{request_id}"


def _platform_response_id_for_prompt(session_id: str, request_id: JsonRpcId, params: dict[str, Any]) -> str:
    value = params.get("_platformResponseId")
    if isinstance(value, str) and value:
        return value
    return _platform_response_id(session_id, request_id)


def _client_label(websocket: WebSocket) -> str:
    client = websocket.client
    if client is None:
        return "unknown"
    return f"{client.host}:{client.port}"


def _bridge_container_key_summary(params: dict[str, Any]) -> list[str]:
    summaries: list[str] = []
    for key in ("parameters", "arguments", "metadata", "context", "extra", "runtimeOptions", "runtime_options"):
        value = params.get(key)
        if isinstance(value, dict):
            summaries.append(f"{key}={','.join(sorted(value.keys()))}")
    return summaries


def _log_ws_trace(
    direction: str,
    payload: dict[str, Any],
    *,
    client: str | None = None,
    method: object = None,
    request_id: JsonRpcId = None,
    session_id: str | None = None,
) -> None:
    if not ACP_WS_TRACE_PAYLOADS:
        return
    serialized = diagnostic_json(payload, max_chars=ACP_WS_TRACE_MAX_CHARS)
    logger.info(
        "acp ws trace direction=%s client=%s method=%s request_id=%s session_id=%s payload=%s",
        direction,
        client,
        method,
        request_id,
        session_id or _session_id(_params(payload.get("params"))),
        serialized,
    )


def _runtime_event_type(update: dict[str, Any]) -> str | None:
    meta = update.get("_meta")
    if not isinstance(meta, dict):
        return None
    event = meta.get("jetlinksRuntimeEvent")
    if not isinstance(event, dict):
        return None
    value = event.get("type")
    return value if isinstance(value, str) else None


def _content_text_chars(update: dict[str, Any]) -> int:
    content = update.get("content")
    if not isinstance(content, dict):
        return 0
    text = content.get("text")
    return len(text) if isinstance(text, str) else 0


def _platform_type_for_update(update: dict[str, Any]) -> str | None:
    if update.get("sessionUpdate") != "agent_message_chunk":
        return None
    content = _platform_update_content(update)
    return "session.response_chunk" if content else None


def _platform_update_content(update: dict[str, Any]) -> str:
    content = update.get("content")
    if not isinstance(content, dict):
        return ""
    text = content.get("text")
    if isinstance(text, str):
        return text
    return ""


def _platform_chunks_from_result(result: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    raw_result = result.get("result")
    if isinstance(raw_result, dict):
        reply = raw_result.get("reply")
        if isinstance(reply, str) and reply:
            chunks.append(reply)
    for block in result.get("content") if isinstance(result.get("content"), list) else []:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text and text not in chunks:
            chunks.append(text)
    return chunks


def _platform_event_content_chars(params: dict[str, Any]) -> int:
    chunk = params.get("chunk")
    if not isinstance(chunk, dict):
        return 0
    content = chunk.get("content")
    return len(content) if isinstance(content, str) else 0
