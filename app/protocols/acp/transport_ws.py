from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
from typing import Any

import httpx
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.diagnostics import diagnostic_json, env_flag, env_int
from app.protocols.acp.dispatcher import AcpDispatcher
from app.protocols.acp.event_broker import acp_event_broker
from app.protocols.acp.schemas import AcpWebSocketSession, JsonRpcId


ACP_PROMPT_KEEPALIVE_SECONDS = max(
    1.0,
    float(os.getenv("ACP_PROMPT_KEEPALIVE_SECONDS", "10") or "10"),
)
ACP_PROMPT_KEEPALIVE_ENABLED = env_flag("ACP_PROMPT_KEEPALIVE_ENABLED", "0")
ACP_PROMPT_KEEPALIVE_TOOL_CALL_ID = "acp-prompt-keepalive"
ACP_PROMPT_PROGRESS_TEXT = "正在处理，请等待..."
ACP_PROMPT_WAITING_TEXT = "已收到请求，正在处理，请等待..."
ACP_PROMPT_STILL_WAITING_TEXT = "仍在处理，请等待..."
ACP_WS_TRACE_PAYLOADS = env_flag("ACP_WS_TRACE_PAYLOADS", "1")
ACP_WS_TRACE_MAX_CHARS = env_int("ACP_WS_TRACE_MAX_CHARS", 100)
ACP_WS_RESULT_PREVIEW_MAX_CHARS = env_int("ACP_WS_RESULT_PREVIEW_MAX_CHARS", 1200)

logger = logging.getLogger("uvicorn.error")
_ACP_WS_CONNECTION_IDS = itertools.count(1)


async def handle_acp_websocket(websocket: WebSocket, dispatcher: AcpDispatcher | None = None) -> None:
    """Serve ACP-shaped JSON-RPC over WebSocket.

    ACP's common editor transport is stdio. This transport keeps the ACP
    lifecycle and JSON-RPC envelope, then uses WebSocket for browser clients.
    """

    await websocket.accept(subprotocol="acp.v1")
    client_label = _client_label(websocket)
    connection_id = f"acp-ws-{next(_ACP_WS_CONNECTION_IDS)}"
    connected_at = time.monotonic()
    logger.info(
        "acp ws connected connection_id=%s client=%s path=%s requested_subprotocols=%s accepted_subprotocol=%s",
        connection_id,
        client_label,
        websocket.url.path,
        websocket.headers.get("sec-websocket-protocol"),
        "acp.v1",
    )
    sessions: dict[str, AcpWebSocketSession] = {}
    prompt_tasks: dict[str, asyncio.Task[None]] = {}
    keepalive_sessions: set[str] = set()
    platform_sessions: set[str] = set()
    platform_response_ids: dict[str, str] = {}
    send_lock = asyncio.Lock()
    active_dispatcher = dispatcher or AcpDispatcher()

    async def send_update(session_id: str, update: dict[str, Any]) -> None:
        async with send_lock:
            await _send_session_update(websocket, session_id, update, connection_id=connection_id)
            await acp_event_broker.publish(
                _subscription_keys(sessions, session_id),
                "session/update",
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": session_id,
                        "update": update,
                    },
                },
            )
            if session_id in platform_sessions:
                await _send_platform_update(
                    websocket,
                    session_id,
                    update,
                    response_id=platform_response_ids.get(session_id),
                    connection_id=connection_id,
                )

    async def send_result(request_id: JsonRpcId, result: dict[str, Any]) -> None:
        async with send_lock:
            await _send_result(websocket, request_id, result, connection_id=connection_id)

    async def send_error(request_id: JsonRpcId, code: int, message: str) -> None:
        async with send_lock:
            await _send_error(websocket, request_id, code, message, connection_id=connection_id)

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
            if ACP_PROMPT_KEEPALIVE_ENABLED:
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
                await _publish_prompt_result(
                    sessions,
                    task_session_id,
                    request_id,
                    {"stopReason": "cancelled"},
                )
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
                await _publish_prompt_error(sessions, task_session_id, request_id, -32004, str(exc))
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
                await _publish_prompt_error(sessions, task_session_id, request_id, -32602, error_message)
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
                await _publish_prompt_error(sessions, task_session_id, request_id, -32602, str(exc))
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_error_end(task_session_id, request_id, str(exc))
        except Exception as exc:
            error_message = _user_visible_error_message(exc)
            user_text = f"请求处理失败：{error_message}"
            logger.exception(
                "acp prompt failed session_id=%s request_id=%s",
                task_session_id,
                request_id,
            )
            if task_session_id is not None and task_session_id in platform_sessions:
                await send_platform_error_end(task_session_id, request_id, error_message)
            elif task_session_id is not None:
                await send_prompt_error_message(task_session_id, error_message)
            if request_id is not None:
                error_result = _prompt_error_result(sessions, task_session_id, user_text, error_message)
                await send_result(request_id, error_result)
                await _publish_prompt_result(sessions, task_session_id, request_id, error_result)
        else:
            logger.info("acp prompt completed session_id=%s request_id=%s", task_session_id, request_id)
            if task_session_id is not None and task_session_id in platform_sessions:
                if not platform_stream_started:
                    await send_platform_response_start(task_session_id, request_id)
                await send_platform_result_chunks(task_session_id, result)
                await send_platform_response_end(task_session_id, request_id, result)
            if request_id is not None:
                await send_result(request_id, result)
                await _publish_prompt_result(sessions, task_session_id, request_id, result)
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
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": text},
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "acp.prompt.wait_message",
                        "data": {"sequence": sequence, "text": text},
                    }
                },
            },
        )

    async def send_prompt_error_message(session_id: str, message: str) -> None:
        if session_id not in sessions:
            return
        await send_update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": f"请求处理失败：{message}"},
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "acp.prompt.error",
                        "data": {"message": message},
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
                connection_id=connection_id,
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
                    connection_id=connection_id,
                )

    async def send_platform_response_end(session_id: str, request_id: JsonRpcId, result: dict[str, Any]) -> None:
        end_params = _platform_response_end_params(request_id, result)
        logger.info(
            "\n===== ACP 平台响应结束 | platform response_end =====\n"
            "连接ID: %s\n"
            "会话ID: %s\n"
            "请求ID: %s\n"
            "响应ID: %s\n"
            "停止原因: %s\n"
            "线程ID: %s\n"
            "运行状态: %s\n"
            "运行ID: %s\n"
            "回复字符数: %s\n"
            "内容项数: %s\n"
            "回复内容(最多 %s 字符):\n%s\n"
            "完整参数预览:\n%s\n"
            "===== ACP 平台响应结束完成 =====",
            connection_id,
            session_id,
            request_id,
            platform_response_ids.get(session_id),
            end_params.get("stopReason"),
            end_params.get("threadId"),
            end_params.get("status"),
            _platform_result_run_id(end_params),
            len(end_params.get("reply")) if isinstance(end_params.get("reply"), str) else 0,
            len(end_params.get("content")) if isinstance(end_params.get("content"), list) else 0,
            ACP_WS_RESULT_PREVIEW_MAX_CHARS,
            _preview_log_text(_string_or_empty(end_params.get("reply")), max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
            diagnostic_json(end_params, max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
        )
        async with send_lock:
            await _send_platform_event(
                websocket,
                session_id,
                "session.response_end",
                end_params,
                response_id=platform_response_ids.get(session_id),
                connection_id=connection_id,
            )

    async def send_platform_error_end(session_id: str, request_id: JsonRpcId, message: str) -> None:
        async with send_lock:
            await _send_platform_event(
                websocket,
                session_id,
                "session.response_chunk",
                {"chunk": {"content": f"请求处理失败：{message}"}},
                response_id=platform_response_ids.get(session_id),
                connection_id=connection_id,
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
                connection_id=connection_id,
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
                "acp ws received connection_id=%s client=%s method=%s request_id=%s session_id=%s param_keys=%s "
                "bridge_container_keys=%s active_sessions=%s",
                connection_id,
                client_label,
                method,
                request_id,
                _session_id(params),
                sorted(params.keys()),
                _bridge_container_key_summary(params),
                sorted(sessions.keys()),
            )
            _log_ws_trace(
                "recv",
                message,
                client=client_label,
                connection_id=connection_id,
                method=method,
                request_id=request_id,
            )

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
    except WebSocketDisconnect as exc:
        logger.info(
            "acp ws disconnected connection_id=%s client=%s code=%s reason=%s active_sessions=%s",
            connection_id,
            client_label,
            exc.code,
            exc.reason,
            sorted(sessions.keys()),
        )
        return
    finally:
        active_sessions = sorted(sessions.keys())
        pending_prompt_count = sum(1 for task in prompt_tasks.values() if not task.done())
        for task in prompt_tasks.values():
            if not task.done():
                task.cancel()
        if prompt_tasks:
            await asyncio.gather(*prompt_tasks.values(), return_exceptions=True)
        await active_dispatcher.close(sessions)
        logger.info(
            "acp ws cleanup complete connection_id=%s client=%s duration_ms=%s closed_sessions=%s "
            "cancelled_prompt_tasks=%s platform_sessions=%s keepalive_sessions=%s",
            connection_id,
            client_label,
            int((time.monotonic() - connected_at) * 1000),
            active_sessions,
            pending_prompt_count,
            sorted(platform_sessions),
            sorted(keepalive_sessions),
        )


async def _send_session_update(
    websocket: WebSocket,
    session_id: str,
    update: dict[str, Any],
    *,
    connection_id: str | None = None,
) -> None:
    update_text = _content_text(update)
    logger.info(
        "acp ws send update session_id=%s session_update=%s runtime_event=%s tool_call_id=%s status=%s "
        "content_text_chars=%s content_preview=%s",
        session_id,
        update.get("sessionUpdate"),
        _runtime_event_type(update),
        update.get("toolCallId"),
        update.get("status"),
        len(update_text),
        _preview_log_text(update_text, max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
    )
    payload = {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {
            "sessionId": session_id,
            "update": update,
        },
    }
    _log_ws_trace("send", payload, connection_id=connection_id, method="session/update", session_id=session_id)
    await _safe_send_json(
        websocket,
        payload,
        connection_id=connection_id,
        method="session/update",
        session_id=session_id,
    )


async def _send_platform_update(
    websocket: WebSocket,
    session_id: str,
    update: dict[str, Any],
    *,
    response_id: str | None = None,
    connection_id: str | None = None,
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
        connection_id=connection_id,
    )


async def _send_platform_event(
    websocket: WebSocket,
    session_id: str,
    event_type: str,
    params: dict[str, Any] | None = None,
    *,
    response_id: str | None = None,
    connection_id: str | None = None,
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
        "acp ws send platform event session_id=%s type=%s response_id=%s content_text_chars=%s content_preview=%s",
        session_id,
        event_type,
        resolved_response_id,
        _platform_event_content_chars(event_params),
        _preview_log_text(_platform_event_content(event_params), max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
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
    _log_ws_trace("send", session_event_payload, connection_id=connection_id, method="session.event", session_id=session_id)
    if not await _safe_send_json(
        websocket,
        session_event_payload,
        connection_id=connection_id,
        method="session.event",
        session_id=session_id,
    ):
        return
    _log_ws_trace("send", agent_message_payload, connection_id=connection_id, method="agent.message", session_id=session_id)
    await _safe_send_json(
        websocket,
        agent_message_payload,
        connection_id=connection_id,
        method="agent.message",
        session_id=session_id,
    )


async def _send_result(
    websocket: WebSocket,
    request_id: JsonRpcId,
    result: dict[str, Any],
    *,
    connection_id: str | None = None,
) -> None:
    result_payload = result.get("result")
    result_data = result_payload if isinstance(result_payload, dict) else {}
    reply = result_data.get("reply")
    metadata = result_data.get("metadata")
    metadata_data = metadata if isinstance(metadata, dict) else {}
    logger.info(
        "\n===== ACP WebSocket 最终返回 | ws jsonrpc result =====\n"
        "连接ID: %s\n"
        "请求ID: %s\n"
        "停止原因: %s\n"
        "线程ID: %s\n"
        "运行ID: %s\n"
        "运行状态: %s\n"
        "回复字符数: %s\n"
        "内容项数: %s\n"
        "回复内容(最多 %s 字符):\n%s\n"
        "完整结果预览:\n%s\n"
        "===== ACP WebSocket 最终返回结束 =====",
        connection_id,
        request_id,
        result.get("stopReason"),
        result.get("threadId"),
        metadata_data.get("run_id"),
        result_data.get("status"),
        len(reply) if isinstance(reply, str) else 0,
        len(result.get("content") or []) if isinstance(result.get("content"), list) else 0,
        ACP_WS_RESULT_PREVIEW_MAX_CHARS,
        _preview_log_text(_string_or_empty(reply), max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
        diagnostic_json(result, max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
    )
    payload = {"jsonrpc": "2.0", "id": request_id, "result": result}
    _log_ws_trace("send", payload, connection_id=connection_id, method="result", request_id=request_id)
    await _safe_send_json(
        websocket,
        payload,
        connection_id=connection_id,
        method="result",
        request_id=request_id,
    )


async def _send_error(
    websocket: WebSocket,
    request_id: JsonRpcId,
    code: int,
    message: str,
    *,
    connection_id: str | None = None,
) -> None:
    logger.info("acp ws send error request_id=%s code=%s message=%s", request_id, code, message)
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }
    _log_ws_trace("send", payload, connection_id=connection_id, method="error", request_id=request_id)
    await _safe_send_json(
        websocket,
        payload,
        connection_id=connection_id,
        method="error",
        request_id=request_id,
    )


async def _safe_send_json(
    websocket: WebSocket,
    payload: dict[str, Any],
    *,
    connection_id: str | None = None,
    method: str | None = None,
    request_id: JsonRpcId | None = None,
    session_id: str | None = None,
) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except RuntimeError as exc:
        message = str(exc)
        if "websocket.close" not in message and "response already completed" not in message:
            raise
        logger.info(
            "acp ws send skipped connection_id=%s method=%s request_id=%s session_id=%s reason=websocket_closed",
            connection_id,
            method,
            request_id,
            session_id,
        )
        return False


def _user_visible_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        status_text = f"{response.status_code} {response.reason_phrase}".strip()
        message = f"模型连接/调用失败：模型服务返回 {status_text}"
        if exc.request is not None:
            message = f"{message}；地址：{exc.request.url}"
        detail = response.text.strip()
        if detail:
            message = f"{message}；响应：{detail[:1000]}"
        return message
    if isinstance(exc, httpx.TimeoutException):
        return f"模型连接超时：{exc}"
    if isinstance(exc, httpx.ConnectError):
        return f"模型连接失败：{exc}"
    if isinstance(exc, httpx.HTTPError):
        return f"模型连接/调用失败：{exc}"
    return str(exc)


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


def _subscription_keys(sessions: dict[str, AcpWebSocketSession], session_id: str | None) -> set[str]:
    keys = {session_id or ""}
    session = sessions.get(session_id or "")
    if session is not None:
        keys.add(session.thread_id)
    return {key for key in keys if key}


async def _publish_prompt_result(
    sessions: dict[str, AcpWebSocketSession],
    session_id: str | None,
    request_id: JsonRpcId,
    result: dict[str, Any],
) -> None:
    await acp_event_broker.publish(
        _subscription_keys(sessions, session_id),
        "result",
        {"jsonrpc": "2.0", "id": request_id, "result": result},
    )


async def _publish_prompt_error(
    sessions: dict[str, AcpWebSocketSession],
    session_id: str | None,
    request_id: JsonRpcId,
    code: int,
    message: str,
) -> None:
    await acp_event_broker.publish(
        _subscription_keys(sessions, session_id),
        "error",
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        },
    )


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


def _prompt_error_result(
    sessions: dict[str, AcpWebSocketSession],
    session_id: str | None,
    reply: str,
    error_message: str,
) -> dict[str, Any]:
    session = sessions.get(session_id or "")
    thread_id = session.thread_id if session is not None else session_id or ""
    agent_name = session.agent_name if session is not None else "default"
    content = [{"type": "text", "text": reply}]
    result = {
        "agent": agent_name,
        "thread_id": thread_id,
        "status": "failed",
        "reply": reply,
        "content": content,
        "artifacts": [],
        "metadata": {"error": error_message},
    }
    return {
        "stopReason": "end_turn",
        "threadId": thread_id,
        "agentName": agent_name,
        "content": content,
        "result": result,
    }


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
    connection_id: str | None = None,
    method: object = None,
    request_id: JsonRpcId = None,
    session_id: str | None = None,
) -> None:
    if not ACP_WS_TRACE_PAYLOADS:
        return
    serialized = diagnostic_json(payload, max_chars=ACP_WS_TRACE_MAX_CHARS)
    logger.info(
        "acp ws trace direction=%s connection_id=%s client=%s method=%s request_id=%s session_id=%s payload=%s",
        direction,
        connection_id,
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
    return len(_content_text(update))


def _content_text(update: dict[str, Any]) -> str:
    content = update.get("content")
    if not isinstance(content, dict):
        return ""
    text = content.get("text")
    return text if isinstance(text, str) else ""


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


def _platform_event_content(params: dict[str, Any]) -> str:
    chunk = params.get("chunk")
    if isinstance(chunk, dict):
        content = chunk.get("content")
        return content if isinstance(content, str) else ""
    reply = params.get("reply")
    return reply if isinstance(reply, str) else ""


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


def _platform_response_end_params(request_id: JsonRpcId, result: dict[str, Any]) -> dict[str, Any]:
    raw_result = result.get("result")
    result_data = raw_result if isinstance(raw_result, dict) else {}
    reply = result_data.get("reply")
    content = result.get("content")
    if not isinstance(content, list):
        content = result_data.get("content")
    params = {
        "requestId": request_id,
        "stopReason": result.get("stopReason"),
        "threadId": result.get("threadId"),
        "agentName": result.get("agentName"),
        "status": result_data.get("status"),
        "reply": reply if isinstance(reply, str) else "",
        "content": content if isinstance(content, list) else [],
        "metadata": result_data.get("metadata") if isinstance(result_data.get("metadata"), dict) else {},
        "result": raw_result,
    }
    return params


def _platform_result_run_id(params: dict[str, Any]) -> object:
    result = params.get("result")
    if not isinstance(result, dict):
        return None
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata.get("run_id")


def _string_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _preview_log_text(value: str, *, max_chars: int) -> str:
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated chars={len(value) - max_chars}>"


def _platform_event_content_chars(params: dict[str, Any]) -> int:
    return len(_platform_event_content(params))
