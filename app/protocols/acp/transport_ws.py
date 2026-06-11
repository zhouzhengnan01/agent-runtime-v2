from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from anyio import BrokenResourceError, ClosedResourceError
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.diagnostics import diagnostic_json, env_flag, env_int
from app.protocols.acp.dispatcher import AcpDispatcher
from app.protocols.acp.schemas import AcpWebSocketSession, JsonRpcId


ACP_PROMPT_KEEPALIVE_SECONDS = max(
    1.0,
    float(os.getenv("ACP_PROMPT_KEEPALIVE_SECONDS", "5") or "5"),
)
ACP_PROMPT_KEEPALIVE_ENABLED = env_flag("ACP_PROMPT_KEEPALIVE_ENABLED", "1")
ACP_PROMPT_KEEPALIVE_TOOL_CALL_ID = "acp-prompt-keepalive"
ACP_PROMPT_PROGRESS_TEXT = "正在处理，请等待..."
ACP_PROMPT_WAITING_TEXT = "已收到请求，正在处理，请等待..."
ACP_PROMPT_STILL_WAITING_TEXT = "仍在处理，请等待..."
ACP_WS_TRACE_PAYLOADS = env_flag("ACP_WS_TRACE_PAYLOADS", "0")
ACP_WS_TRACE_MAX_CHARS = env_int("ACP_WS_TRACE_MAX_CHARS", 100)
ACP_WS_RESULT_PREVIEW_MAX_CHARS = env_int("ACP_WS_RESULT_PREVIEW_MAX_CHARS", 1200)
ACP_WS_THREAD_EVENT_DUMP_ENABLED = env_flag("ACP_WS_THREAD_EVENT_DUMP_ENABLED", "1")
ACP_REVIEW_RESULT_FILE_WATCH_ENABLED = env_flag("ACP_REVIEW_RESULT_FILE_WATCH_ENABLED", "1")
ACP_REVIEW_RESULT_FILE_WATCH_SECONDS = max(
    0.0,
    float(os.getenv("ACP_REVIEW_RESULT_FILE_WATCH_SECONDS", "90") or "90"),
)
ACP_REVIEW_RESULT_FILE_POLL_SECONDS = max(
    0.05,
    float(os.getenv("ACP_REVIEW_RESULT_FILE_POLL_SECONDS", "0.2") or "0.2"),
)
ACP_REVIEW_RESULT_FILENAME = "parking_abnormal_review_result.json"
ACP_WS_THREAD_ROOT = Path(
    os.getenv(
        "ACP_WS_THREAD_ROOT",
        str(Path(__file__).resolve().parents[3] / ".runtime" / "threads"),
    )
)

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
    logger.debug(
        "acp ws connected connection_id=%s client=%s path=%s requested_subprotocols=%s accepted_subprotocol=%s",
        connection_id,
        client_label,
        websocket.url.path,
        websocket.headers.get("sec-websocket-protocol"),
        "acp.v1",
    )
    sessions: dict[str, AcpWebSocketSession] = {}
    prompt_tasks: dict[str, asyncio.Task[None]] = {}
    prompt_task_sessions: dict[str, str] = {}
    prompt_task_ids = itertools.count(1)
    keepalive_sessions: set[str] = set()
    send_lock = asyncio.Lock()
    connection_closed = asyncio.Event()
    active_dispatcher = dispatcher or AcpDispatcher()

    def mark_connection_closed(reason: str) -> None:
        if not connection_closed.is_set():
            logger.info("acp ws marked closed connection_id=%s reason=%s", connection_id, reason)
        connection_closed.set()

    async def send_update(session_id: str, update: dict[str, Any]) -> bool:
        async with send_lock:
            sent = await _send_session_update(websocket, session_id, update, connection_id=connection_id)
        if not sent:
            mark_connection_closed("session_update_send_failed")
        return sent

    async def send_prompt_update(
        session_id: str,
        update: dict[str, Any],
        *,
        platform_events_enabled: bool,
        platform_response_id: str | None,
    ) -> bool:
        async with send_lock:
            if not await _send_session_update(websocket, session_id, update, connection_id=connection_id):
                sent = False
            elif platform_events_enabled:
                sent = await _send_platform_update(
                    websocket,
                    session_id,
                    update,
                    response_id=platform_response_id,
                    connection_id=connection_id,
                )
            else:
                sent = True
        if not sent:
            mark_connection_closed("prompt_update_send_failed")
        return sent

    async def send_result(request_id: JsonRpcId, result: dict[str, Any]) -> bool:
        async with send_lock:
            sent = await _send_result(websocket, request_id, result, connection_id=connection_id)
        if not sent:
            mark_connection_closed("result_send_failed")
        return sent

    async def send_error(request_id: JsonRpcId, code: int, message: str) -> bool:
        async with send_lock:
            sent = await _send_error(websocket, request_id, code, message, connection_id=connection_id)
        if not sent:
            mark_connection_closed("error_send_failed")
        return sent

    async def run_prompt(
        request_id: JsonRpcId,
        params: dict[str, Any],
        task_session_id: str | None,
        *,
        platform_request_id: JsonRpcId = None,
        platform_keepalive_chunks: bool = True,
        platform_events_enabled: bool = True,
    ) -> None:
        keepalive_task: asyncio.Task[None] | None = None
        review_result_watch_task: asyncio.Task[None] | None = None
        platform_stream_started = False
        platform_final_sent = False
        prompt_result_sent = False
        platform_final_send_lock = asyncio.Lock()
        platform_event_request_id = platform_request_id if platform_request_id is not None else request_id
        platform_response_id: str | None = None
        owner_task = asyncio.current_task()

        def raise_if_connection_closed() -> None:
            if connection_closed.is_set():
                raise asyncio.CancelledError()

        async def send_platform_final_result(
            session_id: str,
            final_result: dict[str, Any],
            *,
            source: str,
            cancel_prompt_task: bool = False,
        ) -> bool:
            nonlocal platform_final_sent, prompt_result_sent
            async with platform_final_send_lock:
                if platform_final_sent and (request_id is None or prompt_result_sent):
                    return False
                if platform_events_enabled:
                    _dump_platform_event(
                        session_id,
                        "jetlinks.review_result.finalized",
                        {
                            "source": source,
                            "threadId": final_result.get("threadId"),
                            "status": _result_status(final_result),
                            "replyChars": _result_reply_chars(final_result),
                        },
                        response_id=platform_response_id,
                        connection_id=connection_id,
                        delivery="internal.final_result",
                    )
                if keepalive_task is not None:
                    keepalive_task.cancel()
                if platform_events_enabled and not platform_final_sent:
                    await send_platform_result_chunks(
                        session_id,
                        final_result,
                        response_id=platform_response_id,
                    )
                if request_id is not None and not prompt_result_sent:
                    await send_result(request_id, final_result)
                    prompt_result_sent = True
                if platform_events_enabled and not platform_final_sent:
                    await send_platform_response_end(
                        session_id,
                        platform_event_request_id,
                        final_result,
                        response_id=platform_response_id,
                    )
                    platform_final_sent = True
                if keepalive_task is not None:
                    await asyncio.gather(keepalive_task, return_exceptions=True)
                if cancel_prompt_task:
                    current_task = asyncio.current_task()
                    if owner_task is not None and owner_task is not current_task and not owner_task.done():
                        owner_task.cancel()
                return True

        if task_session_id is not None:
            if platform_events_enabled:
                platform_response_id = _platform_response_id_for_prompt(
                    task_session_id,
                    platform_event_request_id,
                    params,
                )
                if not await send_platform_response_start(
                    task_session_id,
                    platform_event_request_id,
                    response_id=platform_response_id,
                ):
                    raise asyncio.CancelledError()
                platform_stream_started = True
            if ACP_PROMPT_KEEPALIVE_ENABLED:
                if not await send_prompt_progress(
                    task_session_id,
                    sequence=0,
                    platform_events_enabled=platform_events_enabled,
                    platform_response_id=platform_response_id,
                ):
                    raise asyncio.CancelledError()
                if not await send_prompt_wait_message(
                    task_session_id,
                    sequence=0,
                    platform_events_enabled=platform_events_enabled,
                    platform_response_id=platform_response_id,
                ):
                    raise asyncio.CancelledError()
                keepalive_task = asyncio.create_task(
                    send_prompt_keepalive(
                        task_session_id,
                        platform_keepalive_chunks=platform_keepalive_chunks,
                        platform_events_enabled=platform_events_enabled,
                        platform_response_id=platform_response_id,
                    )
                )
            if (
                _should_watch_review_result_file(sessions, task_session_id)
                and ACP_REVIEW_RESULT_FILE_WATCH_ENABLED
            ):
                review_result_watch_task = asyncio.create_task(
                    watch_review_result_file(task_session_id, send_platform_final_result)
                )
        logger.debug("acp prompt started session_id=%s request_id=%s", task_session_id, request_id)
        try:
            async def dispatch_update(session_id: str, update: dict[str, Any]) -> None:
                raise_if_connection_closed()
                sent = await send_prompt_update(
                    session_id,
                    update,
                    platform_events_enabled=platform_events_enabled,
                    platform_response_id=platform_response_id,
                )
                if not sent and session_id == task_session_id:
                    raise asyncio.CancelledError()
                if not platform_events_enabled or session_id != task_session_id or platform_final_sent:
                    return
                final_result = _platform_final_result_from_update(update, sessions.get(session_id))
                if final_result is None:
                    return
                await send_platform_final_result(session_id, final_result, source="runtime_update")

            result = await active_dispatcher.dispatch(sessions, "prompt", params, dispatch_update)
        except asyncio.CancelledError:
            logger.debug(
                "acp prompt cancelled session_id=%s request_id=%s platform_final_sent=%s",
                task_session_id,
                request_id,
                platform_final_sent,
            )
            if platform_final_sent or prompt_result_sent:
                return
            final_result = _platform_review_result_from_session(sessions, task_session_id)
            if (
                final_result is not None
                and task_session_id is not None
                and await send_platform_final_result(task_session_id, final_result, source="cancel_existing_result")
            ):
                return
            if request_id is not None and not prompt_result_sent:
                await send_result(request_id, {"stopReason": "cancelled"})
                prompt_result_sent = True
            if platform_events_enabled and task_session_id is not None:
                await send_platform_response_end(
                    task_session_id,
                    platform_event_request_id,
                    {"stopReason": "cancelled", "result": {"reply": "请求已取消。"}},
                    response_id=platform_response_id,
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
            if platform_events_enabled and task_session_id is not None:
                await send_platform_error_end(
                    task_session_id,
                    platform_event_request_id,
                    str(exc),
                    response_id=platform_response_id,
                )
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
            if platform_events_enabled and task_session_id is not None:
                await send_platform_error_end(
                    task_session_id,
                    platform_event_request_id,
                    error_message,
                    response_id=platform_response_id,
                )
        except ValueError as exc:
            logger.warning(
                "acp prompt value error session_id=%s request_id=%s error=%s",
                task_session_id,
                request_id,
                exc,
            )
            if request_id is not None:
                await send_error(request_id, -32602, str(exc))
            if platform_events_enabled and task_session_id is not None:
                await send_platform_error_end(
                    task_session_id,
                    platform_event_request_id,
                    str(exc),
                    response_id=platform_response_id,
                )
        except Exception as exc:
            error_message = _user_visible_error_message(exc)
            user_text = f"请求处理失败：{error_message}"
            logger.exception(
                "acp prompt failed session_id=%s request_id=%s",
                task_session_id,
                request_id,
            )
            if not platform_events_enabled and task_session_id is not None:
                await send_prompt_error_message(task_session_id, error_message)
            if request_id is not None and not prompt_result_sent:
                await send_result(request_id, _prompt_error_result(sessions, task_session_id, user_text, error_message))
                prompt_result_sent = True
            if platform_events_enabled and task_session_id is not None:
                await send_platform_error_end(
                    task_session_id,
                    platform_event_request_id,
                    error_message,
                    response_id=platform_response_id,
                )
        else:
            logger.info(
                "acp prompt completed session_id=%s request_id=%s stop_reason=%s thread_id=%s run_id=%s "
                "status=%s reply_chars=%s content_items=%s result_preview=%s",
                task_session_id,
                request_id,
                result.get("stopReason"),
                result.get("threadId"),
                _result_run_id(result),
                _result_status(result),
                _result_reply_chars(result),
                _result_content_items(result),
                diagnostic_json(result, max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
            )
            if platform_events_enabled and task_session_id is not None:
                if not platform_final_sent and not platform_stream_started:
                    await send_platform_response_start(
                        task_session_id,
                        platform_event_request_id,
                        response_id=platform_response_id,
                    )
                if not platform_final_sent:
                    await send_platform_result_chunks(
                        task_session_id,
                        result,
                        response_id=platform_response_id,
                    )
            if request_id is not None and not prompt_result_sent:
                await send_result(request_id, result)
                prompt_result_sent = True
            if platform_events_enabled and task_session_id is not None:
                if not platform_final_sent:
                    await send_platform_response_end(
                        task_session_id,
                        platform_event_request_id,
                        result,
                        response_id=platform_response_id,
                    )
                    platform_final_sent = True
        finally:
            if review_result_watch_task is not None:
                review_result_watch_task.cancel()
                await asyncio.gather(review_result_watch_task, return_exceptions=True)
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
                if (
                    task_session_id is not None
                    and task_session_id in sessions
                    and task_session_id in keepalive_sessions
                ):
                    await send_prompt_keepalive_completed(
                        task_session_id,
                        platform_events_enabled=platform_events_enabled,
                        platform_response_id=platform_response_id,
                    )
                    keepalive_sessions.discard(task_session_id)
            current_task = asyncio.current_task()
            for task_key, task in list(prompt_tasks.items()):
                if task is current_task:
                    prompt_tasks.pop(task_key, None)
                    prompt_task_sessions.pop(task_key, None)
                    break

    async def send_prompt_keepalive(
        session_id: str,
        *,
        platform_keepalive_chunks: bool,
        platform_events_enabled: bool,
        platform_response_id: str | None,
    ) -> None:
        sequence = 0
        while True:
            await asyncio.sleep(ACP_PROMPT_KEEPALIVE_SECONDS)
            if connection_closed.is_set():
                return
            if session_id not in sessions:
                return
            sequence += 1
            keepalive_sessions.add(session_id)
            logger.debug("acp prompt keepalive session_id=%s sequence=%s", session_id, sequence)
            if platform_keepalive_chunks and platform_events_enabled:
                if not await send_platform_keepalive_chunk(
                    session_id,
                    sequence=sequence,
                    response_id=platform_response_id,
                ):
                    return
            if not await send_prompt_progress(
                session_id,
                sequence=sequence,
                platform_events_enabled=platform_events_enabled,
                platform_response_id=platform_response_id,
            ):
                return
            if not await send_prompt_wait_message(
                session_id,
                sequence=sequence,
                platform_events_enabled=platform_events_enabled,
                platform_response_id=platform_response_id,
            ):
                return
            sent = await send_prompt_update(
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
                platform_events_enabled=platform_events_enabled,
                platform_response_id=platform_response_id,
            )
            if not sent:
                return

    async def watch_review_result_file(session_id: str, on_result: Any) -> None:
        deadline = time.monotonic() + ACP_REVIEW_RESULT_FILE_WATCH_SECONDS
        while time.monotonic() <= deadline:
            if session_id not in sessions:
                await asyncio.sleep(ACP_REVIEW_RESULT_FILE_POLL_SECONDS)
                continue
            if not _should_watch_review_result_file(sessions, session_id):
                await asyncio.sleep(ACP_REVIEW_RESULT_FILE_POLL_SECONDS)
                continue
            final_result = _platform_review_result_from_session(sessions, session_id)
            if final_result is not None:
                await on_result(
                    session_id,
                    final_result,
                    source=ACP_REVIEW_RESULT_FILENAME,
                    cancel_prompt_task=True,
                )
                return
            await asyncio.sleep(ACP_REVIEW_RESULT_FILE_POLL_SECONDS)

    async def send_prompt_progress(
        session_id: str,
        *,
        sequence: int,
        platform_events_enabled: bool = False,
        platform_response_id: str | None = None,
    ) -> bool:
        if session_id not in sessions:
            return False
        return await send_prompt_update(
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
            platform_events_enabled=platform_events_enabled,
            platform_response_id=platform_response_id,
        )

    async def send_prompt_wait_message(
        session_id: str,
        *,
        sequence: int,
        platform_events_enabled: bool = False,
        platform_response_id: str | None = None,
    ) -> bool:
        if session_id not in sessions:
            return False
        text = ACP_PROMPT_WAITING_TEXT if sequence <= 0 else ACP_PROMPT_STILL_WAITING_TEXT
        return await send_prompt_update(
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
            platform_events_enabled=platform_events_enabled,
            platform_response_id=platform_response_id,
        )

    async def send_prompt_error_message(
        session_id: str,
        message: str,
        *,
        platform_events_enabled: bool = False,
        platform_response_id: str | None = None,
    ) -> bool:
        if session_id not in sessions:
            return False
        return await send_prompt_update(
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
            platform_events_enabled=platform_events_enabled,
            platform_response_id=platform_response_id,
        )

    async def send_prompt_keepalive_completed(
        session_id: str,
        *,
        platform_events_enabled: bool = False,
        platform_response_id: str | None = None,
    ) -> bool:
        return await send_prompt_update(
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
            platform_events_enabled=platform_events_enabled,
            platform_response_id=platform_response_id,
        )

    async def send_platform_keepalive_chunk(
        session_id: str,
        *,
        sequence: int,
        response_id: str | None = None,
    ) -> bool:
        text = ACP_PROMPT_WAITING_TEXT if sequence == 1 else ACP_PROMPT_STILL_WAITING_TEXT
        async with send_lock:
            sent = await _send_platform_event(
                websocket,
                session_id,
                "session.response_chunk",
                {"chunk": {"content": text}},
                response_id=response_id,
                connection_id=connection_id,
            )
        if not sent:
            mark_connection_closed("platform_keepalive_send_failed")
        return sent

    async def send_platform_response_start(
        session_id: str,
        request_id: JsonRpcId,
        *,
        response_id: str | None = None,
    ) -> bool:
        async with send_lock:
            sent = await _send_platform_event(
                websocket,
                session_id,
                "session.response_start",
                {"requestId": request_id},
                response_id=response_id,
                connection_id=connection_id,
            )
        if not sent:
            mark_connection_closed("platform_start_send_failed")
        return sent

    async def send_platform_result_chunks(
        session_id: str,
        result: dict[str, Any],
        *,
        response_id: str | None = None,
    ) -> bool:
        for chunk in _platform_chunks_from_result(result):
            async with send_lock:
                sent = await _send_platform_event(
                    websocket,
                    session_id,
                    "session.response_chunk",
                    {"chunk": {"content": chunk}},
                    response_id=response_id,
                    connection_id=connection_id,
                )
            if not sent:
                mark_connection_closed("platform_chunk_send_failed")
                return False
        return True

    async def send_platform_response_end(
        session_id: str,
        request_id: JsonRpcId,
        result: dict[str, Any],
        *,
        response_id: str | None = None,
    ) -> bool:
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
            response_id,
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
            sent = await _send_platform_event(
                websocket,
                session_id,
                "session.response_end",
                end_params,
                response_id=response_id,
                connection_id=connection_id,
            )
        if not sent:
            mark_connection_closed("platform_end_send_failed")
        return sent

    async def send_platform_error_end(
        session_id: str,
        request_id: JsonRpcId,
        message: str,
        *,
        response_id: str | None = None,
    ) -> bool:
        logger.warning(
            "acp platform response error session_id=%s request_id=%s response_id=%s error=%s",
            session_id,
            request_id,
            response_id,
            message,
        )
        async with send_lock:
            chunk_sent = await _send_platform_event(
                websocket,
                session_id,
                "session.response_chunk",
                {"chunk": {"content": f"请求处理失败：{message}"}},
                response_id=response_id,
                connection_id=connection_id,
            )
            end_sent = await _send_platform_event(
                websocket,
                session_id,
                "session.response_end",
                {
                    "requestId": request_id,
                    "stopReason": "error",
                    "error": message,
                },
                response_id=response_id,
                connection_id=connection_id,
            )
        sent = chunk_sent and end_sent
        if not sent:
            mark_connection_closed("platform_error_send_failed")
        return sent

    try:
        while True:
            message = await websocket.receive_json()
            if not isinstance(message, dict):
                await send_error(None, -32600, "JSON-RPC message must be an object")
                continue

            request_id = _request_id(message)
            method = message.get("method")
            params = _params(message.get("params"))
            logger.debug(
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
                if task_session_id is None and len(sessions) == 1:
                    task_session_id = next(iter(sessions))
                if task_session_id is not None and _session_id(params) is None:
                    params = {**params, "sessionId": task_session_id}
                incoming_platform_response_id = _string_or_empty(params.get("_platformResponseId"))
                platform_prompt = normalized_method == "agent.command" and task_session_id is not None
                task_request_id = request_id
                if platform_prompt and task_session_id is not None:
                    params = {
                        **params,
                        "_platformResponseId": incoming_platform_response_id
                        or _platform_response_id(task_session_id, request_id),
                    }
                task = asyncio.create_task(
                    run_prompt(
                        task_request_id,
                        params,
                        task_session_id,
                        platform_request_id=request_id if platform_prompt else None,
                        platform_keepalive_chunks=normalized_method != "agent.command",
                        platform_events_enabled=platform_prompt,
                    )
                )
                if task_session_id is not None:
                    task_key = _prompt_task_key(task_session_id, task_request_id, next(prompt_task_ids))
                    prompt_tasks[task_key] = task
                    prompt_task_sessions[task_key] = task_session_id
                continue

            if normalized_method == "session.init":
                result = await active_dispatcher.dispatch(sessions, method, params, send_update)
                if request_id is not None:
                    await send_result(request_id, result)
                continue

            if normalized_method in {"cancel", "session/cancel"}:
                session_id = _session_id(params)
                if session_id is not None:
                    for task_key, task_session_id in list(prompt_task_sessions.items()):
                        if task_session_id != session_id:
                            continue
                        cancel_task = prompt_tasks.pop(task_key, None)
                        prompt_task_sessions.pop(task_key, None)
                        if cancel_task is not None and not cancel_task.done():
                            cancel_task.cancel()

            if normalized_method in {"close_session", "session/close"}:
                session_id = _session_id(params)
                if session_id is not None:
                    for task_key, task_session_id in list(prompt_task_sessions.items()):
                        if task_session_id != session_id:
                            continue
                        close_task = prompt_tasks.pop(task_key, None)
                        prompt_task_sessions.pop(task_key, None)
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
        logger.debug(
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
        logger.debug(
            "acp ws cleanup complete connection_id=%s client=%s duration_ms=%s closed_sessions=%s "
            "cancelled_prompt_tasks=%s keepalive_sessions=%s",
            connection_id,
            client_label,
            int((time.monotonic() - connected_at) * 1000),
            active_sessions,
            pending_prompt_count,
            sorted(keepalive_sessions),
        )


async def _send_session_update(
    websocket: WebSocket,
    session_id: str,
    update: dict[str, Any],
    *,
    connection_id: str | None = None,
) -> bool:
    update_text = _content_text(update)
    logger.debug(
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
    return await _safe_send_json(
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
) -> bool:
    event_type = _platform_type_for_update(update)
    if event_type is None:
        return True
    return await _send_platform_event(
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
) -> bool:
    event_params = params or {}
    resolved_response_id = response_id or _platform_response_id(session_id, None)
    headers = {
        "responseId": resolved_response_id,
        "messageId": f"{resolved_response_id}:{event_type}",
        "origin": "assistant",
        "provider": "jetlinks-agent-runtime-v2",
    }
    logger.debug(
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
    _dump_platform_event(
        session_id,
        event_type,
        event_params,
        response_id=resolved_response_id,
        connection_id=connection_id,
        delivery="session.event.attempt",
    )
    if not await _safe_send_json(
        websocket,
        session_event_payload,
        connection_id=connection_id,
        method="session.event",
        session_id=session_id,
    ):
        _dump_platform_event(
            session_id,
            event_type,
            event_params,
            response_id=resolved_response_id,
            connection_id=connection_id,
            delivery="session.event.skipped",
            error="send_failed",
        )
        return False
    _dump_platform_event(
        session_id,
        event_type,
        event_params,
        response_id=resolved_response_id,
        connection_id=connection_id,
        delivery="session.event.sent",
    )
    _log_ws_trace("send", agent_message_payload, connection_id=connection_id, method="agent.message", session_id=session_id)
    _dump_platform_event(
        session_id,
        event_type,
        event_params,
        response_id=resolved_response_id,
        connection_id=connection_id,
        delivery="agent.message.attempt",
    )
    if not await _safe_send_json(
        websocket,
        agent_message_payload,
        connection_id=connection_id,
        method="agent.message",
        session_id=session_id,
    ):
        _dump_platform_event(
            session_id,
            event_type,
            event_params,
            response_id=resolved_response_id,
            connection_id=connection_id,
            delivery="agent.message.skipped",
            error="send_failed",
        )
        return False
    _dump_platform_event(
        session_id,
        event_type,
        event_params,
        response_id=resolved_response_id,
        connection_id=connection_id,
        delivery="agent.message.sent",
    )
    return True


async def _send_result(
    websocket: WebSocket,
    request_id: JsonRpcId,
    result: dict[str, Any],
    *,
    connection_id: str | None = None,
) -> bool:
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
    return await _safe_send_json(
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
) -> bool:
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
    return await _safe_send_json(
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
    event_type: str | None = None
    event_params: dict[str, Any] | None = None
    if method in {"session.event", "agent.message"}:
        params = _params(payload.get("params"))
        raw_type = params.get("type")
        if isinstance(raw_type, str):
            event_type = raw_type
        raw_event_params = params.get("params")
        if isinstance(raw_event_params, dict):
            event_params = raw_event_params
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, ClosedResourceError, BrokenResourceError):
        logger.warning(
            "acp ws send skipped connection_id=%s method=%s request_id=%s session_id=%s reason=websocket_disconnected "
            "payload_preview=%s",
            connection_id,
            method,
            request_id,
            session_id,
            diagnostic_json(payload, max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
        )
        if session_id and event_type and event_params is not None:
            _dump_platform_event(
                session_id,
                event_type,
                event_params,
                response_id=None,
                connection_id=connection_id,
                delivery=f"{method}.websocket_disconnected",
                error="websocket_disconnected",
            )
        return False
    except RuntimeError as exc:
        message = str(exc)
        if not _is_websocket_closed_error(message):
            raise
        logger.warning(
            "acp ws send skipped connection_id=%s method=%s request_id=%s session_id=%s reason=websocket_closed "
            "error=%s payload_preview=%s",
            connection_id,
            method,
            request_id,
            session_id,
            message,
            diagnostic_json(payload, max_chars=ACP_WS_RESULT_PREVIEW_MAX_CHARS),
        )
        if session_id and event_type and event_params is not None:
            _dump_platform_event(
                session_id,
                event_type,
                event_params,
                response_id=None,
                connection_id=connection_id,
                delivery=f"{method}.websocket_closed",
                error=message,
            )
        return False


def _is_websocket_closed_error(message: str) -> bool:
    return (
        "websocket.close" in message
        or "response already completed" in message
        or 'Cannot call "send" once a close message has been sent.' in message
        or "ClientDisconnected" in message
    )


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


def _result_run_id(result: dict[str, Any]) -> str | None:
    result_payload = result.get("result")
    if not isinstance(result_payload, dict):
        return None
    metadata = result_payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    run_id = metadata.get("run_id")
    return run_id if isinstance(run_id, str) else None


def _result_status(result: dict[str, Any]) -> str | None:
    result_payload = result.get("result")
    if not isinstance(result_payload, dict):
        return None
    status = result_payload.get("status")
    return status if isinstance(status, str) else None


def _result_reply_chars(result: dict[str, Any]) -> int:
    result_payload = result.get("result")
    if not isinstance(result_payload, dict):
        return 0
    reply = result_payload.get("reply")
    return len(reply) if isinstance(reply, str) else 0


def _result_content_items(result: dict[str, Any]) -> int:
    content = result.get("content")
    return len(content) if isinstance(content, list) else 0


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


def _prompt_task_key(session_id: str, request_id: JsonRpcId, sequence: int) -> str:
    if request_id is None:
        return f"{session_id}:notification:{sequence}"
    return f"{session_id}:{request_id}:{sequence}"


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
    session_update = update.get("sessionUpdate")
    if session_update not in {"agent_message_chunk", "agent_thought_chunk"}:
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


def _platform_final_result_from_update(
    update: dict[str, Any],
    session: AcpWebSocketSession | None = None,
) -> dict[str, Any] | None:
    meta = update.get("_meta")
    if not isinstance(meta, dict):
        return None
    raw_event = meta.get("jetlinksRuntimeEvent")
    if not isinstance(raw_event, dict) or raw_event.get("type") != "run.completed":
        return None
    data = raw_event.get("data")
    if not isinstance(data, dict):
        return None
    raw_result = data.get("result")
    if not isinstance(raw_result, dict):
        return None
    result_payload = _normalize_structured_json_result_payload(raw_result, session)
    content = result_payload.get("content")
    if not isinstance(content, list):
        reply = result_payload.get("reply")
        content = [{"type": "text", "text": reply}] if isinstance(reply, str) and reply else []
    return {
        "stopReason": "end_turn",
        "threadId": result_payload.get("thread_id") or result_payload.get("threadId"),
        "agentName": result_payload.get("agent"),
        "content": content,
        "result": result_payload,
    }


def _normalize_structured_json_result_payload(
    result_payload: dict[str, Any],
    session: AcpWebSocketSession | None,
) -> dict[str, Any]:
    if not _should_normalize_structured_json_payload(session):
        return result_payload
    reply = result_payload.get("reply")
    if not isinstance(reply, str):
        return result_payload
    normalized_reply = _validated_json_text_from_reply(reply)
    if normalized_reply is None or normalized_reply == reply:
        return result_payload
    normalized = dict(result_payload)
    normalized["reply"] = normalized_reply
    normalized["content"] = _replace_primary_text_content(result_payload.get("content"), reply, normalized_reply)
    return normalized


def _should_normalize_structured_json_payload(session: AcpWebSocketSession | None) -> bool:
    if session is None:
        return False
    runtime_options = session.runtime_options if isinstance(session.runtime_options, dict) else {}
    response_format = runtime_options.get("response_format") or runtime_options.get("responseFormat")
    app_template_name = (
        session.app_template_name
        or runtime_options.get("app_template_name")
        or runtime_options.get("appTemplateName")
    )
    return response_format == "json" or app_template_name == "ParkingAbnormalEventMonitoring"


def _validated_json_text_from_reply(reply: str) -> str | None:
    candidate = _extract_fenced_json(reply.strip())
    if candidate is None:
        candidate = reply.strip()
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_fenced_json(text: str) -> str | None:
    if not text.startswith("```"):
        return None
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0].strip().startswith("```"):
        return None
    closing_index: int | None = None
    for index in range(len(lines) - 1, 0, -1):
        if lines[index].strip() == "```":
            closing_index = index
            break
    if closing_index is None:
        return None
    return "\n".join(lines[1:closing_index]).strip()


def _replace_primary_text_content(raw_content: Any, old_text: str, new_text: str) -> list[dict[str, Any]]:
    content = raw_content if isinstance(raw_content, list) else []
    updated: list[dict[str, Any]] = []
    replaced = False
    for block in content:
        if not isinstance(block, dict):
            continue
        if not replaced and block.get("type") == "text" and block.get("text") == old_text:
            next_block = dict(block)
            next_block["text"] = new_text
            updated.append(next_block)
            replaced = True
            continue
        updated.append(block)
    if not replaced:
        return [{"type": "text", "text": new_text}, *updated]
    return updated


def _should_watch_review_result_file(
    sessions: dict[str, AcpWebSocketSession],
    session_id: str | None,
) -> bool:
    session = sessions.get(session_id or "")
    if session is None:
        return False
    runtime_options = session.runtime_options if isinstance(session.runtime_options, dict) else {}
    return runtime_options.get("workflow") == "parking_abnormal_review"


def _platform_review_result_from_session(
    sessions: dict[str, AcpWebSocketSession],
    session_id: str | None,
) -> dict[str, Any] | None:
    session = sessions.get(session_id or "")
    if session is None:
        return None
    result_path = ACP_WS_THREAD_ROOT / session.thread_id / "outputs" / ACP_REVIEW_RESULT_FILENAME
    if not result_path.exists() or not result_path.is_file():
        return None
    try:
        reply = result_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not reply:
        return None
    try:
        json.loads(reply)
    except json.JSONDecodeError:
        return None
    result_payload = {
        "agent": session.agent_name,
        "thread_id": session.thread_id,
        "status": "completed",
        "reply": reply,
        "content": [{"type": "text", "text": reply}],
        "artifacts": [],
        "metadata": {
            "workflow": "parking_abnormal_review",
            "result_source": f"outputs/{ACP_REVIEW_RESULT_FILENAME}",
        },
    }
    return {
        "stopReason": "end_turn",
        "threadId": session.thread_id,
        "agentName": session.agent_name,
        "content": result_payload["content"],
        "result": result_payload,
    }


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


def _dump_platform_event(
    session_id: str,
    event_type: str,
    params: dict[str, Any],
    *,
    response_id: str | None,
    connection_id: str | None,
    delivery: str,
    error: str | None = None,
) -> None:
    if not ACP_WS_THREAD_EVENT_DUMP_ENABLED or not session_id:
        return
    try:
        thread_root = Path(__file__).resolve().parents[3] / ".runtime" / "threads" / session_id / "outputs"
        thread_root.mkdir(parents=True, exist_ok=True)
        event_path = thread_root / "acp-platform-events.jsonl"
        record = {
            "ts": time.time(),
            "sessionId": session_id,
            "connectionId": connection_id,
            "responseId": response_id,
            "type": event_type,
            "delivery": delivery,
            "params": params,
        }
        if error:
            record["error"] = error
        with event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("acp thread event dump failed session_id=%s type=%s", session_id, event_type)
