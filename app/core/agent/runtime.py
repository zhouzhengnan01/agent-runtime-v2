from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx

from app.core.artifacts import ArtifactStore
from app.core.artifacts.store import ThreadPaths
from app.core.artifacts.preview import guess_mime_type
from app.core.agent.execution_context import ExecutionContext
from app.core.agent.execution_context import build_execution_context
from app.core.agent.session import SessionConversationStore
from app.core.agent.session_kernel import AgentSessionManager
from app.core.apps import AppTemplateRegistry
from app.core.apps.models import AppModelOption
from app.core.apps.runtime_options import merge_runtime_options_with_template
from app.core.agent.tool_loop import ToolCallingAgentLoop
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.config.secrets import SecretCodec
from app.core.events import EventRecorder, RunEventStore
from app.core.llm import OpenAICompatibleClient
from app.core.memory import MarkdownMemoryStore, MemoryStore
from app.core.agent.input_required import required_inputs_for_request, required_inputs_for_result
from app.core.routing.workflow_router import WorkflowRouter
from app.core.skills import SkillDefinition, SkillRegistry, SkillRunner
from app.core.skills.aliases import expand_skill_aliases
from app.core.tools import ToolInvocationService
from app.core.workflow import WorkflowRegistry
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, Role, RuntimeOptions


logger = logging.getLogger("uvicorn.error")
REMOTE_ATTACHMENT_MAX_BYTES = 50 * 1024 * 1024
REMOTE_ATTACHMENT_TIMEOUT_SECONDS = 30.0
WORKFLOW_THREAD_WORKERS = max(1, int(os.getenv("JETLINKS_WORKFLOW_THREAD_WORKERS", "16") or "16"))
PARKING_REVIEW_THREAD_WORKERS = max(
    1,
    int(os.getenv("JETLINKS_PARKING_REVIEW_THREAD_WORKERS", "4") or "4"),
)
WORKFLOW_QUEUE_WARN_SECONDS = max(0.0, float(os.getenv("JETLINKS_WORKFLOW_QUEUE_WARN_SECONDS", "1") or "1"))
PARKING_REVIEW_MAX_QUEUE_BACKLOG = max(0, int(os.getenv("PARKING_REVIEW_MAX_QUEUE_BACKLOG", "16") or "16"))
PARKING_REVIEW_DEDUP_ENABLED = os.getenv("PARKING_REVIEW_DEDUP_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PARKING_REVIEW_DEDUP_TTL_SECONDS = max(
    0.0,
    float(os.getenv("PARKING_REVIEW_DEDUP_TTL_SECONDS", "120") or "120"),
)
_WORKFLOW_QUEUE_LOCK = threading.Lock()
_WORKFLOW_QUEUE_PENDING = 0
_PARKING_REVIEW_QUEUE_PENDING = 0
FIXED_REPLY_ENV = "JETLINKS_AGENT_FIXED_REPLY"
FIXED_REPLY_ENABLED_ENV = "JETLINKS_AGENT_FIXED_REPLY_ENABLED"
FIXED_REPLY_FOREVER_ENV = "JETLINKS_AGENT_FIXED_REPLY_FOREVER"
FIXED_REPLY_INTERVAL_ENV = "JETLINKS_AGENT_FIXED_REPLY_INTERVAL_SECONDS"
DEFAULT_FIXED_REPLY = (
    '[{"reviewSourceId":"debug-review-source",'
    '"reviewEventId":"debug-review-event",'
    '"hit":0,'
    '"result":"联调固定响应：agent-v2 已收到请求并返回固定内容。"}]'
)


@dataclass
class RuntimeQueueSnapshot:
    name: str
    pending: int
    workers: int
    max_backlog: int | None
    overloaded: bool
    rejected_total: int
    last_overloaded_at: float | None
    last_rejected_at: float | None
    last_reject_reason: str | None


class RuntimeQueueState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_by_queue: dict[str, int] = {}
        self._rejected_total_by_queue: dict[str, int] = {}
        self._last_overloaded_at_by_queue: dict[str, float] = {}
        self._last_rejected_at_by_queue: dict[str, float] = {}
        self._last_reject_reason_by_queue: dict[str, str] = {}

    def increment(self, queue_name: str) -> int:
        with self._lock:
            backlog = self._pending_by_queue.get(queue_name, 0)
            self._pending_by_queue[queue_name] = backlog + 1
            return backlog

    def backlog(self, queue_name: str) -> int:
        with self._lock:
            return self._pending_by_queue.get(queue_name, 0)

    def decrement(self, queue_name: str) -> int:
        with self._lock:
            backlog = max(0, self._pending_by_queue.get(queue_name, 0) - 1)
            self._pending_by_queue[queue_name] = backlog
            return backlog

    def record_rejection(self, queue_name: str, *, reason: str) -> None:
        now = time.time()
        with self._lock:
            self._rejected_total_by_queue[queue_name] = self._rejected_total_by_queue.get(queue_name, 0) + 1
            self._last_rejected_at_by_queue[queue_name] = now
            self._last_reject_reason_by_queue[queue_name] = reason

    def rejected_total(self, queue_name: str) -> int:
        with self._lock:
            return self._rejected_total_by_queue.get(queue_name, 0)

    def last_overloaded_at(self, queue_name: str) -> float | None:
        with self._lock:
            return self._last_overloaded_at_by_queue.get(queue_name)

    def last_rejected_at(self, queue_name: str) -> float | None:
        with self._lock:
            return self._last_rejected_at_by_queue.get(queue_name)

    def last_reject_reason(self, queue_name: str) -> str | None:
        with self._lock:
            return self._last_reject_reason_by_queue.get(queue_name)

    def snapshot(
        self,
        *,
        queue_name: str,
        workers: int,
        max_backlog: int | None = None,
    ) -> RuntimeQueueSnapshot:
        with self._lock:
            pending = self._pending_by_queue.get(queue_name, 0)
            overloaded = max_backlog is not None and max_backlog > 0 and pending >= max_backlog
            if overloaded:
                self._last_overloaded_at_by_queue[queue_name] = time.time()
        return RuntimeQueueSnapshot(
            name=queue_name,
            pending=pending,
            workers=workers,
            max_backlog=max_backlog,
            overloaded=overloaded,
            rejected_total=self.rejected_total(queue_name),
            last_overloaded_at=self.last_overloaded_at(queue_name),
            last_rejected_at=self.last_rejected_at(queue_name),
            last_reject_reason=self.last_reject_reason(queue_name),
        )


RUNTIME_QUEUE_STATE = RuntimeQueueState()


def _extract_review_source_id_for_overload(text: str) -> str:
    payload = _parse_json_object_for_overload(text)
    if isinstance(payload, dict):
        for key in ("reviewSourceId", "review_source_id", "sourceId", "source_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        headers = payload.get("headers")
        if isinstance(headers, dict):
            for key in ("reviewSourceId", "review_source_id"):
                value = headers.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    for pattern in (
        r"reviewSourceId为\[([^\]]+)\]",
        r"复判事件来源reviewSourceId为\[([^\]]+)\]",
        r"复判事件来源id为\[([^\]]+)\]",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _parse_json_object_for_overload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = json.loads(stripped)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _workflow_queue_increment() -> int:
    global _WORKFLOW_QUEUE_PENDING
    backlog = RUNTIME_QUEUE_STATE.increment("default_workflow")
    with _WORKFLOW_QUEUE_LOCK:
        _WORKFLOW_QUEUE_PENDING = RUNTIME_QUEUE_STATE.backlog("default_workflow")
    return backlog


def _workflow_queue_backlog() -> int:
    backlog = RUNTIME_QUEUE_STATE.backlog("default_workflow")
    with _WORKFLOW_QUEUE_LOCK:
        _WORKFLOW_QUEUE_PENDING = backlog
    return backlog


def _workflow_queue_decrement() -> int:
    global _WORKFLOW_QUEUE_PENDING
    backlog = RUNTIME_QUEUE_STATE.decrement("default_workflow")
    with _WORKFLOW_QUEUE_LOCK:
        _WORKFLOW_QUEUE_PENDING = backlog
    return backlog


def _parking_review_queue_increment() -> int:
    global _PARKING_REVIEW_QUEUE_PENDING
    backlog = RUNTIME_QUEUE_STATE.increment("parking_review")
    with _WORKFLOW_QUEUE_LOCK:
        _PARKING_REVIEW_QUEUE_PENDING = RUNTIME_QUEUE_STATE.backlog("parking_review")
    return backlog


def _parking_review_queue_backlog() -> int:
    backlog = RUNTIME_QUEUE_STATE.backlog("parking_review")
    with _WORKFLOW_QUEUE_LOCK:
        _PARKING_REVIEW_QUEUE_PENDING = backlog
    return backlog


def _parking_review_queue_decrement() -> int:
    global _PARKING_REVIEW_QUEUE_PENDING
    backlog = RUNTIME_QUEUE_STATE.decrement("parking_review")
    with _WORKFLOW_QUEUE_LOCK:
        _PARKING_REVIEW_QUEUE_PENDING = backlog
    return backlog


@dataclass(frozen=True)
class _DirectJsonArtifactIntent:
    filename: str
    content: str
    reason: str


@dataclass(frozen=True)
class _WorkflowDedupEntry:
    workflow_name: str
    dedup_key: str
    reply: str
    status: str
    metadata: dict[str, Any]
    cached_at: float


@dataclass(frozen=True)
class _WorkflowDedupDecision:
    workflow_name: str
    dedup_key: str
    source: str
    owner_future: Future[_WorkflowDedupEntry] | None = None
    wait_future: Future[_WorkflowDedupEntry] | None = None
    cache_entry: _WorkflowDedupEntry | None = None


class AgentRuntime:
    """Stateless runtime shared by HTTP and CLI entrypoints.

    High-level flow:
    1. Normalize the request and derive effective runtime options.
    2. Short-circuit if required inputs are missing.
    3. Dispatch either to an explicit workflow or to the default agent loop.
    4. Persist conversation state and run events back into the thread workspace.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        workflow_router: object | None = None,
        workflow_registry: WorkflowRegistry | None = None,
        memory_store: MemoryStore | None = None,
        markdown_memory_store: MarkdownMemoryStore | None = None,
        run_event_store: RunEventStore | None = None,
        session_store: SessionConversationStore | None = None,
        session_manager: AgentSessionManager | None = None,
        model_config: ModelConfig | dict[str, object] | None = None,
        app_template_registry: AppTemplateRegistry | None = None,
        skills: list[str] | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.memory_store = memory_store or MemoryStore()
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()
        self.run_event_store = run_event_store or RunEventStore()
        self.session_store = session_store or SessionConversationStore()
        self.session_manager = session_manager or AgentSessionManager()
        self.workflow_registry = workflow_registry or WorkflowRegistry.builtin(self.artifact_store)
        self.workflow_router = workflow_router
        self.workflow_executor = ThreadPoolExecutor(
            max_workers=WORKFLOW_THREAD_WORKERS,
            thread_name_prefix="jetlinks-workflow",
        )
        self.parking_review_executor = ThreadPoolExecutor(
            max_workers=PARKING_REVIEW_THREAD_WORKERS,
            thread_name_prefix="jetlinks-parking-review",
        )
        self._workflow_dedup_lock = threading.Lock()
        self._workflow_inflight: dict[tuple[str, str], Future[_WorkflowDedupEntry]] = {}
        self._workflow_cache: dict[tuple[str, str], _WorkflowDedupEntry] = {}
        self.workflow_executors: dict[str, str] = {
            "default_workflow": "workflow_executor",
            "parking_review": "parking_review_executor",
        }
        self.model_config, self._model_config_fields = self._normalize_model_config(model_config)
        self.app_template_registry = app_template_registry or AppTemplateRegistry()
        self.secret_codec = SecretCodec(self.app_template_registry.root_dir)
        self.skills = self._normalize_skills(skills)
        self.agent_loop = ToolCallingAgentLoop(
            ToolInvocationService(
                root_dir=self.app_template_registry.root_dir,
                artifact_store=self.artifact_store,
                memory_store=self.memory_store,
                markdown_memory_store=self.markdown_memory_store,
            )
        )

    async def run(self, agent_config: AgentConfig, request: ChatRequest) -> AgentRunResult:
        result, _events = await self.run_with_events(agent_config, request)
        return result

    async def run_with_events(
        self, agent_config: AgentConfig, request: ChatRequest
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        execution = self._execution_context(agent_config, request)
        fixed_reply = self._fixed_reply_text()
        if fixed_reply is not None:
            return self._fixed_reply_result(execution, fixed_reply)
        if execution.input_required:
            # Preflight exit: do not enter workflows or tool calling when the
            # runtime already knows the request cannot proceed yet.
            recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
            events = [
                recorder.emit(
                    "run.started",
                    {
                        "run_id": recorder.run_id,
                        "workflow": execution.workflow_name or "agent_loop",
                        "execution_mode": execution.workflow_name or "agent_loop",
                        "stateless": execution.agent_config.runtime.stateless,
                    },
                )
            ]
            self.session_manager.begin_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                agent=execution.agent_config.name,
                workflow=execution.workflow_name or "agent_loop",
            )
            result = self._input_required_result(
                execution.agent_config,
                execution.paths.thread_id,
                execution.input_required,
                recorder.run_id,
                execution.workflow_name,
                execution.request.runtime_options,
            )
            events.append(recorder.emit("agent.message", {"text": result.reply}))
            events.append(recorder.emit("run.completed", {"result": result.model_dump()}))
            self.session_store.save(
                execution.paths,
                self._conversation_with_result(execution.conversation, result),
                run_id=recorder.run_id,
            )
            self._persist_events(execution.agent_config, execution.request, execution.paths.thread_id, events, result)
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="completed",
            )
            return result, events
        if execution.workflow_name is not None:
            overload_result = self._parking_review_queue_overload_result(execution)
            if overload_result is not None:
                return overload_result
            dedup_decision = self._workflow_dedup_decision(execution)
            if dedup_decision is not None and dedup_decision.cache_entry is not None:
                return self._workflow_dedup_materialize_result(execution, dedup_decision)
            if dedup_decision is not None and dedup_decision.wait_future is not None:
                return await self._workflow_dedup_wait_result(execution, dedup_decision)
            # Workflow path: a named workflow owns the full execution instead of
            # the generic tool-calling loop.
            try:
                result, events = await self._run_workflow_in_executor(
                    lambda: self._run_workflow_with_events_sync(execution),
                    execution,
                )
            except BaseException as exc:
                if dedup_decision is not None:
                    self._workflow_dedup_fail(dedup_decision, exc)
                raise
            if dedup_decision is not None:
                self._workflow_dedup_complete(dedup_decision, result)
            return result, events

        # Default path: merge thread history and let the agent loop decide when
        # to answer directly versus when to call tools.
        conversation = self._conversation_with_vision_attachments(execution.conversation, execution.request.attachments, execution.paths)
        conversation = self._mark_conversation_has_attachments(conversation, execution.request.attachments)
        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": "agent_loop",
                "execution_mode": "agent_loop",
                "stateless": execution.agent_config.runtime.stateless,
            },
        )
        self.session_manager.begin_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            agent=execution.agent_config.name,
            workflow="agent_loop",
        )
        try:
            direct_result = self._try_complete_direct_json_artifact_turn(execution, recorder)
            if direct_result is not None:
                return direct_result, recorder.events
            direct_skill_result = self._try_auto_execute_primary_skill_turn(execution, recorder)
            if direct_skill_result is not None:
                return direct_skill_result, recorder.events
            loop_result = await self.agent_loop.run(
                agent_config=execution.agent_config,
                messages=conversation,
                thread_id=execution.paths.thread_id,
                recorder=recorder,
                runtime_options=execution.request.runtime_options,
            )
        except Exception:
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="failed",
            )
            raise
        self._enrich_required_inputs(loop_result.result, execution.request)
        self._replace_final_result_event(recorder.events, loop_result.result)
        self.session_store.save(execution.paths, self._strip_internal_message_fields(loop_result.messages), run_id=recorder.run_id)
        self._persist_events(
            execution.agent_config,
            execution.request,
            execution.paths.thread_id,
            recorder.events,
            loop_result.result,
        )
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="completed" if loop_result.result.status == "completed" else "failed",
        )
        return loop_result.result, recorder.events

    async def stream(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[str]:
        try:
            async for event in self.iter_events(agent_config, request):
                yield self._event(event)
        except Exception as exc:
            yield self._event(ChatEvent(type="run.failed", data={"agent": agent_config.name, "error": str(exc)}))

    async def iter_events(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        execution = self._execution_context(agent_config, request)
        fixed_reply = self._fixed_reply_text()
        if fixed_reply is not None:
            if self._fixed_reply_forever_enabled():
                async for event in self._stream_fixed_reply_forever_events(execution, fixed_reply):
                    yield event
                return
            _result, events = self._fixed_reply_result(execution, fixed_reply)
            for event in events:
                yield event
            return
        if execution.input_required:
            async for event in self._stream_input_required_events(execution):
                yield event
            return
        if execution.workflow_name is not None:
            if self.workflow_registry.get(execution.workflow_name) is None:
                raise ValueError(f"Workflow is not registered: {execution.workflow_name}")
            overload_result = self._parking_review_queue_overload_result(execution)
            if overload_result is not None:
                _result, events = overload_result
                for event in events:
                    yield event
                return
            dedup_decision = self._workflow_dedup_decision(execution)
            if dedup_decision is not None and dedup_decision.cache_entry is not None:
                _result, events = self._workflow_dedup_materialize_result(execution, dedup_decision)
                for event in events:
                    yield event
                return
            if dedup_decision is not None and dedup_decision.wait_future is not None:
                _result, events = await self._workflow_dedup_wait_result(execution, dedup_decision)
                for event in events:
                    yield event
                return
            try:
                async for event in self._stream_workflow_events(execution):
                    yield event
            except BaseException as exc:
                if dedup_decision is not None:
                    self._workflow_dedup_fail(dedup_decision, exc)
                raise
            else:
                result = getattr(execution, "_workflow_final_result", None)
                if dedup_decision is not None and isinstance(result, AgentRunResult):
                    self._workflow_dedup_complete(dedup_decision, result)
                return

        async for event in self._stream_agent_loop_events(execution):
            yield event

    async def _stream_workflow_events(self, execution: ExecutionContext) -> AsyncIterator[ChatEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatEvent | BaseException | None] = asyncio.Queue()
        captured_events: list[ChatEvent] = []
        final_result: AgentRunResult | None = None

        def on_event(event: ChatEvent) -> None:
            if event.type == "run.started":
                queue_metrics = getattr(execution, "_workflow_queue_metrics", None)
                if isinstance(queue_metrics, dict):
                    event.data.update(queue_metrics)
            if event.type in {"run.completed", "run.failed"}:
                raw_result = event.data.get("result")
                if isinstance(raw_result, dict):
                    event_result = AgentRunResult.model_validate(raw_result)
                    self._enrich_required_inputs(event_result, execution.request)
                    event.data["result"] = event_result.model_dump()
            captured_events.append(event)
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_workflow() -> None:
            nonlocal final_result, captured_events
            workflow = self.workflow_registry.get(execution.workflow_name or "")
            if workflow is None:
                raise ValueError(f"Workflow is not registered: {execution.workflow_name}")
            try:
                final_result, returned_events = workflow.run_with_events(
                    agent_config=execution.agent_config,
                    messages=self._workflow_messages(execution.conversation),
                    attachments=execution.request.attachments,
                    thread_id=execution.paths.thread_id,
                    on_event=on_event,
                    workflow_name=execution.workflow_name or "agent_loop",
                    runtime_options=execution.request.runtime_options,
                )
                if not captured_events:
                    captured_events = list(returned_events)
                    self._apply_workflow_queue_metrics(captured_events, execution)
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(self._run_workflow_in_executor(run_workflow, execution))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    yield ChatEvent(type="run.failed", data={"agent": execution.agent_config.name, "error": str(item)})
                    break
                yield item
        finally:
            await task
            if captured_events:
                thread_id = (
                    final_result.thread_id
                    if final_result is not None
                    else str(captured_events[0].data.get("thread_id") or execution.request.runtime_options.thread_id or "")
                )
                if final_result is not None:
                    self._enrich_required_inputs(final_result, execution.request)
                    self._replace_final_result_event(captured_events, final_result)
                    self.session_store.save(
                        execution.paths,
                        self._conversation_with_result(execution.conversation, final_result),
                        run_id=self._run_id(captured_events),
                    )
                self._persist_events(execution.agent_config, execution.request, thread_id, captured_events, final_result)
                if final_result is not None:
                    object.__setattr__(execution, "_workflow_final_result", final_result)

    async def _run_workflow_in_executor(self, func: Callable[[], Any], execution: ExecutionContext) -> Any:
        loop = asyncio.get_running_loop()
        queued_at = time.monotonic()
        queue_name = self._workflow_queue_name(execution)
        queued_backlog = self._workflow_queue_increment(execution)
        queue_workers = self._workflow_queue_workers(execution)
        logger.info(
            "workflow queue submitted workflow=%s thread_id=%s queue=%s workers=%s queue_backlog=%s",
            execution.workflow_name,
            execution.paths.thread_id,
            queue_name,
            queue_workers,
            queued_backlog,
        )

        def wrapped() -> Any:
            wait_seconds = time.monotonic() - queued_at
            start_backlog = self._workflow_queue_decrement(execution)
            object.__setattr__(
                execution,
                "_workflow_queue_metrics",
                {
                    "queue_backlog": start_backlog,
                    "queue_wait_ms": round(wait_seconds * 1000, 1),
                    "queue_workers": queue_workers,
                    "queue_name": queue_name,
                },
            )
            log = logger.warning if wait_seconds >= WORKFLOW_QUEUE_WARN_SECONDS else logger.info
            log(
                "workflow queue acquired workflow=%s thread_id=%s queue=%s wait_ms=%.1f workers=%s queue_backlog=%s",
                execution.workflow_name,
                execution.paths.thread_id,
                queue_name,
                wait_seconds * 1000,
                queue_workers,
                start_backlog,
            )
            started_at = time.monotonic()
            try:
                return func()
            except Exception:
                logger.exception(
                    "workflow execution failed workflow=%s thread_id=%s elapsed_ms=%.1f",
                    execution.workflow_name,
                    execution.paths.thread_id,
                    (time.monotonic() - started_at) * 1000,
                )
                raise
            finally:
                logger.info(
                    "workflow execution finished workflow=%s thread_id=%s queue=%s elapsed_ms=%.1f",
                    execution.workflow_name,
                    execution.paths.thread_id,
                    queue_name,
                    (time.monotonic() - started_at) * 1000,
                )

        return await loop.run_in_executor(self._workflow_executor_for(execution), wrapped)

    def _run_workflow_with_events_sync(self, execution: ExecutionContext) -> tuple[AgentRunResult, list[ChatEvent]]:
        workflow = self.workflow_registry.get(execution.workflow_name or "")
        if workflow is None:
            raise ValueError(f"Workflow is not registered: {execution.workflow_name}")
        result, events = workflow.run_with_events(
            agent_config=execution.agent_config,
            messages=self._workflow_messages(execution.conversation),
            attachments=execution.request.attachments,
            thread_id=execution.paths.thread_id,
            workflow_name=execution.workflow_name,
            runtime_options=execution.request.runtime_options,
        )
        self._apply_workflow_queue_metrics(events, execution)
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=self._run_id(events),
        )
        self._enrich_required_inputs(result, execution.request)
        self._replace_final_result_event(events, result)
        self._persist_events(execution.agent_config, execution.request, result.thread_id, events, result)
        object.__setattr__(execution, "_workflow_final_result", result)
        return result, events

    @staticmethod
    def _apply_workflow_queue_metrics(events: list[ChatEvent], execution: ExecutionContext) -> None:
        queue_metrics = getattr(execution, "_workflow_queue_metrics", None)
        if not isinstance(queue_metrics, dict):
            return
        for event in events:
            if event.type == "run.started":
                event.data.update(queue_metrics)
                return

    def _parking_review_queue_overload_result(
        self,
        execution: ExecutionContext,
    ) -> tuple[AgentRunResult, list[ChatEvent]] | None:
        if execution.workflow_name != "parking_abnormal_review":
            return None
        if PARKING_REVIEW_MAX_QUEUE_BACKLOG <= 0:
            return None
        queue_backlog = _parking_review_queue_backlog()
        if queue_backlog < PARKING_REVIEW_MAX_QUEUE_BACKLOG:
            return None
        self.queue_state().record_rejection("parking_review", reason="parking_review_queue_overloaded")

        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        workflow = execution.workflow_name
        recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": workflow,
                "execution_mode": workflow,
                "queue_backlog": queue_backlog,
                "queue_wait_ms": 0.0,
                "queue_workers": PARKING_REVIEW_THREAD_WORKERS,
                "queue_name": "parking_review",
                "queue_rejected": True,
                "queue_reject_reason": "parking_review_queue_overloaded",
            },
        )
        reply = self._parking_review_overload_reply(execution)
        recorder.emit(
            "review.skill_invocation.failed",
            {
                "skill_name": "",
                "reason": "parking_review_queue_overloaded",
                "queue_backlog": queue_backlog,
                "queue_workers": PARKING_REVIEW_THREAD_WORKERS,
            },
        )
        artifact = self.artifact_store.write_text_artifact(
            execution.paths,
            "parking_abnormal_review_result.json",
            reply,
        )
        recorder.emit(
            "artifact.created",
            {"path": f"outputs/{artifact.name}", "name": artifact.name, "mime_type": artifact.mime_type},
        )
        recorder.emit("agent.message", {"text": reply})
        result = AgentRunResult(
            agent=execution.agent_config.name,
            thread_id=execution.paths.thread_id,
            status="completed",
            reply=reply,
            artifacts=[artifact],
            metadata={
                "workflow": workflow,
                "queue_backlog": queue_backlog,
                "queue_workers": PARKING_REVIEW_THREAD_WORKERS,
                "queue_name": "parking_review",
                "queue_rejected": True,
                "artifact_path": f"outputs/{artifact.name}",
            },
        )
        recorder.emit("run.completed", {"result": result.model_dump()})
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(execution.agent_config, execution.request, execution.paths.thread_id, recorder.events, result)
        logger.warning(
            "parking review queue overloaded thread_id=%s queue_backlog=%s workers=%s max_queue_backlog=%s",
            execution.paths.thread_id,
            queue_backlog,
            PARKING_REVIEW_THREAD_WORKERS,
            PARKING_REVIEW_MAX_QUEUE_BACKLOG,
        )
        return result, recorder.events

    def _workflow_dedup_decision(self, execution: ExecutionContext) -> _WorkflowDedupDecision | None:
        if not PARKING_REVIEW_DEDUP_ENABLED:
            return None
        if execution.workflow_name != "parking_abnormal_review":
            return None
        workflow = self.workflow_registry.get(execution.workflow_name or "")
        if workflow is None:
            return None
        dedup_key = self._workflow_dedup_key(workflow, execution)
        if dedup_key is None:
            return None
        registry_key = (execution.workflow_name, dedup_key)
        now = time.monotonic()
        with self._workflow_dedup_lock:
            cache_entry = self._workflow_cache.get(registry_key)
            if cache_entry is not None and now - cache_entry.cached_at <= PARKING_REVIEW_DEDUP_TTL_SECONDS:
                logger.info(
                    "workflow dedup cache hit workflow=%s dedup_key=%s thread_id=%s ttl_seconds=%.1f",
                    execution.workflow_name,
                    dedup_key,
                    execution.paths.thread_id,
                    PARKING_REVIEW_DEDUP_TTL_SECONDS,
                )
                return _WorkflowDedupDecision(
                    workflow_name=execution.workflow_name,
                    dedup_key=dedup_key,
                    source="cache",
                    cache_entry=cache_entry,
                )
            if cache_entry is not None:
                self._workflow_cache.pop(registry_key, None)
            wait_future = self._workflow_inflight.get(registry_key)
            if wait_future is not None:
                logger.info(
                    "workflow dedup wait workflow=%s dedup_key=%s thread_id=%s",
                    execution.workflow_name,
                    dedup_key,
                    execution.paths.thread_id,
                )
                return _WorkflowDedupDecision(
                    workflow_name=execution.workflow_name,
                    dedup_key=dedup_key,
                    source="inflight",
                    wait_future=wait_future,
                )
            owner_future: Future[_WorkflowDedupEntry] = Future()
            self._workflow_inflight[registry_key] = owner_future
            logger.info(
                "workflow dedup owner workflow=%s dedup_key=%s thread_id=%s",
                execution.workflow_name,
                dedup_key,
                execution.paths.thread_id,
            )
            return _WorkflowDedupDecision(
                workflow_name=execution.workflow_name,
                dedup_key=dedup_key,
                source="owner",
                owner_future=owner_future,
            )

    def _workflow_dedup_key(self, workflow: object, execution: ExecutionContext) -> str | None:
        dedup_key = getattr(workflow, "dedup_key", None)
        if not callable(dedup_key):
            return None
        try:
            value = dedup_key(
                self._workflow_messages(execution.conversation),
                execution.request.attachments,
                execution.request.runtime_options,
            )
        except Exception:
            logger.exception(
                "workflow dedup key failed workflow=%s thread_id=%s",
                execution.workflow_name,
                execution.paths.thread_id,
            )
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    async def _workflow_dedup_wait_result(
        self,
        execution: ExecutionContext,
        decision: _WorkflowDedupDecision,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        if decision.wait_future is None:
            raise RuntimeError("workflow dedup wait requires an inflight future")
        entry = await asyncio.wrap_future(decision.wait_future)
        return self._workflow_dedup_materialize_result(
            execution,
            _WorkflowDedupDecision(
                workflow_name=decision.workflow_name,
                dedup_key=decision.dedup_key,
                source="inflight",
                cache_entry=entry,
            ),
        )

    def _workflow_dedup_materialize_result(
        self,
        execution: ExecutionContext,
        decision: _WorkflowDedupDecision,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        entry = decision.cache_entry
        if entry is None:
            raise RuntimeError("workflow dedup materialization requires a cached entry")
        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        workflow = execution.workflow_name or entry.workflow_name
        recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": workflow,
                "execution_mode": workflow,
                "deduped": True,
                "dedup_source": decision.source,
                "dedup_key": entry.dedup_key,
            },
        )
        artifact = self.artifact_store.write_text_artifact(
            execution.paths,
            "parking_abnormal_review_result.json",
            entry.reply,
        )
        recorder.emit(
            "artifact.created",
            {"path": f"outputs/{artifact.name}", "name": artifact.name, "mime_type": artifact.mime_type},
        )
        recorder.emit("agent.message", {"text": entry.reply})
        metadata = dict(entry.metadata)
        metadata.update(
            {
                "workflow": workflow,
                "artifact_path": f"outputs/{artifact.name}",
                "deduped": True,
                "dedup_source": decision.source,
                "dedup_key": entry.dedup_key,
            }
        )
        result = AgentRunResult(
            agent=execution.agent_config.name,
            thread_id=execution.paths.thread_id,
            status=cast(Any, entry.status),
            reply=entry.reply,
            artifacts=[artifact],
            metadata=metadata,
        )
        recorder.emit("run.completed", {"result": result.model_dump()})
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(execution.agent_config, execution.request, execution.paths.thread_id, recorder.events, result)
        logger.info(
            "workflow dedup materialized workflow=%s dedup_key=%s thread_id=%s source=%s",
            workflow,
            entry.dedup_key,
            execution.paths.thread_id,
            decision.source,
        )
        return result, recorder.events

    def _workflow_dedup_complete(
        self,
        decision: _WorkflowDedupDecision,
        result: AgentRunResult,
    ) -> None:
        if decision.owner_future is None:
            return
        entry = _WorkflowDedupEntry(
            workflow_name=decision.workflow_name,
            dedup_key=decision.dedup_key,
            reply=result.reply,
            status=result.status,
            metadata={key: value for key, value in result.metadata.items() if key != "run_id"},
            cached_at=time.monotonic(),
        )
        registry_key = (decision.workflow_name, decision.dedup_key)
        with self._workflow_dedup_lock:
            if not decision.owner_future.done():
                decision.owner_future.set_result(entry)
            if PARKING_REVIEW_DEDUP_TTL_SECONDS > 0:
                self._workflow_cache[registry_key] = entry
            self._workflow_inflight.pop(registry_key, None)
        logger.info(
            "workflow dedup completed workflow=%s dedup_key=%s reply_chars=%s",
            decision.workflow_name,
            decision.dedup_key,
            len(result.reply),
        )

    def _workflow_dedup_fail(self, decision: _WorkflowDedupDecision, exc: BaseException) -> None:
        if decision.owner_future is None:
            return
        registry_key = (decision.workflow_name, decision.dedup_key)
        with self._workflow_dedup_lock:
            if not decision.owner_future.done():
                decision.owner_future.set_exception(exc)
            self._workflow_inflight.pop(registry_key, None)

    @staticmethod
    def _workflow_queue_name(execution: ExecutionContext) -> str:
        if execution.workflow_name == "parking_abnormal_review":
            return "parking_review"
        return "default_workflow"

    @staticmethod
    def _workflow_queue_workers(execution: ExecutionContext) -> int:
        if execution.workflow_name == "parking_abnormal_review":
            return PARKING_REVIEW_THREAD_WORKERS
        return WORKFLOW_THREAD_WORKERS

    def _workflow_queue_increment(self, execution: ExecutionContext) -> int:
        if execution.workflow_name == "parking_abnormal_review":
            return _parking_review_queue_increment()
        return _workflow_queue_increment()

    def _workflow_queue_decrement(self, execution: ExecutionContext) -> int:
        if execution.workflow_name == "parking_abnormal_review":
            return _parking_review_queue_decrement()
        return _workflow_queue_decrement()

    def _workflow_executor_for(self, execution: ExecutionContext) -> ThreadPoolExecutor:
        executor_attr = self.workflow_executors[self._workflow_queue_name(execution)]
        return cast(ThreadPoolExecutor, getattr(self, executor_attr))

    @staticmethod
    def queue_state() -> RuntimeQueueState:
        return RUNTIME_QUEUE_STATE

    def queue_state_snapshot(self) -> dict[str, RuntimeQueueSnapshot]:
        return {
            "default_workflow": self.queue_state().snapshot(
                queue_name="default_workflow",
                workers=WORKFLOW_THREAD_WORKERS,
                max_backlog=None,
            ),
            "parking_review": self.queue_state().snapshot(
                queue_name="parking_review",
                workers=PARKING_REVIEW_THREAD_WORKERS,
                max_backlog=PARKING_REVIEW_MAX_QUEUE_BACKLOG,
            ),
        }

    @staticmethod
    def _parking_review_overload_reply(execution: ExecutionContext) -> str:
        prompt_text = next(
            (message.content for message in reversed(execution.request.messages) if message.role == "user"),
            "",
        )
        review_source_id = _extract_review_source_id_for_overload(prompt_text)
        review_event_id = f"{review_source_id}_event_1" if review_source_id else "queue_overloaded_event_1"
        return json.dumps(
            [
                {
                    "reviewSourceId": review_source_id,
                    "reviewEventId": review_event_id,
                    "hit": 0,
                    "result": "系统复判任务积压，已按证据不足处理，请稍后重试。",
                }
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def _stream_agent_loop_events(self, execution: ExecutionContext) -> AsyncIterator[ChatEvent]:
        conversation = self._conversation_with_vision_attachments(execution.conversation, execution.request.attachments, execution.paths)
        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        yield recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": "agent_loop",
                "execution_mode": "agent_loop",
                "stateless": execution.agent_config.runtime.stateless,
            },
        )
        self.session_manager.begin_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            agent=execution.agent_config.name,
            workflow="agent_loop",
        )

        emitted = 1
        try:
            direct_result = self._try_complete_direct_json_artifact_turn(execution, recorder)
        except Exception:
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="failed",
            )
            raise
        if direct_result is not None:
            for event in recorder.events[emitted:]:
                yield event
            return
        try:
            direct_skill_result = self._try_auto_execute_primary_skill_turn(execution, recorder)
        except Exception:
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="failed",
            )
            raise
        if direct_skill_result is not None:
            for event in recorder.events[emitted:]:
                yield event
            return

        llm = OpenAICompatibleClient(execution.agent_config, runtime_options=execution.request.runtime_options)
        if not llm.configured:
            yield recorder.emit(
                "llm.started",
                {
                    "model": llm.model,
                    "temperature": llm.temperature,
                    "top_p": llm.top_p,
                    "max_tokens": llm.max_tokens,
                    "request_timeout_seconds": llm.request_timeout_seconds,
                    "configured": llm.configured,
                },
            )
            chunks: list[str] = []
            async for chunk in llm.stream_complete(execution.agent_config.prompts.system, conversation):
                chunks.append(chunk)
                yield recorder.emit("agent.message.delta", {"text": chunk})

            reply = "".join(chunks)
            final_messages = [*conversation, {"role": "assistant", "content": reply}]
            yield recorder.emit("agent.message", {"text": reply})
            result = AgentRunResult(
                agent=execution.agent_config.name,
                thread_id=execution.paths.thread_id,
                reply=reply,
                metadata={
                    "workflow": "agent_loop",
                    "run_id": recorder.run_id,
                    "llm_configured": llm.configured,
                    "model": llm.model,
                    "temperature": llm.temperature,
                    "top_p": llm.top_p,
                    "max_tokens": llm.max_tokens,
                    "request_timeout_seconds": llm.request_timeout_seconds,
                    "tool_rounds": 0,
                    "tool_call_count": 0,
                },
            )
            self._enrich_required_inputs(result, execution.request)
            yield recorder.emit("run.completed", {"result": result.model_dump()})
            self.session_store.save(
                execution.paths,
                self._strip_internal_message_fields(final_messages),
                run_id=recorder.run_id,
            )
            self._persist_events(execution.agent_config, execution.request, execution.paths.thread_id, recorder.events, result)
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="completed",
            )
            return

        try:
            loop_result = await self.agent_loop.run(
                agent_config=execution.agent_config,
                messages=conversation,
                thread_id=execution.paths.thread_id,
                recorder=recorder,
                runtime_options=execution.request.runtime_options,
                emit_message_delta=True,
            )
        except Exception:
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="failed",
            )
            raise
        self._enrich_required_inputs(loop_result.result, execution.request)
        self._replace_final_result_event(recorder.events, loop_result.result)
        for event in recorder.events[emitted:]:
            yield event
        self.session_store.save(
            execution.paths,
            self._strip_internal_message_fields(loop_result.messages),
            run_id=recorder.run_id,
        )
        self._persist_events(
            execution.agent_config,
            execution.request,
            execution.paths.thread_id,
            recorder.events,
            loop_result.result,
        )
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="completed" if loop_result.result.status == "completed" else "failed",
        )

    async def _stream_input_required_events(self, execution: ExecutionContext) -> AsyncIterator[ChatEvent]:
        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        yield recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": execution.workflow_name or "agent_loop",
                "execution_mode": execution.workflow_name or "agent_loop",
                "stateless": execution.agent_config.runtime.stateless,
            },
        )
        self.session_manager.begin_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            agent=execution.agent_config.name,
            workflow=execution.workflow_name or "agent_loop",
        )
        result = self._input_required_result(
            execution.agent_config,
            execution.paths.thread_id,
            execution.input_required,
            recorder.run_id,
            execution.workflow_name,
            execution.request.runtime_options,
        )
        yield recorder.emit("agent.message.delta", {"text": result.reply})
        yield recorder.emit("agent.message", {"text": result.reply})
        yield recorder.emit("run.completed", {"result": result.model_dump()})
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(execution.agent_config, execution.request, execution.paths.thread_id, recorder.events, result)
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="completed",
        )

    def _try_complete_direct_json_artifact_turn(
        self,
        execution: ExecutionContext,
        recorder: EventRecorder,
    ) -> AgentRunResult | None:
        intent = self._direct_json_artifact_intent(execution.request)
        if intent is None:
            return None

        artifact = self.artifact_store.write_text_artifact(execution.paths, intent.filename, intent.content)
        result = AgentRunResult(
            agent=execution.agent_config.name,
            thread_id=execution.paths.thread_id,
            reply=f"已生成 JSON 文件：outputs/{artifact.name}",
            artifacts=[artifact],
            metadata={
                "workflow": "agent_loop",
                "run_id": recorder.run_id,
                "direct_artifact_generation": True,
                "direct_artifact_reason": intent.reason,
                "tool_rounds": 0,
                "tool_call_count": 0,
                "mode": execution.request.runtime_options.mode or "edit",
            },
        )
        recorder.emit(
            "artifact.created",
            {
                "artifact": artifact.model_dump(mode="json"),
                "filename": artifact.name,
                "reason": intent.reason,
            },
        )
        recorder.emit("agent.message", {"text": result.reply, "content": result.content})
        recorder.emit("run.completed", {"result": result.model_dump()})
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(
            execution.agent_config,
            execution.request,
            execution.paths.thread_id,
            recorder.events,
            result,
        )
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="completed",
        )
        return result

    def _try_auto_execute_primary_skill_turn(
        self,
        execution: ExecutionContext,
        recorder: EventRecorder,
    ) -> AgentRunResult | None:
        runtime_options = execution.request.runtime_options
        if not runtime_options.selected_skills:
            return None
        skill_name = runtime_options.selected_skills[0].strip()
        if not skill_name:
            return None
        config_auto_execute = runtime_options.config_options.get("auto_execute_primary_skill") is True
        try:
            skill = SkillRegistry(self.app_template_registry.root_dir).get(skill_name)
        except KeyError as exc:
            if config_auto_execute:
                return self._auto_primary_skill_failed_result(
                    execution,
                    recorder,
                    skill_name=skill_name,
                    error=f"自动执行的 Skill 未注册或不可用：{skill_name}",
                    reason="skill_not_found",
                    cause=exc,
                )
            return None
        if not skill.executable:
            if config_auto_execute:
                return self._auto_primary_skill_failed_result(
                    execution,
                    recorder,
                    skill_name=skill_name,
                    error=f"自动执行的 Skill 不可执行：{skill_name}",
                    reason="skill_not_executable",
                )
            return None
        auto_execute = (
            config_auto_execute
            or skill.auto_execute is True
        )
        logger.debug(
            "primary skill auto-execute decision thread_id=%s skill=%s app_template=%s config_auto=%s skill_auto=%s executable=%s selected_skills=%s",
            execution.paths.thread_id,
            skill_name,
            runtime_options.app_template_name,
            runtime_options.config_options.get("auto_execute_primary_skill") is True,
            skill.auto_execute is True,
            skill.executable,
            runtime_options.selected_skills,
        )
        if not auto_execute:
            return None

        spec = self._auto_primary_skill_spec(skill_name, execution.request)
        recorder.emit("skill.started", {"skill_name": skill_name, "spec": spec, "auto_execute": True})
        recorder.emit("tool.started", {"tool_name": skill_name, "tool_call_id": f"auto-skill-{skill_name}"})
        try:
            run_result = SkillRunner(self.artifact_store, root_dir=self.app_template_registry.root_dir).run(
                skill_name,
                spec,
                execution.paths,
            )
        except Exception as exc:
            return self._auto_primary_skill_failed_result(
                execution,
                recorder,
                skill_name=skill_name,
                error=f"自动执行 Skill 失败：{skill_name}。原因：{exc}",
                reason="skill_execution_failed",
                cause=exc,
            )
        artifacts = [artifact.model_dump(mode="json") for artifact in run_result.outputs]
        recorder.emit(
            "tool.completed",
            {
                "tool_name": skill_name,
                "tool_call_id": f"auto-skill-{skill_name}",
                "is_error": False,
                "structured_content": {
                    "skill_name": skill_name,
                    "thread_id": execution.paths.thread_id,
                    "artifacts": artifacts,
                    "data": run_result.data,
                },
            },
        )
        recorder.emit(
            "skill.completed",
            {
                "skill_name": skill_name,
                "output_count": len(run_result.outputs),
                "data": run_result.data,
                "auto_execute": True,
            },
        )
        for artifact in run_result.outputs:
            recorder.emit("artifact.created", {"artifact": artifact.model_dump(mode="json")})

        primary = run_result.data.get("primary_artifact") or (run_result.outputs[0].path if run_result.outputs else "")
        reply = f"已生成大屏产物：{primary}" if primary else f"已执行技能：{skill_name}"
        result = AgentRunResult(
            agent=execution.agent_config.name,
            thread_id=execution.paths.thread_id,
            reply=reply,
            artifacts=run_result.outputs,
            metadata={
                "workflow": "agent_loop",
                "run_id": recorder.run_id,
                "auto_execute_primary_skill": True,
                "skill_name": skill_name,
                "skill_data": run_result.data,
                "tool_rounds": 0,
                "tool_call_count": 1,
                "mode": runtime_options.mode or "edit",
            },
        )
        recorder.emit("agent.message", {"text": result.reply, "content": result.content})
        recorder.emit("run.completed", {"result": result.model_dump()})
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(
            execution.agent_config,
            execution.request,
            execution.paths.thread_id,
            recorder.events,
            result,
        )
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="completed",
        )
        return result

    def _auto_primary_skill_failed_result(
        self,
        execution: ExecutionContext,
        recorder: EventRecorder,
        *,
        skill_name: str,
        error: str,
        reason: str,
        cause: BaseException | None = None,
    ) -> AgentRunResult:
        if cause is not None:
            logger.warning(
                "primary skill auto-execute failed thread_id=%s skill=%s reason=%s error=%s",
                execution.paths.thread_id,
                skill_name,
                reason,
                cause,
            )
        recorder.emit(
            "tool.failed",
            {
                "tool_name": skill_name,
                "tool_call_id": f"auto-skill-{skill_name}",
                "is_error": True,
                "error": error,
                "reason": reason,
            },
        )
        recorder.emit(
            "skill.failed",
            {
                "skill_name": skill_name,
                "error": error,
                "reason": reason,
                "auto_execute": True,
            },
        )
        result = AgentRunResult(
            agent=execution.agent_config.name,
            thread_id=execution.paths.thread_id,
            status="failed",
            reply=error,
            metadata={
                "workflow": "agent_loop",
                "run_id": recorder.run_id,
                "auto_execute_primary_skill": True,
                "skill_name": skill_name,
                "error": error,
                "error_reason": reason,
                "tool_rounds": 0,
                "tool_call_count": 1,
                "mode": execution.request.runtime_options.mode or "edit",
            },
        )
        recorder.emit("agent.message", {"text": result.reply, "content": result.content})
        recorder.emit("run.failed", {"error": error, "result": result.model_dump()})
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(
            execution.agent_config,
            execution.request,
            execution.paths.thread_id,
            recorder.events,
            result,
        )
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="failed",
        )
        return result

    @classmethod
    def _auto_primary_skill_spec(cls, skill_name: str, request: ChatRequest) -> dict[str, Any]:
        text = cls._last_user_text(request)
        return {
            "skill_name": skill_name,
            "objective": text or skill_name,
            "message": text,
        }

    @classmethod
    def _direct_json_artifact_intent(cls, request: ChatRequest) -> _DirectJsonArtifactIntent | None:
        text = cls._last_user_text(request)
        normalized = text.lower()
        if not normalized:
            return None
        if "json" not in normalized:
            return None
        if not any(keyword in normalized for keyword in ("文件", "file")):
            return None
        if not any(keyword in normalized for keyword in ("生成", "创建", "写", "保存", "输出", "create", "generate", "write", "save")):
            return None
        if not any(keyword in normalized for keyword in ("emoji", "emo", "emajl", "表情")):
            return None

        payload = {"emoji": "😊"}
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        return _DirectJsonArtifactIntent(
            filename=cls._json_artifact_filename(text),
            content=content,
            reason="json_emoji_file_intent",
        )

    @staticmethod
    def _json_artifact_filename(text: str) -> str:
        match = re.search(r"(?i)([A-Za-z0-9_.-]+\.json)\b", text)
        if match is not None:
            filename = match.group(1).strip(" .")
            if filename and "/" not in filename and "\\" not in filename:
                return filename
        return "emoji.json"

    @staticmethod
    def _request_required_inputs(request: ChatRequest) -> list[dict[str, Any]]:
        required_inputs = required_inputs_for_request(request)
        if request.runtime_options.mode != "yolo":
            return required_inputs
        return [item for item in required_inputs if item.get("type") != "dataset"]

    @staticmethod
    def _enrich_required_inputs(result: AgentRunResult, request: ChatRequest) -> None:
        required_inputs = required_inputs_for_result(result, request)
        if not required_inputs:
            return
        result.metadata = {
            **result.metadata,
            "requires_input": True,
            "required_inputs": required_inputs,
        }

    @staticmethod
    def _input_required_result(
        agent_config: AgentConfig,
        thread_id: str,
        required_inputs: list[dict[str, object]],
        run_id: str,
        workflow_name: str | None,
        runtime_options: RuntimeOptions | None = None,
    ) -> AgentRunResult:
        labels = "、".join(_input_type_label(item.get("type")) for item in required_inputs) or "文件"
        reply = f"请先上传{labels}后继续。"
        return AgentRunResult(
            agent=agent_config.name,
            thread_id=thread_id,
            reply=reply,
            metadata={
                "workflow": workflow_name or "agent_loop",
                "run_id": run_id,
                "requires_input": True,
                "required_inputs": required_inputs,
                "tool_rounds": 0,
                "tool_call_count": 0,
                "mode": (runtime_options.mode if runtime_options is not None and runtime_options.mode else "edit"),
            },
        )

    @staticmethod
    def _fixed_reply_text() -> str | None:
        explicit = os.getenv(FIXED_REPLY_ENV)
        if explicit is not None:
            return explicit.strip() or DEFAULT_FIXED_REPLY
        enabled = (os.getenv(FIXED_REPLY_ENABLED_ENV) or "").strip().lower()
        if enabled in {"1", "true", "yes", "on"}:
            return DEFAULT_FIXED_REPLY
        return None

    @staticmethod
    def _fixed_reply_forever_enabled() -> bool:
        enabled = (os.getenv(FIXED_REPLY_FOREVER_ENV) or "").strip().lower()
        return enabled in {"1", "true", "yes", "on"}

    @staticmethod
    def _fixed_reply_interval_seconds() -> float:
        raw = (os.getenv(FIXED_REPLY_INTERVAL_ENV) or "5").strip()
        try:
            return max(0.1, float(raw))
        except ValueError:
            return 5.0

    async def _stream_fixed_reply_forever_events(
        self,
        execution: ExecutionContext,
        reply: str,
    ) -> AsyncIterator[ChatEvent]:
        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        yield recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": "fixed_reply_forever",
                "execution_mode": "fixed_reply_forever",
                "stateless": execution.agent_config.runtime.stateless,
            },
        )
        self.session_manager.begin_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            agent=execution.agent_config.name,
            workflow="fixed_reply_forever",
        )
        interval_seconds = self._fixed_reply_interval_seconds()
        try:
            while True:
                yield recorder.emit(
                    "agent.message.delta",
                    {
                        "text": reply,
                        "fixed_reply": True,
                        "fixed_reply_forever": True,
                        "interval_seconds": interval_seconds,
                    },
                )
                await asyncio.sleep(interval_seconds)
        finally:
            self.session_manager.finish_turn(
                thread_id=execution.paths.thread_id,
                run_id=recorder.run_id,
                status="cancelled",
            )

    def _fixed_reply_result(self, execution: ExecutionContext, reply: str) -> tuple[AgentRunResult, list[ChatEvent]]:
        recorder = EventRecorder(agent=execution.agent_config.name, thread_id=execution.paths.thread_id)
        events = [
            recorder.emit(
                "run.started",
                {
                    "run_id": recorder.run_id,
                    "workflow": "fixed_reply",
                    "execution_mode": "fixed_reply",
                    "stateless": execution.agent_config.runtime.stateless,
                },
            )
        ]
        self.session_manager.begin_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            agent=execution.agent_config.name,
            workflow="fixed_reply",
        )
        result = AgentRunResult(
            agent=execution.agent_config.name,
            thread_id=execution.paths.thread_id,
            reply=reply,
            metadata={
                "workflow": "fixed_reply",
                "run_id": recorder.run_id,
                "fixed_reply": True,
                "tool_rounds": 0,
                "tool_call_count": 0,
                "mode": (
                    execution.request.runtime_options.mode
                    if execution.request.runtime_options is not None and execution.request.runtime_options.mode
                    else "edit"
                ),
            },
        )
        events.append(recorder.emit("agent.message.delta", {"text": reply}))
        events.append(recorder.emit("agent.message", {"text": reply}))
        events.append(recorder.emit("run.completed", {"result": result.model_dump()}))
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=recorder.run_id,
        )
        self._persist_events(execution.agent_config, execution.request, execution.paths.thread_id, events, result)
        self.session_manager.finish_turn(
            thread_id=execution.paths.thread_id,
            run_id=recorder.run_id,
            status="completed",
        )
        return result, events

    @staticmethod
    def _replace_final_result_event(events: list[ChatEvent], result: AgentRunResult) -> None:
        for index in range(len(events) - 1, -1, -1):
            if events[index].type in {"run.completed", "run.failed"}:
                events[index].data["result"] = result.model_dump()
                return

    def _persist_events(
        self,
        agent_config: AgentConfig,
        request: ChatRequest,
        thread_id: str,
        events: list[ChatEvent],
        result: AgentRunResult | None,
    ) -> None:
        if not events:
            return
        run_id = str(events[0].data.get("run_id") or f"run-{uuid4().hex[:12]}")
        if result is not None:
            result.metadata.setdefault("run_id", run_id)
        self.run_event_store.save(
            run_id=run_id,
            agent=agent_config.name,
            thread_id=thread_id,
            events=events,
            result=result.model_dump(mode="json") if result is not None else None,
            agent_snapshot=self._agent_snapshot(agent_config),
            request_snapshot=self._request_snapshot(request),
        )

    @staticmethod
    def _workflow_messages(conversation: list[dict[str, object]]) -> list[Message]:
        messages: list[Message] = []
        for item in conversation:
            role = item.get("role")
            content = item.get("content")
            if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
                messages.append(Message(role=cast(Role, role), content=content))
        return messages

    @staticmethod
    def _conversation_with_result(
        conversation: list[dict[str, object]],
        result: AgentRunResult | None,
    ) -> list[dict[str, object]]:
        if result is None or not result.reply:
            return conversation
        if conversation and conversation[-1].get("role") == "assistant" and conversation[-1].get("content") == result.reply:
            return conversation
        return [*conversation, {"role": "assistant", "content": result.reply}]

    @classmethod
    def _conversation_with_vision_attachments(
        cls,
        conversation: list[dict[str, Any]],
        attachments: list[Attachment],
        paths: ThreadPaths,
    ) -> list[dict[str, Any]]:
        vision_attachments = cls._vision_attachment_payloads(attachments, paths)
        if not vision_attachments or not conversation:
            return conversation
        updated = [dict(message) for message in conversation]
        for index in range(len(updated) - 1, -1, -1):
            if updated[index].get("role") == "user":
                existing = updated[index].get("_attachments")
                merged = [*existing, *vision_attachments] if isinstance(existing, list) else vision_attachments
                updated[index]["_attachments"] = merged
                return updated
        return conversation

    @staticmethod
    def _mark_conversation_has_attachments(
        conversation: list[dict[str, Any]],
        attachments: list[Attachment],
    ) -> list[dict[str, Any]]:
        if not attachments or not conversation:
            return conversation
        updated = [dict(message) for message in conversation]
        for index in range(len(updated) - 1, -1, -1):
            if updated[index].get("role") == "user":
                updated[index]["_has_attachments"] = True
                return updated
        return conversation

    @classmethod
    def _vision_attachment_payloads(cls, attachments: list[Attachment], paths: ThreadPaths) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for attachment in attachments:
            mime_type = (attachment.mime_type or "").split(";", 1)[0].strip().lower()
            if not mime_type.startswith("image/"):
                continue
            if mime_type == "image/svg+xml":
                continue
            payload = attachment.model_dump(mode="python")
            local_path = cls._attachment_local_path(attachment.path or "", paths)
            if local_path is not None:
                payload["_local_path"] = str(local_path)
            payloads.append(payload)
        return payloads

    @staticmethod
    def _attachment_local_path(raw_path: str, paths: ThreadPaths) -> Path | None:
        normalized = raw_path.replace("\\", "/").strip()
        if not normalized:
            return None
        virtual_roots = {
            "/mnt/user-data/uploads": paths.uploads,
            "/mnt/user-data/outputs": paths.outputs,
        }
        for prefix, root in virtual_roots.items():
            if normalized == prefix or normalized.startswith(prefix + "/"):
                suffix = normalized[len(prefix):].lstrip("/")
                candidate = (root / suffix).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    return None
                return candidate
        candidate = Path(normalized).expanduser()
        if not candidate.is_absolute():
            return None
        try:
            return candidate.resolve()
        except OSError:
            return None

    @staticmethod
    def _strip_internal_message_fields(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stripped: list[dict[str, Any]] = []
        for message in messages:
            clean = dict(message)
            clean.pop("_attachments", None)
            clean.pop("_has_attachments", None)
            stripped.append(clean)
        return stripped

    @staticmethod
    def _run_id(events: list[ChatEvent]) -> str:
        if not events:
            return ""
        return str(events[0].data.get("run_id") or "")

    @staticmethod
    def _agent_snapshot(agent_config: AgentConfig) -> dict[str, object]:
        payload = agent_config.model_dump(mode="json")
        model = payload.get("model")
        if isinstance(model, dict):
            if model.get("api_key"):
                model["api_key"] = "********"
            if model.get("api_key_enc"):
                model["api_key_enc"] = "********"
        return payload

    @staticmethod
    def _request_snapshot(request: ChatRequest) -> dict[str, object]:
        payload = request.model_dump(mode="json")
        runtime_options = payload.get("runtime_options")
        if isinstance(runtime_options, dict) and runtime_options.get("api_key"):
            runtime_options["api_key"] = "********"
        return payload

    def _should_use_workflow(self, agent_config: AgentConfig, request: ChatRequest) -> bool:
        return self._selected_workflow_name(agent_config, request) is not None

    def _selected_workflow_name(self, agent_config: AgentConfig, request: ChatRequest) -> str | None:
        return self._explicit_workflow_name(request)

    def _prepare_execution(self, agent_config: AgentConfig, request: ChatRequest) -> tuple[AgentConfig, ChatRequest]:
        request = self._effective_request(request)
        return self._effective_agent_config(agent_config, request), request

    def _execution_context(self, agent_config: AgentConfig, request: ChatRequest) -> ExecutionContext:
        return build_execution_context(
            agent_config=agent_config,
            request=request,
            artifact_store=self.artifact_store,
            session_store=self.session_store,
            prepare_execution=self._prepare_execution,
            selected_workflow_name=self._selected_workflow_name,
            request_required_inputs=self._request_required_inputs,
        )

    def _effective_agent_config(self, agent_config: AgentConfig, request: ChatRequest) -> AgentConfig:
        updates: dict[str, object] = {}
        if self.skills is not None:
            updates["skills"] = list(self.skills)
        selected_skills = self._normalize_skills(
            request.runtime_options.selected_skills,
            root_dir=self.app_template_registry.root_dir,
            extra_aliases=self._runtime_skill_aliases(request.runtime_options),
        )
        if selected_skills:
            raw_skills = updates.get("skills")
            base_skills = list(raw_skills) if isinstance(raw_skills, list) else list(agent_config.skills)
            seen = set(base_skills)
            for skill_name in selected_skills:
                if skill_name not in seen:
                    base_skills.append(skill_name)
                    seen.add(skill_name)
            updates["skills"] = base_skills
        if not updates:
            return agent_config
        return agent_config.model_copy(update=updates, deep=True)

    def _effective_request(self, request: ChatRequest) -> ChatRequest:
        # Request normalization happens once here so all downstream execution
        # paths observe the same derived runtime options, attachments, and
        # message context.
        runtime_options = self._runtime_options_with_message_capabilities(request.runtime_options, request.messages)
        runtime_options = self._runtime_options_with_app_template_defaults(runtime_options)
        runtime_options = self._runtime_options_with_skill_aliases(runtime_options)
        runtime_options = self._effective_runtime_options(runtime_options)
        runtime_options = self._runtime_options_with_composite_skills(runtime_options)
        attachments = self._expanded_video_record_attachments(request.attachments)
        if not attachments and runtime_options.thread_id:
            attachments = self._thread_file_attachments(runtime_options.thread_id)
        if attachments and runtime_options.thread_id:
            attachments = self._materialize_remote_attachments(attachments, runtime_options.thread_id)
        attachments = self._attachments_with_review_task_context(runtime_options, attachments)
        messages = self._messages_with_attachment_context(request.messages, attachments)
        runtime_options = self._runtime_options_with_routed_primary_skill(runtime_options, messages, attachments)
        if runtime_options is request.runtime_options and messages is request.messages and attachments is request.attachments:
            return request
        return request.model_copy(
            update={"runtime_options": runtime_options, "messages": messages, "attachments": attachments},
            deep=True,
        )

    def _runtime_options_with_app_template_defaults(self, runtime_options: RuntimeOptions) -> RuntimeOptions:
        app_template_name = (runtime_options.app_template_name or "").strip()
        if not app_template_name:
            return runtime_options
        try:
            template = self.app_template_registry.get(app_template_name)
        except (KeyError, ValueError):
            return runtime_options
        merged = merge_runtime_options_with_template(runtime_options, template)
        updates: dict[str, object] = {}
        # Some clients send selected_skills/selected_mcp_tools as explicit empty
        # lists while relying on app_template_name for the actual defaults.
        if not merged.selected_skills and template.selected_skills:
            updates["selected_skills"] = list(template.selected_skills)
        if not merged.selected_mcp_tools and template.selected_mcp_tools:
            updates["selected_mcp_tools"] = list(template.selected_mcp_tools)
        if updates:
            merged = merged.model_copy(update=updates, deep=True)
        return merged

    def _runtime_options_with_message_capabilities(
        self,
        runtime_options: RuntimeOptions,
        messages: list[Message],
    ) -> RuntimeOptions:
        if runtime_options.selected_skills:
            return runtime_options
        if runtime_options.workflow and runtime_options.workflow not in {"agent_loop", "default"}:
            return runtime_options
        selected_skills = self._selected_skills_from_messages(messages)
        if selected_skills:
            # Preserve the original explicit-field set so downstream model
            # resolution can still tell which runtime options were truly provided
            # by the caller versus which ones are still eligible for app-template
            # or bootstrap defaults.
            return runtime_options.model_copy(update={"selected_skills": selected_skills}, deep=True)
        if (runtime_options.app_template_name or "").strip():
            return runtime_options
        if self._has_runtime_mcp_config(runtime_options):
            return runtime_options
        selected_skills = self._selected_skills_from_routing(messages)
        if not selected_skills:
            return runtime_options
        # Preserve the original explicit-field set so downstream model
        # resolution can still tell which runtime options were truly provided
        # by the caller versus which ones are still eligible for app-template
        # or bootstrap defaults.
        return runtime_options.model_copy(update={"selected_skills": selected_skills}, deep=True)

    @staticmethod
    def _has_runtime_mcp_config(runtime_options: RuntimeOptions) -> bool:
        config_options = runtime_options.config_options
        for key in ("mcpServers", "mcp_servers", "runtime_mcp_tools", "_runtime_mcp_tools"):
            value = config_options.get(key)
            if isinstance(value, list) and value:
                return True
        return False

    def _runtime_options_with_skill_aliases(self, runtime_options: RuntimeOptions) -> RuntimeOptions:
        selected_skills = self._normalize_skills(
            runtime_options.selected_skills,
            root_dir=self.app_template_registry.root_dir,
            extra_aliases=self._runtime_skill_aliases(runtime_options),
        )
        if selected_skills == runtime_options.selected_skills:
            return runtime_options
        return runtime_options.model_copy(update={"selected_skills": selected_skills or []}, deep=True)

    def _runtime_options_with_composite_skills(self, runtime_options: RuntimeOptions) -> RuntimeOptions:
        raw_selected_skills = self._normalize_skills(runtime_options.selected_skills) or []
        selected_skills = self._normalize_skills(
            runtime_options.selected_skills,
            root_dir=self.app_template_registry.root_dir,
            extra_aliases=self._runtime_skill_aliases(runtime_options),
        ) or []
        if not selected_skills:
            return runtime_options
        skill_registry = SkillRegistry(self.app_template_registry.root_dir)
        expanded_skills: list[str] = []
        composite_contexts: list[dict[str, object]] = []
        seen: set[str] = set()
        for skill_name in selected_skills:
            self._append_unique(expanded_skills, seen, skill_name)
            try:
                skill = skill_registry.get(skill_name)
            except KeyError:
                continue
            if not skill.composite:
                continue
            # Composite skills stay in the selected list, but their child skills
            # are expanded so the loop can expose concrete tools to the model.
            composite_contexts.append(self._composite_skill_context(skill))
            for child_skill in skill.child_skills:
                self._append_unique(expanded_skills, seen, child_skill)
        if expanded_skills == raw_selected_skills and not composite_contexts:
            return runtime_options
        config_options = dict(runtime_options.config_options)
        if composite_contexts:
            config_options["composite_skills"] = composite_contexts
        return runtime_options.model_copy(
            update={"selected_skills": expanded_skills, "config_options": config_options},
            deep=True,
        )

    def _runtime_options_with_routed_primary_skill(
        self,
        runtime_options: RuntimeOptions,
        messages: list[Message],
        attachments: list[Attachment],
    ) -> RuntimeOptions:
        selected_skills = [skill.strip() for skill in runtime_options.selected_skills if skill.strip()]
        if len(selected_skills) <= 1:
            return runtime_options
        routing_text = "\n".join(message.content for message in messages if message.role == "user").strip()
        if not routing_text:
            return runtime_options
        review_routing = self._is_review_runtime_options(runtime_options) or self._is_review_skill_routing_request(
            routing_text,
            attachments,
        )
        if not review_routing:
            return runtime_options
        try:
            from app.core.skills.plugins import SkillPluginManager

            plugin_manager = SkillPluginManager(self.app_template_registry.root_dir)
            semantic_values = self._attachment_semantic_values(attachments)
            labels = self._attachment_labels(attachments)
            candidate = self._candidate_from_attachment_target_skill(attachments, selected_skills)
            score = 120 if candidate else 0
            source = "attachment_target_skill" if candidate else "message_attachment_context"
            label_updates: dict[str, str] = {}
            if not candidate:
                candidate = self._candidate_from_scene_skill_aliases(
                    runtime_options,
                    semantic_values,
                    routing_text,
                    selected_skills,
                )
                score = 115 if candidate else 0
                if candidate:
                    source = "scene_skill_aliases"
            if not candidate:
                candidate, score = self._candidate_from_attachment_semantics(
                    plugin_manager,
                    semantic_values,
                    selected_skills,
                )
                if candidate:
                    source = "attachment_event_semantics"
            if not candidate:
                candidate = self._candidate_from_skill_label_aliases(runtime_options, labels, selected_skills)
                score = 100 if candidate else 0
                if candidate:
                    source = "skill_label_aliases"
            if not candidate:
                candidate, score, label_updates = self._candidate_from_attachment_labels(
                    plugin_manager,
                    labels,
                    selected_skills,
                )
                if candidate:
                    source = "attachment_object_labels"
            if not candidate:
                candidate, score = plugin_manager.select_skill_candidate(
                    routing_text,
                    list(attachments),
                    selected_skills,
                )
                if candidate:
                    label_updates = self._skill_label_alias_updates_for_candidate(
                        plugin_manager,
                        labels,
                        selected_skills,
                        candidate,
                    )
        except Exception as exc:
            logger.warning("primary skill routing skipped selected_skills=%s error=%s", selected_skills, exc)
            return runtime_options
        if label_updates:
            runtime_options = self._runtime_options_with_skill_label_alias_updates(runtime_options, label_updates)
        if score <= 0 or not candidate or candidate == selected_skills[0] or candidate not in selected_skills:
            if review_routing and (score <= 0 or not candidate):
                config_options = dict(runtime_options.config_options)
                config_options["skill_routing_error"] = {
                    "reason": "no_skill_match",
                    "source": "attachment_object_labels",
                    "selected_skills": selected_skills,
                    "semantic_values": semantic_values,
                    "labels": labels,
                    "message": (
                        "No selected skill matched the review text, attachment event semantics, or attachment object labels; "
                        "the runtime refused to fall back to the first selected skill."
                    ),
                }
                logger.warning(
                    "primary skill routing failed selected_skills=%s attachments=%s",
                    selected_skills,
                    len(attachments),
                )
                return runtime_options.model_copy(
                    update={"selected_skills": [], "config_options": config_options},
                    deep=True,
                )
            return runtime_options
        routed_skills = [candidate, *[skill for skill in selected_skills if skill != candidate]]
        config_options = dict(runtime_options.config_options)
        config_options["routed_primary_skill"] = {
            "skill_name": candidate,
            "score": score,
            "source": source,
            "semantic_values": semantic_values,
            "labels": labels,
        }
        logger.info(
            "primary skill routed skill=%s score=%s source=%s selected_skills=%s semantic_values=%s labels=%s",
            candidate,
            score,
            source,
            selected_skills,
            semantic_values,
            labels,
        )
        return runtime_options.model_copy(
            update={"selected_skills": routed_skills, "config_options": config_options},
            deep=True,
        )

    def _attachments_with_review_task_context(
        self,
        runtime_options: RuntimeOptions,
        attachments: list[Attachment],
    ) -> list[Attachment]:
        if not attachments:
            return attachments
        task_mappings = self._review_task_mappings(runtime_options.config_options)
        skill_aliases = self._review_task_skill_aliases(runtime_options.config_options)
        event_aliases = self._review_task_event_aliases(runtime_options.config_options)
        if not task_mappings and not skill_aliases and not event_aliases:
            return attachments
        selected_skills = [skill.strip() for skill in runtime_options.selected_skills if skill.strip()]
        app_template_name = (runtime_options.app_template_name or "").strip()
        updated_attachments: list[Attachment] = []
        changed = False
        for attachment in attachments:
            metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
            task_context = self._attachment_review_task_context(attachment)
            if not task_context:
                updated_attachments.append(attachment)
                continue
            metadata_updates = self._review_task_metadata_updates(
                task_context,
                app_template_name=app_template_name,
                task_mappings=task_mappings,
                skill_aliases=skill_aliases,
                event_aliases=event_aliases,
                selected_skills=selected_skills,
                runtime_options=runtime_options,
            )
            if not metadata_updates:
                updated_attachments.append(attachment)
                continue
            merged_metadata = dict(metadata)
            for key, value in metadata_updates.items():
                if key in merged_metadata and str(merged_metadata.get(key) or "").strip():
                    continue
                merged_metadata[key] = value
            if merged_metadata == metadata:
                updated_attachments.append(attachment)
                continue
            updated_attachments.append(attachment.model_copy(update={"metadata": merged_metadata}, deep=True))
            changed = True
        return updated_attachments if changed else attachments

    def _review_task_metadata_updates(
        self,
        task_context: dict[str, str],
        *,
        app_template_name: str,
        task_mappings: dict[str, dict[str, str]],
        skill_aliases: dict[str, str],
        event_aliases: dict[str, dict[str, str]],
        selected_skills: list[str],
        runtime_options: RuntimeOptions,
    ) -> dict[str, str]:
        updates: dict[str, str] = {}
        stream_id = task_context.get("streamId") or ""
        task_id = task_context.get("cvTaskId") or ""
        source_id = task_context.get("sourceId") or ""
        if stream_id:
            updates["streamId"] = stream_id
        if task_id:
            updates["cvTaskId"] = task_id
        if source_id:
            updates["sourceId"] = source_id
        raw_skill_name = ""
        for key in self._review_task_alias_keys(app_template_name, task_context):
            mapping = task_mappings.get(key)
            if mapping:
                raw_skill_name = mapping.get("targetSkill") or ""
                for update_key, update_value in mapping.items():
                    if update_key != "targetSkill":
                        updates[update_key] = update_value
                break
        if not any(key in updates for key in ("applicationScene", "eventTypeName", "cvTaskName")):
            for key in self._review_task_alias_keys(app_template_name, task_context):
                event_update = event_aliases.get(key)
                if event_update:
                    updates.update(event_update)
                    break
        if not raw_skill_name:
            for key in self._review_task_alias_keys(app_template_name, task_context):
                raw_skill_name = skill_aliases.get(key) or ""
                if raw_skill_name:
                    break
        if raw_skill_name:
            allowed = set(selected_skills)
            normalized = self._normalize_skills(
                [raw_skill_name],
                root_dir=self.app_template_registry.root_dir,
                extra_aliases=self._runtime_skill_aliases(runtime_options),
            ) or []
            skill_name = normalized[0] if normalized else raw_skill_name
            if not allowed or skill_name in allowed:
                updates["targetSkill"] = skill_name
        return updates

    @classmethod
    def _review_task_mappings(cls, config_options: dict[str, Any]) -> dict[str, dict[str, str]]:
        value = config_options.get("review_task_mappings")
        if not isinstance(value, dict):
            return {}
        mappings: dict[str, dict[str, str]] = {}
        for raw_key, raw_value in value.items():
            alias_key = str(raw_key or "").strip()
            mapping = cls._review_task_mapping_update(raw_value)
            if alias_key and mapping:
                mappings[alias_key] = mapping
        return mappings

    @classmethod
    def _review_task_mapping_update(cls, value: object) -> dict[str, str]:
        if isinstance(value, str) and value.strip():
            return {"targetSkill": value.strip()}
        if not isinstance(value, dict):
            return {}
        updates: dict[str, str] = {}
        event_value = value.get("event_semantics") or value.get("eventSemantics") or value.get("event") or value
        updates.update(cls._review_task_event_update(event_value))
        skill_name = cls._first_string(
            value.get("skill_name"),
            value.get("skillName"),
            value.get("skill"),
            value.get("targetSkill"),
            value.get("target_skill"),
        )
        if skill_name:
            updates["targetSkill"] = skill_name
        return updates

    @classmethod
    def _review_task_skill_aliases(cls, config_options: dict[str, Any]) -> dict[str, str]:
        aliases: dict[str, str] = {}
        for key in ("review_task_skill_aliases", "task_skill_aliases", "cv_task_skill_aliases"):
            value = config_options.get(key)
            if not isinstance(value, dict):
                continue
            for raw_key, raw_value in value.items():
                alias_key = str(raw_key or "").strip()
                skill_name = cls._first_string(raw_value)
                if alias_key and skill_name:
                    aliases[alias_key] = skill_name
        return aliases

    @classmethod
    def _review_task_event_aliases(cls, config_options: dict[str, Any]) -> dict[str, dict[str, str]]:
        aliases: dict[str, dict[str, str]] = {}
        for key in ("review_task_event_aliases", "task_event_aliases", "cv_task_event_aliases"):
            value = config_options.get(key)
            if not isinstance(value, dict):
                continue
            for raw_key, raw_value in value.items():
                alias_key = str(raw_key or "").strip()
                event_update = cls._review_task_event_update(raw_value)
                if alias_key and event_update:
                    aliases[alias_key] = event_update
        return aliases

    @classmethod
    def _review_task_event_update(cls, value: object) -> dict[str, str]:
        if isinstance(value, str) and value.strip():
            clean = value.strip()
            return {"applicationScene": clean, "eventTypeName": clean, "cvTaskName": clean}
        if not isinstance(value, dict):
            return {}
        updates: dict[str, str] = {}
        for key in (
            "appTemplateName",
            "applicationScene",
            "eventTypeName",
            "cvTaskName",
            "taskName",
            "sceneName",
            "algorithmName",
            "targetSkillName",
        ):
            clean = cls._first_string(value.get(key))
            if clean:
                updates[key] = clean
        return updates

    @classmethod
    def _review_task_alias_keys(cls, app_template_name: str, task_context: dict[str, str]) -> list[str]:
        keys: list[str] = []
        for value in (
            task_context.get("streamId"),
            task_context.get("cvTaskId"),
        ):
            clean = str(value or "").strip()
            if not clean:
                continue
            if app_template_name:
                keys.append(f"{app_template_name}:{clean}")
            keys.append(clean)
        return keys

    @classmethod
    def _attachment_review_task_context(cls, attachment: Attachment) -> dict[str, str]:
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        stream_id = cls._first_string(
            metadata.get("streamId"),
            metadata.get("stream_id"),
            metadata.get("stream"),
        )
        for value in cls._attachment_review_task_uri_candidates(attachment):
            stream_id = stream_id or cls._stream_id_from_attachment_uri(value)
            if stream_id:
                break
        stream_id = unquote(stream_id)
        task_id = cls._first_string(
            metadata.get("cvTaskId"),
            metadata.get("cv_task_id"),
            metadata.get("taskId"),
            metadata.get("task_id"),
        )
        source_id = cls._first_string(metadata.get("sourceId"), metadata.get("source_id"))
        if stream_id:
            parts = [part.strip() for part in stream_id.split("/") if part.strip()]
            if len(parts) >= 2 and not task_id:
                task_id = parts[1]
            if len(parts) >= 3 and not source_id:
                source_id = parts[2]
        context: dict[str, str] = {}
        if stream_id:
            context["streamId"] = stream_id
        if task_id:
            context["cvTaskId"] = task_id
        if source_id:
            context["sourceId"] = source_id
        return context

    @classmethod
    def _attachment_review_task_uri_candidates(cls, attachment: Attachment) -> list[str]:
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        candidates: list[str] = []
        for value in (
            metadata.get("original_uri"),
            metadata.get("uri"),
            metadata.get("url"),
            metadata.get("path"),
            attachment.path,
        ):
            clean = cls._first_string(value)
            if clean and clean not in candidates:
                candidates.append(clean)
        return candidates

    @classmethod
    def _stream_id_from_attachment_uri(cls, uri: str) -> str:
        decoded_uri = cls._decoded_edge_read_uri(uri) or uri
        for text in (decoded_uri, unquote(decoded_uri)):
            match = re.search(r"(?:[?&]|^)streamId=([^&]+)", text)
            if match:
                return unquote(match.group(1)).strip()
        return ""

    @classmethod
    def _decoded_edge_read_uri(cls, uri: str) -> str:
        marker = "/_read/"
        if marker not in uri:
            return ""
        token = uri.split(marker, 1)[1].split("?", 1)[0]
        if "." in token:
            token = token.rsplit(".", 1)[0]
        token = token.strip()
        if not token:
            return ""
        try:
            padded = token + "=" * ((4 - len(token) % 4) % 4)
            return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
        except Exception:
            return ""

    @classmethod
    def _candidate_from_attachment_target_skill(
        cls,
        attachments: list[Attachment],
        selected_skills: list[str],
    ) -> str:
        allowed = set(selected_skills)
        matches: set[str] = set()
        for attachment in attachments:
            metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
            target = cls._first_string(
                metadata.get("targetSkill"),
                metadata.get("target_skill"),
                metadata.get("skillName"),
                metadata.get("skill_name"),
            )
            if target and target in allowed:
                matches.add(target)
        return next(iter(matches)) if len(matches) == 1 else ""

    @classmethod
    def _candidate_from_attachment_semantics(
        cls,
        plugin_manager: Any,
        semantic_values: list[str],
        selected_skills: list[str],
    ) -> tuple[str, int]:
        if not semantic_values:
            return "", 0
        routing_text = "\n".join(semantic_values)
        return plugin_manager.select_skill_candidate(routing_text, [], selected_skills)

    def _candidate_from_scene_skill_aliases(
        self,
        runtime_options: RuntimeOptions,
        semantic_values: list[str],
        routing_text: str,
        selected_skills: list[str],
    ) -> str:
        aliases = self._scene_skill_aliases(runtime_options.config_options)
        if not aliases:
            return ""
        allowed = set(selected_skills)
        matches: set[str] = set()
        semantic_scene_keys = {self._scene_key(value) for value in semantic_values}
        normalized_routing_text = self._scene_key(routing_text)
        for scene_key, raw_skill_names in aliases.items():
            if scene_key in semantic_scene_keys or (
                normalized_routing_text and scene_key in normalized_routing_text
            ):
                for raw_skill_name in raw_skill_names:
                    normalized = self._normalize_skills(
                        [raw_skill_name],
                        root_dir=self.app_template_registry.root_dir,
                        extra_aliases=self._runtime_skill_aliases(runtime_options),
                    ) or []
                    for skill_name in normalized or [raw_skill_name]:
                        if skill_name in allowed:
                            matches.add(skill_name)
        return next(iter(matches)) if len(matches) == 1 else ""

    @classmethod
    def _candidate_from_skill_label_aliases(
        cls,
        runtime_options: RuntimeOptions,
        labels: list[str],
        selected_skills: list[str],
    ) -> str:
        if not labels:
            return ""
        aliases = cls._skill_label_aliases(runtime_options.config_options.get("skill_label_aliases"))
        if not aliases:
            return ""
        allowed = set(selected_skills)
        matches: set[str] = set()
        for label in labels:
            for key in {label.strip(), cls._label_key(label)}:
                if not key:
                    continue
                for skill_name in aliases.get(key, []):
                    if skill_name in allowed:
                        matches.add(skill_name)
        return next(iter(matches)) if len(matches) == 1 else ""

    @classmethod
    def _candidate_from_attachment_labels(
        cls,
        plugin_manager: Any,
        labels: list[str],
        selected_skills: list[str],
    ) -> tuple[str, int, dict[str, str]]:
        if not labels:
            return "", 0, {}
        matches: dict[str, tuple[str, int]] = {}
        for label in labels:
            candidate, score = plugin_manager.select_skill_candidate(
                cls._label_routing_text(label),
                [],
                selected_skills,
            )
            if candidate and score > 0:
                matches[cls._label_key(label)] = (candidate, score)
        candidates = {candidate for candidate, _score in matches.values()}
        if len(candidates) != 1:
            return "", 0, {}
        candidate = next(iter(candidates))
        score = max(score for _candidate, score in matches.values())
        updates = {label: candidate for label, (_candidate, _score) in matches.items()}
        return candidate, score, updates

    @classmethod
    def _skill_label_alias_updates_for_candidate(
        cls,
        plugin_manager: Any,
        labels: list[str],
        selected_skills: list[str],
        candidate: str,
    ) -> dict[str, str]:
        updates: dict[str, str] = {}
        for label in labels:
            skill_name, score = plugin_manager.select_skill_candidate(
                cls._label_routing_text(label),
                [],
                selected_skills,
            )
            if skill_name == candidate and score > 0:
                updates[cls._label_key(label)] = candidate
        return updates

    @classmethod
    def _skill_label_aliases(cls, value: object) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        aliases: dict[str, list[str]] = {}
        for raw_key, raw_value in value.items():
            key = cls._label_key(str(raw_key))
            if not key:
                continue
            names = cls._alias_skill_names(raw_value)
            if names:
                aliases[key] = names
        return aliases

    @classmethod
    def _scene_skill_aliases(cls, config_options: dict[str, Any]) -> dict[str, list[str]]:
        aliases: dict[str, list[str]] = {}
        for config_key in (
            "scene_skill_aliases",
            "review_scene_skill_aliases",
            "event_scene_skill_aliases",
            "cv_scene_skill_aliases",
        ):
            value = config_options.get(config_key)
            if not isinstance(value, dict):
                continue
            for raw_key, raw_value in value.items():
                key = cls._scene_key(str(raw_key))
                if not key:
                    continue
                names = cls._alias_skill_names(raw_value)
                if names:
                    aliases[key] = names
        return aliases

    @staticmethod
    def _alias_skill_names(value: object) -> list[str]:
        names: list[str] = []
        if isinstance(value, str):
            clean = value.strip()
            if clean:
                names.append(clean)
        elif isinstance(value, list):
            for item in value:
                clean = str(item).strip()
                if clean and clean not in names:
                    names.append(clean)
        return names

    @classmethod
    def _runtime_options_with_skill_label_alias_updates(
        cls,
        runtime_options: RuntimeOptions,
        updates: dict[str, str],
    ) -> RuntimeOptions:
        if not updates:
            return runtime_options
        config_options = dict(runtime_options.config_options)
        aliases = dict(config_options.get("skill_label_aliases")) if isinstance(config_options.get("skill_label_aliases"), dict) else {}
        changed = False
        for label, skill_name in sorted(updates.items()):
            if aliases.get(label) == skill_name:
                continue
            aliases[label] = skill_name
            changed = True
        if not changed:
            return runtime_options
        config_options["skill_label_aliases"] = aliases
        cls._persist_upload_app_skill_label_aliases(runtime_options.app_template_name, updates)
        return runtime_options.model_copy(update={"config_options": config_options}, deep=True)

    @classmethod
    def _persist_upload_app_skill_label_aliases(cls, app_template_name: str | None, updates: dict[str, str]) -> None:
        clean_name = str(app_template_name or "").strip()
        if not clean_name or "/" in clean_name or "\\" in clean_name or clean_name in {".", ".."}:
            return
        path = Path(__file__).resolve().parents[3] / "config" / "upload" / "apps" / f"{clean_name}.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            runtime_options = data.setdefault("runtime_options", {})
            if not isinstance(runtime_options, dict):
                runtime_options = {}
                data["runtime_options"] = runtime_options
            config_options = runtime_options.setdefault("config_options", {})
            if not isinstance(config_options, dict):
                config_options = {}
                runtime_options["config_options"] = config_options
            aliases = config_options.setdefault("skill_label_aliases", {})
            if not isinstance(aliases, dict):
                aliases = {}
                config_options["skill_label_aliases"] = aliases
            changed = False
            for label, skill_name in sorted(updates.items()):
                if aliases.get(label) == skill_name:
                    continue
                aliases[label] = skill_name
                changed = True
            if changed:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("persist skill label aliases failed app_template=%s error=%s", clean_name, exc)

    @classmethod
    def _attachment_labels(cls, attachments: list[Attachment]) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for attachment in attachments:
            metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
            objects = metadata.get("objects")
            if not isinstance(objects, list):
                continue
            for item in objects:
                if not isinstance(item, dict):
                    continue
                label = cls._first_string(item.get("label"), item.get("name"), item.get("class"), item.get("type"))
                others = item.get("others")
                if isinstance(others, dict):
                    label = label or cls._first_string(others.get("text"), others.get("label"))
                key = cls._label_key(label)
                if key and key not in seen:
                    labels.append(label.strip())
                    seen.add(key)
        return labels

    @classmethod
    def _attachment_semantic_values(cls, attachments: list[Attachment]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for attachment in attachments:
            metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
            for value in cls._semantic_values_from_metadata(metadata):
                key = value.casefold()
                if key and key not in seen:
                    values.append(value)
                    seen.add(key)
        return values

    @classmethod
    def _semantic_values_from_metadata(cls, metadata: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in (
            "appTemplateName",
            "app_template_name",
            "applicationScene",
            "application_scene",
            "eventTypeName",
            "event_type_name",
            "cvTaskName",
            "cv_task_name",
            "taskName",
            "task_name",
            "sceneName",
            "scene_name",
            "algorithmName",
            "algorithm_name",
            "targetSkillName",
            "target_skill_name",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                cls._append_unique_text(values, value.strip())
        return values

    @classmethod
    def _label_routing_text(cls, label: str) -> str:
        clean = label.strip()
        normalized = cls._label_key(clean)
        spaced = re.sub(r"[_\-.]+", " ", clean).strip()
        return "\n".join(part for part in (clean, normalized, spaced) if part)

    @staticmethod
    def _label_key(label: str) -> str:
        clean = str(label or "").strip().casefold().replace("-", "_")
        clean = re.sub(r"\s+", "_", clean)
        return clean.strip("_")

    @staticmethod
    def _scene_key(value: str) -> str:
        clean = str(value or "").strip().casefold()
        clean = clean.replace("／", "/")
        return re.sub(r"\s+", "", clean)

    @staticmethod
    def _is_review_skill_routing_request(routing_text: str, attachments: list[Attachment]) -> bool:
        if not attachments:
            return False
        text = routing_text.casefold()
        return any(
            marker in text
            for marker in (
                "reviewsourceid",
                "review source",
                "复判",
                "审核",
                "匹配对应的skill",
                "匹配对应的 skill",
                "智能体审核",
            )
        )

    @staticmethod
    def _is_review_runtime_options(runtime_options: RuntimeOptions) -> bool:
        workflow = str(runtime_options.workflow or "").strip().casefold()
        if workflow and ("review" in workflow or "复判" in workflow):
            return True
        return workflow == "parking_abnormal_review"

    @staticmethod
    def _append_unique(target: list[str], seen: set[str], value: str) -> None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        target.append(normalized)
        seen.add(normalized)

    @staticmethod
    def _append_unique_text(target: list[str], value: str) -> None:
        normalized = value.strip()
        if not normalized:
            return
        seen = {item.casefold() for item in target}
        if normalized.casefold() in seen:
            return
        target.append(normalized)

    @staticmethod
    def _composite_skill_context(skill: SkillDefinition) -> dict[str, object]:
        return {
            "name": skill.name,
            "description": skill.description,
            "child_skills": list(skill.child_skills),
            "stages": list(skill.stages),
            "done_when": list(skill.done_when),
        }

    @classmethod
    def _selected_skills_from_messages(cls, messages: list[Message]) -> list[str]:
        for message in reversed(messages):
            if message.role != "user" or "Selected Skills:" not in message.content:
                continue
            match = re.search(r"(?im)^\s*Selected Skills:\s*(.+?)\s*$", message.content)
            if match is None:
                continue
            return cls._normalize_skills([item.strip() for item in match.group(1).split(",")]) or []
        return []

    def _selected_skills_from_routing(self, messages: list[Message]) -> list[str]:
        routing_text = "\n".join(message.content for message in messages if message.role == "user").strip()
        if not routing_text or WorkflowRouter.is_meta_request(routing_text):
            return []
        skill_name, score = WorkflowRouter.manifest_skill_match(
            routing_text,
            SkillRegistry(self.app_template_registry.root_dir).list(executable_only=True),
        )
        if score <= 0 or not skill_name:
            return []
        return [skill_name]

    def _thread_file_attachments(self, thread_id: str) -> list[Attachment]:
        paths = self.artifact_store.prepare_thread(thread_id)
        attachments: list[Attachment] = []
        for root, scope in ((paths.uploads, "uploads"),):
            for file_path in sorted(root.rglob("*")):
                if not file_path.is_file():
                    continue
                try:
                    relative = file_path.resolve().relative_to(root.resolve()).as_posix()
                except ValueError:
                    continue
                attachments.append(
                    Attachment(
                        name=file_path.name,
                        path=f"/mnt/user-data/{scope}/{relative}",
                        mime_type=guess_mime_type(file_path),
                        metadata={"size": file_path.stat().st_size, "scope": scope, "thread_file": True},
                    )
                    )
        return attachments

    @classmethod
    def _expanded_video_record_attachments(cls, attachments: list[Attachment]) -> list[Attachment]:
        expanded: list[Attachment] = list(attachments)
        seen_refs: set[str] = set()
        for attachment in attachments:
            ref = cls._attachment_reference(attachment)
            if ref:
                seen_refs.add(ref)
        changed = False
        for attachment in attachments:
            derived = cls._attachment_record_video(attachment)
            if derived is None:
                continue
            ref = cls._attachment_reference(derived)
            if ref and ref in seen_refs:
                continue
            if ref:
                seen_refs.add(ref)
            expanded.append(derived)
            changed = True
        return expanded if changed else attachments

    @classmethod
    def _attachment_record_video(cls, attachment: Attachment) -> Attachment | None:
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        others = metadata.get("others")
        if not isinstance(others, dict):
            return None
        record = others.get("record")
        if isinstance(record, str):
            uri = record.strip()
            source_meta: dict[str, Any] = {"value": uri}
        elif isinstance(record, dict):
            uri = cls._first_string(record.get("url"), record.get("uri"), record.get("path"), record.get("internalUrl"))
            source_meta = record
        else:
            return None
        if not uri:
            return None
        mime_type = cls._record_video_mime_type(uri, source_meta)
        if not mime_type.startswith("video/"):
            return None
        record_name = cls._record_video_name(uri, source_meta, attachment.name)
        derived_metadata = dict(metadata)
        derived_metadata.update(
            {
                "source": "attachment_record_video",
                "record_source": "metadata.others.record",
                "record_parent_name": attachment.name,
                "record_parent_mime_type": attachment.mime_type,
                "original_uri": uri,
            }
        )
        return Attachment(
            name=record_name,
            path=uri,
            mime_type=mime_type,
            metadata=derived_metadata,
        )

    @staticmethod
    def _attachment_reference(attachment: Attachment) -> str:
        path = str(attachment.path or "").strip()
        if path:
            return f"path:{path}"
        data_base64 = str(attachment.data_base64 or "").strip()
        if data_base64:
            return f"data:{attachment.name}:{attachment.mime_type}:{len(data_base64)}"
        return ""

    @staticmethod
    def _record_video_mime_type(uri: str, metadata: dict[str, Any]) -> str:
        explicit = AgentRuntime._first_string(
            metadata.get("mime_type"),
            metadata.get("mimeType"),
            metadata.get("media_type"),
            metadata.get("mediaType"),
            metadata.get("content_type"),
            metadata.get("contentType"),
        )
        if explicit:
            clean = explicit.split(";", 1)[0].strip().lower()
            if clean:
                return clean
        guessed = guess_mime_type(Path(urlparse(uri).path or "record.mp4"))
        return guessed.split(";", 1)[0].strip().lower() if guessed else ""

    @staticmethod
    def _record_video_name(uri: str, metadata: dict[str, Any], fallback_name: str) -> str:
        explicit = AgentRuntime._first_string(metadata.get("name"), metadata.get("filename"), metadata.get("fileName"))
        if explicit:
            return explicit
        parsed_name = Path(unquote(urlparse(uri).path)).name
        if parsed_name:
            return parsed_name
        fallback = Path(fallback_name or "record").stem or "record"
        return f"{fallback}.mp4"

    @staticmethod
    def _first_string(*values: object) -> str:
        for value in values:
            if isinstance(value, str):
                clean = value.strip()
                if clean:
                    return clean
        return ""

    def _materialize_remote_attachments(self, attachments: list[Attachment], thread_id: str) -> list[Attachment]:
        paths = self.artifact_store.prepare_thread(thread_id)
        materialized: list[Attachment] = []
        changed = False
        for attachment in attachments:
            if not self._should_materialize_remote_attachment(attachment):
                materialized.append(attachment)
                continue
            try:
                updated = self._download_remote_attachment(attachment, paths)
            except Exception as exc:
                logger.warning(
                    "remote attachment download failed thread_id=%s name=%s url=%s error=%s",
                    paths.thread_id,
                    attachment.name,
                    attachment.path,
                    exc,
                )
                metadata = dict(attachment.metadata)
                metadata["download_error"] = self._redacted_remote_attachment_error(str(exc), attachment.path)
                metadata["original_uri"] = metadata.get("uri") or attachment.path
                metadata["remote_download_failed"] = True
                materialized.append(
                    attachment.model_copy(
                        update={
                            "path": attachment.path,
                            "data_base64": None,
                            "metadata": metadata,
                        },
                        deep=True,
                    )
                )
                changed = True
                continue
            materialized.append(updated)
            changed = True
        return materialized if changed else attachments

    @staticmethod
    def _should_materialize_remote_attachment(attachment: Attachment) -> bool:
        raw_path = (attachment.path or "").strip()
        if not raw_path:
            return False
        parsed = urlparse(raw_path)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return False
        mime_type = _attachment_mime_type_for_remote_download(attachment)
        return mime_type.startswith(("image/", "video/"))

    def _download_remote_attachment(self, attachment: Attachment, paths: ThreadPaths) -> Attachment:
        url = str(attachment.path or "").strip()
        with httpx.Client(timeout=REMOTE_ATTACHMENT_TIMEOUT_SECONDS, follow_redirects=True, trust_env=False) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = self._remote_content_length(response.headers.get("content-length"))
                if content_length is not None and content_length > REMOTE_ATTACHMENT_MAX_BYTES:
                    raise ValueError(
                        f"remote attachment is too large: {content_length} bytes > {REMOTE_ATTACHMENT_MAX_BYTES} bytes"
                    )
                header_mime = self._header_mime_type(response.headers.get("content-type"))
                mime_type = header_mime or _attachment_mime_type_for_remote_download(attachment)
                if not self._remote_attachment_mime_allowed(mime_type):
                    raise ValueError(f"remote attachment MIME type is not allowed: {mime_type}")
                filename = self._remote_attachment_filename(url, attachment.name, mime_type)
                target: Path | None = None
                digest = hashlib.sha1()
                bytes_written = 0
                chunks: list[bytes] = []
                try:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if bytes_written > REMOTE_ATTACHMENT_MAX_BYTES:
                            raise ValueError(
                                f"remote attachment is too large: {bytes_written} bytes > {REMOTE_ATTACHMENT_MAX_BYTES} bytes"
                            )
                        digest.update(chunk)
                        chunks.append(chunk)
                    fingerprint = digest.hexdigest()[:12] if bytes_written else uuid4().hex[:12]
                    filename = self._fingerprinted_remote_attachment_filename(filename, fingerprint)
                    target = self._unique_upload_target(paths, filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as handle:
                        for chunk in chunks:
                            handle.write(chunk)
                except Exception:
                    if target is not None:
                        target.unlink(missing_ok=True)
                    raise
        virtual_path = self._virtual_upload_path(paths, target)
        metadata = dict(attachment.metadata)
        metadata.update(
            {
                "original_uri": metadata.get("uri") or url,
                "original_name": attachment.name,
                "uri": virtual_path,
                "downloaded": True,
                "size": target.stat().st_size,
                "sha1": digest.hexdigest() if bytes_written else "",
            }
        )
        logger.info(
            "remote attachment downloaded thread_id=%s original_name=%s name=%s url=%s path=%s size=%s sha1=%s source_id=%s data_id=%s timestamp=%s",
            paths.thread_id,
            attachment.name,
            target.name,
            url,
            virtual_path,
            target.stat().st_size,
            digest.hexdigest() if bytes_written else "",
            metadata.get("sourceId") or metadata.get("source_id"),
            metadata.get("dataId") or metadata.get("data_id"),
            metadata.get("timestamp"),
        )
        return attachment.model_copy(
            update={
                "name": target.name,
                "path": virtual_path,
                "mime_type": mime_type,
                "metadata": metadata,
            },
            deep=True,
        )

    @staticmethod
    def _remote_content_length(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    def _header_mime_type(value: str | None) -> str:
        if value is None:
            return ""
        return value.split(";", 1)[0].strip().lower()

    @staticmethod
    def _remote_attachment_mime_allowed(mime_type: str) -> bool:
        clean = mime_type.split(";", 1)[0].strip().lower()
        return clean.startswith(("image/", "video/"))

    @staticmethod
    def _remote_attachment_filename(url: str, attachment_name: str, mime_type: str) -> str:
        parsed_name = Path(unquote(urlparse(url).path)).name
        raw_name = attachment_name.strip() or parsed_name or "attachment"
        name = Path(raw_name.replace("\\", "/")).name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "attachment"
        if "." not in name:
            extension = mimetypes.guess_extension(mime_type.split(";", 1)[0].strip().lower())
            if extension:
                name = f"{name}{extension}"
        return name[:180]

    @staticmethod
    def _redacted_remote_attachment_error(error: str, raw_url: str | None) -> str:
        redacted = error
        if raw_url:
            clean_url = raw_url.strip()
            if clean_url:
                parsed = urlparse(clean_url)
                safe_url = parsed._replace(query="", fragment="").geturl() if parsed.scheme and parsed.netloc else "<remote-url>"
                redacted = redacted.replace(clean_url, safe_url)
        return redacted[:1000]

    @staticmethod
    def _fingerprinted_remote_attachment_filename(filename: str, fingerprint: str) -> str:
        path = Path(filename)
        stem = path.stem or "attachment"
        suffix = path.suffix
        clean_fingerprint = re.sub(r"[^A-Za-z0-9]+", "", fingerprint)[:12] or uuid4().hex[:12]
        if stem.endswith(f"-{clean_fingerprint}"):
            return filename
        max_stem_len = max(1, 180 - len(suffix) - len(clean_fingerprint) - 1)
        return f"{stem[:max_stem_len]}-{clean_fingerprint}{suffix}"

    @staticmethod
    def _unique_upload_target(paths: ThreadPaths, filename: str) -> Path:
        uploads = paths.uploads.resolve()
        target = (uploads / filename).resolve()
        try:
            target.relative_to(uploads)
        except ValueError as exc:
            raise ValueError("remote attachment upload path traversal blocked") from exc
        if not target.exists():
            return target
        stem = target.stem or "attachment"
        suffix = target.suffix
        for index in range(1, 10_000):
            candidate = (uploads / f"{stem}-{index}{suffix}").resolve()
            try:
                candidate.relative_to(uploads)
            except ValueError as exc:
                raise ValueError("remote attachment upload path traversal blocked") from exc
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not choose a unique upload filename for {filename}")

    @staticmethod
    def _virtual_upload_path(paths: ThreadPaths, target: Path) -> str:
        relative = target.resolve().relative_to(paths.uploads.resolve()).as_posix()
        return f"/mnt/user-data/uploads/{relative}"

    @classmethod
    def _messages_with_attachment_context(
        cls,
        messages: list[Message],
        attachments: list[Attachment],
    ) -> list[Message]:
        if not attachments or not messages:
            return messages
        lines = []
        for index, attachment in enumerate(attachments, start=1):
            metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
            size = metadata.get("size")
            size_text = f", size={size}" if isinstance(size, int | float | str) and str(size) else ""
            path_text = f", path={attachment.path}" if attachment.path else ""
            mime_text = f", mime_type={attachment.mime_type}" if attachment.mime_type else ""
            original_uri = metadata.get("original_uri")
            source_id = metadata.get("sourceId") or metadata.get("source_id")
            data_id = metadata.get("dataId") or metadata.get("data_id")
            timestamp = metadata.get("timestamp")
            sha1 = metadata.get("sha1")
            remote_download_failed = metadata.get("remote_download_failed")
            download_error = metadata.get("download_error")
            metadata_parts = []
            if isinstance(original_uri, str) and original_uri and not remote_download_failed:
                metadata_parts.append(f"original_uri={original_uri}")
            if remote_download_failed:
                metadata_parts.append("remote_download_failed=true")
            if isinstance(download_error, str) and download_error:
                metadata_parts.append(f"download_error={download_error[:200]}")
            if isinstance(source_id, str) and source_id:
                metadata_parts.append(f"sourceId={source_id}")
            if isinstance(data_id, str) and data_id:
                metadata_parts.append(f"dataId={data_id}")
            if isinstance(timestamp, int | float | str) and str(timestamp):
                metadata_parts.append(f"timestamp={timestamp}")
            if isinstance(sha1, str) and sha1:
                metadata_parts.append(f"sha1={sha1[:12]}")
            semantic_values = cls._semantic_values_from_metadata(metadata)
            if semantic_values:
                metadata_parts.append(f"event_semantics={'; '.join(semantic_values[:8])}")
            object_summary = cls._attachment_object_summary(metadata)
            if object_summary:
                metadata_parts.append(f"objects={object_summary}")
            metadata_text = ", " + ", ".join(metadata_parts) if metadata_parts else ""
            lines.append(f"{index}. name={attachment.name}{path_text}{mime_text}{size_text}{metadata_text}")
        if not lines:
            return messages
        context = "\n\nUploaded files available to tools:\n" + "\n".join(lines)
        updated = list(messages)
        for index in range(len(updated) - 1, -1, -1):
            message = updated[index]
            if message.role == "user":
                updated[index] = message.model_copy(update={"content": message.content + context})
                return updated
        return messages

    @classmethod
    def _attachment_object_summary(cls, metadata: dict[str, Any]) -> str:
        objects = metadata.get("objects")
        if not isinstance(objects, list):
            return ""
        parts: list[str] = []
        for item in objects[:12]:
            if not isinstance(item, dict):
                continue
            label = cls._first_string(item.get("label"), item.get("name"), item.get("class"), item.get("type"))
            others = item.get("others")
            if isinstance(others, dict):
                label = label or cls._first_string(others.get("text"), others.get("label"))
            if not label:
                continue
            fields = [f"label={label}"]
            score = item.get("score")
            if isinstance(score, int | float):
                fields.append(f"score={score:.3g}")
            elif isinstance(score, str) and score.strip():
                fields.append(f"score={score.strip()[:24]}")
            box = item.get("box") or item.get("bbox")
            if isinstance(box, list) and box:
                fields.append(f"box={cls._compact_number_list(box[:4])}")
            parts.append("{" + ", ".join(fields) + "}")
        if len(objects) > len(parts):
            remaining = len(objects) - len(parts)
            if remaining > 0:
                parts.append(f"...(+{remaining} objects)")
        return "[" + "; ".join(parts) + "]" if parts else ""

    @staticmethod
    def _compact_number_list(values: list[Any]) -> str:
        rendered: list[str] = []
        for value in values:
            if isinstance(value, int):
                rendered.append(str(value))
            elif isinstance(value, float):
                rendered.append(f"{value:.3g}")
            elif isinstance(value, str) and value.strip():
                rendered.append(value.strip()[:16])
        return "[" + ",".join(rendered) + "]"

    def _effective_runtime_options(self, runtime_options: RuntimeOptions) -> RuntimeOptions:
        updates = self._app_model_runtime_option_updates(runtime_options)
        if updates:
            runtime_options = runtime_options.model_copy(update=updates, deep=True)
        updates = self._app_template_runtime_option_updates(runtime_options)
        if updates:
            runtime_options = runtime_options.model_copy(update=updates, deep=True)
        if self.model_config is None:
            return runtime_options
        updates = self._model_runtime_option_updates(runtime_options)
        if not updates:
            return runtime_options
        return runtime_options.model_copy(update=updates, deep=True)

    def _app_template_runtime_option_updates(self, runtime_options: RuntimeOptions) -> dict[str, object]:
        if "api_key" in runtime_options.model_fields_set or runtime_options.api_key:
            return {}
        app_template_name = (runtime_options.app_template_name or "").strip()
        if not app_template_name:
            return {}
        try:
            template = self.app_template_registry.get(app_template_name)
        except (KeyError, ValueError):
            return {}
        template_options = template.runtime_options if isinstance(template.runtime_options, dict) else {}
        encrypted = self._clean_string(template_options.get("api_key_enc"))
        if not encrypted:
            return {}
        decrypted = self._decrypt_app_runtime_api_key(
            encrypted,
            app_template_name=app_template_name,
            runtime_options=runtime_options,
            template_options=template_options,
        )
        return {"api_key": decrypted} if decrypted else {}

    def _app_model_runtime_option_updates(self, runtime_options: RuntimeOptions) -> dict[str, object]:
        app_template_name = (runtime_options.app_template_name or "").strip()
        if not app_template_name:
            return {}
        try:
            template = self.app_template_registry.get(app_template_name)
        except (KeyError, ValueError):
            return {}
        selected_model = template.select_model(runtime_options.model_type or "chat")
        if selected_model is None:
            return {}
        return self._runtime_option_updates_from_app_model(runtime_options, selected_model, app_template_name)

    def _runtime_option_updates_from_app_model(
        self,
        runtime_options: RuntimeOptions,
        model: AppModelOption,
        app_template_name: str,
    ) -> dict[str, object]:
        explicit = runtime_options.model_fields_set
        force_model_config = runtime_options.config_options.get("force_model_config") is True
        updates: dict[str, object] = {}
        model_name = model.model or model.default_model or model.name
        if model_name and (force_model_config or "model_name" not in explicit):
            updates["model_name"] = model_name
        for option_name in (
            "base_url",
            "api_key",
            "temperature",
            "top_p",
            "max_tokens",
            "request_timeout_seconds",
        ):
            if option_name in explicit and not force_model_config:
                continue
            value = getattr(model, option_name)
            if value is not None:
                updates[option_name] = value
        if (force_model_config or "api_key" not in explicit) and "api_key" not in updates and model.api_key_enc:
            decrypted = self._decrypt_app_model_api_key(model.api_key_enc, app_template_name, model)
            if decrypted:
                updates["api_key"] = decrypted
        if (force_model_config or "api_key" not in explicit) and "api_key" not in updates:
            api_key = self._api_key_from_app_model_env(model)
            if api_key:
                updates["api_key"] = api_key
        logger.info(
            "app model runtime options resolved app_template=%s selected_model=%s force_model_config=%s "
            "explicit_fields=%s update_keys=%s base_url=%s model_name=%s api_key_configured=%s api_key_len=%s "
            "api_key_source=%s",
            app_template_name,
            model.name or model.model or model.default_model or "",
            force_model_config,
            sorted(str(item) for item in explicit),
            sorted(updates),
            updates.get("base_url") or runtime_options.base_url or "",
            updates.get("model_name") or runtime_options.model_name or "",
            bool(updates.get("api_key") or runtime_options.api_key),
            len(str(updates.get("api_key") or runtime_options.api_key or "")),
            self._app_model_api_key_source(model, updates),
        )
        return updates

    @staticmethod
    def _api_key_from_app_model_env(model: AppModelOption) -> str | None:
        env_name = (model.api_key_env or "").strip()
        if not env_name:
            return None
        value = os.getenv(env_name, "").strip()
        return value or None

    @staticmethod
    def _app_model_api_key_source(model: AppModelOption, updates: dict[str, object]) -> str:
        if updates.get("api_key"):
            if model.api_key:
                return "model.api_key"
            if model.api_key_env:
                return f"env:{model.api_key_env}"
            if model.api_key_enc:
                return "model.api_key_enc"
            return "runtime_or_template"
        if model.api_key:
            return "model.api_key_not_applied"
        if model.api_key_env:
            return f"env:{model.api_key_env}:empty_or_not_applied"
        if model.api_key_enc:
            return "model.api_key_enc_not_applied"
        return "none"

    def _decrypt_app_model_api_key(self, encrypted: str, app_template_name: str, model: AppModelOption) -> str | None:
        purposes = [
            f"app:{app_template_name}:model:{model.name or model.model or model.default_model}:api_key",
            f"app:{app_template_name}:model:api_key",
            "agent:default:model:api_key",
        ]
        for purpose in purposes:
            try:
                return self.secret_codec.decrypt(encrypted.strip(), purpose=purpose)
            except ValueError:
                continue
        return None

    def _decrypt_app_runtime_api_key(
        self,
        encrypted: str,
        *,
        app_template_name: str,
        runtime_options: RuntimeOptions,
        template_options: dict[str, object],
    ) -> str | None:
        purposes: list[str] = []
        seen: set[str] = set()
        for model_name in (
            runtime_options.model_name,
            self._clean_string(template_options.get("model_name")),
            self._clean_string(template_options.get("model")),
            self._clean_string(template_options.get("default_model")),
        ):
            clean_model = self._clean_string(model_name)
            if not clean_model or clean_model in seen:
                continue
            seen.add(clean_model)
            purposes.append(f"app:{app_template_name}:model:{clean_model}:api_key")
        purposes.extend(
            [
                f"app:{app_template_name}:runtime_options:api_key",
                f"app:{app_template_name}:model:api_key",
                "agent:default:model:api_key",
            ]
        )
        for purpose in purposes:
            try:
                return self.secret_codec.decrypt(encrypted.strip(), purpose=purpose)
            except ValueError:
                continue
        return None

    def _model_runtime_option_updates(self, runtime_options: RuntimeOptions) -> dict[str, object]:
        if self.model_config is None:
            return {}
        explicit = runtime_options.model_fields_set
        model_config = self.model_config
        updates: dict[str, object] = {}
        model_name = model_config.model if "model" in self._model_config_fields else None
        if model_name is None and "default_model" in self._model_config_fields:
            model_name = model_config.default_model
        if model_name and "model_name" not in explicit:
            updates["model_name"] = model_name
        for option_name, config_name in (
            ("base_url", "base_url"),
            ("api_key", "api_key"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_tokens", "max_tokens"),
            ("request_timeout_seconds", "request_timeout_seconds"),
        ):
            if option_name in explicit or config_name not in self._model_config_fields:
                continue
            value = getattr(model_config, config_name)
            if value is not None:
                updates[option_name] = value
        return updates

    @staticmethod
    def _normalize_model_config(
        model_config: ModelConfig | dict[str, object] | None,
    ) -> tuple[ModelConfig | None, set[str]]:
        if model_config is None:
            return None, set()
        if isinstance(model_config, ModelConfig):
            return model_config, set(model_config.model_fields_set)
        fields = set(model_config)
        return ModelConfig.model_validate(model_config), fields

    @staticmethod
    def _clean_string(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _normalize_skills(
        skills: list[str] | None,
        *,
        root_dir: str | os.PathLike[str] | None = None,
        extra_aliases: object = None,
    ) -> list[str] | None:
        if skills is None:
            return None
        normalized = []
        seen = set()
        for skill in expand_skill_aliases(
            skills,
            Path(root_dir) if root_dir is not None else None,
            extra_aliases=extra_aliases,
        ):
            name = skill.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    @staticmethod
    def _runtime_skill_aliases(runtime_options: RuntimeOptions) -> object:
        return runtime_options.config_options.get("skill_aliases")

    @staticmethod
    def _explicit_workflow_name(request: ChatRequest) -> str | None:
        workflow_name = (request.runtime_options.workflow or "").strip()
        if not workflow_name or workflow_name in {"agent_loop", "default"}:
            return None
        return workflow_name

    @staticmethod
    def _last_user_text(request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content.strip()
        return ""

    @staticmethod
    def _has_generation_context(request: ChatRequest) -> bool:
        return False

    @staticmethod
    def _is_artifact_refinement_request(text: str) -> bool:
        return False

    @staticmethod
    def _is_meta_request(text: str) -> bool:
        normalized = text.lower()
        return any(keyword in normalized for keyword in ("你能干啥", "你能做什么", "what can you do"))

    @staticmethod
    def _event(event: ChatEvent) -> str:
        payload = json.dumps(event.model_dump(), ensure_ascii=False)
        return f"event: {event.type}\ndata: {payload}\n\n"


def _attachment_mime_type_for_remote_download(attachment: Attachment) -> str:
    explicit = (attachment.mime_type or "").split(";", 1)[0].strip().lower()
    if explicit:
        return explicit
    guessed_from_name = guess_mime_type(Path(attachment.name or "attachment"))
    if guessed_from_name != "application/octet-stream":
        return guessed_from_name
    raw_path = str(attachment.path or "").strip()
    parsed_name = Path(unquote(urlparse(raw_path).path)).name
    if parsed_name:
        guessed_from_url = guess_mime_type(Path(parsed_name))
        if guessed_from_url != "application/octet-stream":
            return guessed_from_url
    return guessed_from_name


def _input_type_label(value: object) -> str:
    return {
        "image": "图片",
        "video": "视频",
        "audio": "音频",
        "dataset": "数据集",
        "model": "模型文件",
        "model_config": "模型配置",
    }.get(str(value), "文件")
