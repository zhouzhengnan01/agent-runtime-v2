from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
from collections.abc import AsyncIterator
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


@dataclass(frozen=True)
class _DirectJsonArtifactIntent:
    filename: str
    content: str
    reason: str


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
            # Workflow path: a named workflow owns the full execution instead of
            # the generic tool-calling loop.
            return await asyncio.to_thread(self._run_workflow_with_events_sync, execution)

        # Default path: merge thread history and let the agent loop decide when
        # to answer directly versus when to call tools.
        conversation = self._conversation_with_vision_attachments(execution.conversation, execution.request.attachments, execution.paths)
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
            async for event in self._stream_workflow_events(execution):
                yield event
            return

        async for event in self._stream_agent_loop_events(execution):
            yield event

    async def _stream_workflow_events(self, execution: ExecutionContext) -> AsyncIterator[ChatEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatEvent | BaseException | None] = asyncio.Queue()
        captured_events: list[ChatEvent] = []
        final_result: AgentRunResult | None = None

        def on_event(event: ChatEvent) -> None:
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
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(run_workflow))
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
        self.session_store.save(
            execution.paths,
            self._conversation_with_result(execution.conversation, result),
            run_id=self._run_id(events),
        )
        self._enrich_required_inputs(result, execution.request)
        self._replace_final_result_event(events, result)
        self._persist_events(execution.agent_config, execution.request, result.thread_id, events, result)
        return result, events

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
        logger.info(
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
        runtime_options = self._effective_runtime_options(runtime_options)
        runtime_options = self._runtime_options_with_composite_skills(runtime_options)
        attachments = request.attachments
        if not attachments and runtime_options.thread_id:
            attachments = self._thread_file_attachments(runtime_options.thread_id)
        if attachments and runtime_options.thread_id:
            attachments = self._materialize_remote_attachments(attachments, runtime_options.thread_id)
        messages = self._messages_with_attachment_context(request.messages, attachments)
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
        if not selected_skills:
            selected_skills = self._selected_skills_from_routing(messages)
        if not selected_skills:
            return runtime_options
        # Preserve the original explicit-field set so downstream model
        # resolution can still tell which runtime options were truly provided
        # by the caller versus which ones are still eligible for app-template
        # or bootstrap defaults.
        return runtime_options.model_copy(update={"selected_skills": selected_skills}, deep=True)

    def _runtime_options_with_composite_skills(self, runtime_options: RuntimeOptions) -> RuntimeOptions:
        raw_selected_skills = self._normalize_skills(runtime_options.selected_skills) or []
        selected_skills = self._normalize_skills(
            runtime_options.selected_skills,
            root_dir=self.app_template_registry.root_dir,
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

    @staticmethod
    def _append_unique(target: list[str], seen: set[str], value: str) -> None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        target.append(normalized)
        seen.add(normalized)

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
        for root, scope in ((paths.uploads, "uploads"), (paths.outputs, "outputs")):
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
                            "path": None,
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
        with httpx.Client(timeout=REMOTE_ATTACHMENT_TIMEOUT_SECONDS, follow_redirects=True) as client:
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
    def _normalize_skills(skills: list[str] | None, *, root_dir: str | os.PathLike[str] | None = None) -> list[str] | None:
        if skills is None:
            return None
        normalized = []
        seen = set()
        for skill in expand_skill_aliases(skills, Path(root_dir) if root_dir is not None else None):
            name = skill.strip()
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

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
