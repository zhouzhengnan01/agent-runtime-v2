from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

from app.core.artifacts import ArtifactStore
from app.core.agent.session import SessionConversationStore
from app.core.agent.tool_loop import ToolCallingAgentLoop
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.events import EventRecorder, RunEventStore
from app.core.llm import OpenAICompatibleClient
from app.core.memory import MarkdownMemoryStore, MemoryStore
from app.core.tools import ToolInvocationService
from app.core.workflow import WorkflowRegistry
from app.schemas import AgentRunResult, ChatEvent, ChatRequest, Message, RuntimeOptions


class AgentRuntime:
    """Stateless runtime shared by HTTP and CLI entrypoints."""

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        workflow_router: object | None = None,
        workflow_registry: WorkflowRegistry | None = None,
        memory_store: MemoryStore | None = None,
        markdown_memory_store: MarkdownMemoryStore | None = None,
        run_event_store: RunEventStore | None = None,
        session_store: SessionConversationStore | None = None,
        model_config: ModelConfig | dict[str, object] | None = None,
        skills: list[str] | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.memory_store = memory_store or MemoryStore()
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()
        self.run_event_store = run_event_store or RunEventStore()
        self.session_store = session_store or SessionConversationStore()
        self.workflow_registry = workflow_registry or WorkflowRegistry.builtin(self.artifact_store)
        self.workflow_router = workflow_router
        self.model_config, self._model_config_fields = self._normalize_model_config(model_config)
        self.skills = self._normalize_skills(skills)
        self.agent_loop = ToolCallingAgentLoop(
            ToolInvocationService(
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
        agent_config, request = self._prepare_execution(agent_config, request)
        if not request.messages:
            raise ValueError("messages must not be empty")

        workflow_name = self._selected_workflow_name(agent_config, request)
        if workflow_name is not None:
            workflow = self.workflow_registry.get(workflow_name)
            if workflow is None:
                raise ValueError(f"Workflow is not registered: {workflow_name}")
            paths = self.artifact_store.prepare_thread(request.runtime_options.thread_id)
            conversation = self.session_store.merge(self.session_store.load(paths), request.messages)
            result, events = workflow.run_with_events(
                agent_config=agent_config,
                messages=self._workflow_messages(conversation),
                attachments=request.attachments,
                thread_id=paths.thread_id,
                workflow_name=workflow_name,
                runtime_options=request.runtime_options,
            )
            self.session_store.save(paths, self._conversation_with_result(conversation, result), run_id=self._run_id(events))
            self._persist_events(agent_config, request, result.thread_id, events, result)
            return result, events

        paths = self.artifact_store.prepare_thread(request.runtime_options.thread_id)
        conversation = self.session_store.merge(self.session_store.load(paths), request.messages)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id)
        recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": "agent_loop",
                "execution_mode": "agent_loop",
                "stateless": agent_config.runtime.stateless,
            },
        )
        loop_result = await self.agent_loop.run(
            agent_config=agent_config,
            messages=conversation,
            thread_id=paths.thread_id,
            recorder=recorder,
            runtime_options=request.runtime_options,
        )
        self.session_store.save(paths, loop_result.messages, run_id=recorder.run_id)
        self._persist_events(agent_config, request, paths.thread_id, recorder.events, loop_result.result)
        return loop_result.result, recorder.events

    async def stream(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[str]:
        try:
            async for event in self.iter_events(agent_config, request):
                yield self._event(event)
        except Exception as exc:
            yield self._event(ChatEvent(type="run.failed", data={"agent": agent_config.name, "error": str(exc)}))

    async def iter_events(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        agent_config, request = self._prepare_execution(agent_config, request)
        if not request.messages:
            raise ValueError("messages must not be empty")

        workflow_name = self._selected_workflow_name(agent_config, request)
        if workflow_name is not None:
            if self.workflow_registry.get(workflow_name) is None:
                raise ValueError(f"Workflow is not registered: {workflow_name}")
            async for event in self._stream_workflow_events(agent_config, request, workflow_name):
                yield event
            return

        async for event in self._stream_agent_loop_events(agent_config, request):
            yield event

    async def _stream_workflow_events(
        self, agent_config: AgentConfig, request: ChatRequest, workflow_name: str
    ) -> AsyncIterator[ChatEvent]:
        paths = self.artifact_store.prepare_thread(request.runtime_options.thread_id)
        conversation = self.session_store.merge(self.session_store.load(paths), request.messages)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatEvent | BaseException | None] = asyncio.Queue()
        captured_events: list[ChatEvent] = []
        final_result: AgentRunResult | None = None

        def on_event(event: ChatEvent) -> None:
            captured_events.append(event)
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_workflow() -> None:
            nonlocal final_result, captured_events
            workflow = self.workflow_registry.get(workflow_name)
            if workflow is None:
                raise ValueError(f"Workflow is not registered: {workflow_name}")
            try:
                final_result, returned_events = workflow.run_with_events(
                    agent_config=agent_config,
                    messages=self._workflow_messages(conversation),
                    attachments=request.attachments,
                    thread_id=paths.thread_id,
                    on_event=on_event,
                    workflow_name=workflow_name,
                    runtime_options=request.runtime_options,
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
                    yield ChatEvent(type="run.failed", data={"agent": agent_config.name, "error": str(item)})
                    break
                yield item
        finally:
            await task
            if captured_events:
                thread_id = (
                    final_result.thread_id
                    if final_result is not None
                    else str(captured_events[0].data.get("thread_id") or request.runtime_options.thread_id or "")
                )
                if final_result is not None:
                    self.session_store.save(
                        paths,
                        self._conversation_with_result(conversation, final_result),
                        run_id=self._run_id(captured_events),
                    )
                self._persist_events(agent_config, request, thread_id, captured_events, final_result)

    async def _stream_agent_loop_events(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        paths = self.artifact_store.prepare_thread(request.runtime_options.thread_id)
        conversation = self.session_store.merge(self.session_store.load(paths), request.messages)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id)
        yield recorder.emit(
            "run.started",
            {
                "run_id": recorder.run_id,
                "workflow": "agent_loop",
                "execution_mode": "agent_loop",
                "stateless": agent_config.runtime.stateless,
            },
        )

        llm = OpenAICompatibleClient(agent_config, runtime_options=request.runtime_options)
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
            async for chunk in llm.stream_complete(agent_config.prompts.system, conversation):
                chunks.append(chunk)
                yield recorder.emit("agent.message.delta", {"text": chunk})

            reply = "".join(chunks)
            final_messages = [*conversation, {"role": "assistant", "content": reply}]
            yield recorder.emit("agent.message", {"text": reply})
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
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
            yield recorder.emit("run.completed", {"result": result.model_dump()})
            self.session_store.save(paths, final_messages, run_id=recorder.run_id)
            self._persist_events(agent_config, request, paths.thread_id, recorder.events, result)
            return

        emitted = 1
        loop_result = await self.agent_loop.run(
            agent_config=agent_config,
            messages=conversation,
            thread_id=paths.thread_id,
            recorder=recorder,
            runtime_options=request.runtime_options,
            emit_message_delta=True,
        )
        for event in recorder.events[emitted:]:
            yield event
        self.session_store.save(paths, loop_result.messages, run_id=recorder.run_id)
        self._persist_events(agent_config, request, paths.thread_id, recorder.events, loop_result.result)

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
                messages.append(Message(role=role, content=content))
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
        return self._effective_agent_config(agent_config), self._effective_request(request)

    def _effective_agent_config(self, agent_config: AgentConfig) -> AgentConfig:
        updates: dict[str, object] = {}
        if self.skills is not None:
            updates["skills"] = list(self.skills)
        if not updates:
            return agent_config
        return agent_config.model_copy(update=updates, deep=True)

    def _effective_request(self, request: ChatRequest) -> ChatRequest:
        runtime_options = self._effective_runtime_options(request.runtime_options)
        if runtime_options is request.runtime_options:
            return request
        return request.model_copy(update={"runtime_options": runtime_options}, deep=True)

    def _effective_runtime_options(self, runtime_options: RuntimeOptions) -> RuntimeOptions:
        if self.model_config is None:
            return runtime_options
        updates = self._model_runtime_option_updates(runtime_options)
        if not updates:
            return runtime_options
        data = runtime_options.model_dump(mode="python")
        data.update(updates)
        return RuntimeOptions.model_validate(data)

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
            ("model_env", "model_env"),
            ("base_url_env", "base_url_env"),
            ("api_key_env", "api_key_env"),
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
    def _normalize_skills(skills: list[str] | None) -> list[str] | None:
        if skills is None:
            return None
        normalized = []
        seen = set()
        for skill in skills:
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
