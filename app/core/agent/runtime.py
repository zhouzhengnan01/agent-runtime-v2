from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from app.core.artifacts import ArtifactStore
from app.core.agent.tool_loop import ToolCallingAgentLoop
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm import OpenAICompatibleClient
from app.core.memory import MarkdownMemoryStore, MemoryStore
from app.core.routing import WorkflowRouter
from app.core.tools import ToolInvocationService
from app.core.workflow import WorkflowRegistry
from app.schemas import AgentRunResult, ChatEvent, ChatRequest


class AgentRuntime:
    """Stateless runtime shared by HTTP and CLI entrypoints."""

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        workflow_router: WorkflowRouter | None = None,
        workflow_registry: WorkflowRegistry | None = None,
        memory_store: MemoryStore | None = None,
        markdown_memory_store: MarkdownMemoryStore | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.memory_store = memory_store or MemoryStore()
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()
        self.workflow_registry = workflow_registry or WorkflowRegistry.builtin(self.artifact_store)
        self.workflow_router = workflow_router or WorkflowRouter(available_workflows=self.workflow_registry.names())
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
        if not request.messages:
            raise ValueError("messages must not be empty")

        workflow_name = self._selected_workflow_name(agent_config, request)
        if workflow_name is not None:
            workflow = self.workflow_registry.get(workflow_name)
            if workflow is None:
                raise ValueError(f"Workflow is not registered: {workflow_name}")
            return workflow.run_with_events(
                agent_config=agent_config,
                messages=request.messages,
                attachments=request.attachments,
                thread_id=request.runtime_options.thread_id,
                workflow_name=workflow_name,
                runtime_options=request.runtime_options,
            )

        paths = self.artifact_store.prepare_thread(request.runtime_options.thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id)
        recorder.emit("run.started", {"workflow": "agent_loop", "stateless": agent_config.runtime.stateless})
        loop_result = await self.agent_loop.run(
            agent_config=agent_config,
            messages=request.messages,
            thread_id=paths.thread_id,
            recorder=recorder,
            runtime_options=request.runtime_options,
        )
        return loop_result.result, recorder.events

    async def stream(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[str]:
        try:
            async for event in self.iter_events(agent_config, request):
                yield self._event(event)
        except Exception as exc:
            yield self._event(ChatEvent(type="run.failed", data={"agent": agent_config.name, "error": str(exc)}))

    async def iter_events(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[ChatEvent]:
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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[ChatEvent | BaseException | None] = asyncio.Queue()

        def on_event(event: ChatEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def run_workflow() -> None:
            workflow = self.workflow_registry.get(workflow_name)
            if workflow is None:
                raise ValueError(f"Workflow is not registered: {workflow_name}")
            try:
                workflow.run_with_events(
                    agent_config=agent_config,
                    messages=request.messages,
                    attachments=request.attachments,
                    thread_id=request.runtime_options.thread_id,
                    on_event=on_event,
                    workflow_name=workflow_name,
                    runtime_options=request.runtime_options,
                )
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

    async def _stream_agent_loop_events(self, agent_config: AgentConfig, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        paths = self.artifact_store.prepare_thread(request.runtime_options.thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id)
        yield recorder.emit("run.started", {"workflow": "agent_loop", "stateless": agent_config.runtime.stateless})

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
            async for chunk in llm.stream_complete(agent_config.prompts.system, request.messages):
                chunks.append(chunk)
                yield recorder.emit("agent.message.delta", {"text": chunk})

            reply = "".join(chunks)
            yield recorder.emit("agent.message", {"text": reply})
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                reply=reply,
                metadata={
                    "workflow": "agent_loop",
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
            return

        emitted = 1
        await self.agent_loop.run(
            agent_config=agent_config,
            messages=request.messages,
            thread_id=paths.thread_id,
            recorder=recorder,
            runtime_options=request.runtime_options,
            emit_message_delta=True,
        )
        for event in recorder.events[emitted:]:
            yield event

    def _should_use_workflow(self, agent_config: AgentConfig, request: ChatRequest) -> bool:
        return self._selected_workflow_name(agent_config, request) is not None

    def _selected_workflow_name(self, agent_config: AgentConfig, request: ChatRequest) -> str | None:
        return self._explicit_workflow_name(request) or self._workflow_for_selected_skills(
            agent_config,
            request,
        )

    @staticmethod
    def _explicit_workflow_name(request: ChatRequest) -> str | None:
        workflow_name = (request.runtime_options.workflow or "").strip()
        if not workflow_name or workflow_name == "agent_loop":
            return None
        return workflow_name

    @staticmethod
    def _workflow_for_selected_skills(agent_config: AgentConfig, request: ChatRequest) -> str | None:
        selected_skills = [name.strip() for name in request.runtime_options.selected_skills if name.strip()]
        if not selected_skills:
            return None
        skill_registry = WorkflowRouter().skill_registry
        for skill_name in selected_skills:
            if agent_config.skills and skill_name not in agent_config.skills:
                continue
            try:
                skill = skill_registry.get(skill_name)
            except KeyError:
                continue
            if skill.runner_path is None:
                continue
            if skill.generation:
                workflow_name = agent_config.workflows.get("generation", "artifact_workflow")
            else:
                workflow_name = agent_config.workflows.get("vision_behavior", "evidence_first_detection")
            if workflow_name and workflow_name != "agent_loop":
                return workflow_name
        return None

    @staticmethod
    def _last_user_text(request: ChatRequest) -> str:
        return WorkflowRouter.last_user_text(request)

    @staticmethod
    def _has_generation_context(request: ChatRequest) -> bool:
        return WorkflowRouter().has_generation_context(request)

    @staticmethod
    def _is_artifact_refinement_request(text: str) -> bool:
        return WorkflowRouter().is_artifact_refinement_request(text)

    @staticmethod
    def _is_meta_request(text: str) -> bool:
        return WorkflowRouter().is_meta_request(text)

    @staticmethod
    def _event(event: ChatEvent) -> str:
        payload = json.dumps(event.model_dump(), ensure_ascii=False)
        return f"event: {event.type}\ndata: {payload}\n\n"
