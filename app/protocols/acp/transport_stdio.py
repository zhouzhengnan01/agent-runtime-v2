from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

from acp import Agent, Client, PROTOCOL_VERSION, run_agent
from acp import helpers as acp_helpers
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AudioContentBlock,
    CloseSessionResponse,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerStdio,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionForkCapabilities,
    SessionInfo,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    SetSessionModelResponse,
    SseMcpServer,
    TextContentBlock,
)

from app.core.agent import AgentRuntime
from app.core.config import AgentConfig, AgentConfigLoader
from app.core.runtime import ModelManager
from app.protocols.acp.content import prompt_parts_from_sdk_blocks
from app.protocols.acp.session_files import delete_thread_files
from app.schemas import ChatEvent, ChatRequest, Message, RuntimeOptions


PromptBlock = TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock


@dataclass
class AcpStdioSession:
    session_id: str
    thread_id: str
    agent_name: str
    cwd: str
    model_id: str | None = None
    model_name: str | None = None
    runtime_options: dict[str, Any] = field(default_factory=dict)


class JetLinksAcpStdioAgent:
    """Official ACP stdio agent wrapper around the internal AgentRuntime."""

    def __init__(
        self,
        agent_name: str = "default",
        loader: AgentConfigLoader | None = None,
        runtime: AgentRuntime | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.loader = loader or AgentConfigLoader()
        self.runtime = runtime or AgentRuntime()
        self.model_manager = model_manager or ModelManager()
        self.sessions: dict[str, AcpStdioSession] = {}
        self.active_prompt_tasks: dict[str, asyncio.Task[str]] = {}
        self.client: Client | None = None
        self.loader.load(agent_name)

    def on_connect(self, conn: Client) -> None:
        self.client = conn

    async def initialize(
        self,
        protocol_version: int,
        *_: Any,
        **__: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_info=Implementation(
                name="jetlinks-agent-runtime-v2",
                title="JetLinks Agent Runtime v2",
                version="0.1.0",
            ),
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    audio=True,
                    embedded_context=True,
                ),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                    fork=SessionForkCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_: Any,
    ) -> NewSessionResponse:
        del mcp_servers
        session_id = f"acp-session-{uuid4().hex[:12]}"
        self.sessions[session_id] = AcpStdioSession(
            session_id=session_id,
            thread_id=f"acp-{uuid4().hex[:12]}",
            agent_name=self.agent_name,
            cwd=cwd,
        )
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        prompt: list[PromptBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        session = self.sessions.get(session_id)
        if session is None:
            session = AcpStdioSession(
                session_id=session_id,
                thread_id=f"acp-{uuid4().hex[:12]}",
                agent_name=self.agent_name,
                cwd=".",
            )
            self.sessions[session_id] = session

        text, attachments = prompt_parts_from_sdk_blocks(prompt)
        runtime_options = self._prompt_runtime_options(session, kwargs)
        request = ChatRequest(
            messages=[Message(role="user", content=text or " ")],
            attachments=attachments,
            runtime_options=runtime_options,
        )
        agent_config = self.loader.load(session.agent_name)
        task = asyncio.create_task(self._run_prompt_session(session, agent_config, request, message_id))
        previous = self.active_prompt_tasks.get(session.session_id)
        if previous is not None and not previous.done():
            previous.cancel()
        self.active_prompt_tasks[session.session_id] = task
        try:
            result_status = await task
        except asyncio.CancelledError:
            result_status = "cancelled"
        finally:
            if self.active_prompt_tasks.get(session.session_id) is task:
                self.active_prompt_tasks.pop(session.session_id, None)

        if result_status == "completed":
            return PromptResponse(stop_reason="end_turn", user_message_id=message_id)
        if result_status == "cancelled":
            return PromptResponse(stop_reason="cancelled", user_message_id=message_id)
        return PromptResponse(stop_reason="refusal", user_message_id=message_id)

    async def cancel(self, session_id: str, **_: Any) -> None:
        task = self.active_prompt_tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()

    async def _run_prompt_session(
        self,
        session: AcpStdioSession,
        agent_config: AgentConfig,
        request: ChatRequest,
        message_id: str | None,
    ) -> str:
        result_status = "completed"
        agent_message_delta_seen = False
        async for event in self.runtime.iter_events(agent_config, request):
            await self._send_runtime_event(
                session.session_id,
                event,
                message_id,
                suppress_agent_message=agent_message_delta_seen,
            )
            if event.type == "agent.message.delta":
                agent_message_delta_seen = True
            if event.type == "run.failed":
                result_status = "failed"
        return result_status

    async def list_sessions(self, cursor: str | None = None, cwd: str | None = None, **_: Any) -> ListSessionsResponse:
        del cursor
        sessions = [
            SessionInfo(
                session_id=session.session_id,
                cwd=session.cwd,
                title=f"JetLinks ACP {session.agent_name}",
            )
            for session in self.sessions.values()
            if cwd is None or session.cwd == cwd
        ]
        return ListSessionsResponse(sessions=sessions)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_: Any,
    ) -> LoadSessionResponse | None:
        del mcp_servers
        session = self.sessions.get(session_id)
        if session is None:
            return None
        session.cwd = cwd
        return LoadSessionResponse()

    async def close_session(self, session_id: str, **_: Any) -> CloseSessionResponse:
        task = self.active_prompt_tasks.pop(session_id, None)
        if task is not None and not task.done():
            task.cancel()
        self.sessions.pop(session_id, None)
        return CloseSessionResponse()

    async def set_session_model(self, model_id: str, session_id: str, **_: Any) -> SetSessionModelResponse | None:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        session.model_id = model_id
        resolved = self._runtime_options_for_model(model_id)
        if resolved:
            session.runtime_options = _merge_runtime_options(session.runtime_options, resolved)
        else:
            session.runtime_options["model_name"] = model_id
        session.model_name = _string(session.runtime_options.get("model_name")) or model_id
        return SetSessionModelResponse()

    async def set_session_mode(self, mode_id: str, session_id: str, **_: Any) -> SetSessionModeResponse | None:
        if session_id not in self.sessions:
            return None
        del mode_id
        return SetSessionModeResponse()

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **_: Any,
    ) -> SetSessionConfigOptionResponse | None:
        del config_id, session_id, value
        return SetSessionConfigOptionResponse(config_options=[])

    async def authenticate(self, method_id: str, **_: Any) -> AuthenticateResponse:
        del method_id
        return AuthenticateResponse()

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_: Any,
    ) -> ForkSessionResponse:
        del mcp_servers
        source = self.sessions.get(session_id)
        new_session_id = f"acp-session-{uuid4().hex[:12]}"
        self.sessions[new_session_id] = AcpStdioSession(
            session_id=new_session_id,
            thread_id=f"acp-{uuid4().hex[:12]}",
            agent_name=source.agent_name if source is not None else self.agent_name,
            cwd=cwd,
            model_id=source.model_id if source is not None else None,
            model_name=source.model_name if source is not None else None,
            runtime_options=dict(source.runtime_options) if source is not None else {},
        )
        return ForkSessionResponse(session_id=new_session_id)

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_: Any,
    ) -> ResumeSessionResponse:
        del mcp_servers
        session = self.sessions.get(session_id)
        if session is None:
            self.sessions[session_id] = AcpStdioSession(
                session_id=session_id,
                thread_id=f"acp-{uuid4().hex[:12]}",
                agent_name=self.agent_name,
                cwd=cwd,
            )
        else:
            session.cwd = cwd
        return ResumeSessionResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        normalized = method.removeprefix("_").replace("-", "_")
        if normalized in {"jetlinks/session/delete_files", "jetlinks/delete_session_files"}:
            session_id = _required_session_id(params)
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown ACP session: {session_id}")
            result = delete_thread_files(self.runtime.artifact_store, session.thread_id, params)
            result["sessionId"] = session_id
            return result
        return {"method": method, "params": params, "handled": False}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    async def _send_runtime_event(
        self,
        session_id: str,
        event: ChatEvent,
        message_id: str | None,
        *,
        suppress_agent_message: bool = False,
    ) -> None:
        if self.client is None:
            return
        for update in _event_to_sdk_updates(event, message_id, suppress_agent_message=suppress_agent_message):
            await self.client.session_update(session_id=session_id, update=update)

    def _prompt_runtime_options(self, session: AcpStdioSession, kwargs: dict[str, Any]) -> RuntimeOptions:
        merged = dict(session.runtime_options)
        if session.model_id is not None:
            resolved = self._runtime_options_for_model(session.model_id)
            if resolved:
                merged.update(resolved)
            elif session.model_name is not None:
                merged["model_name"] = session.model_name
        elif session.model_name is not None:
            merged["model_name"] = session.model_name
        override = _runtime_options_from_kwargs(kwargs)
        requested_model = _string(override.get("model_name"))
        merged.update(override)
        if requested_model is not None:
            resolved = self._runtime_options_for_model(requested_model)
            if resolved:
                merged.update(resolved)
        merged["thread_id"] = session.thread_id
        return RuntimeOptions.model_validate(merged)

    def _runtime_options_for_model(self, model_id: str) -> dict[str, Any]:
        options = self.model_manager.runtime_options_for(model_id)
        if options is None:
            return {}
        return options.model_dump(mode="python", exclude_none=True)


async def run_acp_stdio_agent(agent_name: str = "default") -> None:
    await run_agent(cast(Agent, JetLinksAcpStdioAgent(agent_name=agent_name)), use_unstable_protocol=True)


def _prompt_text(prompt: list[PromptBlock]) -> str:
    text, _attachments = prompt_parts_from_sdk_blocks(prompt)
    return text or " "


def _workflow_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    runtime_options = _params(kwargs.get("runtimeOptions") or kwargs.get("runtime_options"))
    jetlinks_meta = _params(kwargs.get("jetlinks"))
    return _string(
        runtime_options.get("workflow")
        or kwargs.get("workflow")
        or jetlinks_meta.get("workflow")
    )


def _runtime_options_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    options = {
        "workflow": _workflow_from_kwargs(kwargs),
        "selected_skills": _string_list_from_kwargs(kwargs, "selectedSkills", "selected_skills"),
        "selected_mcp_tools": _string_list_from_kwargs(kwargs, "selectedMcpTools", "selected_mcp_tools"),
    }
    runtime_options = _params(kwargs.get("runtimeOptions") or kwargs.get("runtime_options"))
    jetlinks_meta = _params(kwargs.get("jetlinks"))
    aliases = {
        "modelName": "model_name",
        "modelId": "model_name",
        "model_name": "model_name",
        "baseUrl": "base_url",
        "base_url": "base_url",
        "apiKey": "api_key",
        "api_key": "api_key",
        "temperature": "temperature",
        "topP": "top_p",
        "top_p": "top_p",
        "maxTokens": "max_tokens",
        "max_tokens": "max_tokens",
        "requestTimeoutSeconds": "request_timeout_seconds",
        "request_timeout_seconds": "request_timeout_seconds",
    }
    for source in (runtime_options, kwargs, jetlinks_meta):
        for raw_name, field_name in aliases.items():
            if raw_name in source:
                options[field_name] = source[raw_name]
    return {key: value for key, value in options.items() if value is not None and value != []}


def _merge_runtime_options(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {**base, **override}
    if not merged:
        return {}
    return RuntimeOptions.model_validate(merged).model_dump(mode="python", exclude_none=True)


def _string_list_from_kwargs(kwargs: dict[str, Any], camel_name: str, snake_name: str) -> list[str]:
    runtime_options = _params(kwargs.get("runtimeOptions") or kwargs.get("runtime_options"))
    jetlinks_meta = _params(kwargs.get("jetlinks"))
    raw_value = (
        runtime_options.get(camel_name)
        or runtime_options.get(snake_name)
        or kwargs.get(camel_name)
        or kwargs.get(snake_name)
        or jetlinks_meta.get(camel_name)
        or jetlinks_meta.get(snake_name)
    )
    if not isinstance(raw_value, list):
        return []
    return [item.strip() for item in raw_value if isinstance(item, str) and item.strip()]


def _event_to_sdk_updates(event: ChatEvent, message_id: str | None, *, suppress_agent_message: bool = False) -> list[Any]:
    data = event.data
    if event.type == "agent.message.delta":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        update = acp_helpers.update_agent_message_text(text)
        update.field_meta = _event_meta(event)
        return [update]
    if event.type == "agent.message":
        if suppress_agent_message:
            return []
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        update = acp_helpers.update_agent_message_text(text)
        update.message_id = message_id
        update.field_meta = _event_meta(event)
        return [update]
    if event.type == "skill.started":
        skill_name = _skill_name(data) or "skill"
        update = acp_helpers.start_tool_call(
            _skill_tool_call_id(data, skill_name),
            skill_name,
            kind="other",
            status="in_progress",
        )
        update.field_meta = _event_meta(event)
        return [update]
    if event.type == "tool.started":
        tool_name = _string(data.get("tool_name")) or "tool"
        update = acp_helpers.start_tool_call(
            _tool_call_id(_string(data.get("tool_call_id")) or tool_name),
            tool_name,
            kind="other",
            status="in_progress",
        )
        update.field_meta = _event_meta(event)
        return [update]
    if event.type == "tool.completed":
        return [
            _tool_call_update(
                event,
                _tool_call_id(_string(data.get("tool_call_id")) or _string(data.get("tool_name")) or "tool"),
                title=_string(data.get("tool_name")) or "tool",
                status="completed",
                raw_output=data.get("structured_content"),
            )
        ]
    if event.type == "tool.failed":
        return [
            _tool_call_update(
                event,
                _tool_call_id(_string(data.get("tool_call_id")) or _string(data.get("tool_name")) or "tool"),
                title=_string(data.get("tool_name")) or "tool",
                status="failed",
                raw_output=data.get("structured_content"),
            )
        ]
    if event.type == "skill.completed":
        skill_name = _skill_name(data) or "skill"
        return [
            _tool_call_update(
                event,
                _skill_tool_call_id(data, skill_name),
                title=skill_name,
                status="completed",
                raw_output=_runtime_event_raw_output(event),
            )
        ]
    if event.type == "verifier.started":
        update = acp_helpers.start_tool_call(
            "runtime-verifier",
            "verifier",
            kind="other",
            status="in_progress",
        )
        update.field_meta = _event_meta(event)
        return [update]
    if event.type == "verifier.completed":
        return [
            _tool_call_update(
                event,
                "runtime-verifier",
                title="verifier",
                status="completed",
                raw_output=_runtime_event_raw_output(event),
            )
        ]
    if event.type == "run.failed":
        return [_thought_update(event)]
    return [_thought_update(event)]


def _thought_update(event: ChatEvent) -> Any:
    update = acp_helpers.update_agent_thought_text(_runtime_event_summary(event))
    update.field_meta = _event_meta(event)
    return update


def _tool_call_update(
    event: ChatEvent,
    tool_call_id: str,
    *,
    title: str | None = None,
    status: str | None = None,
    raw_output: Any | None = None,
) -> Any:
    update = acp_helpers.update_tool_call(
        tool_call_id,
        title=title,
        kind="other",
        status=status,
        raw_output=raw_output,
    )
    update.field_meta = _event_meta(event)
    return update


def _event_meta(event: ChatEvent) -> dict[str, Any]:
    return {"jetlinksRuntimeEvent": event.model_dump()}


def _runtime_event_summary(event: ChatEvent) -> str:
    data = event.data
    if event.type in {"tool.completed", "tool.failed"}:
        tool_name = _string(data.get("tool_name")) or "tool"
        status = "failed" if event.type == "tool.failed" else "completed"
        return f"{tool_name}: {status}"
    if event.type == "run.started":
        workflow = _string(data.get("workflow"))
        return f"run started: {workflow}" if workflow else "run started"
    if event.type == "run.completed":
        result = _params(data.get("result"))
        return f"run completed: {_string(result.get('status')) or 'completed'}"
    if event.type == "run.failed":
        return f"run failed: {_string(data.get('error')) or 'failed'}"
    if event.type == "llm.started":
        return f"llm started: {_string(data.get('model')) or 'model'}"
    if event.type == "tools.available":
        tool_count = data.get("tool_count")
        return f"tools available: {tool_count}" if isinstance(tool_count, int) else "tools available"
    if event.type == "tool.calls.started":
        tool_calls = data.get("tool_calls")
        count = len(tool_calls) if isinstance(tool_calls, list) else 0
        return f"tool calls started: {count}"
    if event.type == "skill.selected":
        skill = _params(data.get("skill"))
        return f"skill selected: {_string(skill.get('name')) or _string(data.get('skill_name')) or 'skill'}"
    if event.type == "skill.started":
        return f"skill started: {_string(data.get('skill_name')) or 'skill'}"
    if event.type == "artifact.created":
        artifact = _params(data.get("artifact"))
        return f"artifact: {_string(artifact.get('name')) or 'created'}"
    if event.type == "preview.ready":
        artifact = _params(data.get("artifact"))
        return f"preview ready: {_string(artifact.get('name')) or 'artifact'}"
    if event.type == "verifier.started":
        return "verification: started"
    if event.type == "verifier.completed":
        verification = _params(data.get("verification"))
        return "verification: passed" if verification.get("passed") is True else "verification: failed"
    if event.type == "skill.completed":
        return f"skill completed: {_string(data.get('skill_name')) or ''}".strip()
    return event.type


def _runtime_event_raw_output(event: ChatEvent) -> object:
    data = event.data
    if event.type == "artifact.created":
        return data.get("artifact")
    if event.type == "verifier.completed":
        return data.get("verification")
    if event.type == "run.failed":
        return data.get("result") or data.get("error")
    if event.type == "skill.completed":
        return data.get("data") or {"output_count": data.get("output_count")}
    return data


def _skill_name(data: dict[str, Any]) -> str | None:
    return _string(data.get("skill_name")) or _string(_params(data.get("skill")).get("name"))


def _skill_tool_call_id(data: dict[str, Any], skill_name: str) -> str:
    attempt = data.get("attempt")
    suffix = f"{skill_name}-{attempt}" if isinstance(attempt, int) else skill_name
    return _tool_call_id(f"skill-{suffix}")


def _tool_call_id(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return f"runtime-{normalized or 'tool'}"


def _params(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _required_session_id(params: dict[str, Any]) -> str:
    session_id = _string(params.get("sessionId") or params.get("session_id"))
    if session_id is None:
        raise ValueError("sessionId is required")
    return session_id
