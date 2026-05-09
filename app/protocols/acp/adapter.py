from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.agent import AgentRuntime
from app.core.apps import AppTemplate, AppTemplateRegistry
from app.core.config import AgentConfigLoader
from app.core.runtime import ModelManager
from app.protocols.acp.content import prompt_parts_from_dict_blocks
from app.protocols.acp.external_backend import ExternalAcpSession, prompt_blocks_from_params, prompt_response_payload
from app.protocols.acp.schemas import AcpWebSocketSession
from app.protocols.acp.session_files import delete_thread_files
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


AcpUpdateSender = Callable[[str, dict[str, Any]], Awaitable[None]]

_RUNTIME_OPTION_ALIASES = {
    "threadId": "thread_id",
    "thread_id": "thread_id",
    "userId": "user_id",
    "user_id": "user_id",
    "projectId": "project_id",
    "project_id": "project_id",
    "workflow": "workflow",
    "selectedSkills": "selected_skills",
    "selected_skills": "selected_skills",
    "selectedMcpTools": "selected_mcp_tools",
    "selected_mcp_tools": "selected_mcp_tools",
    "modelId": "model_name",
    "model_id": "model_name",
    "modelName": "model_name",
    "model_name": "model_name",
    "modelEnv": "model_env",
    "model_env": "model_env",
    "baseUrl": "base_url",
    "base_url": "base_url",
    "baseUrlEnv": "base_url_env",
    "base_url_env": "base_url_env",
    "apiKey": "api_key",
    "api_key": "api_key",
    "apiKeyEnv": "api_key_env",
    "api_key_env": "api_key_env",
    "temperature": "temperature",
    "topP": "top_p",
    "top_p": "top_p",
    "maxTokens": "max_tokens",
    "max_tokens": "max_tokens",
    "requestTimeoutSeconds": "request_timeout_seconds",
    "request_timeout_seconds": "request_timeout_seconds",
    "responseFormat": "response_format",
    "response_format": "response_format",
}

_RUNTIME_OPTION_RESPONSE_ALIASES = {
    "thread_id": "threadId",
    "user_id": "userId",
    "project_id": "projectId",
    "selected_skills": "selectedSkills",
    "selected_mcp_tools": "selectedMcpTools",
    "model_name": "modelName",
    "model_env": "modelEnv",
    "base_url": "baseUrl",
    "base_url_env": "baseUrlEnv",
    "api_key": "apiKey",
    "api_key_env": "apiKeyEnv",
    "top_p": "topP",
    "max_tokens": "maxTokens",
    "request_timeout_seconds": "requestTimeoutSeconds",
    "response_format": "responseFormat",
}


class AcpRuntimeAdapter:
    """Translate ACP-shaped requests into the internal AgentRuntime contract."""

    def __init__(
        self,
        loader: AgentConfigLoader | None = None,
        runtime: AgentRuntime | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.loader = loader or AgentConfigLoader()
        self.runtime = runtime or AgentRuntime()
        self.model_manager = model_manager or ModelManager()
        self.app_registry = AppTemplateRegistry(self.loader.root_dir)

    def initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": 1,
            "agentInfo": {
                "name": "jetlinks-agent-runtime-v2",
                "version": "0.1.0",
            },
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {
                    "image": True,
                    "audio": True,
                    "embeddedContext": True,
                },
                "sessionCapabilities": {
                    "list": True,
                    "close": True,
                    "fork": True,
                    "resume": True,
                    "cancel": True,
                },
                "transports": ["websocket"],
            },
            "_meta": {
                "jetlinks": {
                    "modelManagement": "server",
                    "runtimeOptions": sorted(set(_RUNTIME_OPTION_RESPONSE_ALIASES.values()) | {"workflow", "temperature"}),
                    "artifactWorkspace": True,
                }
            },
        }

    def new_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        app_template = self._app_template_from_params(params)
        session_runtime_options = _new_session_runtime_options(params, app_template)
        agent_name = (
            _string(params.get("agentName") or params.get("agent_name") or params.get("agent"))
            or (app_template.agent_name if app_template is not None else None)
            or "default"
        )
        agent_config = self.loader.load(agent_name)
        thread_id = (
            _string(session_runtime_options.get("thread_id"))
            or _string(params.get("threadId") or params.get("thread_id"))
            or f"acp-{uuid4().hex[:12]}"
        )
        session_id = _string(params.get("sessionId") or params.get("session_id")) or f"acp-session-{uuid4().hex[:12]}"
        cwd = _string(params.get("cwd")) or str(Path.cwd())
        stored_runtime_options = _without_thread_id(session_runtime_options)
        model_id = _string(stored_runtime_options.get("model_name")) or self._current_model_id()
        if model_id is not None:
            stored_runtime_options["model_name"] = model_id
        stored_runtime_options = self._resolve_model_runtime_options(stored_runtime_options)
        model_name = _string(stored_runtime_options.get("model_name"))
        sessions[session_id] = AcpWebSocketSession(
            session_id=session_id,
            thread_id=thread_id,
            agent_name=agent_name,
            cwd=cwd,
            backend_type=agent_config.backend.type,
            model_id=model_id,
            model_name=model_name,
            app_template_name=app_template.name if app_template is not None else None,
            runtime_options=stored_runtime_options,
        )
        return {
            "sessionId": session_id,
            "threadId": thread_id,
            "agentName": agent_name,
            "cwd": cwd,
            "backend": {"type": agent_config.backend.type},
            "appTemplateName": app_template.name if app_template is not None else None,
            "runtimeOptions": _runtime_options_response(stored_runtime_options),
            "models": {
                "currentModelId": model_id,
                "currentModelName": model_name,
                "availableModels": self.model_manager.list_public_payloads(),
            },
        }

    async def prompt(
        self,
        sessions: dict[str, AcpWebSocketSession],
        params: dict[str, Any],
        send_update: AcpUpdateSender,
    ) -> dict[str, Any]:
        session_id = _string(params.get("sessionId") or params.get("session_id"))
        if not session_id or session_id not in sessions:
            session_result = self.new_session(sessions, params)
            session_id = str(session_result["sessionId"])

        session = sessions[session_id]
        agent_name = _string(params.get("agentName") or params.get("agent_name") or params.get("agent")) or session.agent_name
        agent = self.loader.load(agent_name)
        thread_id = _resolve_thread_id(params, session)
        runtime_options = self._prompt_runtime_options(params, session, thread_id)
        if agent.backend.type == "acp_stdio":
            return await self._prompt_external(session, agent.backend, params, send_update, runtime_options.workflow)

        messages = _messages_from_params(params)
        attachments = _attachments_from_params(params)
        request = ChatRequest(
            messages=messages,
            attachments=attachments,
            runtime_options=runtime_options,
        )

        result: AgentRunResult | None = None
        last_error = ""
        agent_message_delta_seen = False
        async for event in self.runtime.iter_events(agent, request):
            update = _event_to_update(event, suppress_agent_message=agent_message_delta_seen)
            if update is not None:
                await send_update(session_id, update)
            if event.type == "agent.message.delta":
                agent_message_delta_seen = True
            if event.type in {"run.completed", "run.failed"}:
                raw_result = event.data.get("result")
                if isinstance(raw_result, dict):
                    result = AgentRunResult.model_validate(raw_result)
                last_error = _string(event.data.get("error")) or last_error

        if result is None:
            result = AgentRunResult(
                agent=agent.name,
                thread_id=thread_id,
                status="failed",
                reply=f"执行失败：{last_error or 'runtime did not return a final result'}",
                metadata={"error": last_error or "runtime did not return a final result"},
            )

        return {
            "stopReason": "end_turn",
            "threadId": result.thread_id,
            "agentName": result.agent,
            "result": result.model_dump(),
        }

    async def close_sessions(self, sessions: dict[str, AcpWebSocketSession]) -> None:
        for session in list(sessions.values()):
            backend = session.backend
            if isinstance(backend, ExternalAcpSession):
                await backend.close()

    async def cancel_session(self, sessions: dict[str, AcpWebSocketSession], session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            return
        backend = session.backend
        if isinstance(backend, ExternalAcpSession):
            await backend.cancel()

    async def _prompt_external(
        self,
        session: AcpWebSocketSession,
        backend_config: Any,
        params: dict[str, Any],
        send_update: AcpUpdateSender,
        workflow: str | None,
    ) -> dict[str, Any]:
        backend = session.backend
        if not isinstance(backend, ExternalAcpSession):
            backend = await ExternalAcpSession.create(
                backend_config,
                session.cwd,
                artifact_store=self.runtime.artifact_store,
                thread_id=session.thread_id,
            )
            session.backend = backend
            session.backend_session_id = backend.backend_session_id
        response = await backend.prompt(
            session.session_id,
            prompt_blocks_from_params(params),
            send_update,
            workflow=workflow,
        )
        payload = prompt_response_payload(response)
        payload["threadId"] = session.thread_id
        payload["agentName"] = session.agent_name
        payload["backend"] = {
            "type": "acp_stdio",
            "sessionId": session.backend_session_id,
        }
        return payload

    def list_agents(self) -> dict[str, Any]:
        return {
            "agents": [
                {
                    "name": agent.name,
                    "displayName": agent.display_name,
                    "description": agent.description,
                }
                for agent in self.loader.list_agents()
            ]
        }

    def list_sessions(self, sessions: dict[str, AcpWebSocketSession]) -> dict[str, Any]:
        return {
            "sessions": [
                {
                    "sessionId": session.session_id,
                    "threadId": session.thread_id,
                    "agentName": session.agent_name,
                    "cwd": session.cwd,
                    "backend": {"type": session.backend_type},
                    "modelId": session.model_id,
                    "modelName": session.model_name,
                    "appTemplateName": session.app_template_name,
                    "runtimeOptions": _runtime_options_response(session.runtime_options),
                }
                for session in sessions.values()
            ]
        }

    def load_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown ACP session: {session_id}")
        cwd = _string(params.get("cwd"))
        if cwd is not None:
            session.cwd = cwd
        raw_options = _runtime_options_payload(params)
        requested_model = _string(raw_options.get("model_name"))
        session.runtime_options = self._merge_and_resolve_runtime_options(session.runtime_options, raw_options)
        if requested_model is not None:
            session.model_id = requested_model
        session.model_name = _string(session.runtime_options.get("model_name")) or session.model_name
        return {}

    async def close_session(self, sessions: dict[str, AcpWebSocketSession], session_id: str) -> dict[str, Any]:
        session = sessions.pop(session_id, None)
        if session is not None and isinstance(session.backend, ExternalAcpSession):
            await session.backend.close()
        return {}

    def set_session_model(
        self,
        sessions: dict[str, AcpWebSocketSession],
        session_id: str,
        model_name: str,
    ) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown ACP session: {session_id}")
        session.model_id = model_name
        resolved_options = self._runtime_options_for_model(model_name)
        if resolved_options:
            session.runtime_options = _merge_runtime_options(session.runtime_options, resolved_options)
        else:
            session.runtime_options["model_name"] = model_name
        session.model_name = _string(session.runtime_options.get("model_name")) or model_name
        return {"sessionId": session_id, "modelId": model_name, "modelName": session.model_name}

    def set_session_mode(
        self,
        sessions: dict[str, AcpWebSocketSession],
        session_id: str,
        mode_id: str,
    ) -> dict[str, Any]:
        if session_id not in sessions:
            raise ValueError(f"Unknown ACP session: {session_id}")
        return {"sessionId": session_id, "modeId": mode_id}

    def set_config_option(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        if session_id not in sessions:
            raise ValueError(f"Unknown ACP session: {session_id}")
        return {"configOptions": []}

    def authenticate(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        return {}

    def delete_session_files(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown ACP session: {session_id}")
        result = delete_thread_files(self.runtime.artifact_store, session.thread_id, params)
        result["sessionId"] = session_id
        return result

    def fork_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        source_session_id = _required_session_id(params)
        source = sessions.get(source_session_id)
        if source is None:
            raise ValueError(f"Unknown ACP session: {source_session_id}")
        session_id = _string(params.get("newSessionId") or params.get("new_session_id")) or f"acp-session-{uuid4().hex[:12]}"
        raw_options = _runtime_options_payload(params)
        runtime_options = self._merge_and_resolve_runtime_options(source.runtime_options, raw_options)
        requested_model = _string(raw_options.get("model_name"))
        thread_id = (
            _string(runtime_options.get("thread_id"))
            or _string(params.get("threadId") or params.get("thread_id"))
            or f"acp-{uuid4().hex[:12]}"
        )
        cwd = _string(params.get("cwd")) or source.cwd
        sessions[session_id] = AcpWebSocketSession(
            session_id=session_id,
            thread_id=thread_id,
            agent_name=source.agent_name,
            cwd=cwd,
            backend_type=source.backend_type,
            model_id=requested_model or source.model_id,
            model_name=_string(runtime_options.get("model_name")) or source.model_name,
            app_template_name=source.app_template_name,
            runtime_options=_without_thread_id(runtime_options),
        )
        return {
            "sessionId": session_id,
            "threadId": thread_id,
            "agentName": source.agent_name,
            "cwd": cwd,
            "appTemplateName": source.app_template_name,
            "runtimeOptions": _runtime_options_response(_without_thread_id(runtime_options)),
        }

    def resume_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        session = sessions.get(session_id)
        cwd = _string(params.get("cwd")) or str(Path.cwd())
        if session is None:
            app_template = self._app_template_from_params(params)
            runtime_options = _new_session_runtime_options(params, app_template)
            agent_name = (
                _string(params.get("agentName") or params.get("agent_name") or params.get("agent"))
                or (app_template.agent_name if app_template is not None else None)
                or "default"
            )
            agent_config = self.loader.load(agent_name)
            model_id = _string(runtime_options.get("model_name"))
            runtime_options = self._resolve_model_runtime_options(runtime_options)
            sessions[session_id] = AcpWebSocketSession(
                session_id=session_id,
                thread_id=(
                    _string(runtime_options.get("thread_id"))
                    or _string(params.get("threadId") or params.get("thread_id"))
                    or f"acp-{uuid4().hex[:12]}"
                ),
                agent_name=agent_name,
                cwd=cwd,
                backend_type=agent_config.backend.type,
                model_id=model_id,
                model_name=_string(runtime_options.get("model_name")),
                app_template_name=app_template.name if app_template is not None else None,
                runtime_options=_without_thread_id(runtime_options),
            )
        else:
            session.cwd = cwd
            raw_options = _runtime_options_payload(params)
            requested_model = _string(raw_options.get("model_name"))
            session.runtime_options = self._merge_and_resolve_runtime_options(session.runtime_options, raw_options)
            if requested_model is not None:
                session.model_id = requested_model
            session.model_name = _string(session.runtime_options.get("model_name")) or session.model_name
        return {}

    def _prompt_runtime_options(
        self,
        params: dict[str, Any],
        session: AcpWebSocketSession,
        thread_id: str,
    ) -> RuntimeOptions:
        merged = dict(session.runtime_options)
        if session.model_id is not None:
            resolved = self._runtime_options_for_model(session.model_id)
            if resolved:
                merged.update(resolved)
            elif session.model_name is not None:
                merged["model_name"] = session.model_name
        elif session.model_name is not None:
            merged["model_name"] = session.model_name
        merged = self._merge_and_resolve_runtime_options(merged, _runtime_options_payload(params))
        merged["thread_id"] = thread_id
        return RuntimeOptions.model_validate(merged)

    def _merge_and_resolve_runtime_options(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = _merge_runtime_options(base, override)
        requested_model = _string(override.get("model_name"))
        if requested_model is None:
            return merged
        resolved = self._runtime_options_for_model(requested_model)
        if not resolved:
            return merged
        return _merge_runtime_options(merged, resolved)

    def _resolve_model_runtime_options(self, runtime_options: dict[str, Any]) -> dict[str, Any]:
        model_id = _string(runtime_options.get("model_name"))
        if model_id is None:
            return runtime_options
        resolved = self._runtime_options_for_model(model_id)
        if not resolved:
            return runtime_options
        return _merge_runtime_options(runtime_options, resolved)

    def _runtime_options_for_model(self, model_id: str) -> dict[str, Any]:
        options = self.model_manager.runtime_options_for(model_id)
        if options is None:
            return {}
        return options.model_dump(mode="python", exclude_none=True)

    def _current_model_id(self) -> str | None:
        model = self.model_manager.default()
        return model.id if model is not None else None

    def _app_template_from_params(self, params: dict[str, Any]) -> AppTemplate | None:
        template_name = _app_template_name(params)
        if template_name is None:
            return None
        try:
            return self.app_registry.get(template_name)
        except KeyError as exc:
            raise ValueError(f"Unknown ACP app template: {template_name}") from exc


def _resolve_thread_id(params: dict[str, Any], session: AcpWebSocketSession) -> str:
    runtime_options = _runtime_options_payload(params)
    return _string(runtime_options.get("thread_id")) or session.thread_id


def _new_session_runtime_options(params: dict[str, Any], app_template: AppTemplate | None) -> dict[str, Any]:
    template_options = _template_runtime_options(app_template)
    request_options = _runtime_options_payload(params)
    return _merge_runtime_options(template_options, request_options)


def _template_runtime_options(app_template: AppTemplate | None) -> dict[str, Any]:
    if app_template is None:
        return {}
    payload = _runtime_options_payload({"runtimeOptions": app_template.runtime_options})
    if app_template.workflow is not None and "workflow" not in payload:
        payload["workflow"] = app_template.workflow
    if app_template.selected_skills and "selected_skills" not in payload:
        payload["selected_skills"] = app_template.selected_skills
    if app_template.selected_mcp_tools and "selected_mcp_tools" not in payload:
        payload["selected_mcp_tools"] = app_template.selected_mcp_tools
    return payload


def _merge_runtime_options(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = {**base, **override}
    if not merged:
        return {}
    return RuntimeOptions.model_validate(merged).model_dump(mode="python", exclude_none=True)


def _without_thread_id(runtime_options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in runtime_options.items() if key != "thread_id"}


def _runtime_options_payload(params: dict[str, Any]) -> dict[str, Any]:
    meta = _params(params.get("_meta"))
    jetlinks_meta = _params(meta.get("jetlinks"))
    sources = [
        params,
        _params(jetlinks_meta.get("runtimeOptions") or jetlinks_meta.get("runtime_options")),
        _params(params.get("runtimeOptions") or params.get("runtime_options")),
    ]
    payload: dict[str, Any] = {}
    for source in sources:
        for raw_name, field_name in _RUNTIME_OPTION_ALIASES.items():
            if raw_name in source:
                payload[field_name] = source[raw_name]
    return payload


def _runtime_options_response(runtime_options: dict[str, Any]) -> dict[str, Any]:
    response: dict[str, Any] = {}
    for field_name, value in runtime_options.items():
        if field_name == "api_key":
            continue
        alias = _RUNTIME_OPTION_RESPONSE_ALIASES.get(field_name, field_name)
        response[alias] = value
    return response


def _app_template_name(params: dict[str, Any]) -> str | None:
    meta = _params(params.get("_meta"))
    jetlinks_meta = _params(meta.get("jetlinks"))
    return _string(
        params.get("appTemplateName")
        or params.get("app_template_name")
        or params.get("app")
        or jetlinks_meta.get("appTemplateName")
        or jetlinks_meta.get("app_template_name")
        or jetlinks_meta.get("app")
    )


def _messages_from_params(params: dict[str, Any]) -> list[Message]:
    raw_messages = params.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        messages = [Message.model_validate(item) for item in raw_messages if isinstance(item, dict)]
        prompt_text, _attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
        if prompt_text:
            messages.append(Message(role="user", content=prompt_text))
        return messages

    text, prompt_attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
    if not text and not prompt_attachments:
        raise ValueError("prompt text or messages are required")
    return [Message(role="user", content=text or " ")]


def _attachments_from_params(params: dict[str, Any]) -> list[Attachment]:
    _prompt_text, prompt_attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
    raw_attachments = params.get("attachments")
    if not isinstance(raw_attachments, list):
        return prompt_attachments
    explicit_attachments = [Attachment.model_validate(item) for item in raw_attachments if isinstance(item, dict)]
    return [*prompt_attachments, *explicit_attachments]


def _event_to_update(event: ChatEvent, *, suppress_agent_message: bool = False) -> dict[str, Any] | None:
    data = event.data
    base: dict[str, Any] = {
        "_meta": {"jetlinksRuntimeEvent": event.model_dump()},
    }
    if event.type == "agent.message.delta":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        return {
            **base,
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
    if event.type == "agent.message":
        if suppress_agent_message:
            return None
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        return {
            **base,
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
    if event.type == "tool.started":
        tool_name = _string(data.get("tool_name")) or "tool"
        return {
            **base,
            "sessionUpdate": "tool_call",
            "toolCallId": _tool_call_id(data, tool_name),
            "title": tool_name,
            "kind": "other",
            "status": "in_progress",
        }
    if event.type in {"tool.completed", "tool.failed"}:
        tool_name = _string(data.get("tool_name")) or "tool"
        return {
            **base,
            "sessionUpdate": "tool_call_update",
            "toolCallId": _tool_call_id(data, tool_name),
            "title": tool_name,
            "kind": "other",
            "status": "failed" if event.type == "tool.failed" else "completed",
            "rawOutput": data.get("structured_content"),
        }
    if event.type == "skill.started":
        skill_name = _skill_name(data) or "skill"
        return {
            **base,
            "sessionUpdate": "tool_call",
            "toolCallId": _skill_tool_call_id(data, skill_name),
            "title": skill_name,
            "kind": "other",
            "status": "in_progress",
        }
    if event.type == "skill.completed":
        skill_name = _skill_name(data) or "skill"
        return {
            **base,
            "sessionUpdate": "tool_call_update",
            "toolCallId": _skill_tool_call_id(data, skill_name),
            "title": skill_name,
            "kind": "other",
            "status": "completed",
            "rawOutput": _runtime_event_raw_output(event),
        }
    if event.type == "verifier.started":
        return {
            **base,
            "sessionUpdate": "tool_call",
            "toolCallId": "runtime-verifier",
            "title": "verifier",
            "kind": "other",
            "status": "in_progress",
        }
    if event.type == "verifier.completed":
        return {
            **base,
            "sessionUpdate": "tool_call_update",
            "toolCallId": "runtime-verifier",
            "title": "verifier",
            "kind": "other",
            "status": "completed",
            "rawOutput": _runtime_event_raw_output(event),
        }
    return {
        **base,
        "sessionUpdate": "agent_thought_chunk",
        "content": {"type": "text", "text": _runtime_event_summary(event)},
    }


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
    return _tool_call_id({}, f"skill-{suffix}")


def _tool_call_id(data: dict[str, Any], fallback: str) -> str:
    raw_id = _string(data.get("tool_call_id"))
    if raw_id is not None:
        return raw_id
    sequence = data.get("sequence")
    if isinstance(sequence, int):
        return f"runtime-{sequence}"
    normalized = "".join(char if char.isalnum() or char in "._-" else "-" for char in fallback)
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
