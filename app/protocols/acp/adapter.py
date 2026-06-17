from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
from contextlib import suppress
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.agent import AgentRuntime
from app.core.apps import AppTemplate, AppTemplateRegistry
from app.core.artifacts import ThreadPaths
from app.core.config import AgentConfigLoader
from app.core.diagnostics import diagnostic_json, env_flag, env_int, redact_diagnostic_value
from app.core.runtime import ModelManager
from app.core.skills.aliases import expand_skill_aliases
from app.protocols.acp.content import prompt_parts_from_dict_blocks
from app.protocols.acp.external_backend import ExternalAcpSession, prompt_blocks_from_params, prompt_response_payload
from app.protocols.acp.input_required import stop_reason_for_result
from app.protocols.acp.schemas import AcpTerminal, AcpWebSocketSession
from app.protocols.acp.session_files import delete_thread_files
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


AcpUpdateSender = Callable[[str, dict[str, Any]], Awaitable[None]]
logger = logging.getLogger("uvicorn.error")
ACP_ADAPTER_TRACE_PAYLOADS = env_flag("ACP_ADAPTER_TRACE_PAYLOADS", "0")
ACP_ADAPTER_TRACE_MAX_CHARS = env_int("ACP_ADAPTER_TRACE_MAX_CHARS", 100)
ACP_ADAPTER_RESULT_PREVIEW_MAX_CHARS = env_int("ACP_ADAPTER_RESULT_PREVIEW_MAX_CHARS", 1200)
ACP_ADAPTER_SESSION_LOG_MAX_CHARS = env_int("ACP_ADAPTER_SESSION_LOG_MAX_CHARS", 1200)

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
    "appTemplateName": "app_template_name",
    "app_template_name": "app_template_name",
    "modelType": "model_type",
    "model_type": "model_type",
    "modelId": "model_name",
    "model_id": "model_name",
    "modelName": "model_name",
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
    "responseFormat": "response_format",
    "response_format": "response_format",
    "mode": "mode",
    "modeId": "mode",
    "mode_id": "mode",
    "configOptions": "config_options",
    "config_options": "config_options",
    "sandboxProfile": "sandbox_profile",
    "sandbox_profile": "sandbox_profile",
}

_REVIEW_TARGET_KEYS = (
    "target",
    "targetName",
    "target_name",
    "taskTargetName",
    "task_target_name",
    "taskTarget",
    "task_target",
    "cvTaskName",
    "cv_task_name",
    "applicationScene",
    "application_scene",
    "eventTypeName",
    "event_type_name",
    "targetSkill",
    "target_skill",
)

_RUNTIME_OPTION_RESPONSE_ALIASES = {
    "thread_id": "threadId",
    "user_id": "userId",
    "project_id": "projectId",
    "selected_skills": "selectedSkills",
    "selected_mcp_tools": "selectedMcpTools",
    "app_template_name": "appTemplateName",
    "model_type": "modelType",
    "model_name": "modelName",
    "base_url": "baseUrl",
    "api_key": "apiKey",
    "top_p": "topP",
    "max_tokens": "maxTokens",
    "request_timeout_seconds": "requestTimeoutSeconds",
    "response_format": "responseFormat",
    "config_options": "configOptions",
    "sandbox_profile": "sandboxProfile",
}

_MODE_IDS = {"plan", "edit", "autonomous", "safe", "yolo"}
_REVIEW_SOURCE_ID_RE = re.compile(r"reviewSourceId\s*(?:为|=|:)?\s*[\[【]([^\]】]+)[\]】]")
_CONFIG_OPTION_DEFINITIONS = [
    {
        "id": "modelName",
        "type": "string",
        "title": "Model",
        "runtimeOption": "modelName",
    },
    {
        "id": "temperature",
        "type": "number",
        "title": "Temperature",
        "runtimeOption": "temperature",
    },
    {
        "id": "selectedSkills",
        "type": "string_array",
        "title": "Skills",
        "runtimeOption": "selectedSkills",
    },
    {
        "id": "workflow",
        "type": "string",
        "title": "Workflow",
        "runtimeOption": "workflow",
    },
    {
        "id": "sandboxProfile",
        "type": "string",
        "title": "Sandbox Profile",
        "runtimeOption": "sandboxProfile",
    },
]


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
                "loadSession": True,
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
                    "runtimeOptions": sorted(
                        set(_RUNTIME_OPTION_RESPONSE_ALIASES.values()) | {"workflow", "temperature", "mode"}
                    ),
                    "artifactWorkspace": True,
                    "modes": [
                        {"id": "plan", "title": "Plan", "description": "Plan-only responses without tool execution."},
                        {"id": "edit", "title": "Edit", "description": "Default balanced tool use."},
                        {
                            "id": "autonomous",
                            "title": "Autonomous",
                            "description": "Allow broader tool use and more tool rounds.",
                        },
                        {
                            "id": "yolo",
                            "title": "YOLO",
                            "description": "Auto-approve permission requests and run with autonomous tool access.",
                        },
                        {"id": "safe", "title": "Safe", "description": "Read-only and low-risk tool use."},
                    ],
                    "configOptions": _CONFIG_OPTION_DEFINITIONS,
                    "methods": [
                        "authenticate",
                        "fs/read_text_file",
                        "fs/write_text_file",
                        "initialize",
                        "session/cancel",
                        "session/close",
                        "session/fork",
                        "session/list",
                        "session/load",
                        "session/new",
                        "session/prompt",
                        "session/request_permission",
                        "session/resume",
                        "session/set_config_option",
                        "session/set_mode",
                        "session/set_model",
                        "terminal/create",
                        "terminal/kill",
                        "terminal/output",
                        "terminal/release",
                        "terminal/wait_for_exit",
                    ],
                }
            },
        }

    def new_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        app_template = self._app_template_from_params(params)
        session_runtime_options = _new_session_runtime_options(params, app_template, self.loader.root_dir)
        agent_name = (
            _agent_name_from_meta(params)
            or (app_template.agent_name if app_template is not None else None)
            or "default"
        )
        agent_config = self.loader.load(agent_name)
        thread_id = (
            _string(session_runtime_options.get("thread_id"))
            or f"acp-{uuid4().hex[:12]}"
        )
        session_id = _string(params.get("sessionId") or params.get("session_id")) or thread_id
        cwd = _string(params.get("cwd")) or str(Path.cwd())
        stored_runtime_options = _without_thread_id(session_runtime_options)
        model_id = _string(stored_runtime_options.get("model_name")) or self._current_model_id()
        if model_id is not None:
            stored_runtime_options["model_name"] = model_id
        if not _force_model_config(stored_runtime_options):
            stored_runtime_options = self._resolve_model_runtime_options(stored_runtime_options)
        model_name = _string(stored_runtime_options.get("model_name"))
        mcp_servers = _mcp_servers_from_params(params)
        logger.debug(
            "acp session new session_id=%s thread_id=%s agent=%s app_template=%s selected_skills=%s mode=%s config_option_keys=%s param_keys=%s meta_keys=%s bridge_container_keys=%s",
            session_id,
            thread_id,
            agent_name,
            app_template.name if app_template is not None else None,
            stored_runtime_options.get("selected_skills") or [],
            stored_runtime_options.get("mode"),
            sorted(_params(stored_runtime_options.get("config_options")).keys()),
            sorted(params.keys()),
            sorted(_params(params.get("_meta")).keys()),
            _bridge_container_key_summary(params),
        )
        logger.info(
            "acp session new detail session_id=%s thread_id=%s agent=%s app_template=%s runtime_options=%s mcp_servers=%s params_summary=%s",
            session_id,
            thread_id,
            agent_name,
            app_template.name if app_template is not None else None,
            diagnostic_json(
                _runtime_options_response(stored_runtime_options),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
            diagnostic_json(
                _mcp_servers_response(mcp_servers),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
            diagnostic_json(
                _acp_param_log_summary(params),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
        )
        _dump_acp_request_debug(
            thread_id,
            "session.new",
            params,
            app_template=app_template.name if app_template is not None else None,
            workflow=stored_runtime_options.get("workflow") if isinstance(stored_runtime_options, dict) else None,
            selected_skills=stored_runtime_options.get("selected_skills") if isinstance(stored_runtime_options, dict) else None,
        )
        sessions[session_id] = AcpWebSocketSession(
            session_id=session_id,
            thread_id=thread_id,
            agent_name=agent_name,
            cwd=cwd,
            mcp_servers=mcp_servers,
            backend_type=agent_config.backend.type,
            model_id=model_id,
            model_name=model_name,
            app_template_name=app_template.name if app_template is not None else None,
            runtime_options=stored_runtime_options,
            mode_id=_mode_from_options(stored_runtime_options),
            config_options=_config_options_from_runtime_options(stored_runtime_options),
        )
        return {
            "sessionId": session_id,
            "threadId": thread_id,
            "agentName": agent_name,
            "cwd": cwd,
            "mcpServers": _mcp_servers_response(mcp_servers),
            "backend": {"type": agent_config.backend.type},
            "appTemplateName": app_template.name if app_template is not None else None,
            "runtimeOptions": _runtime_options_response(stored_runtime_options),
            "modeId": sessions[session_id].mode_id,
            "configOptions": sessions[session_id].config_options,
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
            if session_id:
                restored_params = _restored_session_new_params(session_id)
                if restored_params is not None:
                    logger.info(
                        "acp prompt restored missing session from thread files session_id=%s restored_app_template=%s",
                        session_id,
                        _app_template_name(restored_params),
                    )
                    self.new_session(sessions, restored_params)
                else:
                    logger.warning(
                        "acp prompt missing session and no persisted session.new found session_id=%s",
                        session_id,
                    )
            if not session_id or session_id not in sessions:
                session_result = self.new_session(sessions, params)
                session_id = str(session_result["sessionId"])

        session = sessions[session_id]
        agent_name = _agent_name_from_meta(params) or session.agent_name
        agent = self.loader.load(agent_name)
        prompt_app_template = self._app_template_from_params(params)
        if prompt_app_template is not None and prompt_app_template.name != session.app_template_name:
            raw_options = _runtime_options_payload(params)
            template_options = _template_runtime_options(prompt_app_template)
            app_model_options = _app_model_runtime_options(prompt_app_template, template_options, raw_options)
            session.runtime_options = _merge_app_template_runtime_options(
                session.runtime_options,
                template_options,
                app_model_options,
                raw_options,
                root_dir=self.loader.root_dir,
            )
            session.app_template_name = prompt_app_template.name
            session.mode_id = _mode_from_options(session.runtime_options, fallback=session.mode_id)
            session.config_options = _config_options_from_runtime_options(session.runtime_options)
            session.agent_name = _agent_name_from_meta(params) or prompt_app_template.agent_name or session.agent_name
            model_name = _string(session.runtime_options.get("model_name"))
            if model_name is not None:
                session.model_id = model_name
                session.model_name = model_name
            agent_name = session.agent_name
            agent = self.loader.load(agent_name)
        logger.info(
            "acp prompt app routing session_id=%s thread_id=%s review_source_id=%s app_template=%s agent=%s selected_skills=%s workflow=%s",
            session_id,
            session.thread_id,
            _review_source_id_from_params(params) or "",
            session.app_template_name or "",
            agent_name,
            session.runtime_options.get("selected_skills") if isinstance(session.runtime_options, dict) else None,
            session.runtime_options.get("workflow") if isinstance(session.runtime_options, dict) else None,
        )
        _dump_acp_request_debug(
            session.thread_id,
            "session.prompt",
            params,
            app_template=session.app_template_name,
            workflow=session.runtime_options.get("workflow") if isinstance(session.runtime_options, dict) else None,
            selected_skills=session.runtime_options.get("selected_skills") if isinstance(session.runtime_options, dict) else None,
        )
        thread_id = _resolve_thread_id(params, session)
        runtime_options = self._prompt_runtime_options(params, session, thread_id)
        logger.debug(
            "acp prompt routing session_id=%s thread_id=%s agent=%s app_template=%s selected_skills=%s selected_mcp_tools=%s mode=%s config_option_keys=%s param_keys=%s meta_keys=%s bridge_container_keys=%s",
            session_id,
            thread_id,
            agent_name,
            session.app_template_name,
            runtime_options.selected_skills,
            runtime_options.selected_mcp_tools,
            runtime_options.mode,
            sorted(runtime_options.config_options.keys()),
            sorted(params.keys()),
            sorted(_params(params.get("_meta")).keys()),
            _bridge_container_key_summary(params),
        )
        logger.info(
            "acp prompt detail-lite session_id=%s thread_id=%s agent=%s app_template=%s runtime_options=%s session_mcp_servers=%s prompt_mcp_servers=%s params_summary=%s",
            session_id,
            thread_id,
            agent_name,
            session.app_template_name,
            diagnostic_json(
                runtime_options.model_dump(mode="python", exclude_none=True, exclude={"api_key"}),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
            diagnostic_json(
                _mcp_servers_response(session.mcp_servers),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
            diagnostic_json(
                _mcp_servers_response(_mcp_servers_from_params(params)),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
            diagnostic_json(
                _acp_param_log_summary(params),
                max_chars=ACP_ADAPTER_SESSION_LOG_MAX_CHARS,
            ),
        )
        if agent.backend.type == "acp_stdio":
            return await self._prompt_external(session, agent.backend, params, send_update, runtime_options.workflow)

        messages = _messages_from_params(params)
        attachments = _attachments_from_params(params)
        if ACP_ADAPTER_TRACE_PAYLOADS:
            logger.info(
                "acp prompt detail session_id=%s thread_id=%s app_template=%s params=%s runtime_options=%s messages=%s attachments=%s",
                session_id,
                thread_id,
                session.app_template_name,
                diagnostic_json(params, max_chars=ACP_ADAPTER_TRACE_MAX_CHARS),
                diagnostic_json(
                    runtime_options.model_dump(mode="python", exclude_none=True),
                    max_chars=ACP_ADAPTER_TRACE_MAX_CHARS,
                ),
                diagnostic_json(
                    [message.model_dump(mode="python") for message in messages],
                    max_chars=ACP_ADAPTER_TRACE_MAX_CHARS,
                ),
                diagnostic_json(
                    [attachment.model_dump(mode="python") for attachment in attachments],
                    max_chars=ACP_ADAPTER_TRACE_MAX_CHARS,
                ),
            )
        request = ChatRequest(
            messages=messages,
            attachments=attachments,
            runtime_options=runtime_options,
        )

        result: AgentRunResult | None = None
        last_error = ""
        agent_message_delta_seen = False
        structured_json_updates = _should_normalize_structured_json(runtime_options, session)
        async for event in self.runtime.iter_events(agent, request):
            if ACP_ADAPTER_TRACE_PAYLOADS:
                logger.info(
                    "acp runtime event detail session_id=%s thread_id=%s event_type=%s data=%s",
                    session_id,
                    thread_id,
                    event.type,
                    diagnostic_json(event.data, max_chars=ACP_ADAPTER_TRACE_MAX_CHARS),
                )
            if structured_json_updates and event.type in {"agent.message", "agent.message.delta"}:
                if event.type == "agent.message.delta":
                    agent_message_delta_seen = True
                continue
            update = _event_to_update(event, suppress_agent_message=agent_message_delta_seen)
            if update is not None:
                await send_update(session_id, update)
            if event.type == "agent.message.delta":
                agent_message_delta_seen = True
            if event.type in {"run.completed", "run.failed"}:
                raw_result = event.data.get("result")
                if isinstance(raw_result, dict):
                    result = AgentRunResult.model_validate(raw_result)
                    for resource_update in _resource_content_updates(event, result.content):
                        await send_update(session_id, resource_update)
                last_error = _string(event.data.get("error")) or last_error

        if result is None:
            result = AgentRunResult(
                agent=agent.name,
                thread_id=thread_id,
                status="failed",
                reply=f"执行失败：{last_error or 'runtime did not return a final result'}",
                metadata={"error": last_error or "runtime did not return a final result"},
            )

        result = _normalize_structured_json_result(result, runtime_options, session)
        if structured_json_updates:
            final_message_update = _final_structured_json_message_update(result)
            if final_message_update is not None:
                await send_update(session_id, final_message_update)
        stop_reason = stop_reason_for_result(result, request)
        payload = {
            "stopReason": stop_reason,
            "threadId": result.thread_id,
            "agentName": result.agent,
            "content": result.content,
            "result": result.model_dump(),
        }
        logger.info(
            "\n===== ACP 调度结果 | acp dispatch result =====\n"
            "会话ID: %s\n"
            "线程ID: %s\n"
            "智能体: %s\n"
            "停止原因: %s\n"
            "运行状态: %s\n"
            "工作流: %s\n"
            "运行ID: %s\n"
            "回复字符数: %s\n"
            "内容项数: %s\n"
            "回复内容(最多 %s 字符):\n%s\n"
            "元数据:\n%s\n"
            "===== ACP 调度结果结束 =====",
            session_id,
            result.thread_id,
            result.agent,
            stop_reason,
            result.status,
            result.metadata.get("workflow"),
            result.metadata.get("run_id"),
            len(result.reply),
            len(result.content),
            ACP_ADAPTER_RESULT_PREVIEW_MAX_CHARS,
            _preview_log_text(result.reply, max_chars=ACP_ADAPTER_RESULT_PREVIEW_MAX_CHARS),
            diagnostic_json(result.metadata, max_chars=ACP_ADAPTER_RESULT_PREVIEW_MAX_CHARS),
        )
        if result.metadata.get("requires_input") is True:
            payload["_meta"] = {
                "jetlinks": {
                    "stopReason": "input_required",
                    "requiresInput": True,
                    "requiredInputs": result.metadata.get("required_inputs") or [],
                }
            }
        return payload

    async def close_sessions(self, sessions: dict[str, AcpWebSocketSession]) -> None:
        for session in list(sessions.values()):
            await self._close_session_resources(session)

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
                mcp_servers=session.mcp_servers,
            )
            session.backend = backend
            session.backend_session_id = backend.backend_session_id
        response = await backend.prompt(
            session.session_id,
            prompt_blocks_from_params(params),
            send_update,
            workflow=workflow,
            yolo_mode=session.mode_id == "yolo",
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
                    "mcpServers": _mcp_servers_response(session.mcp_servers),
                    "backend": {"type": session.backend_type},
                    "modelId": session.model_id,
                    "modelName": session.model_name,
                    "appTemplateName": session.app_template_name,
                    "runtimeOptions": _runtime_options_response(session.runtime_options),
                    "modeId": session.mode_id,
                    "configOptions": session.config_options,
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
        if _has_mcp_servers(params):
            session.mcp_servers = _mcp_servers_from_params(params)
        raw_options = _runtime_options_payload(params)
        requested_model = _string(raw_options.get("model_name"))
        session.runtime_options = self._merge_and_resolve_runtime_options(session.runtime_options, raw_options)
        session.mode_id = _mode_from_options(session.runtime_options, fallback=session.mode_id)
        session.config_options = _config_options_from_runtime_options(session.runtime_options)
        if requested_model is not None:
            session.model_id = requested_model
        session.model_name = _string(session.runtime_options.get("model_name")) or session.model_name
        return {}

    async def close_session(self, sessions: dict[str, AcpWebSocketSession], session_id: str) -> dict[str, Any]:
        session = sessions.pop(session_id, None)
        if session is not None:
            await self._close_session_resources(session)
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
            session.runtime_options = _merge_runtime_options(
                session.runtime_options,
                resolved_options,
                root_dir=self.loader.root_dir,
            )
        else:
            session.runtime_options["model_name"] = model_name
        session.model_name = _string(session.runtime_options.get("model_name")) or model_name
        session.config_options = _config_options_from_runtime_options(session.runtime_options)
        return {"sessionId": session_id, "modelId": model_name, "modelName": session.model_name}

    def update_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown ACP session: {session_id}")

        raw_options = _session_update_runtime_options(params, self.loader.root_dir)
        app_template = self._app_template_from_params(params)
        base_options = dict(session.runtime_options)
        if app_template is not None:
            template_options = _template_runtime_options(app_template)
            app_model_options = _app_model_runtime_options(app_template, template_options, raw_options)
            base_options = _merge_app_template_runtime_options(
                base_options,
                template_options,
                app_model_options,
                raw_options,
                root_dir=self.loader.root_dir,
            )
            session.app_template_name = app_template.name
        else:
            base_options = self._merge_and_resolve_runtime_options(base_options, raw_options)

        session.runtime_options = base_options
        session.mode_id = _mode_from_options(session.runtime_options, fallback=session.mode_id)
        session.config_options = _config_options_from_runtime_options(session.runtime_options)
        model_name = _string(session.runtime_options.get("model_name"))
        if model_name is not None:
            session.model_id = model_name
            session.model_name = model_name

        return {
            "sessionId": session_id,
            "threadId": session.thread_id,
            "agentName": session.agent_name,
            "appTemplateName": session.app_template_name,
            "runtimeOptions": _runtime_options_response(session.runtime_options),
            "modeId": session.mode_id,
            "configOptions": session.config_options,
            "models": {
                "currentModelId": session.model_id,
                "currentModelName": session.model_name,
                "availableModels": self.model_manager.list_public_payloads(),
            },
        }

    def set_session_mode(
        self,
        sessions: dict[str, AcpWebSocketSession],
        session_id: str,
        mode_id: str,
    ) -> dict[str, Any]:
        if session_id not in sessions:
            raise ValueError(f"Unknown ACP session: {session_id}")
        mode = _normalize_mode(mode_id)
        session = sessions[session_id]
        session.mode_id = mode
        session.runtime_options["mode"] = mode
        return {"sessionId": session_id, "modeId": mode}

    def set_config_option(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        session = sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown ACP session: {session_id}")
        config_id = _string(params.get("configId") or params.get("config_id"))
        if config_id is None:
            return {"configOptions": _CONFIG_OPTION_DEFINITIONS, "values": session.config_options}
        value = params.get("value")
        runtime_option = _config_option_runtime_update(config_id, value)
        session.runtime_options = self._merge_and_resolve_runtime_options(session.runtime_options, runtime_option)
        session.mode_id = _mode_from_options(session.runtime_options, fallback=session.mode_id)
        session.config_options = _config_options_from_runtime_options(session.runtime_options)
        return {"configOptions": _CONFIG_OPTION_DEFINITIONS, "values": session.config_options}

    def authenticate(self, params: dict[str, Any]) -> dict[str, Any]:
        del params
        return {}

    async def request_permission(
        self,
        sessions: dict[str, AcpWebSocketSession],
        params: dict[str, Any],
        send_update: AcpUpdateSender | None = None,
    ) -> dict[str, Any]:
        session_id = _required_session_id(params)
        if session_id not in sessions:
            raise ValueError(f"Unknown ACP session: {session_id}")
        if sessions[session_id].mode_id == "yolo":
            option_id = _auto_permission_option(params.get("options"))
            return {"outcome": {"outcome": "selected", "optionId": option_id or "allow"}}
        option_id = _string(params.get("selectedOptionId") or params.get("selected_option_id"))
        if option_id is not None:
            return {"outcome": {"outcome": "selected", "optionId": option_id}}
        if params.get("approved") is True:
            return {"outcome": {"outcome": "approved"}}
        if send_update is not None:
            await send_update(
                session_id,
                {
                    "sessionUpdate": "permission_request",
                    "permissionRequest": {
                        "id": _string(params.get("permissionRequestId") or params.get("permission_request_id"))
                        or f"permission-{uuid4().hex[:12]}",
                        "title": _string(params.get("title")) or "Permission request",
                        "description": _string(params.get("description") or params.get("reason")) or "",
                        "options": params.get("options") if isinstance(params.get("options"), list) else [],
                    },
                    "_meta": {"jetlinksPermissionRequest": params},
                },
            )
        return {"outcome": {"outcome": "cancelled"}}

    def read_text_file(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session = _required_session(sessions, params)
        path = _required_path(params)
        target = self._resolve_read_path(session, path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        line = _positive_int(params.get("line")) or 1
        limit = _positive_int(params.get("limit"))
        start = max(0, line - 1)
        selected = lines[start : start + limit] if limit is not None else lines[start:]
        return {"content": "\n".join(selected)[:200_000]}

    def write_text_file(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session = _required_session(sessions, params)
        path = _required_path(params)
        content = params.get("content")
        if not isinstance(content, str):
            raise ValueError("content is required")
        target = self._resolve_write_path(session, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {}

    async def create_terminal(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session = _required_session(sessions, params)
        command = params.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command is required")
        args = params.get("args")
        if args is not None and not (isinstance(args, list) and all(isinstance(item, str) for item in args)):
            raise ValueError("args must be a list of strings")
        cwd = self._resolve_terminal_cwd(session, _string(params.get("cwd")))
        output_limit = _positive_int(params.get("outputByteLimit") or params.get("output_byte_limit")) or 200_000
        env = os.environ.copy()
        raw_env = params.get("env")
        if isinstance(raw_env, list):
            for item in raw_env:
                if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("value"), str):
                    env[item["name"]] = item["value"]
        process = await asyncio.create_subprocess_exec(
            command,
            *(args or []),
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        terminal_id = f"terminal-{uuid4().hex[:12]}"
        terminal = AcpTerminal(process=process, output_limit=output_limit)
        terminal.reader_task = asyncio.create_task(_capture_terminal_output(terminal))
        session.terminals[terminal_id] = terminal
        return {"terminalId": terminal_id, "terminal_id": terminal_id}

    def terminal_output(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        terminal = _required_terminal(_required_session(sessions, params), params)
        exit_status = _terminal_exit_status(terminal.process.returncode)
        return {
            "output": terminal.output.decode("utf-8", errors="replace"),
            "truncated": terminal.truncated,
            "exitStatus": exit_status,
            "exit_status": exit_status,
        }

    async def wait_for_terminal_exit(
        self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]
    ) -> dict[str, Any]:
        terminal = _required_terminal(_required_session(sessions, params), params)
        returncode = await terminal.process.wait()
        if terminal.reader_task is not None:
            with suppress(asyncio.CancelledError):
                await terminal.reader_task
        if returncode >= 0:
            return {"exitCode": returncode, "exit_code": returncode}
        return {"signal": _signal_name(-returncode)}

    async def kill_terminal(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        terminal = _required_terminal(_required_session(sessions, params), params)
        if terminal.process.returncode is None:
            terminal.process.kill()
            await terminal.process.wait()
        if terminal.reader_task is not None:
            with suppress(asyncio.CancelledError):
                await terminal.reader_task
        return {}

    async def release_terminal(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session = _required_session(sessions, params)
        terminal_id = _required_terminal_id(params)
        terminal = session.terminals.pop(terminal_id, None)
        if terminal is not None:
            await _terminate_terminal(terminal)
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
            or f"acp-{uuid4().hex[:12]}"
        )
        cwd = _string(params.get("cwd")) or source.cwd
        mcp_servers = _mcp_servers_from_params(params) if _has_mcp_servers(params) else list(source.mcp_servers)
        sessions[session_id] = AcpWebSocketSession(
            session_id=session_id,
            thread_id=thread_id,
            agent_name=source.agent_name,
            cwd=cwd,
            mcp_servers=mcp_servers,
            backend_type=source.backend_type,
            model_id=requested_model or source.model_id,
            model_name=_string(runtime_options.get("model_name")) or source.model_name,
            app_template_name=source.app_template_name,
            runtime_options=_without_thread_id(runtime_options),
            mode_id=source.mode_id,
            config_options=_config_options_from_runtime_options(_without_thread_id(runtime_options)),
        )
        return {
            "sessionId": session_id,
            "threadId": thread_id,
            "agentName": source.agent_name,
            "cwd": cwd,
            "mcpServers": _mcp_servers_response(mcp_servers),
            "appTemplateName": source.app_template_name,
            "runtimeOptions": _runtime_options_response(_without_thread_id(runtime_options)),
        }

    def resume_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        session_id = _required_session_id(params)
        session = sessions.get(session_id)
        cwd = _string(params.get("cwd")) or str(Path.cwd())
        if session is None:
            app_template = self._app_template_from_params(params)
            runtime_options = _new_session_runtime_options(params, app_template, self.loader.root_dir)
            agent_name = (
                _agent_name_from_meta(params)
                or (app_template.agent_name if app_template is not None else None)
                or "default"
            )
            agent_config = self.loader.load(agent_name)
            model_id = _string(runtime_options.get("model_name"))
            if not _force_model_config(runtime_options):
                runtime_options = self._resolve_model_runtime_options(runtime_options)
            mcp_servers = _mcp_servers_from_params(params)
            sessions[session_id] = AcpWebSocketSession(
                session_id=session_id,
                thread_id=(
                    _string(runtime_options.get("thread_id"))
                    or session_id
                ),
                agent_name=agent_name,
                cwd=cwd,
                mcp_servers=mcp_servers,
                backend_type=agent_config.backend.type,
                model_id=model_id,
                model_name=_string(runtime_options.get("model_name")),
                app_template_name=app_template.name if app_template is not None else None,
                runtime_options=_without_thread_id(runtime_options),
                mode_id=_mode_from_options(runtime_options),
                config_options=_config_options_from_runtime_options(runtime_options),
            )
        else:
            session.cwd = cwd
            if _has_mcp_servers(params):
                session.mcp_servers = _mcp_servers_from_params(params)
            raw_options = _runtime_options_payload(params)
            requested_model = _string(raw_options.get("model_name"))
            session.runtime_options = self._merge_and_resolve_runtime_options(session.runtime_options, raw_options)
            session.mode_id = _mode_from_options(session.runtime_options, fallback=session.mode_id)
            session.config_options = _config_options_from_runtime_options(session.runtime_options)
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
                merged = _merge_runtime_options(resolved, merged, root_dir=self.loader.root_dir)
            elif session.model_name is not None:
                merged["model_name"] = session.model_name
        elif session.model_name is not None:
            merged["model_name"] = session.model_name
        raw_options = _runtime_options_payload(params)
        merged = self._merge_and_resolve_runtime_options(merged, raw_options)
        if session.app_template_name is not None:
            app_template = self._app_template_from_name(session.app_template_name)
            if app_template is not None:
                template_options = _template_runtime_options(app_template)
                app_model_options = _app_model_runtime_options(app_template, template_options, raw_options)
                if _force_model_config(template_options, raw_options) and app_model_options:
                    merged = _merge_runtime_options(merged, app_model_options, root_dir=self.loader.root_dir)
        merged["thread_id"] = thread_id
        merged["mode"] = _mode_from_options(merged, fallback=session.mode_id)
        raw_config_options = merged.get("config_options")
        config_options = dict(raw_config_options) if isinstance(raw_config_options, dict) else {}
        config_options["session_cwd"] = session.cwd
        review_target = _review_target_payload(params)
        if review_target:
            config_options["review_target"] = review_target
        if session.mcp_servers:
            config_options["mcpServers"] = list(session.mcp_servers)
        merged["config_options"] = config_options
        if session.app_template_name is not None and "app_template_name" not in merged:
            merged["app_template_name"] = session.app_template_name
        if isinstance(merged.get("selected_skills"), list):
            merged["selected_skills"] = expand_skill_aliases(merged["selected_skills"], self.loader.root_dir)
        return RuntimeOptions.model_validate(merged)

    def _merge_and_resolve_runtime_options(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = _merge_runtime_options(base, override, root_dir=self.loader.root_dir)
        requested_model = _string(override.get("model_name"))
        if requested_model is None:
            return merged
        resolved = self._runtime_options_for_model(requested_model)
        if not resolved:
            return merged
        return _merge_runtime_options(merged, resolved, root_dir=self.loader.root_dir)

    def _resolve_model_runtime_options(self, runtime_options: dict[str, Any]) -> dict[str, Any]:
        model_id = _string(runtime_options.get("model_name"))
        if model_id is None:
            return runtime_options
        resolved = self._runtime_options_for_model(model_id)
        if not resolved:
            return runtime_options
        return _merge_runtime_options(runtime_options, resolved, root_dir=self.loader.root_dir)

    def _runtime_options_for_model(self, model_id: str) -> dict[str, Any]:
        options = self.model_manager.runtime_options_for(model_id)
        if options is None:
            return {}
        payload = options.model_dump(mode="python", exclude_none=True)
        model_fields = {
            "model_name",
            "base_url",
            "api_key",
            "temperature",
            "top_p",
            "max_tokens",
            "request_timeout_seconds",
        }
        return {key: value for key, value in payload.items() if key in model_fields}

    def _current_model_id(self) -> str | None:
        model = self.model_manager.default()
        return model.id if model is not None else None

    async def _close_session_resources(self, session: AcpWebSocketSession) -> None:
        for terminal_id in list(session.terminals):
            terminal = session.terminals.pop(terminal_id)
            await _terminate_terminal(terminal)
        backend = session.backend
        if isinstance(backend, ExternalAcpSession):
            await backend.close()

    def _thread_paths(self, session: AcpWebSocketSession) -> ThreadPaths:
        return self.runtime.artifact_store.prepare_thread(session.thread_id)

    def _resolve_read_path(self, session: AcpWebSocketSession, raw_path: str) -> Path:
        return _resolve_session_path(self._thread_paths(session), session.cwd, raw_path, for_write=False)

    def _resolve_write_path(self, session: AcpWebSocketSession, raw_path: str) -> Path:
        return _resolve_session_path(self._thread_paths(session), session.cwd, raw_path, for_write=True)

    def _resolve_terminal_cwd(self, session: AcpWebSocketSession, raw_cwd: str | None) -> Path:
        if raw_cwd is None:
            return self._thread_paths(session).workspace
        target = self._resolve_write_path(session, raw_cwd)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _app_template_from_params(self, params: dict[str, Any]) -> AppTemplate | None:
        template_name = _app_template_name(params)
        if template_name is None:
            template_name = self._app_template_name_from_review_source(params)
        return self._app_template_from_name(template_name)

    def _app_template_from_name(self, template_name: str | None) -> AppTemplate | None:
        if template_name is None:
            return None
        template = self.app_registry.find(template_name)
        if template is not None:
            return template
        raise ValueError(f"Unknown ACP app template: {template_name}")

    def _app_template_name_from_review_source(self, params: dict[str, Any]) -> str | None:
        review_source_id = _review_source_id_from_params(params)
        if review_source_id is None:
            return None
        for candidate in _review_source_template_candidates(review_source_id):
            template = self.app_registry.find(candidate)
            if template is None:
                continue
            logger.info(
                "acp inferred app template from reviewSourceId app_template=%s review_source_id=%s candidate=%s",
                template.name,
                review_source_id,
                candidate,
            )
            return template.name
        return None


def _resolve_thread_id(params: dict[str, Any], session: AcpWebSocketSession) -> str:
    runtime_options = _runtime_options_payload(params)
    return _string(runtime_options.get("thread_id")) or session.thread_id


def _new_session_runtime_options(
    params: dict[str, Any],
    app_template: AppTemplate | None,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    template_options = _template_runtime_options(app_template)
    request_options = _runtime_options_payload(params)
    app_model_options = _app_model_runtime_options(app_template, template_options, request_options)
    return _merge_app_template_runtime_options(
        {},
        template_options,
        app_model_options,
        request_options,
        root_dir=root_dir,
    )


def _template_runtime_options(app_template: AppTemplate | None) -> dict[str, Any]:
    if app_template is None:
        return {}
    payload = _runtime_options_from_source(app_template.runtime_options)
    raw_config_options = app_template.runtime_options.get("config_options") if isinstance(app_template.runtime_options, dict) else None
    if isinstance(raw_config_options, dict):
        payload["config_options"] = dict(raw_config_options)
    if app_template.workflow is not None and "workflow" not in payload:
        payload["workflow"] = app_template.workflow
    if app_template.selected_skills and "selected_skills" not in payload:
        payload["selected_skills"] = app_template.selected_skills
    if app_template.selected_mcp_tools and "selected_mcp_tools" not in payload:
        payload["selected_mcp_tools"] = app_template.selected_mcp_tools
    return payload


def _app_model_runtime_options(
    app_template: AppTemplate | None,
    template_options: dict[str, Any],
    request_options: dict[str, Any],
) -> dict[str, Any]:
    force_model_config = _force_model_config(template_options, request_options)
    if app_template is None or (_string(request_options.get("model_name")) is not None and not force_model_config):
        return {}
    model_type = _string(request_options.get("model_type")) or _string(template_options.get("model_type")) or "chat"
    model = app_template.select_model(model_type)
    if model is None:
        return {}

    options: dict[str, Any] = {}
    model_name = model.model or model.default_model or model.name
    if model_name:
        options["model_name"] = model_name
    for field_name in (
        "base_url",
        "api_key",
        "temperature",
        "top_p",
        "max_tokens",
        "request_timeout_seconds",
    ):
        value = getattr(model, field_name)
        if value is not None:
            options[field_name] = value
    return options


def _merge_runtime_options(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    merged = {**base, **override}
    config_options = _merge_config_options(base.get("config_options"), override.get("config_options"))
    if config_options:
        merged["config_options"] = config_options
    if not merged:
        return {}
    normalized = RuntimeOptions.model_validate(merged).model_dump(mode="python", exclude_none=True)
    if "selected_skills" in normalized:
        normalized["selected_skills"] = expand_skill_aliases(normalized["selected_skills"], root_dir)
    return normalized


def _merge_app_template_runtime_options(
    base: dict[str, Any],
    template_options: dict[str, Any],
    app_model_options: dict[str, Any],
    request_options: dict[str, Any],
    *,
    root_dir: Path | None = None,
) -> dict[str, Any]:
    merged = _merge_runtime_options(base, template_options, root_dir=root_dir)
    if _force_model_config(template_options, request_options):
        merged = _merge_runtime_options(merged, request_options, root_dir=root_dir)
        return _merge_runtime_options(merged, app_model_options, root_dir=root_dir)
    merged = _merge_runtime_options(merged, app_model_options, root_dir=root_dir)
    return _merge_runtime_options(merged, request_options, root_dir=root_dir)


def _merge_config_options(base: object, override: object) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(base, dict):
        merged.update(base)
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _force_model_config(*sources: dict[str, Any]) -> bool:
    for source in sources:
        config_options = source.get("config_options")
        if isinstance(config_options, dict) and config_options.get("force_model_config") is True:
            return True
    return False


def _without_thread_id(runtime_options: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in runtime_options.items() if key != "thread_id"}


def _normalize_mode(value: str | None, fallback: str = "edit") -> str:
    if value in _MODE_IDS:
        return value
    return fallback if fallback in _MODE_IDS else "edit"


def _mode_from_options(runtime_options: dict[str, Any], fallback: str = "edit") -> str:
    return _normalize_mode(_string(runtime_options.get("mode")), fallback=fallback)


def _config_options_from_runtime_options(runtime_options: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field_name, response_name in _RUNTIME_OPTION_RESPONSE_ALIASES.items():
        if field_name in runtime_options and field_name != "api_key":
            values[response_name] = runtime_options[field_name]
    if runtime_options.get("workflow") is not None:
        values["workflow"] = runtime_options["workflow"]
    if runtime_options.get("temperature") is not None:
        values["temperature"] = runtime_options["temperature"]
    if runtime_options.get("mode") is not None:
        values["mode"] = runtime_options["mode"]
    return values


def _config_option_runtime_update(config_id: str, value: object) -> dict[str, Any]:
    if config_id in {"modelName", "modelId"}:
        return {"model_name": _string(value)}
    if config_id == "temperature":
        return {"temperature": value}
    if config_id == "selectedSkills":
        return {"selected_skills": value if isinstance(value, list) else []}
    if config_id == "workflow":
        return {"workflow": _string(value)}
    if config_id == "sandboxProfile":
        return {"sandbox_profile": _string(value)}
    if config_id in {"mode", "modeId"}:
        return {"mode": _normalize_mode(_string(value))}
    raise ValueError(f"Unknown ACP config option: {config_id}")


def _auto_permission_option(raw_options: object) -> str | None:
    if not isinstance(raw_options, list):
        return "allow"
    fallback: str | None = None
    for item in raw_options:
        if not isinstance(item, dict):
            continue
        option_id = _string(item.get("id") or item.get("optionId") or item.get("option_id") or item.get("name"))
        if option_id is None:
            continue
        if fallback is None:
            fallback = option_id
        kind = _string(item.get("kind")) or ""
        name = (_string(item.get("name")) or "").lower()
        if kind in {"allow_once", "allow_always"} or "allow" in name or "approve" in name:
            return option_id
    return fallback or "allow"


def _runtime_options_payload(params: dict[str, Any]) -> dict[str, Any]:
    meta = _params(params.get("_meta"))
    sources = [
        params,
        _params(params.get("runtimeOptions") or params.get("runtime_options")),
        _params(meta.get("runtimeOptions") or meta.get("runtime_options")),
    ]
    for container in _bridge_payload_containers(params):
        sources.extend(
            [
                container,
                _params(container.get("runtimeOptions") or container.get("runtime_options")),
            ]
        )
    payload: dict[str, Any] = {}
    for source in sources:
        payload.update(_runtime_options_from_source(source))
    session_tools = _session_init_tools_payload(params)
    if session_tools is not None:
        raw_config_options = payload.get("config_options")
        config_options = dict(raw_config_options) if isinstance(raw_config_options, dict) else {}
        config_options["session_init_tools"] = session_tools
        payload["config_options"] = config_options
    return payload


def _bridge_payload_containers(params: dict[str, Any]) -> list[dict[str, Any]]:
    meta = _params(params.get("_meta"))
    containers: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in (params, meta):
        for key in (
            "parameters",
            "parameter",
            "params",
            "expands",
            "expand",
            "extensions",
            "metadata",
            "meta",
            "arguments",
            "argument",
            "context",
            "extra",
        ):
            container = _params(source.get(key))
            if not container:
                continue
            identity = id(container)
            if identity in seen:
                continue
            seen.add(identity)
            containers.append(container)
            nested = _params(container.get("_meta"))
            if nested:
                nested_identity = id(nested)
                if nested_identity not in seen:
                    seen.add(nested_identity)
                    containers.append(nested)
    return containers


def _bridge_container_key_summary(params: dict[str, Any]) -> list[str]:
    summary: list[str] = []
    for container in _bridge_payload_containers(params):
        summary.append(",".join(sorted(container.keys())))
    return summary


def _runtime_options_from_source(source: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_name, field_name in _RUNTIME_OPTION_ALIASES.items():
        if raw_name in source:
            payload[field_name] = source[raw_name]
    return payload


def _review_target_payload(params: dict[str, Any]) -> dict[str, str]:
    sources = _review_target_sources(params)
    for source in sources:
        for key in _REVIEW_TARGET_KEYS:
            if key not in source:
                continue
            value = _review_target_text(source.get(key))
            if value:
                return {"value": value, "source": key}
    return {}


def _review_target_sources(params: dict[str, Any]) -> list[dict[str, Any]]:
    meta = _params(params.get("_meta"))
    sources = [
        params,
        meta,
        _params(params.get("runtimeOptions") or params.get("runtime_options")),
        _params(meta.get("runtimeOptions") or meta.get("runtime_options")),
    ]
    sources.extend(_review_target_prompt_sources(params.get("prompt")))
    for container in _bridge_payload_containers(params):
        sources.append(container)
        sources.append(_params(container.get("runtimeOptions") or container.get("runtime_options")))
        sources.extend(_review_target_prompt_sources(container.get("prompt")))
    return [source for source in sources if source]


def _review_target_prompt_sources(prompt: Any) -> list[dict[str, Any]]:
    if not isinstance(prompt, list):
        return []
    sources: list[dict[str, Any]] = []
    for block in prompt:
        if not isinstance(block, dict):
            continue
        sources.append(block)
        meta = _params(block.get("_meta"))
        if meta:
            sources.append(meta)
        resource = _params(block.get("resource"))
        if resource:
            sources.append(resource)
            resource_meta = _params(resource.get("_meta"))
            if resource_meta:
                sources.append(resource_meta)
    return sources


def _review_target_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "name", "value", "id", "code"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _session_init_tools_payload(params: dict[str, Any]) -> object:
    meta = _params(params.get("_meta"))
    runtime_options = _params(meta.get("runtimeOptions") or meta.get("runtime_options"))
    sources = [params, meta, runtime_options]
    for container in _bridge_payload_containers(params):
        sources.append(container)
        sources.append(_params(container.get("runtimeOptions") or container.get("runtime_options")))
    for source in sources:
        for key in (
            "sessionInitTools",
            "session_init_tools",
            "sessioninittools",
            "clientTools",
            "client_tools",
            "dynamicTools",
            "dynamic_tools",
        ):
            if key in source:
                return source[key]
    return None


def _has_mcp_servers(params: dict[str, Any]) -> bool:
    if "mcpServers" in params or "mcp_servers" in params:
        return True
    return any("mcpServers" in container or "mcp_servers" in container for container in _bridge_payload_containers(params))


def _mcp_servers_from_params(params: dict[str, Any]) -> list[dict[str, Any]]:
    raw_servers = params.get("mcpServers")
    if raw_servers is None:
        raw_servers = params.get("mcp_servers")
    if raw_servers is None:
        for container in _bridge_payload_containers(params):
            raw_servers = container.get("mcpServers")
            if raw_servers is None:
                raw_servers = container.get("mcp_servers")
            if raw_servers is not None:
                break
    if not isinstance(raw_servers, list):
        return []
    servers: list[dict[str, Any]] = []
    for raw_server in raw_servers:
        server = _mcp_server_from_source(raw_server)
        if server is not None:
            servers.append(server)
    return servers


def _mcp_server_from_source(raw_server: object) -> dict[str, Any] | None:
    if not isinstance(raw_server, dict):
        return None
    server: dict[str, Any] = {}
    for key in ("name", "type", "url", "command"):
        value = _string(raw_server.get(key))
        if value:
            server[key] = value
    for key in ("headers", "args", "env"):
        value = raw_server.get(key)
        if isinstance(value, list):
            server[key] = [dict(item) if isinstance(item, dict) else item for item in value]
    meta = _params(raw_server.get("_meta"))
    if meta:
        server["_meta"] = meta
    return server if server else None


def _mcp_servers_response(mcp_servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_redacted_mcp_server(server) for server in mcp_servers]


def _redacted_mcp_server(server: dict[str, Any]) -> dict[str, Any]:
    payload = dict(server)
    headers = payload.get("headers")
    if isinstance(headers, list):
        payload["headers"] = [_redacted_header(header) for header in headers]
    env = payload.get("env")
    if isinstance(env, list):
        payload["env"] = [_redacted_env(item) for item in env]
    return payload


def _redacted_header(header: object) -> object:
    if not isinstance(header, dict):
        return header
    payload = dict(header)
    if "value" in payload and _sensitive_name(_string(payload.get("name"))):
        payload["value"] = "********"
    return payload


def _redacted_env(item: object) -> object:
    if not isinstance(item, dict):
        return item
    payload = dict(item)
    if "value" in payload and _sensitive_name(_string(payload.get("name"))):
        payload["value"] = "********"
    return payload


def _sensitive_name(name: str | None) -> bool:
    if name is None:
        return False
    lowered = name.lower()
    return any(token in lowered for token in ("authorization", "token", "secret", "key", "password"))


def _session_update_runtime_options(params: dict[str, Any], root_dir: Path | None = None) -> dict[str, Any]:
    payload = _runtime_options_payload(params)
    if isinstance(params.get("configOptions"), dict) or isinstance(params.get("config_options"), dict):
        payload.pop("config_options", None)
    for raw_container in (
        params.get("update"),
        params.get("sessionUpdate"),
        params.get("session_update"),
        params.get("configOptions"),
        params.get("config_options"),
        params.get("values"),
    ):
        container = _params(raw_container)
        if container:
            payload = _merge_runtime_options(payload, _runtime_options_payload(container), root_dir=root_dir)
    return payload


def _runtime_options_response(runtime_options: dict[str, Any]) -> dict[str, Any]:
    response: dict[str, Any] = {}
    for field_name, value in runtime_options.items():
        if field_name == "api_key":
            continue
        alias = _RUNTIME_OPTION_RESPONSE_ALIASES.get(field_name, field_name)
        response[alias] = value
    return response


def _acp_param_log_summary(params: dict[str, Any]) -> dict[str, Any]:
    meta = _params(params.get("_meta"))
    runtime_options = _runtime_options_payload(params)
    bridge_containers = _bridge_payload_containers(params)
    return {
        "param_keys": sorted(params.keys()),
        "meta_keys": sorted(meta.keys()),
        "bridge_container_keys": [sorted(container.keys()) for container in bridge_containers],
        "runtime_options": _runtime_options_response(runtime_options),
        "mcp_servers": _mcp_servers_response(_mcp_servers_from_params(params)),
        "has_runtime_options": bool(runtime_options),
        "has_mcp_servers": bool(_mcp_servers_from_params(params)),
        "session_init_tools_present": _session_init_tools_payload(params) is not None,
    }


def _app_template_name(params: dict[str, Any]) -> str | None:
    meta = _params(params.get("_meta"))
    top_runtime_options = _params(params.get("runtimeOptions") or params.get("runtime_options"))
    runtime_options = _params(meta.get("runtimeOptions") or meta.get("runtime_options"))
    bridge_sources: list[dict[str, Any]] = []
    for container in _bridge_payload_containers(params):
        bridge_sources.append(container)
        nested_runtime_options = _params(container.get("runtimeOptions") or container.get("runtime_options"))
        if nested_runtime_options:
            bridge_sources.append(nested_runtime_options)
    for source in (
        meta,
        params,
        top_runtime_options,
        runtime_options,
        *bridge_sources,
    ):
        value = _template_name_from_source(source)
        if value is not None:
            return value
    return None


def _template_name_from_source(source: dict[str, Any]) -> str | None:
    return _string(
        source.get("appTemplateName")
        or source.get("app_template_name")
        or source.get("app")
        or source.get("templateName")
        or source.get("template_name")
        or source.get("templateId")
        or source.get("template_id")
        or source.get("appId")
        or source.get("app_id")
        or source.get("taskName")
        or source.get("task_name")
        or source.get("agentId")
        or source.get("agent_id")
    )


def _dump_acp_request_debug(
    thread_id: str,
    phase: str,
    params: dict[str, Any],
    *,
    app_template: str | None = None,
    workflow: str | None = None,
    selected_skills: object = None,
) -> None:
    if not thread_id:
        return
    try:
        outputs_dir = Path(__file__).resolve().parents[3] / ".runtime" / "threads" / thread_id / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": phase,
            "threadId": thread_id,
            "appTemplateName": app_template,
            "reviewSourceId": _review_source_id_from_params(params),
            "workflow": workflow,
            "selectedSkills": selected_skills,
            "params": redact_diagnostic_value(params),
        }
        with (outputs_dir / "acp-request-debug.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("acp request debug dump failed thread_id=%s phase=%s", thread_id, phase)


def _restored_session_new_params(session_id: str) -> dict[str, Any] | None:
    try:
        debug_path = (
            Path(__file__).resolve().parents[3]
            / ".runtime"
            / "threads"
            / session_id
            / "outputs"
            / "acp-request-debug.jsonl"
        )
        if not debug_path.exists():
            return None
        restored: dict[str, Any] | None = None
        for line in debug_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("phase") != "session.new":
                continue
            params = record.get("params")
            if not isinstance(params, dict):
                continue
            restored = dict(params)
            restored.setdefault("sessionId", session_id)
            restored.setdefault("threadId", session_id)
            app_template = _string(record.get("appTemplateName"))
            if app_template is not None and _app_template_name(restored) is None:
                meta = _params(restored.get("_meta"))
                restored["_meta"] = {**meta, "appTemplateName": app_template}
        return restored
    except Exception:
        logger.exception("acp persisted session restore failed session_id=%s", session_id)
        return None


def _review_source_id_from_params(params: dict[str, Any]) -> str | None:
    sources: list[object] = [params.get("reviewSourceId"), params.get("review_source_id")]
    for container in _bridge_payload_containers(params):
        sources.extend([container.get("reviewSourceId"), container.get("review_source_id")])

    prompt_text, _attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
    if prompt_text:
        sources.append(prompt_text)
    for message in params.get("messages") if isinstance(params.get("messages"), list) else []:
        if isinstance(message, dict):
            sources.append(message.get("content"))

    for source in sources:
        text = _string(source)
        if text is None:
            continue
        if "reviewSourceId" in text:
            match = _REVIEW_SOURCE_ID_RE.search(text)
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        elif "_" in text:
            return text
    return None


def _review_source_template_candidates(review_source_id: str) -> list[str]:
    parts = [part.strip() for part in review_source_id.split("_") if part.strip()]
    candidates: list[str] = []
    for end in range(len(parts), 0, -1):
        candidate = "_".join(parts[:end])
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _agent_name_from_meta(params: dict[str, Any]) -> str | None:
    meta = _params(params.get("_meta"))
    for source in (meta, params, *_bridge_payload_containers(params)):
        value = _string(source.get("agentName") or source.get("agent_name") or source.get("agent"))
        if value is not None:
            return value
    return None


def _messages_from_params(params: dict[str, Any]) -> list[Message]:
    raw_messages = params.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        messages = [Message.model_validate(item) for item in raw_messages if isinstance(item, dict)]
        prompt_text, _attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
        if prompt_text and not _message_matches_user_text(messages[-1] if messages else None, prompt_text):
            messages.append(Message(role="user", content=prompt_text))
        return messages

    text, prompt_attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
    if not text and not prompt_attachments:
        text = _agent_command_content(params)
    if not text and not prompt_attachments:
        raise ValueError("prompt text or messages are required")
    return [Message(role="user", content=text or " ")]


def _message_matches_user_text(message: Message | None, text: str) -> bool:
    return message is not None and message.role == "user" and message.content == text


def _attachments_from_params(params: dict[str, Any]) -> list[Attachment]:
    _prompt_text, prompt_attachments = prompt_parts_from_dict_blocks(params.get("prompt"))
    raw_attachments = params.get("attachments")
    if not isinstance(raw_attachments, list):
        return prompt_attachments
    explicit_attachments = [Attachment.model_validate(item) for item in raw_attachments if isinstance(item, dict)]
    return [*prompt_attachments, *explicit_attachments]


def _normalize_structured_json_result(
    result: AgentRunResult,
    runtime_options: RuntimeOptions,
    session: AcpWebSocketSession,
) -> AgentRunResult:
    if not _should_normalize_structured_json(runtime_options, session):
        return result
    normalized_reply = _validated_json_text_from_reply(result.reply)
    if normalized_reply is None or normalized_reply == result.reply:
        return result
    content = _replace_primary_text_content(result.content, result.reply, normalized_reply)
    return result.model_copy(update={"reply": normalized_reply, "content": content})


def _should_normalize_structured_json(runtime_options: RuntimeOptions, session: AcpWebSocketSession) -> bool:
    return (
        runtime_options.response_format == "json"
        or session.app_template_name == "ParkingAbnormalEventMonitoring"
        or runtime_options.app_template_name == "ParkingAbnormalEventMonitoring"
    )


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


def _replace_primary_text_content(content: list[dict[str, Any]], old_text: str, new_text: str) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    replaced = False
    for block in content:
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


def _final_structured_json_message_update(result: AgentRunResult) -> dict[str, Any] | None:
    if not result.reply.strip():
        return None
    return {
        "_meta": {
            "jetlinksRuntimeEvent": {
                "type": "agent.message",
                "data": {
                    "agent": result.agent,
                    "thread_id": result.thread_id,
                    "text": result.reply,
                    "final": True,
                },
            },
            "jetlinksPlan": None,
            "jetlinksDiff": None,
        },
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": result.reply},
    }


def _preview_log_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated chars={len(value) - max_chars}>"


def _agent_command_content(params: dict[str, Any]) -> str | None:
    command = _string(params.get("command"))
    if command is not None and command.lower() != "chat":
        return None
    for source in (params, *_bridge_payload_containers(params)):
        value = _string(source.get("content") or source.get("text") or source.get("message"))
        if value is not None:
            return value
    return None


def _event_to_update(event: ChatEvent, *, suppress_agent_message: bool = False) -> dict[str, Any] | None:
    data = event.data
    base: dict[str, Any] = {
        "_meta": {
            "jetlinksRuntimeEvent": event.model_dump(),
            "jetlinksPlan": _plan_update(event),
            "jetlinksDiff": _diff_update(event),
        },
    }
    if event.type == "agent.message.delta":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        if not text:
            return None
        return {
            **base,
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        }
    if event.type == "agent.message":
        if suppress_agent_message:
            return None
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        if not text:
            return None
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


def _resource_content_updates(event: ChatEvent, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = _string(block.get("type"))
        if block_type not in {"resource_link", "resource"}:
            continue
        updates.append(
            {
                "_meta": {
                    "jetlinksRuntimeEvent": event.model_dump(),
                    "jetlinksPlan": _plan_update(event),
                    "jetlinksDiff": _diff_update(event),
                },
                "sessionUpdate": "agent_message_chunk",
                "content": dict(block),
            }
        )
    return updates


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


def _plan_update(event: ChatEvent) -> dict[str, Any] | None:
    data = event.data
    if event.type == "run.started":
        return {"type": "run", "status": "in_progress", "title": "Run started", "detail": _runtime_event_summary(event)}
    if event.type == "llm.request.started":
        return {
            "type": "llm",
            "status": "in_progress",
            "title": "LLM request",
            "round": data.get("round"),
            "tools": data.get("tools") if isinstance(data.get("tools"), list) else [],
        }
    if event.type == "tool.calls.started":
        return {
            "type": "tool_plan",
            "status": "in_progress",
            "title": "Tool calls",
            "items": data.get("tool_calls") if isinstance(data.get("tool_calls"), list) else [],
        }
    if event.type in {"tool.started", "tool.completed", "tool.failed"}:
        return {
            "type": "tool",
            "status": "failed" if event.type == "tool.failed" else ("completed" if event.type == "tool.completed" else "in_progress"),
            "title": _string(data.get("tool_name")) or "tool",
            "id": _tool_call_id(data, _string(data.get("tool_name")) or "tool"),
        }
    if event.type in {"skill.started", "skill.completed"}:
        return {
            "type": "skill",
            "status": "completed" if event.type == "skill.completed" else "in_progress",
            "title": _skill_name(data) or "skill",
        }
    if event.type in {"run.completed", "run.failed"}:
        return {
            "type": "run",
            "status": "failed" if event.type == "run.failed" else "completed",
            "title": "Run completed" if event.type == "run.completed" else "Run failed",
            "detail": _runtime_event_summary(event),
        }
    return None


def _diff_update(event: ChatEvent) -> dict[str, Any] | None:
    data = event.data
    if event.type == "tool.completed":
        structured = _params(data.get("structured_content"))
        path = _string(structured.get("path"))
        if path is not None:
            return {
                "type": "file_change",
                "status": "completed",
                "path": path,
                "source": _string(data.get("tool_name")) or "tool",
                "operation": "updated",
            }
    if event.type == "artifact.created":
        artifact = _params(data.get("artifact"))
        path = _string(artifact.get("path"))
        if path is not None:
            return {
                "type": "artifact_change",
                "status": "completed",
                "path": path,
                "source": "artifact",
                "operation": "created",
            }
    return None


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


def _required_session(sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> AcpWebSocketSession:
    session_id = _required_session_id(params)
    session = sessions.get(session_id)
    if session is None:
        raise ValueError(f"Unknown ACP session: {session_id}")
    return session


def _required_path(params: dict[str, Any]) -> str:
    path = _string(params.get("path"))
    if path is None:
        raise ValueError("path is required")
    return path


def _required_terminal_id(params: dict[str, Any]) -> str:
    terminal_id = _string(params.get("terminalId") or params.get("terminal_id"))
    if terminal_id is None:
        raise ValueError("terminalId is required")
    return terminal_id


def _required_terminal(session: AcpWebSocketSession, params: dict[str, Any]) -> AcpTerminal:
    terminal_id = _required_terminal_id(params)
    terminal = session.terminals.get(terminal_id)
    if terminal is None:
        raise ValueError(f"Unknown ACP terminal: {terminal_id}")
    return terminal


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _resolve_session_path(paths: ThreadPaths, cwd: str, raw_path: str, *, for_write: bool) -> Path:
    normalized = raw_path.replace("\\", "/")
    virtual_roots = {
        "/mnt/user-data/workspace": paths.workspace,
        "/mnt/user-data/uploads": paths.uploads,
        "/mnt/user-data/outputs": paths.outputs,
    }
    for prefix, root in virtual_roots.items():
        if normalized == prefix or normalized.startswith(prefix + "/"):
            if for_write and root == paths.uploads:
                raise ValueError("uploads is read-only over ACP fs/write_text_file")
            suffix = normalized[len(prefix) :].lstrip("/")
            return _bounded_path(root, suffix)

    if normalized.startswith("/"):
        cwd_path = Path(cwd).expanduser().resolve()
        if str(cwd_path) == "/":
            raise ValueError("absolute host paths require a non-root session cwd")
        candidate = Path(normalized).expanduser().resolve()
        try:
            candidate.relative_to(cwd_path)
        except ValueError as exc:
            raise ValueError("absolute host paths must stay within session cwd") from exc
        return candidate

    if for_write:
        return _bounded_path(paths.workspace, normalized)
    return _bounded_path(paths.workspace, normalized)


def _bounded_path(root: Path, raw_path: str) -> Path:
    base = root.resolve()
    candidate = (base / raw_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("ACP file path traversal blocked") from exc
    return candidate


async def _capture_terminal_output(terminal: AcpTerminal) -> None:
    stream = terminal.process.stdout
    if stream is None:
        return
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        terminal.output.extend(chunk)
        if len(terminal.output) > terminal.output_limit:
            terminal.truncated = True
            if terminal.output_limit <= 0:
                terminal.output.clear()
            else:
                del terminal.output[: len(terminal.output) - terminal.output_limit]


async def _terminate_terminal(terminal: AcpTerminal) -> None:
    if terminal.process.returncode is None:
        terminal.process.terminate()
        with suppress(TimeoutError):
            await asyncio.wait_for(terminal.process.wait(), timeout=2)
        if terminal.process.returncode is None:
            terminal.process.kill()
            await terminal.process.wait()
    if terminal.reader_task is not None:
        terminal.reader_task.cancel()
        with suppress(asyncio.CancelledError):
            await terminal.reader_task


def _terminal_exit_status(returncode: int | None) -> dict[str, Any] | None:
    if returncode is None:
        return None
    if returncode >= 0:
        return {"exitCode": returncode, "exit_code": returncode}
    return {"signal": _signal_name(-returncode)}


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"
