from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.agent import AgentRuntime
from app.core.config import AgentConfigLoader
from app.protocols.acp.external_backend import ExternalAcpSession, prompt_blocks_from_params, prompt_response_payload
from app.protocols.acp.schemas import AcpWebSocketSession
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


AcpUpdateSender = Callable[[str, dict[str, Any]], Awaitable[None]]


class AcpRuntimeAdapter:
    """Translate ACP-shaped requests into the internal AgentRuntime contract."""

    def __init__(self, loader: AgentConfigLoader | None = None, runtime: AgentRuntime | None = None) -> None:
        self.loader = loader or AgentConfigLoader()
        self.runtime = runtime or AgentRuntime()

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
                    "embeddedContext": True,
                },
                "sessionCapabilities": {
                    "list": False,
                    "resume": False,
                    "fork": False,
                    "cancel": True,
                },
                "transports": ["websocket"],
            },
        }

    def new_session(self, sessions: dict[str, AcpWebSocketSession], params: dict[str, Any]) -> dict[str, Any]:
        agent_name = _string(params.get("agentName") or params.get("agent_name") or params.get("agent")) or "default"
        agent_config = self.loader.load(agent_name)
        thread_id = _string(params.get("threadId") or params.get("thread_id")) or f"acp-{uuid4().hex[:12]}"
        session_id = _string(params.get("sessionId") or params.get("session_id")) or f"acp-session-{uuid4().hex[:12]}"
        cwd = _string(params.get("cwd")) or str(Path.cwd())
        sessions[session_id] = AcpWebSocketSession(
            session_id=session_id,
            thread_id=thread_id,
            agent_name=agent_name,
            cwd=cwd,
            backend_type=agent_config.backend.type,
        )
        return {
            "sessionId": session_id,
            "threadId": thread_id,
            "agentName": agent_name,
            "cwd": cwd,
            "backend": {"type": agent_config.backend.type},
            "models": {
                "currentModelId": None,
                "availableModels": [],
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
        if agent.backend.type == "acp_stdio":
            return await self._prompt_external(session, agent.backend, params, send_update)

        thread_id = _resolve_thread_id(params, session)
        workflow = _resolve_workflow(params)
        messages = _messages_from_params(params)
        attachments = _attachments_from_params(params)
        request = ChatRequest(
            messages=messages,
            attachments=attachments,
            runtime_options=RuntimeOptions(
                thread_id=thread_id,
                workflow=workflow,
                selected_skills=_resolve_string_list(params, "selectedSkills", "selected_skills"),
                selected_mcp_tools=_resolve_string_list(params, "selectedMcpTools", "selected_mcp_tools"),
            ),
        )

        result: AgentRunResult | None = None
        last_error = ""
        async for event in self.runtime.iter_events(agent, request):
            await send_update(session_id, _event_to_update(event))
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

    async def _prompt_external(
        self,
        session: AcpWebSocketSession,
        backend_config: Any,
        params: dict[str, Any],
        send_update: AcpUpdateSender,
    ) -> dict[str, Any]:
        backend = session.backend
        if not isinstance(backend, ExternalAcpSession):
            backend = await ExternalAcpSession.create(backend_config, session.cwd)
            session.backend = backend
            session.backend_session_id = backend.backend_session_id
        response = await backend.prompt(
            session.session_id,
            prompt_blocks_from_params(params),
            send_update,
            workflow=_resolve_workflow(params),
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


def _resolve_thread_id(params: dict[str, Any], session: AcpWebSocketSession) -> str:
    runtime_options = _params(params.get("runtimeOptions") or params.get("runtime_options"))
    return (
        _string(runtime_options.get("threadId") or runtime_options.get("thread_id"))
        or _string(params.get("threadId") or params.get("thread_id"))
        or session.thread_id
    )


def _resolve_workflow(params: dict[str, Any]) -> str | None:
    runtime_options = _params(params.get("runtimeOptions") or params.get("runtime_options"))
    return _string(runtime_options.get("workflow") or params.get("workflow"))


def _resolve_string_list(params: dict[str, Any], camel_name: str, snake_name: str) -> list[str]:
    runtime_options = _params(params.get("runtimeOptions") or params.get("runtime_options"))
    raw_value = runtime_options.get(camel_name) or runtime_options.get(snake_name) or params.get(camel_name) or params.get(snake_name)
    if not isinstance(raw_value, list):
        return []
    return [item.strip() for item in raw_value if isinstance(item, str) and item.strip()]


def _messages_from_params(params: dict[str, Any]) -> list[Message]:
    raw_messages = params.get("messages")
    if isinstance(raw_messages, list) and raw_messages:
        return [Message.model_validate(item) for item in raw_messages if isinstance(item, dict)]

    prompt = params.get("prompt")
    text = _extract_prompt_text(prompt)
    if not text:
        raise ValueError("prompt text or messages are required")
    return [Message(role="user", content=text)]


def _attachments_from_params(params: dict[str, Any]) -> list[Attachment]:
    raw_attachments = params.get("attachments")
    if not isinstance(raw_attachments, list):
        return []
    return [Attachment.model_validate(item) for item in raw_attachments if isinstance(item, dict)]


def _extract_prompt_text(prompt: object) -> str:
    if isinstance(prompt, str):
        return prompt.strip()
    if not isinstance(prompt, list):
        return ""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        text = block.get("text") or block.get("content")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(part.strip() for part in parts if part.strip())


def _event_to_update(event: ChatEvent) -> dict[str, Any]:
    data = event.data
    base: dict[str, Any] = {
        "sessionUpdate": "runtime_event",
        "_meta": {"jetlinksRuntimeEvent": event.model_dump()},
    }
    if event.type == "agent.message.delta":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        return {
            **base,
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
            "text": text,
        }
    if event.type == "agent.message":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        return {
            **base,
            "sessionUpdate": "agent_message",
            "content": {"type": "text", "text": text},
            "text": text,
        }
    if event.type in {"skill.selected", "skill.started"}:
        tool_name = _string(data.get("skill_name")) or _string(_params(data.get("skill")).get("name")) or event.type
        return {
            **base,
            "sessionUpdate": "tool_call_start",
            "toolCallId": f"runtime-{data.get('sequence', 0)}",
            "title": tool_name,
            "kind": "other",
            "status": "in_progress",
        }
    if event.type in {"skill.completed", "artifact.created", "verifier.completed", "run.failed"}:
        return {
            **base,
            "sessionUpdate": "tool_call_update",
            "toolCallId": f"runtime-{data.get('sequence', 0)}",
            "status": "completed" if event.type != "run.failed" else "failed",
            "content": [{"type": "text", "text": _runtime_event_summary(event)}],
        }
    return base


def _runtime_event_summary(event: ChatEvent) -> str:
    data = event.data
    if event.type == "artifact.created":
        artifact = _params(data.get("artifact"))
        return f"artifact: {_string(artifact.get('name')) or 'created'}"
    if event.type == "verifier.completed":
        verification = _params(data.get("verification"))
        return "verification: passed" if verification.get("passed") is True else "verification: failed"
    if event.type == "skill.completed":
        return f"skill completed: {_string(data.get('skill_name')) or ''}".strip()
    return event.type


def _params(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
