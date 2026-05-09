from __future__ import annotations

from typing import Any

from app.protocols.acp.adapter import AcpRuntimeAdapter, AcpUpdateSender
from app.protocols.acp.schemas import AcpWebSocketSession


class AcpDispatcher:
    """Dispatch ACP method names to protocol adapter operations."""

    def __init__(self, adapter: AcpRuntimeAdapter | None = None) -> None:
        self.adapter = adapter or AcpRuntimeAdapter()

    async def dispatch(
        self,
        sessions: dict[str, AcpWebSocketSession],
        method: str,
        params: dict[str, Any],
        send_update: AcpUpdateSender,
    ) -> dict[str, Any]:
        normalized = method.replace("-", "_")
        if normalized in {"initialize", "connection_initialize"}:
            return self.adapter.initialize()
        if normalized in {"new_session", "newsession", "session/new"}:
            return self.adapter.new_session(sessions, params)
        if normalized in {"session/load", "load_session"}:
            return self.adapter.load_session(sessions, params)
        if normalized in {"prompt", "session/prompt"}:
            return await self.adapter.prompt(sessions, params, send_update)
        if normalized in {"session/list", "list_sessions"}:
            return self.adapter.list_sessions(sessions)
        if normalized in {"session/close", "close_session"}:
            session_id = _required_session_id(params)
            return await self.adapter.close_session(sessions, session_id)
        if normalized in {"session/fork", "fork_session"}:
            return self.adapter.fork_session(sessions, params)
        if normalized in {"session/resume", "resume_session"}:
            return self.adapter.resume_session(sessions, params)
        if normalized in {"session/set_mode", "set_session_mode"}:
            session_id = _required_session_id(params)
            mode_id = _required_mode_id(params)
            return self.adapter.set_session_mode(sessions, session_id, mode_id)
        if normalized in {"session/set_model", "set_session_model"}:
            session_id = _required_session_id(params)
            model_name = _required_model_name(params)
            return self.adapter.set_session_model(sessions, session_id, model_name)
        if normalized in {"session/set_config_option", "set_config_option"}:
            return self.adapter.set_config_option(sessions, params)
        if normalized == "authenticate":
            return self.adapter.authenticate(params)
        if normalized in {"list_agents", "jetlinks/list_agents", "jetlinks/agents/list"}:
            return self.adapter.list_agents()
        if normalized in {"jetlinks/session/delete_files", "jetlinks/delete_session_files"}:
            return self.adapter.delete_session_files(sessions, params)
        if normalized in {"cancel", "session/cancel"}:
            cancel_session_id = _session_id(params)
            if cancel_session_id is not None:
                await self.adapter.cancel_session(sessions, cancel_session_id)
            return {"stopReason": "cancelled"}
        raise ValueError(f"Unsupported ACP method: {method}")

    async def close(self, sessions: dict[str, AcpWebSocketSession]) -> None:
        await self.adapter.close_sessions(sessions)


def _session_id(params: dict[str, Any]) -> str | None:
    value = params.get("sessionId") or params.get("session_id")
    return value if isinstance(value, str) and value.strip() else None


def _required_session_id(params: dict[str, Any]) -> str:
    session_id = _session_id(params)
    if session_id is None:
        raise ValueError("sessionId is required")
    return session_id


def _required_model_name(params: dict[str, Any]) -> str:
    value = params.get("modelId") or params.get("model_id") or params.get("modelName") or params.get("model_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("modelId is required")
    return value


def _required_mode_id(params: dict[str, Any]) -> str:
    value = params.get("modeId") or params.get("mode_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("modeId is required")
    return value
