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
        if normalized in {"prompt", "session/prompt"}:
            return await self.adapter.prompt(sessions, params, send_update)
        if normalized in {"list_agents", "jetlinks/list_agents", "jetlinks/agents/list"}:
            return self.adapter.list_agents()
        if normalized in {"cancel", "session/cancel"}:
            return {"stopReason": "cancelled"}
        raise ValueError(f"Unsupported ACP method: {method}")

    async def close(self, sessions: dict[str, AcpWebSocketSession]) -> None:
        await self.adapter.close_sessions(sessions)
