from __future__ import annotations

from fastapi import APIRouter, WebSocket

from app.api.auth import require_websocket_runtime_token
from app.core.agent import AgentRuntime
from app.core.config import AgentConfigLoader
from app.protocols.acp import AcpDispatcher, AcpRuntimeAdapter, AcpWebSocketSession, JsonRpcId, handle_acp_websocket


router = APIRouter(prefix="/api/acp", tags=["acp"])
loader = AgentConfigLoader()
runtime = AgentRuntime()

__all__ = ["AcpWebSocketSession", "JsonRpcId", "loader", "router", "runtime"]


@router.websocket("/ws")
async def acp_websocket(websocket: WebSocket, token: str | None = None) -> None:
    await require_websocket_runtime_token(websocket, token=token)
    adapter = AcpRuntimeAdapter(loader=loader, runtime=runtime)
    dispatcher = AcpDispatcher(adapter)
    await handle_acp_websocket(websocket, dispatcher)
