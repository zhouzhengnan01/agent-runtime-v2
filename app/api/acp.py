from __future__ import annotations

from fastapi import APIRouter, WebSocket

from app.api.auth import require_websocket_runtime_token
from app.core.runtime import default_container
from app.protocols.acp import AcpDispatcher, AcpRuntimeAdapter, AcpWebSocketSession, JsonRpcId, handle_acp_websocket


router = APIRouter(prefix="/api/acp", tags=["acp"])
loader = default_container.loader
runtime = default_container.runtime
model_manager = default_container.model_manager

__all__ = ["AcpWebSocketSession", "JsonRpcId", "loader", "model_manager", "router", "runtime"]


@router.websocket("/ws")
async def acp_websocket(websocket: WebSocket, token: str | None = None) -> None:
    await require_websocket_runtime_token(websocket, token=token)
    adapter = AcpRuntimeAdapter(loader=loader, runtime=runtime, model_manager=model_manager)
    dispatcher = AcpDispatcher(adapter)
    await handle_acp_websocket(websocket, dispatcher)
