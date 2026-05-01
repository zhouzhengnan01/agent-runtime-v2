from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response

from app.core.artifacts import ArtifactStore
from app.core.mcp import McpToolRegistry
from app.core.skills import SkillRunner
from app.protocols.mcp import (
    MCP_PROTOCOL_VERSION,
    McpDispatcher,
    McpRequestHeaders,
    handle_mcp_json_rpc,
)


router = APIRouter(tags=["mcp"])
registry = McpToolRegistry()
artifact_store = ArtifactStore()
skill_runner = SkillRunner(artifact_store)

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "artifact_store",
    "registry",
    "router",
    "skill_runner",
]


@router.get("/api/mcp/tools")
async def list_mcp_tools() -> dict[str, list[dict[str, Any]]]:
    return {"tools": [tool.to_payload() for tool in registry.list_custom(include_disabled=True)]}


@router.get("/api/mcp/runtime-tools")
async def list_mcp_runtime_tools() -> dict[str, list[dict[str, Any]]]:
    return {"tools": [tool.to_payload() for tool in registry.list(include_disabled=True)]}


@router.get("/api/mcp/tools/{tool_name}")
async def get_mcp_tool(tool_name: str) -> dict[str, Any]:
    try:
        return registry.get_custom(tool_name).to_payload()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/api/mcp/tools/{tool_name}")
async def save_mcp_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("tool") if isinstance(payload.get("tool"), dict) else payload
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=400, detail="MCP tool manifest must be a JSON object.")
    try:
        return registry.save_custom_tool(tool_name, manifest).to_payload()
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/mcp")
async def mcp_json_rpc(
    request: Request,
    mcp_method: str | None = Header(default=None, alias="Mcp-Method"),
    mcp_name: str | None = Header(default=None, alias="Mcp-Name"),
) -> Response:
    dispatcher = McpDispatcher(
        registry=registry,
        artifact_store=artifact_store,
        skill_runner=skill_runner,
    )
    headers = McpRequestHeaders(method=mcp_method, name=mcp_name)
    return await handle_mcp_json_rpc(request, dispatcher, headers)
