from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.api.auth import require_admin_token
from app.core.mcp import McpToolRegistry
from app.core.runtime import default_container
from app.core.skills import SkillRunner
from app.protocols.mcp import (
    MCP_PROTOCOL_VERSION,
    McpDispatcher,
    McpRequestHeaders,
    handle_mcp_json_rpc,
)


router = APIRouter(tags=["mcp"])
registry = McpToolRegistry()
artifact_store = default_container.runtime.artifact_store
skill_runner = SkillRunner(artifact_store)
_test_user_loop_counts: dict[str, int] = {}

__all__ = [
    "MCP_PROTOCOL_VERSION",
    "artifact_store",
    "registry",
    "router",
    "skill_runner",
]


@router.get("/api/mcp/tools")
async def list_mcp_tools() -> dict[str, list[dict[str, Any]]]:
    return {"tools": [tool.to_payload() for tool in registry.list(include_disabled=True, include_skill_tools=False)]}


@router.get("/api/mcp/tools/{tool_name}")
async def get_mcp_tool(tool_name: str) -> dict[str, Any]:
    try:
        return registry.get(tool_name).to_payload()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/api/mcp/tools/{tool_name}", dependencies=[Depends(require_admin_token)])
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


@router.post("/api/mcp/test-users/reset")
async def reset_test_users_mcp(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_id = _test_users_session_id(payload or {})
    _test_user_loop_counts.pop(session_id, None)
    return {"ok": True, "sessionId": session_id}


@router.post("/api/mcp/test-users")
async def test_users_mcp_json_rpc(request: Request) -> Response:
    """Small deterministic MCP server for conditional loop smoke tests."""

    payload = await request.json()
    if not isinstance(payload, dict):
        return _test_users_error(None, -32600, "JSON-RPC payload must be an object.")
    request_id = payload.get("id")
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        return _test_users_error(request_id, -32600, "JSON-RPC method is required.")

    if method == "initialize":
        return _test_users_result(
            request_id,
            {
                "protocolVersion": str(
                    (payload.get("params") if isinstance(payload.get("params"), dict) else {}).get(
                        "protocolVersion"
                    )
                    or MCP_PROTOCOL_VERSION
                ),
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "jetlinks-test-users-mcp", "version": "0.1.0"},
            },
        )
    if method == "notifications/initialized":
        return Response(status_code=202)
    if method == "ping":
        return _test_users_result(request_id, {})
    if method == "tools/list":
        return _test_users_result(
            request_id,
            {
                "tools": [
                    {
                        "name": "query_users",
                        "description": (
                            "Test-only user query tool. By default the first two calls return "
                            "two super-admin rows and later calls return an empty rows array. "
                            "Set mode=always_present to keep returning rows for infinite-loop "
                            "smoke tests."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "keyword": {
                                    "type": "string",
                                    "description": "Optional user keyword.",
                                },
                                "sessionId": {
                                    "type": "string",
                                    "description": "Optional test session id for counter isolation.",
                                },
                                "mode": {
                                    "type": "string",
                                    "enum": ["eventually_empty", "always_present"],
                                    "description": (
                                        "eventually_empty stops after emptyAfter calls; always_present "
                                        "keeps returning rows."
                                    ),
                                },
                                "emptyAfter": {
                                    "type": "integer",
                                    "description": "1-based call index that starts returning empty rows.",
                                    "minimum": 1,
                                    "default": 3,
                                },
                            },
                            "additionalProperties": True,
                        },
                    }
                ]
            },
        )
    if method == "tools/call":
        return _test_users_tool_call(request_id, payload.get("params"))
    return _test_users_error(request_id, -32601, f"Unsupported MCP method: {method}")


def _test_users_tool_call(request_id: object, raw_params: object) -> Response:
    params = raw_params if isinstance(raw_params, dict) else {}
    if params.get("name") != "query_users":
        return _test_users_error(request_id, -32602, "Unknown test MCP tool.")
    arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
    session_id = _test_users_session_id(arguments)
    call_count = _test_user_loop_counts.get(session_id, 0) + 1
    _test_user_loop_counts[session_id] = call_count
    rows = _test_users_rows_for_call(arguments, call_count)
    structured = {
        "rows": rows,
        "count": len(rows),
        "callIndex": call_count,
        "sessionId": session_id,
        "mode": _test_users_mode(arguments),
    }
    return _test_users_result(
        request_id,
        {
            "content": [{"type": "text", "text": _json_text(structured)}],
            "structuredContent": structured,
            "isError": False,
        },
    )


def _test_users_rows_for_call(arguments: dict[str, Any], call_count: int) -> list[dict[str, Any]]:
    if _test_users_mode(arguments) == "always_present":
        return _test_super_admin_rows()
    return [] if call_count >= _test_users_empty_after(arguments) else _test_super_admin_rows()


def _test_users_mode(arguments: dict[str, Any]) -> str:
    value = str(arguments.get("mode") or "eventually_empty").strip().lower()
    if value in {"always_present", "always-present", "infinite", "always"}:
        return "always_present"
    return "eventually_empty"


def _test_users_empty_after(arguments: dict[str, Any]) -> int:
    try:
        value = int(arguments.get("emptyAfter") or arguments.get("empty_after") or 3)
    except (TypeError, ValueError):
        return 3
    return max(1, min(value, 1000000))


def _test_super_admin_rows() -> list[dict[str, Any]]:
    return [
        {
            "username": "admin",
            "name": "超级管理员",
            "role": "超级管理员",
            "status": 1,
        },
        {
            "username": "pm_8356f1b5b99601d1",
            "name": "超级管理员",
            "role": "超级管理员",
            "status": 1,
        },
    ]


def _test_users_session_id(arguments: dict[str, Any]) -> str:
    value = arguments.get("sessionId") or arguments.get("session_id") or "default"
    return str(value).strip()[:80] or "default"


def _test_users_result(request_id: object, result: dict[str, Any]) -> Response:
    if request_id is None:
        return Response(status_code=202)

    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _test_users_error(request_id: object, code: int, message: str) -> Response:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


def _json_text(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)
