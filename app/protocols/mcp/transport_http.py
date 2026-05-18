from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from app.protocols.mcp.dispatcher import McpDispatcher
from app.protocols.mcp.schemas import McpRequestHeaders


async def handle_mcp_json_rpc(
    request: Request,
    dispatcher: McpDispatcher,
    headers: McpRequestHeaders | None = None,
) -> Response:
    payload = await request.json()
    if not isinstance(payload, dict):
        return _error(None, -32600, "JSON-RPC payload must be an object.")
    request_id = payload.get("id")
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        return _error(request_id, -32600, "JSON-RPC method is required.")

    active_headers = headers or McpRequestHeaders()
    if active_headers.method and active_headers.method != method:
        return _error(request_id, -32001, "Mcp-Method header does not match JSON-RPC method.")

    raw_params = payload.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    if method == "tools/call" and active_headers.name and active_headers.name != params.get("name"):
        return _error(request_id, -32002, "Mcp-Name header does not match tools/call name.")

    try:
        result = dispatcher.dispatch(method, params)
    except KeyError as exc:
        return _error(request_id, -32602, str(exc))
    except ValueError as exc:
        return _error(request_id, -32602, str(exc))
    except Exception as exc:
        return _error(request_id, -32000, str(exc))

    if request_id is None:
        return Response(status_code=202)
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id: object, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})

