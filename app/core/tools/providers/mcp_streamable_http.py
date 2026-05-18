from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app.core.config import AgentConfigLoader
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class McpStreamableHttpToolProvider:
    """Call a remote MCP server over Streamable HTTP."""

    source_type = "mcp_streamable_http"

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        try:
            result = self._call(tool, arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return result

    def list_remote_tools(self, server: dict[str, Any]) -> list[ToolDefinition]:
        tools, _error = self.list_remote_tools_with_error(server)
        return tools

    def list_remote_tools_with_error(self, server: dict[str, Any]) -> tuple[list[ToolDefinition], str | None]:
        probe = ToolDefinition(
            name=f"{_safe_tool_prefix(server)}__list_tools",
            title=f"{_string(server.get('name')) or 'remote'} tools",
            description="List remote MCP tools.",
            source=_source_from_server(server, operation="list_tools"),
            editable=False,
        )
        result = self.call(probe, {})
        if result.is_error:
            error = _string(result.structured_content.get("error")) or result.content[0].get("text", "")
            return [], error
        raw_tools = result.structured_content.get("tools")
        tools = raw_tools if isinstance(raw_tools, list) else []
        prefix = _safe_tool_prefix(server)
        definitions: list[ToolDefinition] = []
        for raw_tool in tools:
            if not isinstance(raw_tool, dict):
                continue
            remote_name = _string(raw_tool.get("name"))
            if not remote_name:
                continue
            input_schema = raw_tool.get("inputSchema") or raw_tool.get("input_schema") or {
                "type": "object",
                "additionalProperties": True,
            }
            definitions.append(
                ToolDefinition(
                    name=_safe_tool_name(f"{prefix}__{remote_name}"),
                    title=_string(raw_tool.get("title")) or remote_name,
                    description=_string(raw_tool.get("description")) or remote_name,
                    input_schema=input_schema if isinstance(input_schema, dict) else {},
                    output_schema={},
                    enabled=True,
                    source=_source_from_server(server, operation="call", server_tool=remote_name),
                    editable=False,
                )
            )
        return definitions, None

    def _call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        url = _string(tool.source.get("url"))
        if not url:
            raise ValueError("mcp_streamable_http source.url is required")

        operation = _string(tool.source.get("operation")) or "call"
        timeout = _bounded_float(tool.source.get("timeout_seconds"), default=30.0, minimum=1.0, maximum=300.0)
        protocol_version = _string(tool.source.get("protocol_version")) or "2025-06-18"
        headers = _headers(tool, arguments, protocol_version)
        next_id = 1

        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            initialized = _post_mcp(
                client,
                url,
                headers,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": protocol_version,
                        "capabilities": {},
                        "clientInfo": {"name": "jetlinks-agent-runtime-v2", "version": "0.1.0"},
                    },
                },
            )
            next_id += 1
            session_id = initialized.session_id
            if session_id:
                headers["Mcp-Session-Id"] = session_id
            _post_mcp(client, url, headers, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

            if operation == "list_tools":
                listed = _post_mcp(
                    client,
                    url,
                    headers,
                    {"jsonrpc": "2.0", "id": next_id, "method": "tools/list", "params": {}},
                )
                return _list_tools_result(tool, listed.payload)
            if operation != "call":
                raise ValueError(f"Unsupported mcp_streamable_http operation: {operation}")

            server_tool = _server_tool_name(tool, arguments)
            tool_arguments = _server_arguments(tool, arguments)
            called = _post_mcp(
                client,
                url,
                headers,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "tools/call",
                    "params": {"name": server_tool, "arguments": tool_arguments},
                },
            )
            return _tool_call_result(tool, called.payload)


class _McpHttpPayload:
    def __init__(self, payload: dict[str, Any], session_id: str | None = None) -> None:
        self.payload = payload
        self.session_id = session_id


def _post_mcp(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> _McpHttpPayload:
    response = client.post(url, headers=headers, json=payload)
    if response.status_code == 202:
        return _McpHttpPayload({}, _session_id(response))
    if response.status_code >= 400:
        detail = response.text.strip()
        message = f"remote MCP HTTP {response.status_code}"
        if detail:
            message = f"{message}: {detail[:1000]}"
        raise RuntimeError(message)

    parsed = _response_payload(response, payload.get("id"))
    if "error" in parsed:
        raise RuntimeError(str(parsed["error"]))
    result = parsed.get("result")
    return _McpHttpPayload(result if isinstance(result, dict) else {}, _session_id(response))


def _response_payload(response: httpx.Response, request_id: object) -> dict[str, Any]:
    text = response.text.strip()
    if not text:
        return {}
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type or text.startswith("event:") or text.startswith("data:"):
        return _sse_payload(text, request_id)
    data = response.json()
    return data if isinstance(data, dict) else {}


def _sse_payload(text: str, request_id: object) -> dict[str, Any]:
    fallback: dict[str, Any] = {}
    for block in text.split("\n\n"):
        data_lines: list[str] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            fallback = payload
            if request_id is None or payload.get("id") == request_id:
                return payload
    return fallback


def _session_id(response: httpx.Response) -> str | None:
    value = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
    return value if value and value.strip() else None


def _list_tools_result(tool: ToolDefinition, payload: dict[str, Any]) -> ToolInvocationResult:
    raw_tools = payload.get("tools")
    tools = raw_tools if isinstance(raw_tools, list) else []
    names = [str(item.get("name")) for item in tools if isinstance(item, dict) and item.get("name")]
    text = "Remote MCP tools:\n" + "\n".join(f"- {name}" for name in names) if names else "Remote MCP returned no tools."
    return ToolInvocationResult(
        content=[{"type": "text", "text": text}],
        structured_content={"tool_name": tool.name, "tools": tools},
        is_error=False,
    )


def _tool_call_result(tool: ToolDefinition, payload: dict[str, Any]) -> ToolInvocationResult:
    content = payload.get("content")
    if not isinstance(content, list):
        content = [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]
    structured = payload.get("structuredContent")
    return ToolInvocationResult(
        content=[item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in content],
        structured_content=structured if isinstance(structured, dict) else {"tool_name": tool.name, "mcp_result": payload},
        is_error=bool(payload.get("isError")),
    )


def _server_tool_name(tool: ToolDefinition, arguments: dict[str, Any]) -> str:
    configured = _string(tool.source.get("server_tool"))
    if configured:
        return configured
    from_arguments = _string(arguments.get("tool_name") or arguments.get("name"))
    if from_arguments:
        return from_arguments
    raise ValueError("mcp_streamable_http source.server_tool or arguments.tool_name is required")


def _server_arguments(tool: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_nested = arguments.get("arguments")
    if isinstance(raw_nested, dict):
        return {str(key): value for key, value in raw_nested.items() if not str(key).startswith("_")}
    defaults = _argument_defaults(tool.input_schema)
    merged = {**defaults, **arguments}
    return {
        str(key): value
        for key, value in merged.items()
        if str(key) not in {"tool_name", "name", "arguments"} and not str(key).startswith("_")
    }


def _source_from_server(
    server: dict[str, Any],
    *,
    operation: str,
    server_tool: str | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "type": McpStreamableHttpToolProvider.source_type,
        "operation": operation,
        "url": _string(server.get("url")),
        "auth": "none",
    }
    if server_tool:
        source["server_tool"] = server_tool
    headers = _headers_from_server(server)
    if headers:
        source["headers"] = headers
    meta = server.get("_meta")
    if isinstance(meta, dict):
        protocol_version = _string(meta.get("protocolVersion") or meta.get("protocol_version"))
        if protocol_version:
            source["protocol_version"] = protocol_version
        timeout = meta.get("timeoutSeconds") or meta.get("timeout_seconds")
        if isinstance(timeout, int | float):
            source["timeout_seconds"] = timeout
    return source


def _headers_from_server(server: dict[str, Any]) -> dict[str, str]:
    raw_headers = server.get("headers")
    if not isinstance(raw_headers, list):
        return {}
    headers: dict[str, str] = {}
    for item in raw_headers:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"))
        value = _string(item.get("value"))
        if name and value:
            headers[name] = value
    return headers


def _headers(tool: ToolDefinition, arguments: dict[str, Any], protocol_version: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": protocol_version,
    }
    api_key = _api_key(tool, arguments) if _uses_api_key(tool, arguments) else ""
    raw_headers = tool.source.get("headers")
    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            raw_value = str(value)
            if _has_api_key_placeholder(raw_value) and not api_key:
                continue
            rendered = _expand_secret_value(raw_value, api_key=api_key)
            if rendered:
                headers[str(key)] = rendered
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _uses_api_key(tool: ToolDefinition, arguments: dict[str, Any]) -> bool:
    auth = _string(tool.source.get("auth")).lower()
    if auth in {"none", "false", "disabled", "off", "no"}:
        return False
    if _string(arguments.get("_mcp_api_key")):
        return True
    if auth in {"bearer", "api_key", "apikey", "token"}:
        return True
    if _string(tool.source.get("api_key_env")) or _string_list(tool.source.get("api_key_envs")):
        return True
    if _string(tool.source.get("api_key_agent")):
        return True
    raw_headers = tool.source.get("headers")
    if isinstance(raw_headers, dict):
        return any(_has_api_key_placeholder(str(value)) for value in raw_headers.values())
    return False


def _api_key(tool: ToolDefinition, arguments: dict[str, Any]) -> str:
    argument_key = _string(arguments.get("_mcp_api_key"))
    if argument_key:
        return argument_key
    env_names = _string_list(tool.source.get("api_key_envs"))
    env_name = _string(tool.source.get("api_key_env"))
    if env_name:
        env_names.insert(0, env_name)
    for name in [*env_names, "DASHSCOPE_API_KEY", "BAILIAN_API_KEY"]:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    agent_name = _string(tool.source.get("api_key_agent")) or "default"
    try:
        api_key = AgentConfigLoader().load(agent_name).model.api_key
    except Exception:
        return ""
    return api_key.strip() if isinstance(api_key, str) else ""


def _expand_secret_value(value: str, *, api_key: str) -> str:
    return value.replace("{api_key}", api_key).replace("${api_key}", api_key)


def _has_api_key_placeholder(value: str) -> bool:
    return "{api_key}" in value or "${api_key}" in value


def _argument_defaults(input_schema: dict[str, Any]) -> dict[str, Any]:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    defaults: dict[str, Any] = {}
    for key, schema in properties.items():
        if isinstance(key, str) and isinstance(schema, dict) and "default" in schema:
            defaults[key] = schema["default"]
    return defaults


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _safe_tool_prefix(server: dict[str, Any]) -> str:
    return _safe_tool_name(_string(server.get("name")) or "remote_mcp")


def _safe_tool_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)
    safe = safe.strip("._-") or "remote_mcp_tool"
    return safe[:128]


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value) if isinstance(value, (str, int, float)) else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))
