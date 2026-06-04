from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.tools.providers.mcp_resources import (
    list_resource_templates_result,
    list_resources_result,
    read_resource_result,
    resource_cursor_params,
    resource_list_description,
    resource_list_input_schema,
    resource_template_list_description,
    resource_template_list_input_schema,
    resource_read_description,
    resource_read_input_schema,
    resource_uri,
    should_list_mcp_resource_templates,
    should_list_mcp_resources,
    should_list_mcp_tools,
)
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class McpStdioToolProvider:
    """Invoke an external MCP stdio server for a single configured tool call."""

    source_type = "mcp_stdio"

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.root_dir = Path(__file__).resolve().parents[4]

    def call(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> ToolInvocationResult:
        try:
            response = self._call(tool, arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return self._result(tool, response)

    def list_remote_tools(self, server: dict[str, Any]) -> list[ToolDefinition]:
        tools, _error = self.list_remote_tools_with_error(server)
        return tools

    def list_remote_tools_with_error(
        self, server: dict[str, Any]
    ) -> tuple[list[ToolDefinition], str | None]:
        probe = ToolDefinition(
            name=f"{_safe_tool_prefix(server)}__discover",
            title=f"{_string(server.get('name')) or 'stdio'} discovery",
            description="Discover remote MCP stdio tools and resources.",
            source=_source_from_server(server, operation="discover"),
            editable=False,
        )
        try:
            raw_tools, raw_resources, raw_templates, resources_available, errors = (
                self._discover_remote_capabilities(
                    probe,
                    {},
                    server,
                )
            )
        except Exception as exc:
            return [], str(exc)
        definitions = self._tool_definitions_from_raw_tools(server, raw_tools)
        if resources_available:
            definitions.extend(
                self._resource_tool_definitions(server, raw_resources, raw_templates)
            )
        if not definitions and errors:
            return [], "; ".join(errors)
        return definitions, None

    def _tool_definitions_from_raw_tools(
        self,
        server: dict[str, Any],
        raw_tools: list[dict[str, Any]],
    ) -> list[ToolDefinition]:
        prefix = _safe_tool_prefix(server)
        definitions: list[ToolDefinition] = []
        for raw_tool in raw_tools:
            remote_name = _string(raw_tool.get("name"))
            if not remote_name:
                continue
            input_schema = (
                raw_tool.get("inputSchema")
                or raw_tool.get("input_schema")
                or {
                    "type": "object",
                    "additionalProperties": True,
                }
            )
            definitions.append(
                ToolDefinition(
                    name=_safe_tool_name(f"{prefix}__{remote_name}"),
                    title=_string(raw_tool.get("title")) or remote_name,
                    description=_string(raw_tool.get("description")) or remote_name,
                    input_schema=input_schema if isinstance(input_schema, dict) else {},
                    output_schema={},
                    enabled=True,
                    source=_source_from_server(server, server_tool=remote_name),
                    editable=False,
                )
            )
        return definitions

    def _resource_tool_definitions(
        self,
        server: dict[str, Any],
        resources: list[dict[str, Any]],
        resource_templates: list[dict[str, Any]],
    ) -> list[ToolDefinition]:
        prefix = _safe_tool_prefix(server)
        definitions = [
            ToolDefinition(
                name=_safe_tool_name(f"{prefix}__mcp_list_resources"),
                title=f"{_string(server.get('name')) or 'stdio'} resources",
                description=resource_list_description(server),
                input_schema=resource_list_input_schema(),
                output_schema={},
                enabled=True,
                source=_source_from_server(server, operation="list_resources"),
                editable=False,
            ),
        ]
        if resource_templates:
            definitions.append(
                ToolDefinition(
                    name=_safe_tool_name(f"{prefix}__mcp_list_resource_templates"),
                    title=f"{_string(server.get('name')) or 'stdio'} resource templates",
                    description=resource_template_list_description(server),
                    input_schema=resource_template_list_input_schema(),
                    output_schema={},
                    enabled=True,
                    source=_source_from_server(
                        server, operation="list_resource_templates"
                    ),
                    editable=False,
                )
            )
        definitions.append(
            ToolDefinition(
                name=_safe_tool_name(f"{prefix}__mcp_read_resource"),
                title=f"{_string(server.get('name')) or 'stdio'} read resource",
                description=resource_read_description(
                    server, resources, resource_templates
                ),
                input_schema=resource_read_input_schema(resources, resource_templates),
                output_schema={},
                enabled=True,
                source=_source_from_server(server, operation="read_resource"),
                editable=False,
            )
        )
        return definitions

    def _call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
        command = _string(tool.source.get("command"))
        if not command:
            raise ValueError("mcp_stdio source.command is required")
        operation = _string(tool.source.get("operation")) or "call"
        server_tool = _string(tool.source.get("server_tool")) or tool.name
        timeout = _bounded_int(
            tool.source.get("timeout_seconds"), default=15, minimum=1, maximum=120
        )
        proc = self._start_process(tool, arguments, command)
        next_id = 1
        try:
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "jetlinks-agent-runtime-v2",
                            "version": "0.1.0",
                        },
                    },
                },
            )
            self._read_response(proc, next_id, timeout)
            self._send(
                proc,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )
            next_id += 1
            if operation == "list_resources":
                self._send(
                    proc,
                    {
                        "jsonrpc": "2.0",
                        "id": next_id,
                        "method": "resources/list",
                        "params": resource_cursor_params(arguments),
                    },
                )
                return self._read_response(proc, next_id, timeout)
            if operation == "list_resource_templates":
                self._send(
                    proc,
                    {
                        "jsonrpc": "2.0",
                        "id": next_id,
                        "method": "resources/templates/list",
                        "params": resource_cursor_params(arguments),
                    },
                )
                return self._read_response(proc, next_id, timeout)
            if operation == "read_resource":
                self._send(
                    proc,
                    {
                        "jsonrpc": "2.0",
                        "id": next_id,
                        "method": "resources/read",
                        "params": {"uri": resource_uri(tool, arguments)},
                    },
                )
                return self._read_response(proc, next_id, timeout)
            if operation != "call":
                raise ValueError(f"Unsupported mcp_stdio operation: {operation}")
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "tools/call",
                    "params": {
                        "name": server_tool,
                        "arguments": self._server_arguments(tool, arguments),
                    },
                },
            )
            return self._read_response(proc, next_id, timeout)
        finally:
            _close_process(proc)

    def _discover_remote_capabilities(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        server: dict[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        bool,
        list[str],
    ]:
        command = _string(tool.source.get("command"))
        if not command:
            raise ValueError("mcp_stdio source.command is required")
        timeout = _bounded_int(
            tool.source.get("timeout_seconds"), default=15, minimum=1, maximum=120
        )
        proc = self._start_process(tool, arguments, command)
        next_id = 1
        tools: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        resource_templates: list[dict[str, Any]] = []
        resources_available = False
        errors: list[str] = []
        try:
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "jetlinks-agent-runtime-v2",
                            "version": "0.1.0",
                        },
                    },
                },
            )
            initialized = self._read_response(proc, next_id, timeout)
            self._send(
                proc,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )
            next_id += 1
            capabilities = initialized.get("capabilities")
            if should_list_mcp_tools(capabilities):
                try:
                    self._send(
                        proc,
                        {
                            "jsonrpc": "2.0",
                            "id": next_id,
                            "method": "tools/list",
                            "params": {},
                        },
                    )
                    response = self._read_response(proc, next_id, timeout)
                    raw_tools = response.get("tools")
                    tools = (
                        [dict(item) for item in raw_tools if isinstance(item, dict)]
                        if isinstance(raw_tools, list)
                        else []
                    )
                except Exception as exc:
                    errors.append(f"tools/list failed: {exc}")
                finally:
                    next_id += 1
            if should_list_mcp_resources(capabilities, server):
                try:
                    self._send(
                        proc,
                        {
                            "jsonrpc": "2.0",
                            "id": next_id,
                            "method": "resources/list",
                            "params": {},
                        },
                    )
                    response = self._read_response(proc, next_id, timeout)
                    raw_resources = response.get("resources")
                    resources = (
                        [dict(item) for item in raw_resources if isinstance(item, dict)]
                        if isinstance(raw_resources, list)
                        else []
                    )
                    resources_available = True
                except Exception as exc:
                    errors.append(f"resources/list failed: {exc}")
                finally:
                    next_id += 1
            if should_list_mcp_resource_templates(capabilities, server):
                try:
                    self._send(
                        proc,
                        {
                            "jsonrpc": "2.0",
                            "id": next_id,
                            "method": "resources/templates/list",
                            "params": {},
                        },
                    )
                    response = self._read_response(proc, next_id, timeout)
                    raw_templates = response.get("resourceTemplates") or response.get(
                        "resource_templates"
                    )
                    resource_templates = (
                        [dict(item) for item in raw_templates if isinstance(item, dict)]
                        if isinstance(raw_templates, list)
                        else []
                    )
                    resources_available = True
                except Exception as exc:
                    errors.append(f"resources/templates/list failed: {exc}")
            return tools, resources, resource_templates, resources_available, errors
        finally:
            _close_process(proc)

    def _list_tools(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> list[dict[str, Any]]:
        command = _string(tool.source.get("command"))
        if not command:
            raise ValueError("mcp_stdio source.command is required")
        timeout = _bounded_int(
            tool.source.get("timeout_seconds"), default=15, minimum=1, maximum=120
        )
        proc = self._start_process(tool, arguments, command)
        next_id = 1
        try:
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "jetlinks-agent-runtime-v2",
                            "version": "0.1.0",
                        },
                    },
                },
            )
            self._read_response(proc, next_id, timeout)
            self._send(
                proc,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            )
            next_id += 1
            self._send(
                proc,
                {"jsonrpc": "2.0", "id": next_id, "method": "tools/list", "params": {}},
            )
            response = self._read_response(proc, next_id, timeout)
            tools = response.get("tools")
            return tools if isinstance(tools, list) else []
        finally:
            _close_process(proc)

    def _start_process(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        command: str,
    ) -> subprocess.Popen[str]:
        args = [
            _expand_runtime_value(str(item), tool, arguments, self.artifact_store)
            for item in _string_list(tool.source.get("args"))
        ]
        cwd = _expand_runtime_value(
            _string(tool.source.get("cwd")) or str(self.root_dir),
            tool,
            arguments,
            self.artifact_store,
        )
        env = os.environ.copy()
        raw_env = tool.source.get("env")
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                env[str(key)] = _expand_runtime_value(
                    str(value), tool, arguments, self.artifact_store
                )
        return subprocess.Popen(
            [command, *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _server_arguments(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        defaults = _argument_defaults(tool.input_schema)
        merged = {**defaults, **arguments}
        return {
            key: _expand_argument_value(value, tool, arguments, self.artifact_store)
            for key, value in merged.items()
            if not key.startswith("_")
        }

    @staticmethod
    def _send(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("MCP stdio process has no stdin")
        proc.stdin.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        proc.stdin.flush()

    def _read_response(
        self, proc: subprocess.Popen[str], request_id: int, timeout: int
    ) -> dict[str, Any]:
        if proc.stdout is None:
            raise RuntimeError("MCP stdio process has no stdout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"MCP stdio process exited with code {proc.returncode}: {self._stderr(proc)}"
                )
            ready, _, _ = select.select([proc.stdout.fileno()], [], [], 0.1)
            if not ready:
                continue
            line = proc.stdout.readline()
            if not line:
                continue
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(str(message["error"]))
            result = message.get("result")
            if not isinstance(result, dict):
                return {}
            return result
        raise TimeoutError(
            f"MCP stdio tool call timed out after {timeout}s: {self._stderr(proc)}"
        )

    @staticmethod
    def _stderr(proc: subprocess.Popen[str]) -> str:
        if proc.stderr is None:
            return ""
        try:
            ready, _, _ = select.select([proc.stderr.fileno()], [], [], 0)
            return proc.stderr.read(4000) if ready else ""
        except Exception:
            return ""

    @staticmethod
    def _result(tool: ToolDefinition, response: dict[str, Any]) -> ToolInvocationResult:
        operation = _string(tool.source.get("operation")) or "call"
        if operation == "list_resources":
            return list_resources_result(tool, response)
        if operation == "list_resource_templates":
            return list_resource_templates_result(tool, response)
        if operation == "read_resource":
            return read_resource_result(tool, response)
        content = response.get("content")
        if not isinstance(content, list):
            content = [
                {"type": "text", "text": json.dumps(response, ensure_ascii=False)}
            ]
        structured = response.get("structuredContent")
        return ToolInvocationResult(
            content=[
                item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                for item in content
            ],
            structured_content=(
                structured
                if isinstance(structured, dict)
                else {"tool_name": tool.name, "mcp_result": response}
            ),
            is_error=bool(response.get("isError")),
        )


def _close_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if isinstance(value, (str, int, float)) else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _expand_runtime_value(
    value: str,
    tool: ToolDefinition,
    arguments: dict[str, Any],
    artifact_store: ArtifactStore,
) -> str:
    thread_id = str(arguments.get("_thread_id") or "mcp-stdio")
    paths = artifact_store.prepare_thread(thread_id)
    return (
        value.replace("{thread_id}", thread_id)
        .replace("{workspace}", str(paths.workspace.resolve()))
        .replace("{outputs}", str(paths.outputs.resolve()))
        .replace("{runtime_root}", str(artifact_store.root_dir.resolve()))
        .replace("{tool_name}", tool.name)
    )


def _expand_argument_value(
    value: object,
    tool: ToolDefinition,
    arguments: dict[str, Any],
    artifact_store: ArtifactStore,
) -> object:
    if isinstance(value, str):
        return _expand_runtime_value(value, tool, arguments, artifact_store)
    if isinstance(value, list):
        return [
            _expand_argument_value(item, tool, arguments, artifact_store)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _expand_argument_value(item, tool, arguments, artifact_store)
            for key, item in value.items()
        }
    return value


def _argument_defaults(input_schema: dict[str, Any]) -> dict[str, Any]:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    defaults: dict[str, Any] = {}
    for key, schema in properties.items():
        if isinstance(key, str) and isinstance(schema, dict) and "default" in schema:
            defaults[key] = schema["default"]
    return defaults


def _source_from_server(
    server: dict[str, Any],
    *,
    server_tool: str | None = None,
    operation: str = "call",
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "type": McpStdioToolProvider.source_type,
        "command": _string(server.get("command")),
        "operation": operation,
    }
    if server_tool:
        source["server_tool"] = server_tool
    args = _string_list(server.get("args"))
    if args:
        source["args"] = args
    env = _env_from_server(server)
    if env:
        source["env"] = env
    cwd = _string(server.get("cwd"))
    if cwd:
        source["cwd"] = cwd
    meta = server.get("_meta")
    if isinstance(meta, dict):
        timeout = meta.get("timeoutSeconds") or meta.get("timeout_seconds")
        if isinstance(timeout, int | float):
            source["timeout_seconds"] = timeout
        cwd = _string(meta.get("cwd"))
        if cwd:
            source["cwd"] = cwd
    return source


def _env_from_server(server: dict[str, Any]) -> dict[str, str]:
    raw_env = server.get("env")
    if isinstance(raw_env, dict):
        return {str(key): str(value) for key, value in raw_env.items()}
    if not isinstance(raw_env, list):
        return {}
    env: dict[str, str] = {}
    for item in raw_env:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name"))
        value = _string(item.get("value"))
        if name:
            env[name] = value
    return env


def _safe_tool_prefix(server: dict[str, Any]) -> str:
    return _safe_tool_name(_string(server.get("name")) or "stdio_mcp")


def _safe_tool_name(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value
    )
    safe = safe.strip("._-") or "stdio_mcp_tool"
    return safe[:128]
