from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class McpStdioToolProvider:
    """Invoke an external MCP stdio server for a single configured tool call."""

    source_type = "mcp_stdio"

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.root_dir = Path(__file__).resolve().parents[4]

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
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

    def list_remote_tools_with_error(self, server: dict[str, Any]) -> tuple[list[ToolDefinition], str | None]:
        probe = ToolDefinition(
            name=f"{_safe_tool_prefix(server)}__list_tools",
            title=f"{_string(server.get('name')) or 'stdio'} tools",
            description="List remote MCP stdio tools.",
            source=_source_from_server(server),
            editable=False,
        )
        try:
            raw_tools = self._list_tools(probe, {})
        except Exception as exc:
            return [], str(exc)
        prefix = _safe_tool_prefix(server)
        definitions: list[ToolDefinition] = []
        for raw_tool in raw_tools:
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
                    source=_source_from_server(server, server_tool=remote_name),
                    editable=False,
                )
            )
        return definitions, None

    def _call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
        command = _string(tool.source.get("command"))
        if not command:
            raise ValueError("mcp_stdio source.command is required")
        server_tool = _string(tool.source.get("server_tool")) or tool.name
        timeout = _bounded_int(tool.source.get("timeout_seconds"), default=15, minimum=1, maximum=120)
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
                        "clientInfo": {"name": "jetlinks-agent-runtime-v2", "version": "0.1.0"},
                    },
                },
            )
            self._read_response(proc, next_id, timeout)
            self._send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            next_id += 1
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": next_id,
                    "method": "tools/call",
                    "params": {"name": server_tool, "arguments": self._server_arguments(tool, arguments)},
                },
            )
            return self._read_response(proc, next_id, timeout)
        finally:
            _close_process(proc)

    def _list_tools(self, tool: ToolDefinition, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        command = _string(tool.source.get("command"))
        if not command:
            raise ValueError("mcp_stdio source.command is required")
        timeout = _bounded_int(tool.source.get("timeout_seconds"), default=15, minimum=1, maximum=120)
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
                        "clientInfo": {"name": "jetlinks-agent-runtime-v2", "version": "0.1.0"},
                    },
                },
            )
            self._read_response(proc, next_id, timeout)
            self._send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
            next_id += 1
            self._send(proc, {"jsonrpc": "2.0", "id": next_id, "method": "tools/list", "params": {}})
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
        args = [_expand_runtime_value(str(item), tool, arguments, self.artifact_store) for item in _string_list(tool.source.get("args"))]
        cwd = _expand_runtime_value(_string(tool.source.get("cwd")) or str(self.root_dir), tool, arguments, self.artifact_store)
        env = os.environ.copy()
        raw_env = tool.source.get("env")
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                env[str(key)] = _expand_runtime_value(str(value), tool, arguments, self.artifact_store)
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

    def _server_arguments(self, tool: ToolDefinition, arguments: dict[str, Any]) -> dict[str, Any]:
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
        proc.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    def _read_response(self, proc: subprocess.Popen[str], request_id: int, timeout: int) -> dict[str, Any]:
        if proc.stdout is None:
            raise RuntimeError("MCP stdio process has no stdout")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"MCP stdio process exited with code {proc.returncode}: {self._stderr(proc)}")
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
        raise TimeoutError(f"MCP stdio tool call timed out after {timeout}s: {self._stderr(proc)}")

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
        content = response.get("content")
        if not isinstance(content, list):
            content = [{"type": "text", "text": json.dumps(response, ensure_ascii=False)}]
        structured = response.get("structuredContent")
        return ToolInvocationResult(
            content=[item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in content],
            structured_content=structured if isinstance(structured, dict) else {"tool_name": tool.name, "mcp_result": response},
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
        return [_expand_argument_value(item, tool, arguments, artifact_store) for item in value]
    if isinstance(value, dict):
        return {key: _expand_argument_value(item, tool, arguments, artifact_store) for key, item in value.items()}
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


def _source_from_server(server: dict[str, Any], *, server_tool: str | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {
        "type": McpStdioToolProvider.source_type,
        "command": _string(server.get("command")),
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
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)
    safe = safe.strip("._-") or "stdio_mcp_tool"
    return safe[:128]
