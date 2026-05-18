from __future__ import annotations

import json
from pathlib import Path
import sys

import httpx
from pytest import MonkeyPatch

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner
from app.core.tools import ToolInvocationService, ToolRegistry
from app.core.tools.providers.mcp_streamable_http import McpStreamableHttpToolProvider
from app.core.tools.schemas import ToolDefinition


def test_tool_invocation_service_keeps_manual_tools_and_skills_as_providers(tmp_path: Path) -> None:
    config_dir = tmp_path / "config" / "mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "tools.json").write_text(
        """
{
  "tools": [
    {
      "name": "jetlinks_runtime_status",
      "title": "JetLinks Runtime Status",
      "description": "Return a simple status message.",
      "enabled": true,
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"},
      "source": {"type": "manual", "response_template": "ok"}
    }
  ]
}
""",
        encoding="utf-8",
    )

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    registry = ToolRegistry(tmp_path, artifact_store=store, skill_runner=SkillRunner(store))
    service = ToolInvocationService(registry=registry, artifact_store=store, skill_runner=SkillRunner(store))

    tools = {tool.name: tool for tool in service.list_tools()}
    assert tools["jetlinks_runtime_status"].source["type"] == "manual"
    assert tools["markdown-rendering"].source["type"] == "skill"

    manual = service.call_tool("jetlinks_runtime_status", {"probe": True})
    assert manual.is_error is False
    assert manual.content[0]["text"] == "ok"

    skill = service.call_tool(
        "markdown-rendering",
        {"_thread_id": "tool-skill-test", "title": "Tool Layer", "summary": "Skill provider output"},
    )
    assert skill.is_error is False
    assert skill.structured_content["skill_name"] == "markdown-rendering"
    assert skill.structured_content["thread_id"] == "tool-skill-test"
    assert skill.structured_content["artifacts"][0]["name"] == "result.md"


def test_local_patch_file_replaces_unique_text_in_session_cwd(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    target = workspace / "src" / "large.py"
    target.write_text("A = 1\nTARGET_STATUS = \"BROKEN\"\nB = 2\n", encoding="utf-8")
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path / "runtime"))

    result = service.call_tool(
        "local_patch_file",
        {
            "_thread_id": "patch-test",
            "_session_cwd": str(workspace),
            "path": "src/large.py",
            "old_text": 'TARGET_STATUS = "BROKEN"',
            "new_text": 'TARGET_STATUS = "PATCHED"',
        },
    )

    assert result.is_error is False
    assert 'TARGET_STATUS = "PATCHED"' in target.read_text(encoding="utf-8")
    assert result.structured_content["path"] == "src/large.py"
    assert result.structured_content["replacements"] == 1
    assert "-TARGET_STATUS" in result.structured_content["diff"]
    assert "+TARGET_STATUS" in result.structured_content["diff"]


def test_local_patch_file_rejects_non_unique_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "dup.txt"
    target.write_text("same\nsame\n", encoding="utf-8")
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path / "runtime"))

    result = service.call_tool(
        "local_patch_file",
        {
            "_thread_id": "patch-test",
            "_session_cwd": str(workspace),
            "path": "dup.txt",
            "old_text": "same",
            "new_text": "changed",
        },
    )

    assert result.is_error is True
    assert "expected exactly one" in result.structured_content["error"]
    assert target.read_text(encoding="utf-8") == "same\nsame\n"


def test_mcp_streamable_http_provider_calls_remote_tool(monkeypatch: MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer env-key"
        payload = json.loads(request.content.decode("utf-8"))
        seen.append(payload)
        if payload["method"] == "initialize":
            return httpx.Response(
                200,
                headers={"Mcp-Session-Id": "session-1"},
                json={"jsonrpc": "2.0", "id": payload["id"], "result": {"protocolVersion": "2025-06-18"}},
            )
        if payload["method"] == "notifications/initialized":
            assert request.headers["mcp-session-id"] == "session-1"
            return httpx.Response(202)
        if payload["method"] == "tools/call":
            assert request.headers["mcp-session-id"] == "session-1"
            assert payload["params"] == {"name": "remote_search", "arguments": {"query": "JetLinks"}}
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "content": [{"type": "text", "text": "remote ok"}],
                        "structuredContent": {"items": [{"title": "JetLinks"}]},
                        "isError": False,
                    },
                },
            )
        raise AssertionError(payload)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    monkeypatch.setattr(httpx, "Client", lambda **_: original_client(transport=httpx.MockTransport(handler)))
    tool = ToolDefinition(
        name="remote_search_wrapper",
        title="Remote Search",
        description="Remote search",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        source={
            "type": "mcp_streamable_http",
            "url": "https://example.test/mcp",
            "server_tool": "remote_search",
            "headers": {"Authorization": "Bearer {api_key}"},
        },
    )

    result = McpStreamableHttpToolProvider().call(tool, {"query": "JetLinks"})

    assert result.is_error is False
    assert result.content[0]["text"] == "remote ok"
    assert result.structured_content["items"][0]["title"] == "JetLinks"
    assert [payload["method"] for payload in seen] == ["initialize", "notifications/initialized", "tools/call"]


def test_mcp_streamable_http_provider_lists_tools_from_sse(monkeypatch: MonkeyPatch) -> None:
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        payload = json.loads(request.content.decode("utf-8"))
        if payload["method"] == "initialize":
            body = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{}}\n\n'
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=body)
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        if payload["method"] == "tools/list":
            body = (
                'event: message\n'
                'data: {"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"remote_weather"}]}}\n\n'
            )
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=body)
        raise AssertionError(payload)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    monkeypatch.setattr(httpx, "Client", lambda **_: original_client(transport=httpx.MockTransport(handler)))
    tool = ToolDefinition(
        name="remote_list",
        title="Remote List",
        description="Remote list",
        source={
            "type": "mcp_streamable_http",
            "url": "https://example.test/mcp",
            "operation": "list_tools",
        },
    )

    result = McpStreamableHttpToolProvider().call(tool, {})

    assert result.is_error is False
    assert result.structured_content["tools"][0]["name"] == "remote_weather"
    assert "remote_weather" in result.content[0]["text"]


def test_mcp_streamable_http_provider_builds_runtime_tools_from_server(monkeypatch: MonkeyPatch) -> None:
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer runtime-token"
        payload = json.loads(request.content.decode("utf-8"))
        if payload["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {}})
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        if payload["method"] == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "tools": [
                            {
                                "name": "runtime_status",
                                "description": "Return runtime status.",
                                "inputSchema": {"type": "object", "properties": {"detail": {"type": "boolean"}}},
                            }
                        ]
                    },
                },
            )
        raise AssertionError(payload)

    monkeypatch.setattr(httpx, "Client", lambda **_: original_client(transport=httpx.MockTransport(handler)))
    tools = McpStreamableHttpToolProvider().list_remote_tools(
        {
            "name": "jetlinks-session",
            "url": "https://example.test/mcp",
            "type": "http",
            "headers": [{"name": "Authorization", "value": "Bearer runtime-token"}],
        }
    )

    assert [tool.name for tool in tools] == ["jetlinks-session__runtime_status"]
    assert tools[0].source["server_tool"] == "runtime_status"
    assert tools[0].source["headers"]["Authorization"] == "Bearer runtime-token"
    assert tools[0].input_schema["properties"]["detail"]["type"] == "boolean"


def test_tool_invocation_service_caches_runtime_mcp_tool_discovery(monkeypatch: MonkeyPatch) -> None:
    original_client = httpx.Client
    seen_methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_methods.append(payload["method"])
        if payload["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {}})
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        if payload["method"] == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"tools": [{"name": "runtime_status"}]},
                },
            )
        raise AssertionError(payload)

    monkeypatch.setenv("RUNTIME_MCP_TOOLS_CACHE_SECONDS", "60")
    monkeypatch.setattr(httpx, "Client", lambda **_: original_client(transport=httpx.MockTransport(handler)))
    service = ToolInvocationService()
    servers = [{"name": "jetlinks-session", "url": "https://example.test/mcp", "type": "http"}]

    first = service.discover_runtime_mcp_tools(servers)
    second = service.discover_runtime_mcp_tools(servers)

    assert [tool.name for tool in first.tools] == ["jetlinks-session__runtime_status"]
    assert [tool.name for tool in second.tools] == ["jetlinks-session__runtime_status"]
    assert seen_methods == ["initialize", "notifications/initialized", "tools/list"]


def test_tool_invocation_service_reports_runtime_mcp_policy_failures(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RUNTIME_MCP_ALLOWED_HOSTS", raising=False)
    service = ToolInvocationService()

    result = service.discover_runtime_mcp_tools(
        [{"name": "metadata", "url": "http://169.254.169.254/latest", "type": "http"}]
    )

    assert result.tools == []
    assert len(result.failures) == 1
    assert result.failures[0].reason == "blocked_host"
    assert result.failures[0].endpoint == "http://169.254.169.254"


def test_tool_invocation_service_supports_runtime_mcp_allowed_host_policy(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_MCP_ALLOWED_HOSTS", "allowed.example.test")
    service = ToolInvocationService()

    result = service.discover_runtime_mcp_tools(
        [{"name": "denied", "url": "https://denied.example.test/mcp", "type": "http"}]
    )

    assert result.tools == []
    assert len(result.failures) == 1
    assert result.failures[0].reason == "host_not_allowed"


def test_tool_invocation_service_discovers_and_calls_runtime_stdio_mcp_tools(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    server_script = tmp_path / "stdio_mcp_server.py"
    server_script.write_text(
        """
from __future__ import annotations

import json
import sys

for line in sys.stdin:
    payload = json.loads(line)
    method = payload.get("method")
    if method == "notifications/initialized":
        continue
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18"}
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "runtime_status",
                    "description": "Return runtime status.",
                    "inputSchema": {"type": "object", "properties": {"detail": {"type": "boolean"}}},
                }
            ]
        }
    elif method == "tools/call":
        result = {
            "content": [{"type": "text", "text": "stdio runtime ok"}],
            "structuredContent": {"tool": payload["params"]["name"], "arguments": payload["params"]["arguments"]},
        }
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("RUNTIME_MCP_STDIO_ENABLED", "true")
    service = ToolInvocationService()
    servers = [
        {
            "name": "stdio-session",
            "type": "stdio",
            "command": sys.executable,
            "args": [str(server_script)],
        }
    ]

    discovery = service.discover_runtime_mcp_tools(servers)

    assert discovery.failures == []
    assert [tool.name for tool in discovery.tools] == ["stdio-session__runtime_status"]
    result = service.call_tool(
        "stdio-session__runtime_status",
        {
            "detail": True,
            "_runtime_mcp_tools": [tool.to_payload() for tool in discovery.tools],
        },
    )
    assert result.is_error is False
    assert result.content[0]["text"] == "stdio runtime ok"
    assert result.structured_content["tool"] == "runtime_status"
    assert result.structured_content["arguments"] == {"detail": True}


def test_tool_invocation_service_requires_runtime_stdio_mcp_opt_in(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RUNTIME_MCP_STDIO_ENABLED", raising=False)
    service = ToolInvocationService()

    result = service.discover_runtime_mcp_tools(
        [{"name": "stdio-session", "type": "stdio", "command": sys.executable}]
    )

    assert result.tools == []
    assert len(result.failures) == 1
    assert result.failures[0].reason == "stdio_disabled"


def test_mcp_streamable_http_provider_does_not_send_unrelated_llm_key(monkeypatch: MonkeyPatch) -> None:
    seen_headers: list[httpx.Headers] = []
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers)
        payload = json.loads(request.content.decode("utf-8"))
        if payload["method"] == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {}})
        if payload["method"] == "notifications/initialized":
            return httpx.Response(202)
        if payload["method"] == "tools/call":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"content": [{"type": "text", "text": "public ok"}]},
                },
            )
        raise AssertionError(payload)

    monkeypatch.setenv("LLM_API_KEY", "llm-key-should-not-be-sent")
    monkeypatch.setattr(httpx, "Client", lambda **_: original_client(transport=httpx.MockTransport(handler)))
    tool = ToolDefinition(
        name="public_remote_wrapper",
        title="Public Remote",
        description="Public remote",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        source={
            "type": "mcp_streamable_http",
            "url": "https://example.test/mcp",
            "server_tool": "public_search",
        },
    )

    result = McpStreamableHttpToolProvider().call(tool, {"query": "JetLinks"})

    assert result.is_error is False
    assert result.content[0]["text"] == "public ok"
    assert seen_headers
    assert all("authorization" not in headers for headers in seen_headers)
