from __future__ import annotations

import json
from pathlib import Path

import httpx

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


def test_mcp_streamable_http_provider_calls_remote_tool(monkeypatch) -> None:
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


def test_mcp_streamable_http_provider_lists_tools_from_sse(monkeypatch) -> None:
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


def test_mcp_streamable_http_provider_does_not_send_unrelated_llm_key(monkeypatch) -> None:
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
