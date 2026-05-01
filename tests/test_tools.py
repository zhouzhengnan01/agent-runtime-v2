from __future__ import annotations

from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner
from app.core.tools.providers.mcp_stdio import McpStdioToolProvider
from app.core.tools.schemas import ToolDefinition
from app.core.tools import ToolInvocationService, ToolRegistry


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


def test_mcp_stdio_provider_maps_external_mcp_result(tmp_path: Path, monkeypatch) -> None:
    tool = ToolDefinition(
        name="mcp_fs_read_text_file",
        title="MCP read file",
        description="Read via external MCP.",
        source={"type": "mcp_stdio", "server_tool": "read_text_file"},
    )
    provider = McpStdioToolProvider(artifact_store=ArtifactStore(root_dir=tmp_path / "runtime"))
    seen: dict[str, object] = {}

    def fake_call(self: McpStdioToolProvider, call_tool: ToolDefinition, arguments: dict[str, object]) -> dict[str, object]:
        seen["tool"] = call_tool.name
        seen["arguments"] = arguments
        return {
            "content": [{"type": "text", "text": "hello from external mcp"}],
            "structuredContent": {"content": "hello from external mcp"},
        }

    monkeypatch.setattr(McpStdioToolProvider, "_call", fake_call)

    result = provider.call(tool, {"path": "/tmp/example.txt", "_thread_id": "mcp-stdio-unit"})

    assert result.is_error is False
    assert result.content[0]["text"] == "hello from external mcp"
    assert result.structured_content["content"] == "hello from external mcp"
    assert seen["tool"] == "mcp_fs_read_text_file"


def test_mcp_stdio_provider_expands_schema_defaults(tmp_path: Path, monkeypatch) -> None:
    tool = ToolDefinition(
        name="mcp_fs_list_directory",
        title="MCP list directory",
        description="List via external MCP.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "{workspace}"},
            },
        },
        source={"type": "mcp_stdio", "server_tool": "list_directory"},
    )
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    provider = McpStdioToolProvider(artifact_store=store)

    arguments = provider._server_arguments(tool, {"_thread_id": "default-path"})

    assert arguments == {"path": str(store.prepare_thread("default-path").workspace.resolve())}
