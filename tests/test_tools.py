from __future__ import annotations

from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner
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

