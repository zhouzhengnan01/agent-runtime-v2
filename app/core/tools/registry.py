from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRegistry, SkillRunner
from app.core.tools.providers.delegate import delegate_tool_definitions
from app.core.tools.providers.local import local_tool_definitions
from app.core.tools.providers.markdown_memory import markdown_memory_tool_definitions
from app.core.tools.providers.memory import memory_tool_definitions
from app.core.tools.providers.skill import SkillToolProvider
from app.core.tools.schemas import ToolDefinition


_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class ToolRegistry:
    """Protocol-neutral registry for internal and configured tools."""

    def __init__(
        self,
        root_dir: Path | None = None,
        skill_registry: SkillRegistry | None = None,
        artifact_store: ArtifactStore | None = None,
        skill_runner: SkillRunner | None = None,
    ) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.config_path = self.root_dir / "config" / "mcp" / "tools.json"
        self.skill_provider = SkillToolProvider(
            artifact_store=artifact_store,
            skill_registry=skill_registry or SkillRegistry(self.root_dir),
            skill_runner=skill_runner,
        )

    def list(self, *, include_disabled: bool = False, include_skill_tools: bool = True) -> List[ToolDefinition]:
        skill_tools = self._skill_tools() if include_skill_tools else []
        tools = self._builtin_tools() + skill_tools
        custom = self._custom_tools()
        skill_names = {tool.name for tool in tools}
        tools.extend(tool for tool in custom if tool.name not in skill_names)
        result = tools if include_disabled else [tool for tool in tools if tool.enabled]
        return sorted(result, key=lambda tool: tool.name)

    def get(self, name: str) -> ToolDefinition:
        for tool in self.list(include_disabled=True):
            if tool.name == name:
                return tool
        raise KeyError(f"Unknown tool: {name}")

    def save_custom_tool(self, name: str, data: dict[str, Any]) -> ToolDefinition:
        _validate_tool_name(name)
        if name in {tool.name for tool in self._skill_tools()}:
            raise ValueError(f"Tool name conflicts with a skill-backed tool: {name}")
        tool = _tool_from_data({**data, "name": name}, editable=True)
        tools = [item.to_payload() for item in self._custom_tools() if item.name != name]
        tools.append(tool.to_payload())
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        target = self.config_path.with_suffix(".json.tmp")
        target.write_text(json.dumps({"tools": tools}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(self.config_path)
        return tool

    def _skill_tools(self) -> List[ToolDefinition]:
        return self.skill_provider.list()

    @staticmethod
    def _builtin_tools() -> List[ToolDefinition]:
        return (
            local_tool_definitions()
            + delegate_tool_definitions()
            + memory_tool_definitions()
            + markdown_memory_tool_definitions()
        )

    def _custom_tools(self) -> List[ToolDefinition]:
        if not self.config_path.is_file():
            return []
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw_tools = data.get("tools") if isinstance(data, dict) else []
        if not isinstance(raw_tools, list):
            return []
        tools: List[ToolDefinition] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            try:
                tools.append(_tool_from_data(item, editable=True))
            except ValueError:
                continue
        return tools


def _tool_from_data(data: dict[str, Any], *, editable: bool) -> ToolDefinition:
    name = str(data.get("name") or "").strip()
    _validate_tool_name(name)
    input_schema = _dict_value(data.get("input_schema")) or _dict_value(data.get("inputSchema")) or {
        "type": "object",
        "additionalProperties": True,
    }
    output_schema = _dict_value(data.get("output_schema")) or _dict_value(data.get("outputSchema")) or {}
    source = _dict_value(data.get("source")) or {"type": "manual"}
    return ToolDefinition(
        name=name,
        title=str(data.get("title") or name),
        description=str(data.get("description") or name),
        input_schema=input_schema,
        output_schema=output_schema,
        enabled=bool(data.get("enabled", True)),
        source=source,
        editable=editable,
    )


def _dict_value(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _validate_tool_name(name: str) -> None:
    if not _TOOL_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid tool name: {name}")
