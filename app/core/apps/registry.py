from __future__ import annotations

import builtins
import json
from pathlib import Path

from app.core.apps.models import AppTemplate
from app.core.config import AgentConfigLoader
from app.core.mcp import McpToolRegistry
from app.core.skills import SkillRegistry


class AppTemplateRegistry:
    """Load one-click Workbench application templates from config/apps."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.path = self.root_dir / "config" / "apps" / "templates.json"

    def list(self) -> builtins.list[AppTemplate]:
        return sorted(self._read_templates(), key=lambda template: (template.category, template.title, template.name))

    def get(self, name: str) -> AppTemplate:
        safe_name = self._safe_name(name)
        for template in self._read_templates():
            if template.name == safe_name:
                return template
        raise KeyError(f"App template not found: {safe_name}")

    def _read_templates(self) -> builtins.list[AppTemplate]:
        if not self.path.is_file():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items = raw.get("templates") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return []
        templates: builtins.list[AppTemplate] = []
        for item in items:
            if isinstance(item, dict):
                templates.append(AppTemplate.model_validate(item))
        return templates

    def validate_references(self) -> builtins.list[str]:
        """Return template reference problems without failing template loading."""
        agents = {agent.name for agent in AgentConfigLoader(self.root_dir).list_agents()}
        skills = {skill.name for skill in SkillRegistry(self.root_dir).list()}
        mcp_tools = {tool.name for tool in McpToolRegistry(self.root_dir).list(include_disabled=True)}
        workflows = {"agent_loop", "artifact_workflow", "evidence_first_detection"}
        problems: builtins.list[str] = []
        for template in self._read_templates():
            prefix = f"{template.name}:"
            if template.agent_name not in agents:
                problems.append(f"{prefix} unknown agent {template.agent_name}")
            if template.workflow and template.workflow not in workflows:
                problems.append(f"{prefix} unknown workflow {template.workflow}")
            for skill_name in template.selected_skills:
                if skill_name not in skills:
                    problems.append(f"{prefix} unknown skill {skill_name}")
            for tool_name in template.selected_mcp_tools:
                if tool_name not in mcp_tools:
                    problems.append(f"{prefix} unknown MCP tool {tool_name}")
        return problems

    @staticmethod
    def _safe_name(name: str) -> str:
        safe_name = name.strip()
        if not safe_name:
            raise ValueError("App template name must not be empty.")
        if any(char in safe_name for char in "/\\"):
            raise ValueError("App template name must not contain path separators.")
        return safe_name
