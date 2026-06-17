from __future__ import annotations

import builtins
import json
import re
from pathlib import Path

from app.core.apps.models import AppTemplate
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfigLoader
from app.core.mcp import McpToolRegistry
from app.core.resources import ResourceRoots, first_existing_file, merged_json_paths
from app.core.skills.aliases import expand_skill_aliases
from app.core.skills import SkillRegistry
from app.core.workflow import WorkflowRegistry


_APP_TEMPLATE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class AppTemplateRegistry:
    """Load one-click Workbench application templates from config/apps."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.resources = ResourceRoots.from_project_root(self.root_dir)
        self.config_dirs = self.resources.config_dirs("apps")
        self.config_dir = self.resources.writable_config_dir("apps")

    def list(self) -> builtins.list[AppTemplate]:
        return sorted(self._read_templates(), key=lambda template: (template.category, template.title, template.name))

    def get(self, name: str) -> AppTemplate:
        safe_name = self._safe_name(name)
        path = first_existing_file(self.config_dirs, f"{safe_name}.json")
        if path is not None:
            return self._load_template(path)
        for template in self._read_templates():
            if template.name == safe_name:
                return template
        raise KeyError(f"App template not found: {safe_name}")

    def find(self, name: str) -> AppTemplate | None:
        raw_name = name.strip()
        if not raw_name:
            return None
        if _APP_TEMPLATE_NAME_PATTERN.fullmatch(raw_name):
            path = first_existing_file(self.config_dirs, f"{raw_name}.json")
            if path is not None:
                return self._load_template(path)
        lowered = raw_name.casefold()
        for template in self._read_templates():
            aliases = [alias for alias in template.aliases if alias]
            if template.name == raw_name or template.title == raw_name or raw_name in aliases:
                return template
            if (
                template.name.casefold() == lowered
                or template.title.casefold() == lowered
                or any(alias.casefold() == lowered for alias in aliases)
            ):
                return template
        return None

    def _read_templates(self) -> builtins.list[AppTemplate]:
        if not any(directory.is_dir() for directory in self.config_dirs):
            return []
        templates_by_name: dict[str, AppTemplate] = {}
        for collection_path in reversed([path for path in (directory / "templates.json" for directory in self.config_dirs) if path.is_file()]):
            try:
                for template in self._load_template_collection(collection_path):
                    templates_by_name[template.name] = template
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        for path in merged_json_paths(self.config_dirs, skip_names={"templates.json"}):
            if path.name.startswith(".") or path.name == "templates.json":
                continue
            try:
                template = self._load_template(path)
            except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            templates_by_name[template.name] = template
        return list(templates_by_name.values())

    def _load_template(self, path: Path) -> AppTemplate:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"App template must be a JSON object: {path}")
        return _normalize_template(AppTemplate.model_validate(data), self.root_dir)

    def _load_template_collection(self, path: Path) -> builtins.list[AppTemplate]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"App template collection must be a JSON object: {path}")
        raw_templates = data.get("templates")
        if not isinstance(raw_templates, list):
            raise ValueError(f"App template collection must contain templates list: {path}")
        return [_normalize_template(AppTemplate.model_validate(item), self.root_dir) for item in raw_templates if isinstance(item, dict)]

    def validate_references(self) -> builtins.list[str]:
        """Return template reference problems without failing template loading."""
        agents = {agent.name for agent in AgentConfigLoader(self.root_dir).list_agents()}
        skills = {skill.name for skill in SkillRegistry(self.root_dir).list()}
        mcp_tools = {tool.name for tool in McpToolRegistry(self.root_dir).list(include_disabled=True)}
        workflows = {"agent_loop", *WorkflowRegistry.builtin(ArtifactStore(), self.root_dir).names()}
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
        if not _APP_TEMPLATE_NAME_PATTERN.fullmatch(safe_name):
            raise ValueError(
                "App template name must be 1-128 characters and contain only letters, numbers, dots, underscores, or hyphens."
            )
        return safe_name


def _normalize_template(template: AppTemplate, root_dir: Path | None = None) -> AppTemplate:
    selected_skills = expand_skill_aliases(template.selected_skills, root_dir)
    if selected_skills == template.selected_skills:
        return template
    return template.model_copy(update={"selected_skills": selected_skills}, deep=True)
