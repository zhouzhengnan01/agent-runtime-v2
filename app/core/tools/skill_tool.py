import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.skills import get_skill_registry
from app.core.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)


def _get_int_setting(name: str, default: int) -> int:
    try:
        value = getattr(settings, name, None)
        if value is not None:
            return int(value)
    except Exception:
        pass

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


@register_tool("skill")
class SkillTool(BaseTool):
    name = "skill"
    description = "Load a skill to get detailed instructions for a specific task."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill identifier",
            }
        },
        "required": ["name"],
    }
    require_confirmation = False
    estimated_time = 1.0

    def setup(self) -> None:
        self._registry = get_skill_registry()
        self._refresh_definition()

    def _refresh_definition(self) -> None:
        skills = self._registry.list_skills()
        max_items = _get_int_setting("SKILL_MAX_ITEMS", 50)

        listed = skills[:max_items] if max_items > 0 else skills
        enums = [skill.name for skill in listed]

        if not listed:
            self.description = (
                "Load a skill to get detailed instructions for a specific task. "
                "No skills are currently available."
            )
        else:
            desc_lines = [
                "Load a skill to get detailed instructions for a specific task.",
                "Skills provide specialized knowledge and step-by-step guidance.",
                "Use this when a task matches an available skill's description.",
                "Only the skills listed here are available:",
                "<available_skills>",
            ]
            for skill in listed:
                desc_lines.extend(
                    [
                        "  <skill>",
                        f"    <name>{skill.name}</name>",
                        f"    <description>{skill.description}</description>",
                        "  </skill>",
                    ]
                )
            desc_lines.append("</available_skills>")
            if max_items > 0 and len(skills) > max_items:
                desc_lines.append(f"(showing {max_items} of {len(skills)} skills)")
            self.description = " ".join(desc_lines)

        param_desc = "Skill identifier from available_skills"
        if enums:
            sample = ", ".join([f"'{name}'" for name in enums[:3]])
            param_desc = f"{param_desc} (e.g., {sample}, ...)"

        name_schema = {
            "type": "string",
            "description": param_desc,
        }
        if enums:
            name_schema["enum"] = enums

        self.parameters = {
            "type": "object",
            "properties": {
                "name": name_schema,
            },
            "required": ["name"],
        }

    def _format_output(self, skill_name: str, content: str, base_dir: str, description: str, tags: List[str]) -> str:
        parts = [f"## Skill: {skill_name}", ""]
        if description:
            parts.append(f"**Description**: {description}")
            parts.append("")
        parts.append(f"**Base directory**: {base_dir}")
        if tags:
            parts.append(f"**Tags**: {', '.join(tags)}")
        parts.append("")
        parts.append(content.strip())
        return "\n".join(part for part in parts if part is not None)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        self._refresh_definition()
        name = str(kwargs.get("name") or "").strip()
        if not name:
            raise ValueError("Skill name is required")

        skill = self._registry.get_skill(name)
        if not skill:
            available = [item.name for item in self._registry.list_skills()]
            raise ValueError(f'Skill "{name}" not found. Available skills: {", ".join(available) or "none"}')

        content = self._registry.read_skill_content(skill)
        max_chars = _get_int_setting("SKILL_MAX_CHARS", 12000)
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars] + "\n...(truncated)"

        base_dir = str(Path(skill.location).resolve().parent)
        output = self._format_output(skill.name, content, base_dir, skill.description, skill.tags)

        return {
            "message": output,
            "skill": skill.name,
            "description": skill.description,
            "base_dir": base_dir,
            "tags": skill.tags,
        }

    async def acall(self, params: Any, **kwargs: Any) -> Dict[str, Any]:
        if isinstance(params, str):
            try:
                import json

                params = json.loads(params)
            except Exception:
                params = {"name": params}
        if not isinstance(params, dict):
            params = {}
        return self.run(**params)
