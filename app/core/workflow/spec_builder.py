from __future__ import annotations

from typing import Any

from app.core.skills.registry import SkillRegistry
from app.core.skills.plugins import SkillPluginManager
from app.schemas import Attachment, Message


class SpecBuilder:
    """Build first-pass specs through installed skill plugins.

    The core runtime only creates the common envelope. Skill-specific routing
    and spec enrichment are delegated to plugin hooks when they exist.
    """

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        plugin_manager: SkillPluginManager | None = None,
    ) -> None:
        self.skill_registry = skill_registry or SkillRegistry()
        self.plugin_manager = plugin_manager or SkillPluginManager(self.skill_registry.root_dir)

    def build(
        self,
        messages: list[Message],
        attachments: list[Attachment],
        allowed_skills: list[str],
        selected_skill: str | None = None,
    ) -> dict[str, Any]:
        user_text = self._last_user_text(messages)
        context_text = self._user_text_context(messages)
        routing_text = self._routing_text(user_text, context_text)
        skill_name = self._selected_skill(selected_skill, allowed_skills) or self._followup_generation_skill(
            user_text,
            self._prior_user_text_context(messages),
            attachments,
            allowed_skills,
        ) or self.plugin_manager.select_skill(
            routing_text,
            attachments,
            allowed_skills,
        )
        skill = self.skill_registry.get(skill_name) if skill_name else None
        quality = list(skill.quality_template) if skill is not None else []
        refinement_requested = self._is_artifact_refinement_followup(user_text)
        base: dict[str, Any] = {
            "skill_name": skill_name,
            "title": self._title(user_text, skill_name),
            "summary": routing_text,
            "refinement_requested": refinement_requested,
            "has_visual_evidence": bool(attachments),
            "attachments": [attachment.model_dump() for attachment in attachments],
            "quality_requirements": quality,
            "verification_rules": quality,
        }
        return self.plugin_manager.build_spec(skill_name, user_text, routing_text, attachments, base)

    def _selected_skill(self, selected_skill: str | None, allowed_skills: list[str]) -> str | None:
        skill_name = (selected_skill or "").strip()
        if not skill_name:
            return None
        if allowed_skills and skill_name not in allowed_skills:
            return None
        try:
            skill = self.skill_registry.get(skill_name)
        except KeyError:
            return None
        return skill.name if skill.executable else None

    @staticmethod
    def _last_user_text(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content.strip()
        return ""

    @staticmethod
    def _user_text_context(messages: list[Message]) -> str:
        parts = [message.content.strip() for message in messages if message.role == "user" and message.content.strip()]
        return "\n".join(parts)

    @staticmethod
    def _prior_user_text_context(messages: list[Message]) -> str:
        user_parts = [message.content.strip() for message in messages if message.role == "user" and message.content.strip()]
        return "\n".join(user_parts[:-1])

    def _followup_generation_skill(
        self,
        user_text: str,
        prior_context_text: str,
        attachments: list[Attachment],
        allowed_skills: list[str],
    ) -> str | None:
        if not prior_context_text:
            return None
        if not (self._is_contextual_format_followup(user_text) or self._is_artifact_refinement_followup(user_text)):
            return None
        skill_name, score = self.plugin_manager.select_skill_candidate(
            prior_context_text,
            attachments,
            allowed_skills,
        )
        if not skill_name or score <= 0:
            return None
        try:
            skill = self.skill_registry.get(skill_name)
        except KeyError:
            return None
        return skill.name if skill.generation else None

    @classmethod
    def _routing_text(cls, user_text: str, context_text: str) -> str:
        if cls._is_contextual_format_followup(user_text) or cls._is_artifact_refinement_followup(user_text):
            return context_text
        return user_text

    @staticmethod
    def _is_contextual_format_followup(user_text: str) -> bool:
        text = SpecBuilder._normalized_text(user_text)
        format_keywords = (
            "drawio",
            "draw.io",
            "ppt",
            "pptx",
            "excel",
            "xlsx",
            "xmind",
            "markdown",
            "md",
            "架构图",
            "拓扑",
            "流程图",
            "原型图",
            "线框图",
            "diagram",
            "prototype",
            "wireframe",
            "格式",
        )
        followup_markers = (
            "能不能",
            "可以",
            "改成",
            "换成",
            "转成",
            "转换",
            "导出",
            "格式",
            "用",
            "加上",
            "再加",
            "增加",
            "新增",
            "添加",
            "补充",
            "加入",
            "扩展",
            "带上",
        )
        return any(keyword in text for keyword in format_keywords) and any(
            marker in text for marker in followup_markers
        )

    @staticmethod
    def _is_artifact_refinement_followup(user_text: str) -> bool:
        text = SpecBuilder._normalized_text(user_text)
        return any(
            marker in text
            for marker in (
                "美化",
                "优化",
                "调整",
                "修改",
                "重做",
                "重新生成",
                "重新设计",
                "换个",
                "太丑",
                "不好看",
                "不满意",
                "不是我想要",
                "改进",
                "更好看",
                "高级",
                "精致",
                "漂亮",
                "加上",
                "再加",
                "增加",
                "新增",
                "添加",
                "补充",
                "加入",
                "扩展",
                "带上",
            )
        )

    @staticmethod
    def _normalized_text(value: str) -> str:
        text = value.lower()
        return text.replace("draw.io", "drawio").replace("draw io", "drawio").replace("draw_io", "drawio")

    @staticmethod
    def _title(user_text: str, skill_name: str) -> str:
        compact = user_text.strip().replace("\n", " ")
        if compact:
            return compact[:48]
        return skill_name.replace("-", " ").title() if skill_name else "Untitled"
