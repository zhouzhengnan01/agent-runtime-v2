from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from app.core.config import AgentConfig
from app.core.llm import OpenAICompatibleClient
from app.core.skills import SkillDefinition, SkillRegistry
from app.schemas import ChatRequest, Message


@dataclass(frozen=True)
class WorkflowSelection:
    workflow_name: str | None
    skill_name: str | None = None
    score: int = 0
    mode: str = "local"
    reason: str = ""


class WorkflowRouter:
    """Suggest workflow plugins from skill manifests for UI or compatibility callers."""

    def __init__(
        self,
        skill_registry: SkillRegistry | None = None,
        available_workflows: set[str] | None = None,
    ) -> None:
        self.skill_registry = skill_registry or SkillRegistry()
        self.available_workflows = set(available_workflows or set())

    def select(self, agent_config: AgentConfig, request: ChatRequest) -> str | None:
        return self.select_with_details(agent_config, request).workflow_name

    def select_with_details(self, agent_config: AgentConfig, request: ChatRequest) -> WorkflowSelection:
        routing_text = self.routing_text(request)
        if not routing_text:
            return WorkflowSelection(workflow_name=None, reason="empty_request")
        if self._is_agent_capability_question(routing_text):
            return WorkflowSelection(workflow_name=None, reason="capability_question")

        allowed_skills = self._allowed_skills(agent_config)
        if not allowed_skills:
            return WorkflowSelection(workflow_name=None, reason="no_allowed_skills")

        skills = self.skill_registry.list(allowed_skills, executable_only=True)
        if not skills:
            return WorkflowSelection(workflow_name=None, reason="no_executable_skills")

        llm_candidate = self._select_skill_with_llm(agent_config, request, skills)
        if llm_candidate is not None:
            workflow_name = self._workflow_for_skill(agent_config, llm_candidate)
            if self._is_registered_workflow(workflow_name):
                return WorkflowSelection(
                    workflow_name=workflow_name,
                    skill_name=llm_candidate.name,
                    score=100,
                    mode="llm",
                    reason="skill_manifest_match",
                )

        skill_name, score = self._select_skill_from_manifest(routing_text, skills)
        if request.attachments and score <= 0:
            detection_skills = [skill for skill in skills if not skill.generation]
            if len(detection_skills) == 1:
                skill_name = detection_skills[0].name
                score = 1
        if not skill_name or score <= 0:
            return WorkflowSelection(workflow_name=None, skill_name=skill_name or None, score=score, reason="no_skill_match")

        skill = self.skill_registry.get(skill_name)
        workflow_name = self._workflow_for_skill(agent_config, skill)
        if not self._is_registered_workflow(workflow_name):
            return WorkflowSelection(
                workflow_name=None,
                skill_name=skill.name,
                score=score,
                mode="local",
                reason="workflow_not_registered",
            )
        return WorkflowSelection(
            workflow_name=workflow_name,
            skill_name=skill.name,
            score=score,
            mode="local",
            reason="plugin_score_match" if workflow_name is not None else "skill_without_workflow",
        )

    @staticmethod
    def last_user_text(request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content.strip()
        return ""

    @classmethod
    def routing_text(cls, request: ChatRequest) -> str:
        user_text = cls.last_user_text(request)
        context_text = "\n".join(message.content.strip() for message in request.messages if message.role == "user")
        if cls._is_followup_request(user_text):
            return context_text
        return user_text

    def has_generation_context(self, request: ChatRequest) -> bool:
        routing_text = self.routing_text(request)
        skill_name, score = self._select_skill_from_manifest(
            routing_text,
            [skill for skill in self.skill_registry.list(executable_only=True) if skill.generation],
        )
        return bool(skill_name and score > 0)

    def is_artifact_refinement_request(self, text: str) -> bool:
        return self._is_followup_request(text)

    @staticmethod
    def is_meta_request(text: str) -> bool:
        return WorkflowRouter._is_agent_capability_question(text)

    def _allowed_skills(self, agent_config: AgentConfig) -> list[str]:
        if agent_config.skills:
            return agent_config.skills
        return [skill.name for skill in self.skill_registry.list(executable_only=True)]

    @staticmethod
    def _workflow_for_skill(agent_config: AgentConfig, skill: SkillDefinition) -> str | None:
        if skill.generation:
            return "artifact_workflow"
        return "evidence_first_detection"

    def _is_registered_workflow(self, workflow_name: str | None) -> bool:
        return bool(workflow_name and workflow_name != "agent_loop" and workflow_name in self.available_workflows)

    def _select_skill_with_llm(
        self,
        agent_config: AgentConfig,
        request: ChatRequest,
        skills: list[SkillDefinition],
    ) -> SkillDefinition | None:
        if not self._llm_routing_enabled(agent_config):
            return None
        client = OpenAICompatibleClient(agent_config, runtime_options=request.runtime_options)
        if not client.configured:
            return None
        prompt = self._llm_prompt(request, skills)
        try:
            raw = client.complete_sync(self._llm_system_prompt(), [Message(role="user", content=prompt)])
            payload = self._parse_json_object(raw)
        except Exception:
            return None
        skill_name = payload.get("skill_name")
        if not isinstance(skill_name, str) or skill_name == "agent_loop":
            return None
        allowed = {skill.name: skill for skill in skills}
        return allowed.get(skill_name)

    @classmethod
    def _select_skill_from_manifest(cls, routing_text: str, skills: list[SkillDefinition]) -> tuple[str, int]:
        normalized = cls._normalize(routing_text)
        best_name = ""
        best_score = 0
        for skill in skills:
            score = cls._manifest_score(normalized, skill)
            if score > best_score:
                best_name = skill.name
                best_score = score
        return best_name, best_score

    @classmethod
    def _manifest_score(cls, normalized_text: str, skill: SkillDefinition) -> int:
        routing = skill.routing or {}
        score = 0
        for keyword in cls._strings(routing.get("keywords")):
            if cls._normalize(keyword) in normalized_text:
                score = max(score, 100)
        for example in cls._strings(routing.get("examples")):
            overlap = cls._token_overlap(normalized_text, cls._normalize(example))
            if overlap:
                score = max(score, min(90, 20 + overlap * 10))
        summary = routing.get("summary") if isinstance(routing.get("summary"), str) else skill.description
        overlap = cls._token_overlap(normalized_text, cls._normalize(str(summary)))
        if overlap >= 2:
            score = max(score, min(70, overlap * 10))
        return score

    @staticmethod
    def _llm_routing_enabled(agent_config: AgentConfig) -> bool:
        raw = os.getenv(agent_config.routing.llm_workflow_router_env)
        if raw is None:
            return agent_config.routing.llm_workflow_router
        return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}

    @staticmethod
    def _llm_system_prompt() -> str:
        return (
            "You route a stateless agent request to one declared skill. "
            "Use only the provided skill descriptions and schemas. "
            "Return only JSON like {\"skill_name\":\"...\"} or {\"skill_name\":\"agent_loop\"}."
        )

    def _llm_prompt(self, request: ChatRequest, skills: list[SkillDefinition]) -> str:
        skills_payload = [
            {
                "name": skill.name,
                "description": skill.description,
                "generation": skill.generation,
                "output_kind": skill.output_kind,
                "quality_template": list(skill.quality_template),
                "routing": skill.routing or {},
                "input_schema": self._compact_schema(skill.input_schema),
            }
            for skill in skills
        ]
        return (
            "Conversation user text:\n"
            f"{self.routing_text(request)}\n\n"
            "Attachments:\n"
            f"{json.dumps([attachment.model_dump() for attachment in request.attachments], ensure_ascii=False)}\n\n"
            "Available skills:\n"
            f"{json.dumps(skills_payload, ensure_ascii=False, indent=2)}\n\n"
            "Choose a skill only when the request clearly asks for that capability. "
            "Otherwise choose agent_loop."
        )

    @staticmethod
    def _compact_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        return {
            "required": schema.get("required", []),
            "properties": sorted((schema.get("properties") or {}).keys()) if isinstance(schema.get("properties"), dict) else [],
        }

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Router LLM did not return a JSON object")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Router LLM returned non-object JSON")
        return parsed

    @staticmethod
    def _is_agent_capability_question(text: str) -> bool:
        normalized = text.lower()
        return any(marker in normalized for marker in ("你能", "你会", "你的能力", "帮助", "help", "what can you do"))

    @staticmethod
    def _is_followup_request(text: str) -> bool:
        normalized = text.lower()
        return any(
            marker in normalized
            for marker in (
                "能不能",
                "可以",
                "改成",
                "换成",
                "转成",
                "转换",
                "导出",
                "格式",
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
    def _strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _normalize(value: str) -> str:
        return value.lower().replace("draw.io", "drawio").replace("draw io", "drawio").replace("draw_io", "drawio")

    @staticmethod
    def _token_overlap(text: str, candidate: str) -> int:
        text_tokens = {token for token in text.replace("/", " ").replace("-", " ").split() if len(token) >= 2}
        candidate_tokens = {
            token for token in candidate.replace("/", " ").replace("-", " ").split() if len(token) >= 2
        }
        if not text_tokens or not candidate_tokens:
            return 0
        return len(text_tokens & candidate_tokens)
