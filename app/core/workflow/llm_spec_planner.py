from __future__ import annotations

import json
import os
from typing import Any

from app.core.config import AgentConfig
from app.core.llm import OpenAICompatibleClient
from app.schemas import Message, RuntimeOptions


GENERATION_SKILLS = {
    "drawio-generation",
    "pptx-generation",
    "excel-generation",
    "xmind-generation",
    "markdown-rendering",
}


class LlmSpecPlanner:
    """Use an LLM to enrich deterministic specs while keeping rendering local."""

    def plan(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        base_spec: dict[str, Any],
        runtime_options: RuntimeOptions | None = None,
    ) -> dict[str, Any]:
        skill_name = str(base_spec.get("skill_name") or "")
        if skill_name not in GENERATION_SKILLS:
            return self._with_planner(base_spec, mode="local", enabled=False, reason="non_generation_skill")
        if not self._enabled():
            return self._with_planner(base_spec, mode="local", enabled=False, reason="disabled")

        client = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
        if not client.configured:
            return self._with_planner(base_spec, mode="local", enabled=False, reason="llm_not_configured")

        prompt = self._prompt(messages, base_spec)
        raw = client.complete_sync(self._system_prompt(), [Message(role="user", content=prompt)])
        planned = self._parse_json_object(raw)
        merged = self._merge(base_spec, planned)
        return self._with_planner(
            merged,
            mode="llm",
            enabled=True,
            model=client.model,
            raw_chars=len(raw),
        )

    @staticmethod
    def _enabled() -> bool:
        raw = os.getenv("LLM_SPEC_PLANNER", "1").strip().lower()
        return raw not in {"0", "false", "no", "off", "disabled"}

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a senior JetLinks IoT solution architect and document designer. "
            "Return only valid JSON. Do not include markdown fences or commentary. "
            "You design rich structured specs for deterministic renderers; do not generate final files."
        )

    @staticmethod
    def _prompt(messages: list[Message], base_spec: dict[str, Any]) -> str:
        transcript = "\n".join(f"{message.role}: {message.content}" for message in messages[-8:])
        skill_name = str(base_spec.get("skill_name") or "")
        contracts = {
            "drawio-generation": (
                "For drawio-generation, provide diagram_type, visual_style, swimlanes, color_semantics, nodes, edges, lane_nodes. "
                "nodes must be 10-24 concise Chinese labels. edges must be pairs of existing node labels. "
                "lane_nodes must be an object whose keys exactly match swimlanes and values are ordered node labels from nodes. "
                "Keep lane_nodes semantically balanced: protocol/gateway devices in access, parsing/rules/forwarding in core, "
                "databases/caches/device shadows in storage, alerts/notifications/monitoring/consoles in operations. "
                "Use domain-specific JetLinks concepts from the request."
            ),
            "pptx-generation": (
                "For pptx-generation, provide 6-8 slides. Each slide has title and kind. "
                "Kinds should include cover, agenda, architecture, process, plan, risks, summary where appropriate."
            ),
            "excel-generation": (
                "For excel-generation, provide sheets, fields, validations, formulas, freeze_panes. "
                "Fields are objects with name, type, required."
            ),
            "xmind-generation": (
                "For xmind-generation, provide root_topic and at least 4 topics. "
                "Each topic has title and 2-5 children."
            ),
            "markdown-rendering": (
                "For markdown-rendering, provide sections and required_blocks. "
                "Include mermaid, table, echarts, attachment_block when relevant."
            ),
        }
        return (
            "Conversation:\n"
            f"{transcript}\n\n"
            "Current deterministic base spec:\n"
            f"{json.dumps(base_spec, ensure_ascii=False, indent=2)}\n\n"
            "Task:\n"
            f"Enhance this spec for skill_name={skill_name}. Keep skill_name unchanged. "
            "Preserve safety fields such as attachments, has_visual_evidence, quality_requirements, verification_rules, "
            "and refinement_requested unless you have a more specific value. "
            f"{contracts.get(skill_name, '')}\n\n"
            "Return one JSON object only."
        )

    @classmethod
    def _parse_json_object(cls, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = cls._strip_fence(text)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM planner did not return a JSON object")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM planner returned non-object JSON")
        return parsed

    @staticmethod
    def _strip_fence(text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @classmethod
    def _merge(cls, base_spec: dict[str, Any], planned: dict[str, Any]) -> dict[str, Any]:
        skill_name = str(base_spec.get("skill_name") or "")
        merged = dict(base_spec)
        for key, value in planned.items():
            if key in {"attachments", "has_visual_evidence", "quality_requirements", "verification_rules"}:
                continue
            if key == "skill_name":
                continue
            merged[key] = value
        merged["skill_name"] = skill_name
        normalized = cls._normalize_for_skill(merged)
        return cls._preserve_minimum_valid_shape(base_spec, normalized)

    @classmethod
    def _preserve_minimum_valid_shape(cls, base_spec: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
        skill_name = str(spec.get("skill_name") or "")
        if skill_name == "drawio-generation":
            if len(cls._strings(spec.get("nodes"))) < 8:
                spec["nodes"] = cls._strings(base_spec.get("nodes"))
            if len(cls._pairs(spec.get("edges"))) < 6:
                spec["edges"] = cls._pairs(base_spec.get("edges"))
            if len(cls._strings(spec.get("swimlanes"))) < 2:
                spec["swimlanes"] = cls._strings(base_spec.get("swimlanes"))
        elif skill_name == "pptx-generation":
            if not 6 <= len(cls._slides(spec.get("slides"))) <= 8:
                spec["slides"] = cls._slides(base_spec.get("slides"))
        elif skill_name == "xmind-generation":
            if len(cls._topics(spec.get("topics"))) < 3:
                spec["topics"] = cls._topics(base_spec.get("topics"))
        elif skill_name == "markdown-rendering":
            if len(cls._strings(spec.get("sections"))) < 3:
                spec["sections"] = cls._strings(base_spec.get("sections"))
            if len(cls._strings(spec.get("required_blocks"))) < 2:
                spec["required_blocks"] = cls._strings(base_spec.get("required_blocks"))
        return spec

    @classmethod
    def _normalize_for_skill(cls, spec: dict[str, Any]) -> dict[str, Any]:
        skill_name = str(spec.get("skill_name") or "")
        if skill_name == "drawio-generation":
            spec["nodes"] = cls._strings(spec.get("nodes")) or cls._strings(spec.get("components")) or []
            spec["edges"] = cls._pairs(spec.get("edges")) or cls._pairs(spec.get("relationships")) or []
            spec["swimlanes"] = cls._strings(spec.get("swimlanes")) or ["交互层", "Agent 编排层", "技能执行层", "产物与质量层"]
            spec["lane_nodes"] = cls._lane_nodes(spec.get("lane_nodes"), spec["swimlanes"], spec["nodes"])
            if not isinstance(spec.get("color_semantics"), dict):
                spec["color_semantics"] = {}
            spec["diagram_type"] = str(spec.get("diagram_type") or "layered_architecture")
            spec["visual_style"] = str(spec.get("visual_style") or "polished")
        elif skill_name == "pptx-generation":
            spec["slides"] = cls._slides(spec.get("slides"))
        elif skill_name == "excel-generation":
            spec["sheets"] = cls._strings(spec.get("sheets")) or ["设备模板", "状态汇总"]
            spec["fields"] = cls._fields(spec.get("fields"))
            if not isinstance(spec.get("validations"), dict):
                spec["validations"] = {}
            spec["formulas"] = cls._strings(spec.get("formulas"))
        elif skill_name == "xmind-generation":
            spec["root_topic"] = str(spec.get("root_topic") or spec.get("title") or "JetLinks Agent Runtime v2")
            spec["topics"] = cls._topics(spec.get("topics"))
        elif skill_name == "markdown-rendering":
            spec["sections"] = cls._strings(spec.get("sections"))
            spec["required_blocks"] = cls._strings(spec.get("required_blocks"))
        return spec

    @staticmethod
    def _strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _pairs(value: object) -> list[list[str]]:
        if not isinstance(value, list):
            return []
        pairs: list[list[str]] = []
        for item in value:
            if isinstance(item, list | tuple) and len(item) >= 2:
                source = str(item[0]).strip()
                target = str(item[1]).strip()
                if source and target:
                    pairs.append([source, target])
            elif isinstance(item, dict):
                source = str(item.get("source") or item.get("from") or "").strip()
                target = str(item.get("target") or item.get("to") or "").strip()
                if source and target:
                    pairs.append([source, target])
        return pairs

    @classmethod
    def _lane_nodes(cls, value: object, swimlanes: list[str], nodes: list[str]) -> dict[str, list[str]]:
        if not swimlanes:
            return {}
        node_set = set(nodes)
        raw_items: list[tuple[str, object]] = []
        if isinstance(value, dict):
            raw_items.extend((str(lane), raw_nodes) for lane, raw_nodes in value.items())
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                lane = str(item.get("lane") or item.get("swimlane") or item.get("name") or "").strip()
                raw_nodes = item.get("nodes") or item.get("items")
                raw_items.append((lane, raw_nodes))

        normalized_lanes = {cls._normalize_label(lane): lane for lane in swimlanes}
        grouped: dict[str, list[str]] = {lane: [] for lane in swimlanes}
        for raw_lane, raw_nodes in raw_items:
            matched_lane = normalized_lanes.get(cls._normalize_label(raw_lane))
            if matched_lane is None or not isinstance(raw_nodes, list):
                continue
            for node in raw_nodes:
                node_name = str(node).strip()
                if node_name in node_set and node_name not in grouped[matched_lane]:
                    grouped[matched_lane].append(node_name)
        return {lane: lane_nodes for lane, lane_nodes in grouped.items() if lane_nodes}

    @staticmethod
    def _normalize_label(value: str) -> str:
        return value.lower().replace(" ", "").replace("-", "").replace("_", "")

    @staticmethod
    def _slides(value: object) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        slides: list[dict[str, str]] = []
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or f"Slide {idx}").strip()
            kind = str(item.get("kind") or "content").strip()
            slides.append({"title": title, "kind": kind})
        return slides

    @staticmethod
    def _fields(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        fields: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name:
                fields.append(
                    {
                        "name": name,
                        "type": str(item.get("type") or "string"),
                        "required": bool(item.get("required", False)),
                    }
                )
        return fields

    @staticmethod
    def _topics(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        topics: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            children = [str(child).strip() for child in item.get("children", []) if str(child).strip()]
            if title:
                topics.append({"title": title, "children": children})
        return topics

    @staticmethod
    def _with_planner(spec: dict[str, Any], **planner: object) -> dict[str, Any]:
        enriched = dict(spec)
        enriched["planner"] = planner
        return enriched
