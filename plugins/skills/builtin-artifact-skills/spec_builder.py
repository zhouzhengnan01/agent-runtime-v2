from __future__ import annotations

from typing import Any

from app.schemas import Attachment, Message


class SpecBuilder:
    """Deterministic first-pass spec builder.

    This is intentionally local and predictable. Later versions can call an LLM
    for richer spec completion while keeping the same structured contract.
    """

    def build(self, messages: list[Message], attachments: list[Attachment], allowed_skills: list[str]) -> dict[str, Any]:
        user_text = self._last_user_text(messages)
        context_text = self._user_text_context(messages)
        routing_text = self._routing_text(user_text, context_text)
        skill_name = self._select_skill(routing_text, attachments, allowed_skills)
        refinement_requested = self._is_artifact_refinement_followup(user_text)
        base: dict[str, Any] = {
            "skill_name": skill_name,
            "title": self._title(user_text, skill_name),
            "summary": routing_text,
            "refinement_requested": refinement_requested,
            "has_visual_evidence": bool(attachments),
            "attachments": [attachment.model_dump() for attachment in attachments],
            "quality_requirements": self._quality_requirements(skill_name),
            "verification_rules": self._quality_requirements(skill_name),
        }
        if skill_name == "drawio-generation":
            base.update(self._drawio_spec(routing_text))
        elif skill_name == "pptx-generation":
            base.update(self._ppt_spec(routing_text))
        elif skill_name == "excel-generation":
            base.update(self._excel_spec())
        elif skill_name == "xmind-generation":
            base.update(self._xmind_spec())
        elif skill_name == "markdown-rendering":
            base.update(self._markdown_spec())
        elif skill_name == "behavior-detection":
            base.update(self._behavior_spec(user_text, attachments))
        elif skill_name == "behavior-review":
            base.update(self._behavior_review_spec(user_text, attachments))
        return base

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
    def _select_skill(user_text: str, attachments: list[Attachment], allowed_skills: list[str]) -> str:
        text = SpecBuilder._normalized_text(user_text)
        candidates: list[tuple[str, tuple[str, ...]]] = [
            ("behavior-review", ("复判", "复核", "二次审核", "人工审核", "告警审核", "review")),
            ("behavior-detection", ("翻越", "摔倒", "逗留", "禁区", "行为", "识别", "intrusion")),
            (
                "drawio-generation",
                ("drawio", "draw.io", "draw io", "架构图", "拓扑", "流程图", "原型图", "线框图", "prototype", "wireframe", "diagram"),
            ),
            ("pptx-generation", ("ppt", "pptx", "幻灯片", "演示文稿")),
            ("excel-generation", ("excel", "xlsx", "表格", "模板")),
            ("xmind-generation", ("xmind", "思维导图", "脑图")),
            ("markdown-rendering", ("markdown", "md", "文档", "报告")),
        ]
        for skill, keywords in candidates:
            if skill in allowed_skills and any(keyword in text for keyword in keywords):
                return skill
        if attachments and "behavior-detection" in allowed_skills:
            return "behavior-detection"
        if "markdown-rendering" in allowed_skills:
            return "markdown-rendering"
        return allowed_skills[0] if allowed_skills else "markdown-rendering"

    @staticmethod
    def _normalized_text(value: str) -> str:
        text = value.lower()
        return text.replace("draw.io", "drawio").replace("draw io", "drawio").replace("draw_io", "drawio")

    @staticmethod
    def _title(user_text: str, skill_name: str) -> str:
        compact = user_text.strip().replace("\n", " ")
        lower = compact.lower()
        if skill_name == "drawio-generation":
            if any(keyword in lower for keyword in ("原型", "prototype", "wireframe")):
                return "交互原型图"
            if "jetlinks" in lower and ("iot" in lower or "架构" in compact):
                return "JetLinks IoT 平台架构图"
            if "架构" in compact:
                return "系统架构图"
        if compact:
            return compact[:48]
        return skill_name.replace("-", " ").title()

    @staticmethod
    def _quality_requirements(skill_name: str) -> list[str]:
        defaults = {
            "drawio-generation": ["nodes", "edges", "swimlanes", "layering", "semantic_colors"],
            "pptx-generation": ["cover", "agenda", "architecture", "risks", "summary"],
            "excel-generation": ["fields", "types", "validation", "summary"],
            "xmind-generation": ["multi_level_topics", "notes", "tasks"],
            "markdown-rendering": ["mermaid", "table", "echarts_block", "attachment_block"],
            "behavior-detection": ["evidence_first", "no_visual_score_without_visual_evidence"],
            "behavior-review": ["evidence_first", "review_decision", "evidence_gaps", "actions"],
        }
        return defaults.get(skill_name, [])

    @staticmethod
    def _drawio_spec(user_text: str) -> dict[str, Any]:
        is_jetlinks = "jetlinks" in user_text.lower()
        is_prototype = any(keyword in user_text.lower() for keyword in ("原型", "prototype", "wireframe"))
        is_polished = SpecBuilder._is_artifact_refinement_followup(user_text)
        color_semantics = (
            {
                "entry": "#e0f2fe",
                "agent": "#d1fae5",
                "skill": "#fef3c7",
                "artifact": "#ede9fe",
                "quality": "#ffe4e6",
            }
            if is_polished
            else {
                "entry": "#dbeafe",
                "agent": "#dcfce7",
                "skill": "#fef3c7",
                "artifact": "#fce7f3",
                "quality": "#fee2e2",
            }
        )
        if is_prototype:
            return {
                "diagram_type": "prototype_wireframe",
                "visual_style": "polished" if is_polished else "standard",
                "swimlanes": ["页面结构", "核心交互", "输出区域", "状态反馈"],
                "color_semantics": color_semantics,
                "nodes": [
                    "左侧导航",
                    "Agent 列表",
                    "中间聊天区",
                    "任务输入框",
                    "技能快捷入口",
                    "ACP WS / SSE 切换",
                    "右侧文件预览",
                    "校验结果",
                    "Spec 详情",
                    "执行事件",
                    "下载操作",
                    "错误提示",
                ],
                "edges": [
                    ["左侧导航", "Agent 列表"],
                    ["Agent 列表", "中间聊天区"],
                    ["任务输入框", "技能快捷入口"],
                    ["技能快捷入口", "ACP WS / SSE 切换"],
                    ["ACP WS / SSE 切换", "执行事件"],
                    ["执行事件", "右侧文件预览"],
                    ["右侧文件预览", "下载操作"],
                    ["执行事件", "校验结果"],
                    ["执行事件", "Spec 详情"],
                    ["校验结果", "错误提示"],
                ],
            }

        nodes = [
            "用户入口",
            "MiniMax 风格工作台",
            "无状态 Agent Runtime",
            "JSON Agent 配置",
            "结构化 Spec Builder",
            "Skill Registry",
            "Skill Runner",
            "Artifact Store",
            "Preview Service",
            "内容级 Verifier",
            "自动修正重试",
            "Agent 最终回复",
        ]
        if is_jetlinks:
            nodes.extend(["JetLinks 设备管理", "规则引擎", "数据可视化"])
        edges = [
            ["用户入口", "MiniMax 风格工作台"],
            ["MiniMax 风格工作台", "无状态 Agent Runtime"],
            ["无状态 Agent Runtime", "JSON Agent 配置"],
            ["无状态 Agent Runtime", "结构化 Spec Builder"],
            ["结构化 Spec Builder", "Skill Registry"],
            ["Skill Registry", "Skill Runner"],
            ["Skill Runner", "Artifact Store"],
            ["Artifact Store", "Preview Service"],
            ["Artifact Store", "内容级 Verifier"],
            ["内容级 Verifier", "自动修正重试"],
            ["内容级 Verifier", "Agent 最终回复"],
            ["Agent 最终回复", "MiniMax 风格工作台"],
        ]
        if is_jetlinks:
            edges.extend(
                [
                    ["JetLinks 设备管理", "规则引擎"],
                    ["规则引擎", "数据可视化"],
                    ["数据可视化", "结构化 Spec Builder"],
                ]
            )
        return {
            "diagram_type": "layered_architecture",
            "visual_style": "polished" if is_polished else "standard",
            "swimlanes": ["交互层", "Agent 编排层", "技能执行层", "产物与质量层"],
            "color_semantics": color_semantics,
            "nodes": nodes,
            "edges": edges,
        }

    @staticmethod
    def _ppt_spec(user_text: str) -> dict[str, Any]:
        title = user_text.strip()[:48] or "JetLinks Agent Runtime v2"
        return {
            "slides": [
                {"title": title, "kind": "cover"},
                {"title": "目录", "kind": "agenda"},
                {"title": "总体架构", "kind": "architecture"},
                {"title": "生成链路", "kind": "process"},
                {"title": "实施计划", "kind": "plan"},
                {"title": "风险与控制", "kind": "risks"},
                {"title": "总结", "kind": "summary"},
            ]
        }

    @staticmethod
    def _excel_spec() -> dict[str, Any]:
        return {
            "sheets": ["设备模板", "状态汇总"],
            "fields": [
                {"name": "设备ID", "type": "string", "required": True},
                {"name": "设备名称", "type": "string", "required": True},
                {"name": "产品类型", "type": "enum", "required": True},
                {"name": "在线状态", "type": "enum", "required": True},
                {"name": "最后活跃时间", "type": "datetime", "required": False},
            ],
            "validations": {"在线状态": ["在线", "离线", "告警"]},
            "formulas": ["COUNTIF", "COUNTA"],
            "freeze_panes": "A2",
        }

    @staticmethod
    def _xmind_spec() -> dict[str, Any]:
        return {
            "root_topic": "JetLinks Agent Runtime v2",
            "topics": [
                {"title": "Agent 配置", "children": ["JSON", "模型", "工具", "权限"]},
                {"title": "Workflow", "children": ["Spec Builder", "Skill Runner", "Verifier", "Retry"]},
                {"title": "Artifact", "children": ["Preview", "Download", "解析校验"]},
                {"title": "交互体验", "children": ["中间聊天", "右侧预览", "事件流"]},
            ],
        }

    @staticmethod
    def _markdown_spec() -> dict[str, Any]:
        return {
            "sections": ["目标", "架构", "流程", "接口", "验收标准"],
            "required_blocks": ["mermaid", "table", "echarts", "attachment_block"],
        }

    @staticmethod
    def _behavior_spec(user_text: str, attachments: list[Attachment]) -> dict[str, Any]:
        text_rules: list[str] = []
        if any(keyword in user_text for keyword in ("翻越", "围栏", "禁区")):
            text_rules.append("翻越进入禁区")
        if "摔倒" in user_text:
            text_rules.append("老人摔倒")
        if "逗留" in user_text:
            text_rules.append("长时间逗留")
        if not text_rules:
            text_rules.append("待确认行为类型")
        return {
            "evidence_mode": "visual_or_structured" if attachments else "text_only",
            "text_rule_candidates": text_rules,
            "must_not_emit_visual_score_without_evidence": True,
        }

    @staticmethod
    def _behavior_review_spec(user_text: str, attachments: list[Attachment]) -> dict[str, Any]:
        spec = SpecBuilder._behavior_spec(user_text, attachments)
        spec.update(
            {
                "event_id": "待生成",
                "original_decision": "critical" if attachments else "needs_visual_confirmation",
                "review_policy": "evidence_first",
                "manual_review_required": not bool(attachments),
            }
        )
        return spec


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Attachment],
    allowed_skills: list[str],
) -> int:
    selected = SpecBuilder._select_skill(routing_text, attachments, allowed_skills)
    return 100 if selected == skill_name else 0


def build_spec(
    skill_name: str,
    user_text: str,
    routing_text: str,
    attachments: list[Attachment],
    base_spec: dict[str, Any],
) -> dict[str, Any]:
    builder = SpecBuilder()
    spec = dict(base_spec)
    spec["title"] = builder._title(user_text, skill_name)
    quality = builder._quality_requirements(skill_name)
    spec["quality_requirements"] = quality
    spec["verification_rules"] = quality
    if skill_name == "drawio-generation":
        spec.update(builder._drawio_spec(routing_text))
    elif skill_name == "pptx-generation":
        spec.update(builder._ppt_spec(routing_text))
    elif skill_name == "excel-generation":
        spec.update(builder._excel_spec())
    elif skill_name == "xmind-generation":
        spec.update(builder._xmind_spec())
    elif skill_name == "markdown-rendering":
        spec.update(builder._markdown_spec())
    elif skill_name == "behavior-detection":
        spec.update(builder._behavior_spec(user_text, attachments))
    elif skill_name == "behavior-review":
        spec.update(builder._behavior_review_spec(user_text, attachments))
    return spec
