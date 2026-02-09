"""
Database bootstrap: seed default templates and agents.

This runs at startup to ensure a usable out-of-the-box experience on a fresh DB.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import settings
from app.db.session import SessionLocal
from app.models.agent import Agent
from app.models.agent_template import AgentTemplate
from app.models.tool import Tool

logger = logging.getLogger(__name__)


def _convert_schema_to_inputs(parameters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not parameters or "properties" not in parameters:
        return []

    inputs: List[Dict[str, Any]] = []
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []) or [])

    for prop_id, prop_def in properties.items():
        prop_type = prop_def.get("type", "string")
        value_type: Dict[str, Any] = {"type": "string"}

        if prop_type == "integer":
            value_type["type"] = "int"
        elif prop_type == "number":
            value_type["type"] = "float"
        elif prop_type == "boolean":
            value_type["type"] = "boolean"
        elif prop_type == "array":
            value_type["type"] = "array"
            if "items" in prop_def:
                value_type["elementType"] = prop_def["items"]
        elif prop_type == "object":
            value_type["type"] = "object"
            if "properties" in prop_def:
                value_type["properties"] = prop_def["properties"]

        tool_input = {
            "id": prop_id,
            "name": prop_def.get("title", prop_id),
            "valueType": value_type,
            "required": prop_id in required,
            "description": prop_def.get("description"),
        }

        if "default" in prop_def:
            tool_input["default"] = prop_def["default"]

        inputs.append(tool_input)

    return inputs


def seed_default_tools() -> None:
    """Ensure internal tools are registered in the database."""
    try:
        from app.core.tools import TOOL_REGISTRY
    except Exception as exc:
        logger.warning("⚠️ Tool registry import failed, skip tool seed: %s", exc)
        return

    db = SessionLocal()
    try:
        created = 0
        updated = 0

        for tool_id, tool_cls in TOOL_REGISTRY.items():
            tool_name = getattr(tool_cls, "name", tool_id) or tool_id
            description = getattr(tool_cls, "description", "") or ""
            inputs = getattr(tool_cls, "inputs", None)
            parameters = getattr(tool_cls, "parameters", None)
            output = getattr(tool_cls, "output", None)
            async_mode = bool(getattr(tool_cls, "async_mode", False))
            estimated_time = getattr(tool_cls, "estimated_time", None)
            timeout = getattr(tool_cls, "timeout", None)

            if not inputs and parameters:
                inputs = _convert_schema_to_inputs(parameters)

            if output is None:
                output = {"type": "object"}

            tool = db.query(Tool).filter(Tool.tool_id == tool_id).first()
            if not tool:
                tool = Tool(
                    tool_id=tool_id,
                    name=tool_name,
                    description=description,
                    inputs=inputs or [],
                    output=output,
                    is_async=async_mode,
                    execution_mode="async" if async_mode else "sync",
                    estimated_time=int(estimated_time or 3),
                    timeout=int(timeout or 30),
                    source="internal",
                    is_active=True,
                )
                db.add(tool)
                created += 1
                continue

            changed = False
            if not tool.name:
                tool.name = tool_name
                changed = True
            if not tool.description and description:
                tool.description = description
                changed = True
            if not tool.inputs and inputs:
                tool.inputs = inputs
                changed = True
            if not tool.output and output:
                tool.output = output
                changed = True
            if tool.source != "internal":
                tool.source = "internal"
                changed = True
            if tool.is_async is None:
                tool.is_async = async_mode
                changed = True
            if tool.execution_mode is None:
                tool.execution_mode = "async" if async_mode else "sync"
                changed = True
            if tool.estimated_time is None and estimated_time is not None:
                tool.estimated_time = int(estimated_time)
                changed = True
            if tool.timeout is None and timeout is not None:
                tool.timeout = int(timeout)
                changed = True
            if tool.is_active is None:
                tool.is_active = True
                changed = True

            if changed:
                updated += 1

        if created or updated:
            db.commit()
            logger.info(
                "✅ Default tools ensured (created=%s, updated=%s)",
                created,
                updated,
            )
        else:
            logger.info("ℹ️  Default tools already present")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _upsert_template(
    *,
    template_id: str,
    name: str,
    description: str,
    class_path: str,
    framework: str = "langchain",
    type_: Optional[str] = None,
    default_tools: Optional[List[Any]] = None,
    default_config: Optional[Dict[str, Any]] = None,
    capabilities: Optional[List[str]] = None,
) -> bool:
    """Ensure template exists and key fields are correct.

    Returns True if created/updated.
    """
    db = SessionLocal()
    try:
        now = datetime.now()
        template = (
            db.query(AgentTemplate)
            .filter(AgentTemplate.id == template_id)
            .first()
        )
        if not template:
            template = (
                db.query(AgentTemplate)
                .filter(AgentTemplate.template_id == template_id)
                .first()
            )

        changed = False
        if not template:
            template = AgentTemplate(
                id=template_id,
                template_id=template_id,
                name=name,
                description=description,
                type=type_ or template_id,
                class_path=class_path,
                config={},
                default_tools=default_tools or [],
                default_config=default_config or {},
                capabilities=capabilities or [],
                is_active=True,
                framework=framework,
                created_at=now,
                updated_at=now,
            )
            db.add(template)
            changed = True
        else:
            # Keep user customizations, only repair the known-good core fields.
            if not template.template_id:
                template.template_id = template_id
                changed = True
            if not template.name:
                template.name = name
                changed = True
            if not template.description:
                template.description = description
                changed = True
            if not template.type:
                template.type = type_ or template_id
                changed = True
            if template.class_path != class_path:
                template.class_path = class_path
                changed = True
            if template.framework != framework:
                template.framework = framework
                changed = True
            if template.is_active is not True:
                template.is_active = True
                changed = True
            if changed:
                template.updated_at = now

        if changed:
            db.commit()
        return changed
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _create_agent_if_missing(
    *,
    agent_id: str,
    name: str,
    type_: str,
    description: str = "",
    prompt: str = "",
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    tools: Optional[List[str]] = None,
    opening_statement: Optional[str] = None,
    suggested_questions: bool = True,
) -> bool:
    """Ensure agent exists (do not overwrite if already present)."""
    db = SessionLocal()
    try:
        existing = db.query(Agent).filter(Agent.id == agent_id).first()
        if existing:
            return False

        now = datetime.now()
        agent = Agent(
            id=agent_id,
            name=name,
            description=description,
            type=type_,
            config={
                "template": type_,
                "model": model or settings.DEFAULT_LLM_MODEL,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools or [],
                # historical compatibility
                "prompt": prompt,
                "system_prompt": prompt,
            },
            status="active",
            created_by="system",
            opening_statement=opening_statement,
            suggested_questions=suggested_questions,
            output_format="text",
            created_at=now,
            updated_at=now,
        )
        db.add(agent)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def seed_default_templates_and_agents() -> None:
    """Idempotent: can be safely called on every startup."""
    changed_any = False

    # Default templates (for DB-backed listing + factory DB loading).
    changed_any |= _upsert_template(
        template_id="tool_calling",
        name="工具调用模板",
        description="通用工具调用智能体（支持工具确认/流式输出）",
        class_path="app.core.agents.template_agent.tool_calling_template.ToolCallingAgent",
        framework="langchain",
        type_="tool_calling",
        capabilities=["streaming", "tools", "confirmation"],
    )
    changed_any |= _upsert_template(
        template_id="stepwise_tool",
        name="复杂任务编排模板",
        description="面向复杂任务的逐步规划与工具链编排，支持会话Markdown记录",
        class_path="app.core.agents.template_agent.stepwise_tool_template.StepwiseToolAgent",
        framework="langchain",
        type_="stepwise_tool",
        capabilities=["streaming", "tools", "confirmation", "stepwise", "session_markdown"],
    )
    changed_any |= _upsert_template(
        template_id="video_inspection",
        name="视频巡检模板",
        description="视频/图片分析与巡检报告生成",
        class_path="app.core.agents.template_agent.video_inspection_template.VideoInspectionAgent",
        framework="langchain",
        type_="video_inspection",
        capabilities=["streaming", "multimodal"],
    )

    changed_any |= _upsert_template(
        template_id="video_patrol",
        name="视频巡检(视频流)模板",
        description="面向 RTSP/视频流 的实时巡检与轮询任务模板（支持 JSON 事件输出）",
        class_path="app.core.agents.template_agent.video_patrol_template.VideoPatrolAgent",
        framework="langchain",
        type_="video_patrol",
        capabilities=["streaming", "multimodal", "machine"],
    )

    # Default agents (used by frontend/websocket endpoints).
    changed_any |= _create_agent_if_missing(
        agent_id="iotHomeAgent",
        name="IoT智能助手",
        type_="tool_calling",
        description="物联网设备与平台运维智能助手",
        prompt="你是 JetLinks IoT 智能助手。请用中文回答，必要时调用工具获取实时数据。",
        opening_statement="你好！我是 IoT 智能助手，有什么可以帮你？",
        suggested_questions=True,
    )
    changed_any |= _create_agent_if_missing(
        agent_id="video_inspection",
        name="视频巡检智能体",
        type_="video_inspection",
        description="专业的视频/图片巡检与安全分析智能体",
        prompt="你是专业的视频巡检智能体，擅长从视频/图片中识别安全隐患并生成结构化报告。",
        opening_statement="你好！我是视频巡检智能体，请上传视频/图片或描述你的巡检需求。",
        suggested_questions=True,
    )

    if changed_any:
        logger.info("✅ Default templates/agents ensured (seeded or repaired)")
    else:
        logger.info("ℹ️  Default templates/agents already present")
