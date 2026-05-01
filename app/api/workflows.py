from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.core.agent import AgentRuntime


router = APIRouter(prefix="/api/workflows", tags=["workflows"])
runtime = AgentRuntime()


@router.get("")
async def list_workflows() -> dict[str, list[dict[str, Any]]]:
    names = sorted(runtime.workflow_registry.names())
    return {"workflows": [_workflow_payload(name) for name in names]}


def _workflow_payload(name: str) -> dict[str, Any]:
    descriptions = {
        "artifact_workflow": "确定性产物生成流程：结构化 Spec、Skill 执行、产物预览和校验。",
        "evidence_first_detection": "证据优先行为检测流程：无视觉或结构化证据时只输出文本规则判断。",
    }
    return {
        "name": name,
        "description": descriptions.get(name, "Workflow plugin registered in WorkflowRegistry."),
        "enabled": True,
        "trigger": "explicit_runtime_options",
        "request_example": {
            "runtime_options": {
                "workflow": name,
            },
        },
    }
