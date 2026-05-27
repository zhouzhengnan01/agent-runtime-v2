from __future__ import annotations

import uuid
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillDefinition, SkillRegistry, SkillRunner
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult
from app.schemas import ArtifactRef, artifact_content_block


class SkillToolProvider:
    source_type = "skill"

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_runner: SkillRunner | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_runner = skill_runner or SkillRunner(self.artifact_store)

    def list(self) -> list[ToolDefinition]:
        self.skill_registry.reload()
        return [_tool_from_skill(skill) for skill in self.skill_registry.list(executable_only=True)]

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        skill_name = str(tool.source.get("skill_name") or tool.name)
        thread_id = str(arguments.get("_thread_id") or f"mcp-{uuid.uuid4().hex[:10]}")
        spec = {key: value for key, value in arguments.items() if not key.startswith("_")}
        spec["skill_name"] = skill_name
        paths = self.artifact_store.prepare_thread(thread_id)
        result = self.skill_runner.run(skill_name, spec, paths)
        artifacts = [artifact.model_dump() for artifact in result.outputs]
        artifact_text = "\n".join(
            f"- {artifact['name']} ({artifact_content_block(ArtifactRef.model_validate(artifact))['path']})"
            for artifact in artifacts
        ) or "- no artifacts"
        requires_input = result.data.get("requires_input") is True
        raw_required_inputs = result.data.get("required_inputs")
        required_inputs: list[object] = raw_required_inputs if isinstance(raw_required_inputs, list) else []
        content = [
            {
                "type": "text",
                "text": _input_required_text(required_inputs) if requires_input else f"Executed {skill_name}.\n{artifact_text}",
            }
        ]
        if not requires_input:
            content.extend(artifact_content_block(artifact) for artifact in result.outputs)
        return ToolInvocationResult(
            content=content,
            structured_content={
                "skill_name": skill_name,
                "thread_id": thread_id,
                "artifacts": artifacts,
                "data": result.data,
                "requires_input": requires_input,
                "required_inputs": required_inputs,
            },
            is_error=False,
        )


def _tool_from_skill(skill: SkillDefinition) -> ToolDefinition:
    return ToolDefinition(
        name=skill.name,
        title=skill.name,
        description=skill.description,
        input_schema=skill.input_schema or {"type": "object", "additionalProperties": True},
        output_schema=skill.output_schema or {},
        enabled=True,
        editable=False,
        source={"type": "skill", "skill_name": skill.name, "output_kind": skill.output_kind},
    )


def _input_required_text(required_inputs: list[object]) -> str:
    if not required_inputs:
        return "Skill requires additional input before execution can continue."
    lines = ["Skill requires additional input before execution can continue:"]
    for item in required_inputs:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or item.get("type") or "input")
        reason = str(item.get("reason") or "Required input is missing.")
        lines.append(f"- {stage}: {reason}")
    return "\n".join(lines)
