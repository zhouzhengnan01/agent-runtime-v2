from __future__ import annotations

from dataclasses import dataclass

from app.core.agent.primary_skill_context import PrimarySkillContext
from app.core.tools import ToolDefinition


@dataclass(frozen=True)
class ToolExposure:
    visible_tools: list[ToolDefinition]
    priority_tools: list[ToolDefinition]
    secondary_tools: list[ToolDefinition]
    hidden_tools: list[ToolDefinition]
    metadata: dict[str, object]


def select_tools_for_phase(
    tools: list[ToolDefinition],
    *,
    phase: str,
    primary_skill_context: PrimarySkillContext | None = None,
) -> ToolExposure:
    stage_priority = _stage_priority_tools(tools, primary_skill_context)
    if stage_priority:
        secondary = [tool for tool in tools if tool not in stage_priority]
        return ToolExposure(
            visible_tools=[*stage_priority, *secondary],
            priority_tools=stage_priority,
            secondary_tools=secondary,
            hidden_tools=[],
            metadata={
                "phase": phase,
                "policy": "primary_stage_owner_priority",
                "visible_tool_count": len(tools),
                "active_stage": primary_skill_context.active_stage_name if primary_skill_context is not None else "",
            },
        )
    primary_skill_priority = _primary_skill_tool(tools, primary_skill_context)
    if primary_skill_priority:
        secondary = _stable_order([tool for tool in tools if tool not in primary_skill_priority])
        return ToolExposure(
            visible_tools=[*primary_skill_priority, *secondary],
            priority_tools=primary_skill_priority,
            secondary_tools=secondary,
            hidden_tools=[],
            metadata={
                "phase": phase,
                "policy": "primary_skill_tool_priority",
                "visible_tool_count": len(tools),
                "primary_skill": primary_skill_context.skill_name if primary_skill_context is not None else "",
            },
        )
    if phase == "explore":
        priority = [tool for tool in tools if _is_read_only_tool(tool) or _is_inventory_tool(tool)]
        secondary = [tool for tool in tools if _is_delegate_tool(tool) and tool not in priority]
        hidden = [tool for tool in tools if tool not in priority and tool not in secondary]
        visible = priority
        if visible:
            return ToolExposure(
                visible_tools=visible,
                priority_tools=priority,
                secondary_tools=secondary,
                hidden_tools=hidden,
                metadata={
                    "phase": phase,
                    "policy": "read_preferred",
                    "visible_tool_count": len(visible),
                },
            )
    if phase == "verify":
        priority = [tool for tool in tools if _is_verification_friendly_tool(tool)]
        secondary = [tool for tool in tools if _is_delegate_tool(tool) and tool not in priority]
        hidden = [tool for tool in tools if tool not in priority and tool not in secondary]
        visible = priority
        if visible:
            return ToolExposure(
                visible_tools=visible,
                priority_tools=priority,
                secondary_tools=secondary,
                hidden_tools=hidden,
                metadata={
                    "phase": phase,
                    "policy": "verification_preferred",
                    "visible_tool_count": len(visible),
                },
            )
    if phase == "execute":
        priority = _stable_order([tool for tool in tools if _is_execution_tool(tool)])
        secondary = _stable_order([tool for tool in tools if tool not in priority])
        visible = _stable_order([*priority, *secondary])
        return ToolExposure(
            visible_tools=visible,
            priority_tools=priority,
            secondary_tools=secondary,
            hidden_tools=[],
            metadata={
                "phase": phase,
                "policy": "execution_priority",
                "visible_tool_count": len(visible),
            },
        )
    return ToolExposure(
        visible_tools=tools,
        priority_tools=tools,
        secondary_tools=[],
        hidden_tools=[],
        metadata={
            "phase": phase,
            "policy": "all_selected_tools",
            "visible_tool_count": len(tools),
        },
    )


def _is_read_only_tool(tool: ToolDefinition) -> bool:
    source_type = str(tool.source.get("type") or "")
    operation = str(tool.source.get("operation") or "")
    return (
        (source_type == "local" and operation in {"read_file", "search_text", "present_files"})
        or (source_type == "artifact_workspace" and operation in {"manifest_read", "list", "read"})
        or source_type in {"memory", "markdown_memory", "manual"}
    )


def _is_inventory_tool(tool: ToolDefinition) -> bool:
    return tool.name in {"present_files", "artifact_list", "artifact_manifest_read"}


def _is_delegate_tool(tool: ToolDefinition) -> bool:
    return str(tool.source.get("type") or "") == "delegate"


def _is_verification_friendly_tool(tool: ToolDefinition) -> bool:
    source_type = str(tool.source.get("type") or "")
    operation = str(tool.source.get("operation") or "")
    return (
        _is_read_only_tool(tool)
        or _is_delegate_tool(tool)
        or (source_type == "local" and operation in {"shell_command", "validate_yolo_training_inputs"})
        or (source_type == "skill" and tool.name.endswith(("evaluator", "verifier")))
        or (source_type == "artifact_workspace" and operation == "manifest_update")
    )


def _is_execution_tool(tool: ToolDefinition) -> bool:
    source_type = str(tool.source.get("type") or "")
    operation = str(tool.source.get("operation") or "")
    return (
        source_type == "skill"
        or (
            source_type == "local"
            and operation in {"write_file", "patch_file", "shell_command", "todo", "extract_archive", "validate_yolo_training_inputs"}
        )
        or (source_type == "artifact_workspace" and operation in {"write", "patch", "manifest_update"})
    )


def _stage_priority_tools(
    tools: list[ToolDefinition],
    primary_skill_context: PrimarySkillContext | None,
) -> list[ToolDefinition]:
    if primary_skill_context is None or not primary_skill_context.active_stage_owner_skills:
        return []
    owner_skills = set(primary_skill_context.active_stage_owner_skills)
    priority = [
        tool
        for tool in tools
        if str(tool.source.get("type") or "") == "skill" and tool.name in owner_skills
    ]
    return priority


def _primary_skill_tool(
    tools: list[ToolDefinition],
    primary_skill_context: PrimarySkillContext | None,
) -> list[ToolDefinition]:
    if primary_skill_context is None or not primary_skill_context.skill_name:
        return []
    return [
        tool
        for tool in tools
        if str(tool.source.get("type") or "") == "skill" and tool.name == primary_skill_context.skill_name
    ]


def _stable_order(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    def rank(tool: ToolDefinition) -> tuple[int, str]:
        source_type = str(tool.source.get("type") or "")
        operation = str(tool.source.get("operation") or "")
        if source_type == "local" and operation == "read_file":
            return (0, tool.name)
        if source_type == "artifact_workspace" and operation in {"manifest_read", "list", "read"}:
            return (1, tool.name)
        if source_type == "local" and operation == "search_text":
            return (2, tool.name)
        if source_type == "local" and operation == "patch_file":
            return (3, tool.name)
        if source_type == "local" and operation == "write_file":
            return (4, tool.name)
        if source_type == "local" and operation in {"extract_archive", "validate_yolo_training_inputs"}:
            return (5, tool.name)
        if source_type == "artifact_workspace" and operation in {"write", "patch", "manifest_update"}:
            return (6, tool.name)
        if source_type == "local" and operation in {"todo", "shell_command"}:
            return (7, tool.name)
        if source_type == "skill":
            return (8, tool.name)
        return (9, tool.name)

    return sorted(tools, key=rank)
