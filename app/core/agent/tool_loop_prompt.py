from __future__ import annotations

from typing import Any

from app.core.config import AgentConfig
from app.core.agent.primary_skill_context import PrimarySkillContext, primary_skill_stage_prompt
from app.core.skills import SkillRegistry
from app.core.tools import ToolInvocationService
from app.schemas import RuntimeOptions

PRIMARY_SKILL_GUIDANCE = (
    "Primary skill guidance is active. Treat the first selected skill as the "
    "default first capability for this run. When the user's request matches "
    "it, prefer using that skill before other selected skills. Use other "
    "selected skills only when they are clearly needed to complete the "
    "request or support the primary skill outcome."
)

COMPOSITE_SKILL_GUIDANCE = (
    "Composite skill guidance is active. Treat each composite skill as an "
    "optional high-level capability map, not as a fixed workflow. Use the "
    "available child skill tools autonomously, pass artifacts from one stage "
    "to the next, and do not claim completion until the done_when conditions "
    "are satisfied."
)


def build_system_prompt(
    agent_config: AgentConfig,
    runtime_options: RuntimeOptions | None,
    *,
    tool_service: ToolInvocationService,
    primary_skill_context: PrimarySkillContext | None = None,
) -> tuple[str, int]:
    """Build the system prompt used by the default tool loop.

    The assembly order is intentional:
    1. base agent prompt
    2. primary skill guidance
    3. composite skill guidance
    4. long-term memory context
    """

    prompt = prompt_with_primary_skill(
        agent_config.prompts.system,
        runtime_options,
        tool_service=tool_service,
    )
    prompt = prompt_with_composite_skills(prompt, runtime_options)
    prompt = prompt_with_primary_skill_stage_context(prompt, primary_skill_context)
    if not agent_config.memory.enabled or not agent_config.memory.inject_context:
        return prompt, 0
    try:
        memories = tool_service.memory_store.list(
            agent_config.name,
            limit=agent_config.memory.max_items,
            scope=agent_config.memory.scope,
        )
    except Exception:
        return prompt, 0
    if not memories:
        return prompt, 0
    lines = []
    for item in memories:
        tag_text = f" tags={','.join(item.tags)}" if item.tags else ""
        lines.append(f"- {item.text} (id={item.id}{tag_text})")
    memory_prompt = "\n".join(
        [
            "可用长期记忆如下。仅在与当前请求相关时使用；不要编造未列出的记忆。",
            *lines,
        ]
    )
    return f"{prompt}\n\n{memory_prompt}", len(memories)


def prompt_with_primary_skill_stage_context(prompt: str, context: PrimarySkillContext | None) -> str:
    rendered = primary_skill_stage_prompt(context)
    if not rendered:
        return prompt
    return f"{prompt}\n\n{rendered}"


def prompt_with_primary_skill(
    prompt: str,
    runtime_options: RuntimeOptions | None,
    *,
    tool_service: ToolInvocationService,
) -> str:
    if runtime_options is None:
        return prompt
    selected_skills = runtime_options.selected_skills
    if not selected_skills:
        return prompt
    primary_skill_name = selected_skills[0].strip()
    if not primary_skill_name:
        return prompt
    try:
        skill = SkillRegistry(tool_service.root_dir).get(primary_skill_name)
    except KeyError:
        return prompt
    rendered = render_primary_skill_context(skill)
    if not rendered:
        return prompt
    guidance = "\n\n".join([PRIMARY_SKILL_GUIDANCE, rendered])
    return f"{prompt}\n\n{guidance}"


def prompt_with_composite_skills(prompt: str, runtime_options: RuntimeOptions | None) -> str:
    if runtime_options is None:
        return prompt
    raw_contexts = runtime_options.config_options.get("composite_skills")
    if not isinstance(raw_contexts, list):
        return prompt
    sections: list[str] = []
    for item in raw_contexts:
        if not isinstance(item, dict):
            continue
        rendered = render_composite_skill_context(item)
        if rendered:
            sections.append(rendered)
    if not sections:
        return prompt
    guidance = "\n\n".join([COMPOSITE_SKILL_GUIDANCE, *sections])
    return f"{prompt}\n\n{guidance}"


def render_composite_skill_context(item: dict[str, Any]) -> str:
    name = str(item.get("name") or "").strip()
    if not name:
        return ""
    description = str(item.get("description") or "").strip()
    child_skills = string_list(item.get("child_skills"))
    done_when = string_list(item.get("done_when"))
    lines = [f"Composite skill: {name}"]
    if description:
        lines.append(f"Purpose: {description}")
    if child_skills:
        lines.append(f"Child skills: {', '.join(child_skills)}")
    stages = item.get("stages")
    if isinstance(stages, list) and stages:
        lines.append("Stages:")
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            stage_id = str(stage.get("id") or stage.get("name") or "").strip()
            skill = str(stage.get("skill") or "").strip()
            produces = string_list(stage.get("produces"))
            produce_text = f" -> produces {', '.join(produces)}" if produces else ""
            label = stage_id or skill
            lines.append(f"- {label}: use {skill}{produce_text}".strip())
    if done_when:
        lines.append("Done when:")
        lines.extend(f"- {condition}" for condition in done_when)
    return "\n".join(lines)


def render_primary_skill_context(skill: Any) -> str:
    name = str(getattr(skill, "name", "") or "").strip()
    if not name:
        return ""
    description = str(getattr(skill, "description", "") or "").strip()
    routing = getattr(skill, "routing", None)
    routing_summary = ""
    if isinstance(routing, dict):
        raw_summary = routing.get("summary")
        if isinstance(raw_summary, str):
            routing_summary = raw_summary.strip()
    input_schema = getattr(skill, "input_schema", None)
    required_inputs: list[str] = []
    if isinstance(input_schema, dict):
        raw_required = input_schema.get("required")
        if isinstance(raw_required, list):
            required_inputs = [
                str(item).strip()
                for item in raw_required
                if isinstance(item, str) and item.strip() and item.strip() != "skill_name"
            ]
    quality_template = string_list(getattr(skill, "quality_template", ()))
    lines = [f"Primary skill: {name}"]
    if description:
        lines.append(f"Purpose: {description}")
    if routing_summary:
        lines.append(f"When to use: {routing_summary}")
    if required_inputs:
        lines.append(f"Required inputs: {', '.join(required_inputs)}")
    if quality_template:
        lines.append(f"Quality focus: {', '.join(quality_template)}")
    return "\n".join(lines)


def string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
