from __future__ import annotations

from typing import Any

from app.core.config import AgentConfig
from app.core.agent.primary_skill_context import PrimarySkillContext, primary_skill_stage_prompt
from app.core.skills import SkillRegistry
from app.core.skills.context_files import SkillMarkdownContext, load_skill_markdown_context
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

EXECUTION_POLICY_GUIDANCE = (
    "Runtime execution policy is active. Follow these app-level execution "
    "rules when choosing tools and deciding whether to continue. Treat them "
    "as operational guidance, while still respecting required inputs, safety "
    "requirements, and explicit user instructions."
)

ITERATIVE_TOOL_GUIDANCE = (
    "Iterative tool guidance is active for conditional tool-loop behavior. When the user gives a repeat-until or "
    "stop condition, inspect each tool result and continue calling the relevant "
    "tool while the latest result still satisfies the continue condition. Stop "
    "only after the stop condition is observed or the tool-round limit is reached. "
    "Do not claim the task is complete after a single tool call when the latest "
    "tool result still contains the value the user said must disappear."
)

YOLO_TRAINING_GUIDANCE = """YOLO training agent guidance is active.
For YOLO, object detection, dataset generation, auto-annotation, best.pt, or gpu-training-orchestrator requests:
- First inspect available uploads/outputs/workspace with present_files when file state matters.
- If a dataset is an archive (.zip, .tar, .tar.gz, .tgz), call extract_archive before annotation or training. Do not pass archives directly to data-auto-annotation or gpu-training-orchestrator.
- Before calling gpu-training-orchestrator, call validate_yolo_training_inputs with dataset_root, ref_image when available, labels, training, runtime, and split.
- If validate_yolo_training_inputs.ready is false, ask for the missing values or call the tool needed to produce them; do not start training.
- If the user requests dry_run or skip_training, include dry_run=true and skip_training=true in the gpu-training-orchestrator arguments.
- Prefer training.amp=false for GPU YOLO training unless the user explicitly requests AMP; this avoids Ultralytics AMP self-check downloads on restricted-network machines.
- Pass the validated extracted dataset_root to gpu-training-orchestrator. Do not substitute /mnt/user-data/outputs for dataset_root unless validation confirms it contains images.
- After training, inspect artifacts with artifact_list or present_files and only claim training success when best.pt/results.csv or an explicit successful training summary exists.
- If an execution tool is unavailable, say exactly which tool is missing instead of writing scripts that cannot be executed."""


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
    prompt = prompt_with_execution_policy(prompt, runtime_options)
    prompt = prompt_with_yolo_training_guidance(prompt, runtime_options)
    prompt = f"{prompt}\n\n{ITERATIVE_TOOL_GUIDANCE}"
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


def prompt_with_runtime_mcp_discovery_failures(prompt: str, runtime_options: RuntimeOptions | None) -> str:
    if runtime_options is None:
        return prompt
    raw_failures = runtime_options.config_options.get("runtime_mcp_discovery_failures")
    if not isinstance(raw_failures, list):
        return prompt
    lines: list[str] = []
    for item in raw_failures:
        if not isinstance(item, dict):
            continue
        server_name = str(item.get("server_name") or "").strip() or "unnamed"
        server_type = str(item.get("server_type") or "").strip() or "unknown"
        endpoint = str(item.get("endpoint") or "").strip() or "local"
        reason = str(item.get("reason") or "").strip() or "unknown"
        error = str(item.get("error") or "").strip()
        rendered = f"- {server_name} ({server_type}, {endpoint}) failed: {reason}"
        if error:
            rendered = f"{rendered}: {error[:300]}"
        lines.append(rendered)
    if not lines:
        return prompt
    guidance = "\n".join(
        [
            "Runtime MCP discovery notes:",
            "Some requested MCP servers were not exposed as tools. If the user asks about MCP availability or safety, use these facts instead of guessing.",
            *lines,
        ]
    )
    return f"{prompt}\n\n{guidance}"


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
    sections = [PRIMARY_SKILL_GUIDANCE, rendered]
    markdown_context = load_skill_markdown_context(skill)
    rendered_markdown_context = render_skill_markdown_context(markdown_context)
    if rendered_markdown_context:
        sections.append(rendered_markdown_context)
    guidance = "\n\n".join(sections)
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


def prompt_with_execution_policy(prompt: str, runtime_options: RuntimeOptions | None) -> str:
    if runtime_options is None:
        return prompt
    raw_policy = runtime_options.config_options.get("agent_execution_policy")
    if not isinstance(raw_policy, dict):
        raw_policy = runtime_options.config_options.get("executionPolicy")
    if not isinstance(raw_policy, dict):
        return prompt
    rendered = render_execution_policy(raw_policy)
    if not rendered:
        return prompt
    return f"{prompt}\n\n{EXECUTION_POLICY_GUIDANCE}\n{rendered}"


def prompt_with_yolo_training_guidance(prompt: str, runtime_options: RuntimeOptions | None) -> str:
    if runtime_options is None:
        return prompt
    selected = {item.strip() for item in runtime_options.selected_skills if item.strip()}
    mode = str(runtime_options.mode or "").strip()
    policy = runtime_options.config_options.get("agent_execution_policy")
    policy_text = json_like_text(policy)
    enabled = (
        mode == "yolo"
        or "gpu-training-orchestrator" in selected
        or "data-auto-annotation" in selected
        or "image-dataset-generation" in selected
        or "yolo" in policy_text.lower()
        or "训练" in policy_text
    )
    if not enabled:
        return prompt
    return f"{prompt}\n\n{YOLO_TRAINING_GUIDANCE}"


def render_execution_policy(policy: dict[str, Any]) -> str:
    name = str(policy.get("name") or policy.get("id") or "").strip()
    mode = str(policy.get("mode") or "").strip()
    lines: list[str] = []
    title = "Execution policy"
    if name:
        title = f"{title}: {name}"
    lines.append(title)
    if mode:
        lines.append(f"Mode: {mode}")
    stage_order = string_list(policy.get("stage_order") or policy.get("stageOrder"))
    if stage_order:
        lines.append(f"Stage order: {', '.join(stage_order)}")
    instructions = string_list(policy.get("instructions"))
    if instructions:
        lines.append("Instructions:")
        lines.extend(f"- {item}" for item in instructions)
    completion_gates = string_list(policy.get("completion_gates") or policy.get("completionGates"))
    if completion_gates:
        lines.append("Completion gates:")
        lines.extend(f"- {item}" for item in completion_gates)
    return "\n".join(lines).strip()


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


def render_skill_markdown_context(context: SkillMarkdownContext | None) -> str:
    if context is None:
        return ""
    lines = [
        "Primary skill package instructions:",
        f"Source: {context.skill_md_path}",
        "The SKILL.md and declared reference files below have already been loaded into this prompt.",
        "Use the embedded content directly. If a tool read is still needed, use the declared relative path such as references/example.md, not an absolute host path.",
        "",
        "## SKILL.md",
        "",
        context.skill_md,
    ]
    if context.skill_md_truncated:
        lines.append("\n[SKILL.md truncated by runtime context limit]")
    if context.references:
        lines.extend(["", "## Declared Reference Files"])
        for reference in context.references:
            lines.extend(
                [
                    "",
                    f"### {reference.path}",
                    "",
                    reference.content,
                ]
            )
            if reference.truncated:
                lines.append(f"\n[{reference.path} truncated by runtime context limit]")
    return "\n".join(lines).strip()


def string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def json_like_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(str(key))
            parts.append(json_like_text(item))
        return " ".join(parts)
    if isinstance(value, list | tuple):
        return " ".join(json_like_text(item) for item in value)
    return str(value)
