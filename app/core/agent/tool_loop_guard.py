from __future__ import annotations

from typing import Any

from app.schemas import RuntimeOptions


def guard_unverified_completion(
    reply: str,
    runtime_options: RuntimeOptions | None,
    *,
    available_tool_count: int = 0,
    executed_tool_count: int = 0,
    required_inputs: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    mode = runtime_mode(runtime_options)
    if mode != "yolo":
        return reply, {}
    selected_skills = selected_skill_names(runtime_options)
    if not selected_skills:
        return reply, {}
    if available_tool_count <= 0:
        guarded = (
            "已选择 skills，但本轮没有可用工具可执行，不能进入大流程执行。\n"
            "请检查 app 配置中的 selected_skills 是否是本地已安装 skill 名称，而不是平台侧 ID。"
        )
        return guarded, {
            "unavailable_selected_skills_blocked": True,
            "selected_skills": sorted(selected_skills),
        }
    if required_inputs:
        return reply, {}
    if executed_tool_count > 0:
        return reply, {}
    composite_contexts = composite_skill_contexts(runtime_options)
    if not composite_contexts:
        return reply, {}
    completion_gap = missing_composite_completion_evidence(composite_contexts, artifacts or [])
    if completion_gap is None:
        return reply, {}
    guarded = (
        "本次没有执行任何工具调用，且当前 composite skill 缺少完成证据，不能声明大流程已完成。\n"
        f"缺少的完成条件证据：{completion_gap}\n"
        "请继续执行对应子 skill，或提供已经生成的线程产物后再继续。"
    )
    return guarded, {
        "unverified_completion_blocked": True,
        "missing_completion_evidence": completion_gap,
        "selected_skills": sorted(selected_skills),
    }


def runtime_mode(runtime_options: RuntimeOptions | None) -> str:
    if runtime_options is None or runtime_options.mode is None:
        return "edit"
    return runtime_options.mode


def selected_skill_names(runtime_options: RuntimeOptions | None) -> set[str]:
    if runtime_options is None:
        return set()
    return {name.strip() for name in runtime_options.selected_skills if name.strip()}


def composite_skill_contexts(runtime_options: RuntimeOptions | None) -> list[dict[str, Any]]:
    if runtime_options is None:
        return []
    raw_contexts = runtime_options.config_options.get("composite_skills")
    if not isinstance(raw_contexts, list):
        return []
    return [dict(item) for item in raw_contexts if isinstance(item, dict)]


def missing_composite_completion_evidence(
    composite_contexts: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> str | None:
    artifact_names = artifact_evidence_names(artifacts)
    for item in composite_contexts:
        missing = missing_done_when_condition(item, artifact_names)
        if missing is not None:
            return missing
    return None


def artifact_evidence_names(artifacts: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        for value in (item.get("name"), item.get("path")):
            if not isinstance(value, str):
                continue
            normalized = value.strip()
            if normalized:
                names.add(normalized)
    return names


def missing_done_when_condition(composite_context: dict[str, Any], artifact_names: set[str]) -> str | None:
    done_when = string_list(composite_context.get("done_when"))
    if done_when:
        for condition in done_when:
            if not condition_satisfied_by_artifacts(condition, artifact_names):
                return condition
        return None
    stages = composite_context.get("stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        for produced in string_list(stage.get("produces")):
            if not artifact_name_matches(produced, artifact_names):
                return produced
    return None


def condition_satisfied_by_artifacts(condition: str, artifact_names: set[str]) -> bool:
    normalized = condition.strip()
    if not normalized:
        return True
    lowered = normalized.lower()
    if " exists" in lowered:
        candidate = normalized[: lowered.index(" exists")].strip()
        if candidate:
            return artifact_name_matches(candidate, artifact_names)
    return artifact_name_matches(normalized, artifact_names)


def artifact_name_matches(expected: str, artifact_names: set[str]) -> bool:
    target = expected.strip()
    if not target:
        return True
    target_lower = target.lower()
    for artifact_name in artifact_names:
        normalized = artifact_name.strip()
        if not normalized:
            continue
        normalized_lower = normalized.lower()
        if (
            normalized_lower == target_lower
            or normalized_lower.endswith("/" + target_lower)
            or target_lower in normalized_lower
        ):
            return True
    return False


def reply_with_required_inputs(
    reply: str,
    required_inputs: list[dict[str, Any]],
    *,
    tool_call_count: int,
    artifacts: list[dict[str, Any]],
) -> str:
    if not required_inputs:
        return reply
    lines = [
        f"已执行 {tool_call_count} 次工具调用，当前流程需要补充输入后继续。"
        if tool_call_count
        else "当前流程需要补充输入后继续。"
    ]
    if artifacts:
        names = ", ".join(str(item.get("name") or item.get("path") or "artifact") for item in artifacts[:6])
        if len(artifacts) > 6:
            names += f" 等 {len(artifacts)} 个文件"
        lines.append(f"已生成阶段性产物：{names}")
    lines.append("还需要：")
    for item in required_inputs[:8]:
        stage = str(item.get("stage") or item.get("type") or "input")
        reason = str(item.get("reason") or "缺少必要输入。")
        lines.append(f"- {stage}: {reason}")
    if len(required_inputs) > 8:
        lines.append(f"- 还有 {len(required_inputs) - 8} 项输入要求，详见运行元数据。")
    lines.append("补齐后在同一 thread 继续发送，我会基于已有产物接着跑后续步骤。")
    return "\n".join(lines)


def string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
