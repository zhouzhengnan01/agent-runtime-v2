from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.agent.conditional_loop import conditional_loop_policy, latest_tool_message_text
from app.core.agent.primary_skill_context import PrimarySkillContext
from app.core.agent.turn_verifier import TurnVerification
from app.schemas import RuntimeOptions


TURN_POLICY_PROMPT = {
    "explore": (
        "Current turn phase: explore. First inspect the available context, files, "
        "artifacts, and constraints before making irreversible changes. Prefer "
        "reading, searching, listing, and gathering evidence."
    ),
    "execute": (
        "Current turn phase: execute. The task appears actionable. Prefer tools "
        "that produce or update artifacts needed to complete the requested work."
    ),
    "verify": (
        "Current turn phase: verify. Prioritize validation, inspection, and "
        "artifact checks before claiming completion."
    ),
    "finalize": (
        "Current turn phase: finalize. Summarize the completed work accurately, "
        "and avoid claiming outputs that cannot be evidenced from the thread state."
    ),
}


@dataclass(frozen=True)
class TurnPolicy:
    phase: str
    reason: str

    def prompt_fragment(self) -> str:
        guidance = TURN_POLICY_PROMPT.get(self.phase, "")
        if not guidance:
            return ""
        return "\n".join(
            [
                "Turn policy is active.",
                f"Phase: {self.phase}",
                f"Why: {self.reason}",
                guidance,
            ]
        )


@dataclass(frozen=True)
class TurnPolicyInput:
    messages: list[dict[str, Any]]
    available_artifacts: list[dict[str, Any]]
    current_phase: str
    tool_call_count: int
    required_inputs: list[dict[str, Any]]
    latest_tool_results: list[dict[str, Any]]
    latest_assistant_reply: str
    verification: TurnVerification
    primary_skill_context: PrimarySkillContext | None = None
    runtime_options: RuntimeOptions | None = None


def decide_turn_policy(input_state: TurnPolicyInput) -> TurnPolicy:
    """Decide the current phase from accumulated turn state.

    This behaves like a small state machine rather than a one-shot keyword
    classifier. The decision prefers explicit execution signals from tool
    outcomes and required-input state, then falls back to user-intent hints.
    """

    if input_state.verification.verdict == "blocked":
        return TurnPolicy(phase="finalize", reason=input_state.verification.reason)

    if input_state.verification.verdict == "passed":
        return TurnPolicy(phase="finalize", reason=input_state.verification.reason)

    context = input_state.primary_skill_context
    if context is not None and context.stages:
        if context.blocked_stage_signals:
            return TurnPolicy(
                phase="finalize",
                reason=(
                    f"Primary skill is blocked at stage {context.active_stage_name or context.skill_name} "
                    "and is waiting for missing inputs."
                ),
            )
        if input_state.current_phase == "verify" and context.completion_status != "completed":
            return TurnPolicy(
                phase="execute",
                reason=f"Primary skill still has unfinished stage {context.active_stage_name or context.skill_name}.",
            )
        if context.active_stage_name and context.active_stage_owner_skills:
            return TurnPolicy(
                phase="execute",
                reason=(
                    f"Primary skill active stage is {context.active_stage_name}; "
                    f"prefer owner skills {', '.join(context.active_stage_owner_skills)}."
                ),
            )

    loop_policy = conditional_loop_policy(input_state.messages, input_state.runtime_options)
    if (
        loop_policy is not None
        and input_state.tool_call_count > 0
        and input_state.current_phase in {"execute", "verify"}
        and loop_policy.should_continue(latest_tool_message_text(input_state.messages))
    ):
        return TurnPolicy(
            phase="execute",
            reason=(
                "A conditional tool-loop policy is active and the latest tool result "
                "still matches the retry condition; continue executing tools."
            ),
        )

    if input_state.current_phase == "execute" and input_state.tool_call_count > 0:
        if input_state.verification.verdict == "pending":
            return TurnPolicy(phase="verify", reason=input_state.verification.reason)
        if _has_new_artifacts(input_state.latest_tool_results):
            return TurnPolicy(phase="verify", reason="Execution produced artifacts that should be checked.")

    if input_state.current_phase == "verify":
        if input_state.verification.verdict == "pending":
            return TurnPolicy(phase="verify", reason=input_state.verification.reason)
        return TurnPolicy(phase="verify", reason="The run has entered verification and should inspect evidence.")

    if input_state.current_phase == "explore" and input_state.tool_call_count > 0:
        return TurnPolicy(phase="execute", reason="Exploration already gathered context; proceed to execution.")

    text = _joined_message_text(input_state.messages)
    lower_text = text.lower()
    artifact_count = len(input_state.available_artifacts)

    verify_markers = (
        "验证",
        "校验",
        "检查",
        "review",
        "verify",
        "test",
        "评估",
        "benchmark",
    )
    execute_markers = (
        "生成",
        "修改",
        "修复",
        "写",
        "训练",
        "执行",
        "实现",
        "构建",
        "generate",
        "create",
        "write",
        "train",
        "implement",
        "build",
        "fix",
    )
    explore_markers = (
        "看看",
        "看下",
        "看一下",
        "瞅下",
        "分析",
        "阅读",
        "梳理",
        "了解",
        "look at",
        "analyze",
        "inspect",
        "read",
    )

    has_execute_marker = any(marker in lower_text for marker in execute_markers) or any(marker in text for marker in execute_markers)
    has_verify_marker = any(marker in lower_text for marker in verify_markers) or any(marker in text for marker in verify_markers)
    if has_execute_marker:
        return TurnPolicy(phase="execute", reason="The user request appears to ask for concrete changes or outputs.")
    if has_verify_marker:
        return TurnPolicy(phase="verify", reason="The user request emphasizes validation or review.")
    if artifact_count > 0 and _latest_user_message_is_short(input_state.messages):
        return TurnPolicy(phase="verify", reason="Existing thread artifacts are available and the new turn is short.")
    if any(marker in lower_text for marker in explore_markers) or any(marker in text for marker in explore_markers):
        return TurnPolicy(phase="explore", reason="The user request appears exploratory.")
    return TurnPolicy(phase="execute", reason="Default to actionable execution for unresolved work.")


def _has_new_artifacts(latest_tool_results: list[dict[str, Any]]) -> bool:
    for item in latest_tool_results:
        raw_artifacts = item.get("artifacts")
        if isinstance(raw_artifacts, list) and raw_artifacts:
            return True
    return False


def _joined_message_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages
        if str(message.get("role") or "") in {"user", "assistant"}
    )


def _latest_user_message_is_short(messages: list[dict[str, Any]]) -> bool:
    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        return len(str(message.get("content") or "").strip()) <= 24
    return False
