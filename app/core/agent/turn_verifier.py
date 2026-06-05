from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.agent.primary_skill_context import PrimarySkillContext
from app.core.agent.tool_loop_guard import composite_skill_contexts, missing_composite_completion_evidence
from app.schemas import RuntimeOptions

_ARTIFACT_SUITE_SKILLS = {
    "drawio-generation",
    "pptx-generation",
    "excel-generation",
    "xmind-generation",
    "markdown-rendering",
    "deliverables-export",
}
_ARTIFACT_SUITE_REQUIRED_ARTIFACTS = {
    "architecture.drawio",
    "architecture.png",
    "deck.pptx",
    "device-template.xlsx",
    "mindmap.xmind",
    "result.md",
    "commands.txt",
    "steps.docx",
}


@dataclass(frozen=True)
class TurnVerification:
    verdict: str
    reason: str
    missing_evidence: list[str] = field(default_factory=list)


def verify_turn_completion(
    *,
    runtime_options: RuntimeOptions | None,
    artifacts: list[dict[str, Any]],
    required_inputs: list[dict[str, Any]],
    latest_tool_results: list[dict[str, Any]],
    tool_call_count: int,
    primary_skill_context: PrimarySkillContext | None = None,
) -> TurnVerification:
    if required_inputs:
        return TurnVerification(
            verdict="blocked",
            reason="Required inputs are still missing, so the turn must stop and wait.",
        )

    if primary_skill_context is not None and primary_skill_context.stages:
        if primary_skill_context.completion_status == "completed":
            return TurnVerification(
                verdict="passed",
                reason="Primary skill stage board is fully completed.",
            )
        if primary_skill_context.active_stage_name:
            return TurnVerification(
                verdict="pending",
                reason=(
                    f"Primary skill is still progressing through stage "
                    f"{primary_skill_context.active_stage_name}."
                ),
                missing_evidence=list(primary_skill_context.blocked_stage_signals),
            )

    composite_context = composite_skill_contexts(runtime_options)
    if composite_context:
        missing = missing_composite_completion_evidence(composite_context, artifacts)
        if missing is None:
            return TurnVerification(
                verdict="passed",
                reason="Composite completion evidence is satisfied by current thread artifacts.",
            )
        return TurnVerification(
            verdict="pending",
            reason="Composite completion evidence is still incomplete.",
            missing_evidence=[missing],
        )

    structured = _structured_verification_outcome(latest_tool_results)
    if structured is not None:
        return structured

    suite_verification = _artifact_suite_completion(runtime_options, artifacts)
    if suite_verification is not None:
        return suite_verification

    if tool_call_count > 0 and artifacts:
        return TurnVerification(
            verdict="pending",
            reason="Artifacts exist, but no explicit structured completion signal has been produced yet.",
        )

    return TurnVerification(verdict="unknown", reason="No structured verification signal is available yet.")


def _structured_verification_outcome(latest_tool_results: list[dict[str, Any]]) -> TurnVerification | None:
    for item in latest_tool_results:
        verification = item.get("verification")
        if isinstance(verification, dict):
            passed = verification.get("passed")
            if passed is True:
                return TurnVerification(
                    verdict="passed",
                    reason="Latest tool result includes verification.passed=true.",
                )
            if passed is False:
                failed_checks = verification.get("failed_checks")
                detail = (
                    ", ".join(str(entry) for entry in failed_checks)
                    if isinstance(failed_checks, list)
                    else "verification failed"
                )
                return TurnVerification(
                    verdict="pending",
                    reason=f"Latest tool result includes verification failure: {detail}.",
                )

        for key in ("verified", "verification_passed", "done", "completed"):
            if item.get(key) is True:
                return TurnVerification(
                    verdict="passed",
                    reason=f"Latest tool result includes {key}=true.",
                )

        status = str(item.get("status") or item.get("result_status") or "").strip().lower()
        if status in {"completed", "verified", "success", "done"}:
            return TurnVerification(
                verdict="passed",
                reason=f"Latest tool result status is {status}.",
            )
        if status in {"blocked", "needs_input", "pending", "partial"}:
            return TurnVerification(
                verdict="pending",
                reason=f"Latest tool result status is {status}.",
            )
    return None


def _artifact_suite_completion(
    runtime_options: RuntimeOptions | None,
    artifacts: list[dict[str, Any]],
) -> TurnVerification | None:
    if runtime_options is None:
        return None
    if (runtime_options.app_template_name or "").strip() != "artifact-suite":
        return None
    selected_skills = {name.strip() for name in runtime_options.selected_skills if name.strip()}
    if not _ARTIFACT_SUITE_SKILLS.issubset(selected_skills):
        return None
    artifact_names = {str(item.get("name") or "").strip().lower() for item in artifacts if isinstance(item, dict)}
    missing = sorted(name for name in _ARTIFACT_SUITE_REQUIRED_ARTIFACTS if name.lower() not in artifact_names)
    if not missing:
        return TurnVerification(
            verdict="passed",
            reason="Artifact suite required deliverables are all present in the thread workspace.",
        )
    return TurnVerification(
        verdict="pending",
        reason="Artifact suite deliverables are still incomplete.",
        missing_evidence=missing,
    )
