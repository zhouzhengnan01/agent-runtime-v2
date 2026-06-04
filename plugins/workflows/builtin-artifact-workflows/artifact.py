from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.sandbox import (
    SandboxDecision,
    SandboxExecutionError,
    SandboxRunContext,
    SandboxSkillRunner,
    load_sandbox_config,
    load_sandbox_policy,
)
from app.core.skills import SkillRegistry, SkillRunResult, SkillRunner
from app.core.workflow.llm_spec_planner import LlmSpecPlanner
from app.core.workflow.spec_builder import SpecBuilder
from app.core.workflow.verifier import OutputVerifier
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions, VerificationResult


LOGGER = logging.getLogger("uvicorn.error")


class ArtifactWorkflow:
    """Spec -> skill -> artifact -> verifier -> optional repair retry."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        spec_builder: SpecBuilder | None = None,
        spec_planner: LlmSpecPlanner | None = None,
        skill_registry: SkillRegistry | None = None,
        skill_runner: SkillRunner | None = None,
        sandbox_runner: SandboxSkillRunner | None = None,
        verifier: OutputVerifier | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.skill_registry = skill_registry or SkillRegistry()
        self.spec_builder = spec_builder or SpecBuilder(self.skill_registry)
        self.spec_planner = spec_planner or LlmSpecPlanner()
        self.skill_runner = skill_runner or SkillRunner(artifact_store)
        self.sandbox_runner = sandbox_runner or SandboxSkillRunner(artifact_store)
        self.verifier = verifier or OutputVerifier(artifact_store)

    def run(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        attachments: list[Attachment],
        thread_id: str | None,
        runtime_options: RuntimeOptions | None = None,
    ) -> AgentRunResult:
        result, _events = self.run_with_events(agent_config, messages, attachments, thread_id, runtime_options=runtime_options)
        return result

    def run_with_events(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        attachments: list[Attachment],
        thread_id: str | None,
        on_event: Callable[[ChatEvent], None] | None = None,
        workflow_name: str | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        started_at = time.perf_counter()
        paths = self.artifact_store.prepare_thread(thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id, on_emit=on_event)
        workflow_name = workflow_name or agent_config.workflows.get("default", "artifact_workflow")
        recorder.emit(
            "run.started",
            {
                "workflow": workflow_name,
                "stateless": agent_config.runtime.stateless,
                "max_retries": agent_config.runtime.max_retries,
            },
        )

        recorder.emit(
            "spec.started",
            {
                "allowed_skills": self._allowed_skills(agent_config),
                "attachment_count": len(attachments),
                "selected_skill": self._selected_skill(agent_config, runtime_options),
            },
        )
        base_spec = self.spec_builder.build(
            messages,
            attachments,
            self._allowed_skills(agent_config),
            selected_skill=self._selected_skill(agent_config, runtime_options),
        )
        skill_name = str(base_spec["skill_name"])
        spec = self._plan_spec(recorder, agent_config, messages, base_spec, runtime_options)
        skill_name = str(spec["skill_name"])
        recorder.emit(
            "spec.completed",
            {
                "skill_name": skill_name,
                "spec": spec,
            },
        )

        selected_skill_names = self._selected_skill_names(agent_config, runtime_options)
        try:
            skill = self.skill_registry.get(skill_name)
            skill_payload = skill.to_event_payload()
        except KeyError:
            skill_payload = self.skill_runner.plugin_manager.get_loaded_skill(skill_name).definition.to_event_payload()
        recorder.emit(
            "skill.selected",
            {
                "skill": skill_payload,
                "selected_skills": selected_skill_names,
            },
        )

        retry_count = 0
        if len(selected_skill_names) > 1:
            run_result, verification = self._run_selected_skill_sequence(
                recorder,
                agent_config,
                messages,
                attachments,
                paths,
                runtime_options,
                selected_skill_names,
                spec,
                retry_count,
            )
        else:
            sandbox_config = load_sandbox_config()
            sandbox_policy = load_sandbox_policy(sandbox_config)
            sandbox_decision = sandbox_policy.resolve(skill_name, sandbox_config)
            recorder.emit("sandbox.policy", sandbox_decision.model_dump())
            spec["sandbox"] = sandbox_decision.model_dump()
            run_result = self._run_skill_attempt(recorder, skill_name, spec, paths, retry_count, sandbox_decision)
            verification = self._verify_attempt(recorder, spec, run_result, retry_count)

        if (
            len(selected_skill_names) <= 1
            and
            not verification.passed
            and agent_config.quality.auto_repair
            and agent_config.runtime.max_retries > 0
        ):
            retry_count = 1
            spec["repair_notes"] = verification.failed_checks
            recorder.emit(
                "retry.started",
                {
                    "attempt": retry_count,
                    "failed_checks": verification.failed_checks,
                },
            )
            run_result = self._run_skill_attempt(recorder, skill_name, spec, paths, retry_count, sandbox_decision)
            verification = self._verify_attempt(recorder, spec, run_result, retry_count)

        reply = self._build_reply(skill_name, verification, run_result)
        recorder.emit("agent.message", {"text": reply})
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=paths.thread_id,
            status="completed" if verification.passed else "failed",
            reply=reply,
            artifacts=run_result.outputs,
            verification=verification,
            spec=spec,
            metadata={"skill_name": skill_name, "workflow": workflow_name},
        )
        recorder.emit("run.completed" if verification.passed else "run.failed", {"result": result.model_dump()})
        LOGGER.info(
            "artifact_workflow_completed agent=%s thread=%s workflow=%s skill=%s planner=%s status=%s outputs=%s duration_ms=%.1f",
            agent_config.name,
            paths.thread_id,
            workflow_name,
            skill_name,
            spec.get("planner", {}),
            result.status,
            len(run_result.outputs),
            (time.perf_counter() - started_at) * 1000,
        )
        return result, recorder.events

    def _plan_spec(
        self,
        recorder: EventRecorder,
        agent_config: AgentConfig,
        messages: list[Message],
        base_spec: dict[str, Any],
        runtime_options: RuntimeOptions | None,
    ) -> dict[str, Any]:
        skill_name = str(base_spec["skill_name"])
        recorder.emit("spec.planner.started", {"skill_name": skill_name, "mode": "llm"})
        try:
            spec = self.spec_planner.plan(agent_config, messages, base_spec, runtime_options=runtime_options)
        except Exception as exc:
            spec = dict(base_spec)
            spec["planner"] = {"mode": "local", "enabled": True, "fallback": True, "error": str(exc)}
            recorder.emit("spec.planner.failed", {"skill_name": skill_name, "error": str(exc), "fallback": "local"})
            return spec

        raw_planner = spec.get("planner")
        planner: dict[str, Any] = raw_planner if isinstance(raw_planner, dict) else {}
        recorder.emit(
            "spec.planner.completed",
            {
                "skill_name": skill_name,
                "mode": planner.get("mode"),
                "enabled": planner.get("enabled"),
                "model": planner.get("model"),
                "reason": planner.get("reason"),
                "fallback": planner.get("fallback", False),
            },
        )
        return spec

    def _run_skill_attempt(
        self,
        recorder: EventRecorder,
        skill_name: str,
        spec: dict[str, Any],
        paths: ThreadPaths,
        attempt: int,
        sandbox_decision: SandboxDecision,
    ) -> SkillRunResult:
        recorder.emit(
            "skill.started",
            {
                "skill_name": skill_name,
                "attempt": attempt,
                "execution_mode": sandbox_decision.execution_mode,
                "sandbox_profile": sandbox_decision.profile_name,
            },
        )
        if sandbox_decision.use_sandbox:
            run_result = self._run_sandbox_skill_attempt(recorder, skill_name, spec, paths, attempt, sandbox_decision)
        else:
            run_result = self.skill_runner.run(skill_name, spec, paths)
        recorder.emit(
            "skill.completed",
            {
                "skill_name": skill_name,
                "attempt": attempt,
                "execution_mode": run_result.data.get("execution_mode", sandbox_decision.execution_mode),
                "output_count": len(run_result.outputs),
                "data": run_result.data,
            },
        )
        for artifact in run_result.outputs:
            artifact_data = artifact.model_dump()
            recorder.emit("artifact.created", {"artifact": artifact_data, "attempt": attempt})
            recorder.emit("preview.ready", {"artifact": artifact_data, "attempt": attempt})
        return run_result

    def _run_selected_skill_sequence(
        self,
        recorder: EventRecorder,
        agent_config: AgentConfig,
        messages: list[Message],
        attachments: list[Attachment],
        paths: ThreadPaths,
        runtime_options: RuntimeOptions | None,
        selected_skill_names: list[str],
        initial_spec: dict[str, Any],
        retry_count: int,
    ) -> tuple[SkillRunResult, VerificationResult]:
        outputs = []
        data: dict[str, Any] = {
            "sequence": [],
            "selected_skill_count": len(selected_skill_names),
        }
        checks = []
        failed_checks: list[str] = []
        sandbox_config = load_sandbox_config()
        sandbox_policy = load_sandbox_policy(sandbox_config)
        for index, selected_skill_name in enumerate(selected_skill_names, start=1):
            selected_spec = self.spec_builder.build(
                messages,
                attachments,
                self._allowed_skills(agent_config),
                selected_skill=selected_skill_name,
            )
            selected_spec.update(
                {
                    "sequence": {
                        "index": index,
                        "total": len(selected_skill_names),
                        "selected_skills": selected_skill_names,
                    },
                    "planner": initial_spec.get("planner", selected_spec.get("planner", {})),
                }
            )
            if runtime_options is not None:
                selected_spec["runtime_options"] = runtime_options.model_dump()
            sandbox_decision = sandbox_policy.resolve(selected_skill_name, sandbox_config)
            recorder.emit(
                "sandbox.policy",
                {
                    **sandbox_decision.model_dump(),
                    "skill_name": selected_skill_name,
                    "sequence_index": index,
                },
            )
            selected_spec["sandbox"] = sandbox_decision.model_dump()
            run_result = self._run_skill_attempt(
                recorder,
                selected_skill_name,
                selected_spec,
                paths,
                retry_count,
                sandbox_decision,
            )
            verification = self._verify_attempt(recorder, selected_spec, run_result, retry_count)
            outputs.extend(run_result.outputs)
            data["sequence"].append(
                {
                    "skill_name": selected_skill_name,
                    "status": "completed" if verification.passed else "failed",
                    "output_count": len(run_result.outputs),
                    "artifact_names": [artifact.name for artifact in run_result.outputs],
                    "failed_checks": verification.failed_checks,
                    "data": run_result.data,
                }
            )
            checks.extend(verification.checks)
            failed_checks.extend(f"{selected_skill_name}:{name}" for name in verification.failed_checks)

        sequence_result = SkillRunResult(
            skill_name=selected_skill_names[0],
            outputs=outputs,
            data=data,
        )
        sequence_verification = VerificationResult(
            passed=not failed_checks,
            retry_count=retry_count,
            checks=checks,
            failed_checks=failed_checks,
        )
        return sequence_result, sequence_verification

    def _run_sandbox_skill_attempt(
        self,
        recorder: EventRecorder,
        skill_name: str,
        spec: dict[str, Any],
        paths: ThreadPaths,
        attempt: int,
        sandbox_decision: SandboxDecision,
    ) -> SkillRunResult:
        sandbox_config = load_sandbox_config()
        recorder.emit(
            "sandbox.started",
            {
                "skill_name": skill_name,
                "attempt": attempt,
                "profile_name": sandbox_decision.profile_name,
                "image": sandbox_decision.profile.image if sandbox_decision.profile else None,
            },
        )
        try:
            result = self.sandbox_runner.run(
                SandboxRunContext(
                    skill_name=skill_name,
                    spec=spec,
                    paths=paths,
                    decision=sandbox_decision,
                    config=sandbox_config,
                )
            )
            recorder.emit(
                "sandbox.completed",
                {
                    "skill_name": skill_name,
                    "attempt": attempt,
                    "profile_name": sandbox_decision.profile_name,
                    "output_count": len(result.outputs),
                },
            )
            return result
        except SandboxExecutionError as exc:
            recorder.emit(
                "sandbox.failed",
                {
                    "skill_name": skill_name,
                    "attempt": attempt,
                    "profile_name": sandbox_decision.profile_name,
                    "fallback_to_local": sandbox_decision.fallback_to_local,
                    "error": str(exc),
                    "details": exc.data,
                },
            )
            if not sandbox_decision.fallback_to_local:
                raise
            recorder.emit(
                "sandbox.fallback",
                {
                    "skill_name": skill_name,
                    "attempt": attempt,
                    "profile_name": sandbox_decision.profile_name,
                    "reason": str(exc),
                },
            )
            result = self.skill_runner.run(skill_name, spec, paths)
            result.data["execution_mode"] = "local"
            result.data["sandbox_fallback"] = True
            result.data["sandbox_error"] = str(exc)
            return result

    def _verify_attempt(
        self,
        recorder: EventRecorder,
        spec: dict[str, Any],
        run_result: SkillRunResult,
        retry_count: int,
    ) -> VerificationResult:
        recorder.emit("verifier.started", {"retry_count": retry_count})
        verification = self.verifier.verify(spec, run_result, retry_count=retry_count)
        recorder.emit("verifier.completed", {"verification": verification.model_dump()})
        return verification

    def _build_reply(self, skill_name: str, verification: VerificationResult, run_result: SkillRunResult) -> str:
        plugin_reply = self.skill_runner.plugin_manager.format_reply(skill_name, verification, run_result)
        if plugin_reply:
            return plugin_reply
        error_text = self._run_error_text(run_result)
        artifact_lines = [
            f"- {artifact.name}（{artifact.kind}，{artifact.mime_type}，{artifact.path}）"
            for artifact in run_result.outputs
        ]
        artifact_text = "\n".join(artifact_lines) if artifact_lines else "- 无文件产物"
        if verification.passed:
            return f"已生成以下文件并通过内容校验。重试次数：{verification.retry_count}。\n{artifact_text}\n可在右侧「文件」面板预览或下载。"
        if error_text:
            return f"执行失败，未生成可用结果。重试次数：{verification.retry_count}。\n\n原因：{error_text}\n\n{artifact_text}"
        return f"已生成以下文件，但内容校验未完全通过。重试次数：{verification.retry_count}。\n{artifact_text}\n可在右侧「校验」面板查看失败项。"

    @staticmethod
    def _run_error_text(run_result: SkillRunResult) -> str:
        returncode = run_result.data.get("returncode")
        if returncode in (None, 0):
            return ""
        error_type = str(run_result.data.get("error_type") or "").strip()
        message = str(run_result.data.get("message") or run_result.data.get("stderr") or "").strip()
        if not message:
            return f"工具进程退出码 {returncode}"
        first_line = message.splitlines()[0].strip()
        if error_type:
            return f"{error_type}: {first_line}"
        return first_line

    def _allowed_skills(self, agent_config: AgentConfig) -> list[str]:
        if agent_config.skills:
            return agent_config.skills
        return [skill.name for skill in self.skill_registry.list(executable_only=True)]

    def _selected_skill(self, agent_config: AgentConfig, runtime_options: RuntimeOptions | None) -> str | None:
        if runtime_options is None:
            return None
        allowed_skills = self._allowed_skills(agent_config)
        for raw_name in runtime_options.selected_skills:
            skill_name = raw_name.strip()
            if not skill_name:
                continue
            if allowed_skills and skill_name not in allowed_skills:
                continue
            try:
                skill = self.skill_registry.get(skill_name)
            except KeyError:
                continue
            if skill.executable:
                return skill.name
        return None

    def _selected_skill_names(self, agent_config: AgentConfig, runtime_options: RuntimeOptions | None) -> list[str]:
        if runtime_options is None:
            return []
        allowed_skills = self._allowed_skills(agent_config)
        names: list[str] = []
        for raw_name in runtime_options.selected_skills:
            skill_name = raw_name.strip()
            if not skill_name or skill_name in names:
                continue
            if allowed_skills and skill_name not in allowed_skills:
                continue
            try:
                skill = self.skill_registry.get(skill_name)
            except KeyError:
                continue
            if skill.executable:
                names.append(skill.name)
        return names
