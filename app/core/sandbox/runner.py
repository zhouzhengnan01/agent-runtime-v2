from __future__ import annotations

import json
import posixpath
import shlex
import stat
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.sandbox.config import SandboxConfig
from app.core.sandbox.env_cache import SkillEnvironmentCache
from app.core.sandbox.policy import SandboxDecision, SandboxProfile
from app.core.skills import SkillRunResult
from app.core.skills.plugins import SkillPluginManager
from app.schemas import ArtifactRef


SANDBOX_WORKSPACE_DIR = "/mnt/user-data/workspace"
SANDBOX_OUTPUTS_DIR = "/mnt/user-data/outputs"
SANDBOX_REQUEST_PATH = f"{SANDBOX_WORKSPACE_DIR}/request.json"


class SandboxExecutionError(RuntimeError):
    def __init__(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.data = data or {}


@dataclass(frozen=True)
class SandboxRunContext:
    skill_name: str
    spec: dict[str, Any]
    paths: ThreadPaths
    decision: SandboxDecision
    config: SandboxConfig


class SandboxSkillRunner:
    """Run eligible skills in an OpenSandbox profile image.

    The runtime sends a stable request JSON to the sandbox and expects the
    profile image to provide the configured adapter command. Generated files
    are copied back from /mnt/user-data/outputs into the local ArtifactStore.
    """

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.environment_cache = SkillEnvironmentCache()
        self.skill_plugin_manager = SkillPluginManager()

    def run(self, context: SandboxRunContext) -> SkillRunResult:
        if context.decision.profile is None:
            raise SandboxExecutionError("Sandbox decision has no profile.", data=context.decision.model_dump())
        profile = self._profile_with_dependency_image(context)
        try:
            return self._run_with_opensandbox(context, profile)
        except SandboxExecutionError:
            raise
        except Exception as exc:
            raise SandboxExecutionError(str(exc), data={"error_type": type(exc).__name__}) from exc

    def _run_with_opensandbox(self, context: SandboxRunContext, profile: SandboxProfile) -> SkillRunResult:
        from opensandbox.config.connection_sync import ConnectionConfigSync
        from opensandbox.models.execd import RunCommandOpts
        from opensandbox.models.filesystem import SearchEntry
        from opensandbox.sync.sandbox import SandboxSync

        connection = ConnectionConfigSync(
            api_key=context.config.opensandbox_api_key,
            domain=_connection_domain(context.config.opensandbox_domain),
            protocol=context.config.opensandbox_protocol,
            request_timeout=timedelta(seconds=context.config.status_timeout_seconds),
        )
        sandbox = None
        try:
            sandbox = SandboxSync.create(
                image=profile.image,
                timeout=timedelta(seconds=context.config.opensandbox_timeout_seconds),
                ready_timeout=timedelta(seconds=min(60, max(1, profile.timeout_seconds))),
                env=profile.env,
                metadata={"skill_name": context.skill_name, "thread_id": context.paths.thread_id},
                connection_config=connection,
            )
            mkdir = sandbox.commands.run(
                f"mkdir -p {shlex.quote(SANDBOX_WORKSPACE_DIR)} {shlex.quote(SANDBOX_OUTPUTS_DIR)}",
                opts=RunCommandOpts(timeout=timedelta(seconds=30)),
            )
            _raise_if_execution_failed("Create sandbox directories", mkdir)
            sandbox.files.write_file(SANDBOX_REQUEST_PATH, _request_json(context, profile), mode=0o644)

            if profile.runtime_install and profile.python_requirements:
                install_command = "python -m pip install " + " ".join(
                    shlex.quote(requirement) for requirement in profile.python_requirements
                )
                install = sandbox.commands.run(
                    install_command,
                    opts=RunCommandOpts(
                        working_directory=SANDBOX_WORKSPACE_DIR,
                        timeout=timedelta(seconds=profile.timeout_seconds),
                        envs=profile.env or None,
                    ),
                )
                _raise_if_execution_failed("Install sandbox profile requirements", install)

            command = _render_profile_command(profile)
            execution = sandbox.commands.run(
                command,
                opts=RunCommandOpts(
                    working_directory=SANDBOX_WORKSPACE_DIR,
                    timeout=timedelta(seconds=profile.timeout_seconds),
                    envs=profile.env or None,
                ),
            )
            _raise_if_execution_failed("Run sandbox skill adapter", execution)

            entries = sandbox.files.search(SearchEntry(path=SANDBOX_OUTPUTS_DIR, pattern="*"))
            outputs: list[ArtifactRef] = []
            for entry in entries:
                if not stat.S_ISREG(entry.mode):
                    continue
                rel = _relative_output_path(entry.path)
                if rel is None:
                    continue
                content = sandbox.files.read_bytes(entry.path)
                target = self._write_output_bytes(context.paths, rel, content)
                outputs.append(self.artifact_store.to_artifact_ref(context.paths.thread_id, target))
            if not outputs:
                raise SandboxExecutionError(
                    "Sandbox skill adapter completed without output files.",
                    data={"command": command, "stdout": _stdout_text(execution), "stderr": _stderr_text(execution)},
                )
            return SkillRunResult(
                context.skill_name,
                outputs,
                {
                    "execution_mode": "sandbox",
                    "sandbox_profile": context.decision.profile_name,
                    "sandbox_image": profile.image,
                    "sandbox_command": command,
                    "stdout": _stdout_text(execution),
                    "stderr": _stderr_text(execution),
                },
            )
        finally:
            if sandbox is not None:
                try:
                    sandbox.kill()
                except Exception:
                    pass
                try:
                    sandbox.close()
                except Exception:
                    pass

    def _profile_with_dependency_image(self, context: SandboxRunContext) -> SandboxProfile:
        profile = context.decision.profile
        if profile is None or not context.config.skill_env_cache_enabled:
            return profile
        try:
            loaded = self.skill_plugin_manager.get_loaded_skill(context.skill_name)
            package_root = loaded.manifest_path.parent if loaded.manifest_path is not None else None
            environment = self.environment_cache.prepare(
                skill_name=context.skill_name,
                package_root=package_root,
                profile=profile,
                config=context.config,
            )
        except Exception as exc:
            if context.decision.fallback_to_local:
                raise SandboxExecutionError(
                    "Prepare sandbox dependency image failed.",
                    data={"error": str(exc), "fallback_to_local": True},
                ) from exc
            raise
        if environment.status != "ready":
            if environment.status == "base":
                return profile
            raise SandboxExecutionError(
                "Sandbox dependency image is not ready.",
                data={
                    "requirements_hash": environment.requirements_hash,
                    "status": environment.status,
                    "message": environment.message,
                },
            )
        return profile.model_copy(update={"image": environment.image, "runtime_install": False})

    def _write_output_bytes(self, paths: ThreadPaths, relative_path: str, content: bytes) -> Path:
        target = (paths.outputs / relative_path).resolve()
        outputs = paths.outputs.resolve()
        try:
            target.relative_to(outputs)
        except ValueError as exc:
            raise SandboxExecutionError(f"Sandbox output path traversal blocked: {relative_path}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target


def _request_json(context: SandboxRunContext, profile: SandboxProfile) -> str:
    payload = {
        "request_schema_version": context.decision.request_schema_version,
        "skill_name": context.skill_name,
        "thread_id": context.paths.thread_id,
        "spec": context.spec,
        "profile": profile.model_dump(),
        "workspace_dir": SANDBOX_WORKSPACE_DIR,
        "outputs_dir": SANDBOX_OUTPUTS_DIR,
    }
    return json.dumps(payload, ensure_ascii=False)


def _render_profile_command(profile: SandboxProfile) -> str:
    return profile.command.format(
        request_path=shlex.quote(SANDBOX_REQUEST_PATH),
        workspace_dir=shlex.quote(SANDBOX_WORKSPACE_DIR),
        outputs_dir=shlex.quote(SANDBOX_OUTPUTS_DIR),
    )


def _connection_domain(domain: str) -> str:
    return domain.removeprefix("http://").removeprefix("https://").rstrip("/")


def _relative_output_path(path: str) -> str | None:
    normalized = posixpath.normpath(path)
    outputs = posixpath.normpath(SANDBOX_OUTPUTS_DIR)
    if normalized == outputs or not normalized.startswith(outputs + "/"):
        return None
    relative = posixpath.relpath(normalized, outputs)
    if relative.startswith("../") or relative == "..":
        return None
    return relative


def _raise_if_execution_failed(label: str, execution: Any) -> None:
    error = getattr(execution, "error", None)
    exit_code = getattr(execution, "exit_code", None)
    if error is not None or (exit_code is not None and exit_code != 0):
        raise SandboxExecutionError(
            f"{label} failed.",
            data={
                "exit_code": exit_code,
                "error": str(error) if error is not None else None,
                "stdout": _stdout_text(execution),
                "stderr": _stderr_text(execution),
            },
        )


def _stdout_text(execution: Any) -> str:
    logs = getattr(execution, "logs", None)
    messages = getattr(logs, "stdout", []) if logs is not None else []
    return "\n".join(str(getattr(message, "text", message)) for message in messages)


def _stderr_text(execution: Any) -> str:
    logs = getattr(execution, "logs", None)
    messages = getattr(logs, "stderr", []) if logs is not None else []
    return "\n".join(str(getattr(message, "text", message)) for message in messages)
