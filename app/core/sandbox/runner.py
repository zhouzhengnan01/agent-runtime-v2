from __future__ import annotations

import json
import os
import posixpath
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass, replace
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
SANDBOX_INPUTS_DIR = f"{SANDBOX_WORKSPACE_DIR}/inputs"
SANDBOX_SKILLS_DIR = "/mnt/user-data/skills"
LOCAL_SUBPROCESS_ROOT = "sandbox-run"
LOCAL_SUBPROCESS_WORKSPACE = "workspace"
LOCAL_SUBPROCESS_OUTPUTS = "outputs"
LOCAL_SUBPROCESS_INPUTS = "inputs"
LOCAL_SUBPROCESS_SKILLS = "skills"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
            if context.config.provider == "local_subprocess":
                return self._run_with_local_subprocess(context, profile)
            return self._run_with_opensandbox(context, profile)
        except SandboxExecutionError:
            raise
        except Exception as exc:
            raise SandboxExecutionError(str(exc), data={"error_type": type(exc).__name__}) from exc

    def _run_with_local_subprocess(self, context: SandboxRunContext, profile: SandboxProfile) -> SkillRunResult:
        root = context.paths.root / LOCAL_SUBPROCESS_ROOT
        workspace = root / LOCAL_SUBPROCESS_WORKSPACE
        outputs_dir = root / LOCAL_SUBPROCESS_OUTPUTS
        inputs_dir = root / LOCAL_SUBPROCESS_INPUTS
        skills_dir = root / LOCAL_SUBPROCESS_SKILLS
        for path in (workspace, outputs_dir, inputs_dir, skills_dir):
            path.mkdir(parents=True, exist_ok=True)

        package_dir = self._copy_skill_package(skills_dir, context)
        context = self._copy_referenced_inputs(inputs_dir, context)
        request_path = workspace / "request.json"
        request_path.write_text(
            _local_request_json(context, profile, workspace=workspace, outputs_dir=outputs_dir),
            encoding="utf-8",
        )

        command = _render_local_profile_command(
            profile,
            request_path=request_path,
            workspace_dir=workspace,
            outputs_dir=outputs_dir,
            package_dir=package_dir,
        )
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            env={**dict(profile.env), **_local_subprocess_env()},
            capture_output=True,
            text=True,
            timeout=profile.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise SandboxExecutionError(
                "Run local subprocess skill adapter failed.",
                data={
                    "exit_code": completed.returncode,
                    "command": command,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )

        outputs: list[ArtifactRef] = []
        for file_path in sorted(outputs_dir.rglob("*")):
            if not file_path.is_file():
                continue
            relative_path = file_path.relative_to(outputs_dir).as_posix()
            target = self._write_output_bytes(context.paths, relative_path, file_path.read_bytes())
            outputs.append(self.artifact_store.to_artifact_ref(context.paths.thread_id, target))
        if not outputs:
            raise SandboxExecutionError(
                "Local subprocess skill adapter completed without output files.",
                data={"command": command, "stdout": completed.stdout, "stderr": completed.stderr},
            )
        return SkillRunResult(
            context.skill_name,
            outputs,
            {
                "execution_mode": "local_subprocess",
                "sandbox_profile": context.decision.profile_name,
                "sandbox_command": command,
                "sandbox_package_dir": str(package_dir) if package_dir is not None else "",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

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
                f"mkdir -p {shlex.quote(SANDBOX_WORKSPACE_DIR)} {shlex.quote(SANDBOX_OUTPUTS_DIR)} {shlex.quote(SANDBOX_INPUTS_DIR)} {shlex.quote(SANDBOX_SKILLS_DIR)}",
                opts=RunCommandOpts(timeout=timedelta(seconds=30)),
            )
            _raise_if_execution_failed("Create sandbox directories", mkdir)
            package_dir = self._upload_skill_package(sandbox, context)
            context = self._upload_referenced_inputs(sandbox, context)
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

            command = _render_profile_command(profile, package_dir=package_dir)
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
                    "sandbox_package_dir": package_dir,
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
        if profile is None:
            raise SandboxExecutionError("Sandbox decision has no profile.", data=context.decision.model_dump())
        if not context.config.skill_env_cache_enabled:
            return profile
        try:
            loaded = self.skill_plugin_manager.get_loaded_skill(context.skill_name)
            package_root = loaded.plugin.root if loaded.plugin is not None else loaded.manifest_path.parent
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

    def _copy_skill_package(self, skills_dir: Path, context: SandboxRunContext) -> Path | None:
        try:
            loaded = self.skill_plugin_manager.get_loaded_skill(context.skill_name)
        except KeyError:
            return None
        if loaded.plugin is None:
            return None
        package_root = loaded.plugin.root.resolve()
        package_dir = skills_dir / _safe_sandbox_name(context.skill_name)
        for file_path in sorted(package_root.rglob("*")):
            if not file_path.is_file() or _skip_package_file(file_path):
                continue
            relative = file_path.relative_to(package_root)
            target = package_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file_path.read_bytes())
            target.chmod(file_path.stat().st_mode & 0o777 or 0o644)
        return package_dir

    def _copy_referenced_inputs(self, inputs_dir: Path, context: SandboxRunContext) -> SandboxRunContext:
        spec = dict(context.spec)
        for key in ("image_path", "image_paths", "path", "paths", "input_path", "input_dir"):
            if key not in spec:
                continue
            spec[key] = self._copy_input_value(inputs_dir, context.paths, spec[key], key)
        return replace(context, spec=spec)

    def _copy_input_value(self, inputs_dir: Path, paths: ThreadPaths, value: object, key: str) -> object:
        if isinstance(value, list):
            return [self._copy_input_value(inputs_dir, paths, item, key) for item in value]
        if not isinstance(value, str) or not value.strip():
            return value
        source = Path(value).expanduser()
        if not source.exists():
            return value
        try:
            source.resolve().relative_to(paths.root.resolve())
        except ValueError:
            return value
        if source.is_dir():
            target_dir = inputs_dir / _safe_sandbox_name(source.name or key)
            target_dir.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.rglob("*")):
                if not child.is_file() or child.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                target = target_dir / child.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(child.read_bytes())
            return str(target_dir)
        if source.is_file():
            target = inputs_dir / _safe_sandbox_name(source.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            return str(target)
        return value

    def _upload_skill_package(self, sandbox: Any, context: SandboxRunContext) -> str:
        from opensandbox.models.execd import RunCommandOpts

        try:
            loaded = self.skill_plugin_manager.get_loaded_skill(context.skill_name)
        except KeyError:
            return ""
        if loaded.plugin is None:
            return ""
        package_root = loaded.plugin.root.resolve()
        package_dir = f"{SANDBOX_SKILLS_DIR}/{_safe_sandbox_name(context.skill_name)}"
        mkdir = sandbox.commands.run(
            f"mkdir -p {shlex.quote(package_dir)}",
            opts=RunCommandOpts(timeout=timedelta(seconds=30)),
        )
        _raise_if_execution_failed("Create sandbox skill package directory", mkdir)
        for file_path in sorted(package_root.rglob("*")):
            if not file_path.is_file() or _skip_package_file(file_path):
                continue
            relative = file_path.relative_to(package_root).as_posix()
            target = posixpath.join(package_dir, relative)
            parent = posixpath.dirname(target)
            mkdir = sandbox.commands.run(
                f"mkdir -p {shlex.quote(parent)}",
                opts=RunCommandOpts(timeout=timedelta(seconds=30)),
            )
            _raise_if_execution_failed("Create sandbox skill package subdirectory", mkdir)
            sandbox.files.write_file(target, file_path.read_bytes(), mode=file_path.stat().st_mode & 0o777 or 0o644)
        return package_dir

    def _upload_referenced_inputs(self, sandbox: Any, context: SandboxRunContext) -> SandboxRunContext:
        spec = dict(context.spec)
        for key in ("image_path", "image_paths", "path", "paths", "input_path", "input_dir"):
            if key not in spec:
                continue
            spec[key] = self._upload_input_value(sandbox, context.paths, spec[key], key)
        return replace(context, spec=spec)

    def _upload_input_value(self, sandbox: Any, paths: ThreadPaths, value: object, key: str) -> object:
        from opensandbox.models.execd import RunCommandOpts

        if isinstance(value, list):
            return [self._upload_input_value(sandbox, paths, item, key) for item in value]
        if not isinstance(value, str) or not value.strip():
            return value
        source = Path(value).expanduser()
        if not source.exists():
            return value
        try:
            source.resolve().relative_to(paths.root.resolve())
        except ValueError:
            # Only auto-transfer files already inside this thread workspace.
            return value
        if source.is_dir():
            target_dir = f"{SANDBOX_INPUTS_DIR}/{_safe_sandbox_name(source.name or key)}"
            mkdir = sandbox.commands.run(
                f"mkdir -p {shlex.quote(target_dir)}",
                opts=RunCommandOpts(timeout=timedelta(seconds=30)),
            )
            _raise_if_execution_failed("Create sandbox input directory", mkdir)
            for child in sorted(source.rglob("*")):
                if not child.is_file() or child.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                relative = child.relative_to(source).as_posix()
                target = posixpath.join(target_dir, relative)
                parent = posixpath.dirname(target)
                mkdir = sandbox.commands.run(
                    f"mkdir -p {shlex.quote(parent)}",
                    opts=RunCommandOpts(timeout=timedelta(seconds=30)),
                )
                _raise_if_execution_failed("Create sandbox input subdirectory", mkdir)
                sandbox.files.write_file(target, child.read_bytes(), mode=0o644)
            return target_dir
        if source.is_file():
            target = f"{SANDBOX_INPUTS_DIR}/{_safe_sandbox_name(source.name)}"
            sandbox.files.write_file(target, source.read_bytes(), mode=0o644)
            return target
        return value


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


def _local_request_json(
    context: SandboxRunContext,
    profile: SandboxProfile,
    *,
    workspace: Path,
    outputs_dir: Path,
) -> str:
    payload = {
        "request_schema_version": context.decision.request_schema_version,
        "skill_name": context.skill_name,
        "thread_id": context.paths.thread_id,
        "spec": context.spec,
        "profile": profile.model_dump(),
        "workspace_dir": str(workspace),
        "outputs_dir": str(outputs_dir),
    }
    return json.dumps(payload, ensure_ascii=False)


def _render_profile_command(profile: SandboxProfile, *, package_dir: str = "") -> str:
    return profile.command.format(
        request_path=shlex.quote(SANDBOX_REQUEST_PATH),
        workspace_dir=shlex.quote(SANDBOX_WORKSPACE_DIR),
        outputs_dir=shlex.quote(SANDBOX_OUTPUTS_DIR),
        package_dir=shlex.quote(package_dir),
    )


def _render_local_profile_command(
    profile: SandboxProfile,
    *,
    request_path: Path,
    workspace_dir: Path,
    outputs_dir: Path,
    package_dir: Path | None,
) -> str:
    command = profile.command
    if command.startswith("python "):
        command = f"{shlex.quote(sys.executable)} {command.removeprefix('python ')}"
    return command.format(
        request_path=shlex.quote(str(request_path)),
        workspace_dir=shlex.quote(str(workspace_dir)),
        outputs_dir=shlex.quote(str(outputs_dir)),
        package_dir=shlex.quote(str(package_dir or "")),
    )


def _local_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    python_path = str(Path(__file__).resolve().parents[3])
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{python_path}{os.pathsep}{existing}" if existing else python_path
    return env


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


def _safe_sandbox_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
    return cleaned.strip(".-") or "item"


def _skip_package_file(path: Path) -> bool:
    parts = set(path.parts)
    return "__pycache__" in parts or ".pytest_cache" in parts or path.suffix in {".pyc", ".pyo"}
