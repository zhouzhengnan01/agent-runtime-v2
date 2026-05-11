from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from string import Template
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.skills.local_subprocess import LocalSubprocessEnvironmentCache
from app.core.skills.runner_types import SkillRunResult
from app.schemas import ArtifactRef


def can_run_generic(manifest: dict[str, Any]) -> bool:
    execution = manifest.get("execution")
    return isinstance(execution, dict) and _string(execution.get("type")) in {
        "python_script",
        "template",
        "http",
        "command",
    }


def run_generic_skill(
    skill_name: str,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
    *,
    package_root: Path | None = None,
) -> SkillRunResult:
    execution = manifest.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"Skill {skill_name} has no generic execution config.")
    execution_type = _string(execution.get("type"))
    if execution_type == "python_script":
        return _run_python_script(skill_name, execution, spec, paths, artifact_store, package_root=package_root)
    if execution_type == "template":
        return _run_template(skill_name, execution, spec, paths, artifact_store)
    if execution_type == "http":
        return _run_http(skill_name, execution, spec, paths, artifact_store)
    if execution_type == "command":
        return _run_command(skill_name, execution, spec, paths, artifact_store, package_root=package_root)
    raise ValueError(f"Unsupported generic execution type for {skill_name}: {execution_type}")


def _run_python_script(
    skill_name: str,
    execution: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
    *,
    package_root: Path | None,
) -> SkillRunResult:
    script = _string(execution.get("script"))
    if not script:
        raise ValueError("Python script execution requires execution.script.")
    script_path = _safe_package_path(script, package_root)
    if not script_path.is_file():
        raise ValueError(f"Python script not found: {script}")
    python = _python_executable(skill_name, execution, package_root)
    timeout = _bounded_int(execution.get("timeout_seconds"), default=120, minimum=1, maximum=1800)
    input_mode = _string(execution.get("input_mode")) or "stdin_json"
    args = _python_script_args(execution, spec, paths)
    stdin = _python_script_stdin(input_mode, skill_name, spec, paths)
    env = os.environ.copy()
    env.update(_render_mapping(execution.get("env"), spec))
    completed = subprocess.run(
        [python, str(script_path), *args],
        cwd=str(package_root) if package_root is not None else str(script_path.parent),
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    outputs = _python_script_outputs(execution, completed, spec, paths, artifact_store)
    data = _python_script_data(completed.stdout)
    data.update({"execution_type": "python_script", "execution_runtime": _execution_runtime(execution), "returncode": completed.returncode})
    if completed.returncode != 0:
        data["stderr"] = completed.stderr
    return SkillRunResult(skill_name=skill_name, outputs=outputs, data=data)


def _run_template(
    skill_name: str,
    execution: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> SkillRunResult:
    filename = _safe_filename(_render_value(_string(execution.get("filename")) or f"{skill_name}.txt", spec))
    template = _string(execution.get("template"))
    if not template:
        template = json.dumps(_public_spec(spec), ensure_ascii=False, indent=2)
    content = _render_value(template, spec)
    artifact = artifact_store.write_text_artifact(paths, filename, content)
    return SkillRunResult(skill_name=skill_name, outputs=[artifact], data={"execution_type": "template"})


def _run_http(
    skill_name: str,
    execution: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> SkillRunResult:
    url = _render_value(_string(execution.get("url")), spec)
    if not url:
        raise ValueError("HTTP execution requires execution.url.")
    method = (_string(execution.get("method")) or "POST").upper()
    headers = _render_mapping(execution.get("headers"), spec)
    timeout = _bounded_int(execution.get("timeout_seconds"), default=30, minimum=1, maximum=300)
    body_template = _string(execution.get("body_template"))
    body_value = _render_value(body_template, spec) if body_template else json.dumps(_public_spec(spec), ensure_ascii=False)
    data = body_value.encode("utf-8") if method not in {"GET", "HEAD"} else None
    if data is not None and not any(key.lower() == "content-type" for key in headers):
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        status = exc.code
        content_type = exc.headers.get("Content-Type", "")

    filename = _safe_filename(_render_value(_string(execution.get("filename")) or f"{skill_name}-response.json", spec))
    artifact = artifact_store.write_bytes_artifact(paths, filename, response_body)
    return SkillRunResult(
        skill_name=skill_name,
        outputs=[artifact],
        data={"execution_type": "http", "status": status, "content_type": content_type},
    )


def _run_command(
    skill_name: str,
    execution: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
    *,
    package_root: Path | None,
) -> SkillRunResult:
    command = _string(execution.get("command"))
    if not command:
        raise ValueError("Command execution requires execution.command.")
    args = [_render_value(str(item), spec) for item in _string_list(execution.get("args"))]
    cwd = _safe_cwd(_render_value(_string(execution.get("cwd")), spec), package_root)
    timeout = _bounded_int(execution.get("timeout_seconds"), default=60, minimum=1, maximum=600)
    stdin_template = _string(execution.get("stdin_template"))
    stdin = _render_value(stdin_template, spec) if stdin_template else None
    env = os.environ.copy()
    env.update(_render_mapping(execution.get("env"), spec))
    completed = subprocess.run(
        [command, *args],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout_name = _safe_filename(_render_value(_string(execution.get("stdout_filename")) or f"{skill_name}-stdout.txt", spec))
    stderr_name = _safe_filename(_render_value(_string(execution.get("stderr_filename")) or f"{skill_name}-stderr.txt", spec))
    outputs: list[ArtifactRef] = [artifact_store.write_text_artifact(paths, stdout_name, completed.stdout)]
    if completed.stderr:
        outputs.append(artifact_store.write_text_artifact(paths, stderr_name, completed.stderr))
    outputs.extend(_configured_output_artifacts(execution, spec, paths, artifact_store))
    return SkillRunResult(
        skill_name=skill_name,
        outputs=outputs,
        data={"execution_type": "command", "returncode": completed.returncode},
    )


def _configured_output_artifacts(
    execution: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> list[ArtifactRef]:
    raw_outputs = execution.get("outputs")
    if not isinstance(raw_outputs, list):
        return []
    artifacts: list[ArtifactRef] = []
    for item in raw_outputs:
        if not isinstance(item, dict):
            continue
        path_value = _render_value(_string(item.get("path")), spec)
        if not path_value:
            continue
        candidate = (paths.outputs / path_value).resolve()
        try:
            candidate.relative_to(paths.outputs.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            artifact_store.upsert_artifact(paths, candidate)
            artifacts.append(artifact_store.to_artifact_ref(paths.thread_id, candidate))
    return artifacts


def _python_script_args(execution: dict[str, Any], spec: dict[str, Any], paths: ThreadPaths) -> list[str]:
    return [
        _expand_runtime_token(_render_value(str(item), spec), paths)
        for item in _string_list(execution.get("args"))
    ]


def _python_script_stdin(input_mode: str, skill_name: str, spec: dict[str, Any], paths: ThreadPaths) -> str | None:
    if input_mode == "stdin_json":
        return json.dumps(
            {
                "skill_name": skill_name,
                "spec": _expand_runtime_tokens(_public_spec(spec), paths),
                "thread_id": paths.thread_id,
                "workspace_dir": str(paths.workspace.resolve()),
                "uploads_dir": str(paths.uploads.resolve()),
                "outputs_dir": str(paths.outputs.resolve()),
            },
            ensure_ascii=False,
        )
    if input_mode in {"args", "none", "json_file"}:
        if input_mode == "json_file":
            payload_path = paths.workspace / f"{skill_name}-input.json"
            payload_path.write_text(
                json.dumps(_expand_runtime_tokens(_public_spec(spec), paths), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return None
    raise ValueError(f"Unsupported python_script input_mode: {input_mode}")


def _python_script_outputs(
    execution: dict[str, Any],
    completed: subprocess.CompletedProcess[str],
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> list[ArtifactRef]:
    outputs: list[ArtifactRef] = []
    outputs.extend(_declared_result_outputs(completed.stdout, paths, artifact_store))
    outputs.extend(_configured_output_artifacts(execution, spec, paths, artifact_store))
    if execution.get("collect_outputs", True):
        existing = {artifact.path for artifact in outputs}
        for file_path in sorted(paths.outputs.rglob("*")):
            if file_path.is_file():
                artifact_store.upsert_artifact(paths, file_path)
                artifact = artifact_store.to_artifact_ref(paths.thread_id, file_path)
                if artifact.path not in existing:
                    outputs.append(artifact)
                    existing.add(artifact.path)
    stdout_filename = _string(execution.get("stdout_filename"))
    if stdout_filename and completed.stdout:
        outputs.append(artifact_store.write_text_artifact(paths, stdout_filename, completed.stdout))
    stderr_filename = _string(execution.get("stderr_filename"))
    if stderr_filename and completed.stderr:
        outputs.append(artifact_store.write_text_artifact(paths, stderr_filename, completed.stderr))
    return outputs


def _declared_result_outputs(stdout: str, paths: ThreadPaths, artifact_store: ArtifactStore) -> list[ArtifactRef]:
    data = _json_object(stdout)
    raw_outputs = data.get("outputs") or data.get("artifacts")
    if not isinstance(raw_outputs, list):
        return []
    artifacts: list[ArtifactRef] = []
    for item in raw_outputs:
        if not isinstance(item, dict):
            continue
        path_value = _string(item.get("path"))
        if not path_value:
            continue
        candidate = (paths.outputs / path_value).resolve()
        try:
            candidate.relative_to(paths.outputs.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            artifact_store.upsert_artifact(paths, candidate)
            artifacts.append(artifact_store.to_artifact_ref(paths.thread_id, candidate))
    return artifacts


def _python_script_data(stdout: str) -> dict[str, Any]:
    data = _json_object(stdout)
    return data if data else {"stdout": stdout}


def _json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _python_executable(skill_name: str, execution: dict[str, Any], package_root: Path | None) -> str:
    runtime = _execution_runtime(execution)
    configured = _string(execution.get("python"))
    if runtime != "local_subprocess":
        return configured or "python"
    environment = LocalSubprocessEnvironmentCache().prepare(
        skill_name=skill_name,
        package_root=package_root,
        base_python=configured or None,
        install_timeout_seconds=_bounded_int(execution.get("install_timeout_seconds"), default=1800, minimum=1, maximum=7200),
    )
    if environment.status == "failed":
        raise RuntimeError(f"Prepare local_subprocess environment failed: {environment.message}")
    return str(environment.python)


def _execution_runtime(execution: dict[str, Any]) -> str:
    return _string(execution.get("runtime")) or _string(execution.get("mode")) or "in_process"


def _render_value(template: str, spec: dict[str, Any]) -> str:
    values = {key: _template_value(value) for key, value in _public_spec(spec).items()}
    values.setdefault("spec_json", json.dumps(_public_spec(spec), ensure_ascii=False))
    return Template(template).safe_substitute(values)


def _render_mapping(value: object, spec: dict[str, Any]) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _render_value(str(item), spec) for key, item in value.items()}


def _expand_runtime_token(value: str, paths: ThreadPaths) -> str:
    return (
        value.replace("{thread_id}", paths.thread_id)
        .replace("{workspace}", str(paths.workspace.resolve()))
        .replace("{uploads}", str(paths.uploads.resolve()))
        .replace("{outputs}", str(paths.outputs.resolve()))
        .replace("/mnt/user-data/uploads/", str(paths.uploads.resolve()) + "/")
        .replace("/mnt/user-data/outputs/", str(paths.outputs.resolve()) + "/")
    )


def _expand_runtime_tokens(value: Any, paths: ThreadPaths) -> Any:
    if isinstance(value, str):
        return _expand_runtime_token(value, paths)
    if isinstance(value, list):
        return [_expand_runtime_tokens(item, paths) for item in value]
    if isinstance(value, dict):
        return {key: _expand_runtime_tokens(item, paths) for key, item in value.items()}
    return value


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if not key.startswith("_")}


def _template_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _safe_filename(value: str) -> str:
    name = value.strip().replace("\\", "/").lstrip("/")
    name = re.sub(r"/+", "/", name)
    if not name or name.endswith("/"):
        return "result.txt"
    return name


def _safe_cwd(value: str, package_root: Path | None) -> Path | None:
    if not value:
        return package_root
    root = package_root or Path.cwd()
    candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Command cwd must stay inside the skill package root.") from exc
    return candidate


def _safe_package_path(value: str, package_root: Path | None) -> Path:
    root = package_root or Path.cwd()
    candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Python script path must stay inside the skill package root.") from exc
    return candidate


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return shlex.split(value)
    return []


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if isinstance(value, (str, int, float)) else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))
