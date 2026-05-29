from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

LOG_FILENAMES = {
    "gpu-training-orchestrator-stdout.txt",
    "gpu-training-orchestrator-stderr.txt",
    "yolo-training-stdout.txt",
    "yolo-training-stderr.txt",
    "yolo-training-epochs.txt",
}
KEY_RUN_ARTIFACT_NAMES = {
    "args.yaml",
    "run_summary.json",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "labels.jpg",
    "labels_correlogram.jpg",
}
KEY_WEIGHT_NAMES = {"best.pt", "last.pt"}
MAX_BATCH_VISUALS = 8
BATCH_VISUAL_RE = re.compile(r"^(?:train_batch|val_batch|predictions).*\.(?:png|jpg|jpeg)$", re.IGNORECASE)


def run(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any) -> dict[str, Any]:
    del skill_name
    package_root = Path(__file__).resolve().parent
    script_path = package_root / "scripts" / "run_prepared_yolo_training.py"
    if not script_path.exists():
        raise FileNotFoundError(f"run_prepared_yolo_training.py not found: {script_path}")

    normalized = _normalize_training_spec(spec, paths)
    request_path = Path(paths.workspace) / "gpu-training-orchestrator-training-input.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    config_dir = Path(paths.workspace) / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings_file = config_dir / "settings.json"
    if not settings_file.exists():
        settings_file.write_text("{}", encoding="utf-8")
    env["YOLO_CONFIG_DIR"] = str(config_dir.resolve())
    env["ULTRALYTICS_SETTINGS"] = str(settings_file.resolve())
    env["ULTRALYTICS_HOME"] = str(config_dir.resolve())
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    cmd = _training_command(normalized, script_path, request_path)
    completed = _run_command_with_live_logs(cmd, package_root, env, normalized)
    stdout_text = _decode_bytes(completed.stdout)
    stderr_text = _decode_bytes(completed.stderr)
    _write_skill_logs(normalized, stdout_text, stderr_text)
    outputs = _collect_user_visible_outputs(normalized, paths, artifact_store)
    return {
        "skill_name": "gpu-training-orchestrator",
        "outputs": outputs,
        "data": {
            "execution_type": "prepared_yolo_training",
            "returncode": completed.returncode,
            "stdout": stdout_text[-4000:] if stdout_text else "",
            "stderr": stderr_text[-4000:] if stderr_text else "",
            "final_reply": _build_reply(normalized, stdout_text, stderr_text, completed.returncode),
        },
    }


def _normalize_training_spec(spec: dict[str, Any], paths: Any) -> dict[str, Any]:
    payload = dict(spec)
    output_root = Path(paths.outputs).resolve()
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}

    data_yaml = (
        payload.get("data_yaml")
        or payload.get("dataset_yaml")
        or dataset.get("data_yaml")
        or dataset.get("dataset_yaml")
        or str(output_root / "prepared_data" / "dataset.yaml")
    )
    project_dir = output.get("project_dir") or payload.get("project_dir") or str(output_root / "training_run")
    run_name = output.get("run_name") or payload.get("run_name") or "."

    return {
        "runtime": {
            "conda_env_name": runtime.get("conda_env_name") if runtime.get("conda_env_name") is not None else payload.get("conda_env_name", ""),
            "enforce_conda_env": bool(runtime.get("enforce_conda_env", False)),
        },
        "dataset": {"data_yaml": str(data_yaml)},
        "training": {
            "task": training.get("task") or payload.get("training_task") or "detect",
            "model": training.get("model") or payload.get("model") or "yolo11n.pt",
            "epochs": int(training.get("epochs") or payload.get("epochs") or 50),
            "imgsz": int(training.get("imgsz") or payload.get("imgsz") or 640),
            "batch": int(training.get("batch") or payload.get("batch") or 16),
            "device": training.get("device") or payload.get("device") or "0",
            "workers": int(training.get("workers") or payload.get("workers") or 4),
            "patience": int(training.get("patience") or payload.get("patience") or 8),
        },
        "output": {"project_dir": str(project_dir), "run_name": str(run_name)},
    }


def _training_command(spec: dict[str, Any], script_path: Path, request_path: Path) -> list[str]:
    conda_env_name = str(spec.get("runtime", {}).get("conda_env_name") or "").strip()
    if not conda_env_name:
        return [sys.executable, str(script_path), "--input", str(request_path)]
    current_env = str(os.environ.get("CONDA_DEFAULT_ENV", "") or "").strip()
    if current_env == conda_env_name:
        return [sys.executable, str(script_path), "--input", str(request_path)]
    conda_exe = str(os.environ.get("CONDA_EXE", "") or "").strip() or "conda"
    return [conda_exe, "run", "--no-capture-output", "-n", conda_env_name, "python", str(script_path), "--input", str(request_path)]


def _run_command_with_live_logs(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    spec: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    log_dir = _resolve_log_dir(spec)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_targets = [
        log_dir / "gpu-training-orchestrator-stdout.txt",
        log_dir / "yolo-training-stdout.txt",
    ]
    stderr_targets = [
        log_dir / "gpu-training-orchestrator-stderr.txt",
        log_dir / "yolo-training-stderr.txt",
    ]
    for target in [*stdout_targets, *stderr_targets]:
        target.write_text("", encoding="utf-8")

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(
        target=_pump_stream_to_logs,
        args=(process.stdout, stdout_targets, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_stream_to_logs,
        args=(process.stderr, stderr_targets, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return subprocess.CompletedProcess(cmd, returncode, "".join(stdout_chunks), "".join(stderr_chunks))


def _pump_stream_to_logs(stream: Any, targets: list[Path], chunks: list[str]) -> None:
    if stream is None:
        return
    handles = [target.open("a", encoding="utf-8", newline="") for target in targets]
    try:
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            chunks.append(chunk)
            for handle in handles:
                handle.write(chunk)
                handle.flush()
    finally:
        for handle in handles:
            handle.close()
        try:
            stream.close()
        except Exception:
            pass


def _write_skill_logs(spec: dict[str, Any], stdout_text: str, stderr_text: str) -> None:
    log_dir = _resolve_log_dir(spec)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "gpu-training-orchestrator-stdout.txt").write_text(stdout_text or "", encoding="utf-8")
    (log_dir / "gpu-training-orchestrator-stderr.txt").write_text(stderr_text or "", encoding="utf-8")
    (log_dir / "yolo-training-stdout.txt").write_text(stdout_text or "", encoding="utf-8")
    (log_dir / "yolo-training-stderr.txt").write_text(stderr_text or "", encoding="utf-8")
    _write_clean_yolo_log(stdout_text, log_dir)


def _write_clean_yolo_log(stdout_text: str, log_dir: Path) -> None:
    clean = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", stdout_text or "").replace("\r", "\n")
    selected: list[str] = []
    for line in clean.splitlines():
        stripped = " ".join(line.strip().split())
        stripped = re.sub(r":\s+\d+%.*$", "", stripped)
        if not stripped:
            continue
        lower = stripped.lower()
        if (
            "epoch" in lower and "box_loss" in lower
            or re.match(r"^\d+/\d+\s+", stripped)
            or stripped.startswith("all ")
            or stripped.startswith("Starting training")
            or "epochs completed in" in stripped
            or stripped.startswith("Results saved to")
        ):
            selected.append(stripped)
    (log_dir / "yolo-training-epochs.txt").write_text("\n".join(selected).strip() + ("\n" if selected else ""), encoding="utf-8")


def _resolve_log_dir(spec: dict[str, Any]) -> Path:
    project_dir = Path(str(spec.get("output", {}).get("project_dir") or "")).resolve()
    if project_dir.name == "training_run":
        return project_dir.parent / "logs"
    return project_dir / "logs"


def _collect_user_visible_outputs(spec: dict[str, Any], paths: Any, artifact_store: Any) -> list[Any]:
    project_dir = Path(str(spec.get("output", {}).get("project_dir") or ""))
    run_name = str(spec.get("output", {}).get("run_name") or ".")
    run_root = (project_dir / run_name).resolve()
    outputs: list[Any] = []
    seen: set[str] = set()
    for file_path in _key_training_artifacts(run_root):
        _append_artifact(outputs, seen, paths, artifact_store, file_path)
    log_dir = _resolve_log_dir(spec)
    if log_dir.is_dir():
        for file_path in sorted(log_dir.iterdir()):
            if file_path.is_file() and file_path.name in LOG_FILENAMES:
                _append_artifact(outputs, seen, paths, artifact_store, file_path)
    return outputs


def _key_training_artifacts(run_root: Path) -> list[Path]:
    if not run_root.is_dir():
        return []
    candidates: list[Path] = []
    batch_visuals: list[Path] = []
    for file_path in sorted(run_root.rglob("*")):
        if not file_path.is_file():
            continue
        name = file_path.name
        if name in KEY_WEIGHT_NAMES and file_path.parent.name == "weights":
            candidates.append(file_path)
            continue
        if name in KEY_RUN_ARTIFACT_NAMES:
            candidates.append(file_path)
            continue
        if BATCH_VISUAL_RE.match(name):
            batch_visuals.append(file_path)
    candidates.extend(batch_visuals[:MAX_BATCH_VISUALS])
    return _dedupe_paths(candidates)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for file_path in paths:
        key = str(file_path.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(file_path)
    return unique


def _append_artifact(outputs: list[Any], seen: set[str], paths: Any, artifact_store: Any, file_path: Path) -> None:
    if not file_path.is_file():
        return
    key = str(file_path.resolve()).casefold()
    if key in seen:
        return
    try:
        artifact_store.upsert_artifact(paths, file_path)
        outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))
        seen.add(key)
    except Exception:
        return


def _decode_bytes(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for encoding in ("utf-8", "gb18030", "gbk"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")


def _build_reply(spec: dict[str, Any], stdout_text: str, stderr_text: str, returncode: int) -> str:
    project_dir = Path(str(spec.get("output", {}).get("project_dir") or ""))
    run_name = str(spec.get("output", {}).get("run_name") or ".")
    run_summary = project_dir / run_name / "run_summary.json"
    payload = {
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "dataset_yaml": str(spec.get("dataset", {}).get("data_yaml") or ""),
        "project_dir": str(project_dir),
        "run_name": run_name,
        "run_summary": str(run_summary) if run_summary.exists() else "",
        "stdout_tail": stdout_text[-1200:] if stdout_text else "",
        "stderr_tail": stderr_text[-1200:] if stderr_text else "",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
