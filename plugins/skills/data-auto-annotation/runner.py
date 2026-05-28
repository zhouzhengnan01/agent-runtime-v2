from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LOG_FILENAMES = {
    "data-auto-annotation-stdout.txt",
    "data-auto-annotation-stderr.txt",
    "image-dataset-generation-stdout.txt",
    "image-dataset-generation-stderr.txt",
    "synthetic-planner-stdout.txt",
    "synthetic-planner-stderr.txt",
    "dataset-preparation-stdout.txt",
    "dataset-preparation-stderr.txt",
    "model-generated-spec.json",
    "model-generated-spec.txt",
}


def run(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any) -> dict[str, Any]:
    del skill_name
    package_root = Path(__file__).resolve().parent
    script_path = package_root / "scripts" / "run_data_preparation_pipeline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"run_data_preparation_pipeline.py not found: {script_path}")

    normalized = _normalize_spec(spec, paths)
    request_path = Path(paths.workspace) / "data-auto-annotation-pipeline-input.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path), "--input-json", str(request_path)],
        cwd=str(package_root),
        capture_output=True,
        env=env,
        check=False,
    )
    stdout_text = _decode_bytes(completed.stdout)
    stderr_text = _decode_bytes(completed.stderr)
    _write_skill_logs(normalized, stdout_text, stderr_text)
    outputs = _collect_outputs(normalized, paths, artifact_store)
    return {
        "skill_name": "data-auto-annotation",
        "outputs": outputs,
        "data": {
            "execution_type": "data_preparation_pipeline",
            "returncode": completed.returncode,
            "stdout": stdout_text[-4000:] if stdout_text else "",
            "stderr": stderr_text[-4000:] if stderr_text else "",
            "prepared_dataset": _prepared_dataset_path(normalized),
            "dataset_yaml": _dataset_yaml_path(normalized),
            "summary_path": _summary_path(normalized),
            "synthetic_plan": _synthetic_plan_path(normalized),
        },
    }


def _normalize_spec(spec: dict[str, Any], paths: Any) -> dict[str, Any]:
    payload = dict(spec)
    ctx = payload.get("workflow_context") if isinstance(payload.get("workflow_context"), dict) else {}
    output_root = Path(paths.outputs).resolve()
    dataset_archive = str(payload.get("dataset_archive") or "").strip()
    if dataset_archive:
        payload["dataset_root"] = str(_unpack_dataset_archive(dataset_archive, output_root / "uploaded_dataset", paths))
        payload.pop("dataset_archive", None)
    elif not payload.get("dataset_root") and payload.get("image_path"):
        payload["dataset_root"] = payload.get("image_path")
    payload.setdefault("work_dir", str(output_root / "pipeline_work"))
    payload.setdefault("output_dir", str(output_root / "prepared_data"))
    payload.setdefault("task", "")
    payload.setdefault("skip_generation", False)
    payload.setdefault("max_synthetic", 2000)
    payload.setdefault("split", {"train": 0.7, "val": 0.2, "test": 0.1})
    payload.setdefault("training", {})
    payload["training"].setdefault("task", "detect")
    if ctx:
        payload["workflow_context"] = ctx
    return payload


def _unpack_dataset_archive(raw_path: str, target_dir: Path, paths: Any) -> Path:
    source = _resolve_uploaded_local_path(Path(paths.root), raw_path)
    if not source.exists():
        raise FileNotFoundError(f"找不到上传的文件: {source}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    lower_name = source.name.lower()
    if lower_name.endswith(".tar.gz"):
        shutil.unpack_archive(str(source), str(target_dir), format="gztar")
    elif lower_name.endswith(".tar"):
        shutil.unpack_archive(str(source), str(target_dir), format="tar")
    elif lower_name.endswith(".zip"):
        shutil.unpack_archive(str(source), str(target_dir), format="zip")
    else:
        shutil.unpack_archive(str(source), str(target_dir))
    child_dirs = [p for p in target_dir.iterdir() if p.is_dir()]
    return child_dirs[0] if len(child_dirs) == 1 else target_dir


def _resolve_uploaded_local_path(thread_root: Path, raw_path: str) -> Path:
    value = (raw_path or "").strip()
    direct = Path(value).expanduser()
    if direct.exists():
        return direct.resolve()
    normalized = value.replace("\\", "/")
    uploads_dir = thread_root / "uploads"
    outputs_dir = thread_root / "outputs"
    for prefix, base in (("/mnt/user-data/uploads/", uploads_dir), ("d:/mnt/user-data/uploads/", uploads_dir), ("/mnt/user-data/outputs/", outputs_dir), ("d:/mnt/user-data/outputs/", outputs_dir)):
        if normalized.lower().startswith(prefix):
            rel = normalized[len(prefix):]
            candidate = (base / rel).resolve()
            if candidate.exists():
                return candidate
    basename = Path(normalized).name
    for base in (uploads_dir, outputs_dir):
        candidate = (base / basename).resolve()
        if candidate.exists():
            return candidate
    return direct.resolve() if direct.is_absolute() else (thread_root / value).resolve()


def _log_dir(spec: dict[str, Any]) -> Path:
    work_dir = Path(str(spec.get("work_dir") or "")).resolve()
    if work_dir.name == "pipeline_work":
        return work_dir.parent / "logs"
    return work_dir / "logs"


def _write_skill_logs(spec: dict[str, Any], stdout_text: str, stderr_text: str) -> None:
    log_dir = _log_dir(spec)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "data-auto-annotation-pipeline-stdout.txt").write_text(stdout_text or "", encoding="utf-8")
    (log_dir / "data-auto-annotation-pipeline-stderr.txt").write_text(stderr_text or "", encoding="utf-8")


def _prepared_root(spec: dict[str, Any]) -> Path:
    return Path(str(spec.get("output_dir") or "")).resolve()


def _prepared_dataset_path(spec: dict[str, Any]) -> str:
    value = _prepared_root(spec) / "prepared_dataset"
    return str(value) if value.exists() else ""


def _dataset_yaml_path(spec: dict[str, Any]) -> str:
    value = _prepared_root(spec) / "dataset.yaml"
    return str(value) if value.exists() else ""


def _summary_path(spec: dict[str, Any]) -> str:
    value = _prepared_root(spec) / "data_preparation_summary.json"
    return str(value) if value.exists() else ""


def _synthetic_plan_path(spec: dict[str, Any]) -> str:
    value = Path(str(spec.get("work_dir") or "")).resolve() / "synthetic_plan.json"
    return str(value) if value.exists() else ""


def _collect_outputs(spec: dict[str, Any], paths: Any, artifact_store: Any) -> list[Any]:
    outputs: list[Any] = []
    prepared_images = _prepared_root(spec) / "prepared_dataset" / "images"
    for split in ("train", "val", "test"):
        split_dir = prepared_images / split
        if not split_dir.is_dir():
            continue
        for file_path in sorted(split_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                artifact_store.upsert_artifact(paths, file_path)
                outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))
            except Exception:
                continue
    log_dir = _log_dir(spec)
    if log_dir.is_dir():
        for file_path in sorted(log_dir.iterdir()):
            if not file_path.is_file() or file_path.name not in LOG_FILENAMES:
                continue
            try:
                artifact_store.upsert_artifact(paths, file_path)
                outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))
            except Exception:
                continue
    return outputs


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
