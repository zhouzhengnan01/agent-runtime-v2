from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from collections.abc import Callable
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_LOG_ARTIFACTS = 8
LOG_FILENAMES = {
    "data-auto-annotation-stdout.txt",
    "data-auto-annotation-stderr.txt",
    "data-auto-annotation-pipeline-stdout.txt",
    "data-auto-annotation-pipeline-stderr.txt",
    "image-dataset-generation-stdout.txt",
    "image-dataset-generation-stderr.txt",
    "image-dataset-produce-stdout.txt",
    "image-dataset-produce-stderr.txt",
    "synthetic-planner-stdout.txt",
    "synthetic-planner-stderr.txt",
    "dataset-preparation-stdout.txt",
    "dataset-preparation-stderr.txt",
    "model-generated-spec.json",
    "model-generated-spec.txt",
}
KEY_PREPARED_ARTIFACT_NAMES = {
    "dataset.yaml",
    "data_preparation_summary.json",
}
KEY_WORK_ARTIFACT_NAMES = {
    "real_coco.json",
    "synthetic_coco.json",
    "merged_coco.json",
    "synthetic_plan.json",
    "dataset_inspection.json",
}


def run(
    skill_name: str,
    spec: dict[str, Any],
    paths: Any,
    artifact_store: Any,
    on_event: Callable[[str, dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    if _has_dataset_pipeline_input(spec):
        return _run_data_preparation_pipeline(skill_name, spec, paths, artifact_store, package_root, on_event)
    return _run_sam3_annotation(skill_name, spec, paths, artifact_store, package_root, on_event)


def _run_data_preparation_pipeline(
    skill_name: str,
    spec: dict[str, Any],
    paths: Any,
    artifact_store: Any,
    package_root: Path,
    on_event: Callable[[str, dict[str, Any]], Any] | None,
) -> dict[str, Any]:
    script_path = package_root / "scripts" / "run_data_preparation_pipeline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"run_data_preparation_pipeline.py not found: {script_path}")
    normalized = _normalize_spec(spec, paths)
    register_artifacts = _register_artifacts(normalized)
    request_path = Path(paths.workspace) / "data-auto-annotation-pipeline-input.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    returncode, stdout_bytes, stderr_bytes = _run_pipeline_streaming(
        [sys.executable, str(script_path), "--input-json", str(request_path)],
        package_root,
        env,
        paths,
        artifact_store,
        on_event,
        register_artifacts=register_artifacts,
    )
    stdout_text = _decode_bytes(stdout_bytes)
    stderr_text = _decode_bytes(stderr_bytes)
    _write_skill_logs(normalized, stdout_text, stderr_text)
    outputs = _collect_outputs(normalized, paths, artifact_store, register_artifacts=register_artifacts)
    if register_artifacts:
        _emit_artifacts(outputs, on_event)
    return {
        "skill_name": skill_name,
        "outputs": outputs,
        "data": {
            "execution_type": "data_preparation_pipeline",
            "returncode": returncode,
            "stdout": stdout_text[-4000:] if stdout_text else "",
            "stderr": stderr_text[-4000:] if stderr_text else "",
            "prepared_dataset": _prepared_dataset_path(normalized),
            "dataset_yaml": _dataset_yaml_path(normalized),
            "summary_path": _summary_path(normalized),
            "synthetic_plan": _synthetic_plan_path(normalized),
        },
    }


def _run_sam3_annotation(
    skill_name: str,
    spec: dict[str, Any],
    paths: Any,
    artifact_store: Any,
    package_root: Path,
    on_event: Callable[[str, dict[str, Any]], Any] | None,
) -> dict[str, Any]:
    script_path = package_root / "scripts" / "sam3-predict.py"
    if not script_path.exists():
        raise FileNotFoundError(f"sam3-predict.py not found: {script_path}")
    request_path = Path(paths.workspace) / "data-auto-annotation-input.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_spec = _normalize_sam3_spec(spec)
    request_path.write_text(json.dumps(normalized_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path = Path(paths.outputs) / "annotations.coco.json"
    command = [
        sys.executable,
        str(script_path),
        "--annotation-provider",
        str(normalized_spec["annotation_provider"]),
        "--url",
        os.environ.get("SAM3_PREDICT_URL", "$SAM3_PREDICT_URL"),
        "--input-json",
        str(request_path),
        "--connect-timeout",
        "5",
        "--timeout",
        "12",
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, cwd=str(package_root), capture_output=True, text=False, check=False)
    stdout_text = _decode_bytes(completed.stdout)
    stderr_text = _decode_bytes(completed.stderr)
    outputs = _collect_all_output_artifacts(paths, artifact_store)
    if stdout_text:
        outputs.append(artifact_store.write_text_artifact(paths, "data-auto-annotation-stdout.txt", stdout_text))
    if stderr_text:
        outputs.append(artifact_store.write_text_artifact(paths, "data-auto-annotation-stderr.txt", stderr_text))
    _emit_artifacts(outputs, on_event)
    data = _extract_last_json_object(stdout_text)
    data.update(
        {
            "execution_type": "sam3_annotation",
            "returncode": completed.returncode,
            "stdout": stdout_text[-4000:] if stdout_text else "",
        }
    )
    if stderr_text:
        data["stderr"] = stderr_text[-4000:]
    return {"skill_name": skill_name, "outputs": outputs, "data": data}


def _has_dataset_pipeline_input(spec: dict[str, Any]) -> bool:
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    dataset = spec.get("dataset") if isinstance(spec.get("dataset"), dict) else {}
    return bool(str(spec.get("dataset_root") or dataset.get("root_dir") or ctx.get("dataset_root") or "").strip())


def _run_pipeline_streaming(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    paths: Any,
    artifact_store: Any,
    on_event: Callable[[str, dict[str, Any]], Any] | None,
    *,
    register_artifacts: bool,
) -> tuple[int, bytes, bytes]:
    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    emitted: set[str] = set()

    def pump(stream: Any, chunks: list[bytes], is_stderr: bool) -> None:
        try:
            while True:
                chunk = stream.readline()
                if not chunk:
                    break
                chunks.append(chunk)
                text = _decode_bytes(chunk)
                _handle_stream_line(text, paths, artifact_store, on_event, emitted, register_artifacts=register_artifacts)
        finally:
            stream.close()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, stdout_chunks, False), daemon=True),
        threading.Thread(target=pump, args=(proc.stderr, stderr_chunks, True), daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = proc.wait()
    for thread in threads:
        thread.join()
    return returncode, b"".join(stdout_chunks), b"".join(stderr_chunks)


def _handle_stream_line(
    text: str,
    paths: Any,
    artifact_store: Any,
    on_event: Callable[[str, dict[str, Any]], Any] | None,
    emitted: set[str],
    *,
    register_artifacts: bool,
) -> None:
    clean = (text or "").strip()
    if not clean:
        return
    generated_image = _generated_image_from_line(clean)
    if generated_image is not None:
        _emit_file_artifact(
            generated_image,
            paths,
            artifact_store,
            on_event,
            emitted,
            event_type="data_preparation.image_generated",
            register_artifacts=register_artifacts,
        )
    coco_path = _coco_path_from_line(clean)
    if coco_path is not None:
        _emit_file_artifact(
            coco_path,
            paths,
            artifact_store,
            on_event,
            emitted,
            event_type="data_preparation.image_annotated",
            register_artifacts=register_artifacts,
        )


def _generated_image_from_line(line: str) -> Path | None:
    patterns = (
        r"\[data-prep\] annotating composite image immediately:\s*(.+)",
        r"\[data-prep\] annotating image-dataset-produce image immediately:\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            path = Path(match.group(1).strip().strip("\"'"))
            return path.resolve() if path.is_file() else None
    return None


def _coco_path_from_line(line: str) -> Path | None:
    patterns = (
        r"per_image_coco_path:\s*(.+)",
        r"\[data-prep\] per-image real coco written:\s*(.+)",
        r"\[data-prep\] synthetic coco written:\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            path = Path(match.group(1).strip().strip("\"'"))
            return path.resolve() if path.is_file() else None
    return None


def _emit_file_artifact(
    file_path: Path,
    paths: Any,
    artifact_store: Any,
    on_event: Callable[[str, dict[str, Any]], Any] | None,
    emitted: set[str],
    *,
    event_type: str,
    register_artifacts: bool,
) -> None:
    if on_event is None or not file_path.is_file():
        return
    key = str(file_path.resolve()).casefold()
    if key in emitted:
        return
    try:
        if register_artifacts:
            artifact_store.upsert_artifact(paths, file_path)
        artifact = artifact_store.to_artifact_ref(paths.thread_id, file_path)
    except Exception:
        return
    emitted.add(key)
    payload = {"artifact": artifact.model_dump()}
    on_event(event_type, payload)
    if register_artifacts:
        on_event("artifact.created", payload)
        on_event("preview.ready", payload)


def _emit_artifacts(outputs: list[Any], on_event: Callable[[str, dict[str, Any]], Any] | None) -> None:
    if on_event is None:
        return
    for artifact in outputs:
        payload = {"artifact": artifact.model_dump()}
        on_event("artifact.created", payload)
        on_event("preview.ready", payload)


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


def _collect_outputs(spec: dict[str, Any], paths: Any, artifact_store: Any, *, register_artifacts: bool = True) -> list[Any]:
    outputs: list[Any] = []
    seen: set[str] = set()
    for file_path in _key_data_prep_artifacts(spec):
        _append_artifact(outputs, seen, paths, artifact_store, file_path, register_artifacts=register_artifacts)
    log_dir = _log_dir(spec)
    if log_dir.is_dir():
        log_count = 0
        for file_path in sorted(log_dir.iterdir()):
            if not file_path.is_file() or file_path.name not in LOG_FILENAMES:
                continue
            if file_path.stat().st_size <= 0:
                continue
            if log_count >= MAX_LOG_ARTIFACTS:
                continue
            _append_artifact(outputs, seen, paths, artifact_store, file_path, register_artifacts=register_artifacts)
            log_count += 1
    return outputs


def _key_data_prep_artifacts(spec: dict[str, Any]) -> list[Path]:
    prepared_root = _prepared_root(spec)
    work_dir = Path(str(spec.get("work_dir") or "")).resolve()
    candidates: list[Path] = []
    for name in KEY_PREPARED_ARTIFACT_NAMES:
        candidates.append(prepared_root / name)
    for name in KEY_WORK_ARTIFACT_NAMES:
        candidates.append(work_dir / name)
    return _dedupe_paths([path for path in candidates if path.is_file()])


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


def _collect_all_output_artifacts(paths: Any, artifact_store: Any) -> list[Any]:
    outputs: list[Any] = []
    seen: set[str] = set()
    for file_path in sorted(Path(paths.outputs).rglob("*")):
        _append_artifact(outputs, seen, paths, artifact_store, file_path)
    return outputs


def _append_artifact(
    outputs: list[Any],
    seen: set[str],
    paths: Any,
    artifact_store: Any,
    file_path: Path,
    *,
    register_artifacts: bool = True,
) -> None:
    if not file_path.is_file():
        return
    key = str(file_path.resolve()).casefold()
    if key in seen:
        return
    try:
        if register_artifacts:
            artifact_store.upsert_artifact(paths, file_path)
        outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))
        seen.add(key)
    except Exception:
        return


def _register_artifacts(spec: dict[str, Any]) -> bool:
    value = spec.get("register_artifacts")
    if isinstance(value, bool):
        return value
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    value = ctx.get("register_artifacts")
    return value if isinstance(value, bool) else True


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


def _normalize_sam3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    image_path = str(spec.get("image_path") or spec.get("input_dir") or "").strip()
    labels = spec.get("labels") if isinstance(spec.get("labels"), list) else []
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    provider = _normalize_annotation_provider(
        spec.get("annotation_provider") or ctx.get("annotation_provider") or _default_annotation_provider()
    )
    return {"image_path": image_path, "labels": labels, "annotation_provider": provider}


def _default_annotation_provider() -> str:
    return _normalize_annotation_provider(os.getenv("ANNOTATION_PROVIDER", "locate_sam3"))


def _normalize_annotation_provider(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"locate", "locateanything", "locate_anything", "locate_sam3"}:
        return "locate_sam3"
    return "sam3"


def _extract_last_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        return {}
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = parsed
    return result
