from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def run(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    if _has_dataset_pipeline_input(spec):
        script = package_root / "scripts" / "run_data_preparation_pipeline.py"
        request_path = paths.workspace / "data-auto-annotation-pipeline-input.json"
        request_path.write_text(json.dumps(_normalize_pipeline_spec(spec, paths), ensure_ascii=False, indent=2), encoding="utf-8")
        command = [sys.executable, str(script), "--input-json", str(request_path)]
        execution_type = "data_preparation_pipeline"
    else:
        script = package_root / "scripts" / "sam3-predict.py"
        request_path = paths.workspace / "data-auto-annotation-input.json"
        request_path.write_text(json.dumps(_normalize_sam3_spec(spec), ensure_ascii=False, indent=2), encoding="utf-8")
        request_json = request_path.read_text(encoding="utf-8")
        output_path = paths.outputs / "annotations.coco.json"
        command = [
            sys.executable,
            str(script),
            "--url",
            os.environ.get("SAM3_PREDICT_URL", "$SAM3_PREDICT_URL"),
            "--input-json",
            request_json,
            "--connect-timeout",
            "5",
            "--timeout",
            "12",
            "--output",
            str(output_path),
        ]
        execution_type = "sam3_annotation"

    completed = subprocess.run(command, cwd=str(package_root), capture_output=True, text=False, check=False)
    stdout_text = _decode_bytes(completed.stdout)
    stderr_text = _decode_bytes(completed.stderr)

    outputs = []
    for file_path in sorted(paths.outputs.rglob("*")):
        if file_path.is_file():
            artifact_store.upsert_artifact(paths, file_path)
            outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))
    if stdout_text:
        outputs.append(artifact_store.write_text_artifact(paths, "data-auto-annotation-stdout.txt", stdout_text))
    if stderr_text:
        outputs.append(artifact_store.write_text_artifact(paths, "data-auto-annotation-stderr.txt", stderr_text))

    data = _extract_last_json_object(stdout_text)
    data.update(
        {
            "execution_type": execution_type,
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


def _normalize_pipeline_spec(spec: dict[str, Any], paths: Any) -> dict[str, Any]:
    normalized = dict(spec)
    ctx = normalized.get("workflow_context") if isinstance(normalized.get("workflow_context"), dict) else {}
    normalized.setdefault("work_dir", str((paths.outputs / "yolo_training_flow" / "pipeline_work").resolve()))
    normalized.setdefault("output_dir", str((paths.outputs / "yolo_training_flow" / "prepared_data").resolve()))
    normalized.setdefault("labels", ctx.get("labels") or ctx.get("class_names") or normalized.get("class_names") or [])
    normalized.setdefault("split_requested", True)
    normalized.setdefault("skip_generation", False)
    return normalized


def _normalize_sam3_spec(spec: dict[str, Any]) -> dict[str, Any]:
    image_path = str(spec.get("image_path") or spec.get("input_dir") or "").strip()
    labels = spec.get("labels") if isinstance(spec.get("labels"), list) else []
    return {"image_path": image_path, "labels": labels}


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
