from __future__ import annotations

import json
import os
import subprocess
import re
from pathlib import Path
from typing import Any


def run(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any) -> dict[str, Any]:
    del skill_name
    package_root = Path(__file__).resolve().parent
    script_path = package_root / "scripts" / "run_in_conda.ps1"
    if not script_path.exists():
        raise FileNotFoundError(f"run_in_conda.ps1 not found: {script_path}")

    request_path = paths.workspace / "gpu-training-orchestrator-input.json"
    spec = _normalize_training_spec(spec, paths)
    request_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    config_dir = paths.workspace / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings_file = config_dir / "settings.json"
    if not settings_file.exists():
        settings_file.write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env["YOLO_CONFIG_DIR"] = str(config_dir.resolve())
    env["ULTRALYTICS_SETTINGS"] = str(settings_file.resolve())
    env["ULTRALYTICS_HOME"] = str(config_dir.resolve())

    completed = subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script_path),
            "-InputJsonPath",
            str(request_path),
        ],
        cwd=str(package_root),
        capture_output=True,
        env=env,
        check=False,
    )
    stdout_text = _decode_bytes(completed.stdout)
    stderr_text = _decode_bytes(completed.stderr)

    outputs = []
    for file_path in sorted(paths.outputs.rglob("*")):
        if not file_path.is_file():
            continue
        artifact_store.upsert_artifact(paths, file_path)
        outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))

    if stdout_text:
        outputs.append(artifact_store.write_text_artifact(paths, "gpu-training-orchestrator-stdout.txt", stdout_text))
    if stderr_text:
        outputs.append(artifact_store.write_text_artifact(paths, "gpu-training-orchestrator-stderr.txt", stderr_text))

    final_reply = _build_skill_reply(paths, stdout_text, stderr_text)

    return {
        "skill_name": "gpu-training-orchestrator",
        "outputs": outputs,
        "data": {
            "execution_type": "conda_ps1",
            "returncode": completed.returncode,
            "stdout": stdout_text[-4000:] if stdout_text else "",
            "stderr": stderr_text[-4000:] if stderr_text else "",
            "final_reply": final_reply,
        },
    }


def _normalize_training_spec(spec: dict[str, Any], paths: Any) -> dict[str, Any]:
    ctx = spec.get("workflow_context") if isinstance(spec.get("workflow_context"), dict) else {}
    overrides = str(spec.get("overrides_text") or "")
    run_name = str(ctx.get("run_name") or f"smoking_yolo_{paths.thread_id}")
    project_dir = str(ctx.get("project_dir") or (Path(paths.outputs) / "training_runs").resolve())
    dataset_root = str(ctx.get("dataset_root") or "")
    coco_json = str(ctx.get("coco_json") or "")
    class_names = _normalize_labels(ctx.get("class_names") or ctx.get("labels") or spec.get("class_names") or spec.get("labels"))

    normalized = {
        "skill_name": "gpu-training-orchestrator",
        "overrides_text": overrides,
        "runtime": {
            "conda_env_name": "yolo_jetson",
            "enforce_conda_env": True,
        },
        "dataset": {
            "root_dir": dataset_root,
            "coco_json": coco_json,
            "split": {"train": 0.7, "val": 0.2, "test": 0.1},
            "class_names": class_names,
            "copy_images": True,
        },
        "training": {
            "task": "detect",
            "model": "yolo11n.pt",
            "epochs": 50,
            "imgsz": 640,
            "batch": 8,
            "device": "0",
            "workers": 4,
            "patience": 20,
        },
        "output": {"project_dir": project_dir, "run_name": run_name},
    }
    _apply_overrides_text(normalized, overrides)
    return normalized


def _apply_overrides_text(config: dict[str, Any], overrides_text: str) -> None:
    """Apply simple key=value hints from the user's prompt to the generated input.json."""
    if not overrides_text.strip():
        return

    runtime = config.setdefault("runtime", {})
    training = config.setdefault("training", {})
    dataset = config.setdefault("dataset", {})
    split = dataset.setdefault("split", {})

    text = overrides_text.strip()

    conda_env = _find_string_override(text, ("conda_env_name", "conda_env", "env"))
    if conda_env:
        runtime["conda_env_name"] = conda_env

    model = _find_string_override(text, ("model",))
    if model:
        training["model"] = model

    task = _find_string_override(text, ("task",))
    if task in {"detect", "segment"}:
        training["task"] = task

    device = _find_string_override(text, ("device",))
    if device:
        training["device"] = device

    for key in ("epochs", "imgsz", "batch", "workers", "patience"):
        value = _find_int_override(text, key)
        if value is not None:
            training[key] = value

    split_changed = False
    for key in ("train", "val", "test"):
        value = _find_float_override(text, f"dataset.split.{key}")
        if value is None:
            value = _find_float_override(text, f"split.{key}")
        if value is not None:
            split[key] = value
            split_changed = True

    if split_changed:
        total = sum(float(split.get(key, 0) or 0) for key in ("train", "val", "test"))
        if total > 0 and abs(total - 1.0) > 1e-6:
            for key in ("train", "val", "test"):
                split[key] = round(float(split.get(key, 0) or 0) / total, 6)


def _normalize_labels(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[,，;；\s]+", value)
    elif isinstance(value, list):
        parts = [str(item) for item in value]
    else:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for item in parts:
        label = item.strip().strip("\"'`，,;；。")
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _find_string_override(text: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        match = re.search(rf"(?<![\w.]){re.escape(key)}\s*[:=]\s*([^\s,，;；]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'")
    return ""


def _find_int_override(text: str, key: str) -> int | None:
    match = re.search(rf"(?<![\w.]){re.escape(key)}\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _find_float_override(text: str, key: str) -> float | None:
    match = re.search(rf"(?<![\w.]){re.escape(key)}\s*[:=]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


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


def _build_skill_reply(paths: Any, stdout_text: str, stderr_text: str) -> str:
    summary = _read_run_summary(paths)
    failed = _is_training_failed(summary, stdout_text, stderr_text)
    if failed:
        reason = _extract_failure_reason(stderr_text, stdout_text)
        return "\n".join(
            [
                "YOLO 模型训练失败",
                "",
                f"失败原因：{reason}",
                "",
                "请先修复数据或配置后重试，重点检查：`dataset.coco_json` 是否为真实 COCO 标注文件（包含 images/annotations/categories）。",
            ]
        ).strip()

    eval_block = _read_eval_block_from_stdout(stdout_text)
    best_pt = _find_best_pt(paths)

    model = summary.get("model") or "yolo11n.pt"
    task = summary.get("task") or "detect"
    train_dir = summary.get("train_save_dir") or ""
    num_images = summary.get("num_images")
    num_categories = summary.get("num_categories")
    split_counts = summary.get("split_counts") or {}
    conda_env = summary.get("conda_env_name") or "yolo_jetson"
    eval_error = str(summary.get("eval_error") or "").strip()

    split_text = "-"
    if isinstance(split_counts, dict):
        split_text = f"训练 {split_counts.get('train', '-')} / 验证 {split_counts.get('val', '-')} / 测试 {split_counts.get('test', '-')}"

    payload = {
        "status": "completed",
        "message": "YOLO training completed. Use these facts to generate the final user-facing answer.",
        "conda_env": conda_env,
        "model": model,
        "task": task,
        "num_images": num_images,
        "num_categories": num_categories,
        "split": split_text,
        "train_dir": train_dir,
        "best_pt": best_pt,
        "results_png": str(Path(train_dir) / "results.png") if train_dir else "",
        "results_csv": str(Path(train_dir) / "results.csv") if train_dir else "",
        "eval_block": eval_block,
        "eval_error": eval_error,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _is_training_failed(summary: dict[str, Any], stdout_text: str, stderr_text: str) -> bool:
    if not summary:
        return True
    text = f"{stdout_text}\n{stderr_text}".lower()
    if "training failed" in text or "[error]" in text or "traceback" in text:
        return True
    return False


def _extract_failure_reason(stderr_text: str, stdout_text: str) -> str:
    for source in (stderr_text, stdout_text):
        lines = [ln.strip() for ln in source.splitlines() if ln.strip()]
        for ln in reversed(lines):
            lower = ln.lower()
            if "valueerror" in lower or "filenotfounderror" in lower or "training failed" in lower or "[error]" in lower:
                return ln
    return "训练进程异常退出（详见 gpu-training-orchestrator-stderr.txt）"


def _read_run_summary(paths: Any) -> dict[str, Any]:
    candidates = sorted(Path(paths.outputs).rglob("run_summary.json"))
    if not candidates:
        return {}
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_best_pt(paths: Any) -> str:
    candidates = sorted(Path(paths.outputs).rglob("best.pt"))
    if not candidates:
        return ""
    return str(candidates[-1])


def _read_eval_block_from_stdout(stdout_text: str) -> str:
    if not stdout_text.strip():
        return ""
    ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    clean_text = ansi.sub("", stdout_text)
    lines = [ln.strip() for ln in clean_text.splitlines() if ln.strip()]

    summary_line = ""
    for ln in lines:
        if "summary (fused)" in ln and "YOLO" in ln:
            summary_line = ln

    class_header = ""
    class_rows: list[str] = []
    for i, ln in enumerate(lines):
        if re.search(r"\bClass\s+Images\s+Instances\s+Box\(P", ln):
            class_header = ln
            rows: list[str] = []
            for j in range(i + 1, min(i + 8, len(lines))):
                row = lines[j]
                if row.startswith("all") or row.startswith("person") or row.startswith("cigarette"):
                    rows.append(row)
            if rows:
                class_rows = rows

    out: list[str] = []
    if summary_line:
        out.append(summary_line)
    if class_header:
        out.append(class_header)
    out.extend(class_rows)
    return "\n".join(out).strip()
