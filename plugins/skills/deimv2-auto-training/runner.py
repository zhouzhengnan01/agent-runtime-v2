from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


KEY_ARTIFACT_NAMES = {
    "dataset.yml",
    "train.yml",
    "dataset_summary.json",
    "training_summary.json",
    "run_summary.json",
    "train.log",
    "best_stg1.pth",
    "best_stg2.pth",
    "last.pth",
}
DEFAULT_MODEL_VARIANT = "deimv2-dinov3-s"
DEFAULT_BACKBONE_CHECKPOINT = "models/deimv2/vitt_distill.pt"
MODEL_VARIANTS = {
    "deimv2-dinov3-s": {
        "template_config": "configs/deimv2/deimv2_dinov3_s_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_s_coco.pth",
    },
    "deimv2-dinov3-m": {
        "template_config": "configs/deimv2/deimv2_dinov3_m_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_m_coco.pth",
    },
    "deimv2-dinov3-l": {
        "template_config": "configs/deimv2/deimv2_dinov3_l_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_l_coco.pth",
    },
    "deimv2-dinov3-x": {
        "template_config": "configs/deimv2/deimv2_dinov3_x_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_x_coco.pth",
    },
}


def _normalize_model_variant(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return DEFAULT_MODEL_VARIANT
    normalized = raw.replace("_", "-").replace(" ", "")
    normalized = normalized.removesuffix("-coco")
    if normalized in {"s", "m", "l", "x"}:
        return f"deimv2-dinov3-{normalized}"
    if normalized in {"dinov3-s", "dinov3-m", "dinov3-l", "dinov3-x"}:
        return f"deimv2-{normalized}"
    return normalized if normalized in MODEL_VARIANTS else DEFAULT_MODEL_VARIANT
LOG_FILENAMES = {
    "deimv2-auto-training-stdout.txt",
    "deimv2-auto-training-stderr.txt",
    "deimv2-training-stdout.txt",
    "deimv2-training-stderr.txt",
    "deimv2-training-train.log",
    "deimv2-training-epochs.txt",
}


def run(skill_name: str, spec: dict[str, Any], paths: Any, artifact_store: Any, on_event: Any = None) -> dict[str, Any]:
    del skill_name, on_event
    package_root = Path(__file__).resolve().parent
    normalized = _normalize_spec(spec, paths, package_root)
    request_dir = Path(paths.workspace)
    request_dir.mkdir(parents=True, exist_ok=True)
    request_path = request_dir / "deimv2-auto-training-input.json"
    request_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    project_dir = Path(normalized["output"]["project_dir"]).resolve()
    workflow_log_dir = _resolve_workflow_log_dir(project_dir)
    prepared_dir = project_dir / "prepared_dataset"
    prepare_input = {
        "dataset_root": normalized["dataset"]["root_dir"],
        "coco_json": normalized["dataset"]["coco_json"],
        "output_dir": str(prepared_dir),
        "class_names": normalized["dataset"].get("class_names", []),
        "split": normalized["dataset"].get("split", {}),
        "seed": normalized["dataset"].get("seed", 42),
    }
    prepare_input_path = request_dir / "deimv2-prepare-dataset-input.json"
    prepare_input_path.write_text(json.dumps(prepare_input, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    prepare_proc = subprocess.run(
        [sys.executable, str(package_root / "scripts" / "prepare_deimv2_dataset.py"), "--input", str(prepare_input_path)],
        cwd=str(package_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    if prepare_proc.returncode != 0:
        _write_skill_logs(normalized, prepare_proc.stdout, prepare_proc.stderr, "", "")
        outputs = _collect_outputs(project_dir, paths, artifact_store)
        return _result(outputs, prepare_proc.returncode, prepare_proc.stdout, prepare_proc.stderr, "deimv2_prepare_dataset_failed")

    train_input = {
        "deimv2_root": normalized.get("deimv2_root", ""),
        "dataset_root": str(prepared_dir),
        "work_dir": str(project_dir),
        "training": normalized["training"],
    }
    train_input_path = request_dir / "deimv2-training-input.json"
    train_input_path.write_text(json.dumps(train_input, ensure_ascii=False, indent=2), encoding="utf-8")
    cmd = [sys.executable, str(package_root / "scripts" / "run_deimv2_training.py"), "--input", str(train_input_path)]
    if normalized.get("dry_run") or normalized.get("skip_training"):
        cmd.append("--dry-run")
    train_proc = subprocess.run(
        cmd,
        cwd=str(package_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    _write_skill_logs(normalized, prepare_proc.stdout, prepare_proc.stderr, train_proc.stdout, train_proc.stderr)
    _copy_training_log_to_workflow_logs(project_dir, workflow_log_dir)
    outputs = _collect_outputs(project_dir, paths, artifact_store)
    data = _parse_last_json_object(train_proc.stdout)
    if prepare_proc.stdout:
        data["prepare_stdout"] = prepare_proc.stdout[-4000:]
    if prepare_proc.stderr:
        data["prepare_stderr"] = prepare_proc.stderr[-4000:]
    data.update(
        {
            "execution_type": "deimv2_training",
            "returncode": train_proc.returncode,
            "stdout": train_proc.stdout[-4000:] if train_proc.stdout else "",
            "stderr": train_proc.stderr[-4000:] if train_proc.stderr else "",
            "final_reply": _build_reply(project_dir, train_proc.returncode, data, train_proc.stdout, train_proc.stderr),
        }
    )
    return {"skill_name": "deimv2-auto-training", "outputs": outputs, "data": data}


def _normalize_spec(spec: dict[str, Any], paths: Any, package_root: Path) -> dict[str, Any]:
    payload = dict(spec)
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    runtime = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}

    project_dir = str(output.get("project_dir") or payload.get("project_dir") or (Path(paths.outputs) / "yolo_training_flow" / "deimv2_training_run"))
    class_names = dataset.get("class_names") or payload.get("labels") or payload.get("class_names") or []
    if not isinstance(class_names, list):
        class_names = []
    split = dataset.get("split") or payload.get("split") or {"train": 0.7, "val": 0.2, "test": 0.1}
    if not isinstance(split, dict):
        split = {"train": 0.7, "val": 0.2, "test": 0.1}
    deimv2_root = str(
        payload.get("deimv2_root")
        or os.environ.get("DEIMV2_ROOT", "")
        or (package_root / "vendor" / "deimv2")
    )
    model_variant = _normalize_model_variant(training.get("model_variant") or payload.get("model_variant"))
    model_spec = MODEL_VARIANTS[model_variant]
    disable_tuning_checkpoint = bool(training.get("disable_tuning_checkpoint"))
    tuning_checkpoint = (
        ""
        if disable_tuning_checkpoint
        else training.get("tuning_checkpoint") or os.environ.get("DEIMV2_TUNING_CHECKPOINT", "") or model_spec["tuning_checkpoint"]
    )
    normalized_training = {
        "model_variant": model_variant,
        "template_config": training.get("template_config") or model_spec["template_config"],
        "epochs": int(training.get("epochs") or payload.get("epochs") or 10),
        "img_size": int(training.get("img_size") or training.get("imgsz") or payload.get("imgsz") or 640),
        "batch": int(training.get("batch") or payload.get("batch") or 1),
        "device": training.get("device") or payload.get("device") or "auto",
        "workers": int(training.get("workers") if training.get("workers") is not None else 0),
        "backbone_checkpoint": training.get("backbone_checkpoint") or os.environ.get("DEIMV2_BACKBONE_CHECKPOINT", "") or DEFAULT_BACKBONE_CHECKPOINT,
        "tuning_checkpoint": tuning_checkpoint,
        "disable_tuning_checkpoint": disable_tuning_checkpoint,
        "conda_env": training.get("conda_env") or training.get("conda_env_name") or runtime.get("conda_env_name") or "",
    }
    for key in (
        "warmup_iter",
        "checkpoint_freq",
        "num_top_queries",
        "flat_epoch",
        "no_aug_epoch",
        "lr",
        "learning_rate",
        "weight_decay",
        "betas",
        "optimizer",
        "model_source",
        "model_sha256",
        "model_original_name",
        "model_id",
        "model_extension",
        "deimv2_upload_model_usage",
    ):
        if training.get(key) is not None:
            normalized_training[key] = training[key]
    return {
        "skill_name": "deimv2-auto-training",
        "deimv2_root": deimv2_root,
        "dataset": {
            "root_dir": str(dataset.get("root_dir") or dataset.get("dataset_root") or payload.get("dataset_root") or ""),
            "coco_json": str(dataset.get("coco_json") or payload.get("coco_json") or ""),
            "class_names": [str(item).strip() for item in class_names if str(item).strip()],
            "split": split,
            "seed": int(dataset.get("seed") or payload.get("seed") or 42),
        },
        "training": normalized_training,
        "runtime": dict(runtime),
        "output": {"project_dir": project_dir, "run_name": str(output.get("run_name") or ".")},
        "dry_run": bool(payload.get("dry_run", False)),
        "skip_training": bool(payload.get("skip_training", False)),
        "workflow_context": payload.get("workflow_context") if isinstance(payload.get("workflow_context"), dict) else {},
    }


def _collect_outputs(project_dir: Path, paths: Any, artifact_store: Any) -> list[Any]:
    outputs: list[Any] = []
    seen: set[str] = set()
    if project_dir.exists():
        for file_path in sorted(project_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.name not in KEY_ARTIFACT_NAMES and file_path.suffix.lower() != ".pth":
                continue
            _append_artifact(outputs, seen, paths, artifact_store, file_path)
    log_dir = _resolve_workflow_log_dir(project_dir)
    if log_dir.is_dir():
        for file_path in sorted(log_dir.iterdir()):
            if file_path.is_file() and file_path.name in LOG_FILENAMES:
                _append_artifact(outputs, seen, paths, artifact_store, file_path)
    return outputs


def _append_artifact(outputs: list[Any], seen: set[str], paths: Any, artifact_store: Any, file_path: Path) -> None:
    key = str(file_path.resolve()).casefold()
    if key in seen:
        return
    try:
        artifact_store.upsert_artifact(paths, file_path)
        outputs.append(artifact_store.to_artifact_ref(paths.thread_id, file_path))
        seen.add(key)
    except Exception:
        return


def _result(outputs: list[Any], returncode: int, stdout: str, stderr: str, status: str) -> dict[str, Any]:
    return {
        "skill_name": "deimv2-auto-training",
        "outputs": outputs,
        "data": {
            "execution_type": "deimv2_training",
            "status": status,
            "returncode": returncode,
            "stdout": stdout[-4000:] if stdout else "",
            "stderr": stderr[-4000:] if stderr else "",
            "final_reply": stderr[-2000:] if stderr else stdout[-2000:],
        },
    }


def _resolve_workflow_log_dir(project_dir: Path) -> Path:
    resolved = project_dir.resolve()
    if resolved.name == "deimv2_training_run":
        return resolved.parent / "logs"
    return resolved / "logs"


def _write_skill_logs(normalized: dict[str, Any], prepare_stdout: str, prepare_stderr: str, train_stdout: str, train_stderr: str) -> None:
    project_dir = Path(str(normalized.get("output", {}).get("project_dir") or "")).resolve()
    log_dir = _resolve_workflow_log_dir(project_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    skill_stdout = _sectioned_log(
        [
            ("prepare stdout", prepare_stdout),
            ("train stdout", train_stdout),
        ]
    )
    skill_stderr = _sectioned_log(
        [
            ("prepare stderr", prepare_stderr),
            ("train stderr", train_stderr),
        ]
    )
    (log_dir / "deimv2-auto-training-stdout.txt").write_text(skill_stdout, encoding="utf-8")
    (log_dir / "deimv2-auto-training-stderr.txt").write_text(skill_stderr, encoding="utf-8")
    (log_dir / "deimv2-training-stdout.txt").write_text(train_stdout or "", encoding="utf-8")
    (log_dir / "deimv2-training-stderr.txt").write_text(train_stderr or "", encoding="utf-8")


def _sectioned_log(sections: list[tuple[str, str]]) -> str:
    chunks: list[str] = []
    for title, text in sections:
        if not text:
            continue
        chunks.append(f"--- {title} ---\n{text.rstrip()}")
    return "\n\n".join(chunks) + ("\n" if chunks else "")


def _copy_training_log_to_workflow_logs(project_dir: Path, log_dir: Path) -> None:
    source = project_dir / "logs" / "train.log"
    if not source.is_file():
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    (log_dir / "deimv2-training-train.log").write_text(text, encoding="utf-8")
    _write_deimv2_epoch_log(text, log_dir)


def _write_deimv2_epoch_log(train_log: str, log_dir: Path) -> None:
    selected: list[str] = []
    for line in (train_log or "").replace("\r", "\n").splitlines():
        stripped = " ".join(line.strip().split())
        if not stripped:
            continue
        lower = stripped.lower()
        if (
            stripped.startswith("Epoch:")
            or stripped.startswith("best_stat:")
            or stripped.startswith("Averaged stats:")
            or "average precision" in lower
            or "average recall" in lower
            or stripped.startswith("Training time")
        ):
            selected.append(stripped)
    (log_dir / "deimv2-training-epochs.txt").write_text("\n".join(selected).strip() + ("\n" if selected else ""), encoding="utf-8")


def _parse_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    for index, char in enumerate(text or ""):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = parsed
    return result


def _build_reply(project_dir: Path, returncode: int, data: dict[str, Any], stdout: str, stderr: str) -> str:
    payload = {
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "training_backend": "deimv2",
        "project_dir": str(project_dir),
        "config": data.get("config") or str(project_dir / "configs" / "train.yml"),
        "dataset_config": data.get("dataset_config") or str(project_dir / "configs" / "dataset.yml"),
        "best_checkpoint": data.get("best_checkpoint") or "",
        "stdout_tail": stdout[-1200:] if stdout else "",
        "stderr_tail": stderr[-1200:] if stderr else "",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
