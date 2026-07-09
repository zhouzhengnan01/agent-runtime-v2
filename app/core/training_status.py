from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from app.schemas import ChatEvent


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROGRESS_STATE_NAME = "progress_state.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def runtime_root() -> Path:
    return Path.cwd() / ".runtime" / "threads"


def safe_thread_dir(thread_id: str) -> Path:
    if not thread_id or any(part in thread_id for part in ("..", "/", "\\")):
        raise ValueError(f"Invalid thread_id: {thread_id}")
    root = runtime_root().resolve()
    target = (root / thread_id).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"Thread path escapes runtime root: {thread_id}")
    return target


def safe_run_id(run_id: str) -> str:
    if not run_id or any(part in run_id for part in ("..", "/", "\\")):
        raise ValueError(f"Invalid run_id: {run_id}")
    return run_id


class TrainingProgressWriter:
    """Persist workflow progress events for the training status API."""

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        run_dir: Path,
        workflow: str,
        backend: str,
        app_template_name: str | None,
        selected_skills: list[str],
        user_text: str,
    ) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self.run_dir = run_dir
        self.path = run_dir / PROGRESS_STATE_NAME
        self.workflow = workflow
        self.backend = backend
        self.app_template_name = app_template_name
        self.selected_skills = selected_skills
        self.user_text = user_text

    def initialize(self) -> None:
        state = self._read()
        now = utc_now()
        state.setdefault("schema", "jetlinks-training-progress.v1")
        state.update(
            {
                "thread_id": self.thread_id,
                "run_id": self.run_id,
                "workflow": self.workflow,
                "backend": self.backend,
                "app_template_name": self.app_template_name,
                "status": state.get("status") or "running",
                "phase": state.get("phase") or "run_selected",
                "phase_label": state.get("phase_label") or "run selected",
                "created_at": state.get("created_at") or now,
                "started_at": state.get("started_at") or now,
                "updated_at": now,
                "completed_at": state.get("completed_at"),
                "request": {
                    **(state.get("request") if isinstance(state.get("request"), dict) else {}),
                    "user_text": self.user_text,
                    "selected_skills": self.selected_skills,
                },
                "paths": {
                    **(state.get("paths") if isinstance(state.get("paths"), dict) else {}),
                    "run_dir": str(self.run_dir),
                    "workspace": str(self.run_dir / "workspace"),
                    "outputs": str(self.run_dir),
                },
                "model_request": state.get("model_request")
                if isinstance(state.get("model_request"), dict)
                else {"current": None, "history": []},
                "errors": state.get("errors") if isinstance(state.get("errors"), list) else [],
                "warnings": state.get("warnings") if isinstance(state.get("warnings"), list) else [],
            }
        )
        self._write(state)

    def handle_event(self, event: ChatEvent) -> None:
        state = self._read()
        now = utc_now()
        state["updated_at"] = now
        state["last_event"] = {"type": event.type, "timestamp": event.data.get("timestamp") or now}
        if event.type == "workflow.training_run.selected":
            state["phase"] = "run_selected"
            state["phase_label"] = "run selected"
            state["backend"] = event.data.get("training_backend") or state.get("backend")
        elif event.type == "workflow.request_spec.resolved":
            self._merge_request(state, event.data)
        elif event.type == "llm.started":
            self._mark_llm_started(state, event)
        elif event.type == "llm.completed":
            self._mark_llm_completed(state, event)
        elif event.type == "skill.started":
            self._mark_skill_started(state, event)
        elif event.type == "skill.completed":
            self._mark_skill_completed(state, event)
        elif event.type == "data_preparation.image_generated":
            self._mark_stage_running(state, "generation", "synthetic_generation")
            self._increment_counter(state, "generation", "completed")
            state["phase"] = "synthetic_generation"
            state["phase_label"] = "synthetic generation running"
        elif event.type == "data_preparation.image_annotated":
            self._mark_stage_running(state, "annotation", "annotation")
            self._increment_counter(state, "annotation", "completed")
            state["phase"] = "annotation"
            state["phase_label"] = "annotation running"
        elif event.type == "run.failed":
            state["status"] = "failed"
            state["phase"] = "failed"
            state["phase_label"] = "failed"
            state["completed_at"] = now
            if event.data.get("error"):
                self._append_error(state, str(event.data["error"]))
        elif event.type == "run.completed":
            state["status"] = "completed"
            state["phase"] = "completed"
            state["phase_label"] = "completed"
            state["completed_at"] = now
        self._write(state)

    def _merge_request(self, state: dict[str, Any], data: dict[str, Any]) -> None:
        request = state.setdefault("request", {})
        if not isinstance(request, dict):
            return
        for key in (
            "labels",
            "annotation_prompts",
            "annotation_prompt_map",
            "generation_prompt",
            "task_description",
            "max_synthetic_images",
            "training_backend",
            "training_config",
        ):
            if key in data:
                request[key] = data.get(key)

    def _mark_llm_started(self, state: dict[str, Any], event: ChatEvent) -> None:
        model_request = state.setdefault("model_request", {"current": None, "history": []})
        purpose = str(event.data.get("purpose") or "llm_request")
        current = {
            "status": "running",
            "purpose": purpose,
            "phase_label": _llm_phase_label(purpose),
            "model": event.data.get("model"),
            "configured": event.data.get("configured"),
            "started_at": event.data.get("timestamp") or utc_now(),
            "completed_at": None,
            "elapsed_ms": None,
            "error": None,
        }
        model_request["current"] = current
        state["phase"] = "llm_request"
        state["phase_label"] = current["phase_label"]

    def _mark_llm_completed(self, state: dict[str, Any], event: ChatEvent) -> None:
        model_request = state.setdefault("model_request", {"current": None, "history": []})
        current = model_request.get("current") if isinstance(model_request.get("current"), dict) else {}
        purpose = event.data.get("purpose") or current.get("purpose") or "llm_request"
        error = event.data.get("failure_reason") or event.data.get("error")
        item = {
            **current,
            "status": "failed" if error else "completed",
            "purpose": purpose,
            "completed_at": event.data.get("timestamp") or utc_now(),
            "elapsed_ms": event.data.get("elapsed_ms"),
            "error": error,
            "used_fallback": event.data.get("used_fallback"),
            "reason": event.data.get("reason"),
        }
        history = model_request.setdefault("history", [])
        if isinstance(history, list):
            history.append(item)
            model_request["history"] = history[-50:]
        model_request["current"] = None
        if error:
            self._append_error(state, f"LLM request failed: {purpose}: {error}")

    def _mark_skill_started(self, state: dict[str, Any], event: ChatEvent) -> None:
        skill = str(event.data.get("skill_name") or "")
        if skill == "data-auto-annotation":
            state["phase"] = "data_preparation"
            state["phase_label"] = "data preparation running"
            self._mark_stage_pending(state, "annotation")
            self._mark_stage_pending(state, "generation")
        elif skill in {"gpu-training-orchestrator", "deimv2-auto-training"}:
            state["phase"] = "training"
            state["phase_label"] = "training running"
            training = state.setdefault("training", {})
            if isinstance(training, dict):
                training["status"] = "running"
                training["started_at"] = training.get("started_at") or utc_now()

    def _mark_skill_completed(self, state: dict[str, Any], event: ChatEvent) -> None:
        skill = str(event.data.get("skill_name") or "")
        if skill == "data-auto-annotation":
            state["phase"] = "data_preparation_completed"
            state["phase_label"] = "data preparation completed"
            self._mark_stage_completed_if_started(state, "annotation")
            self._mark_stage_completed_if_started(state, "generation")
        elif skill in {"gpu-training-orchestrator", "deimv2-auto-training"}:
            training = state.setdefault("training", {})
            if isinstance(training, dict):
                training["status"] = "completed"
                training["completed_at"] = utc_now()

    def _increment_counter(self, state: dict[str, Any], block: str, field: str) -> None:
        target = state.setdefault(block, {})
        if isinstance(target, dict):
            target[field] = int(target.get(field) or 0) + 1
            target["updated_at"] = utc_now()

    def _mark_stage_pending(self, state: dict[str, Any], block: str) -> None:
        target = state.setdefault(block, {})
        if isinstance(target, dict):
            target.setdefault("status", "pending")
            target.setdefault("started_at", None)
            target.setdefault("completed_at", None)

    def _mark_stage_running(self, state: dict[str, Any], block: str, phase: str) -> None:
        target = state.setdefault(block, {})
        if isinstance(target, dict):
            target["status"] = "running"
            target["phase"] = phase
            target["started_at"] = target.get("started_at") or utc_now()
            target["completed_at"] = None
            target["updated_at"] = utc_now()

    def _mark_stage_completed_if_started(self, state: dict[str, Any], block: str) -> None:
        target = state.setdefault(block, {})
        if isinstance(target, dict) and target.get("started_at"):
            target["status"] = "completed"
            target["completed_at"] = target.get("completed_at") or utc_now()
            target["updated_at"] = utc_now()

    def _append_error(self, state: dict[str, Any], message: str) -> None:
        errors = state.setdefault("errors", [])
        if isinstance(errors, list):
            errors.append({"message": message[-4000:], "timestamp": utc_now()})
            state["errors"] = errors[-20:]

    def _read(self) -> dict[str, Any]:
        return _read_json(self.path)

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def chain_event_hooks(*hooks: Callable[[ChatEvent], None] | None) -> Callable[[ChatEvent], None]:
    active = [hook for hook in hooks if hook is not None]

    def _handle(event: ChatEvent) -> None:
        for hook in active:
            hook(event)

    return _handle


def build_training_status(thread_id: str, run_id: str | None = None) -> dict[str, Any]:
    thread_dir = safe_thread_dir(thread_id)
    if not thread_dir.exists():
        raise FileNotFoundError(f"Thread not found: {thread_id}")
    run_dir = _resolve_run_dir(thread_dir, run_id)
    state = _read_json(run_dir / PROGRESS_STATE_NAME)
    backend = _detect_backend(run_dir, state)
    selected_run_id = run_dir.name
    now = utc_now()
    started_at = state.get("started_at") or state.get("created_at") or _mtime_iso(run_dir)
    completed_at = state.get("completed_at")
    status = state.get("status") or _infer_status(run_dir)
    return {
        "schema": "jetlinks-training-status.v1",
        "thread_id": thread_id,
        "run_id": selected_run_id,
        "run_sequence": _run_sequence(thread_dir, selected_run_id),
        "is_latest": selected_run_id == _latest_run_id(thread_dir),
        "workflow": state.get("workflow") or "yolo_training_flow",
        "app_template_name": state.get("app_template_name"),
        "backend": backend,
        "status": status,
        "phase": state.get("phase") or _infer_phase(run_dir, status),
        "phase_label": state.get("phase_label") or _phase_label(state.get("phase"), status),
        "last_event": state.get("last_event"),
        "next_expected_phase": _next_expected_phase(state.get("phase"), status),
        "created_at": state.get("created_at") or _mtime_iso(run_dir),
        "started_at": started_at,
        "updated_at": now,
        "completed_at": completed_at,
        "elapsed_seconds": _elapsed_seconds(started_at, completed_at or now),
        "request": state.get("request") if isinstance(state.get("request"), dict) else {},
        "model_request": _model_request_status(state),
        "files": _files_status(thread_dir),
        "dataset": _dataset_status(run_dir),
        "annotation": _annotation_status(run_dir, state),
        "generation": _generation_status(run_dir, state),
        "training": _training_status(run_dir, backend, state),
        "evaluation": _evaluation_status(run_dir, backend),
        "resources": _resource_status(run_dir, backend),
        "paths": _paths_status(thread_dir, run_dir, backend),
        "errors": state.get("errors") if isinstance(state.get("errors"), list) else [],
        "warnings": state.get("warnings") if isinstance(state.get("warnings"), list) else [],
    }


def build_training_stream_status(thread_id: str, run_id: str | None = None) -> dict[str, Any]:
    status = build_training_status(thread_id, run_id=run_id)
    return {
        "thread_id": status.get("thread_id"),
        "run_id": status.get("run_id"),
        "run_sequence": status.get("run_sequence"),
        "is_latest": status.get("is_latest"),
        "backend": status.get("backend"),
        "status": status.get("status"),
        "phase": status.get("phase"),
        "last_event": status.get("last_event"),
        "next_expected_phase": status.get("next_expected_phase"),
        "created_at": status.get("created_at"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "completed_at": status.get("completed_at"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "request": status.get("request") if isinstance(status.get("request"), dict) else {},
        "model_request": status.get("model_request") if isinstance(status.get("model_request"), dict) else {},
        "files": status.get("files") if isinstance(status.get("files"), dict) else {},
        "dataset": status.get("dataset") if isinstance(status.get("dataset"), dict) else {},
        "info": _stream_info(status),
        "evaluation": status.get("evaluation") if isinstance(status.get("evaluation"), dict) else {},
        "resources": _stream_resources(status.get("resources")),
        "paths": status.get("paths") if isinstance(status.get("paths"), dict) else {},
        "errors": status.get("errors") if isinstance(status.get("errors"), list) else [],
        "warnings": status.get("warnings") if isinstance(status.get("warnings"), list) else [],
    }


def list_training_runs(thread_id: str) -> dict[str, Any]:
    thread_dir = safe_thread_dir(thread_id)
    if not thread_dir.exists():
        raise FileNotFoundError(f"Thread not found: {thread_id}")
    latest = _latest_run_id(thread_dir)
    runs = []
    for run_dir in _run_dirs(thread_dir):
        state = _read_json(run_dir / PROGRESS_STATE_NAME)
        status = state.get("status") or _infer_status(run_dir)
        runs.append(
            {
                "run_id": run_dir.name,
                "backend": _detect_backend(run_dir, state),
                "status": status,
                "phase": state.get("phase") or _infer_phase(run_dir, status),
                "phase_label": state.get("phase_label") or _phase_label(state.get("phase"), status),
                "created_at": state.get("created_at") or _mtime_iso(run_dir),
                "updated_at": state.get("updated_at") or _mtime_iso(run_dir),
                "completed_at": state.get("completed_at"),
                "is_latest": run_dir.name == latest,
                "run_dir": str(run_dir),
            }
        )
    return {"thread_id": thread_id, "latest_run_id": latest, "runs": runs}


def _resolve_run_dir(thread_dir: Path, run_id: str | None) -> Path:
    runs_root = thread_dir / "outputs" / "yolo_training_flow" / "runs"
    if run_id:
        target = runs_root / safe_run_id(run_id)
        if not target.is_dir():
            raise FileNotFoundError(f"Training run not found: {run_id}")
        return target
    current = _read_json(thread_dir / "workspace" / "current_training_run.json")
    current_run_id = str(current.get("run_id") or "").strip()
    if current_run_id and (runs_root / current_run_id).is_dir():
        return runs_root / current_run_id
    latest = _latest_run_id(thread_dir)
    if latest:
        return runs_root / latest
    raise FileNotFoundError(f"No training run found for thread: {thread_dir.name}")


def _run_dirs(thread_dir: Path) -> list[Path]:
    runs_root = thread_dir / "outputs" / "yolo_training_flow" / "runs"
    if not runs_root.is_dir():
        return []
    return sorted(
        (path for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("run-")),
        key=lambda item: item.stat().st_mtime,
    )


def _latest_run_id(thread_dir: Path) -> str | None:
    dirs = _run_dirs(thread_dir)
    return dirs[-1].name if dirs else None


def _run_sequence(thread_dir: Path, run_id: str) -> int:
    for index, run_dir in enumerate(_run_dirs(thread_dir), start=1):
        if run_dir.name == run_id:
            return index
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _detect_backend(run_dir: Path, state: dict[str, Any]) -> str:
    if state.get("backend"):
        return str(state["backend"])
    if (run_dir / "deimv2_training_run").exists():
        return "deimv2"
    if (run_dir / "training_run").exists():
        return "yolo"
    return "unknown"


def _infer_status(run_dir: Path) -> str:
    state = _read_json(run_dir / PROGRESS_STATE_NAME)
    if state.get("status"):
        return str(state["status"])
    for summary in _summary_candidates(run_dir):
        data = _read_json(summary)
        if data.get("status"):
            return str(data["status"])
    return "running"


def _infer_phase(run_dir: Path, status: Any) -> str:
    if status in {"completed", "failed"}:
        return str(status)
    backend = _detect_backend(run_dir, {})
    if _first_existing(_training_log_candidates(run_dir, backend)):
        return "training"
    if (run_dir / "prepared_data").exists():
        return "data_preparation_completed"
    return "data_preparation"


def _phase_label(phase: Any, status: Any) -> str:
    if status in {"completed", "failed"}:
        return str(status)
    return {
        "llm_request": "llm request running",
        "run_selected": "run selected",
        "data_preparation": "data preparation running",
        "data_preparation_completed": "data preparation completed",
        "synthetic_generation": "synthetic generation running",
        "annotation": "annotation running",
        "training": "training running",
    }.get(str(phase), "running")


def _next_expected_phase(phase: Any, status: Any) -> str | None:
    if status in {"completed", "failed"}:
        return None
    order = ["llm_request", "data_preparation", "synthetic_generation", "annotation", "training", "completed"]
    try:
        return order[order.index(str(phase)) + 1]
    except (ValueError, IndexError):
        return None


def _model_request_status(state: dict[str, Any]) -> dict[str, Any]:
    model_request = state.get("model_request")
    if not isinstance(model_request, dict):
        return {"current": None, "history": []}
    current = model_request.get("current")
    if isinstance(current, dict) and current.get("status") == "running" and current.get("started_at"):
        current = {**current, "elapsed_ms": int(_elapsed_seconds(str(current["started_at"]), utc_now()) * 1000)}
    return {
        "current": current if isinstance(current, dict) else None,
        "history": model_request.get("history") if isinstance(model_request.get("history"), list) else [],
    }


def _files_status(thread_dir: Path) -> dict[str, Any]:
    uploads = thread_dir / "uploads"
    files: dict[str, Any] = {"uploads_dir": str(uploads), "dataset": None, "image1": None, "image2": None, "models": []}
    for name in ("datasets.zip", "dataset.zip"):
        candidate = uploads / name
        if candidate.is_file():
            files["dataset"] = _file_info(candidate)
            break
    for key in ("image1", "image2"):
        candidate = uploads / f"{key}.zip"
        if candidate.is_file():
            files[key] = _file_info(candidate)
    model_dir = uploads / "models"
    if model_dir.is_dir():
        files["models"] = [_file_info(path) for path in sorted(model_dir.iterdir()) if path.is_file()]
    return files


def _dataset_status(run_dir: Path) -> dict[str, Any]:
    summary = _read_json(run_dir / "prepared_data" / "data_preparation_summary.json")
    if not summary:
        summary = _read_json(run_dir / "pipeline_work" / "data_preparation_summary.json")
    class_names = summary.get("class_names") or summary.get("labels_final") or []
    split_counts = summary.get("split_counts") if isinstance(summary.get("split_counts"), dict) else {}
    source_counts = summary.get("source_counts") if isinstance(summary.get("source_counts"), dict) else {}
    return {
        "status": "completed" if summary else "pending",
        "class_names": class_names if isinstance(class_names, list) else [],
        "num_images": summary.get("num_images") or sum(_int_or_none(v) or 0 for v in split_counts.values()),
        "num_categories": summary.get("num_categories") or (len(class_names) if isinstance(class_names, list) else None),
        "split_counts": split_counts,
        "source_counts": source_counts,
        "prepared_dataset": summary.get("prepared_dataset"),
        "dataset_yaml": summary.get("dataset_yaml"),
        "training_coco": summary.get("training_coco"),
        "training_root": summary.get("training_root"),
    }


def _annotation_status(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    uploaded = run_dir / "uploaded_dataset"
    real_total = _count_images(uploaded)
    real_coco = run_dir / "pipeline_work" / "real_coco.json"
    real_completed = _count_coco_images(real_coco) or _count_files(uploaded, "*_coco.json")
    synthetic_coco = run_dir / "pipeline_work" / "synthetic_coco.json"
    synthetic_completed = _count_coco_images(synthetic_coco) or _count_files(run_dir / "pipeline_work" / "synthetic_annotations", "*_coco.json")
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    prompts = request.get("annotation_prompts") or request.get("labels") or []
    prompts = prompts if isinstance(prompts, list) else []
    state_block = state.get("annotation") if isinstance(state.get("annotation"), dict) else {}
    return {
        "real": _merge_counter_state(
            _annotation_block("real_annotation", real_total, real_completed, prompts, real_coco),
            state_block,
        ),
        "synthetic": _annotation_block("synthetic_annotation", synthetic_completed, synthetic_completed, prompts, synthetic_coco),
    }


def _annotation_block(phase: str, total: int, completed: int, prompts: list[Any], coco_path: Path) -> dict[str, Any]:
    return {
        "status": _counter_status(completed, total),
        "phase": phase,
        "total": total,
        "completed": completed,
        "success": completed,
        "failed": max(0, total - completed) if total and completed < total else 0,
        "current_image": None,
        "prompts": prompts,
        "output_coco": str(coco_path) if coco_path.is_file() else None,
        "updated_at": _mtime_iso(coco_path) if coco_path.is_file() else None,
        "error": None,
    }


def _generation_status(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    summary = _read_json(run_dir / "prepared_data" / "data_preparation_summary.json")
    generated_dir = run_dir / "pipeline_work" / "synthetic_images"
    completed = _count_images(generated_dir)
    planned = _extract_planned_synthetic_count(run_dir, state, summary)
    block = {
        "status": "disabled" if planned == 0 else _counter_status(completed, planned),
        "phase": "synthetic_generation",
        "planned": planned,
        "completed": completed,
        "success": completed,
        "failed": 0,
        "fallback_used": bool(summary.get("synthetic_generation_fallback")),
        "provider": summary.get("synthetic_generation_provider"),
        "output_dir": str(generated_dir) if generated_dir.exists() else None,
        "updated_at": _mtime_iso(generated_dir) if generated_dir.exists() else None,
        "error": summary.get("synthetic_generation_error"),
    }
    state_block = state.get("generation") if isinstance(state.get("generation"), dict) else {}
    return _merge_counter_state(block, state_block)


def _training_status(run_dir: Path, backend: str, state: dict[str, Any]) -> dict[str, Any]:
    if backend == "deimv2":
        return _deimv2_training_status(run_dir, state)
    if backend == "yolo":
        return _yolo_training_status(run_dir, state)
    return {"status": "pending", "backend": backend}


def _deimv2_training_status(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    log_path = _first_existing(_training_log_candidates(run_dir, "deimv2"))
    total_epochs = _parse_deimv2_total_epochs(run_dir)
    latest = _parse_deimv2_train_log(log_path) if log_path else {}
    current_epoch = latest.get("epoch")
    step = latest.get("step")
    total_steps = latest.get("total_steps")
    progress = None
    if isinstance(total_epochs, int) and total_epochs > 0 and isinstance(current_epoch, int):
        step_ratio = (float(step or 0) / float(total_steps or 1)) if total_steps else 0.0
        progress = min(1.0, max(0.0, (current_epoch + step_ratio) / total_epochs))
    return {
        "status": "running" if latest else (state.get("training", {}) or {}).get("status", "pending"),
        "backend": "deimv2",
        "device": _selected_device(run_dir),
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "current_step": step,
        "total_steps": total_steps,
        "progress": progress,
        "lr": latest.get("lr"),
        "loss": latest.get("loss"),
        "latest_metrics": latest.get("metrics") or {},
        "best": latest.get("best") or {},
        "log_path": str(log_path) if log_path else None,
        "checkpoint": _best_checkpoint(run_dir),
    }


def _yolo_training_status(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    csv_path = _first_existing([*run_dir.glob("training_run/**/results.csv"), run_dir / "results.csv"])
    row = _last_csv_row(csv_path) if csv_path else {}
    epoch = _int_or_none(row.get("epoch"))
    return {
        "status": "running" if row else (state.get("training", {}) or {}).get("status", "pending"),
        "backend": "yolo",
        "device": _selected_device(run_dir),
        "current_epoch": epoch,
        "total_epochs": _parse_yolo_total_epochs(run_dir),
        "current_step": None,
        "total_steps": None,
        "progress": None,
        "lr": _float_or_none(row.get("lr/pg0") or row.get("lr0")),
        "loss": _float_or_none(row.get("train/box_loss")),
        "latest_metrics": {key: _float_or_none(value) for key, value in row.items() if key.startswith("metrics/")},
        "best": {},
        "results_csv": str(csv_path) if csv_path else None,
        "checkpoint": _best_checkpoint(run_dir),
    }


def _evaluation_status(run_dir: Path, backend: str) -> dict[str, Any]:
    for summary_path in _summary_candidates(run_dir):
        summary = _read_json(summary_path)
        metrics = summary.get("metrics") or summary.get("eval_results") or summary.get("evaluation") or {}
        if metrics:
            return {"status": "completed", "backend": backend, "metrics": metrics, "summary_path": str(summary_path)}
    return {"status": "pending", "backend": backend, "metrics": {}}


def _resource_status(run_dir: Path, backend: str) -> dict[str, Any]:
    selected_device = _selected_device(run_dir)
    return {
        "timestamp": utc_now(),
        "selected_device": selected_device,
        "accelerator_type": _accelerator_type(selected_device),
        "gpu": _nvidia_smi_status(),
        "npu": _npu_smi_status(),
        "host": _host_status(),
        "backend": backend,
    }


def _merge_counter_state(block: dict[str, Any], state_block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state_block, dict):
        return block
    merged = dict(block)
    for key in ("started_at", "completed_at", "updated_at", "current_image"):
        if state_block.get(key) is not None:
            merged[key] = state_block.get(key)
    if state_block.get("status") in {"pending", "running", "completed", "failed"}:
        merged["status"] = state_block["status"]
    return merged


def _stream_info(status: dict[str, Any]) -> dict[str, Any]:
    phase = str(status.get("phase") or "")
    annotation = status.get("annotation") if isinstance(status.get("annotation"), dict) else {}
    generation = status.get("generation") if isinstance(status.get("generation"), dict) else {}
    training = status.get("training") if isinstance(status.get("training"), dict) else {}
    evaluation = status.get("evaluation") if isinstance(status.get("evaluation"), dict) else {}
    model_request = status.get("model_request") if isinstance(status.get("model_request"), dict) else {}
    if isinstance(model_request.get("current"), dict):
        current = model_request["current"]
        return {
            "stage": "llm",
            "status": current.get("status") or "running",
            "metrics": {
                "purpose": current.get("purpose"),
                "model": current.get("model"),
                "elapsed_ms": current.get("elapsed_ms"),
            },
        }
    if phase == "annotation":
        real = annotation.get("real") if isinstance(annotation.get("real"), dict) else {}
        return {
            "stage": "annotation",
            "status": real.get("status") or "running",
            "metrics": {
                "annotated": real.get("completed") or 0,
                "total": real.get("total") or 0,
                "success": real.get("success") or 0,
                "failed": real.get("failed") or 0,
                "started": bool(real.get("started_at")),
                "completed": real.get("status") == "completed",
            },
        }
    if phase == "synthetic_generation":
        return {
            "stage": "generation",
            "status": generation.get("status") or "running",
            "metrics": {
                "generated": generation.get("completed") or 0,
                "planned": generation.get("planned") or 0,
                "success": generation.get("success") or 0,
                "failed": generation.get("failed") or 0,
                "started": bool(generation.get("started_at")),
                "completed": generation.get("status") in {"completed", "disabled"},
            },
        }
    if phase == "training" or training.get("status") == "running":
        return {
            "stage": "training",
            "status": training.get("status") or "running",
            "metrics": {
                "started": bool(training.get("started_at") or training.get("current_epoch") is not None),
                "completed": training.get("status") == "completed",
                "current_epoch": training.get("current_epoch"),
                "total_epochs": training.get("total_epochs"),
                "current_step": training.get("current_step"),
                "total_steps": training.get("total_steps"),
                "progress": training.get("progress"),
                "loss": training.get("loss"),
                "lr": training.get("lr"),
            },
        }
    if status.get("status") == "completed":
        return {
            "stage": "evaluation",
            "status": "complete",
            "metrics": evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {},
        }
    return {"stage": phase or "unknown", "status": status.get("status") or "running", "metrics": {}}


def _stream_resources(resources: Any) -> dict[str, Any]:
    data = resources if isinstance(resources, dict) else {}
    accelerator_type = str(data.get("accelerator_type") or _accelerator_type(data.get("selected_device")) or "unknown")
    result = {
        "timestamp": data.get("timestamp"),
        "backend": data.get("backend"),
        "selected_device": data.get("selected_device"),
        "accelerator_type": accelerator_type,
        "host": data.get("host") if isinstance(data.get("host"), dict) else {},
    }
    if accelerator_type == "npu":
        result["npu"] = data.get("npu") if isinstance(data.get("npu"), dict) else {"available": False, "devices": []}
    elif accelerator_type == "gpu":
        result["gpu"] = data.get("gpu") if isinstance(data.get("gpu"), dict) else {"available": False, "devices": []}
    elif accelerator_type == "cpu":
        result["cpu"] = result["host"]
    else:
        gpu = data.get("gpu") if isinstance(data.get("gpu"), dict) else {}
        npu = data.get("npu") if isinstance(data.get("npu"), dict) else {}
        if npu.get("available"):
            result["accelerator_type"] = "npu"
            result["npu"] = npu
        elif gpu.get("available"):
            result["accelerator_type"] = "gpu"
            result["gpu"] = gpu
        else:
            result["accelerator_type"] = "cpu"
            result["cpu"] = result["host"]
    return result


def _accelerator_type(selected_device: Any) -> str:
    text = str(selected_device or "").lower()
    if "npu" in text or "ascend" in text:
        return "npu"
    if "cuda" in text or "gpu" in text:
        return "gpu"
    if "cpu" in text:
        return "cpu"
    return "unknown"


def _paths_status(thread_dir: Path, run_dir: Path, backend: str) -> dict[str, Any]:
    training_dir = run_dir / ("deimv2_training_run" if backend == "deimv2" else "training_run")
    return {
        "thread_dir": str(thread_dir),
        "run_dir": str(run_dir),
        "workspace": str(run_dir / "workspace"),
        "outputs": str(run_dir),
        "logs": str(training_dir / "logs") if training_dir.exists() else None,
        "progress_state": str(run_dir / PROGRESS_STATE_NAME),
    }


def _summary_candidates(run_dir: Path) -> list[Path]:
    return [
        run_dir / "run_summary.json",
        run_dir / "deimv2_training_run" / "run_summary.json",
        run_dir / "training_run" / "run_summary.json",
        *run_dir.glob("training_run/**/run_summary.json"),
    ]


def _training_log_candidates(run_dir: Path, backend: str) -> list[Path]:
    if backend == "deimv2":
        return [run_dir / "deimv2_training_run" / "logs" / "train.log", run_dir / "logs" / "deimv2-training-train.log"]
    return [*run_dir.glob("training_run/**/*.log"), run_dir / "logs" / "yolo-training-stdout.txt"]


def _parse_deimv2_train_log(path: Path) -> dict[str, Any]:
    text = _tail_text(path, 256_000)
    latest: dict[str, Any] = {}
    for match in re.finditer(r"Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)/(\d+)\].*?(?:lr:\s*([0-9.eE+-]+))?.*?(?:loss:\s*([0-9.eE+-]+))?", text):
        latest.update(
            {
                "epoch": int(match.group(1)),
                "step": int(match.group(2)),
                "total_steps": int(match.group(3)),
                "lr": _float_or_none(match.group(4)),
                "loss": _float_or_none(match.group(5)),
            }
        )
    best_matches = list(re.finditer(r"best_stat:\s*(\{.*?\})", text))
    if best_matches:
        best_text = best_matches[-1].group(1).replace("'", '"')
        try:
            latest["best"] = json.loads(best_text)
        except json.JSONDecodeError:
            latest["best"] = {"raw": best_matches[-1].group(1)}
    metrics = _parse_last_coco_metrics(text)
    if metrics:
        latest["metrics"] = metrics
    return latest


def _parse_last_coco_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    patterns = {
        "mAP50_95": r"Average Precision\s+\(AP\).*?IoU=0\.50:0\.95.*?area=\s*all.*?=\s*([0-9.]+)",
        "mAP50": r"Average Precision\s+\(AP\).*?IoU=0\.50\s+.*?area=\s*all.*?=\s*([0-9.]+)",
        "mAP75": r"Average Precision\s+\(AP\).*?IoU=0\.75\s+.*?area=\s*all.*?=\s*([0-9.]+)",
        "AR100": r"Average Recall\s+\(AR\).*?IoU=0\.50:0\.95.*?maxDets=100\s*\]\s*=\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            metrics[key] = float(matches[-1])
    return metrics


def _parse_deimv2_total_epochs(run_dir: Path) -> int | None:
    for path in [run_dir / "deimv2_training_run" / "configs" / "train.yml", run_dir / "workspace" / "training_config.json"]:
        if not path.is_file():
            continue
        text = _tail_text(path, 200_000)
        match = re.search(r"^\s*epoch(?:e)?s\s*:\s*(\d+)", text, re.MULTILINE)
        if match:
            return int(match.group(1))
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        training = data.get("training") if isinstance(data, dict) else {}
        if isinstance(training, dict) and training.get("epochs"):
            return _int_or_none(training.get("epochs"))
    return None


def _parse_yolo_total_epochs(run_dir: Path) -> int | None:
    for path in run_dir.glob("training_run/**/args.yaml"):
        text = _tail_text(path, 100_000)
        match = re.search(r"^\s*epochs\s*:\s*(\d+)", text, re.MULTILINE)
        if match:
            return int(match.group(1))
    return None


def _nvidia_smi_status() -> dict[str, Any]:
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.total,memory.used,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=3,
    )
    if result["returncode"] != 0:
        return {"available": False, "error": result["stderr"] or result["stdout"], "devices": []}
    devices = []
    for line in result["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        total = _float_or_none(parts[3])
        used = _float_or_none(parts[4])
        devices.append(
            {
                "index": _int_or_none(parts[0]),
                "name": parts[1],
                "utilization_percent": _float_or_none(parts[2]),
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_free_mb": (total - used) if total is not None and used is not None else None,
                "temperature_c": _float_or_none(parts[5]),
                "power_w": _float_or_none(parts[6]),
            }
        )
    return {"available": True, "devices": devices}


def _npu_smi_status() -> dict[str, Any]:
    result = _run_command(["npu-smi", "info"], timeout=3)
    if result["returncode"] != 0:
        return {"available": False, "error": result["stderr"] or result["stdout"], "devices": []}
    return {"available": True, "devices": _parse_npu_smi_info(result["stdout"]), "raw": result["stdout"][-4000:]}


def _parse_npu_smi_info(text: str) -> list[dict[str, Any]]:
    devices: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        numbers = re.findall(r"(?<![\w.])(\d+)(?![\w.])", line)
        if not numbers:
            continue
        index = _int_or_none(numbers[0])
        if index is None or index > 255:
            continue
        item = devices.setdefault(index, {"index": index})
        mem = re.search(r"(\d+)\s*/\s*(\d+)", line)
        if mem:
            item["memory_used_mb"] = int(mem.group(1))
            item["memory_total_mb"] = int(mem.group(2))
            item["memory_free_mb"] = int(mem.group(2)) - int(mem.group(1))
        percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", line)
        if percents:
            item["utilization_percent"] = _float_or_none(percents[-1])
    return list(devices.values())


def _host_status() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "memory": {
                "total_mb": round(vm.total / 1024 / 1024, 2),
                "used_mb": round(vm.used / 1024 / 1024, 2),
                "percent": vm.percent,
            },
        }
    except Exception:
        return {"cpu_percent": None, "memory": {}}


def _run_command(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": 1, "stdout": "", "stderr": str(exc)}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"name": path.name, "path": str(path), "size": stat.st_size, "mtime": _mtime_iso(path)}


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_file():
        return 1 if root.suffix.lower() in IMAGE_EXTS else 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob(pattern) if path.is_file())


def _count_coco_images(path: Path) -> int:
    data = _read_json(path)
    images = data.get("images") if isinstance(data, dict) else None
    return len(images) if isinstance(images, list) else 0


def _counter_status(completed: int, total: int | None) -> str:
    if not total:
        return "pending" if completed == 0 else "completed"
    if completed >= total:
        return "completed"
    return "running" if completed > 0 else "pending"


def _extract_planned_synthetic_count(run_dir: Path, state: dict[str, Any], summary: dict[str, Any]) -> int:
    for value in [summary.get("synthetic_images_requested"), summary.get("synthetic_images_planned"), summary.get("max_synthetic")]:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    plan = _read_json(run_dir / "pipeline_work" / "synthetic_plan.json")
    for value in [plan.get("recommended_synthetic_count"), plan.get("count")]:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    return _int_or_none(request.get("max_synthetic_images")) or 0


def _selected_device(run_dir: Path) -> str | None:
    for path in [run_dir / "workspace" / "training_config.json", run_dir / "deimv2_training_run" / "configs" / "train.yml"]:
        if not path.is_file():
            continue
        text = _tail_text(path, 200_000)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {}
        training = data.get("training") if isinstance(data, dict) else {}
        if isinstance(training, dict) and training.get("device"):
            return str(training["device"])
        match = re.search(r"^\s*device\s*:\s*['\"]?([^'\"\n]+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def _best_checkpoint(run_dir: Path) -> str | None:
    found = _first_existing(
        [
            run_dir / "deimv2_training_run" / "runs" / "train" / "best_stg1.pth",
            run_dir / "training_run" / "train" / "weights" / "best.pt",
            *run_dir.glob("training_run/**/weights/best.pt"),
        ]
    )
    return str(found) if found else None


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _last_csv_row(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return {}
    return rows[-1] if rows else {}


def _tail_text(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _mtime_iso(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat().replace("+00:00", "Z")
    except OSError:
        return None


def _elapsed_seconds(started_at: str | None, ended_at: str | None) -> float:
    if not started_at or not ended_at:
        return 0.0
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return round(max(0.0, (end - start).total_seconds()), 3)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _llm_phase_label(purpose: str) -> str:
    return {
        "workflow_intent_router": "llm intent parsing",
        "workflow_request_spec": "llm request spec generation",
        "workflow_model_managed_intent_spec": "llm yolo spec generation",
        "workflow_deimv2_model_managed_intent_spec": "llm deimv2 spec generation",
        "workflow_deimv2_dataset_aware_training_spec": "llm deimv2 training config generation",
        "workflow_dataset_aware_training_spec": "llm yolo training config generation",
        "workflow_final_reply": "llm final summary generation",
    }.get(purpose, "llm request running")
