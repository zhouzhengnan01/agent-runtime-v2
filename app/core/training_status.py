from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from app.core.artifacts import ArtifactStore
from app.core.http_training_jobs import (
    ACTIVE_HTTP_TRAINING_STATUSES,
    TERMINAL_HTTP_TRAINING_STATUSES,
    read_current_http_training_job_marker,
)
from app.core.training_queue import training_queue
from app.core.training_annotation_previews import build_original_annotation_resources
from app.core.training_artifact_updates import training_status_artifact_paths
from app.schemas import ChatEvent


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROGRESS_STATE_NAME = "progress_state.json"
DEIMV2_FAILURE_LOG_PATTERNS = (
    re.compile(r"OpenBLAS error:\s*Memory allocation still failed after \d+ retries, giving up\.", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
    re.compile(r"^\s*(?:[\w.]+)?(?:RuntimeError|ValueError|AssertionError|ImportError|ModuleNotFoundError|FileNotFoundError|MemoryError|OSError|TypeError|KeyError|IndexError|AttributeError|NotImplementedError|Exception):\s*.+$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"^\s*(?:error|failed|failure):\s*.+$", re.IGNORECASE | re.MULTILINE),
)


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
        elif event.type == "data_preparation.synthetic_generation.started":
            self._mark_stage_running(state, "generation", "synthetic_generation")
            generation = state.setdefault("generation", {})
            if isinstance(generation, dict):
                self._merge_generation_event(generation, event.data)
            state["phase"] = "synthetic_generation"
            state["phase_label"] = "synthetic generation running"
        elif event.type == "data_preparation.synthetic_generation.finished":
            generation = state.setdefault("generation", {})
            if isinstance(generation, dict):
                self._merge_generation_event(generation, event.data)
                generation["completed_at"] = event.data.get("timestamp") or now
                generation["updated_at"] = now
            state["phase"] = "synthetic_generation"
            state["phase_label"] = f"synthetic generation {event.data.get('status') or 'finished'}"
        elif event.type == "data_preparation.image_annotated":
            self._mark_stage_running(state, "annotation", "annotation")
            self._increment_counter(state, "annotation", "completed")
            state["phase"] = "annotation"
            state["phase_label"] = "annotation running"
        elif event.type == "data_preparation.artifacts_reused":
            real_images = int(event.data.get("real_images") or 0)
            synthetic_images = int(event.data.get("synthetic_images") or 0)
            synthetic_reused = bool(event.data.get("synthetic_reused"))
            now = event.data.get("timestamp") or utc_now()
            state["phase"] = "data_reuse"
            state["phase_label"] = "data preparation artifacts reused"
            state["data_reuse"] = {
                "status": "completed",
                "source_run_id": event.data.get("source_run_id"),
                "target_run_id": event.data.get("target_run_id"),
                "real_images": real_images,
                "synthetic_images": synthetic_images,
                "annotations": int(event.data.get("annotations") or 0),
                "copied_files": int(event.data.get("copied_files") or 0),
                "reuse_level": event.data.get("reuse_level") or ("real_and_synthetic" if synthetic_reused else "real_only"),
                "synthetic_reused": synthetic_reused,
                "completed_at": now,
            }
            state["annotation"] = {
                "status": "completed",
                "phase": "annotation",
                "started_at": now,
                "completed_at": now,
                "updated_at": now,
                "reused": True,
            }
            if synthetic_reused:
                state["generation"] = {
                    "status": "completed",
                    "phase": "synthetic_generation",
                    "planned": synthetic_images,
                    "completed": synthetic_images,
                    "success": synthetic_images,
                    "failed": 0,
                    "started_at": now,
                    "completed_at": now,
                    "updated_at": now,
                    "reused": True,
                }
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
            state["phase"] = "training_completed"
            state["phase_label"] = "training completed"
            training = state.setdefault("training", {})
            if isinstance(training, dict):
                training["status"] = "completed"
                training["completed_at"] = utc_now()
                training["updated_at"] = utc_now()

    def _increment_counter(self, state: dict[str, Any], block: str, field: str) -> None:
        target = state.setdefault(block, {})
        if isinstance(target, dict):
            target[field] = int(target.get(field) or 0) + 1
            target["updated_at"] = utc_now()

    def _merge_generation_event(self, generation: dict[str, Any], data: dict[str, Any]) -> None:
        for key in ("status", "planned", "completed", "success", "failed", "error", "fallback_used", "reason", "reason_code"):
            if data.get(key) is not None:
                generation[key] = data.get(key)
        generation["started_at"] = generation.get("started_at") or data.get("timestamp") or utc_now()
        generation["updated_at"] = data.get("timestamp") or utc_now()

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
        if isinstance(target, dict) and target.get("started_at") and target.get("status") in {None, "pending", "running"}:
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
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        last_error: PermissionError | None = None
        for attempt in range(10):
            try:
                tmp.replace(self.path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        try:
            tmp.unlink(missing_ok=True)
        finally:
            if last_error is not None:
                raise last_error


def chain_event_hooks(*hooks: Callable[[ChatEvent], None] | None) -> Callable[[ChatEvent], None]:
    active = [hook for hook in hooks if hook is not None]

    def _handle(event: ChatEvent) -> None:
        for hook in active:
            try:
                hook(event)
            except Exception:
                continue

    return _handle


def build_training_status(thread_id: str, run_id: str | None = None) -> dict[str, Any]:
    thread_dir = safe_thread_dir(thread_id)
    if not thread_dir.exists():
        raise FileNotFoundError(f"Thread not found: {thread_id}")
    run_dir = _resolve_run_dir(thread_dir, run_id)
    return _build_run_status(thread_dir, run_dir)


def _build_run_status(thread_dir: Path, run_dir: Path) -> dict[str, Any]:
    thread_id = thread_dir.name
    state = _read_json(run_dir / PROGRESS_STATE_NAME)
    backend = _detect_backend(run_dir, state)
    selected_run_id = run_dir.name
    now = utc_now()
    started_at = state.get("started_at") or state.get("created_at") or _mtime_iso(run_dir)
    completed_at = state.get("completed_at")
    training = _training_status(run_dir, backend, state)
    evaluation = _evaluation_status(run_dir, backend)
    status = state.get("status") or _infer_status(run_dir)
    phase = state.get("phase") or _infer_phase(run_dir, status)
    status, phase, completed_at = _normalize_run_phase_status(status, phase, completed_at, training, evaluation)
    errors = state.get("errors") if isinstance(state.get("errors"), list) else []
    if isinstance(training, dict) and training.get("status") == "failed":
        status = "failed"
        phase = "failed"
        completed_at = completed_at or now
        training_error = training.get("error")
        if training_error and not errors:
            errors = [{"message": str(training_error), "timestamp": now}]
    generation = _generation_status(run_dir, state)
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
        "phase": phase,
        "phase_label": (
            state.get("phase_label")
            if state.get("phase") == phase and state.get("phase_label")
            else _phase_label(phase, status)
        ),
        "last_event": state.get("last_event"),
        "next_expected_phase": _next_expected_phase(phase, status),
        "created_at": state.get("created_at") or _mtime_iso(run_dir),
        "started_at": started_at,
        "updated_at": now,
        "completed_at": completed_at,
        "elapsed_seconds": _elapsed_seconds(started_at, completed_at or now),
        "request": state.get("request") if isinstance(state.get("request"), dict) else {},
        "model_request": _model_request_status(state),
        "data_reuse": state.get("data_reuse") if isinstance(state.get("data_reuse"), dict) else {},
        "files": _files_status(thread_dir),
        "dataset": _dataset_status(run_dir),
        "annotation": _annotation_status(run_dir, state, generation),
        "generation": generation,
        "training": training,
        "evaluation": evaluation,
        "resources": _resource_status(run_dir, backend),
        "paths": _paths_status(thread_dir, run_dir, backend),
        "errors": errors,
        "warnings": state.get("warnings") if isinstance(state.get("warnings"), list) else [],
    }


def build_training_task_status(thread_id: str, run_id: str | None = None) -> dict[str, Any]:
    thread_dir = safe_thread_dir(thread_id)
    if not thread_dir.exists():
        raise FileNotFoundError(f"Thread not found: {thread_id}")
    job_marker = read_current_http_training_job_marker(
        thread_id,
        require_active=False,
        allow_previous_instance=False,
    )
    latest_run_id = _latest_run_id(thread_dir)
    if not latest_run_id:
        if run_id is not None or job_marker is None:
            raise FileNotFoundError(f"No training run found for thread: {thread_id}")
        updated_at = job_marker.get("updated_at") or utc_now()
        return {
            "schema": "jetlinks-training-task-status.v1",
            "thread_id": thread_id,
            "status": None,
            "phase": None,
            "phase_label": None,
            "total_runs": 0,
            "latest_run_id": None,
            "current_run_id": None,
            "updated_at": updated_at,
            "queue": training_queue.snapshot(),
            "thread_task_status": _build_thread_task_status(job_marker),
            "task_snapshot": {
                "schema": "jetlinks-training-task-snapshot.v1",
                "thread_id": thread_id,
                "total_runs": 0,
                "latest_run_id": None,
                "selected_run_id": None,
                "current_run_id": None,
                "current_run": None,
                "status_counts": {},
                "history_runs": [],
                "updated_at": updated_at,
            },
        }
    selected_run_dir = _resolve_run_dir(thread_dir, run_id)
    current_status = _build_run_status(
        thread_dir,
        thread_dir / "outputs" / "yolo_training_flow" / "runs" / latest_run_id,
    )
    task_snapshot = _build_task_snapshot(
        thread_dir,
        current_status,
        selected_run_id=selected_run_dir.name,
    )
    current_run = task_snapshot.get("current_run") if isinstance(task_snapshot.get("current_run"), dict) else {}
    thread_task_status = _build_thread_task_status(
        job_marker,
        fallback_status=str(current_run.get("status") or ""),
    )
    return {
        "schema": "jetlinks-training-task-status.v1",
        "thread_id": thread_id,
        "status": current_run.get("status"),
        "phase": current_run.get("phase"),
        "phase_label": current_run.get("phase_label"),
        "total_runs": task_snapshot.get("total_runs"),
        "latest_run_id": task_snapshot.get("latest_run_id"),
        "current_run_id": task_snapshot.get("current_run_id"),
        "updated_at": task_snapshot.get("updated_at"),
        "queue": training_queue.snapshot(),
        "thread_task_status": thread_task_status,
        "task_snapshot": task_snapshot,
    }


def _build_thread_task_status(
    marker: dict[str, Any] | None,
    *,
    fallback_status: str = "",
) -> dict[str, Any]:
    job_status = str(marker.get("status") or "").strip() if marker else ""
    thread_id = str(marker.get("thread_id") or "").strip() if marker else ""
    live_queue_state = training_queue.thread_state(thread_id) if thread_id else None
    queue_status = (
        live_queue_state.get("queue_status")
        if isinstance(live_queue_state, dict)
        else marker.get("queue_status") if marker else None
    )
    queue_position = (
        live_queue_state.get("queue_position")
        if isinstance(live_queue_state, dict)
        else marker.get("queue_position") if marker else None
    )
    has_active_job = job_status in ACTIVE_HTTP_TRAINING_STATUSES
    if has_active_job:
        status = "queued" if job_status == "queued" else "running"
        lifecycle = "active"
    elif job_status in TERMINAL_HTTP_TRAINING_STATUSES:
        status = job_status
        lifecycle = "terminal"
    else:
        normalized_fallback = fallback_status.strip()
        status = "running" if normalized_fallback == "cancelling" else normalized_fallback or None
        lifecycle = "active" if status in {"queued", "running"} else "terminal"
    result = {
        "schema": "jetlinks-training-thread-status.v1",
        "status": status,
        "lifecycle": lifecycle,
        "has_active_job": has_active_job,
        "active_job_id": marker.get("job_id") if has_active_job and marker else None,
        "active_job_status": status if has_active_job else None,
    }
    if queue_status is not None:
        result["queue_status"] = queue_status
        result["queue_position"] = queue_position
    return result


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
        "phase_label": status.get("phase_label"),
        "last_event": status.get("last_event"),
        "next_expected_phase": status.get("next_expected_phase"),
        "created_at": status.get("created_at"),
        "started_at": status.get("started_at"),
        "updated_at": status.get("updated_at"),
        "completed_at": status.get("completed_at"),
        "elapsed_seconds": status.get("elapsed_seconds"),
        "request": status.get("request") if isinstance(status.get("request"), dict) else {},
        "model_request": status.get("model_request") if isinstance(status.get("model_request"), dict) else {},
        "data_reuse": status.get("data_reuse") if isinstance(status.get("data_reuse"), dict) else {},
        "files": status.get("files") if isinstance(status.get("files"), dict) else {},
        "dataset": status.get("dataset") if isinstance(status.get("dataset"), dict) else {},
        "generation": status.get("generation") if isinstance(status.get("generation"), dict) else {},
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


def _build_task_snapshot(
    thread_dir: Path,
    current_status: dict[str, Any],
    *,
    selected_run_id: str,
) -> dict[str, Any]:
    run_dirs = _run_dirs(thread_dir)
    latest_run_dir = run_dirs[-1] if run_dirs else None
    latest_run_id = latest_run_dir.name if latest_run_dir is not None else None
    runs: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        status = current_status if run_dir.name == latest_run_id else _build_run_status(thread_dir, run_dir)
        runs.append(_build_run_snapshot(thread_dir, status))

    status_counts: dict[str, int] = {}
    for run in runs:
        run_status = str(run.get("status") or "unknown")
        status_counts[run_status] = status_counts.get(run_status, 0) + 1

    current_run = runs[-1] if runs else None

    return {
        "schema": "jetlinks-training-task-snapshot.v1",
        "thread_id": thread_dir.name,
        "total_runs": len(run_dirs),
        "latest_run_id": latest_run_id,
        "selected_run_id": selected_run_id,
        "current_run_id": latest_run_id,
        "current_run": current_run,
        "status_counts": status_counts,
        "history_runs": runs,
        "updated_at": utc_now(),
    }


def _build_run_snapshot(thread_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    artifact_items = _snapshot_artifacts(thread_dir, status)
    return {
        **status,
        "has_artifacts": bool(artifact_items),
        "artifact_count": len(artifact_items),
        "artifacts_url": f"/api/artifacts/{thread_dir.name}",
        "artifacts": artifact_items,
    }


def _snapshot_artifacts(thread_dir: Path, current_status: dict[str, Any]) -> list[dict[str, Any]]:
    if not current_status:
        return []
    store = ArtifactStore(root_dir=thread_dir.parent)
    run_id = str(current_status.get("run_id") or "")
    run_sequence = _int_or_none(current_status.get("run_sequence")) or _run_sequence(thread_dir, run_id)
    artifacts: list[dict[str, Any]] = []
    for path in training_status_artifact_paths(current_status):
        try:
            artifact = store.to_artifact_ref(thread_dir.name, path).model_dump()
        except (OSError, ValueError):
            continue
        phase, phase_label = _artifact_phase(path, current_status)
        item = {
            **artifact,
            "run_id": run_id,
            "run_sequence": run_sequence,
            "phase": phase,
            "phase_label": phase_label,
        }
        try:
            annotation_resources = build_original_annotation_resources(
                thread_id=thread_dir.name,
                coco_path=path,
                coco_artifact=artifact,
                artifact_store=store,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            annotation_resources = None
        if annotation_resources is not None:
            original = annotation_resources.get("original")
            image = original.get("image") if isinstance(original, dict) else None
            image_download_url = image.get("download_url") if isinstance(image, dict) else None
            if image_download_url:
                item["image_download_url"] = image_download_url
        artifacts.append(item)
    return artifacts


def _artifact_phase(path: Path, status: dict[str, Any]) -> tuple[str, str]:
    annotation = status.get("annotation") if isinstance(status.get("annotation"), dict) else {}
    generation = status.get("generation") if isinstance(status.get("generation"), dict) else {}
    training = status.get("training") if isinstance(status.get("training"), dict) else {}
    real = annotation.get("real") if isinstance(annotation.get("real"), dict) else {}
    synthetic = annotation.get("synthetic") if isinstance(annotation.get("synthetic"), dict) else {}

    if _same_artifact_path(path, training.get("checkpoint")) or path.suffix.lower() in {
        ".onnx",
        ".pt",
        ".pth",
    }:
        return "training", "model training"
    if path.name in {"run_summary.json", "training_summary.json"}:
        return "training", "model training"
    if _same_artifact_path(path, synthetic.get("output_coco")):
        return "synthetic_annotation", "synthetic image annotation"

    generation_dir = _artifact_path_or_none(generation.get("output_dir"))
    if generation_dir is not None and _path_is_within(path, generation_dir):
        if path.suffix.lower() == ".json":
            return "synthetic_annotation", "synthetic image annotation"
        return "synthetic_generation", "synthetic image generation"

    if _same_artifact_path(path, real.get("output_coco")):
        return "real_annotation", "real image annotation"
    if path.suffix.lower() == ".json" and "coco" in path.name.lower():
        return "real_annotation", "real image annotation"
    return "data_preparation", "data preparation"


def _same_artifact_path(path: Path, value: Any) -> bool:
    other = _artifact_path_or_none(value)
    if other is None:
        return False
    try:
        return path.resolve() == other.resolve()
    except OSError:
        return False


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _artifact_path_or_none(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value)).expanduser()


def _normalize_run_phase_status(
    status: str,
    phase: str,
    completed_at: Any,
    training: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[str, str, Any]:
    if status == "failed" or phase == "failed":
        return "failed", "failed", completed_at
    if status == "completed" or phase == "completed":
        return "completed", "completed", completed_at
    if _is_training_completed(training):
        if evaluation.get("status") == "completed":
            return status or "running", "evaluation", completed_at
        return status or "running", "training_completed", completed_at
    return status or "running", phase or "run_selected", completed_at


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
        "data_reuse": "data preparation artifacts reused",
        "data_preparation_completed": "data preparation completed",
        "synthetic_generation": "synthetic generation running",
        "annotation": "annotation running",
        "training": "training running",
        "training_completed": "training completed",
        "evaluation": "evaluation running",
    }.get(str(phase), "running")


def _next_expected_phase(phase: Any, status: Any) -> str | None:
    if status in {"completed", "failed"}:
        return None
    order = ["llm_request", "data_preparation", "data_reuse", "synthetic_generation", "annotation", "training", "training_completed", "evaluation", "completed"]
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


def _annotation_status(
    run_dir: Path,
    state: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    uploaded = run_dir / "uploaded_dataset"
    real_total = _count_images(uploaded)
    real_coco = run_dir / "pipeline_work" / "real_coco.json"
    real_completed = _count_coco_images(real_coco) or _count_files(uploaded, "*_coco.json")
    synthetic_coco = run_dir / "pipeline_work" / "synthetic_coco.json"
    synthetic_completed = _count_coco_images(synthetic_coco) or _count_files(run_dir / "pipeline_work" / "synthetic_annotations", "*_coco.json")
    synthetic_generated = _count_images(run_dir / "pipeline_work" / "synthetic_images")
    summary = _read_json(run_dir / "prepared_data" / "data_preparation_summary.json")
    synthetic_generation_started = bool(synthetic_generated or synthetic_completed or generation.get("started_at"))
    synthetic_planned = _extract_planned_synthetic_count(run_dir, state, summary) if synthetic_generation_started else 0
    generation_status = str(generation.get("status") or "pending")
    generation_terminal = generation_status in {
        "completed",
        "failed",
        "timeout",
        "skipped",
        "disabled",
    }
    synthetic_total = max(
        synthetic_completed,
        synthetic_generated,
        0 if generation_terminal else synthetic_planned,
    )
    request = state.get("request") if isinstance(state.get("request"), dict) else {}
    prompts = request.get("annotation_prompts") or request.get("labels") or []
    prompts = prompts if isinstance(prompts, list) else []
    state_block = state.get("annotation") if isinstance(state.get("annotation"), dict) else {}
    synthetic_block = _annotation_block(
        "synthetic_annotation",
        synthetic_total,
        synthetic_completed,
        prompts,
        synthetic_coco,
    )
    if generation_terminal and synthetic_generated == 0 and synthetic_completed == 0:
        synthetic_block = _skipped_synthetic_annotation(generation)
        synthetic_block["prompts"] = prompts
    return {
        "real": _merge_counter_state(
            _annotation_block("real_annotation", real_total, real_completed, prompts, real_coco),
            state_block,
        ),
        "synthetic": synthetic_block,
    }


def _annotation_block(phase: str, total: int, completed: int, prompts: list[Any], coco_path: Path) -> dict[str, Any]:
    return {
        "status": _counter_status(completed, total),
        "phase": phase,
        "total": total,
        "completed": completed,
        "success": completed,
        "failed": 0,
        "current_image": None,
        "prompts": prompts,
        "output_coco": str(coco_path) if coco_path.is_file() else None,
        "updated_at": _mtime_iso(coco_path) if coco_path.is_file() else None,
        "error": None,
    }


def _skipped_synthetic_annotation(generation: dict[str, Any]) -> dict[str, Any]:
    generation_status = str(generation.get("status") or "disabled")
    reason_code = str(generation.get("reason_code") or "").strip()
    reason = str(generation.get("reason") or "").strip()
    if not reason_code:
        reason_code = {
            "timeout": "synthetic_generation_timeout",
            "failed": "synthetic_generation_failed",
            "completed": "no_synthetic_images_generated",
        }.get(generation_status, "synthetic_generation_skipped")
    if not reason:
        reason = {
            "timeout": "Synthetic annotation skipped because synthetic generation timed out.",
            "failed": "Synthetic annotation skipped because synthetic generation failed.",
            "completed": "Synthetic annotation skipped because no synthetic images were generated.",
        }.get(generation_status, "Synthetic annotation skipped because synthetic generation was skipped.")
    return {
        "status": "skipped",
        "phase": "synthetic_annotation",
        "total": 0,
        "completed": 0,
        "success": 0,
        "failed": 0,
        "current_image": None,
        "prompts": [],
        "output_coco": None,
        "updated_at": generation.get("completed_at") or generation.get("updated_at"),
        "error": None,
        "started_at": None,
        "completed_at": generation.get("completed_at") or generation.get("updated_at"),
        "reason": reason,
        "reason_code": reason_code,
        "generation_status": generation_status,
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
    failure_error = latest.get("failure_error")
    current_epoch = latest.get("epoch")
    step = latest.get("step")
    total_steps = latest.get("total_steps")
    progress = None
    if isinstance(total_epochs, int) and total_epochs > 0 and isinstance(current_epoch, int):
        step_ratio = (float(step or 0) / float(total_steps or 1)) if total_steps else 0.0
        progress = min(1.0, max(0.0, (current_epoch + step_ratio) / total_epochs))
    state_training = state.get("training", {}) if isinstance(state.get("training"), dict) else {}
    checkpoint = _best_checkpoint(run_dir)
    completed = (
        state_training.get("status") == "completed"
        or _has_completed_summary(run_dir)
        or bool(checkpoint and _has_training_summary(run_dir))
    )
    if completed:
        progress = 1.0 if progress is None or progress >= 0.95 else progress
    return {
        "status": "failed" if failure_error else ("completed" if completed else ("running" if latest else state_training.get("status", "pending"))),
        "backend": "deimv2",
        "device": _selected_device(run_dir),
        "current_epoch": current_epoch,
        "current_epoch_display": (current_epoch + 1) if isinstance(current_epoch, int) else None,
        "total_epochs": total_epochs,
        "current_step": step,
        "total_steps": total_steps,
        "progress": progress,
        "lr": latest.get("lr"),
        "loss": latest.get("loss"),
        "latest_metrics": latest.get("metrics") or {},
        "best": latest.get("best") or {},
        "log_path": str(log_path) if log_path else None,
        "checkpoint": checkpoint,
        "completed": completed,
        "error": failure_error,
    }


def _yolo_training_status(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    csv_path = _first_existing([*run_dir.glob("training_run/**/results.csv"), run_dir / "results.csv"])
    row = _last_csv_row(csv_path) if csv_path else {}
    epoch = _int_or_none(row.get("epoch"))
    total_epochs = _parse_yolo_total_epochs(run_dir)
    state_training = state.get("training", {}) if isinstance(state.get("training"), dict) else {}
    checkpoint = _best_checkpoint(run_dir)
    completed = (
        state_training.get("status") == "completed"
        or _has_completed_summary(run_dir)
        or bool(checkpoint and csv_path)
    )
    progress = None
    if isinstance(epoch, int) and isinstance(total_epochs, int) and total_epochs > 0:
        progress = min(1.0, max(0.0, float(epoch + 1) / float(total_epochs)))
    if completed:
        progress = 1.0 if progress is None or progress >= 0.95 else progress
    return {
        "status": "completed" if completed else ("running" if row else state_training.get("status", "pending")),
        "backend": "yolo",
        "device": _selected_device(run_dir),
        "current_epoch": epoch,
        "current_epoch_display": (epoch + 1) if isinstance(epoch, int) else None,
        "total_epochs": total_epochs,
        "current_step": None,
        "total_steps": None,
        "progress": progress,
        "lr": _float_or_none(row.get("lr/pg0") or row.get("lr0")),
        "loss": _float_or_none(row.get("train/box_loss")),
        "latest_metrics": {key: _float_or_none(value) for key, value in row.items() if key.startswith("metrics/")},
        "best": {},
        "results_csv": str(csv_path) if csv_path else None,
        "checkpoint": checkpoint,
        "completed": completed,
    }


def _evaluation_status(run_dir: Path, backend: str) -> dict[str, Any]:
    for summary_path in _summary_candidates(run_dir):
        summary = _read_json(summary_path)
        metrics = summary.get("metrics") or summary.get("eval_results") or summary.get("evaluation") or {}
        if metrics:
            return {"status": "completed", "backend": backend, "metrics": metrics, "summary_path": str(summary_path)}
    return {"status": "pending", "backend": backend, "metrics": {}}


def _has_training_summary(run_dir: Path) -> bool:
    return any(path.is_file() for path in _summary_candidates(run_dir))


def _has_completed_summary(run_dir: Path) -> bool:
    for summary_path in _summary_candidates(run_dir):
        summary = _read_json(summary_path)
        if not summary:
            continue
        status = str(summary.get("status") or summary.get("training_status") or "").strip().lower()
        if status in {"completed", "complete", "success", "finished"}:
            return True
        if summary.get("metrics") or summary.get("eval_results") or summary.get("evaluation"):
            return True
    return False


def _is_training_completed(training: dict[str, Any]) -> bool:
    return bool(training.get("completed") or training.get("status") == "completed")


def _resource_status(run_dir: Path, backend: str) -> dict[str, Any]:
    selected_device = _selected_device(run_dir)
    accelerator_reservation = _read_json(
        run_dir / "deimv2_training_run" / "accelerator_state.json"
    )
    return {
        "timestamp": utc_now(),
        "selected_device": selected_device,
        "accelerator_type": _accelerator_type(selected_device),
        "gpu": _nvidia_smi_status(),
        "npu": _npu_smi_status(),
        "host": _host_status(),
        "backend": backend,
        "accelerator_reservation": accelerator_reservation or {
            "status": "pending",
            "cpu_fallback": False,
        },
    }


def _merge_counter_state(block: dict[str, Any], state_block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state_block, dict):
        return block
    merged = dict(block)
    for key in ("started_at", "completed_at", "updated_at", "current_image", "planned", "completed", "success", "failed", "error", "fallback_used", "reason", "reason_code"):
        if state_block.get(key) is not None:
            merged[key] = state_block.get(key)
    if state_block.get("status") in {"pending", "running", "completed", "failed", "timeout", "skipped", "disabled"}:
        state_status = state_block["status"]
        if state_status in {"failed", "timeout"} or merged.get("status") != "completed":
            merged["status"] = state_status
    return merged


def _stream_info(status: dict[str, Any]) -> dict[str, Any]:
    phase = str(status.get("phase") or "")
    annotation = status.get("annotation") if isinstance(status.get("annotation"), dict) else {}
    generation = status.get("generation") if isinstance(status.get("generation"), dict) else {}
    training = status.get("training") if isinstance(status.get("training"), dict) else {}
    evaluation = status.get("evaluation") if isinstance(status.get("evaluation"), dict) else {}
    model_request = status.get("model_request") if isinstance(status.get("model_request"), dict) else {}
    current_llm = model_request.get("current") if isinstance(model_request.get("current"), dict) else None
    if status.get("status") == "failed":
        return {"stage": "failed", "status": "failed", "metrics": {}}
    if status.get("status") == "completed" or phase == "completed":
        return {
            "stage": "completed",
            "status": "complete",
            "metrics": {
                **_training_stream_metrics(training),
                "evaluation": evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {},
            },
        }
    if _is_training_completed(training):
        summarizing = bool(current_llm and str(current_llm.get("purpose") or "").startswith("workflow_final"))
        return {
            "stage": "evaluation",
            "status": "summarizing" if summarizing else ("complete" if evaluation.get("status") == "completed" else "completed"),
            "metrics": {
                **_training_stream_metrics(training),
                "training_completed": True,
                "evaluation": evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {},
                "llm_purpose": current_llm.get("purpose") if summarizing and current_llm else None,
                "llm_model": current_llm.get("model") if summarizing and current_llm else None,
            },
        }
    if isinstance(current_llm, dict):
        return {
            "stage": "llm",
            "status": current_llm.get("status") or "running",
            "metrics": {
                "purpose": current_llm.get("purpose"),
                "model": current_llm.get("model"),
                "elapsed_ms": current_llm.get("elapsed_ms"),
            },
        }
    if phase == "data_reuse":
        reuse = status.get("data_reuse") if isinstance(status.get("data_reuse"), dict) else {}
        return {
            "stage": "data_reuse",
            "status": reuse.get("status") or "completed",
            "metrics": {
                "source_run_id": reuse.get("source_run_id"),
                "real_images": reuse.get("real_images") or 0,
                "synthetic_images": reuse.get("synthetic_images") or 0,
                "annotations": reuse.get("annotations") or 0,
                "copied_files": reuse.get("copied_files") or 0,
                "completed": True,
            },
        }
    if phase == "annotation":
        annotation_metrics = _annotation_stream_metrics(annotation)
        return {
            "stage": "annotation",
            "status": annotation_metrics["status"],
            "metrics": annotation_metrics,
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
                "completed": generation.get("status") in {"completed", "disabled", "skipped"},
                "error": generation.get("error"),
                "reason": generation.get("reason"),
                "reason_code": generation.get("reason_code"),
                "fallback_used": bool(generation.get("fallback_used")),
            },
        }
    if phase == "training" or training.get("status") == "running":
        return {
            "stage": "training",
            "status": training.get("status") or "running",
            "metrics": _training_stream_metrics(training),
        }
    if status.get("status") == "completed":
        return {
            "stage": "evaluation",
            "status": "complete",
            "metrics": evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {},
        }
    return {"stage": phase or "unknown", "status": status.get("status") or "running", "metrics": {}}


def _annotation_stream_metrics(annotation: dict[str, Any]) -> dict[str, Any]:
    real = annotation.get("real") if isinstance(annotation.get("real"), dict) else {}
    synthetic = annotation.get("synthetic") if isinstance(annotation.get("synthetic"), dict) else {}
    real_completed = _int_or_none(real.get("completed")) or 0
    real_total = _int_or_none(real.get("total")) or 0
    synthetic_completed = _int_or_none(synthetic.get("completed")) or 0
    synthetic_total = _int_or_none(synthetic.get("total")) or 0
    annotated = real_completed + synthetic_completed
    total = real_total + synthetic_total
    success = (_int_or_none(real.get("success")) or real_completed) + (_int_or_none(synthetic.get("success")) or synthetic_completed)
    failed = (_int_or_none(real.get("failed")) or 0) + (_int_or_none(synthetic.get("failed")) or 0)
    completed = total > 0 and annotated >= total and failed == 0
    status = "completed" if completed else ("running" if annotated > 0 else "pending")
    if real.get("status") == "failed" or synthetic.get("status") == "failed":
        status = "failed"
        completed = False
    return {
        "annotated": annotated,
        "total": total,
        "success": success,
        "failed": failed,
        "started": bool(real.get("started_at") or synthetic.get("started_at") or annotated),
        "completed": completed,
        "status": status,
        "real": {
            "annotated": real_completed,
            "total": real_total,
            "success": _int_or_none(real.get("success")) or real_completed,
            "failed": _int_or_none(real.get("failed")) or 0,
            "completed": real.get("status") == "completed",
        },
        "synthetic": {
            "annotated": synthetic_completed,
            "total": synthetic_total,
            "success": _int_or_none(synthetic.get("success")) or synthetic_completed,
            "failed": _int_or_none(synthetic.get("failed")) or 0,
            "completed": synthetic_total > 0 and synthetic.get("status") == "completed",
        },
    }


def _training_stream_metrics(training: dict[str, Any]) -> dict[str, Any]:
    return {
        "started": bool(training.get("started_at") or training.get("current_epoch") is not None),
        "completed": _is_training_completed(training),
        "current_epoch": training.get("current_epoch"),
        "current_epoch_display": training.get("current_epoch_display"),
        "total_epochs": training.get("total_epochs"),
        "current_step": training.get("current_step"),
        "total_steps": training.get("total_steps"),
        "progress": training.get("progress"),
        "loss": training.get("loss"),
        "lr": training.get("lr"),
    }


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
    failure_error = _deimv2_failure_log_error(text)
    if failure_error:
        latest["failure_error"] = failure_error
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


def _deimv2_failure_log_error(text: str) -> str | None:
    if not text:
        return None
    for pattern in DEIMV2_FAILURE_LOG_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(0).strip()
    return None


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
    current_index: int | None = None
    for raw_line in text.splitlines():
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw_line)
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 3:
            continue

        device_match = re.match(r"^(\d+)\s+(.+?)\s*$", parts[0])
        if device_match and re.search(r"[A-Za-z]", device_match.group(2)):
            index = _int_or_none(device_match.group(1))
            if index is None or index > 255:
                current_index = None
                continue
            item = devices.setdefault(index, {"index": index})
            item["name"] = device_match.group(2).strip()
            if parts[1]:
                item["health"] = parts[1]
            telemetry = re.match(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", parts[-1])
            if telemetry:
                item["power_w"] = _float_or_none(telemetry.group(1))
                item["temperature_c"] = _float_or_none(telemetry.group(2))
            current_index = index
            continue

        if current_index is None or current_index not in devices:
            continue
        if not re.fullmatch(r"(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d+", parts[1]):
            continue
        usage = parts[-1]
        utilization = re.match(r"^\s*(\d+(?:\.\d+)?)", usage)
        if utilization:
            devices[current_index]["utilization_percent"] = _float_or_none(utilization.group(1))
        memory_pairs = re.findall(r"(\d+)\s*/\s*(\d+)", usage)
        if memory_pairs:
            used, total = (int(value) for value in memory_pairs[-1])
            devices[current_index]["memory_used_mb"] = used
            devices[current_index]["memory_total_mb"] = total
            devices[current_index]["memory_free_mb"] = max(0, total - used)

    return [devices[index] for index in sorted(devices)]


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
