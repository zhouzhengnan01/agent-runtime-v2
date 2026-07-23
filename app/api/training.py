from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import signal
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.auth import require_runtime_token
from app.core.http_training_jobs import (
    ACTIVE_HTTP_TRAINING_STATUSES,
    read_current_http_training_job_marker,
    update_http_training_job_marker,
    write_http_training_job_marker,
)
from app.core.runtime import default_container
from app.core.runtime.health_state import ReviewSlot
from app.core.training_artifact_updates import build_training_artifact_session_updates
from app.core.training_status import build_training_status, build_training_stream_status, list_training_runs
from app.protocols.acp.event_broker import acp_event_broker
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


router = APIRouter(prefix="/api/training", tags=["training"], dependencies=[Depends(require_runtime_token)])
loader = default_container.loader
runtime = default_container.runtime

VIRTUAL_UPLOADS_PREFIX = "/mnt/user-data/uploads"
HTTP_JOB_STATUS_INTERVAL_SECONDS = 2.0
HTTP_TRAINING_ACTIVE_STATUSES = ACTIVE_HTTP_TRAINING_STATUSES


class TrainingJobFile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str | None = None
    path: str | None = None
    file_url: str | None = Field(default=None, alias="fileUrl")
    file_name: str | None = Field(default=None, alias="fileName")
    name: str | None = None
    media_type: str | None = Field(default=None, alias="mediaType")
    mime_type: str | None = None
    others: dict[str, Any] = Field(default_factory=dict)


class TrainingJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thread_id: str | None = Field(default=None, alias="threadId")
    agent_name: str = Field(default="default", alias="agentName")
    app_template_name: str | None = Field(default=None, alias="appTemplateName")
    content: str | None = None
    prompt: str | None = None
    runtime_options: dict[str, Any] = Field(default_factory=dict, alias="runtimeOptions")
    files: list[TrainingJobFile] = Field(default_factory=list)


class TrainingJobRecord(BaseModel):
    job_id: str
    thread_id: str
    agent_name: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    run_id: str | None = None
    error: str | None = None
    status_url: str
    artifacts_url: str
    result: dict[str, Any] | None = None


_jobs_lock = asyncio.Lock()
_jobs_by_thread: dict[str, dict[str, Any]] = {}
_jobs_by_id: dict[str, dict[str, Any]] = {}


@router.get("/status/{thread_id}/runs")
async def get_training_runs(thread_id: str) -> dict[str, object]:
    try:
        return list_training_runs(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/status/{thread_id}")
async def get_training_status(thread_id: str, run_id: str | None = Query(default=None)) -> dict[str, object]:
    try:
        return build_training_status(thread_id, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/jobs")
async def create_training_job(request: TrainingJobRequest) -> dict[str, object]:
    thread_id = _normalize_thread_id(request.thread_id)
    agent_name = (request.agent_name or "default").strip() or "default"
    content = (request.content or request.prompt or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="content is required.")

    try:
        agent = loader.load(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    runtime_options = _build_runtime_options(request, thread_id)
    attachments = _build_attachments(thread_id, request.files)
    chat_request = ChatRequest(
        messages=[Message(role="user", content=content)],
        attachments=attachments,
        runtime_options=runtime_options,
    )

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    now = _utc_now()
    record: dict[str, Any] = {
        "job_id": job_id,
        "thread_id": thread_id,
        "agent_name": agent_name,
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "run_id": None,
        "error": None,
        "status_url": f"/api/training/status/{thread_id}",
        "artifacts_url": f"/api/artifacts/{thread_id}",
        "result": None,
        "task": None,
    }

    async with _jobs_lock:
        existing = _jobs_by_thread.get(thread_id)
        if existing and existing.get("status") in {"queued", "running", "cancelling"}:
            raise HTTPException(
                status_code=409,
                detail=f"Training job already active for thread_id={thread_id}. Cancel it or use another thread_id.",
            )
        _jobs_by_thread[thread_id] = record
        _jobs_by_id[job_id] = record
        write_http_training_job_marker(thread_id, record)

    task = asyncio.create_task(_run_training_job(record, agent, chat_request), name=f"training-job-{job_id}")
    record["task"] = task
    return _public_job_record(record)


@router.get("/jobs/{thread_id}")
async def get_training_job(thread_id: str) -> dict[str, object]:
    normalized = _normalize_thread_id(thread_id)
    async with _jobs_lock:
        record = _jobs_by_thread.get(normalized)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Training job not found: {normalized}")
        return _public_job_record(record)


@router.post("/jobs/{thread_id}/cancel")
async def cancel_training_job(thread_id: str) -> dict[str, object]:
    normalized = _normalize_thread_id(thread_id)
    fallback_record = False
    async with _jobs_lock:
        record = _jobs_by_thread.get(normalized)
        if record is None:
            record = _active_disk_training_record(normalized)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Training job not found: {normalized}")
            _jobs_by_thread[normalized] = record
            _jobs_by_id[str(record["job_id"])] = record
            fallback_record = True
        status = str(record.get("status") or "")
        if status in {"completed", "failed", "cancelled"}:
            return {"cancelled": False, "reason": f"job already {status}", "job": _public_job_record(record)}
        record["status"] = "cancelling"
        record["completed_at"] = None
        task = record.get("task")
        update_http_training_job_marker(normalized, status="cancelling", completed_at=None)

    runtime.session_manager.cancel_active_turn(normalized)
    _mark_run_progress_cancelled(
        normalized,
        str(record.get("run_id") or ""),
        status="cancelling",
        message="HTTP cancel requested.",
    )
    if isinstance(task, asyncio.Task) and not task.done():
        task.cancel()
    terminated_processes = _terminate_training_processes(normalized, str(record.get("run_id") or ""))
    record["status"] = "cancelled"
    record["completed_at"] = _utc_now()
    record["error"] = "Training job cancelled by HTTP request."
    record["result"] = _cancelled_http_job_result(normalized, record)
    cancel_message = "HTTP cancel requested."
    if terminated_processes:
        cancel_message = f"HTTP cancel requested; terminated training processes: {terminated_processes}"
    update_http_training_job_marker(
        normalized,
        status="cancelled",
        completed_at=record["completed_at"],
        run_id=record.get("run_id"),
        error=record["error"],
        result=record.get("result"),
    )
    _mark_run_progress_cancelled(
        normalized,
        str(record.get("run_id") or ""),
        status="cancelled",
        message=cancel_message,
    )
    await _publish_training_status(normalized, set())
    await _publish_http_job_session_update(
        normalized,
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": f"http-training-job-{record['job_id']}",
            "title": "training job",
            "kind": "other",
            "status": "cancelled",
            "rawOutput": record["result"],
            "_meta": {
                "jetlinksRuntimeEvent": {
                    "type": "http.training_job.cancelled",
                    "data": {
                        "job_id": record.get("job_id"),
                        "thread_id": normalized,
                        "status": "cancelled",
                        "terminated_processes": terminated_processes,
                    },
                }
            },
        },
    )
    await _publish_http_job_result(normalized, str(record["job_id"]), record["result"])
    return {
        "cancelled": True,
        "job": _public_job_record(record),
        "fallback": fallback_record,
        "terminated_processes": terminated_processes,
    }


async def _run_training_job(record: dict[str, Any], agent: Any, request: ChatRequest) -> None:
    thread_id = str(record["thread_id"])
    job_id = str(record["job_id"])
    agent_name = str(record["agent_name"])
    record["status"] = "running"
    record["started_at"] = _utc_now()
    update_http_training_job_marker(thread_id, status="running", started_at=record["started_at"])
    status_stop = asyncio.Event()
    published_artifacts: set[str] = set()
    status_task = asyncio.create_task(_publish_training_status_loop(thread_id, status_stop, published_artifacts))
    last_status_sent = 0.0
    final_result: AgentRunResult | None = None
    final_reply_sent = False
    try:
        async with ReviewSlot():
            await _publish_http_job_session_update(
                thread_id,
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": f"http-training-job-{job_id}",
                    "title": "training job",
                    "kind": "other",
                    "status": "in_progress",
                    "_meta": {
                        "jetlinksRuntimeEvent": {
                            "type": "http.training_job.started",
                            "data": {"job_id": job_id, "thread_id": thread_id, "agent": agent_name},
                        }
                    },
                },
            )
            async for event in runtime.iter_events(agent, request):
                event_run_id = _training_run_id_from_event(event)
                if event_run_id and not record.get("run_id"):
                    record["run_id"] = event_run_id
                    update_http_training_job_marker(thread_id, run_id=event_run_id)
                result_from_event = _result_from_event(event)
                if result_from_event is not None:
                    final_result = result_from_event
                await _publish_runtime_event(thread_id, event)
                if _event_reply_text(event):
                    final_reply_sent = True
                now = time.monotonic()
                if now - last_status_sent >= HTTP_JOB_STATUS_INTERVAL_SECONDS:
                    await _publish_training_status(thread_id, published_artifacts)
                    last_status_sent = now
        await _publish_training_status(thread_id, published_artifacts)
        if final_result is None:
            final_result = AgentRunResult(
                agent=agent_name,
                thread_id=thread_id,
                status="failed",
                reply="HTTP training job finished without a final result.",
            )
        job_status = _job_status_from_result(final_result)
        record["status"] = job_status
        record["result"] = final_result.model_dump()
        if job_status == "cancelled":
            record["error"] = "Training job cancelled by HTTP request."
            _mark_run_progress_cancelled(
                thread_id,
                str(record.get("run_id") or ""),
                status="cancelled",
                message=record["error"],
            )
        elif final_result.status != "completed":
            record["error"] = final_result.reply
        update_http_training_job_marker(
            thread_id,
            status=record["status"],
            completed_at=_utc_now(),
            run_id=record.get("run_id"),
            error=record.get("error"),
            result=record.get("result"),
        )
        await _publish_training_status(thread_id, published_artifacts)
        if not final_reply_sent:
            await _publish_http_job_final_reply(thread_id, job_id, final_result)
        await _publish_http_job_session_update(
            thread_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": f"http-training-job-{job_id}",
                "title": "training job",
                "kind": "other",
                "status": job_status,
                "rawOutput": final_result.model_dump(),
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "http.training_job.completed",
                        "data": {
                            "job_id": job_id,
                            "thread_id": thread_id,
                            "status": job_status,
                        },
                    }
                },
            },
        )
        await _publish_http_job_result(thread_id, job_id, final_result.model_dump())
    except asyncio.CancelledError:
        runtime.session_manager.cancel_active_turn(thread_id)
        record["status"] = "cancelled"
        record["error"] = "Training job cancelled by HTTP request."
        update_http_training_job_marker(
            thread_id,
            status="cancelled",
            completed_at=_utc_now(),
            run_id=record.get("run_id"),
            error=record["error"],
            result=_cancelled_http_job_result(thread_id, record),
        )
        _mark_run_progress_cancelled(
            thread_id,
            str(record.get("run_id") or ""),
            status="cancelled",
            message=record["error"],
        )
        await _publish_training_status(thread_id, published_artifacts)
        await _publish_http_job_error(thread_id, job_id, -32800, record["error"])
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        update_http_training_job_marker(
            thread_id,
            status="failed",
            completed_at=_utc_now(),
            run_id=record.get("run_id"),
            error=record["error"],
            result=AgentRunResult(
                agent=agent_name,
                thread_id=thread_id,
                status="failed",
                reply=record["error"],
                metadata={"phase": "failed", "status": "failed", "job_id": job_id},
            ).model_dump(),
        )
        _mark_run_progress_cancelled(
            thread_id,
            str(record.get("run_id") or ""),
            status="failed",
            message=str(exc),
        )
        await _publish_training_status(thread_id, published_artifacts)
        await _publish_http_job_error(thread_id, job_id, -32000, str(exc))
    finally:
        status_stop.set()
        await asyncio.gather(status_task, return_exceptions=True)
        record["completed_at"] = _utc_now()


def _build_runtime_options(request: TrainingJobRequest, thread_id: str) -> RuntimeOptions:
    raw = dict(request.runtime_options or {})
    if request.app_template_name and not raw.get("appTemplateName") and not raw.get("app_template_name"):
        raw["appTemplateName"] = request.app_template_name
    raw.setdefault("thread_id", thread_id)
    raw.setdefault("threadId", thread_id)
    raw.setdefault("workflow", "yolo_training_flow")
    normalized = _normalize_runtime_options(raw)
    return RuntimeOptions.model_validate(normalized)


def _normalize_runtime_options(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "threadId": "thread_id",
        "userId": "user_id",
        "projectId": "project_id",
        "appTemplateName": "app_template_name",
        "modelType": "model_type",
        "modelName": "model_name",
        "trainingModelId": "training_model_id",
        "maxSyntheticImages": "max_synthetic_images",
        "reusePreviousDataPreparation": "reuse_previous_data_preparation",
        "reuseFromRunId": "reuse_from_run_id",
        "baseUrl": "base_url",
        "apiKey": "api_key",
        "topP": "top_p",
        "maxTokens": "max_tokens",
        "requestTimeoutSeconds": "request_timeout_seconds",
        "responseFormat": "response_format",
        "selectedSkills": "selected_skills",
        "selectedMcpTools": "selected_mcp_tools",
        "skillParameters": "skill_parameters",
        "configOptions": "config_options",
        "sandboxProfile": "sandbox_profile",
    }
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        normalized[aliases.get(key, key)] = value
    if "deimv2ModelVariant" in raw:
        normalized["deimv2ModelVariant"] = raw["deimv2ModelVariant"]
    return normalized


def _build_attachments(thread_id: str, files: list[TrainingJobFile]) -> list[Attachment]:
    if not files:
        files = _discover_default_upload_files(thread_id)
    attachments: list[Attachment] = []
    for item in files:
        raw_path = item.path or item.file_url or item.others.get("path") or item.others.get("fileUrl")
        if not raw_path:
            continue
        path = str(raw_path)
        role = str(item.role or item.others.get("role") or "").strip()
        name = item.name or item.file_name or Path(path.replace("\\", "/")).name or role or "upload"
        mime_type = item.mime_type or item.media_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
        metadata = {
            "role": role,
            "source": "http_training_jobs",
        }
        for key, value in item.others.items():
            metadata.setdefault(key, value)
        attachments.append(
            Attachment(
                name=name,
                path=path,
                mime_type=mime_type,
                metadata=metadata,
            )
        )
    return attachments


def _discover_default_upload_files(thread_id: str) -> list[TrainingJobFile]:
    paths = runtime.artifact_store.prepare_thread(thread_id)
    uploads = paths.uploads
    candidates: list[tuple[str, tuple[str, ...]]] = [
        ("dataset", ("datasets.zip", "dataset.zip")),
        ("image1", ("image1.zip",)),
        ("image2", ("image2.zip",)),
    ]
    found: list[TrainingJobFile] = []
    for role, names in candidates:
        for name in names:
            path = uploads / name
            if path.is_file():
                relative = path.relative_to(uploads).as_posix()
                found.append(
                    TrainingJobFile(
                        role=role,
                        path=f"{VIRTUAL_UPLOADS_PREFIX}/{relative}",
                        name=name,
                        mediaType=mimetypes.guess_type(name)[0] or "application/zip",
                    )
                )
                break
    return found


def _normalize_thread_id(value: str | None) -> str:
    thread_id = (value or "").strip()
    if not thread_id:
        thread_id = f"http-training-{uuid.uuid4().hex[:12]}"
    if any(part in thread_id for part in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail=f"Invalid thread_id: {thread_id}")
    return thread_id


def _public_job_record(record: dict[str, Any]) -> dict[str, object]:
    return TrainingJobRecord(
        job_id=str(record["job_id"]),
        thread_id=str(record["thread_id"]),
        agent_name=str(record["agent_name"]),
        status=str(record["status"]),
        created_at=str(record["created_at"]),
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        run_id=record.get("run_id"),
        error=record.get("error"),
        status_url=str(record["status_url"]),
        artifacts_url=str(record["artifacts_url"]),
        result=record.get("result"),
    ).model_dump()


def _cancelled_http_job_result(thread_id: str, record: dict[str, Any]) -> dict[str, Any]:
    reply = str(record.get("error") or "Training job cancelled by HTTP request.")
    return AgentRunResult(
        agent=str(record.get("agent_name") or "default"),
        thread_id=thread_id,
        status="failed",
        reply=reply,
        metadata={
            "phase": "cancelled",
            "status": "cancelled",
            "cancelled": True,
            "job_id": record.get("job_id"),
            "run_id": record.get("run_id"),
        },
    ).model_dump()


def _active_disk_training_record(thread_id: str) -> dict[str, Any] | None:
    marker = read_current_http_training_job_marker(thread_id, require_active=True)
    if marker is None:
        return None
    run_id = str(marker.get("run_id") or "").strip()
    if run_id:
        try:
            status = build_training_status(thread_id, run_id=run_id)
        except (FileNotFoundError, ValueError):
            status = {}
    else:
        status = {}
    training = status.get("training") if isinstance(status.get("training"), dict) else {}
    status_value = str(status.get("status") or "")
    training_status = str(training.get("status") or "")
    marker_status = str(marker.get("status") or "")
    if (
        marker_status not in HTTP_TRAINING_ACTIVE_STATUSES
        and status_value not in HTTP_TRAINING_ACTIVE_STATUSES
        and training_status not in HTTP_TRAINING_ACTIVE_STATUSES
    ):
        return None
    now = _utc_now()
    record_status = marker_status
    if record_status not in HTTP_TRAINING_ACTIVE_STATUSES:
        record_status = status_value if status_value in HTTP_TRAINING_ACTIVE_STATUSES else training_status or "running"
    return {
        "job_id": str(marker.get("job_id") or f"thread-{thread_id}"),
        "thread_id": thread_id,
        "agent_name": str(marker.get("agent_name") or "default"),
        "status": record_status,
        "created_at": marker.get("created_at") or status.get("created_at") or now,
        "started_at": marker.get("started_at") or status.get("started_at"),
        "completed_at": marker.get("completed_at") or status.get("completed_at"),
        "run_id": run_id or None,
        "error": marker.get("error"),
        "status_url": f"/api/training/status/{thread_id}",
        "artifacts_url": f"/api/artifacts/{thread_id}",
        "result": None,
        "task": None,
    }


def _run_dir_for_thread(thread_id: str, run_id: str) -> Path | None:
    if not run_id:
        return None
    run_dir = Path.cwd() / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    return run_dir.resolve() if run_dir.is_dir() else None


def _terminate_training_processes(thread_id: str, run_id: str) -> list[int]:
    if not run_id:
        return []
    run_dir = _run_dir_for_thread(thread_id, run_id)
    markers = _training_process_markers(thread_id, run_id, run_dir)
    if not markers:
        return []
    processes = _process_table()
    if not processes:
        return []
    comparable_markers = [marker.lower() for marker in markers] if os.name == "nt" else markers
    candidates: set[int] = set()
    for pid, _ppid, command in processes:
        comparable = command.lower() if os.name == "nt" else command
        if any(marker in comparable for marker in comparable_markers):
            candidates.add(pid)
    if not candidates:
        return []
    by_pid = {pid: (ppid, command) for pid, ppid, command in processes}
    children_by_parent: dict[int, list[int]] = {}
    for pid, ppid, _command in processes:
        children_by_parent.setdefault(ppid, []).append(pid)
    targets = set(candidates)
    changed = True
    while changed:
        changed = False
        for pid, (ppid, command) in by_pid.items():
            if pid in targets:
                continue
            if ppid in targets:
                targets.add(pid)
                changed = True
                continue
            if any(child in targets for child in children_by_parent.get(pid, [])) and _is_training_process_command(command):
                targets.add(pid)
                changed = True
    current_pid = os.getpid()
    targets.discard(current_pid)
    killed: list[int] = []
    for pid in sorted(targets, key=lambda item: _process_depth(item, by_pid), reverse=True):
        if _terminate_process(pid):
            killed.append(pid)
    return killed


def _training_process_markers(thread_id: str, run_id: str, run_dir: Path | None) -> list[str]:
    selected_run_id = run_id
    thread_root = Path.cwd() / ".runtime" / "threads" / thread_id
    markers: list[str] = []
    if run_dir is not None:
        markers.append(str(run_dir))
    if selected_run_id:
        markers.append(str(thread_root / "workspace" / "runs" / selected_run_id))

    input_paths = [
        thread_root / "workspace" / "data-auto-annotation-pipeline-input.json",
        thread_root / "workspace" / "deimv2-auto-training-input.json",
        thread_root / "workspace" / "deimv2-prepare-dataset-input.json",
        thread_root / "workspace" / "deimv2-training-input.json",
    ]
    if selected_run_id:
        input_paths.extend((thread_root / "workspace" / "runs" / selected_run_id).glob("*input.json"))

    evidence = [selected_run_id or ""]
    if run_dir is not None:
        evidence.append(str(run_dir))
    for path in input_paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(item and item in text for item in evidence):
            markers.append(str(path))

    normalized: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        value = marker.strip()
        if not value:
            continue
        key = value.lower() if os.name == "nt" else value
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def _is_training_process_command(command: str) -> bool:
    lowered = (command or "").lower()
    return any(
        marker in lowered
        for marker in (
            "run_deimv2_training.py",
            "prepare_deimv2_dataset.py",
            "run_data_preparation_pipeline.py",
            "sam3-predict.py",
            "image-composite.py",
            "image-generate.py",
        )
    )


def _process_table() -> list[tuple[int, int, str]]:
    if os.name == "nt":
        return _windows_process_table()
    return _posix_process_table()


def _posix_process_table() -> list[tuple[int, int, str]]:
    try:
        completed = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    rows: list[tuple[int, int, str]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        except ValueError:
            continue
    return rows


def _windows_process_table() -> list[tuple[int, int, str]]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0 or not completed.stdout.strip():
        return []
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    rows: list[tuple[int, int, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("ProcessId"))
            ppid = int(item.get("ParentProcessId") or 0)
        except (TypeError, ValueError):
            continue
        rows.append((pid, ppid, str(item.get("CommandLine") or "")))
    return rows


def _process_depth(pid: int, by_pid: dict[int, tuple[int, str]]) -> int:
    depth = 0
    seen: set[int] = set()
    current = pid
    while current in by_pid and current not in seen:
        seen.add(current)
        parent = by_pid[current][0]
        if parent <= 0 or parent == current:
            break
        depth += 1
        current = parent
    return depth


def _terminate_process(pid: int) -> bool:
    try:
        if os.name == "nt":
            completed = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False, timeout=8)
            return completed.returncode == 0
        os.kill(pid, signal.SIGTERM)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _mark_run_progress_cancelled(thread_id: str, run_id: str, *, status: str, message: str) -> None:
    if not run_id:
        return
    path = Path.cwd() / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id / "progress_state.json"
    if not path.is_file():
        return
    try:
        import json

        state = json.loads(path.read_text(encoding="utf-8"))
        now = _utc_now()
        state["status"] = status
        state["phase"] = "cancelled" if status == "cancelled" else status
        state["phase_label"] = "cancelled" if status == "cancelled" else status
        state["updated_at"] = now
        if status in {"cancelled", "failed"}:
            state["completed_at"] = now
        errors = state.setdefault("errors", [])
        if isinstance(errors, list) and message:
            errors.append(message)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


async def _publish_runtime_event(thread_id: str, event: ChatEvent) -> None:
    update = _event_to_session_update(event)
    if update is not None:
        await _publish_http_job_session_update(thread_id, update)
    if event.type in {
        "data_preparation.synthetic_generation.started",
        "data_preparation.synthetic_generation.finished",
    }:
        await _publish_training_status(thread_id)


async def _publish_http_job_session_update(thread_id: str, update: dict[str, Any]) -> None:
    await acp_event_broker.publish(
        {thread_id},
        "session/update",
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": thread_id,
                "update": update,
            },
        },
    )


async def _publish_training_status(thread_id: str, published_artifacts: set[str] | None = None) -> None:
    marker = read_current_http_training_job_marker(thread_id, require_active=False)
    run_id = str((marker or {}).get("run_id") or "").strip()
    if not run_id:
        return
    try:
        status = build_training_stream_status(thread_id, run_id=run_id)
    except (FileNotFoundError, ValueError):
        return
    await acp_event_broker.publish({thread_id}, "training/status", status)
    if published_artifacts is not None:
        try:
            full_status = build_training_status(thread_id, run_id=run_id)
        except (FileNotFoundError, ValueError):
            return
        await _publish_training_artifact_updates(thread_id, full_status, published_artifacts)


async def _publish_training_status_loop(
    thread_id: str,
    stop_event: asyncio.Event,
    published_artifacts: set[str] | None = None,
) -> None:
    """HTTP 触发训练时，独立轮询状态文件并推送给 /api/acp/stream。"""
    while not stop_event.is_set():
        await _publish_training_status(thread_id, published_artifacts)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=HTTP_JOB_STATUS_INTERVAL_SECONDS)
        except TimeoutError:
            continue


async def _publish_training_artifact_updates(
    thread_id: str,
    status: dict[str, Any],
    published_artifacts: set[str],
) -> None:
    updates = build_training_artifact_session_updates(
        thread_id=thread_id,
        status=status,
        artifact_store=runtime.artifact_store,
        published_artifacts=published_artifacts,
    )
    for update in updates:
        await _publish_http_job_session_update(thread_id, update)


async def _publish_http_job_result(thread_id: str, job_id: str, result: dict[str, Any]) -> None:
    await acp_event_broker.publish(
        {thread_id},
        "result",
        {
            "jsonrpc": "2.0",
            "id": job_id,
            "result": result,
        },
    )


async def _publish_http_job_final_reply(thread_id: str, job_id: str, result: AgentRunResult) -> None:
    reply = str(result.reply or "").strip()
    if not reply:
        return
    await _publish_http_job_session_update(
        thread_id,
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": reply},
            "rawOutput": result.model_dump(),
            "_meta": {
                "jetlinksRuntimeEvent": {
                    "type": "http.training_job.reply",
                    "data": {
                        "job_id": job_id,
                        "thread_id": thread_id,
                        "status": result.status,
                        "reply": reply,
                    },
                }
            },
        },
    )


async def _publish_http_job_error(thread_id: str, job_id: str, code: int, message: str) -> None:
    await acp_event_broker.publish(
        {thread_id},
        "error",
        {
            "jsonrpc": "2.0",
            "id": job_id,
            "error": {"code": code, "message": message},
        },
    )


def _event_to_session_update(event: ChatEvent) -> dict[str, Any] | None:
    data = event.data
    base = {"_meta": {"jetlinksRuntimeEvent": event.model_dump(), "jetlinksDiff": _event_diff(event)}}
    if event.type in {"agent.message", "agent.message.delta"}:
        text = str(data.get("text") or data.get("message") or "")
        if not text:
            return None
        return {**base, "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
    if event.type == "skill.started":
        skill_name = _event_skill_name(event)
        return {
            **base,
            "sessionUpdate": "tool_call",
            "toolCallId": f"skill-{skill_name}",
            "title": skill_name,
            "kind": "other",
            "status": "in_progress",
        }
    if event.type == "skill.completed":
        skill_name = _event_skill_name(event)
        return {
            **base,
            "sessionUpdate": "tool_call_update",
            "toolCallId": f"skill-{skill_name}",
            "title": skill_name,
            "kind": "other",
            "status": "completed",
            "rawOutput": data.get("result") or data.get("structured_content") or data,
        }
    if event.type == "run.started":
        return {
            **base,
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "训练任务已开始。"},
        }
    if event.type == "run.failed":
        return {
            **base,
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": str(data.get("error") or "训练任务失败。")},
        }
    if event.type in {"artifact.created", "preview.ready"}:
        artifact = _event_artifact(data)
        artifact_name = str(artifact.get("name") or artifact.get("title") or artifact.get("path") or "artifact")
        text = f"artifact: {artifact_name}" if event.type == "artifact.created" else f"preview ready: {artifact_name}"
        return {
            **base,
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": text},
            "artifact": artifact,
            "rawOutput": artifact,
        }
    if event.type == "run.completed":
        result = data.get("result")
        if isinstance(result, dict):
            text = str(result.get("reply") or "")
            if text:
                return {**base, "sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
        return None
    summary = _event_summary(event)
    if not summary:
        return None
    return {**base, "sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": summary}}


def _event_artifact(data: dict[str, Any]) -> dict[str, Any]:
    artifact = data.get("artifact")
    return artifact if isinstance(artifact, dict) else {}


def _event_reply_text(event: ChatEvent) -> str:
    data = event.data
    if event.type == "run.failed":
        return str(data.get("error") or "").strip()
    if event.type == "run.completed":
        result = data.get("result")
        if isinstance(result, dict):
            return str(result.get("reply") or "").strip()
    return ""


def _training_run_id_from_event(event: ChatEvent) -> str:
    if event.type != "workflow.training_run.selected":
        return ""
    return str(event.data.get("run_id") or "").strip()


def _event_diff(event: ChatEvent) -> dict[str, Any] | None:
    if event.type != "artifact.created":
        return None
    artifact = _event_artifact(event.data)
    path = artifact.get("path")
    if not path:
        return None
    return {
        "type": "artifact_change",
        "status": "completed",
        "path": str(path),
        "source": "artifact",
        "operation": "created",
    }


def _event_skill_name(event: ChatEvent) -> str:
    data = event.data
    return str(data.get("skill_name") or data.get("skill") or data.get("name") or "skill").strip() or "skill"


def _event_summary(event: ChatEvent) -> str:
    if event.type.startswith("llm."):
        purpose = event.data.get("purpose") or "LLM"
        return f"{purpose}: {event.type}"
    if event.type.startswith("workflow.") or event.type.startswith("data_preparation."):
        return event.type
    return ""


def _result_from_event(event: ChatEvent) -> AgentRunResult | None:
    if event.type not in {"run.completed", "run.failed"}:
        return None
    result = event.data.get("result")
    if not isinstance(result, dict):
        return None
    try:
        return AgentRunResult.model_validate(result)
    except Exception:
        return None


def _job_status_from_result(result: AgentRunResult) -> str:
    if result.status == "completed":
        return "completed"
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    phase = str(metadata.get("phase") or metadata.get("status") or "").strip().lower()
    if metadata.get("cancelled") is True or phase == "cancelled":
        return "cancelled"
    return "failed"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
