from __future__ import annotations

import asyncio
import json
import mimetypes
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.api.auth import require_runtime_token
from app.core.runtime import default_container
from app.core.runtime.health_state import ReviewSlot
from app.core.training_status import build_training_status, build_training_stream_status, list_training_runs
from app.protocols.acp.event_broker import acp_event_broker
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


router = APIRouter(prefix="/api/training", tags=["training"], dependencies=[Depends(require_runtime_token)])
loader = default_container.loader
runtime = default_container.runtime

VIRTUAL_UPLOADS_PREFIX = "/mnt/user-data/uploads"
HTTP_JOB_STATUS_INTERVAL_SECONDS = 2.0


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
    async with _jobs_lock:
        record = _jobs_by_thread.get(normalized)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Training job not found: {normalized}")
        status = str(record.get("status") or "")
        if status in {"completed", "failed", "cancelled"}:
            return {"cancelled": False, "reason": f"job already {status}", "job": _public_job_record(record)}
        record["status"] = "cancelling"
        record["completed_at"] = None
        task = record.get("task")

    runtime.session_manager.cancel_active_turn(normalized)
    _mark_latest_progress_cancelled(normalized, status="cancelling", message="HTTP cancel requested.")
    if isinstance(task, asyncio.Task) and not task.done():
        task.cancel()
    return {"cancelled": True, "job": _public_job_record(record)}


async def _run_training_job(record: dict[str, Any], agent: Any, request: ChatRequest) -> None:
    thread_id = str(record["thread_id"])
    job_id = str(record["job_id"])
    agent_name = str(record["agent_name"])
    record["status"] = "running"
    record["started_at"] = _utc_now()
    last_status_sent = 0.0
    final_result: AgentRunResult | None = None
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
                result_from_event = _result_from_event(event)
                if result_from_event is not None:
                    final_result = result_from_event
                await _publish_runtime_event(thread_id, event)
                now = time.monotonic()
                if now - last_status_sent >= HTTP_JOB_STATUS_INTERVAL_SECONDS:
                    await _publish_training_status(thread_id)
                    last_status_sent = now
        await _publish_training_status(thread_id)
        if final_result is None:
            final_result = AgentRunResult(
                agent=agent_name,
                thread_id=thread_id,
                status="failed",
                reply="HTTP training job finished without a final result.",
            )
        record["status"] = "completed" if final_result.status == "completed" else "failed"
        record["result"] = final_result.model_dump()
        record["run_id"] = _latest_run_id(thread_id)
        if final_result.status != "completed":
            record["error"] = final_result.reply
        await _publish_http_job_session_update(
            thread_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": f"http-training-job-{job_id}",
                "title": "training job",
                "kind": "other",
                "status": "completed" if final_result.status == "completed" else "failed",
                "rawOutput": final_result.model_dump(),
                "_meta": {
                    "jetlinksRuntimeEvent": {
                        "type": "http.training_job.completed",
                        "data": {
                            "job_id": job_id,
                            "thread_id": thread_id,
                            "status": final_result.status,
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
        _mark_latest_progress_cancelled(thread_id, status="cancelled", message=record["error"])
        await _publish_training_status(thread_id)
        await _publish_http_job_error(thread_id, job_id, -32800, record["error"])
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        _mark_latest_progress_cancelled(thread_id, status="failed", message=str(exc))
        await _publish_training_status(thread_id)
        await _publish_http_job_error(thread_id, job_id, -32000, str(exc))
    finally:
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
        run_id=record.get("run_id") or _latest_run_id(str(record["thread_id"])),
        error=record.get("error"),
        status_url=str(record["status_url"]),
        artifacts_url=str(record["artifacts_url"]),
        result=record.get("result"),
    ).model_dump()


def _latest_run_id(thread_id: str) -> str | None:
    runs_root = Path.cwd() / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs"
    if not runs_root.is_dir():
        return None
    runs = sorted(path.name for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("run-"))
    return runs[-1] if runs else None


def _mark_latest_progress_cancelled(thread_id: str, *, status: str, message: str) -> None:
    run_id = _latest_run_id(thread_id)
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


async def _publish_training_status(thread_id: str) -> None:
    try:
        status = build_training_stream_status(thread_id)
    except (FileNotFoundError, ValueError):
        return
    await acp_event_broker.publish({thread_id}, "training/status", status)


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
    base = {"_meta": {"jetlinksRuntimeEvent": event.model_dump()}}
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
    if event.type != "run.completed":
        return None
    result = event.data.get("result")
    if not isinstance(result, dict):
        return None
    try:
        return AgentRunResult.model_validate(result)
    except Exception:
        return None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
