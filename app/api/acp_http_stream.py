from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
import re
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.auth import require_runtime_token
from app.core.artifacts import ArtifactStore
from app.core.http_training_jobs import (
    ACTIVE_HTTP_TRAINING_STATUSES,
    TERMINAL_HTTP_TRAINING_STATUSES,
    read_current_http_training_job_marker,
)
from app.core.training_artifact_updates import build_training_artifact_session_updates
from app.core.training_annotation_previews import add_annotation_preview_to_update
from app.core.training_status import build_training_status, build_training_stream_status
from app.protocols.acp.event_broker import acp_event_broker


router = APIRouter(prefix="/api/acp", tags=["acp"])
SSE_HEARTBEAT_SECONDS = 10.0
SSE_STATUS_POLL_SECONDS = 2.0
SSE_TERMINAL_REPLAY_SECONDS = 600.0
MAX_MODEL_UPLOAD_BYTES = max(
    1,
    int(os.getenv("JETLINKS_MAX_MODEL_UPLOAD_BYTES", str(512 * 1024 * 1024))),
)
VIRTUAL_UPLOADS_PREFIX = "/mnt/user-data/uploads"
_SAFE_MODEL_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_. -]+")
model_store = ArtifactStore()


class AcpImagePreviewRequest(BaseModel):
    width: int = Field(ge=1, le=4096)
    height: int = Field(ge=1, le=4096)


class AcpHttpSubscriptionRequest(BaseModel):
    sessionId: str | None = None
    threadId: str | None = None
    imagePreview: AcpImagePreviewRequest | None = None


def _safe_model_filename(value: str) -> str:
    name = Path(value.replace("\\", "/")).name.strip().strip(".")
    cleaned = _SAFE_MODEL_FILENAME_RE.sub("-", name).strip(". ")
    return cleaned[:180] or "model.pt"


def _unique_model_path(root: Path, filename: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / filename
    if not candidate.exists():
        return candidate
    return root / f"{candidate.stem or 'model'}-{uuid.uuid4().hex[:8]}{candidate.suffix}"


def _model_registry_path(workspace: Path) -> Path:
    return workspace / "training_models.json"


def _register_training_model(workspace: Path, model_record: dict[str, object]) -> None:
    registry_path = _model_registry_path(workspace)
    registry: dict[str, object] = {"models": {}}
    if registry_path.exists():
        try:
            current = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        if isinstance(current, dict):
            registry = current
    models = registry.get("models")
    if not isinstance(models, dict):
        models = {}
        registry["models"] = models
    models[str(model_record["modelId"])] = model_record
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sse_payload(payload: dict, *, event: str) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


async def _stream_acp_subscription(request: AcpHttpSubscriptionRequest) -> AsyncIterator[str]:
    keys = {value for value in (request.sessionId, request.threadId) if value and value.strip()}
    if not keys:
        yield _sse_payload(
            {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "sessionId or threadId is required"},
            },
            event="error",
        )
        return

    subscriber_id, queue = await acp_event_broker.subscribe(keys)
    try:
        yield ": connected\n\n"
        last_status_fingerprint = ""
        last_heartbeat = asyncio.get_running_loop().time()
        published_artifacts: set[str] = set()
        published_terminal_replies: set[str] = set()
        published_terminal_results: set[str] = set()
        observed_job_keys: set[str] = set()
        initial_status = _initial_training_status(keys, observed_job_keys)
        if initial_status is not None:
            last_status_fingerprint = _status_fingerprint(initial_status)
            yield _sse_payload(initial_status, event="training/status")
            for payload in _artifact_session_update_payloads(
                keys,
                published_artifacts,
                observed_job_keys,
                image_preview=request.imagePreview,
            ):
                yield _sse_payload(payload, event="session/update")
            for payload in _terminal_reply_session_update_payloads(keys, published_terminal_replies, observed_job_keys):
                yield _sse_payload(payload, event="session/update")
            terminal = _terminal_result_payload(keys, published_terminal_results, observed_job_keys)
            if terminal is not None:
                event, payload = terminal
                yield _sse_payload(payload, event=event)
                return
        while True:
            try:
                published = await asyncio.wait_for(queue.get(), timeout=SSE_STATUS_POLL_SECONDS)
            except TimeoutError:
                yielded_update = False
                polled_status = _initial_training_status(keys, observed_job_keys)
                if polled_status is not None:
                    fingerprint = _status_fingerprint(polled_status)
                    if fingerprint != last_status_fingerprint:
                        last_status_fingerprint = fingerprint
                        last_heartbeat = asyncio.get_running_loop().time()
                        yield _sse_payload(polled_status, event="training/status")
                        yielded_update = True
                    for payload in _artifact_session_update_payloads(
                        keys,
                        published_artifacts,
                        observed_job_keys,
                        image_preview=request.imagePreview,
                    ):
                        last_heartbeat = asyncio.get_running_loop().time()
                        yield _sse_payload(payload, event="session/update")
                        yielded_update = True
                    for payload in _terminal_reply_session_update_payloads(keys, published_terminal_replies, observed_job_keys):
                        last_heartbeat = asyncio.get_running_loop().time()
                        yield _sse_payload(payload, event="session/update")
                        yielded_update = True
                    terminal = _terminal_result_payload(keys, published_terminal_results, observed_job_keys)
                    if terminal is not None:
                        event, payload = terminal
                        yield _sse_payload(payload, event=event)
                        return
                    if yielded_update:
                        continue
                now = asyncio.get_running_loop().time()
                if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                    last_heartbeat = now
                    yield ": keepalive\n\n"
                continue
            payload = _enrich_published_annotation_preview(published.payload, request)
            yield _sse_payload(payload, event=published.event)
            if published.event == "training/status":
                last_status_fingerprint = _status_fingerprint(published.payload)
            last_heartbeat = asyncio.get_running_loop().time()
            if published.event in {"result", "error"}:
                break
    finally:
        await acp_event_broker.unsubscribe(subscriber_id, keys)


def _initial_training_status(keys: set[str], observed_job_keys: set[str] | None = None) -> dict | None:
    """新订阅者进入时先返回一次状态快照，避免只能收到后续增量事件。"""
    for key in keys:
        try:
            marker = _visible_http_training_job_marker(key, observed_job_keys)
            if marker is None:
                continue
            status = _build_stream_status_for_marker(key, marker)
            if status is not None:
                return status
        except (FileNotFoundError, ValueError):
            continue
    return None


def _artifact_session_update_payloads(
    keys: set[str],
    published_artifacts: set[str],
    observed_job_keys: set[str] | None = None,
    *,
    image_preview: AcpImagePreviewRequest | None = None,
) -> list[dict]:
    for key in sorted(keys):
        try:
            marker = _visible_http_training_job_marker(key, observed_job_keys)
            if marker is None:
                continue
            status = _build_full_status_for_marker(key, marker)
            if status is None:
                continue
        except (FileNotFoundError, ValueError):
            continue
        return [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": key,
                    "update": update,
                },
            }
            for update in build_training_artifact_session_updates(
                thread_id=key,
                status=status,
                artifact_store=model_store,
                published_artifacts=published_artifacts,
                preview_width=image_preview.width if image_preview else None,
                preview_height=image_preview.height if image_preview else None,
            )
        ]
    return []


def _enrich_published_annotation_preview(payload: dict, request: AcpHttpSubscriptionRequest) -> dict:
    if payload.get("method") != "session/update":
        return payload
    params = payload.get("params")
    if not isinstance(params, dict):
        return payload
    update = params.get("update")
    if not isinstance(update, dict) or not isinstance(update.get("artifact"), dict):
        return payload
    session_id = str(params.get("sessionId") or request.threadId or request.sessionId or "").strip()
    if not session_id:
        return payload
    enriched_payload = copy.deepcopy(payload)
    enriched_params = enriched_payload["params"]
    enriched_params["update"] = add_annotation_preview_to_update(
        enriched_params["update"],
        thread_id=session_id,
        artifact_store=model_store,
        preview_width=request.imagePreview.width if request.imagePreview else None,
        preview_height=request.imagePreview.height if request.imagePreview else None,
    )
    return enriched_payload


def _terminal_reply_session_update_payloads(
    keys: set[str],
    published_replies: set[str],
    observed_job_keys: set[str] | None = None,
) -> list[dict]:
    for key in sorted(keys):
        marker = _visible_http_training_job_marker(key, observed_job_keys)
        if marker is None:
            continue
        status_value = str(marker.get("status") or "")
        if status_value not in TERMINAL_HTTP_TRAINING_STATUSES:
            continue
        reply_key = ":".join(
            [
                key,
                str(marker.get("job_id") or ""),
                status_value,
                str(marker.get("completed_at") or marker.get("updated_at") or ""),
            ]
        )
        if reply_key in published_replies:
            continue
        try:
            status = _build_stream_status_for_marker(key, marker) or {}
        except (FileNotFoundError, ValueError):
            status = {}
        text = _terminal_reply_text(marker, status)
        if not text:
            continue
        published_replies.add(reply_key)
        return [
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": key,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                        "_meta": {
                            "jetlinksRuntimeEvent": {
                                "type": "http.training_job.reply",
                                "data": {
                                    "job_id": marker.get("job_id"),
                                    "thread_id": key,
                                    "status": status_value,
                                },
                            }
                        },
                    },
                },
            }
        ]
    return []


def _terminal_result_payload(
    keys: set[str],
    published_results: set[str],
    observed_job_keys: set[str] | None = None,
) -> tuple[str, dict] | None:
    for key in sorted(keys):
        marker = _visible_http_training_job_marker(key, observed_job_keys)
        if marker is None:
            continue
        status_value = str(marker.get("status") or "")
        if status_value not in TERMINAL_HTTP_TRAINING_STATUSES:
            continue
        result_key = ":".join(
            [
                key,
                str(marker.get("job_id") or ""),
                status_value,
                str(marker.get("completed_at") or marker.get("updated_at") or ""),
            ]
        )
        if result_key in published_results:
            continue
        try:
            status = _build_stream_status_for_marker(key, marker) or {}
        except (FileNotFoundError, ValueError):
            status = {}
        published_results.add(result_key)
        return (
            "result",
            {
                "jsonrpc": "2.0",
                "id": marker.get("job_id") or f"thread-{key}",
                "result": _terminal_result(marker, key, status),
            },
        )
    return None


def _visible_http_training_job_marker(thread_id: str, observed_job_keys: set[str] | None = None) -> dict | None:
    marker = read_current_http_training_job_marker(thread_id, require_active=False)
    if marker is None:
        return None
    status = str(marker.get("status") or "")
    if status in ACTIVE_HTTP_TRAINING_STATUSES:
        if observed_job_keys is not None:
            observed_job_keys.add(_marker_job_key(thread_id, marker))
        return marker
    if (
        status in TERMINAL_HTTP_TRAINING_STATUSES
        and observed_job_keys is not None
        and _marker_job_key(thread_id, marker) in observed_job_keys
        and _recent_terminal_marker(marker)
    ):
        return marker
    return None


def _build_stream_status_for_marker(thread_id: str, marker: dict) -> dict | None:
    run_id = _marker_run_id(marker)
    if not run_id:
        return None
    status = build_training_stream_status(thread_id, run_id=run_id)
    return status if _status_belongs_to_marker(status, marker) else None


def _build_full_status_for_marker(thread_id: str, marker: dict) -> dict | None:
    run_id = _marker_run_id(marker)
    if not run_id:
        return None
    status = build_training_status(thread_id, run_id=run_id)
    return status if _status_belongs_to_marker(status, marker) else None


def _status_belongs_to_marker(status: dict, marker: dict) -> bool:
    marker_run_id = _marker_run_id(marker)
    status_run_id = str(status.get("run_id") or "").strip()
    if marker_run_id:
        return status_run_id == marker_run_id
    marker_started = _parse_utc_timestamp(str(marker.get("started_at") or marker.get("created_at") or ""))
    status_started = _parse_utc_timestamp(str(status.get("started_at") or status.get("created_at") or ""))
    if marker_started is None or status_started is None:
        return False
    return status_started.timestamp() + 5.0 >= marker_started.timestamp()


def _marker_run_id(marker: dict) -> str:
    return str(marker.get("run_id") or "").strip()


def _marker_job_key(thread_id: str, marker: dict) -> str:
    return ":".join([thread_id, str(marker.get("job_id") or ""), str(marker.get("created_at") or "")])


def _recent_terminal_marker(marker: dict) -> bool:
    timestamp = _parse_utc_timestamp(str(marker.get("completed_at") or marker.get("updated_at") or ""))
    if timestamp is None:
        return False
    age = (datetime.now(UTC) - timestamp).total_seconds()
    return age <= SSE_TERMINAL_REPLAY_SECONDS


def _parse_utc_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _terminal_reply_text(marker: dict, status: dict) -> str:
    marker_reply = _marker_result_reply(marker)
    if marker_reply:
        return marker_reply
    status_value = str(marker.get("status") or status.get("status") or "")
    if status_value == "completed":
        run_id = str(marker.get("run_id") or status.get("run_id") or "").strip()
        if run_id:
            return f"训练已完成，模型产物已生成。run_id: {run_id}"
        return "训练已完成，模型产物已生成。"
    if status_value == "cancelled":
        return str(marker.get("error") or "训练任务已取消。")
    errors = status.get("errors") if isinstance(status.get("errors"), list) else []
    if errors:
        return str(errors[-1])
    return str(marker.get("error") or "训练任务失败。")


def _terminal_result(marker: dict, thread_id: str, status: dict) -> dict:
    marker_result = _marker_result(marker)
    if marker_result:
        result = dict(marker_result)
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        result["metadata"] = {
            **metadata,
            "job_id": metadata.get("job_id") or marker.get("job_id"),
            "run_id": metadata.get("run_id") or marker.get("run_id") or status.get("run_id"),
        }
        result.setdefault("agent", marker.get("agent_name") or "default")
        result.setdefault("thread_id", thread_id)
        result.setdefault("content", [{"type": "text", "text": str(result.get("reply") or "")}])
        result.setdefault("artifacts", [])
        result.setdefault("verification", None)
        result.setdefault("spec", None)
        return result
    status_value = str(marker.get("status") or status.get("status") or "")
    reply = _terminal_reply_text(marker, status)
    result_status = "completed" if status_value == "completed" else "failed"
    metadata = {
        "phase": "cancelled" if status_value == "cancelled" else status_value,
        "status": status_value,
        "job_id": marker.get("job_id"),
        "run_id": marker.get("run_id") or status.get("run_id"),
    }
    if status_value == "cancelled":
        metadata["cancelled"] = True
    return {
        "agent": marker.get("agent_name") or "default",
        "thread_id": thread_id,
        "status": result_status,
        "reply": reply,
        "content": [{"type": "text", "text": reply}] if reply else [],
        "artifacts": [],
        "verification": None,
        "spec": None,
        "metadata": metadata,
    }


def _marker_result(marker: dict) -> dict:
    result = marker.get("result")
    return result if isinstance(result, dict) else {}


def _marker_result_reply(marker: dict) -> str:
    return str(_marker_result(marker).get("reply") or "").strip()


def _status_fingerprint(status: dict) -> str:
    return json.dumps(_stable_status_payload(status), ensure_ascii=False, sort_keys=True, default=str)


def _stable_status_payload(status: dict) -> dict:
    info = status.get("info") if isinstance(status.get("info"), dict) else {}
    metrics = info.get("metrics") if isinstance(info.get("metrics"), dict) else {}
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
        "completed_at": status.get("completed_at"),
        "info": {
            "stage": info.get("stage"),
            "status": info.get("status"),
            "metrics": metrics,
        },
        "evaluation": status.get("evaluation") if isinstance(status.get("evaluation"), dict) else {},
        "errors": status.get("errors") if isinstance(status.get("errors"), list) else [],
        "warnings": status.get("warnings") if isinstance(status.get("warnings"), list) else [],
    }


@router.post("/stream", dependencies=[Depends(require_runtime_token)])
async def acp_http_stream(request: AcpHttpSubscriptionRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_acp_subscription(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/models/{thread_id}", dependencies=[Depends(require_runtime_token)])
async def upload_training_model(thread_id: str, file: UploadFile = File(...)) -> dict[str, object]:
    original_name = _safe_model_filename(file.filename or "model.pt")
    if Path(original_name).suffix.lower() not in {".pt", ".pth"}:
        await file.close()
        raise HTTPException(status_code=400, detail="Only .pt and .pth model files are supported.")

    paths = model_store.prepare_thread(thread_id)
    target = _unique_model_path(paths.uploads / "models", original_name)
    size = 0
    digest = hashlib.sha256()

    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_UPLOAD_BYTES:
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Model file is too large. Max upload size is "
                            f"{MAX_MODEL_UPLOAD_BYTES // 1024 // 1024} MB."
                        ),
                    )
                digest.update(chunk)
                await asyncio.to_thread(handle.write, chunk)
    finally:
        await file.close()

    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded model file is empty.")

    relative = target.relative_to(paths.uploads).as_posix()
    virtual_path = f"{VIRTUAL_UPLOADS_PREFIX}/{relative}"
    sha256 = digest.hexdigest()
    model_id = f"model-{uuid.uuid4().hex}"
    attachment = {
        "name": target.name,
        "path": virtual_path,
        "mime_type": "application/octet-stream",
        "metadata": {
            "modelId": model_id,
            "role": "training_model",
            "source": "user_upload",
            "sha256": sha256,
            "size": size,
        },
    }
    model_record = {
        "modelId": model_id,
        "source": "user_upload",
        "name": target.name,
        "original_name": original_name,
        "path": virtual_path,
        "local_path": str(target.resolve()),
        "mime_type": "application/octet-stream",
        "size": size,
        "sha256": sha256,
        "uploaded_at": datetime.now(UTC).isoformat(),
    }
    _register_training_model(paths.workspace, model_record)
    return {
        "modelId": model_id,
        "thread_id": paths.thread_id,
        "name": target.name,
        "original_name": original_name,
        "path": virtual_path,
        "mime_type": "application/octet-stream",
        "size": size,
        "sha256": sha256,
        "attachment": attachment,
    }
