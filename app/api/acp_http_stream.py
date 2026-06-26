from __future__ import annotations

import asyncio
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
from pydantic import BaseModel

from app.api.auth import require_runtime_token
from app.core.artifacts import ArtifactStore
from app.protocols.acp.event_broker import acp_event_broker


router = APIRouter(prefix="/api/acp", tags=["acp"])
SSE_HEARTBEAT_SECONDS = 10.0
MAX_MODEL_UPLOAD_BYTES = max(
    1,
    int(os.getenv("JETLINKS_MAX_MODEL_UPLOAD_BYTES", str(512 * 1024 * 1024))),
)
VIRTUAL_UPLOADS_PREFIX = "/mnt/user-data/uploads"
_SAFE_MODEL_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_. -]+")
model_store = ArtifactStore()


class AcpHttpSubscriptionRequest(BaseModel):
    sessionId: str | None = None
    threadId: str | None = None


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
        while True:
            try:
                published = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield _sse_payload(published.payload, event=published.event)
            if published.event in {"result", "error"}:
                break
    finally:
        await acp_event_broker.unsubscribe(subscriber_id, keys)


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
