from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.auth import require_runtime_token
from app.core.artifacts import ArtifactStore
from app.core.artifacts.preview import guess_mime_type


router = APIRouter(prefix="/api/uploads", tags=["uploads"])
store = ArtifactStore()

VIRTUAL_UPLOADS_PREFIX = "/mnt/user-data/uploads"
MAX_UPLOAD_GIB = 3
MAX_UPLOAD_BYTES = MAX_UPLOAD_GIB * 1024 * 1024 * 1024
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_. -]+")


@router.post("/{thread_id}", dependencies=[Depends(require_runtime_token)])
async def upload_thread_file(
    thread_id: str,
    file: UploadFile = File(...),
    relative_path: str | None = Form(default=None),
) -> dict[str, object]:
    paths = store.prepare_thread(thread_id)
    relative_parts = _safe_relative_parts(relative_path)
    filename = _safe_filename(relative_parts[-1] if relative_parts else file.filename or "upload.bin")
    target_dir = paths.uploads.joinpath(*relative_parts[:-1]) if relative_parts else paths.uploads
    target = _unique_upload_path(target_dir, filename)
    size = 0

    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Max upload size is {MAX_UPLOAD_GIB} GiB.",
                    )
                await asyncio.to_thread(handle.write, chunk)
    finally:
        await file.close()

    relative = target.relative_to(paths.uploads).as_posix()
    mime_type = file.content_type or guess_mime_type(target)
    return {
        "name": target.name,
        "original_name": filename,
        "path": f"{VIRTUAL_UPLOADS_PREFIX}/{relative}",
        "mime_type": mime_type,
        "size": size,
    }


def _safe_filename(value: str) -> str:
    name = Path(value.replace("\\", "/")).name.strip().strip(".")
    cleaned = _SAFE_FILENAME_RE.sub("-", name).strip(". ")
    return cleaned[:180] or "upload.bin"


def _safe_relative_parts(value: str | None) -> list[str]:
    if not value:
        return []
    parts: list[str] = []
    for part in value.replace("\\", "/").split("/"):
        cleaned = _safe_filename(part)
        if cleaned in {"", ".", ".."}:
            continue
        parts.append(cleaned)
    return parts[:12]


def _unique_upload_path(root: Path, filename: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem or "upload"
    suffix = candidate.suffix
    return root / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"
