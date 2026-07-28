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
_TRAINING_INPUT_CANONICAL_NAMES = {
    "dataset.zip": "datasets.zip",
    "datasets.zip": "datasets.zip",
    "image1.zip": "image1.zip",
    "image2.zip": "image2.zip",
}


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
    canonical_name = _canonical_training_input_name(target_dir, paths.uploads, filename)
    target = (
        target_dir / canonical_name
        if canonical_name
        else _unique_upload_path(target_dir, filename)
    )
    replaced = bool(canonical_name and _training_input_exists(target_dir, canonical_name))
    temporary = _temporary_upload_path(target_dir, target.name)
    size = 0

    try:
        with temporary.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Max upload size is {MAX_UPLOAD_GIB} GiB.",
                    )
                await asyncio.to_thread(handle.write, chunk)
        temporary.replace(target)
        if canonical_name == "datasets.zip":
            (target_dir / "dataset.zip").unlink(missing_ok=True)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
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
        "replaced": replaced,
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


def _canonical_training_input_name(target_dir: Path, uploads_dir: Path, filename: str) -> str | None:
    if target_dir != uploads_dir:
        return None
    return _TRAINING_INPUT_CANONICAL_NAMES.get(filename.casefold())


def _training_input_exists(target_dir: Path, canonical_name: str) -> bool:
    if (target_dir / canonical_name).is_file():
        return True
    return canonical_name == "datasets.zip" and (target_dir / "dataset.zip").is_file()


def _temporary_upload_path(root: Path, filename: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    return root / f".{filename}.{uuid.uuid4().hex}.uploading"
