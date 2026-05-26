from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.core.artifacts import ArtifactStore
from app.core.artifacts.preview import guess_mime_type


router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])
store = ArtifactStore()

ACTIVE_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "image/svg+xml"}


@router.get("/{thread_id}")
async def list_artifacts(thread_id: str) -> dict[str, object]:
    return {"thread_id": thread_id, "artifacts": [artifact.model_dump() for artifact in store.list_artifacts(thread_id)]}


@router.get("/{thread_id}/{path:path}")
async def get_artifact(thread_id: str, path: str, download: bool = False) -> Response:
    try:
        actual_path = store.resolve_virtual_path(thread_id, path)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not actual_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")
    if not actual_path.is_file():
        raise HTTPException(status_code=400, detail=f"Artifact path is not a file: {path}")

    mime_type = guess_mime_type(actual_path)
    if download or mime_type in ACTIVE_CONTENT_TYPES:
        return FileResponse(actual_path, media_type=mime_type, filename=actual_path.name)
    if mime_type.startswith("text/") or _looks_like_text(actual_path):
        return PlainTextResponse(actual_path.read_text(encoding="utf-8"), media_type=mime_type)
    return Response(actual_path.read_bytes(), media_type=mime_type)


def _looks_like_text(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8192]
    except OSError:
        return False
    return b"\x00" not in chunk

