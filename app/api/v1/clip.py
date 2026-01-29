"""
Video clip capture API.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.v1.hls import HlsStartRequest, start_hls
from app.config import settings
from app.schemas.pagination import StandardResponse
from app.services.review_record_service import get_review_record_service


router = APIRouter(prefix="/clip", tags=["Clip"])

CLIP_ROOT = Path(os.getenv("PATROL_CLIP_ROOT", "storage/http_tmp/patrol_review")).resolve()
CLIP_TTL_HOURS = int(os.getenv("PATROL_CLIP_TTL_HOURS", "24") or "24")


class ClipRequest(BaseModel):
    url: str = Field(..., description="视频流URL（RTSP/RTMP/HLS/HTTP）")
    clip_seconds: int = Field(10, ge=1, le=120, description="剪辑时长（秒）")
    stream_id: Optional[str] = Field(None, description="可选：流ID，用于命名目录")
    prefer_hls: bool = Field(True, description="RTSP/RTMP时优先使用本地HLS")
    force_hls: bool = Field(False, description="强制要求HLS成功，否则失败")


def _sanitize_stream_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return cleaned[:64] if cleaned else ""


def _hash_stream_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"stream_{digest}"


def _is_live_stream(url: str) -> bool:
    scheme = (urlparse(url).scheme or "").lower()
    return scheme in {"rtsp", "rtsps", "rtmp", "rtmps"}


def _build_local_base_url() -> str:
    raw = (os.getenv("PATROL_CLIP_LOCAL_BASE_URL") or "").strip()
    if raw:
        return raw.rstrip("/")
    return f"http://127.0.0.1:{settings.PORT}"


def _cleanup_old_files(root: Path, ttl_hours: int) -> None:
    if ttl_hours <= 0:
        return
    cutoff = time.time() - (ttl_hours * 3600)
    try:
        for p in root.rglob("*.mp4"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except Exception:
                continue
    except Exception:
        return


@router.post("", response_model=StandardResponse)
async def create_clip(request: ClipRequest) -> StandardResponse:
    url = (request.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url 不能为空")

    stream_id = _sanitize_stream_id(request.stream_id or "") or _hash_stream_id(url)
    clip_dir = (CLIP_ROOT / stream_id).resolve()
    clip_dir.mkdir(parents=True, exist_ok=True)

    filename = f"clip_{int(time.time() * 1000)}.mp4"
    output_path = clip_dir / filename
    clip_url = f"/storage/http_tmp/patrol_review/{stream_id}/{filename}"

    capture_url = url
    hls_meta: Optional[Dict[str, Any]] = None
    if request.prefer_hls and _is_live_stream(url):
        try:
            hls_resp = await start_hls(HlsStartRequest(url=url, stream_id=stream_id))
            hls_meta = hls_resp.get("data") if isinstance(hls_resp, dict) else None
            playlist_url = (hls_meta or {}).get("playlist_url")
            ready = bool((hls_meta or {}).get("ready"))
            if not ready:
                if request.force_hls:
                    raise HTTPException(status_code=500, detail="HLS 未就绪")
            elif isinstance(playlist_url, str) and playlist_url:
                if playlist_url.startswith(("http://", "https://")):
                    capture_url = playlist_url
                else:
                    capture_url = _build_local_base_url() + playlist_url
        except Exception as e:
            if request.force_hls:
                raise HTTPException(status_code=500, detail=f"HLS启动失败: {e}")

    store = get_review_record_service()
    ok, err = await store.record_clip(
        source_url=capture_url,
        output_path=output_path,
        clip_seconds=request.clip_seconds,
    )
    if not ok:
        raise HTTPException(status_code=500, detail=err or "ffmpeg failed")

    _cleanup_old_files(CLIP_ROOT, CLIP_TTL_HOURS)

    return StandardResponse(
        message="success",
        result={
            "url": clip_url,
            "file_path": str(output_path),
            "source_url": url,
            "capture_url": capture_url,
            "clip_seconds": request.clip_seconds,
            "stream_id": stream_id,
            "hls": hls_meta or None,
        },
        status=200,
        timestamp=int(time.time() * 1000),
    )
