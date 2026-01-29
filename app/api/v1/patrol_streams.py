"""
Patrol stream reachability check.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.schemas.pagination import StandardResponse


router = APIRouter(prefix="/patrol-streams", tags=["Patrol Streams"])

DEFAULT_TIMEOUT_MS = int(os.getenv("PATROL_STREAM_CHECK_TIMEOUT_MS", "800") or "800")
MAX_URLS = int(os.getenv("PATROL_STREAM_CHECK_MAX", "800") or "800")
CACHE_TTL_SEC = float(os.getenv("PATROL_STREAM_CHECK_TTL_SEC", "30") or "30")
MAX_CONCURRENCY = int(os.getenv("PATROL_STREAM_CHECK_CONCURRENCY", "50") or "50")

DEFAULT_PORTS = {
    "rtsp": 554,
    "rtsps": 554,
    "rtmp": 1935,
    "rtmps": 1935,
    "http": 80,
    "https": 443,
}

_CACHE: Dict[str, Dict[str, object]] = {}


class StreamCheckRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, description="流地址列表")
    timeout_ms: Optional[int] = Field(None, description="单个探测超时毫秒")


def _cache_get(url: str, now: float) -> Optional[Dict[str, object]]:
    item = _CACHE.get(url)
    if not item:
        return None
    if now - float(item.get("ts", 0)) > CACHE_TTL_SEC:
        return None
    return item


def _cache_set(url: str, reachable: bool, reason: Optional[str], now: float) -> None:
    _CACHE[url] = {"ts": now, "reachable": reachable, "reason": reason}


def _parse_host_port(url: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = parsed.hostname
    if not host:
        return None, None, "invalid_url"
    port = parsed.port or DEFAULT_PORTS.get(scheme)
    if not port:
        return None, None, "unsupported_scheme"
    return host, port, None


async def _check_tcp(host: str, port: int, timeout_sec: float) -> Tuple[bool, Optional[str]]:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout_sec,
        )
        writer.close()
        if hasattr(writer, "wait_closed"):
            await writer.wait_closed()
        return True, None
    except Exception:
        return False, "connect_failed"


@router.post("/check", response_model=StandardResponse)
async def check_patrol_streams(request: StreamCheckRequest) -> StandardResponse:
    raw_urls = [u.strip() for u in (request.urls or []) if u and u.strip()]
    if not raw_urls:
        return StandardResponse(
            message="success",
            result={"items": [], "checkedAt": int(time.time() * 1000)},
            status=200,
            timestamp=int(time.time() * 1000),
        )
    urls = raw_urls[:MAX_URLS]
    timeout_sec = max(0.1, min(5.0, (request.timeout_ms or DEFAULT_TIMEOUT_MS) / 1000.0))
    now = time.time()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def _check(url: str) -> Dict[str, object]:
        cached = _cache_get(url, now)
        if cached:
            return {
                "url": url,
                "reachable": bool(cached.get("reachable")),
                "reason": cached.get("reason"),
            }

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme in {"", "file"}:
            path = parsed.path if scheme == "file" else url
            ok = bool(path) and os.path.isabs(path) and os.path.exists(path)
            reason = None if ok else "file_missing"
            _cache_set(url, ok, reason, now)
            return {"url": url, "reachable": ok, "reason": reason}

        host, port, reason = _parse_host_port(url)
        if reason:
            _cache_set(url, False, reason, now)
            return {"url": url, "reachable": False, "reason": reason}

        async with sem:
            ok, err = await _check_tcp(host, int(port), timeout_sec)
        _cache_set(url, ok, err, now)
        payload: Dict[str, object] = {"url": url, "reachable": ok}
        if err:
            payload["reason"] = err
        return payload

    items = await asyncio.gather(*[_check(url) for url in urls])
    return StandardResponse(
        message="success",
        result={"items": items, "checkedAt": int(time.time() * 1000)},
        status=200,
        timestamp=int(time.time() * 1000),
    )
