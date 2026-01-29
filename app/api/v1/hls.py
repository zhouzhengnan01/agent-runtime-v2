"""
RTSP/RTMP -> HLS (m3u8) 轻量转发接口
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.shared.jetlinks_video.utils.ffmpeg.python_ffmpeg_utils import (
    ensure_ffmpeg,
    ffbin,
    input_args_for,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hls", tags=["HLS"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HLS_ROOT = PROJECT_ROOT / "storage" / "static" / "hls"

DEFAULT_SEGMENT_TIME = int(os.getenv("HLS_SEGMENT_TIME", "2") or "2")
DEFAULT_LIST_SIZE = int(os.getenv("HLS_LIST_SIZE", "6") or "6")
DEFAULT_TRANSCODE = (os.getenv("HLS_TRANSCODE", "") or "").lower() in {"1", "true", "yes"}
DEFAULT_TRANSCODE_CODEC = (os.getenv("HLS_TRANSCODE_CODEC", "libx264") or "libx264").strip()
DEFAULT_TRANSCODE_PRESET = (os.getenv("HLS_TRANSCODE_PRESET", "veryfast") or "").strip()
DEFAULT_TRANSCODE_TUNE = (os.getenv("HLS_TRANSCODE_TUNE", "zerolatency") or "").strip()
DEFAULT_TRANSCODE_PIX_FMT = (os.getenv("HLS_TRANSCODE_PIX_FMT", "") or "").strip()
START_WAIT_SEC = float(os.getenv("HLS_START_WAIT_SEC", "2.0") or "2.0")


@dataclass
class HlsProcess:
    stream_id: str
    url: str
    process: subprocess.Popen
    output_dir: Path
    playlist_path: Path
    playlist_url: str
    log_path: Path
    cmd: List[str]
    started_at: float


_STREAMS: Dict[str, HlsProcess] = {}
_URL_INDEX: Dict[str, str] = {}
_LOCK = threading.Lock()


class HlsStartRequest(BaseModel):
    url: str = Field(..., description="RTSP/RTMP/HTTP 流地址")
    stream_id: Optional[str] = Field(None, description="自定义流ID")
    force_restart: bool = Field(False, description="强制重启已有进程")
    transcode: Optional[bool] = Field(None, description="强制转码为 H264")
    segment_time: Optional[int] = Field(None, description="HLS 切片时长（秒）")
    list_size: Optional[int] = Field(None, description="HLS 列表长度（段数）")


class HlsStopRequest(BaseModel):
    stream_id: Optional[str] = Field(None, description="流ID")
    url: Optional[str] = Field(None, description="流地址")
    cleanup: bool = Field(False, description="停止后清理切片文件")


def _mask_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if parsed.username or parsed.password:
            netloc = parsed.hostname or ""
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            return parsed._replace(netloc=netloc).geturl()
    except Exception:
        return url
    return url


def _sanitize_stream_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return cleaned[:64] if cleaned else ""


def _hash_stream_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"stream_{digest}"


def _resolve_stream_id(stream_id: Optional[str], url: Optional[str]) -> str:
    if stream_id:
        return _sanitize_stream_id(stream_id)
    if url:
        with _LOCK:
            mapped = _URL_INDEX.get(url)
        return mapped or _hash_stream_id(url)
    return ""


def _is_running(item: HlsProcess) -> bool:
    return item.process and item.process.poll() is None


def _cleanup_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.is_file() and child.suffix in {".ts", ".m3u8", ".log"}:
            try:
                child.unlink()
            except Exception:
                continue


def _build_hls_command(
    url: str,
    output_dir: Path,
    *,
    segment_time: int,
    list_size: int,
    transcode: bool,
) -> List[str]:
    def _strip_hwaccel_args(args: List[str]) -> List[str]:
        cleaned: List[str] = []
        skip = False
        for item in args:
            if skip:
                skip = False
                continue
            if item in {"-hwaccel", "-c:v"}:
                skip = True
                continue
            cleaned.append(item)
        return cleaned

    playlist_path = output_dir / "index.m3u8"
    segment_template = str(output_dir / "seg_%05d.ts")
    input_args = input_args_for(url)
    if transcode and DEFAULT_TRANSCODE_CODEC.endswith("_bm"):
        input_args = _strip_hwaccel_args(input_args)
    cmd: List[str] = [
        ffbin(),
        "-hide_banner",
        "-loglevel",
        "warning",
        *input_args,
        "-i",
        url,
    ]
    if transcode:
        codec = DEFAULT_TRANSCODE_CODEC or "libx264"
        cmd += ["-c:v", codec]
        if codec == "libx264":
            if DEFAULT_TRANSCODE_PRESET:
                cmd += ["-preset", DEFAULT_TRANSCODE_PRESET]
            if DEFAULT_TRANSCODE_TUNE:
                cmd += ["-tune", DEFAULT_TRANSCODE_TUNE]
        pix_fmt = DEFAULT_TRANSCODE_PIX_FMT or "yuv420p"
        if codec.endswith("_bm") and pix_fmt != "bmcodec":
            cmd += ["-vf", f"scale_bm=format={pix_fmt}"]
        cmd += ["-pix_fmt", pix_fmt]
    else:
        cmd += ["-c:v", "copy"]
    cmd += [
        "-an",
        "-f",
        "hls",
        "-hls_time",
        str(segment_time),
        "-hls_list_size",
        str(list_size),
        "-hls_flags",
        "delete_segments+program_date_time+omit_endlist",
        "-hls_segment_filename",
        segment_template,
        str(playlist_path),
    ]
    return cmd


def _wait_for_playlist(process: subprocess.Popen, playlist_path: Path, timeout_sec: float) -> bool:
    if timeout_sec <= 0:
        return playlist_path.exists()
    start = time.time()
    while time.time() - start < timeout_sec:
        if playlist_path.exists() and playlist_path.stat().st_size > 0:
            return True
        if process.poll() is not None:
            return False
        time.sleep(0.2)
    return playlist_path.exists() and playlist_path.stat().st_size > 0


def _stop_process(item: HlsProcess, *, cleanup: bool = False) -> None:
    if item.process and item.process.poll() is None:
        try:
            item.process.terminate()
            item.process.wait(timeout=3)
        except Exception:
            try:
                item.process.kill()
            except Exception:
                pass
    if cleanup:
        _cleanup_outputs(item.output_dir)


@router.post("/start")
async def start_hls(request: HlsStartRequest):
    url = (request.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url 不能为空")

    try:
        ensure_ffmpeg()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg 未就绪: {e}")

    stream_id = _resolve_stream_id(request.stream_id, url)
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id 无效")

    HLS_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = (HLS_ROOT / stream_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    playlist_path = output_dir / "index.m3u8"
    playlist_url = f"/static/hls/{stream_id}/index.m3u8"

    with _LOCK:
        existing = _STREAMS.get(stream_id)
        if existing and _is_running(existing) and not request.force_restart:
            return {
                "success": True,
                "data": {
                    "stream_id": stream_id,
                    "playlist_url": existing.playlist_url,
                    "running": True,
                    "ready": playlist_path.exists(),
                    "reused": True,
                },
            }

        if existing:
            _stop_process(existing, cleanup=False)
            _STREAMS.pop(stream_id, None)

    segment_time = request.segment_time or DEFAULT_SEGMENT_TIME
    list_size = request.list_size or DEFAULT_LIST_SIZE
    transcode = DEFAULT_TRANSCODE if request.transcode is None else request.transcode

    _cleanup_outputs(output_dir)
    cmd = _build_hls_command(
        url,
        output_dir,
        segment_time=segment_time,
        list_size=list_size,
        transcode=transcode,
    )
    log_path = output_dir / "ffmpeg.log"
    log_file = None
    try:
        log_file = open(log_path, "a", encoding="utf-8")
    except Exception:
        log_file = None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log_file or subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动 ffmpeg 失败: {e}")
    finally:
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass

    item = HlsProcess(
        stream_id=stream_id,
        url=url,
        process=proc,
        output_dir=output_dir,
        playlist_path=playlist_path,
        playlist_url=playlist_url,
        log_path=log_path,
        cmd=cmd,
        started_at=time.time(),
    )

    with _LOCK:
        _STREAMS[stream_id] = item
        _URL_INDEX[url] = stream_id

    ready = _wait_for_playlist(proc, playlist_path, START_WAIT_SEC)
    logger.info("HLS start: %s -> %s (ready=%s)", _mask_url(url), playlist_url, ready)

    return {
        "success": True,
        "data": {
            "stream_id": stream_id,
            "playlist_url": playlist_url,
            "running": _is_running(item),
            "ready": ready,
            "reused": False,
        },
    }


@router.post("/stop")
async def stop_hls(request: HlsStopRequest):
    stream_id = _resolve_stream_id(request.stream_id, request.url)
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id/url 不能为空")

    with _LOCK:
        item = _STREAMS.pop(stream_id, None)
        if request.url:
            _URL_INDEX.pop(request.url, None)

    if not item:
        raise HTTPException(status_code=404, detail="未找到对应流")

    _stop_process(item, cleanup=request.cleanup)
    logger.info("HLS stop: %s", stream_id)

    return {
        "success": True,
        "data": {
            "stream_id": stream_id,
            "running": False,
            "playlist_url": item.playlist_url,
        },
    }


@router.get("/status")
async def status_hls(stream_id: Optional[str] = None, url: Optional[str] = None):
    resolved = _resolve_stream_id(stream_id, url)
    if not resolved:
        raise HTTPException(status_code=400, detail="stream_id/url 不能为空")

    with _LOCK:
        item = _STREAMS.get(resolved)

    if not item:
        raise HTTPException(status_code=404, detail="未找到对应流")

    running = _is_running(item)
    ready = item.playlist_path.exists() and item.playlist_path.stat().st_size > 0
    if not running:
        with _LOCK:
            _STREAMS.pop(resolved, None)

    return {
        "success": True,
        "data": {
            "stream_id": resolved,
            "playlist_url": item.playlist_url,
            "running": running,
            "ready": ready,
            "started_at": item.started_at,
        },
    }
