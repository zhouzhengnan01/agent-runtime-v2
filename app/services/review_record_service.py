"""
Review record persistence (filesystem-based).

Stores the latest review calls (request/response/media) on disk and provides
helpers for listing, reading, and post-processing (video recording).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from app.config import settings

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _parse_time_ms(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if not value:
            return None
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    iso = s.replace(" ", "T")
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _coerce_created_at_ms(record: Dict[str, Any]) -> int:
    if not isinstance(record, dict):
        return 0
    parsed = _parse_time_ms(record.get("created_at"))
    if parsed is not None:
        return parsed
    parsed = _parse_time_ms(record.get("created_at_ms"))
    return parsed or 0


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def _safe_text(value: Any, *, max_chars: int) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…(truncated, original_length={len(text)})"


def _coerce_hit(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "hit", "match", "matched"}:
            return 1
        if s in {"0", "false", "no", "n", "miss", "unmatch", "unmatched"}:
            return 0
    return None


def _mask_data_uri(url: str, *, max_chars: int = 256) -> str:
    if not isinstance(url, str):
        return ""
    if not url.startswith("data:"):
        return url
    return _safe_text(url, max_chars=max_chars)


def _sanitize_files(files: Any) -> List[Dict[str, Any]]:
    if not files:
        return []
    sanitized: List[Dict[str, Any]] = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        url = file_item.get("url")
        media_type = file_item.get("media_type") or file_item.get("mediaType")
        sanitized.append(
            {
                "url": _mask_data_uri(url) if isinstance(url, str) else "",
                "media_type": media_type,
                **{k: v for k, v in file_item.items() if k not in {"url", "media_type", "mediaType"}},
            }
        )
    return sanitized


_DATA_URI_IMAGE_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _decode_data_uri_image(data_uri: str) -> Optional[Tuple[str, bytes]]:
    if not isinstance(data_uri, str):
        return None
    data_uri = data_uri.strip()
    if not data_uri.startswith("data:"):
        return None
    match = _DATA_URI_IMAGE_RE.match(data_uri)
    if not match:
        return None
    mime = match.group("mime").strip().lower()
    if not mime.startswith("image/"):
        return None
    raw = re.sub(r"\s+", "", match.group("data") or "")
    if not raw:
        return None
    try:
        payload = base64.b64decode(raw, validate=False)
    except Exception:
        return None
    return mime, payload


def _write_raw_image(payload: bytes, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return False


def _write_image_preview(payload: bytes, path: Path, *, max_side: int = 1024) -> bool:
    try:
        from PIL import Image, ImageOps
    except Exception:
        return _write_raw_image(payload, path)
    try:
        with Image.open(BytesIO(payload)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                    background.alpha_composite(img.convert("RGBA"))
                    img = background.convert("RGB")
                else:
                    img = img.convert("RGB")
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, format="JPEG", quality=85, optimize=True)
        return path.exists() and path.stat().st_size > 0
    except Exception:
        return _write_raw_image(payload, path)


def _resolve_local_path(url: str) -> Optional[Path]:
    if not isinstance(url, str):
        return None
    raw = url.strip()
    if not raw:
        return None
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        path_str = unquote(parsed.path or "")
        if os.name == "nt" and path_str.startswith("/"):
            path_str = path_str.lstrip("/")
        candidate = Path(path_str)
    else:
        candidate = Path(os.path.expanduser(raw))
    if not candidate.is_absolute() or not candidate.exists() or not candidate.is_file():
        return None
    return candidate


def _is_image_path(path: Path) -> bool:
    mime, _ = mimetypes.guess_type(path.name)
    return bool(mime and mime.startswith("image/"))


def _env_flag(*names: str) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
    return False


def _transcode_settings() -> Tuple[str, str, str, str]:
    codec = (
        os.getenv("REVIEW_RECORDS_TRANSCODE_CODEC")
        or os.getenv("HLS_TRANSCODE_CODEC")
        or "libx264"
    ).strip()
    preset = (
        os.getenv("REVIEW_RECORDS_TRANSCODE_PRESET")
        or os.getenv("HLS_TRANSCODE_PRESET")
        or "veryfast"
    ).strip()
    tune = (
        os.getenv("REVIEW_RECORDS_TRANSCODE_TUNE")
        or os.getenv("HLS_TRANSCODE_TUNE")
        or "zerolatency"
    ).strip()
    pix_fmt = (
        os.getenv("REVIEW_RECORDS_TRANSCODE_PIX_FMT")
        or os.getenv("HLS_TRANSCODE_PIX_FMT")
        or "yuv420p"
    ).strip()
    return codec, preset, tune, pix_fmt


@dataclass(frozen=True)
class ReviewRecordPaths:
    record_id: str
    record_dir: Path
    record_json: Path
    video_mp4: Path
    thumb_jpg: Path
    image_jpg: Path

    @property
    def video_static_url(self) -> str:
        return f"/storage/review_records/{self.record_id}/video.mp4"

    @property
    def thumb_static_url(self) -> str:
        return f"/storage/review_records/{self.record_id}/thumb.jpg"

    @property
    def image_static_url(self) -> str:
        return f"/storage/review_records/{self.record_id}/image.jpg"


class ReviewRecordService:
    """
    Review record store.

    - Each call creates: storage/review_records/{record_id}/record.json
    - Optionally records the first video (30s) to: .../video.mp4
    - Enforces global retention (max N records) after writes.
    """

    def __init__(self) -> None:
        self.base_path = Path(getattr(settings, "REVIEW_RECORDS_STORAGE_PATH", "storage/review_records"))
        self.max_records = int(getattr(settings, "REVIEW_RECORDS_MAX_RECORDS", 1000))
        self.clip_seconds = int(getattr(settings, "REVIEW_RECORDS_CLIP_SECONDS", 30))
        self.ffmpeg_bin = str(getattr(settings, "REVIEW_RECORDS_FFMPEG_BIN", "ffmpeg"))
        self.ffmpeg_timeout_seconds = int(
            getattr(settings, "REVIEW_RECORDS_FFMPEG_TIMEOUT", max(60, self.clip_seconds + 20))
        )
        self._lock_file = self.base_path / ".lock"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._fast_index: List[Tuple[int, str, Path]] = []
        self._fast_index_built_at = 0.0
        self._fast_index_base_mtime = 0.0

    @staticmethod
    def _is_useless_summary(text: Any) -> bool:
        s = str(text or "").strip()
        if not s:
            return True
        return bool(re.fullmatch(r"[\s,，。\.、;；:：\-—_]+", s))

    def _summarize_record(self, rec: Dict[str, Any], *, dir_name: str) -> Dict[str, Any]:
        if not isinstance(rec, dict):
            rec = {}
        created_at_ms = _coerce_created_at_ms(rec)
        record_id = str(rec.get("record_id") or dir_name)
        request = rec.get("request") or {}
        ctx = request.get("context") or {}
        params = ctx.get("parameters") or {}

        task = None
        if isinstance(ctx, dict):
            v = ctx.get("task")
            if isinstance(v, str) and v.strip():
                task = v.strip()
        if task is None and isinstance(params, dict):
            for k in ("task", "任务", "task_name", "taskName", "taskTitle", "task_title"):
                v = params.get(k)
                if isinstance(v, str) and v.strip():
                    task = v.strip()
                    break

        first_image_url: Optional[str] = None
        first_video_url: Optional[str] = None
        files = ctx.get("files") or []
        if isinstance(files, list):
            for f in files:
                if not isinstance(f, dict):
                    continue
                media_type = str(f.get("media_type") or f.get("mediaType") or "").lower().strip()
                url = f.get("url")
                if not isinstance(url, str) or not url.strip():
                    continue
                u = url.strip()
                if u.startswith("data:"):
                    continue
                if media_type == "image" and first_image_url is None:
                    if u.startswith("http://") or u.startswith("https://") or u.startswith("/"):
                        first_image_url = u
                        continue
                if media_type == "video" and first_video_url is None:
                    first_video_url = u
                    continue

        source = None
        if isinstance(params, dict):
            for k in ("source", "来源", "from", "source_name", "sourceName"):
                v = params.get(k)
                if isinstance(v, str) and v.strip():
                    source = v.strip()
                    break

        record_time = None
        if isinstance(params, dict):
            for k in (
                "record_time",
                "recordTime",
                "alarm_time",
                "alarmTime",
                "event_time",
                "eventTime",
                "timestamp",
                "ts",
            ):
                v = params.get(k)
                if v is not None and v != "":
                    record_time = v
                    break

        response = rec.get("response") or {}
        success = bool(response.get("success"))
        error = response.get("error")
        data = response.get("data") if isinstance(response, dict) else None
        hit: Optional[int] = None
        if isinstance(response, dict):
            hit = _coerce_hit(response.get("hit"))
        illegal: Optional[bool] = None
        if isinstance(data, dict):
            candidates = [
                data.get("illegal"),
                data.get("is_illegal"),
                data.get("isIllegal"),
                data.get("illegal_flag"),
                data.get("illegalFlag"),
                data.get("violation"),
                data.get("is_violation"),
                data.get("isViolation"),
            ]
            for v in candidates:
                if isinstance(v, bool):
                    illegal = v
                    break
                if isinstance(v, (int, float)) and v in (0, 1):
                    illegal = bool(v)
                    break
                if isinstance(v, str):
                    s = v.strip().lower()
                    if s in {"true", "yes", "y", "1", "illegal"}:
                        illegal = True
                        break
                    if s in {"false", "no", "n", "0", "ok", "legal"}:
                        illegal = False
                        break
        if hit is None and illegal is not None:
            hit = 1 if illegal else 0
        summary = rec.get("meta", {}).get("summary") or response.get("summary_sentence") or ""
        if not summary and isinstance(data, dict):
            for k in ("summary", "结论", "conclusion", "result"):
                if data.get(k):
                    summary = str(data.get(k))
                    break
        if not summary:
            summary = "请求成功" if success else (str(error) if error else "请求失败")
        if self._is_useless_summary(summary):
            msg = request.get("message")
            if isinstance(msg, str) and msg.strip():
                summary = _safe_text(msg.strip(), max_chars=80)

        video = rec.get("video") or {}
        video_status = str(video.get("status") or "")
        if video_status in {"queued", "recording"}:
            review_status = "pending"
        elif success:
            review_status = "done"
        else:
            review_status = "error"

        image_url = first_image_url
        video_url = None
        if isinstance(video, dict):
            v = video.get("source_url")
            if isinstance(v, str) and v.strip():
                video_url = v.strip()
        if video_url is None:
            video_url = first_video_url

        image_static_url: Optional[str] = None
        image_meta = rec.get("image")
        if isinstance(image_meta, dict):
            v = image_meta.get("static_url")
            if isinstance(v, str) and v.strip():
                image_static_url = v.strip()
        if image_static_url is None:
            try:
                paths = self._paths(record_id)
                if paths.image_jpg.exists():
                    image_static_url = paths.image_static_url
            except Exception:
                image_static_url = None

        if image_url is None:
            image_url = image_static_url

        thumbnail = None
        try:
            if video_status == "done":
                paths = self._paths(record_id)
                if paths.thumb_jpg.exists():
                    thumbnail = {"type": "video", "url": paths.thumb_static_url}
                elif isinstance(video.get("static_url"), str) and video.get("static_url"):
                    thumbnail = {"type": "video", "url": str(video.get("static_url"))}

            if thumbnail is None:
                if image_static_url:
                    thumbnail = {"type": "image", "url": image_static_url}
                elif first_image_url:
                    thumbnail = {"type": "image", "url": first_image_url}
        except Exception:
            thumbnail = None

        return {
            "record_id": record_id,
            "created_at": rec.get("created_at"),
            "created_at_ms": created_at_ms,
            "agent_id": rec.get("agent_id"),
            "task": task,
            "source": source,
            "record_time": record_time,
            "review_status": review_status,
            "success": success,
            "summary": summary,
            "hit": hit,
            "illegal": illegal,
            "image_url": image_url,
            "video_url": video_url,
            "thumbnail": thumbnail,
            "http_status": rec.get("meta", {}).get("http_status"),
            "cost_ms": rec.get("meta", {}).get("cost_ms"),
            "video": {
                "status": video.get("status"),
                "static_url": video.get("static_url"),
                "source_url": video.get("source_url"),
                "clip_seconds": video.get("clip_seconds"),
                "error": video.get("error"),
            },
        }

    def _build_transcode_command(
        self,
        *,
        source_url: str,
        seconds: int,
        input_args: List[str],
        output_path: Path,
    ) -> List[str]:
        transcode_codec, transcode_preset, transcode_tune, transcode_pix_fmt = _transcode_settings()
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            *input_args,
            "-i",
            source_url,
            "-t",
            str(seconds),
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-c:v",
            transcode_codec or "libx264",
        ]
        if transcode_codec == "libx264":
            if transcode_preset:
                cmd += ["-preset", transcode_preset]
            if transcode_tune:
                cmd += ["-tune", transcode_tune]
            cmd += ["-crf", "23"]
        else:
            pix_fmt = transcode_pix_fmt or "yuv420p"
            if transcode_codec.endswith("_bm") and pix_fmt != "bmcodec":
                cmd += ["-vf", f"scale_bm=format={pix_fmt}"]
            if pix_fmt:
                cmd += ["-pix_fmt", pix_fmt]

        cmd += [
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        return cmd

    def _paths(self, record_id: str) -> ReviewRecordPaths:
        record_dir = self.base_path / record_id
        return ReviewRecordPaths(
            record_id=record_id,
            record_dir=record_dir,
            record_json=record_dir / "record.json",
            video_mp4=record_dir / "video.mp4",
            thumb_jpg=record_dir / "thumb.jpg",
            image_jpg=record_dir / "image.jpg",
        )

    @contextmanager
    def _fs_lock(self):
        """
        Best-effort cross-process lock for retention cleanup.

        Works on Unix platforms via fcntl.flock; falls back to no-op if unavailable.
        """
        try:
            import fcntl  # type: ignore

            self.base_path.mkdir(parents=True, exist_ok=True)
            with open(self._lock_file, "a", encoding="utf-8") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    yield
                    return
                try:
                    yield
                finally:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
        except Exception:
            yield

    def _write_json_atomic(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(_json_dumps(data), encoding="utf-8")
        os.replace(tmp, path)

    def create_record(
        self,
        *,
        agent_id: str,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> ReviewRecordPaths:
        record_id = uuid.uuid4().hex
        paths = self._paths(record_id)
        paths.record_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "record_id": record_id,
            "created_at": _utcnow_iso(),
            "created_at_ms": _now_ms(),
            "agent_id": agent_id,
            "request": request_payload,
            "response": response_payload,
            "meta": meta,
            "video": {
                "source_url": None,
                "clip_seconds": self.clip_seconds,
                "status": "not_requested",
                "static_url": None,
                "file_path": None,
                "error": None,
                "recorded_at": None,
            },
        }

        self._write_json_atomic(paths.record_json, record)
        # Optional: mirror to MySQL for long-term persistence / querying.
        try:
            from app.services.review_record_db_service import get_review_record_db_service

            get_review_record_db_service().upsert_record(record)
        except Exception:
            pass
        return paths

    def update_record(self, record_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        paths = self._paths(record_id)
        if not paths.record_json.exists():
            return None
        try:
            record = json.loads(paths.record_json.read_text(encoding="utf-8"))
        except Exception:
            record = {}
        merged = {**record, **patch}
        self._write_json_atomic(paths.record_json, merged)
        # Optional: mirror to MySQL for long-term persistence / querying.
        try:
            from app.services.review_record_db_service import get_review_record_db_service

            get_review_record_db_service().upsert_record(merged)
        except Exception:
            pass
        return merged

    def save_image_preview_from_files(
        self,
        record_id: str,
        files: Optional[Iterable[Any]],
    ) -> Optional[str]:
        if not files:
            return None
        paths = self._paths(record_id)
        try:
            if paths.image_jpg.exists() and paths.image_jpg.stat().st_size > 0:
                return paths.image_static_url
        except Exception:
            pass

        for file_item in files:
            url: Optional[str] = None
            media_type: Optional[str] = None
            if isinstance(file_item, dict):
                url = file_item.get("url")
                media_type = file_item.get("media_type") or file_item.get("mediaType")
            else:
                url = getattr(file_item, "url", None)
                media_type = getattr(file_item, "media_type", None) or getattr(file_item, "mediaType", None)

            if not isinstance(url, str) or not url.strip():
                continue
            url = url.strip()

            if isinstance(media_type, str):
                media_type = media_type.strip().lower()
            else:
                media_type = ""
            if not media_type:
                guessed, _ = mimetypes.guess_type(url)
                if guessed:
                    media_type = guessed.lower()
            if media_type and not media_type.startswith("image"):
                continue

            decoded = _decode_data_uri_image(url)
            if decoded:
                _, payload = decoded
                if _write_image_preview(payload, paths.image_jpg):
                    self.update_record(record_id, {"image": {"static_url": paths.image_static_url}})
                    return paths.image_static_url
                continue

            local_path = _resolve_local_path(url)
            if local_path:
                if not (media_type.startswith("image") or _is_image_path(local_path)):
                    continue
                try:
                    payload = local_path.read_bytes()
                except Exception:
                    continue
                if _write_image_preview(payload, paths.image_jpg):
                    self.update_record(record_id, {"image": {"static_url": paths.image_static_url}})
                    return paths.image_static_url
        return None

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        paths = self._paths(record_id)
        if not paths.record_json.exists():
            return None
        try:
            return json.loads(paths.record_json.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_records(
        self,
        *,
        agent_id: Optional[str] = None,
        page_index: int = 0,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        page_index = max(0, int(page_index))
        page_size = max(1, min(200, int(page_size)))

        try:
            from app.services.review_record_db_service import get_review_record_db_service

            db_service = get_review_record_db_service()
            if db_service.enabled:
                db_result = db_service.list_records(
                    agent_id=agent_id,
                    page_index=page_index,
                    page_size=page_size,
                )
                if db_result is not None:
                    raw_records = db_result.get("records") or []
                    records = [rec if isinstance(rec, dict) else {} for rec in raw_records]
                    return {
                        "total": int(db_result.get("total") or 0),
                        "pageIndex": page_index,
                        "pageSize": page_size,
                        "data": [
                            self._summarize_record(
                                rec,
                                dir_name=str(rec.get("record_id") or ""),
                            )
                            for rec in records
                        ],
                    }
        except Exception as e:
            logger.warning("⚠️ ReviewRecord MySQL 分页查询失败（回退到本地文件）: %s", e)

        if not agent_id:
            try:
                now = time.time()
                base_mtime = 0.0
                try:
                    base_mtime = self.base_path.stat().st_mtime
                except Exception:
                    base_mtime = 0.0

                need_rebuild = (
                    not self._fast_index
                    or base_mtime != self._fast_index_base_mtime
                    or (now - self._fast_index_built_at) > 3.0
                )

                if need_rebuild:
                    entries_fast: List[Tuple[int, str, Path]] = []
                    for child in self.base_path.iterdir():
                        if not child.is_dir():
                            continue
                        record_json = child / "record.json"
                        if not record_json.exists():
                            continue
                        try:
                            created_ms = int(record_json.stat().st_mtime * 1000)
                        except Exception:
                            created_ms = 0
                        entries_fast.append((created_ms, child.name, record_json))

                    entries_fast.sort(key=lambda x: x[0], reverse=True)
                    self._fast_index = entries_fast
                    self._fast_index_built_at = now
                    self._fast_index_base_mtime = base_mtime

                total = len(self._fast_index)
                start = page_index * page_size
                end = start + page_size
                page = self._fast_index[start:end]

                data: List[Dict[str, Any]] = []
                for created_ms, dir_name, record_json in page:
                    try:
                        record = json.loads(record_json.read_text(encoding="utf-8"))
                    except Exception:
                        record = {"record_id": dir_name, "created_at_ms": created_ms}
                    data.append(self._summarize_record(record, dir_name=dir_name))

                return {
                    "total": total,
                    "pageIndex": page_index,
                    "pageSize": page_size,
                    "data": data,
                }
            except Exception:
                # Fallback to the full scan when fast path fails.
                pass

        entries: List[Tuple[int, str, Dict[str, Any]]] = []
        for child in self.base_path.iterdir():
            if not child.is_dir():
                continue
            record_json = child / "record.json"
            if not record_json.exists():
                continue
            try:
                record = json.loads(record_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            if agent_id and str(record.get("agent_id") or "") != agent_id:
                continue
            created_ms = _coerce_created_at_ms(record)
            entries.append((created_ms, child.name, record))

        entries.sort(key=lambda x: x[0], reverse=True)
        total = len(entries)
        start = page_index * page_size
        end = start + page_size
        page = entries[start:end]

        return {
            "total": total,
            "pageIndex": page_index,
            "pageSize": page_size,
            "data": [self._summarize_record(rec, dir_name=dir_name) for _, dir_name, rec in page],
        }

    async def record_video_for_record(
        self,
        *,
        record_id: str,
        source_url: str,
    ) -> None:
        """
        Record a 30s clip from a stream URL to mp4 using ffmpeg.

        This runs asynchronously and updates record.json with status and static_url.
        """
        paths = self._paths(record_id)
        if not paths.record_json.exists():
            return

        self.update_record(
            record_id,
            {
                "video": {
                    "source_url": source_url,
                    "clip_seconds": self.clip_seconds,
                    "status": "recording",
                    "static_url": None,
                    "file_path": None,
                    "error": None,
                    "recorded_at": None,
                }
            },
        )

        try:
            local_path = _resolve_local_path(source_url)
            if local_path:
                try:
                    target_path = paths.video_mp4
                    if local_path.resolve() != target_path.resolve():
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(local_path, target_path)
                    if target_path.exists() and target_path.stat().st_size > 0:
                        thumb_static_url = None
                        try:
                            ok, _ = await self._run_ffmpeg(
                                [
                                    self.ffmpeg_bin,
                                    "-y",
                                    "-hide_banner",
                                    "-loglevel",
                                    "error",
                                    "-nostdin",
                                    "-ss",
                                    "1",
                                    "-i",
                                    str(target_path),
                                    "-frames:v",
                                    "1",
                                    "-vf",
                                    "scale=320:-2",
                                    "-q:v",
                                    "4",
                                    str(paths.thumb_jpg),
                                ]
                            )
                            if ok and paths.thumb_jpg.exists() and paths.thumb_jpg.stat().st_size > 0:
                                thumb_static_url = paths.thumb_static_url
                        except Exception:
                            thumb_static_url = None

                        self.update_record(
                            record_id,
                            {
                                "video": {
                                    "source_url": source_url,
                                    "clip_seconds": self.clip_seconds,
                                    "status": "done",
                                    "static_url": paths.video_static_url,
                                    "thumb_static_url": thumb_static_url,
                                    "file_path": str(target_path),
                                    "error": None,
                                    "recorded_at": _utcnow_iso(),
                                }
                            },
                        )
                        return
                except Exception:
                    pass

            if not shutil.which(self.ffmpeg_bin):
                raise FileNotFoundError(f"ffmpeg not found: {self.ffmpeg_bin}")

            # RTSP 默认使用 TCP 传输，避免 UDP 被防火墙拦截导致拉流失败
            input_args: List[str] = []
            try:
                from urllib.parse import urlparse

                scheme = urlparse(source_url).scheme.lower()
            except Exception:
                scheme = ""
            if scheme in {"rtsp", "rtsps"}:
                input_args += ["-rtsp_transport", "tcp"]

            force_transcode = _env_flag("REVIEW_RECORDS_FORCE_TRANSCODE", "PATROL_CLIP_FORCE_TRANSCODE")
            if force_transcode:
                ok, err = await self._run_ffmpeg(
                    self._build_transcode_command(
                        source_url=source_url,
                        seconds=self.clip_seconds,
                        input_args=input_args,
                        output_path=paths.video_mp4,
                    )
                )
                if not ok:
                    ok, err = await self._run_ffmpeg(
                        [
                            self.ffmpeg_bin,
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            *input_args,
                            "-i",
                            source_url,
                            "-t",
                            str(self.clip_seconds),
                            "-c",
                            "copy",
                            "-movflags",
                            "+faststart",
                            str(paths.video_mp4),
                        ]
                    )
            else:
                # Attempt 1: remux/copy (fast)
                ok, err = await self._run_ffmpeg(
                    [
                        self.ffmpeg_bin,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        *input_args,
                        "-i",
                        source_url,
                        "-t",
                        str(self.clip_seconds),
                        "-c",
                        "copy",
                        "-movflags",
                        "+faststart",
                        str(paths.video_mp4),
                    ]
                )

                # Attempt 2: remux video-only (drop audio) to avoid unsupported audio codecs
                if not ok:
                    ok, err = await self._run_ffmpeg(
                        [
                            self.ffmpeg_bin,
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-nostdin",
                            *input_args,
                            "-i",
                            source_url,
                            "-t",
                            str(self.clip_seconds),
                            "-an",
                            "-c",
                            "copy",
                            "-movflags",
                            "+faststart",
                            str(paths.video_mp4),
                        ]
                    )

                # Attempt 3: transcode (more compatible)
                if not ok:
                    ok, err = await self._run_ffmpeg(
                        self._build_transcode_command(
                            source_url=source_url,
                            seconds=self.clip_seconds,
                            input_args=input_args,
                            output_path=paths.video_mp4,
                        )
                    )

            if not ok:
                raise RuntimeError(err or "ffmpeg failed")

            if not paths.video_mp4.exists() or paths.video_mp4.stat().st_size <= 0:
                raise RuntimeError("recorded video is empty")

            thumb_static_url = None
            try:
                ok, _ = await self._run_ffmpeg(
                    [
                        self.ffmpeg_bin,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-ss",
                        "1",
                        "-i",
                        str(paths.video_mp4),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=320:-2",
                        "-q:v",
                        "4",
                        str(paths.thumb_jpg),
                    ]
                )
                if ok and paths.thumb_jpg.exists() and paths.thumb_jpg.stat().st_size > 0:
                    thumb_static_url = paths.thumb_static_url
            except Exception:
                thumb_static_url = None

            self.update_record(
                record_id,
                {
                    "video": {
                        "source_url": source_url,
                        "clip_seconds": self.clip_seconds,
                        "status": "done",
                        "static_url": paths.video_static_url,
                        "thumb_static_url": thumb_static_url,
                        "file_path": str(paths.video_mp4),
                        "error": None,
                        "recorded_at": _utcnow_iso(),
                    }
                },
            )
        except Exception as e:
            self.update_record(
                record_id,
                {
                    "video": {
                        "source_url": source_url,
                        "clip_seconds": self.clip_seconds,
                        "status": "failed",
                        "static_url": None,
                        "file_path": None,
                        "error": str(e),
                        "recorded_at": _utcnow_iso(),
                    }
                },
            )


    async def record_clip(
        self,
        *,
        source_url: str,
        output_path: Path,
        clip_seconds: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        Record a short clip to mp4 using ffmpeg and return (ok, error).
        """
        try:
            seconds = int(clip_seconds or self.clip_seconds)
        except Exception:
            seconds = self.clip_seconds
        if seconds <= 0:
            seconds = self.clip_seconds

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if not shutil.which(self.ffmpeg_bin):
            return False, f"ffmpeg not found: {self.ffmpeg_bin}"

        # RTSP 默认使用 TCP 传输，避免 UDP 被防火墙拦截导致拉流失败
        input_args: List[str] = []
        try:
            from urllib.parse import urlparse

            scheme = urlparse(source_url).scheme.lower()
        except Exception:
            scheme = ""
        if scheme in {"rtsp", "rtsps"}:
            input_args += ["-rtsp_transport", "tcp"]

        force_transcode = _env_flag("REVIEW_RECORDS_FORCE_TRANSCODE", "PATROL_CLIP_FORCE_TRANSCODE")
        if force_transcode:
            ok, err = await self._run_ffmpeg(
                self._build_transcode_command(
                    source_url=source_url,
                    seconds=seconds,
                    input_args=input_args,
                    output_path=output_path,
                )
            )
            if not ok:
                ok, err = await self._run_ffmpeg(
                    [
                        self.ffmpeg_bin,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        *input_args,
                        "-i",
                        source_url,
                        "-t",
                        str(seconds),
                        "-c",
                        "copy",
                        "-movflags",
                        "+faststart",
                        str(output_path),
                    ]
                )
        else:
            # Attempt 1: remux/copy (fast)
            ok, err = await self._run_ffmpeg(
                [
                    self.ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    *input_args,
                    "-i",
                    source_url,
                    "-t",
                    str(seconds),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )

            # Attempt 2: remux video-only (drop audio) to avoid unsupported audio codecs
            if not ok:
                ok, err = await self._run_ffmpeg(
                    [
                        self.ffmpeg_bin,
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        *input_args,
                        "-i",
                        source_url,
                        "-t",
                        str(seconds),
                        "-an",
                        "-c",
                        "copy",
                        "-movflags",
                        "+faststart",
                        str(output_path),
                    ]
                )

            # Attempt 3: transcode (more compatible)
            if not ok:
                ok, err = await self._run_ffmpeg(
                    self._build_transcode_command(
                        source_url=source_url,
                        seconds=seconds,
                        input_args=input_args,
                        output_path=output_path,
                    )
                )

        if not ok:
            return False, err or "ffmpeg failed"

        if not output_path.exists() or output_path.stat().st_size <= 0:
            return False, "recorded clip is empty"

        return True, ""
    async def enforce_retention_async(self) -> None:
        await asyncio.to_thread(self.enforce_retention)

    def enforce_retention(self) -> None:
        """
        Keep only the latest N records globally (by created_at_ms).

        If max_records <= 0, retention is disabled (keep all records).
        """
        if self.max_records <= 0:
            return
        with self._fs_lock():
            entries: List[Tuple[int, Path]] = []
            for child in self.base_path.iterdir():
                if not child.is_dir():
                    continue
                record_json = child / "record.json"
                if not record_json.exists():
                    continue
                created_ms = 0
                try:
                    record = json.loads(record_json.read_text(encoding="utf-8"))
                    created_ms = int(record.get("created_at_ms") or 0)
                except Exception:
                    try:
                        created_ms = int(child.stat().st_mtime * 1000)
                    except Exception:
                        created_ms = 0
                entries.append((created_ms, child))

            entries.sort(key=lambda x: x[0], reverse=True)
            if len(entries) <= self.max_records:
                return

            removed_ids: List[str] = []
            for _, dir_path in entries[self.max_records :]:
                removed_ids.append(dir_path.name)
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                except Exception:
                    pass

            # Optional: keep MySQL mirror in sync.
            if removed_ids:
                try:
                    from app.services.review_record_db_service import get_review_record_db_service

                    get_review_record_db_service().delete_records(removed_ids)
                except Exception:
                    pass

    async def _run_ffmpeg(self, cmd: List[str]) -> Tuple[bool, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.ffmpeg_timeout_seconds)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return False, f"ffmpeg timeout after {self.ffmpeg_timeout_seconds}s"

            out = (stdout or b"") + (stderr or b"")
            text = out.decode("utf-8", errors="replace")
            return proc.returncode == 0, _safe_text(text, max_chars=4000)
        except Exception as e:
            return False, str(e)


_review_record_service: Optional[ReviewRecordService] = None


def get_review_record_service() -> ReviewRecordService:
    global _review_record_service
    if _review_record_service is None:
        _review_record_service = ReviewRecordService()
    return _review_record_service
