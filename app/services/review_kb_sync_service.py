"""
Best-effort sync for review records into JetLinks Knowledge.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)

_COLLECTION_CHECK_INTERVAL_SEC = 300
_COLLECTION_RETRY_INTERVAL_SEC = 60
_collection_ready = False
_collection_checked_at = 0.0


def enqueue_review_kb_sync(record_id: str) -> None:
    if not record_id:
        return
    if not bool(getattr(settings, "REVIEW_KB_ENABLED", False)):
        return
    thread = threading.Thread(target=_sync_record, args=(record_id,), daemon=True)
    thread.start()


def _sync_record(record_id: str) -> None:
    try:
        base_url = _get_base_url()
        if not base_url:
            return
        record = _load_record(record_id)
        if not record:
            return
        if not _should_sync(record):
            return

        session = requests.Session()
        session.headers.update({"Accept": "application/json"})
        if not _ensure_collection(session, base_url):
            return

        markdown = _build_markdown(record)
        data_uri = _build_data_uri(markdown)
        payload = _build_resource_payload(record, data_uri)
        timeout = int(getattr(settings, "REVIEW_KB_TIMEOUT", 15))
        resp = session.post(
            f"{base_url}/api/v1/rag/resource-parse/batch-parse",
            json={"resources": [payload]},
            timeout=timeout,
        )
        if not resp.ok:
            logger.warning(
                "Review KB sync failed: status=%s body=%s",
                resp.status_code,
                _safe_text(resp.text, 800),
            )
            return
        logger.info("Review KB sync queued: %s", record_id)
    except Exception as exc:
        logger.warning("Review KB sync error: %s", exc)


def _get_base_url() -> str:
    base_url = str(getattr(settings, "REVIEW_KB_BASE_URL", "") or "").strip()
    if not base_url:
        base_url = str(getattr(settings, "KNOWLEDGE_SERVICE_URL", "") or "").strip()
    return base_url.rstrip("/")


def _load_record(record_id: str) -> Optional[Dict[str, Any]]:
    base_path = Path(getattr(settings, "REVIEW_RECORDS_STORAGE_PATH", "storage/review_records"))
    record_json = base_path / record_id / "record.json"
    if not record_json.exists():
        logger.warning("Review record not found: %s", record_json)
        return None
    try:
        return json.loads(record_json.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Review record load failed: %s", exc)
        return None


def _should_sync(record: Dict[str, Any]) -> bool:
    response = record.get("response") or {}
    success = bool(response.get("success"))
    if not success and not bool(getattr(settings, "REVIEW_KB_INCLUDE_FAILED", False)):
        return False
    min_hit = int(getattr(settings, "REVIEW_KB_MIN_HIT", 0))
    if min_hit > 0:
        hit_val = _coerce_hit(response.get("hit"))
        if hit_val is None or hit_val < min_hit:
            return False
    return True


def _coerce_hit(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "hit", "match", "matched"}:
            return 1
        if s in {"0", "false", "no", "n", "miss", "unmatch", "unmatched"}:
            return 0
        try:
            return int(float(s))
        except (TypeError, ValueError):
            return None
    return None


def _ensure_collection(session: requests.Session, base_url: str) -> bool:
    global _collection_ready, _collection_checked_at
    now = time.time()
    if _collection_ready and (now - _collection_checked_at) < _COLLECTION_CHECK_INTERVAL_SEC:
        return True
    if (now - _collection_checked_at) < _COLLECTION_RETRY_INTERVAL_SEC:
        return _collection_ready

    _collection_checked_at = now
    timeout = int(getattr(settings, "REVIEW_KB_TIMEOUT", 15))
    payload = {
        "collection_id": getattr(settings, "REVIEW_KB_COLLECTION_ID", "video_search"),
        "name": getattr(settings, "REVIEW_KB_COLLECTION_NAME", "Video Search"),
        "description": getattr(
            settings, "REVIEW_KB_COLLECTION_DESCRIPTION", "Auto-created collection for review results"
        ),
        "collection_type": getattr(settings, "REVIEW_KB_COLLECTION_TYPE", "multimodal"),
    }
    try:
        resp = session.post(
            f"{base_url}/api/v1/rag/collections/",
            json=payload,
            timeout=timeout,
        )
        if resp.ok:
            _collection_ready = True
            return True
        logger.warning(
            "Review KB collection ensure failed: status=%s body=%s",
            resp.status_code,
            _safe_text(resp.text, 800),
        )
    except Exception as exc:
        logger.warning("Review KB collection ensure error: %s", exc)
    _collection_ready = False
    return False


def _build_resource_payload(record: Dict[str, Any], data_uri: str) -> Dict[str, Any]:
    record_id = str(record.get("record_id") or "")
    name = f"review_{record_id}" if record_id else "review_record"
    description = _extract_summary(record)
    return {
        "id": record_id or f"review_{int(time.time())}",
        "classifyId": getattr(settings, "REVIEW_KB_COLLECTION_ID", "video_search"),
        "name": name,
        "description": description or "Review record",
        "type": "file",
        "content": {"url": data_uri},
        "tags": ["review"],
        "bindings": [],
    }


def _build_markdown(record: Dict[str, Any]) -> str:
    record_id = str(record.get("record_id") or "")
    created_at = str(record.get("created_at") or "")
    agent_id = str(record.get("agent_id") or "")
    response = record.get("response") or {}
    success = response.get("success")
    hit = response.get("hit")
    summary = _extract_summary(record)

    compact = _compact_record(record)
    payload = json.dumps(compact, ensure_ascii=False, indent=2)
    payload = _truncate(payload, int(getattr(settings, "REVIEW_KB_MAX_PAYLOAD_CHARS", 20000)))

    lines = [
        "# Review Record",
        "",
        f"- record_id: {record_id}",
        f"- created_at: {created_at}",
        f"- agent_id: {agent_id}",
        f"- success: {success}",
    ]
    if hit is not None:
        lines.append(f"- hit: {hit}")
    if summary:
        lines.append(f"- summary: {summary}")
    lines.extend(["", "## Payload", "```json", payload, "```"])
    return "\n".join(lines)


def _compact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    compact = dict(record)
    meta = compact.get("meta")
    if isinstance(meta, dict) and "llm_raw_response" in meta:
        raw = meta.get("llm_raw_response")
        meta = dict(meta)
        meta["llm_raw_response"] = _truncate(_safe_text(raw, 8000), 8000)
        compact["meta"] = meta
    return compact


def _extract_summary(record: Dict[str, Any]) -> str:
    response = record.get("response") or {}
    meta = record.get("meta") or {}
    summary = meta.get("summary") or response.get("summary_sentence") or ""
    return str(summary or "").strip()


def _build_data_uri(markdown: str) -> str:
    payload = base64.b64encode(markdown.encode("utf-8")).decode("ascii")
    return f"data:text/markdown;base64,{payload}"


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...truncated, original_length={len(text)}"


def _safe_text(text: Any, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n...truncated, original_length={len(text)}"
