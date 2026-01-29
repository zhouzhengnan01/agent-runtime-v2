"""
Daily review report persistence and generation (filesystem-based).
"""

from __future__ import annotations

import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.services.review_record_service import get_review_record_service


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _now_ms() -> int:
    return int(datetime.utcnow().timestamp() * 1000)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False)


def _safe_text(value: Any, *, max_chars: int) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...(truncated, original_length={len(text)})"


def _pad2(n: int) -> str:
    return str(n).zfill(2)


def _format_local_datetime(ms: int) -> str:
    d = datetime.fromtimestamp(ms / 1000)
    return f"{d.year}-{_pad2(d.month)}-{_pad2(d.day)} {_pad2(d.hour)}:{_pad2(d.minute)}"


def _format_local_time(ms: int) -> str:
    d = datetime.fromtimestamp(ms / 1000)
    return f"{_pad2(d.hour)}:{_pad2(d.minute)}"


def _parse_time_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "").replace(" ", "T"))
        return int(dt.timestamp() * 1000)
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _get_record_time_ms(record: Dict[str, Any]) -> int:
    parsed = _parse_time_ms(record.get("record_time"))
    if parsed is not None:
        return parsed
    created = record.get("created_at_ms") or record.get("at") or 0
    try:
        created_ms = int(created)
    except Exception:
        created_ms = 0
    return created_ms if created_ms > 0 else int(datetime.now().timestamp() * 1000)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(max(0, min(len(sorted_vals) - 1, (len(sorted_vals) - 1) * p)))
    return float(sorted_vals[idx])


def _normalize_task_name(value: Any) -> str:
    s = str(value or "").strip()
    return s or "(unnamed)"


def _derive_task_category(task_name: str) -> str:
    s = str(task_name or "").strip()
    if not s:
        return "(unnamed)"
    if "://" in s:
        return s
    m = re.match(r"^(.+?)\s*[-|/>:]\s*(.+)$", s)
    if m and m.group(1) and m.group(2):
        left = str(m.group(1)).strip()
        return left or s
    return s


def _compute_review_status(record: Dict[str, Any]) -> str:
    status = str(record.get("review_status") or "").strip()
    if status:
        return status
    video = record.get("video") or {}
    video_status = str(video.get("status") or "").strip()
    if video_status in {"queued", "recording"}:
        return "pending"
    ok = record.get("success")
    if ok is False:
        return "error"
    return "done"


def _ensure_report_title(markdown: str, report_date: str) -> str:
    title = f"# 复判日报（{report_date}）"
    text = str(markdown or "").strip()
    if not text:
        return title
    lines = text.splitlines()
    first_idx = next((idx for idx, line in enumerate(lines) if line.strip()), None)
    if first_idx is None:
        return title
    first_line = lines[first_idx].strip()
    if first_line.startswith("#"):
        if "复判" not in first_line or report_date not in first_line:
            lines[first_idx] = title
        return "\n".join(lines).strip()
    return f"{title}\n\n{text}"


@dataclass(frozen=True)
class ReviewReportPaths:
    report_id: str
    report_dir: Path
    report_json: Path


class ReviewReportService:
    """Daily report store and generator."""

    def __init__(self) -> None:
        self.base_path = Path(getattr(settings, "REVIEW_REPORTS_STORAGE_PATH", "storage/review_reports"))
        self.retention_days = int(getattr(settings, "REVIEW_REPORTS_RETENTION_DAYS", 180))
        self.max_records = int(getattr(settings, "REVIEW_REPORTS_MAX_RECORDS", 5000))
        self.sample_size_default = int(getattr(settings, "REVIEW_REPORTS_SAMPLE_SIZE", 6))
        self.max_payload_chars = int(getattr(settings, "REVIEW_REPORTS_MAX_PAYLOAD_CHARS", 8500))
        self.default_agent_id = str(getattr(settings, "REVIEW_REPORTS_AGENT_ID", "review_cloud"))
        self.generate_empty = bool(getattr(settings, "REVIEW_REPORTS_GENERATE_EMPTY", True))
        self._lock_file = self.base_path / ".lock"
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _paths(self, report_id: str) -> ReviewReportPaths:
        report_dir = self.base_path / report_id
        return ReviewReportPaths(
            report_id=report_id,
            report_dir=report_dir,
            report_json=report_dir / "report.json",
        )

    def _normalize_date(self, value: Optional[Any]) -> str:
        if value is None or value == "":
            return datetime.now().date().isoformat()
        if isinstance(value, date_cls):
            return value.isoformat()
        if isinstance(value, datetime):
            return value.date().isoformat()
        s = str(value).strip()
        if not s:
            return datetime.now().date().isoformat()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                continue
        try:
            dt = datetime.fromisoformat(s.replace("Z", "").replace(" ", "T"))
            return dt.date().isoformat()
        except Exception:
            return datetime.now().date().isoformat()

    def _get_day_range(self, date_str: str) -> Tuple[int, int]:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime(dt.year, dt.month, dt.day)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        return int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    @contextmanager
    def _fs_lock(self):
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

    def _load_report(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def get_report(self, report_date: str) -> Optional[Dict[str, Any]]:
        report_id = self._normalize_date(report_date)
        paths = self._paths(report_id)
        return self._load_report(paths.report_json)

    def list_reports(self, *, page_index: int = 0, page_size: int = 50) -> Dict[str, Any]:
        page_index = max(0, int(page_index))
        page_size = max(1, min(200, int(page_size)))

        entries: List[Tuple[str, Dict[str, Any]]] = []
        for child in self.base_path.iterdir():
            if not child.is_dir():
                continue
            report_json = child / "report.json"
            if not report_json.exists():
                continue
            report = self._load_report(report_json)
            if not report:
                continue
            report_date = str(report.get("date") or child.name)
            entries.append((report_date, report))

        entries.sort(key=lambda x: x[0], reverse=True)
        total = len(entries)
        start = page_index * page_size
        end = start + page_size
        page = entries[start:end]

        def _summarize(report: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "report_id": report.get("report_id"),
                "date": report.get("date"),
                "agent_id": report.get("agent_id"),
                "base_url": report.get("base_url"),
                "created_at_ms": report.get("created_at_ms"),
                "stats": report.get("stats"),
                "meta": report.get("meta"),
            }

        return {
            "total": total,
            "pageIndex": page_index,
            "pageSize": page_size,
            "data": [_summarize(report) for _, report in page],
        }

    def _map_review_record_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for item in items:
            created_at_ms = item.get("created_at_ms") or item.get("createdAtMs") or item.get("created_atMs") or 0
            try:
                created_at_ms_int = int(created_at_ms)
            except Exception:
                created_at_ms_int = 0
            mapped.append(
                {
                    "id": item.get("record_id"),
                    "created_at_ms": created_at_ms_int,
                    "at": created_at_ms_int or int(datetime.now().timestamp() * 1000),
                    "agent_id": item.get("agent_id"),
                    "task": item.get("task"),
                    "source": item.get("source"),
                    "record_time": item.get("record_time"),
                    "review_status": item.get("review_status"),
                    "illegal": item.get("illegal"),
                    "http_status": item.get("http_status"),
                    "cost_ms": item.get("cost_ms"),
                    "success": item.get("success"),
                    "summary": item.get("summary"),
                    "video": item.get("video"),
                }
            )
        return mapped

    def _fetch_day_records(self, *, start_ms: int, end_ms: int, max_records: int) -> Tuple[List[Dict[str, Any]], int]:
        review_store = get_review_record_service()
        page_index = 0
        total = 0
        all_records: List[Dict[str, Any]] = []
        page_size = 200

        while len(all_records) < max_records:
            result = review_store.list_records(page_index=page_index, page_size=page_size)
            items = result.get("data") or []
            if not items:
                break
            total = int(result.get("total") or total)
            mapped = self._map_review_record_items(items)
            if not mapped:
                break
            all_records.extend(mapped)

            oldest = mapped[-1]
            oldest_ms = _get_record_time_ms(oldest)
            if oldest_ms < start_ms:
                break
            if total and len(all_records) >= total:
                break
            page_index += 1

        day_records = [r for r in all_records if start_ms <= _get_record_time_ms(r) <= end_ms]
        day_records.sort(key=lambda r: _get_record_time_ms(r), reverse=True)
        return day_records, total

    def _build_hourly_buckets(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets = [{"hour": hour, "total": 0, "alarm": 0} for hour in range(24)]
        for record in records:
            ts = _get_record_time_ms(record)
            hour = datetime.fromtimestamp(ts / 1000).hour
            idx = max(0, min(23, hour))
            buckets[idx]["total"] += 1
            if record.get("illegal") is True:
                buckets[idx]["alarm"] += 1
        return buckets

    def _build_counts(self, records: List[Dict[str, Any]], *, alarm_only: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        task_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        for record in records:
            source = str(record.get("source") or "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            if alarm_only and record.get("illegal") is not True:
                continue
            task = _normalize_task_name(record.get("task"))
            category = _derive_task_category(task)
            task_counts[category] = task_counts.get(category, 0) + 1

        def _to_list(counts: Dict[str, int], limit: int = 6) -> List[Dict[str, Any]]:
            return [
                {"name": name, "count": count}
                for name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:limit]
            ]

        return {
            "tasks": _to_list(task_counts),
            "sources": _to_list(source_counts),
        }

    def _build_samples(
        self,
        records: List[Dict[str, Any]],
        filter_fn,
        limit: int,
    ) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for record in records:
            if not filter_fn(record):
                continue
            samples.append(
                {
                    "time": _format_local_datetime(_get_record_time_ms(record)),
                    "task": _normalize_task_name(record.get("task")),
                    "source": str(record.get("source") or "-"),
                    "summary": _safe_text(str(record.get("summary") or ""), max_chars=80),
                    "status": _compute_review_status(record),
                    "illegal": record.get("illegal"),
                    "http_status": record.get("http_status"),
                }
            )
            if len(samples) >= limit:
                break
        return samples

    def _build_report_payload(
        self,
        records: List[Dict[str, Any]],
        *,
        start_ms: int,
        end_ms: int,
        sample_size: int,
    ) -> Dict[str, Any]:
        total = len(records)
        alarm_count = len([r for r in records if r.get("illegal") is True])
        pending_count = len([r for r in records if _compute_review_status(r) == "pending"])
        done_count = len([r for r in records if _compute_review_status(r) == "done"])
        error_count = len([r for r in records if _compute_review_status(r) == "error"])
        alarm_rate = alarm_count / total if total else 0.0
        success_rate = len([r for r in records if r.get("success") is not False]) / total if total else 0.0

        costs = [float(r.get("cost_ms") or 0) for r in records if float(r.get("cost_ms") or 0) > 0]
        avg_cost = sum(costs) / len(costs) if costs else 0.0
        p90_cost = _percentile(costs, 0.9) if costs else 0.0

        buckets = self._build_hourly_buckets(records)
        peak = max(buckets, key=lambda b: b.get("total", 0), default={"total": 0, "hour": 0})
        peak_label = (
            f"{_pad2(int(peak.get('hour', 0)))}:00 - {_pad2((int(peak.get('hour', 0)) + 1) % 24)}:00"
            f" · {peak.get('total', 0)} records"
            if peak.get("total")
            else "No data"
        )

        top_alarm = self._build_counts(records, alarm_only=True)
        top_all = self._build_counts(records, alarm_only=False)

        payload = {
            "date": datetime.fromtimestamp(start_ms / 1000).date().isoformat(),
            "range": {
                "start": _format_local_datetime(start_ms),
                "end": _format_local_datetime(end_ms),
            },
            "summary": {
                "total": total,
                "alarm_count": alarm_count,
                "pending_count": pending_count,
                "done_count": done_count,
                "error_count": error_count,
                "alarm_rate": round(alarm_rate, 4),
                "success_rate": round(success_rate, 4),
                "avg_cost_ms": round(avg_cost, 2),
                "p90_cost_ms": round(p90_cost, 2),
            },
            "peak_hour": peak_label,
            "top_alarm_tasks": top_alarm["tasks"],
            "top_sources": top_all["sources"],
            "hourly": [
                {"hour": f"{_pad2(int(b['hour']))}:00", "total": b["total"], "alarm": b["alarm"]}
                for b in buckets
            ],
            "samples": {
                "alarms": self._build_samples(records, lambda r: r.get("illegal") is True, sample_size),
                "pending": self._build_samples(records, lambda r: _compute_review_status(r) == "pending", sample_size),
            },
        }
        return payload

    def _shrink_payload_for_llm(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        clone = json.loads(json.dumps(payload))

        def _size(value: Any) -> int:
            return len(json.dumps(value, ensure_ascii=False))

        if _size(clone) <= self.max_payload_chars:
            return clone

        samples = clone.get("samples") or {}
        if isinstance(samples, dict):
            if isinstance(samples.get("alarms"), list):
                samples["alarms"] = samples["alarms"][:3]
            if isinstance(samples.get("pending"), list):
                samples["pending"] = samples["pending"][:3]
        if _size(clone) <= self.max_payload_chars:
            return clone

        if isinstance(clone.get("top_alarm_tasks"), list):
            clone["top_alarm_tasks"] = clone["top_alarm_tasks"][:3]
        if isinstance(clone.get("top_sources"), list):
            clone["top_sources"] = clone["top_sources"][:3]
        if _size(clone) <= self.max_payload_chars:
            return clone

        clone.pop("samples", None)
        if _size(clone) <= self.max_payload_chars:
            return clone

        hourly = clone.get("hourly")
        if isinstance(hourly, list):
            clone["hourly"] = [item for idx, item in enumerate(hourly) if idx % 4 == 0]
        return clone

    def _build_report_message(self, payload: Dict[str, Any]) -> str:
        instruction = "\n".join(
            [
                "你是政务安全运营分析师，请根据以下复判数据生成中文 Markdown 日报。",
                "行文要求（政府报告风格）：",
                "1) 标题需包含“复判日报/通报”与日期；措辞正式、客观、简洁。",
                "2) 分章节：总体情况、运行态势、情况分析、风险隐患、处置进展、处理建议、典型证据。",
                "3) 只基于提供的数据，不得编造事实或夸大结论。",
                "4) 关键数据用条目列出，可适度编号。",
                "5) 只输出 Markdown 正文，不要额外解释。",
                "",
                "数据(JSON)：",
            ]
        )
        payload_slim = self._shrink_payload_for_llm(payload)
        payload_str = json.dumps(payload_slim, ensure_ascii=False)
        return f"{instruction}\n{payload_str}"

    async def generate_report(
        self,
        *,
        date_value: Optional[Any] = None,
        base_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        sample_size: Optional[int] = None,
        force: bool = False,
        generated_by: str = "manual",
    ) -> Optional[Dict[str, Any]]:
        report_date = self._normalize_date(date_value)
        if not force:
            existing = self.get_report(report_date)
            if existing:
                return existing

        start_ms, end_ms = self._get_day_range(report_date)
        max_records = int(getattr(settings, "REVIEW_REPORTS_MAX_RECORDS", 5000))
        records, total = self._fetch_day_records(start_ms=start_ms, end_ms=end_ms, max_records=max_records)
        if not records and not self.generate_empty:
            return None

        size = int(sample_size or self.sample_size_default)
        payload = self._build_report_payload(records, start_ms=start_ms, end_ms=end_ms, sample_size=size)

        schema = {
            "type": "object",
            "properties": {
                "report_markdown": {"type": "string", "description": "Markdown report"},
                "key_findings": {"type": "array", "items": {"type": "string"}},
                "risk_points": {"type": "array", "items": {"type": "string"}},
                "action_items": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["report_markdown"],
        }

        message = self._build_report_message(payload)
        agent = (agent_id or self.default_agent_id).strip() or self.default_agent_id

        raw_response = None
        parsed_data = None
        try:
            from app.core.agents.agent_json import process_json_message
            from app.api.v1.agents_json import _parse_json_with_fallback, _fill_missing_fields_by_schema, _repair_json_with_llm

            raw_response = await process_json_message(
                message=message,
                agent_id=agent,
                context={
                    "json_schema": schema,
                    "output_format": "json",
                },
                parameter={},
            )

            parsed_data = _parse_json_with_fallback(raw_response)
            if parsed_data is None:
                parsed_data = await _repair_json_with_llm(
                    raw_response=raw_response,
                    user_message=message,
                    json_schema=schema,
                )
            if isinstance(parsed_data, dict):
                parsed_data = _fill_missing_fields_by_schema(
                    parsed_data,
                    schema,
                    fallback_summary=raw_response,
                )
        except Exception:
            parsed_data = None

        markdown = ""
        if isinstance(parsed_data, dict):
            markdown = str(parsed_data.get("report_markdown") or "").strip()
        if not markdown:
            markdown = str(raw_response or "").strip()
        if not markdown:
            markdown = "(empty report)"
        markdown = _ensure_report_title(markdown, report_date)

        report_id = report_date
        report = {
            "report_id": report_id,
            "date": report_date,
            "agent_id": agent,
            "base_url": base_url,
            "report_markdown": markdown,
            "raw_data": parsed_data,
            "payload": payload,
            "stats": payload.get("summary"),
            "meta": {
                "generated_by": generated_by,
                "record_count": len(records),
                "total_records": total,
                "range": payload.get("range"),
                "peak_hour": payload.get("peak_hour"),
                "sample_size": size,
            },
            "created_at_ms": _now_ms(),
            "created_at": _utcnow_iso(),
        }

        paths = self._paths(report_id)
        paths.report_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(paths.report_json, report)

        try:
            from app.services.review_report_db_service import get_review_report_db_service

            get_review_report_db_service().upsert_report(report)
        except Exception:
            pass

        self.enforce_retention()
        return report

    def enforce_retention(self) -> None:
        removed_ids: List[str] = []
        with self._fs_lock():
            entries: List[Tuple[str, Path]] = []
            for child in self.base_path.iterdir():
                if not child.is_dir():
                    continue
                report_json = child / "report.json"
                if not report_json.exists():
                    continue
                report_date = child.name
                entries.append((report_date, child))

            entries.sort(key=lambda x: x[0], reverse=True)

            if self.retention_days > 0:
                cutoff = datetime.now().date() - timedelta(days=self.retention_days)
                for report_date, path in entries:
                    try:
                        d = datetime.strptime(report_date, "%Y-%m-%d").date()
                    except Exception:
                        continue
                    if d < cutoff:
                        removed_ids.append(report_date)
                        shutil.rmtree(path, ignore_errors=True)

            if self.max_records > 0:
                remaining = [(d, p) for d, p in entries if d not in removed_ids]
                if len(remaining) > self.max_records:
                    for report_date, path in remaining[self.max_records :]:
                        removed_ids.append(report_date)
                        shutil.rmtree(path, ignore_errors=True)

        if removed_ids:
            try:
                from app.services.review_report_db_service import get_review_report_db_service

                get_review_report_db_service().delete_reports(removed_ids)
            except Exception:
                pass


_review_report_service: Optional[ReviewReportService] = None


def get_review_report_service() -> ReviewReportService:
    global _review_report_service
    if _review_report_service is None:
        _review_report_service = ReviewReportService()
    return _review_report_service
