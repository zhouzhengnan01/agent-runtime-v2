from __future__ import annotations

import heapq
import json
import math
import os
import statistics
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return default
    return max(1, value)


class TrainingQueueManager:
    """Single-process FIFO admission queue for complete training workflows."""

    def __init__(
        self,
        *,
        max_running: int | None = None,
        max_waiting: int | None = None,
        default_duration_seconds: int | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.max_running = max_running or _positive_int_env("JETLINKS_TRAINING_MAX_RUNNING", 10)
        self.max_waiting = max_waiting or _positive_int_env("JETLINKS_TRAINING_MAX_WAITING", 10)
        self.default_duration_seconds = default_duration_seconds or _positive_int_env(
            "JETLINKS_TRAINING_DEFAULT_DURATION_SECONDS",
            3600,
        )
        self.history_limit = 20
        self.minimum_history_samples = 3
        self.history_path = history_path
        self._lock = threading.RLock()
        self._running: dict[str, dict[str, Any]] = {}
        self._waiting: deque[dict[str, Any]] = deque()
        self._successful_durations: deque[float] = deque(maxlen=self.history_limit)
        self._updated_at = _utc_now()
        self._load_duration_history()

    def admit(self, record: dict[str, Any]) -> str:
        job_id = str(record["job_id"])
        with self._lock:
            if len(self._running) < self.max_running:
                self._running[job_id] = record
                record["queue_status"] = "running"
                record["queue_position"] = None
                record["queued_at"] = None
                self._touch()
                return "running"
            if len(self._waiting) < self.max_waiting:
                record["queue_status"] = "waiting"
                record["queued_at"] = _utc_now()
                self._waiting.append(record)
                self._refresh_positions()
                self._touch()
                return "waiting"
            record["queue_status"] = "queue_rejected"
            record["queue_position"] = None
            record["queued_at"] = None
            self._touch()
            return "queue_rejected"

    def finish(self, job_id: str, *, promote: bool = True) -> list[dict[str, Any]]:
        promoted: list[dict[str, Any]] = []
        with self._lock:
            finished_record = self._running.pop(job_id, None)
            if finished_record is not None and str(finished_record.get("status") or "") == "completed":
                duration = _record_duration_seconds(finished_record)
                if duration is not None:
                    self._record_successful_duration(duration)
            if promote:
                while self._waiting and len(self._running) < self.max_running:
                    record = self._waiting.popleft()
                    promoted_id = str(record["job_id"])
                    record["queue_status"] = "running"
                    record["queue_position"] = None
                    self._running[promoted_id] = record
                    promoted.append(record)
            self._refresh_positions()
            self._touch()
        return promoted

    def cancel_waiting(self, job_id: str) -> bool:
        with self._lock:
            original = len(self._waiting)
            self._waiting = deque(
                record for record in self._waiting if str(record.get("job_id") or "") != job_id
            )
            removed = len(self._waiting) != original
            if removed:
                self._refresh_positions()
                self._touch()
            return removed

    def clear(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self._lock:
            running = list(self._running.values())
            waiting = list(self._waiting)
            self._running.clear()
            self._waiting.clear()
            self._touch()
            return running, waiting

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            running = len(self._running)
            waiting = len(self._waiting)
            return {
                "schema": "jetlinks-training-queue.v1",
                "max_running": self.max_running,
                "max_waiting": self.max_waiting,
                "running": running,
                "waiting": waiting,
                "available_running_slots": max(0, self.max_running - running),
                "available_waiting_slots": max(0, self.max_waiting - waiting),
                "queue_full": running >= self.max_running and waiting >= self.max_waiting,
                "updated_at": self._updated_at,
            }

    def thread_state(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            for record in self._running.values():
                if str(record.get("thread_id") or "") == thread_id:
                    return _record_queue_state(record)
            for record in self._waiting:
                if str(record.get("thread_id") or "") == thread_id:
                    return _record_queue_state(record)
        return None

    def job_snapshot(self, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            queue_status = str(record.get("queue_status") or "") or None
            running = len(self._running)
            waiting = len(self._waiting)
            estimated_seconds, estimate_type = self._estimate_seconds(record, queue_status)
            return {
                "schema": "jetlinks-training-job-queue.v1",
                "status": queue_status,
                "accepted": queue_status not in {None, "queue_rejected"},
                "starts_immediately": queue_status == "running",
                "position": record.get("queue_position"),
                "queued_at": record.get("queued_at"),
                "max_running": self.max_running,
                "max_waiting": self.max_waiting,
                "running": running,
                "waiting": waiting,
                "available_running_slots": max(0, self.max_running - running),
                "available_waiting_slots": max(0, self.max_waiting - waiting),
                "queue_full": running >= self.max_running and waiting >= self.max_waiting,
                "reason": record.get("error") if queue_status == "queue_rejected" else None,
                "estimate_type": estimate_type,
                "estimated_seconds": estimated_seconds,
                "updated_at": self._updated_at,
            }

    def record_successful_duration(self, duration_seconds: float) -> None:
        with self._lock:
            self._record_successful_duration(duration_seconds)
            self._touch()

    def _record_successful_duration(self, duration_seconds: float) -> None:
        try:
            duration = float(duration_seconds)
        except (TypeError, ValueError):
            return
        if not math.isfinite(duration) or duration <= 0:
            return
        self._successful_durations.append(duration)
        self._write_duration_history()

    def _estimated_task_duration(self) -> tuple[float, str]:
        if len(self._successful_durations) >= self.minimum_history_samples:
            return float(statistics.median(self._successful_durations)), "historical_median"
        return float(self.default_duration_seconds), "default"

    def _estimate_seconds(self, record: dict[str, Any], queue_status: str | None) -> tuple[int, str]:
        duration, _ = self._estimated_task_duration()
        now = datetime.now(UTC)
        if queue_status == "running":
            return int(math.ceil(_remaining_duration(record, duration, now))), "completion"
        if queue_status == "waiting":
            return int(math.ceil(self._waiting_start_seconds(record, duration, now))), "start"
        if queue_status == "queue_rejected":
            remaining = [_remaining_duration(item, duration, now) for item in self._running.values()]
            return int(math.ceil(min(remaining) if remaining else 0.0)), "retry"
        completed_at = _parse_utc_datetime(record.get("completed_at"))
        if queue_status in {"finished", "cancelled"} or completed_at is not None:
            return 0, "completion"
        return int(math.ceil(duration)), "start"

    def _waiting_start_seconds(self, target: dict[str, Any], duration: float, now: datetime) -> float:
        slot_times = [
            _remaining_duration(record, duration, now)
            for record in self._running.values()
        ]
        slot_times.extend(0.0 for _ in range(max(0, self.max_running - len(slot_times))))
        if not slot_times:
            return 0.0
        heapq.heapify(slot_times)
        target_id = str(target.get("job_id") or "")
        for waiting_record in self._waiting:
            starts_in = heapq.heappop(slot_times)
            if str(waiting_record.get("job_id") or "") == target_id:
                return starts_in
            heapq.heappush(slot_times, starts_in + duration)
        return min(slot_times)

    def _load_duration_history(self) -> None:
        if self.history_path is None or not self.history_path.is_file():
            return
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        values = payload.get("durations_seconds") if isinstance(payload, dict) else None
        if not isinstance(values, list):
            return
        for value in values[-self.history_limit :]:
            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                self._successful_durations.append(duration)

    def _write_duration_history(self) -> None:
        if self.history_path is None:
            return
        payload = {
            "schema": "jetlinks-training-duration-history.v1",
            "durations_seconds": list(self._successful_durations),
            "updated_at": _utc_now(),
        }
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.history_path)
        except OSError:
            return

    def _refresh_positions(self) -> None:
        for position, record in enumerate(self._waiting, start=1):
            record["queue_position"] = position

    def _touch(self) -> None:
        self._updated_at = _utc_now()


def _record_queue_state(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": record.get("job_id"),
        "queue_status": record.get("queue_status"),
        "queue_position": record.get("queue_position"),
        "queued_at": record.get("queued_at"),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _record_duration_seconds(record: dict[str, Any]) -> float | None:
    started_at = _parse_utc_datetime(record.get("started_at"))
    completed_at = _parse_utc_datetime(record.get("completed_at"))
    if started_at is None or completed_at is None:
        return None
    duration = (completed_at - started_at).total_seconds()
    return duration if duration > 0 else None


def _remaining_duration(record: dict[str, Any], duration: float, now: datetime) -> float:
    started_at = _parse_utc_datetime(record.get("started_at"))
    if started_at is None:
        return duration
    elapsed = max(0.0, (now - started_at).total_seconds())
    return max(0.0, duration - elapsed)


training_queue = TrainingQueueManager(
    history_path=Path.cwd() / ".runtime" / "server" / "training_duration_history.json"
)
