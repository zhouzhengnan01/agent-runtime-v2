from __future__ import annotations

import builtins
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.cron.models import CronJob, CronJobInput, CronRunStatus, utc_now
from app.core.cron.schedule import CronExpression


class CronJobStore:
    """JSON-backed store for cron job configuration and last-run status."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.path = self.root_dir / "config" / "cron" / "jobs.json"

    def list(self) -> builtins.list[CronJob]:
        return sorted(self._read_jobs().values(), key=lambda job: job.name)

    def get(self, name: str) -> CronJob:
        jobs = self._read_jobs()
        safe_name = self._safe_name(name)
        if safe_name not in jobs:
            raise KeyError(f"Cron job not found: {safe_name}")
        return jobs[safe_name]

    def save(self, name: str, payload: dict[str, Any]) -> CronJob:
        safe_name = self._safe_name(name)
        jobs = self._read_jobs()
        existing = jobs.get(safe_name)
        try:
            data = CronJobInput.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        now = utc_now()
        job = CronJob(
            **data.model_dump(),
            name=safe_name,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            next_run_at=self._next_run_at(data.schedule, data.timezone, now) if data.enabled else None,
            last_run_at=existing.last_run_at if existing else None,
            last_finished_at=existing.last_finished_at if existing else None,
            last_status=existing.last_status if existing else "idle",
            last_error=existing.last_error if existing else "",
            last_thread_id=existing.last_thread_id if existing else "",
            last_reply=existing.last_reply if existing else "",
            run_count=existing.run_count if existing else 0,
        )
        jobs[safe_name] = job
        self._write_jobs(jobs)
        return job

    def delete(self, name: str) -> None:
        safe_name = self._safe_name(name)
        jobs = self._read_jobs()
        if safe_name not in jobs:
            raise KeyError(f"Cron job not found: {safe_name}")
        del jobs[safe_name]
        self._write_jobs(jobs)

    def due(self, now: datetime | None = None) -> builtins.list[CronJob]:
        now = now or utc_now()
        return [job for job in self.list() if job.enabled and job.next_run_at is not None and job.next_run_at <= now]

    def mark_started(self, name: str, *, started_at: datetime, thread_id: str) -> CronJob:
        job = self.get(name)
        updated = job.model_copy(
            update={
                "updated_at": started_at,
                "last_run_at": started_at,
                "last_status": "running",
                "last_error": "",
                "last_thread_id": thread_id,
            }
        )
        self._replace(updated)
        return updated

    def mark_finished(
        self,
        name: str,
        *,
        finished_at: datetime,
        status: CronRunStatus,
        error: str = "",
        reply: str = "",
        thread_id: str = "",
    ) -> CronJob:
        job = self.get(name)
        next_run_at = self._next_run_at(job.schedule, job.timezone, finished_at) if job.enabled else None
        updated = job.model_copy(
            update={
                "updated_at": finished_at,
                "next_run_at": next_run_at,
                "last_finished_at": finished_at,
                "last_status": status,
                "last_error": error[:2000],
                "last_reply": reply[:2000],
                "last_thread_id": thread_id or job.last_thread_id,
                "run_count": job.run_count + 1,
            }
        )
        self._replace(updated)
        return updated

    def _replace(self, job: CronJob) -> None:
        jobs = self._read_jobs()
        jobs[job.name] = job
        self._write_jobs(jobs)

    def _read_jobs(self) -> dict[str, CronJob]:
        if not self.path.is_file():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        items = raw.get("jobs") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return {}
        jobs: dict[str, CronJob] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            job = CronJob.model_validate(item)
            jobs[job.name] = job
        return jobs

    def _write_jobs(self, jobs: dict[str, CronJob]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": [job.to_payload() for job in sorted(jobs.values(), key=lambda item: item.name)]}
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    @staticmethod
    def _safe_name(name: str) -> str:
        safe_name = name.strip()
        if not safe_name:
            raise ValueError("Cron job name must not be empty.")
        if any(char in safe_name for char in "/\\"):
            raise ValueError("Cron job name must not contain path separators.")
        return safe_name

    @staticmethod
    def _next_run_at(schedule: str, timezone: str, after: datetime) -> datetime:
        after_utc = after if after.tzinfo is not None else after.replace(tzinfo=UTC)
        return CronExpression.parse(schedule).next_after(after_utc.astimezone(UTC), timezone)
