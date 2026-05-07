from __future__ import annotations

import asyncio
import os
from contextlib import suppress

from app.core.cron.models import CronSchedulerStatus, utc_now
from app.core.cron.service import CronService
from app.core.cron.store import CronJobStore


class CronScheduler:
    """Small in-process scheduler for enabled cron jobs."""

    def __init__(
        self,
        store: CronJobStore | None = None,
        service: CronService | None = None,
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        self.store = store or CronJobStore()
        self.service = service or CronService(store=self.store)
        self.interval_seconds = interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> CronSchedulerStatus:
        return CronSchedulerStatus(enabled=scheduler_enabled(), running=self.running, interval_seconds=self.interval_seconds)

    async def start(self) -> None:
        if not scheduler_enabled() or self.running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._stop_event = None

    async def tick(self) -> None:
        for job in self.store.due(utc_now()):
            await self.service.run_job(job.name)

    async def _loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue


def scheduler_enabled() -> bool:
    return os.getenv("CRON_SCHEDULER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
