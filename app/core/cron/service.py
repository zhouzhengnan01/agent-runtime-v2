from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from app.core.agent import AgentRuntime
from app.core.config import AgentConfigLoader
from app.core.cron.models import CronRunPayload, utc_now
from app.core.cron.store import CronJobStore
from app.schemas import ChatRequest, Message, RuntimeOptions


class CronService:
    """Execute cron jobs through the existing stateless AgentRuntime."""

    def __init__(
        self,
        store: CronJobStore | None = None,
        loader: AgentConfigLoader | None = None,
        runtime: AgentRuntime | None = None,
    ) -> None:
        self.store = store or CronJobStore()
        self.loader = loader or AgentConfigLoader()
        self.runtime = runtime or AgentRuntime()
        self._lock = asyncio.Lock()

    async def run_job(self, name: str, *, manual: bool = False) -> CronRunPayload:
        async with self._lock:
            job = self.store.get(name)
            thread_id = self._thread_id(job.name, manual=manual)
            self.store.mark_started(job.name, started_at=utc_now(), thread_id=thread_id)
        try:
            agent = self.loader.load(job.agent_name)
            runtime_options = self._runtime_options(job.runtime_options, thread_id)
            result = await self.runtime.run(
                agent,
                ChatRequest(
                    messages=[Message(role="user", content=job.prompt)],
                    runtime_options=runtime_options,
                ),
            )
        except Exception as exc:
            finished = self.store.mark_finished(
                job.name,
                finished_at=utc_now(),
                status="failed",
                error=str(exc),
                thread_id=thread_id,
            )
            return CronRunPayload(job=finished.to_payload(), result=None, error=str(exc))

        finished = self.store.mark_finished(
            job.name,
            finished_at=utc_now(),
            status=result.status,
            error="" if result.status == "completed" else result.reply,
            reply=result.reply,
            thread_id=result.thread_id,
        )
        return CronRunPayload(job=finished.to_payload(), result=result.model_dump(mode="json"))

    @staticmethod
    def _runtime_options(raw_options: dict[str, Any], thread_id: str) -> RuntimeOptions:
        options = dict(raw_options)
        options.setdefault("thread_id", thread_id)
        return RuntimeOptions.model_validate(options)

    @staticmethod
    def _thread_id(job_name: str, *, manual: bool) -> str:
        suffix = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        mode = "manual" if manual else "cron"
        safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in job_name)
        return f"{mode}-{safe_name}-{suffix}"
