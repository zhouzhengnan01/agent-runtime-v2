from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import cron as cron_api
from app.core.agent import AgentRuntime
from app.core.config import AgentConfig, AgentConfigLoader
from app.core.cron import CronExpression, CronJobStore, CronScheduler, CronService
from app.main import create_app
from app.schemas import AgentRunResult, ChatRequest


def test_cron_expression_calculates_next_run_with_timezone() -> None:
    expression = CronExpression.parse("*/15 9-10 * * mon-fri")
    after = datetime(2026, 5, 4, 1, 1, tzinfo=UTC)  # Monday 09:01 Asia/Shanghai.

    next_run = expression.next_after(after, "Asia/Shanghai")

    assert next_run == datetime(2026, 5, 4, 1, 15, tzinfo=UTC)


def test_cron_job_store_saves_lists_and_deletes_jobs(tmp_path: Path) -> None:
    store = CronJobStore(root_dir=tmp_path)

    saved = store.save(
        "daily-summary",
        {
            "title": "Daily Summary",
            "description": "Summarize system state.",
            "enabled": True,
            "schedule": "0 9 * * *",
            "timezone": "Asia/Shanghai",
            "agent_name": "default",
            "prompt": "生成每日摘要",
            "runtime_options": {"workflow": "agent_loop"},
        },
    )

    assert saved.name == "daily-summary"
    assert saved.next_run_at is not None
    assert store.get("daily-summary").prompt == "生成每日摘要"
    assert [job.name for job in store.list()] == ["daily-summary"]

    store.delete("daily-summary")
    assert store.list() == []


def test_cron_job_store_rejects_invalid_schedule(tmp_path: Path) -> None:
    store = CronJobStore(root_dir=tmp_path)

    with pytest.raises(ValueError):
        store.save("bad", {"schedule": "not cron", "agent_name": "default", "prompt": "hello"})


class FakeRuntime(AgentRuntime):
    async def run(self, agent_config: AgentConfig, request: ChatRequest) -> AgentRunResult:
        return AgentRunResult(
            agent=agent_config.name,
            thread_id=request.runtime_options.thread_id or "missing-thread",
            reply=f"ran: {request.messages[-1].content}",
            metadata={"workflow": request.runtime_options.workflow or "agent_loop"},
        )


def test_cron_service_runs_job_and_updates_status(tmp_path: Path) -> None:
    store = CronJobStore(root_dir=tmp_path)
    store.save(
        "manual-job",
        {
            "schedule": "* * * * *",
            "agent_name": "default",
            "prompt": "执行巡检",
            "runtime_options": {"workflow": "agent_loop"},
        },
    )
    service = CronService(store=store, loader=AgentConfigLoader(), runtime=FakeRuntime())

    payload = asyncio.run(service.run_job("manual-job", manual=True))

    assert payload.result is not None
    assert payload.job["last_status"] == "completed"
    assert payload.job["run_count"] == 1
    assert payload.job["last_reply"] == "ran: 执行巡检"


def test_cron_api_crud_and_manual_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = CronJobStore(root_dir=tmp_path)
    service = CronService(store=store, loader=AgentConfigLoader(), runtime=FakeRuntime())
    monkeypatch.setattr(cron_api, "store", store)
    monkeypatch.setattr(cron_api, "service", service)
    monkeypatch.setattr(cron_api, "scheduler", CronScheduler(store=store, service=service))
    monkeypatch.setenv("RUNTIME_API_TOKEN", "admin-secret")
    client = TestClient(create_app())

    denied = client.put(
        "/api/cron/jobs/nightly",
        json={"schedule": "0 1 * * *", "agent_name": "default", "prompt": "夜间巡检"},
    )
    saved = client.put(
        "/api/cron/jobs/nightly?token=admin-secret",
        json={"schedule": "0 1 * * *", "agent_name": "default", "prompt": "夜间巡检"},
    )
    listed = client.get("/api/cron/jobs")
    run = client.post("/api/cron/jobs/nightly/run?token=admin-secret")
    deleted = client.delete("/api/cron/jobs/nightly?token=admin-secret")

    assert denied.status_code == 401
    assert saved.status_code == 200
    assert saved.json()["name"] == "nightly"
    assert listed.json()["jobs"][0]["name"] == "nightly"
    assert run.status_code == 200
    assert run.json()["job"]["last_status"] == "completed"
    assert deleted.status_code == 200


def test_cron_scheduler_tick_runs_due_jobs(tmp_path: Path) -> None:
    store = CronJobStore(root_dir=tmp_path)
    job = store.save("due-job", {"schedule": "* * * * *", "agent_name": "default", "prompt": "到点执行"})
    store._replace(job.model_copy(update={"next_run_at": datetime(2020, 1, 1, tzinfo=UTC)}))
    service = CronService(store=store, loader=AgentConfigLoader(), runtime=FakeRuntime())
    scheduler = CronScheduler(store=store, service=service)

    asyncio.run(scheduler.tick())

    assert store.get("due-job").run_count == 1
