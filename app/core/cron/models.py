from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.core.cron.schedule import CronExpression


CronRunStatus = Literal["idle", "running", "completed", "failed"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class CronJobInput(BaseModel):
    title: str = ""
    description: str = ""
    enabled: bool = True
    schedule: str = "0 * * * *"
    timezone: str = "Asia/Shanghai"
    agent_name: str = "default"
    prompt: str
    runtime_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schedule")
    @classmethod
    def validate_schedule(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        CronExpression.parse(normalized)
        return normalized

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Cron job prompt must not be empty.")
        return value

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Cron job agent_name must not be empty.")
        return value.strip()


class CronJob(CronJobInput):
    name: str
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_finished_at: datetime | None = None
    last_status: CronRunStatus = "idle"
    last_error: str = ""
    last_thread_id: str = ""
    last_reply: str = ""
    run_count: int = 0

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Cron job name must not be empty.")
        if any(char in normalized for char in "/\\"):
            raise ValueError("Cron job name must not contain path separators.")
        return normalized

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CronRunPayload(BaseModel):
    job: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str = ""


class CronSchedulerStatus(BaseModel):
    enabled: bool
    running: bool
    interval_seconds: float
