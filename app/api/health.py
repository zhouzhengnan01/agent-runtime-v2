from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from app.core.runtime import default_container
from app.core.runtime.health_state import readiness_issues, ready_from_state, review_state, uptime_seconds

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "service": "jetlinks-agent-runtime-v2", "ts": time.time()}


@router.get("/live")
async def live() -> dict[str, object]:
    return {"status": "alive", "service": "jetlinks-agent-runtime-v2", "uptime_seconds": round(uptime_seconds(), 3)}


@router.get("/ready")
async def ready() -> dict[str, object]:
    state = await review_state()
    if not ready_from_state(state):
        raise HTTPException(status_code=503, detail={"status": "busy", "issues": readiness_issues(state), **state})
    return {"status": "ready", **state}


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok", "service": "jetlinks-agent-runtime-v2"}


@router.get("/health/ready")
async def health_ready() -> dict[str, object]:
    runtime = default_container.runtime
    queues = runtime.queue_state_snapshot()
    queue_payload = {
        name: {
            "pending": snapshot.pending,
            "workers": snapshot.workers,
            "max_backlog": snapshot.max_backlog,
            "overloaded": snapshot.overloaded,
            "rejected_total": snapshot.rejected_total,
            "last_overloaded_at": snapshot.last_overloaded_at,
            "last_rejected_at": snapshot.last_rejected_at,
            "last_reject_reason": snapshot.last_reject_reason,
        }
        for name, snapshot in queues.items()
    }
    return {
        "status": "ok",
        "service": "jetlinks-agent-runtime-v2",
        "queues": queue_payload,
        "overloaded_queues": [name for name, snapshot in queues.items() if snapshot.overloaded],
    }
