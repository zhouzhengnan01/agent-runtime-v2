from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

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
