from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


@router.get("/health")
def monitor_health() -> dict[str, str]:
    return {"status": "ok"}
