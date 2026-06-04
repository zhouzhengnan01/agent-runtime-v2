from __future__ import annotations

from fastapi import APIRouter

from app.core.sandbox import SandboxStatus, get_sandbox_status


router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


@router.get("/status")
async def sandbox_status() -> SandboxStatus:
    return await get_sandbox_status()
