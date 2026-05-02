from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_admin_token
from app.core.cron import CronJobStore, CronRunPayload, CronScheduler, CronService


router = APIRouter(prefix="/api/cron", tags=["cron"])
store = CronJobStore()
service = CronService(store=store)
scheduler = CronScheduler(store=store, service=service)

__all__ = ["router", "scheduler", "service", "store"]


@router.get("/jobs")
async def list_cron_jobs() -> dict[str, list[dict[str, Any]]]:
    return {"jobs": [job.to_payload() for job in store.list()]}


@router.get("/jobs/{job_name}")
async def get_cron_job(job_name: str) -> dict[str, Any]:
    try:
        return store.get(job_name).to_payload()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/jobs/{job_name}", dependencies=[Depends(require_admin_token)])
async def save_cron_job(job_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("job") if isinstance(payload.get("job"), dict) else payload
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=400, detail="Cron job manifest must be a JSON object.")
    try:
        return store.save(job_name, manifest).to_payload()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/jobs/{job_name}", dependencies=[Depends(require_admin_token)])
async def delete_cron_job(job_name: str) -> dict[str, str]:
    try:
        store.delete(job_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "deleted", "name": job_name}


@router.post("/jobs/{job_name}/run", dependencies=[Depends(require_admin_token)])
async def run_cron_job(job_name: str) -> CronRunPayload:
    try:
        return await service.run_job(job_name, manual=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/scheduler/status")
async def cron_scheduler_status() -> dict[str, Any]:
    return scheduler.status().model_dump()
