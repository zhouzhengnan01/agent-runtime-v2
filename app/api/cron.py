from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import require_admin_token
from app.core.cron import CronExpression, CronJobStore, CronRunPayload, CronScheduler, CronService
from app.core.cron.models import utc_now
from app.core.runtime import default_container


router = APIRouter(prefix="/api/cron", tags=["cron"])
store = CronJobStore()
service = CronService(store=store, loader=default_container.loader, runtime=default_container.runtime)
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


@router.get("/jobs/{job_name}/runs")
async def list_cron_job_runs(job_name: str, limit: int = 50) -> dict[str, list[dict[str, Any]]]:
    try:
        safe_limit = max(1, min(limit, 200))
        return {"runs": [record.to_payload() for record in store.list_runs(job_name, limit=safe_limit)]}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/preview")
async def preview_cron_schedule(payload: dict[str, Any]) -> dict[str, list[str]]:
    schedule = str(payload.get("schedule") or "").strip()
    timezone = str(payload.get("timezone") or "Asia/Shanghai").strip()
    try:
        count = int(payload.get("count") or 5)
        count = max(1, min(count, 20))
        expression = CronExpression.parse(" ".join(schedule.split()))
        current = utc_now()
        runs: list[str] = []
        for _ in range(count):
            current = expression.next_after(current, timezone)
            runs.append(current.isoformat())
        return {"runs": runs}
    except (TypeError, ValueError) as exc:
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
