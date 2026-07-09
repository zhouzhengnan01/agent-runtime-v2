from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import require_runtime_token
from app.core.training_status import build_training_status, list_training_runs


router = APIRouter(prefix="/api/training", tags=["training"], dependencies=[Depends(require_runtime_token)])


@router.get("/status/{thread_id}/runs")
async def get_training_runs(thread_id: str) -> dict[str, object]:
    try:
        return list_training_runs(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/status/{thread_id}")
async def get_training_status(thread_id: str, run_id: str | None = Query(default=None)) -> dict[str, object]:
    try:
        return build_training_status(thread_id, run_id=run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
