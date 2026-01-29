"""
Review records API (filesystem-based).

Used by review.html to list and inspect recorded calls and their captured media.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.review_record_service import get_review_record_service

router = APIRouter()


@router.get("/review-records")
async def list_review_records(
    agentId: Optional[str] = Query(default=None, description="可选：按agentId过滤"),
    pageIndex: int = Query(default=0, ge=0, description="页码，从0开始"),
    pageSize: int = Query(default=50, ge=1, le=200, description="每页数量"),
) -> Dict[str, Any]:
    store = get_review_record_service()
    result = store.list_records(agent_id=agentId, page_index=pageIndex, page_size=pageSize)
    return {
        "status": 200,
        "message": "success",
        "result": result,
        "timestamp": int(time.time() * 1000),
    }


@router.get("/review-records/{record_id}")
async def get_review_record(record_id: str) -> Dict[str, Any]:
    store = get_review_record_service()
    record = store.get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="record not found")
    return {
        "status": 200,
        "message": "success",
        "result": record,
        "timestamp": int(time.time() * 1000),
    }

