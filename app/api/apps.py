from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.apps import AppTemplateRegistry


router = APIRouter(prefix="/api/apps", tags=["apps"])
registry = AppTemplateRegistry()

__all__ = ["registry", "router"]


@router.get("/templates")
async def list_app_templates() -> dict[str, list[dict[str, Any]]]:
    return {"templates": [template.to_payload() for template in registry.list()]}


@router.get("/templates/{template_name}")
async def get_app_template(template_name: str) -> dict[str, Any]:
    try:
        return registry.get(template_name).to_payload()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
