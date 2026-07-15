from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.auth import require_admin_token
from app.core.edge_machine import build_edge_machine_info
from app.schemas import EdgeMachineInfoResponse

router = APIRouter(
    prefix="/api/edge",
    tags=["edge-machine"],
    dependencies=[Depends(require_admin_token)],
)


@router.get("/machine", response_model=EdgeMachineInfoResponse)
def get_edge_machine() -> EdgeMachineInfoResponse:
    return EdgeMachineInfoResponse(**build_edge_machine_info())
