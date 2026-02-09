"""
Deliverables export API.

Generate:
- 命令调用.txt (plain text)
- 步骤说明.docx (Word)
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.schemas.pagination import StandardResponse
from app.services.deliverables_export_service import export_deliverables

router = APIRouter(prefix="/deliverables", tags=["Deliverables"])


class DeliverablesExportRequest(BaseModel):
    commands_text: str = Field(..., description="命令调用.txt 内容（纯文本）")
    steps_markdown: str = Field(..., description="步骤说明内容（支持简单 Markdown）")

    output_dir: Optional[str] = Field(
        default=None,
        description="输出目录（可选）。不传则写入 storage/uploads/documents/deliverables/<job_id>/",
    )
    commands_filename: str = Field(default="命令调用.txt", description="命令文件名（配合 output_dir）")
    steps_filename: str = Field(default="步骤说明.docx", description="步骤文件名（配合 output_dir）")

    commands_out_path: Optional[str] = Field(
        default=None,
        description="命令文件输出路径（可选，优先级高于 output_dir/commands_filename）",
    )
    steps_out_path: Optional[str] = Field(
        default=None,
        description="DOCX 输出路径（可选，优先级高于 output_dir/steps_filename）",
    )


@router.post("/export", response_model=StandardResponse)
async def export_deliverables_api(request: DeliverablesExportRequest) -> StandardResponse:
    try:
        result = export_deliverables(
            commands_text=request.commands_text,
            steps_markdown=request.steps_markdown,
            output_dir=request.output_dir,
            commands_filename=request.commands_filename,
            steps_filename=request.steps_filename,
            commands_out_path=request.commands_out_path,
            steps_out_path=request.steps_out_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StandardResponse(
        message="success",
        result=result.to_dict(),
        status=200,
        timestamp=int(time.time() * 1000),
    )

