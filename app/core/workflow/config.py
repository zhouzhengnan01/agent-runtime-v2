from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


WorkflowTrigger = Literal["explicit_runtime_options"]

_WORKFLOW_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class WorkflowConfig(BaseModel):
    name: str
    display_name: str
    description: str = ""
    enabled: bool = True
    handler: str
    trigger: WorkflowTrigger = "explicit_runtime_options"
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not _WORKFLOW_NAME_RE.fullmatch(normalized):
            raise ValueError(f"Invalid workflow name: {value}")
        return normalized
