from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas import ArtifactRef


@dataclass
class SkillRunResult:
    skill_name: str
    outputs: list[ArtifactRef] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

