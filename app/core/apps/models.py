from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.model_tags import normalize_model_tags


class AppTemplate(BaseModel):
    """Workbench application template for one-click agent setup."""

    name: str
    title: str
    description: str = ""
    category: str = "general"
    icon: str = "spark"
    agent_name: str = "default"
    workflow: str | None = None
    selected_skills: list[str] = Field(default_factory=list)
    selected_mcp_tools: list[str] = Field(default_factory=list)
    prompt_examples: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    model_tags: list[str] = Field(default_factory=list)
    models: list[dict[str, Any]] = Field(default_factory=list)
    runtime_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "title", "agent_name")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("App template name, title and agent_name must not be empty.")
        return normalized

    @field_validator("model_tags", mode="before")
    @classmethod
    def validate_model_tags(cls, value: object) -> list[str]:
        return normalize_model_tags(value)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
