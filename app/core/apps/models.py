from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.model_tags import normalize_model_tags


class AppModelOption(BaseModel):
    """Model option declared by an app template.

    Extra fields are intentionally preserved because app configs may come from
    a platform-side model registry whose schema is wider than the local runtime.
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    model_type: str | None = None
    features: list[str] = Field(default_factory=list)
    priority_features: list[str] = Field(default_factory=list)
    priority: int = 0
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_enc: str | None = None
    tool_choice: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    request_timeout_seconds: float | None = None
    default_model: str | None = None

    @field_validator("features", "priority_features", mode="before")
    @classmethod
    def validate_features(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @property
    def effective_model_type(self) -> str:
        if self.model_type and self.model_type.strip():
            return self.model_type.strip()
        if self.priority_features:
            return self.priority_features[0]
        if self.features:
            return self.features[0]
        return "chat"


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
    models: list[AppModelOption] = Field(default_factory=list)
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
        payload = self.model_dump(mode="json")
        _strip_secret_fields(payload)
        return payload

    def select_model(self, model_type: str = "chat") -> AppModelOption | None:
        return select_app_model(self.models, model_type=model_type)


def select_app_model(models: list[AppModelOption], model_type: str = "chat") -> AppModelOption | None:
    requested = (model_type or "chat").strip() or "chat"
    if not models:
        return None

    def rank(item: tuple[int, AppModelOption]) -> tuple[int, int, int]:
        index, model = item
        if requested in model.priority_features:
            match_rank = 0
        elif model.effective_model_type == requested:
            match_rank = 1
        elif requested in model.features:
            match_rank = 2
        else:
            match_rank = 3
        return (match_rank, model.priority, index)

    selected = min(enumerate(models), key=rank)[1]
    return selected


def _strip_secret_fields(payload: dict[str, Any]) -> None:
    runtime_options = payload.get("runtime_options")
    if isinstance(runtime_options, dict):
        runtime_options.pop("api_key", None)
        runtime_options.pop("api_key_enc", None)
        if not runtime_options:
            payload.pop("runtime_options", None)

    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        return
    for model in raw_models:
        if not isinstance(model, dict):
            continue
        model.pop("api_key", None)
        model.pop("api_key_enc", None)
