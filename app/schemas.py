from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str


class Attachment(BaseModel):
    name: str
    path: str | None = None
    mime_type: str | None = None
    data_base64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeOptions(BaseModel):
    thread_id: str | None = None
    workflow: str | None = None
    selected_skills: list[str] = Field(default_factory=list)
    selected_mcp_tools: list[str] = Field(default_factory=list)
    model_name: str | None = None
    model_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    request_timeout_seconds: float | None = None
    response_format: Literal["text", "json"] = "text"


class ChatRequest(BaseModel):
    messages: list[Message]
    attachments: list[Attachment] = Field(default_factory=list)
    runtime_options: RuntimeOptions = Field(default_factory=RuntimeOptions)


class VerificationCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class VerificationResult(BaseModel):
    passed: bool
    retry_count: int = 0
    checks: list[VerificationCheck] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)


class ArtifactRef(BaseModel):
    thread_id: str
    path: str
    name: str
    mime_type: str
    kind: str
    size: int
    preview_url: str
    download_url: str


class AgentRunResult(BaseModel):
    agent: str
    thread_id: str
    status: Literal["completed", "failed"] = "completed"
    reply: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    verification: VerificationResult | None = None
    spec: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatEvent(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
