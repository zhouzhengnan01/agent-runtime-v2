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
    user_id: str | None = None
    project_id: str | None = None
    workflow: str | None = None
    selected_skills: list[str] = Field(default_factory=list)
    selected_mcp_tools: list[str] = Field(default_factory=list)
    app_template_name: str | None = None
    model_type: str | None = None
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    request_timeout_seconds: float | None = None
    response_format: Literal["text", "json"] = "text"
    mode: Literal["plan", "edit", "autonomous", "safe", "yolo"] | None = None
    config_options: dict[str, Any] = Field(default_factory=dict)
    sandbox_profile: str | None = None


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


def artifact_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip()
    outputs_prefix = "/mnt/user-data/outputs/"
    if normalized.startswith(outputs_prefix):
        return f"outputs/{normalized[len(outputs_prefix):].lstrip('/')}"
    if normalized == "/mnt/user-data/outputs":
        return "outputs"
    if normalized.startswith("mnt/user-data/outputs/"):
        return f"outputs/{normalized[len('mnt/user-data/outputs/'):].lstrip('/')}"
    if normalized.startswith("outputs/") or normalized == "outputs":
        return normalized
    return normalized.lstrip("/")


def artifact_content_block(artifact: ArtifactRef) -> dict[str, Any]:
    relative_path = artifact_relative_path(artifact.path)
    return {
        "type": "resource_link",
        "uri": relative_path,
        "path": relative_path,
        "name": artifact.name,
        "mimeType": artifact.mime_type,
        "size": artifact.size,
        "title": artifact.name,
    }


def agent_result_content(reply: str, artifacts: list[ArtifactRef]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if reply.strip():
        content.append({"type": "text", "text": reply})
    content.extend(artifact_content_block(artifact) for artifact in artifacts)
    return content


class AgentRunResult(BaseModel):
    agent: str
    thread_id: str
    status: Literal["completed", "failed"] = "completed"
    reply: str
    content: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    verification: VerificationResult | None = None
    spec: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.content:
            self.content = agent_result_content(self.reply, self.artifacts)


class ChatEvent(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)
