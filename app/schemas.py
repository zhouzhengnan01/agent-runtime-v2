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
    training_model_id: str | None = None
    deimv2_model_variant: str | None = Field(default=None, alias="deimv2ModelVariant")
    max_synthetic_images: int | None = None
    reuse_previous_data_preparation: Literal["auto", "never", "required"] | bool = "auto"
    reuse_from_run_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    request_timeout_seconds: float | None = None
    response_format: Literal["text", "json"] = "text"
    mode: Literal["plan", "edit", "autonomous", "safe", "yolo"] | None = None
    skill_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)
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


class EdgeRuntimeInstallRequest(BaseModel):
    bundle_url: str | None = None
    bundle_path: str | None = None
    bundle_name: str | None = None
    bundle_sha256: str | None = None
    machine_type: str
    model_id: str
    runtime_name: str = "default"
    force: bool = False


class EdgeRuntimeWeightInstallRequest(BaseModel):
    machine_type: str | None = None
    model_id: str
    runtime_name: str = "default"
    version: str | None = None
    runtime_url: str | None = None
    runtime_path: str | None = None
    runtime_sha256: str | None = None
    runtime_bundle_url: str | None = None
    runtime_bundle_path: str | None = None
    runtime_bundle_sha256: str | None = None
    bundle_url: str | None = None
    bundle_path: str | None = None
    bundle_sha256: str | None = None
    weight_url: str | None = None
    weight_path: str | None = None
    weight_sha256: str | None = None
    target_subdir: str = "weights"
    port: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    install_weight: bool = True
    auto_start: bool = False
    force_runtime: bool = False


class EdgeRuntimeDeployRequest(BaseModel):
    model_id: str
    runtime_name: str = "default"
    machine_type: str | None = None
    version: str | None = None
    runtime_url: str | None = None
    runtime_path: str | None = None
    runtime_sha256: str | None = None
    runtime_bundle_url: str | None = None
    runtime_bundle_path: str | None = None
    runtime_bundle_sha256: str | None = None
    bundle_url: str | None = None
    bundle_path: str | None = None
    bundle_sha256: str | None = None
    weight_url: str | None = None
    weight_path: str | None = None
    weight_sha256: str | None = None
    target_subdir: str = "weights"
    port: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    install_weight: bool = True
    auto_start: bool = True
    force_runtime: bool = False


class EdgeRuntimeStartRequest(BaseModel):
    machine_type: str
    model_id: str
    runtime_name: str = "default"
    port: int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)


class EdgeRuntimeStopRequest(BaseModel):
    machine_type: str
    model_id: str
    runtime_name: str = "default"


class EdgeRuntimeStatusItem(BaseModel):
    machine_type: str
    model_id: str
    runtime_name: str
    bundle_dir: str
    manifest_path: str | None = None
    status: str = "unknown"
    pid: int | None = None
    port: int | None = None
    command: list[str] = Field(default_factory=list)
    log_path: str | None = None
    installed_at: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    health_status: str = "unknown"
    restart_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EdgeMachineRuntimeSummary(BaseModel):
    runtimeRoot: str
    installedRuntimes: int = 0
    runningRuntimes: int = 0
    runtimes: list[dict[str, Any]] = Field(default_factory=list)


class EdgeMachineInfoResponse(BaseModel):
    nodeId: str
    nodeName: str
    nodeRole: str = "edge"
    deviceTypes: list[str] = Field(default_factory=list)
    deviceModel: str | None = None
    machineTypeId: str | None = None
    machineTypeIds: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    managerVersion: str = "0.1.0"
    runtimeGitCommit: str | None = None
    serviceRoot: str
    system: dict[str, str] = Field(default_factory=dict)
    capabilities: dict[str, bool] = Field(default_factory=dict)
    agentRuntime: EdgeMachineRuntimeSummary
    cloudQuery: dict[str, str] = Field(default_factory=dict)
