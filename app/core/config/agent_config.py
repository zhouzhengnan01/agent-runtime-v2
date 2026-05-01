from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AgentBackendConfig(BaseModel):
    type: Literal["local", "acp_stdio"] = "local"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_enc: str | None = None
    model_env: str = "LLM_MODEL"
    default_model: str = "Qwen3.6-35B-A3B"
    base_url_env: str = "LLM_BASE_URL"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.3
    top_p: float | None = None
    max_tokens: int = 4096


class RuntimeConfig(BaseModel):
    stateless: bool = True
    max_tool_rounds: int = 8
    max_retries: int = 1
    workspace_mode: Literal["ephemeral_thread"] = "ephemeral_thread"
    require_verification: bool = True


class QualityConfig(BaseModel):
    auto_repair: bool = True
    verify_outputs: bool = True
    fail_on_missing_artifact: bool = True


class PromptConfig(BaseModel):
    system: str = ""
    response_language: str = "zh-CN"


class RoutingConfig(BaseModel):
    llm_workflow_router: bool = True
    llm_workflow_router_env: str = "LLM_WORKFLOW_ROUTER"


class MemoryConfig(BaseModel):
    enabled: bool = False
    scope: Literal["agent", "global"] = "agent"
    max_items: int = Field(default=20, ge=1, le=100)
    inject_context: bool = False


class AgentConfig(BaseModel):
    name: str
    display_name: str
    description: str = ""
    backend: AgentBackendConfig = Field(default_factory=AgentBackendConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    tools: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    workflows: dict[str, str] = Field(default_factory=dict)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
