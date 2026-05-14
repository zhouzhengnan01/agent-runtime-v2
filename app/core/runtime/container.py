from __future__ import annotations

import builtins
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from app.core.agent import AgentRuntime
from app.core.config import AgentConfig, AgentConfigLoader
from app.core.config.agent_config import ModelConfig
from app.core.model_tags import normalize_model_tags
from app.schemas import RuntimeOptions


class ManagedModel(BaseModel):
    id: str
    display_name: str = ""
    description: str = ""
    config: ModelConfig = Field(default_factory=ModelConfig)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def runtime_options(self) -> RuntimeOptions:
        return RuntimeOptions(
            model_name=self.config.model or self.config.default_model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            request_timeout_seconds=self.config.request_timeout_seconds,
        )

    def public_payload(self) -> dict[str, Any]:
        model_tags = normalize_model_tags(self.capabilities)
        return {
            "id": self.id,
            "name": self.display_name or self.id,
            "displayName": self.display_name or self.id,
            "description": self.description,
            "model": self.config.model or self.config.default_model,
            "baseUrlConfigured": bool(self.config.base_url),
            "apiKeyConfigured": bool(self.config.api_key),
            "capabilities": model_tags,
            "model_tags": model_tags,
            "metadata": dict(self.metadata),
        }


class RuntimeBootstrapConfig(BaseModel):
    model: ModelConfig | dict[str, Any] | None = None
    models: list[ManagedModel | dict[str, Any]] = Field(default_factory=list)
    default_model_id: str | None = None
    skills: list[str] | None = None


def _model_id(value: str) -> str:
    normalized = value.strip()
    return normalized or "default"


class ModelManager:
    def __init__(
        self,
        models: Iterable[ManagedModel | dict[str, Any]] | None = None,
        *,
        default_model_id: str | None = None,
    ) -> None:
        self._models: dict[str, ManagedModel] = {}
        self.default_model_id = default_model_id
        for model in models or []:
            self.register(model)

    def configure(
        self,
        models: Iterable[ManagedModel | dict[str, Any]] | None,
        *,
        default_model_id: str | None = None,
    ) -> None:
        self._models = {}
        self.default_model_id = default_model_id
        for model in models or []:
            self.register(model)

    def register(self, model: ManagedModel | dict[str, Any]) -> ManagedModel:
        normalized = self._normalize(model)
        self._models[normalized.id] = normalized
        if self.default_model_id is None:
            self.default_model_id = normalized.id
        return normalized

    def get(self, model_id: str) -> ManagedModel | None:
        return self._models.get(model_id)

    def list(self) -> list[ManagedModel]:
        return [self._models[key] for key in sorted(self._models)]

    def list_public_payloads(self) -> builtins.list[dict[str, Any]]:
        return [model.public_payload() for model in self.list()]

    def default(self) -> ManagedModel | None:
        if self.default_model_id is None:
            return None
        return self.get(self.default_model_id)

    def runtime_options_for(self, model_id: str) -> RuntimeOptions | None:
        model = self.get(model_id)
        if model is None:
            return None
        return model.runtime_options()

    def configure_from_agent_default(self, agent_config: AgentConfig, *, model_id: str | None = None) -> ManagedModel:
        config = agent_config.model
        candidate_id = model_id or config.model or config.default_model or agent_config.name
        normalized_id = _model_id(candidate_id)
        return self.register(
            ManagedModel(
                id=normalized_id,
                display_name=config.model or config.default_model or normalized_id,
                description=f"Default model for agent {agent_config.name}.",
                config=config,
                capabilities=["chat", "tool_call"] if config.tool_choice == "auto" else ["chat"],
                metadata={"source": "agent_default", "agent": agent_config.name},
            )
        )

    @classmethod
    def _normalize(cls, model: ManagedModel | dict[str, Any]) -> ManagedModel:
        if isinstance(model, ManagedModel):
            return model
        data = dict(model)
        raw_config = data.get("config")
        if isinstance(raw_config, ModelConfig):
            config = raw_config
        elif isinstance(raw_config, dict):
            config = ModelConfig.model_validate(raw_config)
        else:
            config_fields = set(ModelConfig.model_fields)
            config_data = {key: data.pop(key) for key in list(data) if key in config_fields}
            config = ModelConfig.model_validate(config_data)
        data["config"] = config
        data["capabilities"] = normalize_model_tags(data.get("capabilities"))
        return ManagedModel.model_validate(data)


class RuntimeContainer:
    def __init__(
        self,
        *,
        loader: AgentConfigLoader | None = None,
        runtime: AgentRuntime | None = None,
        model_manager: ModelManager | None = None,
        bootstrap: RuntimeBootstrapConfig | dict[str, Any] | None = None,
    ) -> None:
        self.loader = loader or AgentConfigLoader()
        self.model_manager = model_manager or ModelManager()
        self.runtime = runtime or AgentRuntime()
        if bootstrap is not None:
            self.configure(bootstrap)
        else:
            self.configure_default_model_from_agent("default")

    def configure(self, bootstrap: RuntimeBootstrapConfig | dict[str, Any]) -> None:
        config = (
            bootstrap
            if isinstance(bootstrap, RuntimeBootstrapConfig)
            else RuntimeBootstrapConfig.model_validate(bootstrap)
        )
        self.model_manager.configure(config.models, default_model_id=config.default_model_id)
        model_config = self._default_model_config(config)
        normalized_model, fields = AgentRuntime._normalize_model_config(model_config)
        self.runtime.model_config = normalized_model
        self.runtime._model_config_fields = fields
        self.runtime.skills = AgentRuntime._normalize_skills(config.skills)

    def model_runtime_options(self, model_id: str) -> RuntimeOptions | None:
        return self.model_manager.runtime_options_for(model_id)

    def model_payloads(self) -> list[dict[str, Any]]:
        return [model.public_payload() for model in self.model_manager.list()]

    def current_model_id(self) -> str | None:
        model = self.model_manager.default()
        return model.id if model is not None else None

    def configure_default_model_from_agent(self, agent_name: str = "default") -> None:
        if self.model_manager.default() is not None:
            return
        try:
            agent_config = self.loader.load(agent_name)
        except FileNotFoundError:
            return
        self.model_manager.configure_from_agent_default(agent_config)

    def _default_model_config(self, config: RuntimeBootstrapConfig) -> ModelConfig | dict[str, Any] | None:
        if config.model is not None:
            return config.model
        model = self.model_manager.default()
        return model.config if model is not None else None


default_container = RuntimeContainer()
