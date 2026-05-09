from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.workflow.config import WorkflowConfig
from app.core.workflow.plugins import WorkflowPluginManager
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


class WorkflowPlugin(Protocol):
    """Runtime plugin for an optional workflow implementation."""

    def run_with_events(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        attachments: list[Attachment],
        thread_id: str | None,
        on_event: Callable[[ChatEvent], None] | None = None,
        workflow_name: str | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> tuple[AgentRunResult, list[ChatEvent]]: ...


@dataclass(frozen=True)
class RegisteredWorkflow:
    config: WorkflowConfig
    plugin: WorkflowPlugin


class WorkflowRegistry:
    """Registry for optional named workflow plugins."""

    def __init__(
        self,
        plugins: Mapping[str, WorkflowPlugin] | None = None,
        configs: Mapping[str, WorkflowConfig] | None = None,
    ) -> None:
        self._plugins = dict(plugins or {})
        self._configs = dict(configs or {})

    @classmethod
    def builtin(cls, artifact_store: ArtifactStore, root_dir: Path | None = None) -> WorkflowRegistry:
        registry = cls()
        for loaded in WorkflowPluginManager(root_dir).load_workflows(artifact_store).values():
            if loaded.config.enabled:
                registry.register(loaded.config.name, loaded.plugin, config=loaded.config)
        return registry

    def names(self) -> set[str]:
        return set(self._plugins)

    def get(self, name: str) -> WorkflowPlugin | None:
        return self._plugins.get(name)

    def config(self, name: str) -> WorkflowConfig | None:
        return self._configs.get(name)

    def list_registered(self) -> list[RegisteredWorkflow]:
        return [
            RegisteredWorkflow(config=self._configs[name], plugin=self._plugins[name])
            for name in sorted(self._plugins)
            if name in self._configs
        ]

    def register(self, name: str, plugin: WorkflowPlugin, *, config: WorkflowConfig | None = None) -> None:
        self._plugins[name] = plugin
        if config is not None:
            self._configs[name] = config
