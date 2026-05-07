from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.workflow.artifact_workflow import ArtifactWorkflow
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


class WorkflowRegistry:
    """Registry for optional named workflow plugins."""

    def __init__(self, plugins: Mapping[str, WorkflowPlugin] | None = None) -> None:
        self._plugins = dict(plugins or {})

    @classmethod
    def builtin(cls, artifact_store: ArtifactStore) -> WorkflowRegistry:
        artifact_workflow = ArtifactWorkflow(artifact_store)
        return cls(
            {
                "artifact_workflow": artifact_workflow,
                "evidence_first_detection": artifact_workflow,
            }
        )

    def names(self) -> set[str]:
        return set(self._plugins)

    def get(self, name: str) -> WorkflowPlugin | None:
        return self._plugins.get(name)

    def register(self, name: str, plugin: WorkflowPlugin) -> None:
        self._plugins[name] = plugin
