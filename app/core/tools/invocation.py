from __future__ import annotations

from typing import Any, Protocol

from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRunner
from app.core.tools.providers import ManualToolProvider, SkillToolProvider
from app.core.tools.registry import ToolRegistry
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


class ToolProvider(Protocol):
    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult: ...


class ToolInvocationService:
    """Protocol-neutral tool discovery and invocation service."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        artifact_store: ArtifactStore | None = None,
        skill_runner: SkillRunner | None = None,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.registry = registry or ToolRegistry(artifact_store=self.artifact_store, skill_runner=skill_runner)
        self.providers: dict[str, ToolProvider] = {
            ManualToolProvider.source_type: ManualToolProvider(),
            SkillToolProvider.source_type: SkillToolProvider(
                artifact_store=self.artifact_store,
                skill_runner=skill_runner,
            ),
        }

    def list_tools(self, *, include_disabled: bool = False) -> list[ToolDefinition]:
        return self.registry.list(include_disabled=include_disabled)

    def get_tool(self, name: str) -> ToolDefinition:
        return self.registry.get(name)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvocationResult:
        tool = self.registry.get(name)
        if not tool.enabled:
            raise ValueError(f"Tool is disabled: {name}")
        source_type = str(tool.source.get("type") or "manual")
        provider = self.providers.get(source_type)
        if provider is None:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"Unsupported tool source type: {source_type}"}],
                structured_content={"tool_name": name, "source_type": source_type},
                is_error=True,
            )
        return provider.call(tool, arguments)
