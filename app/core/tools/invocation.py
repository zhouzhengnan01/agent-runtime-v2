from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from app.core.artifacts import ArtifactStore
from app.core.memory import MarkdownMemoryStore, MemoryStore
from app.core.skills import SkillRunner
from app.core.tools.providers import (
    DelegateToolProvider,
    LocalToolProvider,
    ManualToolProvider,
    MarkdownMemoryToolProvider,
    MemoryToolProvider,
    SkillToolProvider,
)
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
        memory_store: MemoryStore | None = None,
        markdown_memory_store: MarkdownMemoryStore | None = None,
        root_dir: Path | None = None,
    ) -> None:
        registry_root = getattr(registry, "root_dir", None)
        self.root_dir = root_dir or (
            registry_root if isinstance(registry_root, Path) else Path(__file__).resolve().parents[3]
        )
        self.artifact_store = artifact_store or ArtifactStore()
        self.memory_store = memory_store or MemoryStore()
        self.markdown_memory_store = markdown_memory_store or MarkdownMemoryStore()
        self.registry = registry or ToolRegistry(
            root_dir=self.root_dir,
            artifact_store=self.artifact_store,
            skill_runner=skill_runner,
        )
        self.providers: dict[str, ToolProvider] = {
            LocalToolProvider.source_type: LocalToolProvider(artifact_store=self.artifact_store),
            DelegateToolProvider.source_type: DelegateToolProvider(root_dir=self.root_dir),
            ManualToolProvider.source_type: ManualToolProvider(),
            MemoryToolProvider.source_type: MemoryToolProvider(memory_store=self.memory_store),
            MarkdownMemoryToolProvider.source_type: MarkdownMemoryToolProvider(
                markdown_memory_store=self.markdown_memory_store
            ),
            SkillToolProvider.source_type: SkillToolProvider(
                artifact_store=self.artifact_store,
                skill_runner=skill_runner,
            ),
        }

    def list_tools(self, *, include_disabled: bool = False, include_skill_tools: bool = True) -> list[ToolDefinition]:
        return self.registry.list(include_disabled=include_disabled, include_skill_tools=include_skill_tools)

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
