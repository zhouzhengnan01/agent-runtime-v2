from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.agent.primary_skill_context import PrimarySkillContext
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.core.tools import ToolDefinition


@dataclass
class ToolLoopState:
    """Mutable state for a single tool-loop turn.

    This stays internal to the agent execution layer. It does not affect the
    persisted thread layout, event schema, or public runtime request format.
    """

    llm: OpenAICompatibleClient
    llm_metadata: dict[str, Any]
    conversation: list[dict[str, Any]]
    system_prompt: str
    memory_context_count: int
    tools: list[dict[str, Any]]
    tool_definitions: list[ToolDefinition]
    priority_tool_definitions: list[ToolDefinition] | None = None
    secondary_tool_definitions: list[ToolDefinition] | None = None
    hidden_tool_definitions: list[ToolDefinition] | None = None
    turn_phase: str = "execute"
    turn_policy_reason: str = ""
    primary_skill_context: PrimarySkillContext | None = None
    available_artifacts: list[dict[str, Any]] | None = None
    latest_tool_results: list[dict[str, Any]] | None = None
    verification_verdict: str = "unknown"
    verification_reason: str = ""
    context_compactions: int = 0
    last_context_event: dict[str, Any] | None = None
    rounds: int = 0
    tool_call_count: int = 0
    required_inputs: list[dict[str, Any]] | None = None
    artifacts: list[dict[str, Any]] | None = None
    latest_assistant_reply: str = ""

    def __post_init__(self) -> None:
        if self.last_context_event is None:
            self.last_context_event = {}
        if self.required_inputs is None:
            self.required_inputs = []
        if self.artifacts is None:
            self.artifacts = []
        if self.available_artifacts is None:
            self.available_artifacts = []
        if self.latest_tool_results is None:
            self.latest_tool_results = []
        if self.priority_tool_definitions is None:
            self.priority_tool_definitions = []
        if self.secondary_tool_definitions is None:
            self.secondary_tool_definitions = []
        if self.hidden_tool_definitions is None:
            self.hidden_tool_definitions = []
