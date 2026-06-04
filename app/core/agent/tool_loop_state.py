from __future__ import annotations

from dataclasses import dataclass, field
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
    priority_tool_definitions: list[ToolDefinition] = field(default_factory=list)
    secondary_tool_definitions: list[ToolDefinition] = field(default_factory=list)
    hidden_tool_definitions: list[ToolDefinition] = field(default_factory=list)
    turn_phase: str = "execute"
    turn_policy_reason: str = ""
    primary_skill_context: PrimarySkillContext | None = None
    available_artifacts: list[dict[str, Any]] = field(default_factory=list)
    latest_tool_results: list[dict[str, Any]] = field(default_factory=list)
    verification_verdict: str = "unknown"
    verification_reason: str = ""
    context_compactions: int = 0
    last_context_event: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0
    tool_call_count: int = 0
    required_inputs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    latest_assistant_reply: str = ""
