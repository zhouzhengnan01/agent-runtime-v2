from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.config import AgentConfigLoader
from app.core.llm import OpenAICompatibleClient
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult
from app.schemas import Message, RuntimeOptions


class DelegateToolProvider:
    """Lightweight delegated subtask provider inspired by Hermes delegate_task."""

    source_type = "delegate"

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[4]

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        del tool
        try:
            return self._delegate(arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"delegate_task failed: {exc}"}],
                structured_content={"tool_name": "delegate_task", "error": str(exc)},
                is_error=True,
            )

    def _delegate(self, arguments: dict[str, Any]) -> ToolInvocationResult:
        goal = str(arguments.get("goal") or "").strip()
        if not goal:
            raise ValueError("goal is required")
        context = str(arguments.get("context") or "").strip()
        parent_agent = str(arguments.get("_agent_name") or "default").strip() or "default"
        child_agent_name = str(arguments.get("agent_name") or parent_agent).strip() or parent_agent
        loader = AgentConfigLoader(self.root_dir)
        child_config = loader.load(child_agent_name)
        runtime_options = RuntimeOptions(
            model_name=_optional_str(arguments.get("model_name")),
            temperature=_optional_float(arguments.get("temperature")),
            max_tokens=_optional_int(arguments.get("max_tokens")),
            request_timeout_seconds=_optional_float(arguments.get("request_timeout_seconds")),
        )
        client = OpenAICompatibleClient(child_config, runtime_options=runtime_options)
        reply = client.complete_sync(
            self._system_prompt(child_config.prompts.system),
            [Message(role="user", content=self._user_prompt(goal, context))],
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": reply}],
            structured_content={
                "tool_name": "delegate_task",
                "agent_name": child_config.name,
                "parent_agent_name": parent_agent,
                "goal_chars": len(goal),
                "context_chars": len(context),
                "reply_chars": len(reply),
                "model": client.model,
                "llm_configured": client.configured,
            },
            is_error=False,
        )

    @staticmethod
    def _system_prompt(base_prompt: str) -> str:
        delegate_prompt = (
            "You are a delegated subagent called by a parent JetLinks agent. "
            "Work only on the delegated goal. Do not ask the user questions, do not write shared memory, "
            "and do not claim to have used tools. Return a concise result with findings, decisions, "
            "and any remaining uncertainty."
        )
        base = base_prompt.strip()
        return f"{base}\n\n{delegate_prompt}" if base else delegate_prompt

    @staticmethod
    def _user_prompt(goal: str, context: str) -> str:
        if context:
            return f"Delegated goal:\n{goal}\n\nContext from parent agent:\n{context}"
        return f"Delegated goal:\n{goal}"


def delegate_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="delegate_task",
            title="Delegate Task",
            description=(
                "Delegate a focused independent subtask to a fresh no-tool child model call. "
                "Use for parallel-style analysis, second-pass review, or decomposition; the parent receives only the summary."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "Focused subtask for the child agent."},
                    "context": {"type": "string", "description": "Relevant context to give the child agent."},
                    "agent_name": {
                        "type": "string",
                        "description": "Optional child agent config name. Defaults to the current agent.",
                    },
                    "model_name": {"type": "string", "description": "Optional runtime model override."},
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "max_tokens": {"type": "integer", "minimum": 128, "maximum": 32000},
                    "request_timeout_seconds": {"type": "number", "minimum": 1, "maximum": 3600},
                },
                "required": ["goal"],
            },
            source={"type": DelegateToolProvider.source_type},
            editable=False,
        )
    ]


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
