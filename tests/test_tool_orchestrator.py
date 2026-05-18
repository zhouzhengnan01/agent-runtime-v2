from __future__ import annotations

from typing import Any

from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmToolCall
from app.core.tools.orchestrator import ToolOrchestrator
from app.core.tools.schemas import ToolInvocationResult


class _ToolService:
    def __init__(self) -> None:
        self.arguments: dict[str, Any] | None = None

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolInvocationResult:
        self.arguments = dict(arguments)
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"called {name}"}],
            structured_content={"tool_name": name, "arguments": arguments},
            is_error=False,
        )


def test_tool_orchestrator_preserves_existing_tool_events_and_scoped_arguments() -> None:
    service = _ToolService()
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(name="default", display_name="Default")

    outcome = orchestrator.execute(
        LlmToolCall(id="call-1", name="demo_tool", arguments='{"value": 1}'),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
    )

    assert outcome.result.is_error is False
    assert service.arguments is not None
    assert service.arguments["value"] == 1
    assert service.arguments["_thread_id"] == "thread-tool"
    assert [event.type for event in recorder.events] == ["tool.started", "tool.completed"]
    assert recorder.events[-1].data["arguments"] == {"value": 1}


def test_tool_orchestrator_maps_bad_arguments_to_failed_tool_result() -> None:
    service = _ToolService()
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(name="default", display_name="Default")

    outcome = orchestrator.execute(
        LlmToolCall(id="call-1", name="demo_tool", arguments="[1, 2]"),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
    )

    assert outcome.result.is_error is True
    assert outcome.result.structured_content["error_code"] == "TOOL_ARGUMENTS_INVALID"
    assert [event.type for event in recorder.events] == ["tool.started", "tool.failed"]
    assert "tool arguments must be a JSON object" in outcome.result.to_mcp_result()["content"][0]["text"]
