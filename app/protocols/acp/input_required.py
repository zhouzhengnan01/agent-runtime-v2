from __future__ import annotations

from app.core.agent.input_required import required_inputs_for_result
from app.schemas import AgentRunResult, ChatRequest


END_TURN_STOP_REASON = "end_turn"


def stop_reason_for_result(result: AgentRunResult, request: ChatRequest) -> str:
    """Return the ACP-compatible stop reason and enrich metadata when input is needed."""

    required_inputs = required_inputs_for_result(result, request)
    if not required_inputs:
        return END_TURN_STOP_REASON
    result.metadata = {
        **result.metadata,
        "requires_input": True,
        "required_inputs": required_inputs,
    }
    return END_TURN_STOP_REASON
