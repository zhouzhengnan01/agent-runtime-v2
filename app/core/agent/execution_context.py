from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.core.agent.session import SessionConversationStore
from app.core.artifacts import ArtifactStore
from app.core.artifacts.store import ThreadPaths
from app.core.config import AgentConfig
from app.schemas import ChatRequest


@dataclass(frozen=True)
class ExecutionContext:
    """Per-run execution context shared by sync and streaming paths.

    This is an internal runtime helper only. It does not change the on-disk
    thread layout, the persisted event format, or the public request schema.
    """

    agent_config: AgentConfig
    request: ChatRequest
    paths: ThreadPaths
    workflow_name: str | None
    input_required: list[dict[str, Any]]
    conversation: list[dict[str, Any]]


def build_execution_context(
    *,
    agent_config: AgentConfig,
    request: ChatRequest,
    artifact_store: ArtifactStore,
    session_store: SessionConversationStore,
    prepare_execution: Callable[[AgentConfig, ChatRequest], tuple[AgentConfig, ChatRequest]],
    selected_workflow_name: Callable[[AgentConfig, ChatRequest], str | None],
    request_required_inputs: Callable[[ChatRequest], list[dict[str, Any]]],
) -> ExecutionContext:
    agent_config, request = prepare_execution(agent_config, request)
    if not request.messages:
        raise ValueError("messages must not be empty")
    paths = artifact_store.prepare_thread(request.runtime_options.thread_id)
    conversation = session_store.merge(session_store.load(paths), request.messages)
    return ExecutionContext(
        agent_config=agent_config,
        request=request,
        paths=paths,
        workflow_name=selected_workflow_name(agent_config, request),
        input_required=request_required_inputs(request),
        conversation=conversation,
    )
