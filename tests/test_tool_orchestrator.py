from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmToolCall
from app.core.tools.orchestrator import ToolOrchestrator
from app.core.tools.schemas import ToolInvocationResult
from app.schemas import RuntimeOptions


class _ToolService:
    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.root_dir = Path(__file__).resolve().parents[1]
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


def test_tool_orchestrator_injects_runtime_model_context_for_skills() -> None:
    service = _ToolService()
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(
        name="default",
        display_name="Default",
        model=ModelConfig(
            model="agent-model",
            base_url="http://agent-model.local/v1",
            api_key="agent-key",
            temperature=0.2,
            max_tokens=1024,
        ),
    )

    orchestrator.execute(
        LlmToolCall(id="call-1", name="demo_tool", arguments="{}"),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
        runtime_options=RuntimeOptions(
            model_name="runtime-model",
            base_url="http://runtime-model.local/v1",
            api_key="runtime-key",
            temperature=0.5,
            max_tokens=2048,
            request_timeout_seconds=33,
        ),
    )

    assert service.arguments is not None
    assert service.arguments["_llm_model"] == "runtime-model"
    assert service.arguments["_llm_base_url"] == "http://runtime-model.local/v1"
    assert service.arguments["_llm_api_key"] == "runtime-key"
    assert service.arguments["_llm_temperature"] == 0.5
    assert service.arguments["_llm_max_tokens"] == 2048
    assert service.arguments["_llm_request_timeout_seconds"] == 33
    assert recorder.events[-1].data["arguments"] == {}


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


def test_tool_orchestrator_autofills_upload_file_content_from_workspace(tmp_path: Path) -> None:
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = artifact_store.prepare_thread("thread-tool")
    target = paths.workspace / "background.svg"
    target.write_text("<svg/>", encoding="utf-8")
    service = _ToolService(artifact_store=artifact_store)
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(name="default", display_name="Default")

    outcome = orchestrator.execute(
        LlmToolCall(
            id="call-1",
            name="jetlinks_session__visual-bigscreen_UploadFile",
            arguments='{"fileName": "background.svg", "contentType": "image/svg+xml;charset=UTF-8", "content": null}',
        ),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
    )

    expected_content = base64.b64encode(b"<svg/>").decode("ascii")
    assert outcome.result.is_error is False
    assert service.arguments is not None
    assert service.arguments["content"] == expected_content
    assert service.arguments["contentType"] == "image/svg+xml;charset=UTF-8"
    assert service.arguments["_upload_file_content_autofilled"] is True
    observable = recorder.events[-1].data["arguments"]
    assert observable["content"] == f"<base64 omitted chars={len(expected_content)}>"


def test_tool_orchestrator_preserves_upload_file_existing_content(tmp_path: Path) -> None:
    service = _ToolService(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(name="default", display_name="Default")

    outcome = orchestrator.execute(
        LlmToolCall(
            id="call-1",
            name="jetlinks_session__visual-bigscreen_UploadFile",
            arguments='{"fileName": "missing.svg", "content": "already-encoded"}',
        ),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
    )

    assert outcome.result.is_error is False
    assert service.arguments is not None
    assert service.arguments["content"] == "already-encoded"
    assert "_upload_file_content_autofilled" not in service.arguments


def test_tool_orchestrator_autofills_upload_file_content_from_virtual_path(tmp_path: Path) -> None:
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = artifact_store.prepare_thread("thread-tool")
    target = paths.outputs / "background.svg"
    target.write_text("<svg/>", encoding="utf-8")
    service = _ToolService(artifact_store=artifact_store)
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(name="default", display_name="Default")

    outcome = orchestrator.execute(
        LlmToolCall(
            id="call-1",
            name="jetlinks_session__visual-bigscreen_UploadFile",
            arguments='{"fileName": "ignored.svg", "path": "/mnt/user-data/outputs/background.svg", "content": null}',
        ),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
    )

    assert outcome.result.is_error is False
    assert service.arguments is not None
    assert service.arguments["content"] == base64.b64encode(b"<svg/>").decode("ascii")
    assert service.arguments["fileName"] == "ignored.svg"
    assert service.arguments["contentType"] == "image/svg+xml;charset=UTF-8"


def test_tool_orchestrator_upload_file_missing_file_is_failed_tool_result(tmp_path: Path) -> None:
    service = _ToolService(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    orchestrator = ToolOrchestrator(service)  # type: ignore[arg-type]
    recorder = EventRecorder(agent="default", thread_id="thread-tool")
    agent = AgentConfig(name="default", display_name="Default")

    outcome = orchestrator.execute(
        LlmToolCall(
            id="call-1",
            name="jetlinks_session__visual-bigscreen_UploadFile",
            arguments='{"fileName": "missing.svg", "content": null}',
        ),
        agent_config=agent,
        thread_id="thread-tool",
        recorder=recorder,
    )

    assert outcome.result.is_error is True
    assert outcome.result.structured_content["error_code"] == "TOOL_EXECUTION_FAILED"
    assert "no matching file" in outcome.result.structured_content["error"]
    assert [event.type for event in recorder.events] == ["tool.started", "tool.failed"]
