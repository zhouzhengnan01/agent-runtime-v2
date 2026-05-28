from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from acp import helpers as acp_helpers

from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.runtime import ModelManager
from app.protocols.acp.transport_stdio import JetLinksAcpStdioAgent, _event_to_sdk_updates
from app.schemas import ChatEvent, ChatRequest


class CapturingStdioRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        del agent_config
        self.requests.append(request)
        yield ChatEvent(type="run.completed", data={})


class BlockingStdioRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def iter_events(self, agent_config: Any, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        del agent_config, request
        self.started.set()
        yield ChatEvent(type="run.started", data={})
        await asyncio.sleep(60)


def test_acp_stdio_initialize_new_session_and_prompt() -> None:
    asyncio.run(_run_acp_stdio_flow())


def test_acp_stdio_agent_routes_multimodal_prompt_and_model() -> None:
    asyncio.run(_run_acp_stdio_multimodal_flow())


def test_acp_stdio_cancel_interrupts_active_prompt() -> None:
    asyncio.run(_run_acp_stdio_cancel_flow())


def test_acp_stdio_delete_session_files_extension(tmp_path: Path) -> None:
    asyncio.run(_run_acp_stdio_delete_files_flow(tmp_path))


def test_acp_stdio_maps_tool_events_to_sdk_updates() -> None:
    started = _event_to_sdk_updates(
        ChatEvent(type="tool.started", data={"tool_name": "artifact_write", "tool_call_id": "call-write"}),
        "message-1",
    )
    completed = _event_to_sdk_updates(
        ChatEvent(
            type="tool.completed",
            data={
                "tool_name": "artifact_write",
                "tool_call_id": "call-write",
                "structured_content": {"path": "outputs/result.md"},
            },
        ),
        "message-1",
    )
    delta = _event_to_sdk_updates(ChatEvent(type="agent.message.delta", data={"text": "hello"}), "message-1")

    assert started[0].field_meta["jetlinksRuntimeEvent"]["type"] == "tool.started"
    assert completed[0].field_meta["jetlinksRuntimeEvent"]["type"] == "tool.completed"
    assert completed[0].raw_output == {"path": "outputs/result.md"}
    assert delta[0].field_meta["jetlinksRuntimeEvent"]["type"] == "agent.message.delta"


def test_acp_stdio_skips_empty_agent_message_updates() -> None:
    assert _event_to_sdk_updates(ChatEvent(type="agent.message", data={"text": ""}), "message-1") == []
    assert _event_to_sdk_updates(ChatEvent(type="agent.message.delta", data={"text": ""}), "message-1") == []


async def _run_acp_stdio_flow() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.cli",
        "acp-stdio",
        "--agent",
        "default",
        cwd=project_root,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": 1},
            },
        )
        initialized = await _read(process)
        assert initialized["id"] == 1
        assert initialized["result"]["agentInfo"]["name"] == "jetlinks-agent-runtime-v2"

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(project_root), "mcpServers": []},
            },
        )
        created = await _read(process)
        session_id = created["result"]["sessionId"]
        assert session_id.startswith("acp-session-")

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/list",
                "params": {"cwd": str(project_root)},
            },
        )
        listed = await _read(process)
        assert listed["id"] == 3
        assert any(item["sessionId"] == session_id for item in listed["result"]["sessions"])

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/set_model",
                "params": {"sessionId": session_id, "modelId": "stdio-test-model"},
            },
        )
        set_model = await _read(process)
        assert set_model["id"] == 4
        assert "error" not in set_model

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "人员翻越围栏进入禁区"}],
                    "_meta": {"workflow": "evidence_first_detection"},
                },
            },
        )

        notifications: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(30):
            packet = await _read(process)
            if packet.get("method") == "session/update":
                notifications.append(packet)
                continue
            if packet.get("id") == 5:
                final = packet
                break

        assert final is not None
        assert final["result"]["stopReason"] == "end_turn"
        assert any(
            "待上传" in notification["params"]["update"]["content"]["text"]
            for notification in notifications
            if notification["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        )

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "session/close",
                "params": {"sessionId": session_id},
            },
        )
        closed = await _read(process)
        assert closed["id"] == 6
        assert "error" not in closed
    finally:
        if process.stdin is not None:
            process.stdin.close()
            await process.stdin.wait_closed()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.terminate()
            await process.wait()


async def _run_acp_stdio_multimodal_flow() -> None:
    runtime = CapturingStdioRuntime()
    model_manager = ModelManager(
        [
            {
                "id": "stdio-model",
                "config": {
                    "model": "stdio-runtime-model",
                    "base_url": "http://stdio-model.local/v1",
                    "api_key": "stdio-key",
                },
            }
        ]
    )
    agent = JetLinksAcpStdioAgent(runtime=runtime, model_manager=model_manager)
    created = await agent.new_session(cwd="/tmp", mcp_servers=[])
    await agent.set_session_model("stdio-model", created.session_id)

    response = await agent.prompt(
        [
            acp_helpers.text_block("分析图片"),
            acp_helpers.image_block("aW1hZ2U=", "image/png", uri="file:///camera.png"),
            acp_helpers.resource_link_block("log.txt", "/mnt/user-data/uploads/log.txt", mime_type="text/plain"),
            acp_helpers.resource_block(
                acp_helpers.embedded_text_resource("file:///ctx.txt", "ctx text", mime_type="text/plain")
            ),
        ],
        session_id=created.session_id,
    )

    assert response.stop_reason == "end_turn"
    request = runtime.requests[0]
    assert request.runtime_options.model_name == "stdio-runtime-model"
    assert request.runtime_options.base_url == "http://stdio-model.local/v1"
    assert request.runtime_options.api_key == "stdio-key"
    assert "分析图片" in request.messages[0].content
    assert "Embedded resource (file:///ctx.txt):" in request.messages[0].content
    attachment_types = {attachment.metadata["acp_type"] for attachment in request.attachments}
    assert attachment_types == {"image", "resource_link", "embedded_text_resource"}


async def _run_acp_stdio_cancel_flow() -> None:
    runtime = BlockingStdioRuntime()
    agent = JetLinksAcpStdioAgent(runtime=runtime)
    created = await agent.new_session(cwd="/tmp", mcp_servers=[])
    prompt_task = asyncio.create_task(agent.prompt([acp_helpers.text_block("hold")], session_id=created.session_id))
    await asyncio.wait_for(runtime.started.wait(), timeout=1)
    await agent.cancel(created.session_id)
    response = await asyncio.wait_for(prompt_task, timeout=1)
    assert response.stop_reason == "cancelled"


async def _run_acp_stdio_delete_files_flow(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    agent = JetLinksAcpStdioAgent(runtime=runtime)
    created = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    session = agent.sessions[created.session_id]
    paths = runtime.artifact_store.prepare_thread(session.thread_id)
    workspace_file = paths.workspace / "scratch.txt"
    uploads_file = paths.uploads / "camera.jpg"
    workspace_file.write_text("scratch", encoding="utf-8")
    uploads_file.write_bytes(b"image")

    dry_run = await agent.ext_method(
        "_jetlinks/session/delete_files",
        {"sessionId": created.session_id, "scopes": ["workspace"], "dryRun": True},
    )
    assert dry_run["dryRun"] is True
    assert dry_run["deletedFileCount"] == 1
    assert workspace_file.exists()

    deleted = await agent.ext_method(
        "jetlinks/session/delete_files",
        {"sessionId": created.session_id, "scopes": ["workspace"]},
    )
    assert deleted["sessionId"] == created.session_id
    assert deleted["threadId"] == session.thread_id
    assert deleted["scopes"] == ["workspace"]
    assert deleted["deletedFileCount"] == 1
    assert deleted["deleted"][0]["path"] == "/mnt/user-data/workspace/scratch.txt"
    assert not workspace_file.exists()
    assert paths.workspace.exists()
    assert uploads_file.exists()


async def _send(process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
    await process.stdin.drain()


async def _read(process: asyncio.subprocess.Process) -> dict[str, Any]:
    assert process.stdout is not None
    line = await asyncio.wait_for(process.stdout.readline(), timeout=5)
    if not line:
        stderr = b""
        if process.stderr is not None:
            stderr = await process.stderr.read()
        raise AssertionError(f"ACP stdio process closed before response. stderr={stderr.decode(errors='replace')}")
    return cast(dict[str, Any], json.loads(line))
