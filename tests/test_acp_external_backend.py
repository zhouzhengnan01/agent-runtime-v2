from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import acp as acp_api
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfigLoader
from app.main import create_app
from app.protocols.acp.external_backend import ExternalAcpClient, prompt_blocks_from_params


def test_acp_websocket_can_proxy_external_acp_stdio_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    import json as _json
    (config_dir / "external-behavior.json").write_text(
        _json.dumps({
            "name": "external-behavior",
            "display_name": "External Behavior",
            "description": "Proxy to an external ACP stdio behavior agent.",
            "backend": {
                "type": "acp_stdio",
                "command": sys.executable,
                "args": ["-m", "app.cli", "acp-stdio", "--agent", "default"],
                "cwd": str(project_root),
                "env": {"PYTHONPATH": str(project_root)},
            },
            "runtime": {"stateless": True},
            "tools": [],
            "skills": [],
            "workflows": {},
            "quality": {"auto_repair": False, "verify_outputs": False, "fail_on_missing_artifact": False},
            "prompts": {"system": "", "response_language": "zh-CN"},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(acp_api, "loader", AgentConfigLoader(tmp_path))

    client = TestClient(create_app())
    with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        websocket.receive_json()

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "new_session",
                "params": {
                    "cwd": str(project_root),
                    "_meta": {
                        "agentName": "external-behavior",
                        "runtimeOptions": {"threadId": "external-acp-ws-test"},
                    },
                },
            }
        )
        created = websocket.receive_json()
        session_id = created["result"]["sessionId"]
        assert created["result"]["backend"]["type"] == "acp_stdio"

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "人员翻越围栏进入禁区"}],
                    "_meta": {"runtimeOptions": {"workflow": "evidence_first_detection"}},
                },
            }
        )

        updates: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(30):
            packet = websocket.receive_json()
            if packet.get("method") == "session/update":
                updates.append(packet)
                continue
            if packet.get("id") == 3:
                final = packet
                break

        assert final is not None
        assert final["result"]["stopReason"] == "end_turn"
        assert final["result"]["backend"]["type"] == "acp_stdio"
        assert any(
            "待上传" in update["params"]["update"]["content"]["text"]
            for update in updates
            if update["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        )

        websocket.send_json(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/close",
                "params": {"sessionId": session_id},
            }
        )
        closed = websocket.receive_json()
        assert closed["id"] == 4
        assert "error" not in closed


def test_external_acp_prompt_blocks_preserve_multimodal_prompt() -> None:
    blocks = prompt_blocks_from_params(
        {
            "prompt": [
                {"type": "text", "text": "inspect"},
                {"type": "image", "data": "aW1hZ2U=", "mimeType": "image/png", "uri": "file:///camera.png"},
                {"type": "audio", "data": "YXVkaW8=", "mimeType": "audio/wav"},
                {
                    "type": "resource_link",
                    "name": "log.txt",
                    "uri": "/mnt/user-data/uploads/log.txt",
                    "mimeType": "text/plain",
                },
                {
                    "type": "resource",
                    "resource": {"uri": "file:///ctx.txt", "text": "context", "mimeType": "text/plain"},
                },
            ]
        }
    )

    assert [block.type for block in blocks] == ["text", "image", "audio", "resource_link", "resource"]
    assert blocks[1].mime_type == "image/png"
    assert blocks[2].mime_type == "audio/wav"
    assert blocks[3].uri == "/mnt/user-data/uploads/log.txt"
    assert blocks[4].resource.text == "context"


def test_external_acp_client_file_callbacks_are_bounded(tmp_path: Path) -> None:
    asyncio.run(_run_external_file_callback_flow(tmp_path))


def test_external_acp_client_permission_and_terminal_callbacks(tmp_path: Path) -> None:
    asyncio.run(_run_external_permission_and_terminal_flow(tmp_path))


async def _run_external_file_callback_flow(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "threads")
    client = ExternalAcpClient(store, "external-callbacks")
    paths = store.prepare_thread("external-callbacks")
    (paths.uploads / "input.txt").write_text("u1\nu2", encoding="utf-8")

    await client.write_text_file("line1\nline2", "notes/output.txt", "backend-session")
    written = await client.read_text_file(
        "/mnt/user-data/workspace/notes/output.txt",
        "backend-session",
        line=2,
        limit=1,
    )
    uploaded = await client.read_text_file("/mnt/user-data/uploads/input.txt", "backend-session")

    assert written.content == "line2"
    assert uploaded.content == "u1\nu2"
    assert (paths.workspace / "notes" / "output.txt").read_text(encoding="utf-8") == "line1\nline2"
    with pytest.raises(ValueError, match="path traversal"):
        await client.read_text_file("../outside.txt", "backend-session")
    with pytest.raises(ValueError, match="workspace"):
        await client.write_text_file("x", "/tmp/out.txt", "backend-session")


async def _run_external_permission_and_terminal_flow(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "threads")
    client = ExternalAcpClient(store, "external-terminal")

    permission = await client.request_permission()
    assert permission.outcome.outcome == "cancelled"

    created = await client.create_terminal(
        sys.executable,
        "backend-session",
        args=["-c", "print('hello from acp terminal')"],
    )
    waited = await client.wait_for_terminal_exit("backend-session", created.terminal_id)
    output = await client.terminal_output("backend-session", created.terminal_id)
    released = await client.release_terminal("backend-session", created.terminal_id)

    assert waited.exit_code == 0
    assert output.exit_status is not None
    assert output.exit_status.exit_code == 0
    assert "hello from acp terminal" in output.output
    assert released is not None
