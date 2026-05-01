from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import acp as acp_api
from app.core.config import AgentConfigLoader
from app.main import create_app


def test_acp_websocket_can_proxy_external_acp_stdio_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "config" / "agents"
    config_dir.mkdir(parents=True)
    (config_dir / "external-behavior.json").write_text(
        f"""
{{
  "name": "external-behavior",
  "display_name": "External Behavior",
  "description": "Proxy to an external ACP stdio behavior agent.",
  "backend": {{
    "type": "acp_stdio",
    "command": "{sys.executable}",
    "args": ["-m", "app.cli", "acp-stdio", "--agent", "behavior-detector"],
    "cwd": "{project_root}",
    "env": {{"PYTHONPATH": "{project_root}"}}
  }},
  "runtime": {{"stateless": true}},
  "tools": [],
  "skills": [],
  "workflows": {{}},
  "quality": {{"auto_repair": false, "verify_outputs": false, "fail_on_missing_artifact": false}},
  "prompts": {{"system": "", "response_language": "zh-CN"}}
}}
""",
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
                    "agentName": "external-behavior",
                    "threadId": "external-acp-ws-test",
                    "cwd": str(project_root),
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
