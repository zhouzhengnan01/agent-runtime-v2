from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import mcp as mcp_api
from app.core.artifacts import ArtifactStore
from app.core.mcp import McpToolRegistry
from app.core.skills import SkillRunner
from app.main import create_app


def test_mcp_initialize_and_list_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mcp_runtime(tmp_path, monkeypatch)
    client = TestClient(create_app())

    initialize = client.post(
        "/mcp",
        headers={"Mcp-Method": "initialize"},
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
    )
    assert initialize.status_code == 200
    assert initialize.json()["result"]["capabilities"]["tools"]["listChanged"] is True

    listed = client.post(
        "/mcp",
        headers={"Mcp-Method": "tools/list"},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert listed.status_code == 200
    tool_names = {tool["name"] for tool in listed.json()["result"]["tools"]}
    assert "pptx-generation" not in tool_names
    assert "jetlinks_runtime_status" in tool_names


def test_mcp_call_manual_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mcp_runtime(tmp_path, monkeypatch)
    client = TestClient(create_app())

    response = client.post(
        "/mcp",
        headers={"Mcp-Method": "tools/call", "Mcp-Name": "jetlinks_runtime_status"},
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "jetlinks_runtime_status", "arguments": {"probe": True}},
        },
    )

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is False
    assert "MCP endpoint is reachable" in result["content"][0]["text"]


def test_mcp_management_saves_custom_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mcp_runtime(tmp_path, monkeypatch)
    client = TestClient(create_app())

    response = client.put(
        "/api/mcp/tools/demo_status_tool",
        json={
            "title": "Demo Status",
            "description": "A custom MCP status tool.",
            "enabled": True,
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "source": {"type": "manual", "response_template": "demo ok"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "demo_status_tool"
    assert data["source"]["response_template"] == "demo ok"

    listed = client.get("/api/mcp/tools")
    assert listed.status_code == 200
    listed_names = {tool["name"] for tool in listed.json()["tools"]}
    assert "demo_status_tool" in listed_names
    assert "pptx-generation" not in listed_names


def test_mcp_management_requires_admin_token_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mcp_runtime(tmp_path, monkeypatch)
    monkeypatch.setenv("RUNTIME_API_TOKEN", "admin-secret")
    client = TestClient(create_app())

    denied = client.put(
        "/api/mcp/tools/demo_status_tool",
        json={
            "title": "Demo Status",
            "description": "A custom MCP status tool.",
            "enabled": True,
            "source": {"type": "manual", "response_template": "demo ok"},
        },
    )
    allowed = client.put(
        "/api/mcp/tools/demo_status_tool",
        headers={"Authorization": "Bearer admin-secret"},
        json={
            "title": "Demo Status",
            "description": "A custom MCP status tool.",
            "enabled": True,
            "source": {"type": "manual", "response_template": "demo ok"},
        },
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200


def _patch_mcp_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config" / "mcp"
    config_dir.mkdir(parents=True)
    (config_dir / "tools.json").write_text(
        """
{
  "tools": [
    {
      "name": "jetlinks_runtime_status",
      "title": "JetLinks Runtime Status",
      "description": "Return a simple status message for MCP connectivity checks.",
      "enabled": true,
      "input_schema": {"type": "object"},
      "output_schema": {"type": "object"},
      "source": {
        "type": "manual",
        "response_template": "JetLinks Agent Runtime v2 MCP endpoint is reachable."
      }
    }
  ]
}
""",
        encoding="utf-8",
    )
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    monkeypatch.setattr(mcp_api, "registry", McpToolRegistry(tmp_path))
    monkeypatch.setattr(mcp_api, "artifact_store", store)
    monkeypatch.setattr(mcp_api, "skill_runner", SkillRunner(store))
