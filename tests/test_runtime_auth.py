from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.runtime import default_container
from app.main import create_app


def test_create_app_loads_runtime_bootstrap_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original_models = [model.model_copy(deep=True) for model in default_container.model_manager.list()]
    original_default_model_id = default_container.model_manager.default_model_id
    original_model_config = (
        default_container.runtime.model_config.model_copy(deep=True)
        if default_container.runtime.model_config is not None
        else None
    )
    original_model_config_fields = set(default_container.runtime._model_config_fields)
    original_skills = list(default_container.runtime.skills) if default_container.runtime.skills is not None else None
    bootstrap_path = tmp_path / "runtime.local.json"
    bootstrap_path.write_text(
        json.dumps(
            {
                "default_model_id": "managed-qwen",
                "models": [
                    {
                        "id": "managed-qwen",
                        "display_name": "Managed Qwen",
                        "config": {
                            "model": "runtime-model",
                            "base_url": "http://runtime.local/v1",
                            "api_key": "runtime-key",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("JETLINKS_RUNTIME_BOOTSTRAP", str(bootstrap_path))

    try:
        client = TestClient(create_app())
        with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]) as websocket:
            websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "new_session", "params": {}})
            created = websocket.receive_json()["result"]
    finally:
        default_container.model_manager.configure(original_models, default_model_id=original_default_model_id)
        default_container.runtime.model_config = original_model_config
        default_container.runtime._model_config_fields = original_model_config_fields
        default_container.runtime.skills = original_skills

    assert created["models"]["currentModelId"] == "managed-qwen"
    assert created["models"]["availableModels"][0]["id"] == "managed-qwen"
    assert created["runtimeOptions"]["modelName"] == "runtime-model"
    assert created["runtimeOptions"]["baseUrl"] == "http://runtime.local/v1"


def test_agent_run_is_open_in_development_mode_when_token_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setenv("RUNTIME_API_TOKEN", "runtime-secret")
    monkeypatch.delenv("RUNTIME_AUTH_MODE", raising=False)
    monkeypatch.delenv("JETLINKS_RUNTIME_MODE", raising=False)
    client = TestClient(create_app())

    response = client.post(
        "/api/agents/default/runs",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["workflow"] == "agent_loop"


def test_agent_run_requires_token_in_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setenv("RUNTIME_AUTH_MODE", "production")
    monkeypatch.setenv("RUNTIME_API_TOKEN", "runtime-secret")
    client = TestClient(create_app())

    denied = client.post(
        "/api/agents/default/runs",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    allowed = client.post(
        "/api/agents/default/runs",
        headers={"Authorization": "Bearer runtime-secret"},
        json={"messages": [{"role": "user", "content": "你好"}]},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_agent_run_accepts_query_token_in_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setenv("RUNTIME_AUTH_MODE", "prod")
    monkeypatch.setenv("RUNTIME_API_TOKEN", "runtime-secret")
    client = TestClient(create_app())

    response = client.post(
        "/api/agents/default/runs?token=runtime-secret",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )

    assert response.status_code == 200


def test_agent_stream_requires_token_in_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setenv("RUNTIME_AUTH_MODE", "production")
    monkeypatch.setenv("RUNTIME_API_TOKEN", "runtime-secret")
    client = TestClient(create_app())

    denied = client.post(
        "/api/agents/default/runs/stream",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    allowed = client.post(
        "/api/agents/default/runs/stream",
        headers={"Authorization": "Bearer runtime-secret"},
        json={"messages": [{"role": "user", "content": "你好"}]},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert "run.completed" in allowed.text


def test_acp_websocket_requires_token_in_production_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_llm(monkeypatch)
    monkeypatch.setenv("RUNTIME_AUTH_MODE", "production")
    monkeypatch.setenv("RUNTIME_API_TOKEN", "runtime-secret")
    client = TestClient(create_app())

    try:
        with client.websocket_connect("/api/acp/ws", subprotocols=["acp.v1"]):
            raise AssertionError("expected websocket auth failure")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008
        assert exc.reason == "Missing or invalid runtime token."

    with client.websocket_connect("/api/acp/ws?token=runtime-secret", subprotocols=["acp.v1"]) as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        initialized = websocket.receive_json()

    assert initialized["result"]["agentInfo"]["name"] == "jetlinks-agent-runtime-v2"


def _disable_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_DISABLED", "1")
