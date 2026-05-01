from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pytest import MonkeyPatch

from app.core.config.agent_config import AgentConfig, ModelConfig
from app.core.llm import OpenAICompatibleClient
from app.schemas import Message


def test_model_config_can_use_json_values_without_model_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(
            model="json-model-name",
            base_url="http://llm.local/v1",
            api_key="json-key",
            temperature=0.1,
            max_tokens=128,
        ),
    )

    client = OpenAICompatibleClient(agent)

    assert client.configured is True
    assert client.model == "json-model-name"
    assert client.base_url == "http://llm.local/v1"
    assert client.api_key == "json-key"


def test_model_config_is_configured_without_api_key(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(
            model="json-model-name",
            base_url="http://llm.local/v1",
            api_key=None,
        ),
    )

    client = OpenAICompatibleClient(agent)

    assert client.configured is True
    assert client.model == "json-model-name"
    assert client.base_url == "http://llm.local/v1"
    assert client.api_key == ""
    assert client._headers() == {"Content-Type": "application/json"}


def test_env_overrides_json_model_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://env.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(
            model="json-model-name",
            base_url="http://llm.local/v1",
            api_key="json-key",
        ),
    )

    client = OpenAICompatibleClient(agent)

    assert client.configured is True
    assert client.model == "env-model"
    assert client.base_url == "http://env.local/v1"
    assert client.api_key == "env-key"


def test_complete_uses_json_model_config_without_authorization_header(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    seen: dict[str, Any] = {}

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        seen["url"] = url
        seen["json"] = json
        seen["headers"] = headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(
            model="json-model-name",
            base_url="http://llm.local/v1",
            api_key=None,
        ),
    )

    reply = asyncio.run(OpenAICompatibleClient(agent).complete("system", [Message(role="user", content="hi")]))

    assert reply == "ok"
    assert seen["url"] == "http://llm.local/v1/chat/completions"
    assert seen["json"]["model"] == "json-model-name"
    assert seen["headers"] == {"Content-Type": "application/json"}
