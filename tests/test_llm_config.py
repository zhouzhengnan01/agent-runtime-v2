from __future__ import annotations

import asyncio
from typing import Any

import httpx
from pytest import MonkeyPatch

from app.core.config.agent_config import AgentConfig, ModelConfig
from app.core.llm import OpenAICompatibleClient
from app.schemas import Message, RuntimeOptions


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


def test_request_timeout_can_be_configured_from_json_env_and_runtime_options(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT_SECONDS", raising=False)
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(
            model="json-model-name",
            base_url="http://llm.local/v1",
            request_timeout_seconds=9,
        ),
    )

    assert OpenAICompatibleClient(agent).request_timeout_seconds == 9

    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "7.5")
    assert OpenAICompatibleClient(agent).request_timeout_seconds == 7.5

    client = OpenAICompatibleClient(agent, runtime_options=RuntimeOptions(request_timeout_seconds=2))
    assert client.request_timeout_seconds == 2


def test_model_config_can_disable_tool_choice_auto(monkeypatch: MonkeyPatch) -> None:
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
            tool_choice="none",
        ),
    )
    client = OpenAICompatibleClient(agent)

    payload = client._chat_payload(
        "system",
        [Message(role="user", content="hi")],
        tools=[{"type": "function", "function": {"name": "demo", "parameters": {"type": "object"}}}],
    )

    assert "tools" not in payload
    assert "tool_choice" not in payload


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


def test_complete_with_tools_falls_back_when_auto_tool_choice_is_unsupported(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    payloads: list[dict[str, Any]] = []

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        del self, headers
        payloads.append(json)
        request = httpx.Request("POST", url)
        if len(payloads) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                    }
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "fallback ok"}, "finish_reason": "stop"}]},
            request=request,
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(
            model="json-model-name",
            base_url="http://llm.local/v1",
            api_key="json-key",
            tool_choice="auto",
        ),
    )

    response = asyncio.run(
        OpenAICompatibleClient(agent).complete_with_tools(
            "system",
            [Message(role="user", content="hi")],
            [{"type": "function", "function": {"name": "demo", "parameters": {"type": "object"}}}],
        )
    )

    assert response.content == "fallback ok"
    assert len(payloads) == 2
    assert payloads[0]["tool_choice"] == "auto"
    assert "tools" in payloads[0]
    assert "tool_choice" not in payloads[1]
    assert "tools" not in payloads[1]


def test_http_status_error_includes_upstream_response_body(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    async def fake_post(
        self: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        del self, json, headers
        return httpx.Response(
            400,
            json={"error": {"message": "unsupported parameter: response_format"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model=ModelConfig(model="json-model-name", base_url="http://llm.local/v1", api_key="json-key"),
    )

    try:
        asyncio.run(OpenAICompatibleClient(agent).complete("system", [Message(role="user", content="hi")]))
    except httpx.HTTPStatusError as exc:
        assert "response body:" in str(exc)
        assert "unsupported parameter" in str(exc)
    else:
        raise AssertionError("expected HTTPStatusError")
