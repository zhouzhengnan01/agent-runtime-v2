from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import AgentConfig
from app.schemas import Message, RuntimeOptions


@dataclass(frozen=True)
class LlmToolCall:
    """OpenAI-compatible tool call emitted by a chat model."""

    id: str
    name: str
    arguments: str

    def to_openai_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class LlmChatResponse:
    """Single chat completion response with optional tool calls."""

    content: str = ""
    tool_calls: list[LlmToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


ChatMessageInput = Message | dict[str, Any]


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client."""

    def __init__(
        self,
        agent_config: AgentConfig,
        model_override: str | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> None:
        model_config = agent_config.model
        runtime_options = runtime_options or RuntimeOptions()
        self.disabled = os.getenv("LLM_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}
        model_env = runtime_options.model_env or model_config.model_env
        base_url_env = runtime_options.base_url_env or model_config.base_url_env
        api_key_env = runtime_options.api_key_env or model_config.api_key_env
        self.base_url = (runtime_options.base_url or os.getenv(base_url_env) or model_config.base_url or "").rstrip("/")
        self.api_key = _first_defined(runtime_options.api_key, _env_value(api_key_env), model_config.api_key, "")
        self.model = model_override or runtime_options.model_name or os.getenv(model_env) or model_config.model or model_config.default_model
        self.temperature = runtime_options.temperature if runtime_options.temperature is not None else model_config.temperature
        self.top_p = runtime_options.top_p if runtime_options.top_p is not None else model_config.top_p
        self.max_tokens = runtime_options.max_tokens if runtime_options.max_tokens is not None else model_config.max_tokens
        raw_timeout = (
            runtime_options.request_timeout_seconds
            if runtime_options.request_timeout_seconds is not None
            else os.getenv(model_config.request_timeout_env) or model_config.request_timeout_seconds
        )
        self.request_timeout_seconds = _bounded_timeout(raw_timeout)
        selected_tools = any(name.strip() for name in runtime_options.selected_mcp_tools) or any(
            name.strip() for name in runtime_options.selected_skills
        )
        self.tool_choice = "auto" if selected_tools and model_config.tool_choice == "none" else model_config.tool_choice

    @property
    def configured(self) -> bool:
        return not self.disabled and bool(self.base_url and self.model)

    async def complete(self, system_prompt: str, messages: Sequence[ChatMessageInput]) -> str:
        if not self.configured:
            return self._not_configured_message()

        payload = self._chat_payload(system_prompt, messages)
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
            self._raise_for_status(response)
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    async def complete_with_tools(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessageInput],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        if not self.configured:
            return LlmChatResponse(content=self._not_configured_message())

        payload = self._chat_payload(system_prompt, messages, tools=tools)
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
            if tools and self._is_auto_tool_choice_unsupported(response):
                fallback_payload = self._chat_payload(system_prompt, messages)
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=fallback_payload,
                    headers=self._headers(),
                )
            self._raise_for_status(response)
            data = response.json()
        return self._parse_chat_response(data)

    def complete_sync(self, system_prompt: str, messages: Sequence[ChatMessageInput]) -> str:
        if not self.configured:
            return self._not_configured_message()

        payload = self._chat_payload(system_prompt, messages)
        with httpx.Client(timeout=self.request_timeout_seconds) as client:
            response = client.post(f"{self.base_url}/chat/completions", json=payload, headers=self._headers())
            self._raise_for_status(response)
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    async def stream_complete(self, system_prompt: str, messages: Sequence[ChatMessageInput]) -> AsyncIterator[str]:
        if not self.configured:
            yield self._not_configured_message()
            return

        payload = self._chat_payload(system_prompt, messages, stream=True)
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as response:
                await self._raise_stream_for_status(response)
                async for line in response.aiter_lines():
                    delta = self._delta_from_stream_line(line)
                    if delta == "[DONE]":
                        break
                    if delta:
                        yield delta

    def _chat_payload(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessageInput],
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}]
            + [self._message_payload(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if tools and self.tool_choice == "auto":
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if stream:
            payload["stream"] = True
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text.strip()
            if not body:
                raise
            detail = body[:2000]
            raise httpx.HTTPStatusError(
                f"{exc}; response body: {detail}",
                request=exc.request,
                response=exc.response,
            ) from exc

    @staticmethod
    async def _raise_stream_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raw_body = await response.aread()
            body = raw_body.decode(errors="replace").strip()
            if not body:
                raise
            detail = body[:2000]
            raise httpx.HTTPStatusError(
                f"{exc}; response body: {detail}",
                request=exc.request,
                response=exc.response,
            ) from exc

    @staticmethod
    def _is_auto_tool_choice_unsupported(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        body = response.text.lower()
        return (
            ("tool" in body and "unsupported" in body)
            or ("tool" in body and "not support" in body)
            or ("tool" in body and "not supported" in body)
            or ("tool" in body and "invalid" in body)
            or ("tool" in body and "extra_forbidden" in body)
            or "enable-auto-tool-choice" in body
            or "tool-call-parser" in body
        )

    def _not_configured_message(self) -> str:
        if self.disabled:
            return "v2 无状态 Agent 已收到请求。当前 LLM_DISABLED 已启用，因此返回本地占位响应。"
        return "v2 无状态 Agent 已收到请求。当前 agent JSON、运行时参数或环境变量未配置 LLM base_url/model，因此返回本地占位响应。"

    @staticmethod
    def _message_payload(message: ChatMessageInput) -> dict[str, Any]:
        if isinstance(message, Message):
            return message.model_dump()
        return dict(message)

    @classmethod
    def _parse_chat_response(cls, data: dict[str, Any]) -> LlmChatResponse:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return LlmChatResponse()
        choice = choices[0]
        raw_message = choice.get("message")
        message = raw_message if isinstance(raw_message, dict) else {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")
        raw_usage = data.get("usage")
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        return LlmChatResponse(
            content=content if isinstance(content, str) else "",
            tool_calls=cls._parse_tool_calls(message),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=usage,
        )

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[LlmToolCall]:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            return []
        calls: list[LlmToolCall] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            raw_function = raw_call.get("function")
            function = raw_function if isinstance(raw_function, dict) else {}
            raw_name = function.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                arguments = raw_arguments
            elif raw_arguments is None:
                arguments = "{}"
            else:
                arguments = json.dumps(raw_arguments, ensure_ascii=False)
            raw_id = raw_call.get("id")
            call_id = raw_id if isinstance(raw_id, str) and raw_id else f"call_{index}"
            calls.append(LlmToolCall(id=call_id, name=raw_name, arguments=arguments))
        return calls

    @staticmethod
    def _delta_from_stream_line(line: str) -> str | None:
        if not line.startswith("data:"):
            return None

        raw = line.removeprefix("data:").strip()
        if not raw:
            return None
        if raw == "[DONE]":
            return raw

        try:
            packet: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            return None

        choices = packet.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        choice = choices[0]
        if not isinstance(choice, dict):
            return None

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        return None


def _bounded_timeout(value: str | int | float | None) -> float:
    if value is None:
        return 120.0
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return 120.0
    return max(1.0, min(3600.0, timeout))


def _env_value(name: str | None) -> str | None:
    if not name:
        return None
    return os.environ.get(name) if name in os.environ else None


def _first_defined(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    return ""
