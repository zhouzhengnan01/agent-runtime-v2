from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.core.config import AgentConfig
from app.core.diagnostics import diagnostic_json, env_flag, env_int
from app.schemas import Message, RuntimeOptions


logger = logging.getLogger("uvicorn.error")
LLM_TRACE_PAYLOADS = env_flag("LLM_TRACE_PAYLOADS", "1")
LLM_TRACE_MAX_CHARS = env_int("LLM_TRACE_MAX_CHARS", 100)
LLM_REPLY_TRACE_MAX_CHARS = env_int("LLM_REPLY_TRACE_MAX_CHARS", 4000)
LLM_UPSTREAM_RETRIES = max(0, env_int("LLM_UPSTREAM_RETRIES", 2))
LLM_UPSTREAM_RETRY_DELAY_MS = max(0, env_int("LLM_UPSTREAM_RETRY_DELAY_MS", 300))
LLM_UPSTREAM_RETRY_STATUS_CODES = {500, 502, 503, 504}


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
        self.disabled = False
        self.base_url = (runtime_options.base_url or model_config.base_url or "").rstrip("/")
        self.api_key = _first_defined(runtime_options.api_key, model_config.api_key, "")
        self.model = model_override or runtime_options.model_name or model_config.model or model_config.default_model
        self.temperature = runtime_options.temperature if runtime_options.temperature is not None else model_config.temperature
        self.top_p = runtime_options.top_p if runtime_options.top_p is not None else model_config.top_p
        self.max_tokens = runtime_options.max_tokens if runtime_options.max_tokens is not None else model_config.max_tokens
        self.request_timeout_seconds: float | None = None
        selected_tools = any(name.strip() for name in runtime_options.selected_mcp_tools) or any(
            name.strip() for name in runtime_options.selected_skills
        )
        self.tool_choice = "auto" if selected_tools and model_config.tool_choice == "none" else model_config.tool_choice
        logger.info(
            "\n===== 模型客户端配置 | llm client configured =====\n"
            "模型: %s\n"
            "Base URL: %s\n"
            "请求地址: %s\n"
            "API Key 是否配置: %s\n"
            "API Key 长度: %s\n"
            "Temperature: %s\n"
            "Max Tokens: %s\n"
            "工具选择: %s\n"
            "应用模板: %s\n"
            "已选技能: %s\n"
            "===== 模型客户端配置结束 =====",
            self.model,
            self.base_url,
            self._chat_completions_url(),
            "是" if self.api_key else "否",
            len(self.api_key or ""),
            self.temperature,
            self.max_tokens,
            self.tool_choice,
            runtime_options.app_template_name or "",
            [name for name in runtime_options.selected_skills if name.strip()],
        )

    @property
    def configured(self) -> bool:
        return not self.disabled and bool(self.base_url and self.model)

    async def complete(self, system_prompt: str, messages: Sequence[ChatMessageInput]) -> str:
        if not self.configured:
            logger.info("llm client not configured operation=complete model=%s base_url=%s", self.model, self.base_url)
            return self._not_configured_message()

        payload = self._chat_payload(system_prompt, messages)
        self._log_request("complete", payload)
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await self._post_chat_completion(client, "complete", payload)
            self._raise_for_status(response)
            data = response.json()
        self._log_response_data("complete", data)
        choices = data.get("choices") or []
        if not choices:
            self._log_reply_content("complete", "")
            return ""
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        self._log_reply_content("complete", content)
        return content

    async def complete_with_tools(
        self,
        system_prompt: str,
        messages: Sequence[ChatMessageInput],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        if not self.configured:
            logger.info("llm client not configured operation=complete_with_tools model=%s base_url=%s", self.model, self.base_url)
            return LlmChatResponse(content=self._not_configured_message())

        payload = self._chat_payload(system_prompt, messages, tools=tools)
        self._log_request("complete_with_tools", payload)
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await self._post_chat_completion(client, "complete_with_tools", payload)
            if tools and self._is_auto_tool_choice_unsupported(response):
                fallback_payload = self._chat_payload(system_prompt, messages)
                logger.info(
                    "llm request fallback operation=complete_with_tools reason=auto_tool_choice_unsupported status_code=%s body=%s",
                    response.status_code,
                    response.text[:2000],
                )
                self._log_request("complete_with_tools.fallback", fallback_payload)
                response = await self._post_chat_completion(client, "complete_with_tools.fallback", fallback_payload)
            self._raise_for_status(response)
            data = response.json()
        self._log_response_data("complete_with_tools", data)
        parsed = self._parse_chat_response(data)
        self._log_reply_content("complete_with_tools", parsed.content)
        return parsed

    def complete_sync(self, system_prompt: str, messages: Sequence[ChatMessageInput]) -> str:
        if not self.configured:
            logger.info("llm client not configured operation=complete_sync model=%s base_url=%s", self.model, self.base_url)
            return self._not_configured_message()

        payload = self._chat_payload(system_prompt, messages)
        self._log_request("complete_sync", payload)
        with httpx.Client(timeout=self.request_timeout_seconds) as client:
            response = self._post_chat_completion_sync(client, "complete_sync", payload)
            self._raise_for_status(response)
            data = response.json()
        self._log_response_data("complete_sync", data)
        choices = data.get("choices") or []
        if not choices:
            self._log_reply_content("complete_sync", "")
            return ""
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        self._log_reply_content("complete_sync", content)
        return content

    async def stream_complete(self, system_prompt: str, messages: Sequence[ChatMessageInput]) -> AsyncIterator[str]:
        if not self.configured:
            logger.info("llm client not configured operation=stream_complete model=%s base_url=%s", self.model, self.base_url)
            yield self._not_configured_message()
            return

        payload = self._chat_payload(system_prompt, messages, stream=True)
        self._log_request("stream_complete", payload)
        started_at = time.perf_counter()
        chunk_count = 0
        content_chars = 0
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as response:
                self._log_http_response("stream_complete", response, started_at)
                await self._raise_stream_for_status(response)
                async for line in response.aiter_lines():
                    delta = self._delta_from_stream_line(line)
                    if delta == "[DONE]":
                        break
                    if delta:
                        chunk_count += 1
                        content_chars += len(delta)
                        if LLM_TRACE_PAYLOADS:
                            logger.info(
                                "llm stream delta operation=stream_complete chunk=%s chars=%s text=%s",
                                chunk_count,
                                len(delta),
                                diagnostic_json({"text": delta}, max_chars=LLM_TRACE_MAX_CHARS),
                            )
                        yield delta
        logger.info(
            "llm stream completed operation=stream_complete model=%s chunks=%s content_chars=%s duration_ms=%s",
            self.model,
            chunk_count,
            content_chars,
            round((time.perf_counter() - started_at) * 1000, 3),
        )

    def _log_request(self, operation: str, payload: dict[str, Any]) -> None:
        image_blocks = _payload_image_urls(payload)
        logger.info(
            "\n===== 模型请求 | llm request operation=%s =====\n"
            "操作: %s\n"
            "模型: %s\n"
            "Base URL: %s\n"
            "请求地址: %s\n"
            "超时(秒): %s\n"
            "API Key 是否配置: %s\n"
            "API Key 长度: %s\n"
            "消息数: %s\n"
            "工具数: %s\n"
            "图片块数: %s\n"
            "图片来源: %s\n"
            "是否流式: %s\n"
            "发送内容(最多 %s 字符):\n%s\n"
            "===== 模型请求结束 =====",
            operation,
            operation,
            self.model,
            self.base_url,
            self._chat_completions_url(),
            self.request_timeout_seconds,
            "是" if self.api_key else "否",
            len(self.api_key or ""),
            len(payload.get("messages")) if isinstance(payload.get("messages"), list) else 0,
            len(payload.get("tools")) if isinstance(payload.get("tools"), list) else 0,
            len(image_blocks),
            _image_url_summary(image_blocks),
            "是" if payload.get("stream") is True else "否",
            LLM_TRACE_MAX_CHARS,
            _pretty_json(diagnostic_json(payload, max_chars=LLM_TRACE_MAX_CHARS)) if LLM_TRACE_PAYLOADS else "<disabled>",
        )

    def _log_http_response(self, operation: str, response: httpx.Response, started_at: float) -> None:
        body = _response_json_or_text(response)
        logger.info(
            "\n===== 模型HTTP响应 | llm http response operation=%s =====\n"
            "操作: %s\n"
            "模型: %s\n"
            "状态码: %s\n"
            "耗时(ms): %s\n"
            "响应字符数: %s\n"
            "响应内容(最多 %s 字符):\n%s\n"
            "===== 模型HTTP响应结束 =====",
            operation,
            operation,
            self.model,
            response.status_code,
            round((time.perf_counter() - started_at) * 1000, 3),
            len(str(body)) if isinstance(body, str | dict | list) else 0,
            LLM_TRACE_MAX_CHARS,
            _pretty_json(diagnostic_json(body, max_chars=LLM_TRACE_MAX_CHARS)) if LLM_TRACE_PAYLOADS else "<disabled>",
        )

    def _log_response_data(self, operation: str, data: dict[str, Any]) -> None:
        response = self._parse_chat_response(data)
        logger.info(
            "\n===== 模型解析结果 | llm response data operation=%s =====\n"
            "操作: %s\n"
            "模型: %s\n"
            "结束原因: %s\n"
            "回复字符数: %s\n"
            "工具调用数: %s\n"
            "Token 用量:\n%s\n"
            "解析数据(最多 %s 字符):\n%s\n"
            "===== 模型解析结果结束 =====",
            operation,
            operation,
            self.model,
            response.finish_reason,
            len(response.content or ""),
            len(response.tool_calls),
            _pretty_json(diagnostic_json(response.usage, max_chars=LLM_TRACE_MAX_CHARS)),
            LLM_TRACE_MAX_CHARS,
            _pretty_json(diagnostic_json(data, max_chars=LLM_TRACE_MAX_CHARS)) if LLM_TRACE_PAYLOADS else "<disabled>",
        )

    async def _post_chat_completion(
        self,
        client: httpx.AsyncClient,
        operation: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        url = self._chat_completions_url()
        headers = self._headers()
        attempts = LLM_UPSTREAM_RETRIES + 1
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            try:
                response = await client.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt >= attempts:
                    raise
                self._log_retry(operation, attempt, attempts, reason=exc.__class__.__name__)
                await self._sleep_before_retry()
                continue
            self._log_http_response(operation, response, started_at)
            if not self._should_retry_response(response) or attempt >= attempts:
                return response
            self._log_retry(operation, attempt, attempts, status_code=response.status_code)
            await self._sleep_before_retry()
        raise RuntimeError("unreachable llm retry state")

    def _post_chat_completion_sync(
        self,
        client: httpx.Client,
        operation: str,
        payload: dict[str, Any],
    ) -> httpx.Response:
        url = self._chat_completions_url()
        headers = self._headers()
        attempts = LLM_UPSTREAM_RETRIES + 1
        for attempt in range(1, attempts + 1):
            started_at = time.perf_counter()
            try:
                response = client.post(url, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt >= attempts:
                    raise
                self._log_retry(operation, attempt, attempts, reason=exc.__class__.__name__)
                self._sleep_before_retry_sync()
                continue
            self._log_http_response(operation, response, started_at)
            if not self._should_retry_response(response) or attempt >= attempts:
                return response
            self._log_retry(operation, attempt, attempts, status_code=response.status_code)
            self._sleep_before_retry_sync()
        raise RuntimeError("unreachable llm retry state")

    @staticmethod
    def _should_retry_response(response: httpx.Response) -> bool:
        return response.status_code in LLM_UPSTREAM_RETRY_STATUS_CODES

    async def _sleep_before_retry(self) -> None:
        if LLM_UPSTREAM_RETRY_DELAY_MS > 0:
            await asyncio.sleep(LLM_UPSTREAM_RETRY_DELAY_MS / 1000)

    def _sleep_before_retry_sync(self) -> None:
        if LLM_UPSTREAM_RETRY_DELAY_MS > 0:
            time.sleep(LLM_UPSTREAM_RETRY_DELAY_MS / 1000)

    def _log_retry(
        self,
        operation: str,
        attempt: int,
        attempts: int,
        *,
        status_code: int | None = None,
        reason: str | None = None,
    ) -> None:
        logger.info(
            "llm upstream retry operation=%s model=%s attempt=%s/%s next_attempt=%s status_code=%s reason=%s delay_ms=%s",
            operation,
            self.model,
            attempt,
            attempts,
            attempt + 1,
            status_code,
            reason or "",
            LLM_UPSTREAM_RETRY_DELAY_MS,
        )

    def _log_reply_content(self, operation: str, content: str) -> None:
        logger.info(
            "\n===== 模型回复正文 | llm reply content operation=%s =====\n"
            "操作: %s\n"
            "模型: %s\n"
            "回复字符数: %s\n"
            "回复内容(最多 %s 字符):\n%s\n"
            "===== 模型回复正文结束 =====",
            operation,
            operation,
            self.model,
            len(content),
            LLM_REPLY_TRACE_MAX_CHARS,
            _truncate_log_text(content, LLM_REPLY_TRACE_MAX_CHARS),
        )

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

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions" if self.base_url else ""

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
        payload = dict(message)
        role = payload.get("role")
        content = payload.get("content")
        if role == "user":
            content_blocks = _multimodal_user_content(content, payload.get("_attachments"))
            if content_blocks is not None:
                payload["content"] = content_blocks
        payload.pop("_attachments", None)
        return payload

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


def _first_defined(*values: str | None) -> str:
    for value in values:
        if value is not None:
            return value
    return ""


def _response_json_or_text(response: httpx.Response) -> object:
    try:
        return response.json()
    except (ValueError, httpx.ResponseNotRead):
        pass
    try:
        return {"text": response.text}
    except httpx.ResponseNotRead:
        return {"streaming": True, "reason": "response body not read yet"}


def _pretty_json(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _truncate_log_text(value: str, max_chars: int) -> str:
    if max_chars < 1:
        return ""
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated chars={len(value) - max_chars}>"


def _payload_image_urls(payload: dict[str, Any]) -> list[str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    urls: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_url = block.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _image_url_summary(urls: list[str], *, limit: int = 3) -> str:
    if not urls:
        return "无"
    items: list[str] = []
    for url in urls[:limit]:
        if url.startswith("data:"):
            prefix = url.split(",", 1)[0]
            items.append(f"{prefix},...({len(url)} chars)")
        elif len(url) > 120:
            items.append(url[:117] + "...")
        else:
            items.append(url)
    if len(urls) > limit:
        items.append(f"+{len(urls) - limit}")
    return "; ".join(items)


def _multimodal_user_content(content: object, attachments: object) -> list[dict[str, Any]] | None:
    image_blocks = _attachment_image_blocks(attachments)
    if not image_blocks:
        return None
    blocks: list[dict[str, Any]] = []
    if isinstance(content, list):
        blocks.extend(dict(item) for item in content if isinstance(item, dict))
    else:
        text = str(content or "")
        if text:
            blocks.append({"type": "text", "text": text})
    blocks.extend(image_blocks)
    return blocks


def _attachment_image_blocks(attachments: object) -> list[dict[str, Any]]:
    if not isinstance(attachments, list):
        return []
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_attachment in attachments:
        if not isinstance(raw_attachment, dict):
            continue
        mime_type = _clean_mime_type(raw_attachment.get("mime_type") or raw_attachment.get("mimeType"))
        if not mime_type.startswith("image/"):
            continue
        image_url = _attachment_image_url(raw_attachment, mime_type)
        if not image_url or image_url in seen:
            continue
        seen.add(image_url)
        blocks.append({"type": "image_url", "image_url": {"url": image_url}})
    return blocks


def _attachment_image_url(attachment: dict[str, Any], mime_type: str) -> str:
    data_base64 = attachment.get("data_base64") or attachment.get("dataBase64")
    if isinstance(data_base64, str) and data_base64.strip():
        return _data_uri(mime_type, data_base64.strip())
    path = attachment.get("_local_path") or attachment.get("path")
    if not isinstance(path, str) or not path.strip():
        return ""
    clean_path = path.strip()
    if _is_http_url(clean_path):
        logger.warning("llm image attachment skipped because remote url was not materialized url=%s", clean_path[:500])
        return ""
    local_path = _local_attachment_path(clean_path)
    if local_path is None or not local_path.is_file():
        return ""
    max_bytes = 10 * 1024 * 1024
    file_size = local_path.stat().st_size
    if file_size > max_bytes:
        return ""
    raw = local_path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    local_mime = mime_type or _guess_mime_type(local_path)
    return _data_uri(local_mime, encoded)


def _data_uri(mime_type: str, encoded: str) -> str:
    if encoded.startswith("data:"):
        return encoded
    return f"data:{mime_type or 'application/octet-stream'};base64,{encoded}"


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _local_attachment_path(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        return candidate.resolve()
    except OSError:
        return None


def _clean_mime_type(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.split(";", 1)[0].strip().lower()


def _guess_mime_type(path: Path) -> str:
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"
