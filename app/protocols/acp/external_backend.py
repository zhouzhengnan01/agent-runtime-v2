from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, cast

from acp import Client, PROTOCOL_VERSION, spawn_agent_process
from acp import helpers as acp_helpers
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    NewSessionResponse,
    PromptResponse,
    ReadTextFileResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfoUpdate,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from pydantic import BaseModel

from app.core.config.agent_config import AgentBackendConfig


AcpUpdateSender = Any
PromptBlock = TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
SessionUpdate = (
    UserMessageChunk
    | AgentMessageChunk
    | AgentThoughtChunk
    | ToolCallStart
    | ToolCallProgress
    | AgentPlanUpdate
    | AvailableCommandsUpdate
    | CurrentModeUpdate
    | ConfigOptionUpdate
    | SessionInfoUpdate
    | UsageUpdate
)


class ExternalAcpClient:
    """Client callbacks invoked by an external ACP stdio agent."""

    def __init__(self) -> None:
        self.frontend_session_id: str | None = None
        self.send_update: AcpUpdateSender | None = None

    def on_connect(self, conn: Any) -> None:
        del conn

    async def session_update(self, session_id: str, update: SessionUpdate, **_: Any) -> None:
        del session_id
        if self.frontend_session_id is None or self.send_update is None:
            return
        await self.send_update(self.frontend_session_id, _model_payload(update))

    async def request_permission(self, *_: Any, **__: Any) -> RequestPermissionResponse:
        raise PermissionError("External ACP permission requests are not supported yet.")

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **_: Any,
    ) -> ReadTextFileResponse:
        del path, session_id, limit, line
        raise FileNotFoundError("External ACP file reads are not supported yet.")

    async def write_text_file(self, content: str, path: str, session_id: str, **_: Any) -> None:
        del content, path, session_id
        return None

    async def create_terminal(self, *_: Any, **__: Any) -> CreateTerminalResponse:
        raise RuntimeError("External ACP terminal creation is not supported yet.")

    async def terminal_output(self, *_: Any, **__: Any) -> TerminalOutputResponse:
        raise RuntimeError("External ACP terminal output is not supported yet.")

    async def release_terminal(self, *_: Any, **__: Any) -> None:
        return None

    async def wait_for_terminal_exit(self, *_: Any, **__: Any) -> None:
        return None

    async def kill_terminal(self, *_: Any, **__: Any) -> None:
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"method": method, "params": params, "handled": False}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params


class ExternalAcpSession:
    def __init__(
        self,
        client: ExternalAcpClient,
        connection: Any,
        manager: AbstractAsyncContextManager[Any],
        backend_session_id: str,
    ) -> None:
        self.client = client
        self.connection = connection
        self.manager = manager
        self.backend_session_id = backend_session_id

    @classmethod
    async def create(cls, config: AgentBackendConfig, cwd: str) -> ExternalAcpSession:
        if config.command is None:
            raise ValueError("ACP stdio backend requires backend.command")
        client = ExternalAcpClient()
        manager = spawn_agent_process(
            cast(Client, client),
            config.command,
            *config.args,
            env=config.env or None,
            cwd=Path(config.cwd) if config.cwd else None,
        )
        connection, _process = await manager.__aenter__()
        try:
            await connection.initialize(protocol_version=PROTOCOL_VERSION)
            response: NewSessionResponse = await connection.new_session(cwd=cwd, mcp_servers=[])
        except Exception:
            await manager.__aexit__(None, None, None)
            raise
        return cls(
            client=client,
            connection=connection,
            manager=manager,
            backend_session_id=response.session_id,
        )

    async def prompt(self, frontend_session_id: str, prompt: list[PromptBlock], send_update: AcpUpdateSender) -> PromptResponse:
        self.client.frontend_session_id = frontend_session_id
        self.client.send_update = send_update
        response = cast(PromptResponse, await self.connection.prompt(prompt=prompt, session_id=self.backend_session_id))
        await asyncio.sleep(0.02)
        return response

    async def close(self) -> None:
        await self.manager.__aexit__(None, None, None)


def prompt_blocks_from_params(params: dict[str, Any]) -> list[PromptBlock]:
    prompt = params.get("prompt")
    if isinstance(prompt, list):
        blocks: list[PromptBlock] = []
        for item in prompt:
            if isinstance(item, str):
                blocks.append(acp_helpers.text_block(item))
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                blocks.append(acp_helpers.text_block(text))
        if blocks:
            return blocks

    raw_messages = params.get("messages")
    if isinstance(raw_messages, list):
        text = "\n".join(
            str(item.get("content", "")).strip()
            for item in raw_messages
            if isinstance(item, dict) and item.get("role") == "user" and str(item.get("content", "")).strip()
        )
        if text:
            return [acp_helpers.text_block(text)]

    if isinstance(prompt, str) and prompt.strip():
        return [acp_helpers.text_block(prompt.strip())]
    raise ValueError("prompt text or messages are required")


def prompt_response_payload(response: PromptResponse) -> dict[str, Any]:
    return _model_payload(response)


def _model_payload(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)
