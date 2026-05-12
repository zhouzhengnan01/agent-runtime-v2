from __future__ import annotations

import asyncio
import os
import signal
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from acp import Client, PROTOCOL_VERSION, spawn_agent_process
from acp import helpers as acp_helpers
from acp.schema import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AudioContentBlock,
    AllowedOutcome,
    AvailableCommandsUpdate,
    ConfigOptionUpdate,
    CreateTerminalResponse,
    CurrentModeUpdate,
    DeniedOutcome,
    EmbeddedResourceContentBlock,
    EnvVariable,
    ImageContentBlock,
    KillTerminalResponse,
    NewSessionResponse,
    PromptResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    ResourceContentBlock,
    SessionInfoUpdate,
    TerminalExitStatus,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from pydantic import BaseModel

from app.core.artifacts import ArtifactStore, ThreadPaths
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


@dataclass
class ExternalTerminal:
    process: asyncio.subprocess.Process
    output_limit: int
    output: bytearray = field(default_factory=bytearray)
    truncated: bool = False
    reader_task: asyncio.Task[None] | None = None

    def append_output(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.output.extend(chunk)
        if len(self.output) <= self.output_limit:
            return
        self.truncated = True
        if self.output_limit <= 0:
            self.output.clear()
            return
        del self.output[: len(self.output) - self.output_limit]


class ExternalAcpClient:
    """Client callbacks invoked by an external ACP stdio agent."""

    def __init__(self, artifact_store: ArtifactStore, thread_id: str) -> None:
        self.paths = artifact_store.prepare_thread(thread_id)
        self.frontend_session_id: str | None = None
        self.send_update: AcpUpdateSender | None = None
        self.yolo_mode: bool = False
        self.terminals: dict[str, ExternalTerminal] = {}

    def on_connect(self, conn: Any) -> None:
        del conn

    async def session_update(self, session_id: str, update: SessionUpdate, **_: Any) -> None:
        del session_id
        if self.frontend_session_id is None or self.send_update is None:
            return
        await self.send_update(self.frontend_session_id, _model_payload(update))

    async def request_permission(
        self, options: list[Any] | None = None, *_: Any, **__: Any
    ) -> RequestPermissionResponse:
        if self.yolo_mode:
            option_id = self._permission_option_id(options)
            return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", optionId=option_id))
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **_: Any,
    ) -> ReadTextFileResponse:
        del session_id
        target = _resolve_read_path(self.paths, path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, (line or 1) - 1)
        selected = lines[start : start + limit] if limit is not None and limit > 0 else lines[start:]
        return ReadTextFileResponse(content="\n".join(selected)[:200_000])

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **_: Any,
    ) -> WriteTextFileResponse:
        del session_id
        target = _resolve_workspace_path(self.paths, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return WriteTextFileResponse()

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **_: Any,
    ) -> CreateTerminalResponse:
        del session_id
        terminal_cwd = _resolve_terminal_cwd(self.paths, cwd)
        terminal_cwd.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        for item in env or []:
            environment[item.name] = item.value
        process = await asyncio.create_subprocess_exec(
            command,
            *(args or []),
            cwd=terminal_cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        terminal_id = f"terminal-{uuid4().hex[:12]}"
        terminal = ExternalTerminal(
            process=process,
            output_limit=max(0, output_byte_limit if output_byte_limit is not None else 200_000),
        )
        terminal.reader_task = asyncio.create_task(_capture_terminal_output(terminal))
        self.terminals[terminal_id] = terminal
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(self, session_id: str, terminal_id: str, **_: Any) -> TerminalOutputResponse:
        del session_id
        terminal = self._terminal(terminal_id)
        return TerminalOutputResponse(
            output=_decode_terminal_output(terminal.output),
            truncated=terminal.truncated,
            exit_status=_terminal_exit_status(terminal.process.returncode),
        )

    async def release_terminal(self, session_id: str, terminal_id: str, **_: Any) -> ReleaseTerminalResponse:
        del session_id
        terminal = self.terminals.pop(terminal_id, None)
        if terminal is not None:
            await _terminate_terminal(terminal)
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **_: Any
    ) -> WaitForTerminalExitResponse:
        del session_id
        terminal = self._terminal(terminal_id)
        returncode = await terminal.process.wait()
        if terminal.reader_task is not None:
            with suppress(asyncio.CancelledError):
                await terminal.reader_task
        if returncode >= 0:
            return WaitForTerminalExitResponse(exit_code=returncode)
        return WaitForTerminalExitResponse(signal=_signal_name(-returncode))

    async def kill_terminal(self, session_id: str, terminal_id: str, **_: Any) -> KillTerminalResponse:
        del session_id
        terminal = self._terminal(terminal_id)
        if terminal.process.returncode is None:
            terminal.process.kill()
            await terminal.process.wait()
        if terminal.reader_task is not None:
            with suppress(asyncio.CancelledError):
                await terminal.reader_task
        return KillTerminalResponse()

    def _terminal(self, terminal_id: str) -> ExternalTerminal:
        terminal = self.terminals.get(terminal_id)
        if terminal is None:
            raise ValueError(f"Unknown external ACP terminal: {terminal_id}")
        return terminal

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"method": method, "params": params, "handled": False}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    @staticmethod
    def _permission_option_id(options: list[Any] | None) -> str:
        if not isinstance(options, list) or not options:
            return "allow"
        fallback = "allow"
        for item in options:
            if not isinstance(item, dict):
                continue
            option_id = item.get("optionId") or item.get("option_id") or item.get("id") or item.get("name")
            if not isinstance(option_id, str) or not option_id.strip():
                continue
            if fallback == "allow":
                fallback = option_id
            kind = str(item.get("kind") or "").lower()
            name = str(item.get("name") or "").lower()
            if kind in {"allow_once", "allow_always"} or "allow" in name or "approve" in name:
                return option_id
        return fallback


class ExternalAcpSession:
    def __init__(
        self,
        client: ExternalAcpClient,
        connection: Any,
        manager: AbstractAsyncContextManager[Any],
        process: asyncio.subprocess.Process,
        backend_session_id: str,
    ) -> None:
        self.client = client
        self.connection = connection
        self.manager = manager
        self.process = process
        self.backend_session_id = backend_session_id

    @classmethod
    async def create(
        cls,
        config: AgentBackendConfig,
        cwd: str,
        *,
        artifact_store: ArtifactStore,
        thread_id: str,
    ) -> ExternalAcpSession:
        if config.command is None:
            raise ValueError("ACP stdio backend requires backend.command")
        client = ExternalAcpClient(artifact_store=artifact_store, thread_id=thread_id)
        manager = spawn_agent_process(
            cast(Client, client),
            config.command,
            *config.args,
            env=config.env or None,
            cwd=Path(config.cwd) if config.cwd else None,
        )
        connection, process = await manager.__aenter__()
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
            process=process,
            backend_session_id=response.session_id,
        )

    async def prompt(
        self,
        frontend_session_id: str,
        prompt: list[PromptBlock],
        send_update: AcpUpdateSender,
        workflow: str | None = None,
        yolo_mode: bool = False,
    ) -> PromptResponse:
        self.client.frontend_session_id = frontend_session_id
        self.client.send_update = send_update
        self.client.yolo_mode = yolo_mode
        prompt_metadata: dict[str, Any] = {}
        if workflow:
            prompt_metadata = {"workflow": workflow, "jetlinks": {"workflow": workflow}}
        response = cast(
            PromptResponse,
            await self.connection.prompt(
                prompt=prompt,
                session_id=self.backend_session_id,
                **prompt_metadata,
            ),
        )
        await asyncio.sleep(0.02)
        return response

    async def close(self) -> None:
        await self.manager.__aexit__(None, None, None)
        await _drain_process_pipes(self.process)

    async def cancel(self) -> None:
        await self.connection.cancel(session_id=self.backend_session_id)


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
            block = _prompt_block_from_dict(item)
            if block is not None:
                blocks.append(block)
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


def _prompt_block_from_dict(item: dict[str, Any]) -> PromptBlock | None:
    block_type = str(item.get("type") or "").strip()
    text = item.get("text") or item.get("content")
    if block_type == "text" or isinstance(text, str):
        return acp_helpers.text_block(text) if isinstance(text, str) else None
    if block_type == "image":
        data = _string(item.get("data"))
        mime_type = _string(item.get("mimeType") or item.get("mime_type")) or "image/*"
        return acp_helpers.image_block(data, mime_type, uri=_string(item.get("uri"))) if data else None
    if block_type == "audio":
        data = _string(item.get("data"))
        mime_type = _string(item.get("mimeType") or item.get("mime_type")) or "audio/*"
        return acp_helpers.audio_block(data, mime_type) if data else None
    if block_type == "resource_link":
        uri = _string(item.get("uri"))
        name = _string(item.get("name")) or "resource"
        if uri is None:
            return None
        return acp_helpers.resource_link_block(
            name,
            uri,
            mime_type=_string(item.get("mimeType") or item.get("mime_type")),
            size=item.get("size") if isinstance(item.get("size"), int) else None,
            description=_string(item.get("description")),
            title=_string(item.get("title")),
        )
    if block_type == "resource" and isinstance(item.get("resource"), dict):
        resource = item["resource"]
        uri = _string(resource.get("uri")) or "embedded-resource"
        resource_mime_type = _string(resource.get("mimeType") or resource.get("mime_type"))
        resource_text = _string(resource.get("text"))
        if resource_text is not None:
            return acp_helpers.resource_block(
                acp_helpers.embedded_text_resource(uri, resource_text, mime_type=resource_mime_type)
            )
        blob = _string(resource.get("blob"))
        if blob is not None:
            return acp_helpers.resource_block(acp_helpers.embedded_blob_resource(uri, blob, mime_type=resource_mime_type))
    return None


def prompt_response_payload(response: PromptResponse) -> dict[str, Any]:
    return _model_payload(response)


def _model_payload(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json", by_alias=True, exclude_none=True, exclude_unset=True)


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _resolve_read_path(paths: ThreadPaths, raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    for prefix, root in {
        "/mnt/user-data/workspace/": paths.workspace,
        "/mnt/user-data/uploads/": paths.uploads,
        "/mnt/user-data/outputs/": paths.outputs,
    }.items():
        if normalized.startswith(prefix):
            return _bounded_path(root, normalized[len(prefix) :])
    if normalized.startswith("/"):
        raise ValueError("External ACP file access is limited to /mnt/user-data virtual paths.")
    return _bounded_path(paths.workspace, normalized)


def _resolve_workspace_path(paths: ThreadPaths, raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/")
    prefix = "/mnt/user-data/workspace/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    elif normalized.startswith("/"):
        raise ValueError("External ACP writes are limited to /mnt/user-data/workspace.")
    return _bounded_path(paths.workspace, normalized)


def _resolve_terminal_cwd(paths: ThreadPaths, raw_cwd: str | None) -> Path:
    cwd = _string(raw_cwd)
    if cwd is None:
        return paths.workspace.resolve()
    return _resolve_workspace_path(paths, cwd)


def _bounded_path(root: Path, relative_path: str) -> Path:
    if not relative_path.strip():
        raise ValueError("path is required")
    candidate = (root / relative_path.lstrip("/")).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("External ACP path traversal blocked") from exc
    return candidate


async def _capture_terminal_output(terminal: ExternalTerminal) -> None:
    stdout = terminal.process.stdout
    if stdout is None:
        return
    while True:
        chunk = await stdout.read(4096)
        if not chunk:
            return
        terminal.append_output(chunk)


async def _terminate_terminal(terminal: ExternalTerminal) -> None:
    if terminal.process.returncode is None:
        terminal.process.terminate()
        try:
            await asyncio.wait_for(terminal.process.wait(), timeout=2)
        except TimeoutError:
            terminal.process.kill()
            await terminal.process.wait()
    if terminal.reader_task is not None:
        with suppress(asyncio.CancelledError):
            await terminal.reader_task


async def _drain_process_pipes(process: asyncio.subprocess.Process) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        with suppress(Exception):
            await asyncio.wait_for(stream.read(), timeout=0.5)
    transport = getattr(process, "_transport", None)
    if transport is not None:
        with suppress(Exception):
            transport.close()
        await asyncio.sleep(0)


def _decode_terminal_output(output: bytearray) -> str:
    return bytes(output).decode("utf-8", errors="replace")


def _terminal_exit_status(returncode: int | None) -> TerminalExitStatus | None:
    if returncode is None:
        return None
    if returncode >= 0:
        return TerminalExitStatus(exit_code=returncode)
    return TerminalExitStatus(signal=_signal_name(-returncode))


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"
