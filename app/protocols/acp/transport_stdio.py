from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from acp import Agent, Client, PROTOCOL_VERSION, run_agent
from acp import helpers as acp_helpers
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    InitializeResponse,
    McpServerStdio,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SseMcpServer,
    TextContentBlock,
)

from app.core.agent import AgentRuntime
from app.core.config import AgentConfigLoader
from app.schemas import ChatEvent, ChatRequest, Message, RuntimeOptions


PromptBlock = TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock


@dataclass
class AcpStdioSession:
    session_id: str
    thread_id: str
    agent_name: str
    cwd: str


class JetLinksAcpStdioAgent:
    """Official ACP stdio agent wrapper around the internal AgentRuntime."""

    def __init__(
        self,
        agent_name: str = "default",
        loader: AgentConfigLoader | None = None,
        runtime: AgentRuntime | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.loader = loader or AgentConfigLoader()
        self.runtime = runtime or AgentRuntime()
        self.sessions: dict[str, AcpStdioSession] = {}
        self.client: Client | None = None
        self.loader.load(agent_name)

    def on_connect(self, conn: Client) -> None:
        self.client = conn

    async def initialize(
        self,
        protocol_version: int,
        *_: Any,
        **__: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_info=Implementation(
                name="jetlinks-agent-runtime-v2",
                title="JetLinks Agent Runtime v2",
                version="0.1.0",
            ),
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(
                    image=True,
                    embedded_context=True,
                ),
                session_capabilities=SessionCapabilities(),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **_: Any,
    ) -> NewSessionResponse:
        del mcp_servers
        session_id = f"acp-session-{uuid4().hex[:12]}"
        self.sessions[session_id] = AcpStdioSession(
            session_id=session_id,
            thread_id=f"acp-{uuid4().hex[:12]}",
            agent_name=self.agent_name,
            cwd=cwd,
        )
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        prompt: list[PromptBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> PromptResponse:
        session = self.sessions.get(session_id)
        if session is None:
            session = AcpStdioSession(
                session_id=session_id,
                thread_id=f"acp-{uuid4().hex[:12]}",
                agent_name=self.agent_name,
                cwd=".",
            )
            self.sessions[session_id] = session

        text = _prompt_text(prompt)
        request = ChatRequest(
            messages=[Message(role="user", content=text)],
            runtime_options=RuntimeOptions(
                thread_id=session.thread_id,
                workflow=_workflow_from_kwargs(kwargs),
                selected_skills=_string_list_from_kwargs(kwargs, "selectedSkills", "selected_skills"),
                selected_mcp_tools=_string_list_from_kwargs(kwargs, "selectedMcpTools", "selected_mcp_tools"),
            ),
        )
        agent_config = self.loader.load(session.agent_name)
        result_status = "completed"
        async for event in self.runtime.iter_events(agent_config, request):
            await self._send_runtime_event(session.session_id, event, message_id)
            if event.type == "run.failed":
                result_status = "failed"

        return PromptResponse(stop_reason="end_turn" if result_status == "completed" else "refusal")

    async def cancel(self, session_id: str, **_: Any) -> None:
        del session_id

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"method": method, "params": params, "handled": False}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        del method, params

    async def _send_runtime_event(self, session_id: str, event: ChatEvent, message_id: str | None) -> None:
        if self.client is None:
            return
        for update in _event_to_sdk_updates(event, message_id):
            await self.client.session_update(session_id=session_id, update=update)


async def run_acp_stdio_agent(agent_name: str = "default") -> None:
    await run_agent(cast(Agent, JetLinksAcpStdioAgent(agent_name=agent_name)))


def _prompt_text(prompt: list[PromptBlock]) -> str:
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif isinstance(block, ImageContentBlock):
            parts.append("[image]")
        elif isinstance(block, AudioContentBlock):
            parts.append("[audio]")
        elif isinstance(block, ResourceContentBlock | EmbeddedResourceContentBlock):
            parts.append("[resource]")
    return "\n".join(part.strip() for part in parts if part.strip()) or " "


def _workflow_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    runtime_options = _params(kwargs.get("runtimeOptions") or kwargs.get("runtime_options"))
    jetlinks_meta = _params(kwargs.get("jetlinks"))
    return _string(
        runtime_options.get("workflow")
        or kwargs.get("workflow")
        or jetlinks_meta.get("workflow")
    )


def _string_list_from_kwargs(kwargs: dict[str, Any], camel_name: str, snake_name: str) -> list[str]:
    runtime_options = _params(kwargs.get("runtimeOptions") or kwargs.get("runtime_options"))
    jetlinks_meta = _params(kwargs.get("jetlinks"))
    raw_value = (
        runtime_options.get(camel_name)
        or runtime_options.get(snake_name)
        or kwargs.get(camel_name)
        or kwargs.get(snake_name)
        or jetlinks_meta.get(camel_name)
        or jetlinks_meta.get(snake_name)
    )
    if not isinstance(raw_value, list):
        return []
    return [item.strip() for item in raw_value if isinstance(item, str) and item.strip()]


def _event_to_sdk_updates(event: ChatEvent, message_id: str | None) -> list[Any]:
    data = event.data
    if event.type == "agent.message.delta":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        return [acp_helpers.update_agent_message_text(text)]
    if event.type == "agent.message":
        text = _string(data.get("text")) or _string(data.get("message")) or ""
        update = acp_helpers.update_agent_message_text(text)
        update.message_id = message_id
        return [update]
    if event.type in {"skill.selected", "skill.started"}:
        skill_name = _string(data.get("skill_name")) or _string(_params(data.get("skill")).get("name")) or "runtime"
        return [
            acp_helpers.start_tool_call(
                _tool_call_id(skill_name),
                skill_name,
                kind="other",
                status="in_progress",
            )
        ]
    if event.type == "skill.completed":
        skill_name = _string(data.get("skill_name")) or "runtime"
        return [
            acp_helpers.update_tool_call(
                _tool_call_id(skill_name),
                status="completed",
            )
        ]
    if event.type == "run.failed":
        return [
            acp_helpers.start_tool_call(
                "runtime-run",
                "runtime",
                kind="other",
                status="failed",
                raw_output=data,
            )
        ]
    return []


def _tool_call_id(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "._-" else "-" for char in value)
    return f"runtime-{normalized or 'tool'}"


def _params(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
