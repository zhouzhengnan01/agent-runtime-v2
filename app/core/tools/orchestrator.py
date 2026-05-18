from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmToolCall
from app.core.tools.invocation import ToolInvocationService
from app.core.tools.schemas import ToolInvocationResult
from app.schemas import RuntimeOptions


@dataclass(frozen=True)
class ToolExecutionOutcome:
    result: ToolInvocationResult
    arguments: dict[str, Any]
    duration_ms: float


class ToolOrchestrator:
    """Runtime-level wrapper around tool invocation.

    This keeps the public event contract identical to the previous direct
    `ToolInvocationService.call_tool` path while centralizing argument parsing,
    scoped runtime arguments, error mapping, and audit payload shaping.
    """

    def __init__(self, tool_service: ToolInvocationService) -> None:
        self.tool_service = tool_service

    def execute(
        self,
        tool_call: LlmToolCall,
        *,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None = None,
    ) -> ToolExecutionOutcome:
        started_at = time.perf_counter()
        recorder.emit("tool.started", {"tool_name": tool_call.name, "tool_call_id": tool_call.id})
        arguments: dict[str, Any] = {}
        try:
            arguments = self.tool_arguments(tool_call.arguments)
            self._apply_scoped_arguments(arguments, agent_config, thread_id, runtime_options)
            result = self.tool_service.call_tool(tool_call.name, arguments)
        except Exception as exc:
            error_code = self.tool_error_code(exc)
            result = ToolInvocationResult(
                content=[{"type": "text", "text": f"Tool {tool_call.name} failed: {exc}"}],
                structured_content={
                    "tool_name": tool_call.name,
                    "error": str(exc),
                    "error_code": error_code,
                    "recoverable": error_code in {"TOOL_ARGUMENTS_INVALID", "TOOL_NOT_FOUND", "TOOL_EXECUTION_FAILED"},
                },
                is_error=True,
            )

        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        recorder.emit(
            "tool.completed" if not result.is_error else "tool.failed",
            {
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "is_error": result.is_error,
                "duration_ms": duration_ms,
                "arguments": self.observable_arguments(arguments),
                "error_code": result.structured_content.get("error_code") if result.is_error else None,
                "structured_content": result.structured_content,
            },
        )
        return ToolExecutionOutcome(result=result, arguments=arguments, duration_ms=duration_ms)

    @staticmethod
    def _apply_scoped_arguments(
        arguments: dict[str, Any],
        agent_config: AgentConfig,
        thread_id: str,
        runtime_options: RuntimeOptions | None,
    ) -> None:
        arguments["_thread_id"] = thread_id
        arguments["_agent_name"] = agent_config.name
        arguments["_memory_scope"] = agent_config.memory.scope
        arguments["_markdown_writable_scopes"] = agent_config.memory.markdown_writable_scopes
        arguments["_markdown_max_chars"] = agent_config.memory.markdown_max_chars
        if runtime_options is not None:
            arguments["_user_id"] = runtime_options.user_id
            arguments["_project_id"] = runtime_options.project_id
            session_cwd = runtime_options.config_options.get("session_cwd")
            if isinstance(session_cwd, str) and session_cwd.strip():
                arguments["_session_cwd"] = session_cwd
            runtime_mcp_tools = runtime_options.config_options.get("runtime_mcp_tools")
            if isinstance(runtime_mcp_tools, list):
                arguments["_runtime_mcp_tools"] = runtime_mcp_tools

    @staticmethod
    def tool_arguments(raw_arguments: str) -> dict[str, Any]:
        if not raw_arguments.strip():
            return {}
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def observable_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        visible = {key: value for key, value in arguments.items() if not key.startswith("_")}
        rendered = json.dumps(visible, ensure_ascii=False, default=str)
        if len(rendered) <= 4000:
            return visible
        return {
            "_truncated": True,
            "json_chars": len(rendered),
            "preview": rendered[:3800],
        }

    @staticmethod
    def tool_error_code(exc: Exception) -> str:
        if isinstance(exc, json.JSONDecodeError):
            return "TOOL_ARGUMENTS_INVALID"
        if isinstance(exc, KeyError):
            return "TOOL_NOT_FOUND"
        message = str(exc).lower()
        if "disabled" in message:
            return "TOOL_DISABLED"
        if "arguments" in message and "json" in message:
            return "TOOL_ARGUMENTS_INVALID"
        return "TOOL_EXECUTION_FAILED"
