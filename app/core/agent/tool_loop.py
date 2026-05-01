from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.tools import ToolDefinition, ToolInvocationResult, ToolInvocationService
from app.schemas import AgentRunResult, Message, RuntimeOptions


@dataclass
class ToolLoopResult:
    result: AgentRunResult
    messages: list[dict[str, Any]]
    rounds: int


class ToolCallingAgentLoop:
    """OpenAI tool-calling loop backed by the unified tool service."""

    def __init__(self, tool_service: ToolInvocationService | None = None) -> None:
        self.tool_service = tool_service or ToolInvocationService()

    async def run(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None = None,
        emit_message_delta: bool = False,
    ) -> ToolLoopResult:
        llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
        llm_metadata = {
            "llm_configured": llm.configured,
            "model": llm.model,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "max_tokens": llm.max_tokens,
        }
        recorder.emit(
            "llm.started",
            {
                "model": llm_metadata["model"],
                "temperature": llm_metadata["temperature"],
                "top_p": llm_metadata["top_p"],
                "max_tokens": llm_metadata["max_tokens"],
                "configured": llm_metadata["llm_configured"],
            },
        )

        conversation = [message.model_dump() for message in messages]
        tools = self._openai_tools(agent_config)
        recorder.emit(
            "tools.available",
            {
                "tool_count": len(tools),
                "tools": [tool["function"]["name"] for tool in tools],
            },
        )

        if not tools:
            reply = await llm.complete(agent_config.prompts.system, messages)
            return self._result(
                agent_config,
                thread_id,
                reply,
                0,
                0,
                llm_metadata,
                recorder,
                conversation,
                emit_message_delta=emit_message_delta,
            )

        rounds = 0
        tool_call_count = 0
        final_response = LlmChatResponse()
        max_rounds = max(1, agent_config.runtime.max_tool_rounds)
        for round_index in range(max_rounds):
            rounds = round_index + 1
            final_response = await llm.complete_with_tools(agent_config.prompts.system, conversation, tools)
            assistant_message = self._assistant_message(final_response)
            conversation.append(assistant_message)
            if not final_response.tool_calls:
                reply = final_response.content
                return self._result(
                    agent_config,
                    thread_id,
                    reply,
                    rounds,
                    tool_call_count,
                    llm_metadata,
                    recorder,
                    conversation,
                    emit_message_delta=emit_message_delta,
                )

            recorder.emit(
                "tool.calls.started",
                {
                    "round": rounds,
                    "tool_calls": [
                        {"id": tool_call.id, "name": tool_call.name}
                        for tool_call in final_response.tool_calls
                    ],
                },
            )
            for tool_call in final_response.tool_calls:
                tool_call_count += 1
                tool_result = self._execute_tool_call(tool_call, thread_id, recorder)
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._tool_result_content(tool_result),
                    }
                )

        reply = final_response.content or f"工具调用轮数已达到上限（{max_rounds} 轮），请缩小问题范围或继续追问。"
        return self._result(
            agent_config,
            thread_id,
            reply,
            rounds,
            tool_call_count,
            llm_metadata,
            recorder,
            conversation,
            status="failed" if final_response.tool_calls else "completed",
            completed_event=not final_response.tool_calls,
            emit_message_delta=emit_message_delta,
        )

    def _openai_tools(self, agent_config: AgentConfig) -> list[dict[str, Any]]:
        allowed_names = self._allowed_tool_names(agent_config)
        definitions = self.tool_service.list_tools()
        if allowed_names:
            definitions = [tool for tool in definitions if tool.name in allowed_names]
        return [self._openai_tool(tool) for tool in definitions]

    @staticmethod
    def _allowed_tool_names(agent_config: AgentConfig) -> set[str]:
        return {name for name in [*agent_config.tools, *agent_config.skills] if name}

    @staticmethod
    def _openai_tool(tool: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.title or tool.name,
                "parameters": tool.input_schema or {"type": "object", "additionalProperties": True},
            },
        }

    @staticmethod
    def _assistant_message(response: LlmChatResponse) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            message["tool_calls"] = [tool_call.to_openai_payload() for tool_call in response.tool_calls]
        return message

    def _execute_tool_call(
        self,
        tool_call: LlmToolCall,
        thread_id: str,
        recorder: EventRecorder,
    ) -> ToolInvocationResult:
        recorder.emit("tool.started", {"tool_name": tool_call.name, "tool_call_id": tool_call.id})
        try:
            arguments = self._tool_arguments(tool_call.arguments)
            arguments.setdefault("_thread_id", thread_id)
            result = self.tool_service.call_tool(tool_call.name, arguments)
        except Exception as exc:
            result = ToolInvocationResult(
                content=[{"type": "text", "text": f"Tool {tool_call.name} failed: {exc}"}],
                structured_content={"tool_name": tool_call.name, "error": str(exc)},
                is_error=True,
            )
        recorder.emit(
            "tool.completed" if not result.is_error else "tool.failed",
            {
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "is_error": result.is_error,
                "structured_content": result.structured_content,
            },
        )
        return result

    @staticmethod
    def _tool_arguments(raw_arguments: str) -> dict[str, Any]:
        if not raw_arguments.strip():
            return {}
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def _tool_result_content(result: ToolInvocationResult) -> str:
        payload = result.to_mcp_result()
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _result(
        agent_config: AgentConfig,
        thread_id: str,
        reply: str,
        rounds: int,
        tool_call_count: int,
        llm_metadata: dict[str, Any],
        recorder: EventRecorder,
        messages: list[dict[str, Any]],
        status: str = "completed",
        completed_event: bool = True,
        emit_message_delta: bool = False,
    ) -> ToolLoopResult:
        if emit_message_delta and reply:
            recorder.emit("agent.message.delta", {"text": reply})
        recorder.emit("agent.message", {"text": reply})
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=thread_id,
            status="completed" if status == "completed" else "failed",
            reply=reply,
            metadata={
                "workflow": "agent_loop",
                **llm_metadata,
                "tool_rounds": rounds,
                "tool_call_count": tool_call_count,
            },
        )
        recorder.emit("run.completed" if completed_event else "run.failed", {"result": result.model_dump()})
        return ToolLoopResult(result=result, messages=messages, rounds=rounds)
