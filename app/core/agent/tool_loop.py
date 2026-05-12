from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any

from app.core.agent.context import ContextCompactionResult, ConversationContextManager
from app.core.agent.session import ChatHistoryMessage
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
        messages: Sequence[ChatHistoryMessage],
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
            "request_timeout_seconds": llm.request_timeout_seconds,
        }
        recorder.emit(
            "llm.started",
            {
                "model": llm_metadata["model"],
                "temperature": llm_metadata["temperature"],
                "top_p": llm_metadata["top_p"],
                "max_tokens": llm_metadata["max_tokens"],
                "request_timeout_seconds": llm_metadata["request_timeout_seconds"],
                "configured": llm_metadata["llm_configured"],
            },
        )

        conversation = [self._normalize_message(message) for message in messages]
        context_compactions = 0
        last_context_event: dict[str, Any] = {}
        system_prompt, memory_context_count = self._system_prompt(agent_config)
        if agent_config.memory.enabled:
            recorder.emit(
                "memory.context.loaded",
                {
                    "enabled": True,
                    "inject_context": agent_config.memory.inject_context,
                    "scope": agent_config.memory.scope,
                    "count": memory_context_count,
                },
            )
        tools = self._openai_tools(agent_config, runtime_options)
        if llm.tool_choice == "none":
            tools = []
        recorder.emit(
            "tools.available",
            {
                "tool_count": len(tools),
                "tools": [tool["function"]["name"] for tool in tools],
                "tool_choice": llm.tool_choice,
            },
        )

        if not tools:
            compaction = self._compact_context(agent_config, conversation)
            if compaction.compacted:
                context_compactions += 1
                last_context_event = compaction.event_payload()
                recorder.emit("context.compacted", last_context_event)
                conversation = compaction.messages
            request_started_at = time.perf_counter()
            recorder.emit(
                "llm.request.started",
                {
                    "round": 1,
                    "mode": "chat",
                    "message_count": len(conversation),
                    "tool_count": 0,
                    "tools": [],
                    "tool_choice": llm.tool_choice,
                },
            )
            reply = await llm.complete(system_prompt, conversation)
            reply, guard_metadata = self._guard_unverified_completion(reply, runtime_options)
            recorder.emit(
                "llm.request.completed",
                {
                    "round": 1,
                    "mode": "chat",
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000, 3),
                    "finish_reason": "stop",
                    "tool_call_count": 0,
                    "content_chars": len(reply),
                },
            )
            return self._result(
                agent_config,
                thread_id,
                reply,
                0,
                0,
                llm_metadata,
                recorder,
                conversation,
                context_compactions=context_compactions,
                last_context_event=last_context_event,
                emit_message_delta=emit_message_delta,
                run_id=recorder.run_id,
                mode=self._runtime_mode(runtime_options),
                extra_metadata=guard_metadata,
            )

        rounds = 0
        tool_call_count = 0
        required_inputs: list[dict[str, Any]] = []
        final_response = LlmChatResponse()
        max_rounds = self._max_tool_rounds(agent_config, runtime_options)
        for round_index in range(max_rounds):
            rounds = round_index + 1
            compaction = self._compact_context(agent_config, conversation)
            if compaction.compacted:
                context_compactions += 1
                last_context_event = compaction.event_payload()
                recorder.emit("context.compacted", {**last_context_event, "round": rounds})
                conversation = compaction.messages
            request_started_at = time.perf_counter()
            recorder.emit(
                "llm.request.started",
                {
                    "round": rounds,
                    "mode": "tool_calling",
                    "message_count": len(conversation),
                    "tool_count": len(tools),
                    "tools": self._openai_tool_names(tools),
                    "tool_choice": llm.tool_choice,
                },
            )
            final_response = await llm.complete_with_tools(system_prompt, conversation, tools)
            recorder.emit(
                "llm.request.completed",
                {
                    "round": rounds,
                    "mode": "tool_calling",
                    "duration_ms": round((time.perf_counter() - request_started_at) * 1000, 3),
                    "finish_reason": final_response.finish_reason or "",
                    "tool_call_count": len(final_response.tool_calls),
                    "tool_calls": [
                        {"id": tool_call.id, "name": tool_call.name}
                        for tool_call in final_response.tool_calls
                    ],
                    "content_chars": len(final_response.content or ""),
                    "usage": final_response.usage,
                },
            )
            assistant_message = self._assistant_message(final_response)
            conversation.append(assistant_message)
            if not final_response.tool_calls:
                reply, guard_metadata = self._guard_unverified_completion(
                    final_response.content,
                    runtime_options,
                    tool_count=len(tools),
                )
                return self._result(
                    agent_config,
                    thread_id,
                    reply,
                    rounds,
                    tool_call_count,
                    llm_metadata,
                    recorder,
                    conversation,
                    context_compactions=context_compactions,
                    last_context_event=last_context_event,
                    emit_message_delta=emit_message_delta,
                    run_id=recorder.run_id,
                    mode=self._runtime_mode(runtime_options),
                    extra_metadata=guard_metadata,
                    required_inputs=required_inputs,
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
                tool_result = self._execute_tool_call(
                    tool_call,
                    agent_config,
                    thread_id,
                    recorder,
                    runtime_options=runtime_options,
                )
                required_inputs.extend(self._required_inputs_from_tool_result(tool_result))
                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": self._tool_result_content(
                            tool_result,
                            max_chars=agent_config.runtime.max_tool_result_chars,
                        ),
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
            context_compactions=context_compactions,
            last_context_event=last_context_event,
            emit_message_delta=emit_message_delta,
            run_id=recorder.run_id,
            mode=self._runtime_mode(runtime_options),
            required_inputs=required_inputs,
        )

    @staticmethod
    def _normalize_message(message: ChatHistoryMessage) -> dict[str, Any]:
        if isinstance(message, Message):
            return message.model_dump(mode="python")
        return dict(message)

    @staticmethod
    def _compact_context(agent_config: AgentConfig, conversation: list[dict[str, Any]]) -> ContextCompactionResult:
        if not agent_config.runtime.context_compression_enabled:
            before_chars = len(json.dumps(conversation, ensure_ascii=False, default=str))
            return ContextCompactionResult(
                messages=conversation,
                compacted=False,
                before_chars=before_chars,
                after_chars=before_chars,
                original_message_count=len(conversation),
                compacted_message_count=len(conversation),
                summarized_message_count=0,
            )
        manager = ConversationContextManager(
            max_chars=agent_config.runtime.context_max_chars,
            keep_first_messages=agent_config.runtime.context_keep_first_messages,
            keep_last_messages=agent_config.runtime.context_keep_last_messages,
        )
        return manager.compact(conversation)

    def _openai_tools(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions | None = None,
    ) -> list[dict[str, Any]]:
        mode = self._runtime_mode(runtime_options)
        if mode == "plan":
            return []
        allowed_names = self._allowed_tool_names(agent_config)
        selected_skill_names = self._selected_skill_names(runtime_options)
        selected_mcp_tool_names = self._selected_mcp_tool_names(runtime_options)
        if not allowed_names and not selected_skill_names and not selected_mcp_tool_names:
            return []

        definitions = self.tool_service.list_tools()
        if not agent_config.memory.enabled:
            definitions = [
                tool for tool in definitions if tool.source.get("type") not in {"memory", "markdown_memory"}
            ]
        elif not agent_config.memory.markdown_enabled:
            definitions = [tool for tool in definitions if tool.source.get("type") != "markdown_memory"]
        definitions = [
            tool
            for tool in definitions
            if self._tool_selected_for_run(tool, allowed_names, selected_skill_names, selected_mcp_tool_names)
        ]
        definitions = [tool for tool in definitions if self._tool_allowed_in_mode(tool, mode)]
        return [self._openai_tool(tool) for tool in definitions]

    @staticmethod
    def _tool_selected_for_run(
        tool: ToolDefinition,
        allowed_names: set[str],
        selected_skill_names: set[str],
        selected_mcp_tool_names: set[str],
    ) -> bool:
        if tool.source.get("type") == "skill":
            if selected_skill_names:
                return tool.name in selected_skill_names
            return tool.name in allowed_names
        return (
            tool.name in allowed_names
            or (tool.name in selected_mcp_tool_names and ToolCallingAgentLoop._runtime_selectable_mcp_tool(tool))
        )

    @staticmethod
    def _allowed_tool_names(agent_config: AgentConfig) -> set[str]:
        return {name for name in [*agent_config.tools, *agent_config.skills] if name}

    @staticmethod
    def _selected_skill_names(runtime_options: RuntimeOptions | None) -> set[str]:
        if runtime_options is None:
            return set()
        return {name.strip() for name in runtime_options.selected_skills if name.strip()}

    @staticmethod
    def _selected_mcp_tool_names(runtime_options: RuntimeOptions | None) -> set[str]:
        if runtime_options is None:
            return set()
        return {name.strip() for name in runtime_options.selected_mcp_tools if name.strip()}

    @staticmethod
    def _runtime_selectable_mcp_tool(tool: ToolDefinition) -> bool:
        return tool.source.get("type") in {
            "manual",
            "memory",
            "markdown_memory",
            "local",
            "artifact_workspace",
            "delegate",
            "skill",
        }

    @staticmethod
    def _runtime_mode(runtime_options: RuntimeOptions | None) -> str:
        if runtime_options is None or runtime_options.mode is None:
            return "edit"
        return runtime_options.mode

    @staticmethod
    def _max_tool_rounds(agent_config: AgentConfig, runtime_options: RuntimeOptions | None) -> int:
        base = ToolCallingAgentLoop._base_tool_rounds(agent_config, runtime_options)
        mode = ToolCallingAgentLoop._runtime_mode(runtime_options)
        if mode == "plan":
            return 1
        if mode == "safe":
            return min(base, 2)
        if mode in {"autonomous", "yolo"}:
            return max(base, min(base * 2, 16))
        return base

    @staticmethod
    def _base_tool_rounds(agent_config: AgentConfig, runtime_options: RuntimeOptions | None) -> int:
        configured = max(1, agent_config.runtime.max_tool_rounds)
        if runtime_options is None:
            return configured
        raw_value = runtime_options.config_options.get("max_tool_rounds")
        try:
            requested = int(raw_value)
        except (TypeError, ValueError):
            return configured
        return min(max(1, requested), 32)

    @staticmethod
    def _tool_allowed_in_mode(tool: ToolDefinition, mode: str) -> bool:
        if mode in {"autonomous", "yolo"}:
            return True
        source_type = tool.source.get("type")
        operation = tool.source.get("operation")
        if mode == "safe":
            if source_type == "skill":
                return False
            if source_type == "local" and operation in {"write_file", "shell_command"}:
                return False
            return True
        if mode == "edit":
            return not (source_type == "local" and operation == "shell_command")
        return True

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
    def _openai_tool_names(tools: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for tool in tools:
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
        return names

    @staticmethod
    def _assistant_message(response: LlmChatResponse) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            message["tool_calls"] = [tool_call.to_openai_payload() for tool_call in response.tool_calls]
        return message

    def _execute_tool_call(
        self,
        tool_call: LlmToolCall,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None = None,
    ) -> ToolInvocationResult:
        started_at = time.perf_counter()
        recorder.emit("tool.started", {"tool_name": tool_call.name, "tool_call_id": tool_call.id})
        arguments: dict[str, Any] = {}
        try:
            arguments = self._tool_arguments(tool_call.arguments)
            arguments.setdefault("_thread_id", thread_id)
            arguments.setdefault("_agent_name", agent_config.name)
            arguments.setdefault("_memory_scope", agent_config.memory.scope)
            arguments.setdefault("_markdown_writable_scopes", agent_config.memory.markdown_writable_scopes)
            arguments.setdefault("_markdown_max_chars", agent_config.memory.markdown_max_chars)
            if runtime_options is not None:
                arguments.setdefault("_user_id", runtime_options.user_id)
                arguments.setdefault("_project_id", runtime_options.project_id)
            result = self.tool_service.call_tool(tool_call.name, arguments)
        except Exception as exc:
            error_code = self._tool_error_code(exc)
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
                "arguments": self._observable_arguments(arguments),
                "error_code": result.structured_content.get("error_code") if result.is_error else None,
                "structured_content": result.structured_content,
            },
        )
        return result

    @staticmethod
    def _required_inputs_from_tool_result(result: ToolInvocationResult) -> list[dict[str, Any]]:
        if result.structured_content.get("requires_input") is not True:
            return []
        raw_items = result.structured_content.get("required_inputs")
        if not isinstance(raw_items, list):
            return []
        return [dict(item) for item in raw_items if isinstance(item, dict)]

    @classmethod
    def _guard_unverified_completion(
        cls,
        reply: str,
        runtime_options: RuntimeOptions | None,
        *,
        tool_count: int = 0,
    ) -> tuple[str, dict[str, Any]]:
        mode = cls._runtime_mode(runtime_options)
        if mode != "yolo":
            return reply, {}
        selected_skills = cls._selected_skill_names(runtime_options)
        if not selected_skills:
            return reply, {}
        if tool_count <= 0:
            guarded = (
                "已选择 skills，但本轮没有可用工具可执行，不能进入大流程执行。\n"
                "请检查 app 配置中的 selected_skills 是否是本地已安装 skill 名称，而不是平台侧 ID。"
            )
            return guarded, {"unavailable_selected_skills_blocked": True, "selected_skills": sorted(selected_skills)}
        reply_lower = reply.lower()
        completion_markers = (
            "已生成",
            "已完成",
            "准备就绪",
            "通过内容校验",
            "benchmark 完成",
            "训练完成",
            "评估完成",
            "上线评审通过",
            "completed",
            "generated",
        )
        if not any(marker in reply_lower for marker in completion_markers):
            return reply, {}
        guarded = (
            "本次没有执行任何工具调用，不能声明大流程已完成。\n"
            "请确认已选择对应 app/skills，并提供数据集、GPU 连接、训练产物或评估结果等必要输入。"
        )
        return guarded, {"unverified_completion_blocked": True, "original_reply": reply[:2000]}

    @staticmethod
    def _tool_arguments(raw_arguments: str) -> dict[str, Any]:
        if not raw_arguments.strip():
            return {}
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def _observable_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
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
    def _tool_error_code(exc: Exception) -> str:
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

    def _system_prompt(self, agent_config: AgentConfig) -> tuple[str, int]:
        prompt = agent_config.prompts.system
        if not agent_config.memory.enabled or not agent_config.memory.inject_context:
            return prompt, 0
        try:
            memories = self.tool_service.memory_store.list(
                agent_config.name,
                limit=agent_config.memory.max_items,
                scope=agent_config.memory.scope,
            )
        except Exception:
            return prompt, 0
        if not memories:
            return prompt, 0
        lines = []
        for item in memories:
            tag_text = f" tags={','.join(item.tags)}" if item.tags else ""
            lines.append(f"- {item.text} (id={item.id}{tag_text})")
        memory_prompt = "\n".join(
            [
                "可用长期记忆如下。仅在与当前请求相关时使用；不要编造未列出的记忆。",
                *lines,
            ]
        )
        return f"{prompt}\n\n{memory_prompt}", len(memories)

    @classmethod
    def _tool_result_content(cls, result: ToolInvocationResult, *, max_chars: int) -> str:
        payload = result.to_mcp_result()
        rendered = json.dumps(payload, ensure_ascii=False)
        if len(rendered) <= max_chars:
            return rendered
        return cls._truncated_tool_result_content(result, original_json=rendered, max_chars=max_chars)

    @classmethod
    def _truncated_tool_result_content(
        cls,
        result: ToolInvocationResult,
        *,
        original_json: str,
        max_chars: int,
    ) -> str:
        source_text = cls._tool_result_text(result) or original_json
        marker = (
            "\n\n[tool result truncated before returning to model context; "
            f"original_json_chars={len(original_json)}, max_tool_result_chars={max_chars}. "
            "Use a narrower query, smaller file range, or follow-up tool call if more detail is needed.]"
        )
        clipped = source_text[: max(0, max_chars - len(marker) - 256)]
        while True:
            payload = {
                "content": [{"type": "text", "text": f"{clipped}{marker}"}],
                "structuredContent": {
                    "_truncated": True,
                    "original_json_chars": len(original_json),
                    "max_tool_result_chars": max_chars,
                },
                "isError": result.is_error,
            }
            rendered = json.dumps(payload, ensure_ascii=False)
            if len(rendered) <= max_chars or not clipped:
                return rendered
            overflow = len(rendered) - max_chars
            clipped = clipped[: max(0, len(clipped) - overflow - 16)]

    @staticmethod
    def _tool_result_text(result: ToolInvocationResult) -> str:
        texts: list[str] = []
        for item in result.content:
            raw_text = item.get("text")
            if isinstance(raw_text, str):
                texts.append(raw_text)
            else:
                texts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(texts).strip()

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
        context_compactions: int = 0,
        last_context_event: dict[str, Any] | None = None,
        emit_message_delta: bool = False,
        run_id: str = "",
        mode: str = "edit",
        extra_metadata: dict[str, Any] | None = None,
        required_inputs: list[dict[str, Any]] | None = None,
    ) -> ToolLoopResult:
        final_messages = list(messages)
        if reply and (
            not final_messages
            or final_messages[-1].get("role") != "assistant"
            or final_messages[-1].get("content") != reply
        ):
            final_messages.append({"role": "assistant", "content": reply})
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
                "run_id": run_id,
                **llm_metadata,
                "tool_rounds": rounds,
                "tool_call_count": tool_call_count,
                "mode": mode,
                "context_compaction_count": context_compactions,
                "last_context_compaction": last_context_event or {},
                **({"requires_input": True, "required_inputs": required_inputs} if required_inputs else {}),
                **(extra_metadata or {}),
            },
        )
        recorder.emit("run.completed" if completed_event else "run.failed", {"result": result.model_dump()})
        return ToolLoopResult(result=result, messages=final_messages, rounds=rounds)
