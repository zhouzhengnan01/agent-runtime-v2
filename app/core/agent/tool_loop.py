from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import Any

from app.core.agent.conditional_loop import conditional_loop_policy, latest_tool_message_text
from app.core.agent.context import ContextCompactionResult, ConversationContextManager
from app.core.agent.primary_skill_context import load_primary_skill_context
from app.core.agent.tool_loop_guard import (
    guard_unverified_completion,
    reply_with_required_inputs,
    runtime_mode,
    selected_skill_names,
)
from app.core.agent.tool_loop_prompt import build_system_prompt, prompt_with_runtime_mcp_discovery_failures
from app.core.agent.tool_exposure import select_tools_for_phase
from app.core.agent.session import ChatHistoryMessage
from app.core.agent.tool_loop_state import ToolLoopState
from app.core.agent.turn_policy import TurnPolicyInput, decide_turn_policy
from app.core.agent.turn_verifier import verify_turn_completion
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.tools import ToolDefinition, ToolInvocationResult, ToolInvocationService
from app.core.tools.orchestrator import ToolOrchestrator
from app.schemas import AgentRunResult, ArtifactRef, Message, RuntimeOptions


MAX_TOOL_ROUNDS_LIMIT = 1000


@dataclass
class ToolLoopResult:
    result: AgentRunResult
    messages: list[dict[str, Any]]
    rounds: int


class ToolCallingAgentLoop:
    """基于统一 ToolInvocationService 的工具调用主循环。

    这个类负责默认 agent 的核心执行过程：
    1. 组装 system prompt，包括主 skill、composite skill 和记忆上下文。
    2. 按当前 runtime 过滤出本轮真正可用的 tools。
    3. 调用模型决定是直接回答，还是发起 tool calls。
    4. 执行工具并把结果回填到 conversation，直到模型停止调用工具。
    5. 把最终回复、产物、缺失输入等信息汇总成 AgentRunResult。
    """

    def __init__(self, tool_service: ToolInvocationService | None = None) -> None:
        self.tool_service = tool_service or ToolInvocationService()
        self.tool_orchestrator = ToolOrchestrator(self.tool_service)
        self.artifact_store = self.tool_service.artifact_store

    async def run(
        self,
        agent_config: AgentConfig,
        messages: Sequence[ChatHistoryMessage],
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None = None,
        emit_message_delta: bool = False,
    ) -> ToolLoopResult:
        state = self._prepare_turn_state(agent_config, messages, recorder, runtime_options)
        if not state.tools:
            return await self._run_chat_only_turn(
                state,
                agent_config,
                thread_id,
                recorder,
                runtime_options,
                emit_message_delta=emit_message_delta,
            )
        return await self._run_tool_calling_turn(
            state,
            agent_config,
            thread_id,
            recorder,
            runtime_options,
            emit_message_delta=emit_message_delta,
        )

    def _prepare_turn_state(
        self,
        agent_config: AgentConfig,
        messages: Sequence[ChatHistoryMessage],
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
    ) -> ToolLoopState:
        thread_id = self._thread_id(runtime_options)
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
        conversation = ConversationContextManager.normalize_for_model_prompt(
            [self._normalize_message(message) for message in messages]
        )
        available_artifacts = self._thread_artifacts(thread_id)
        primary_skill_context = load_primary_skill_context(
            root_dir=self.tool_service.root_dir,
            artifact_store=self.artifact_store,
            runtime_options=runtime_options,
            thread_id=thread_id,
            available_artifacts=available_artifacts,
            current_artifacts=[],
            required_inputs=[],
        )
        verification = verify_turn_completion(
            runtime_options=runtime_options,
            artifacts=available_artifacts,
            required_inputs=[],
            latest_tool_results=[],
            tool_call_count=0,
            primary_skill_context=primary_skill_context,
        )
        turn_policy = decide_turn_policy(
            TurnPolicyInput(
                messages=conversation,
                available_artifacts=available_artifacts,
                current_phase="execute",
                tool_call_count=0,
                required_inputs=[],
                latest_tool_results=[],
                latest_assistant_reply="",
                verification=verification,
                primary_skill_context=primary_skill_context,
                runtime_options=runtime_options,
            )
        )
        base_prompt, memory_context_count = build_system_prompt(
            agent_config,
            runtime_options,
            tool_service=self.tool_service,
            primary_skill_context=primary_skill_context,
        )
        system_prompt = self._prompt_with_turn_policy(base_prompt, turn_policy.phase, turn_policy.reason)
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
        tool_definitions = self._tool_definitions_for_run(agent_config, runtime_options, recorder)
        system_prompt = prompt_with_runtime_mcp_discovery_failures(system_prompt, runtime_options)
        exposure = select_tools_for_phase(
            tool_definitions,
            phase=turn_policy.phase,
            primary_skill_context=primary_skill_context,
        )
        tools = [self._openai_tool(tool) for tool in exposure.visible_tools]
        if llm.tool_choice == "none":
            tools = []
        recorder.emit(
            "tools.available",
            {
                "tool_count": len(tools),
                "tools": [tool["function"]["name"] for tool in tools],
                "tool_choice": llm.tool_choice,
                "turn_phase": turn_policy.phase,
                "turn_policy_reason": turn_policy.reason,
                "tool_exposure_policy": exposure.metadata.get("policy"),
                "priority_tools": [tool.name for tool in exposure.priority_tools],
                "secondary_tools": [tool.name for tool in exposure.secondary_tools],
                "hidden_tools": [tool.name for tool in exposure.hidden_tools],
                "verification_verdict": verification.verdict,
                "verification_reason": verification.reason,
                **(primary_skill_context.to_metadata() if primary_skill_context is not None else {}),
            },
        )
        return ToolLoopState(
            llm=llm,
            llm_metadata=llm_metadata,
            conversation=conversation,
            system_prompt=system_prompt,
            memory_context_count=memory_context_count,
            tools=tools,
            tool_definitions=exposure.visible_tools,
            priority_tool_definitions=exposure.priority_tools,
            secondary_tool_definitions=exposure.secondary_tools,
            hidden_tool_definitions=exposure.hidden_tools,
            turn_phase=turn_policy.phase,
            turn_policy_reason=turn_policy.reason,
            primary_skill_context=primary_skill_context,
            available_artifacts=available_artifacts,
            verification_verdict=verification.verdict,
            verification_reason=verification.reason,
        )

    async def _run_chat_only_turn(
        self,
        state: ToolLoopState,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
        *,
        emit_message_delta: bool,
    ) -> ToolLoopResult:
        # 没有可用工具时，仍然沿用同一套事件和结果协议，只是退化成
        # 纯聊天补全，不进入工具调用循环。
        self._apply_context_compaction(state, agent_config, recorder, round_number=None)
        request_started_at = time.perf_counter()
        recorder.emit(
            "llm.request.started",
            {
                "round": 1,
                "mode": "chat",
                "message_count": len(state.conversation),
                "tool_count": 0,
                "tools": [],
                "tool_choice": state.llm.tool_choice,
            },
        )
        reply = await state.llm.complete(state.system_prompt, state.conversation)
        reply, guard_metadata = guard_unverified_completion(
            reply,
            runtime_options,
            available_tool_count=0,
            executed_tool_count=0,
            required_inputs=[],
            artifacts=self._completion_artifacts(state),
        )
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
            state.llm_metadata,
            recorder,
            state.conversation,
            context_compactions=state.context_compactions,
            last_context_event=state.last_context_event,
            emit_message_delta=emit_message_delta,
            run_id=recorder.run_id,
            mode=runtime_mode(runtime_options),
            extra_metadata={
                "turn_phase": state.turn_phase,
                "turn_policy_reason": state.turn_policy_reason,
                "verification_verdict": state.verification_verdict,
                "verification_reason": state.verification_reason,
                **(state.primary_skill_context.to_metadata() if state.primary_skill_context is not None else {}),
                **guard_metadata,
            },
        )

    async def _run_tool_calling_turn(
        self,
        state: ToolLoopState,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
        *,
        emit_message_delta: bool,
    ) -> ToolLoopResult:
        # 标准工具调用路径：
        # 每一轮先让模型决定下一步要不要调工具；如果调工具，就执行并把
        # 结构化结果塞回 conversation，再进入下一轮，直到模型停止调用。
        final_response = LlmChatResponse()
        max_rounds = self._max_tool_rounds(agent_config, runtime_options)
        run_final_pass_after_conditional_repeat = False
        while state.rounds < max_rounds or run_final_pass_after_conditional_repeat:
            run_final_pass_after_conditional_repeat = False
            state.rounds += 1
            self._advance_turn_state(state, agent_config, runtime_options)
            self._apply_context_compaction(state, agent_config, recorder, round_number=state.rounds)
            request_started_at = time.perf_counter()
            recorder.emit(
                "llm.request.started",
                {
                    "round": state.rounds,
                    "mode": "tool_calling",
                    "message_count": len(state.conversation),
                    "tool_count": len(state.tools),
                    "tools": self._openai_tool_names(state.tools),
                    "tool_choice": state.llm.tool_choice,
                    "turn_phase": state.turn_phase,
                },
            )
            final_response = await state.llm.complete_with_tools(
                state.system_prompt,
                state.conversation,
                state.tools,
            )
            state.latest_assistant_reply = final_response.content or ""
            recorder.emit(
                "llm.request.completed",
                {
                    "round": state.rounds,
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
            if self._is_empty_final_response(final_response):
                retrying = state.rounds < max_rounds
                recorder.emit(
                    "llm.empty_response",
                    {
                        "round": state.rounds,
                        "finish_reason": final_response.finish_reason or "",
                        "retrying": retrying,
                        "tool_count": len(state.tools),
                        "tools": self._openai_tool_names(state.tools),
                    },
                )
                if retrying:
                    state.conversation.append(self._empty_response_retry_message(state))
                    continue
                return self._finalize_empty_response_turn(
                    state,
                    agent_config,
                    thread_id,
                    recorder,
                    runtime_options,
                    emit_message_delta=emit_message_delta,
                )
            assistant_message = self._assistant_message(final_response)
            state.conversation.append(assistant_message)
            if not final_response.tool_calls:
                return self._finalize_completed_turn(
                    state,
                    final_response.content,
                    agent_config,
                    thread_id,
                    recorder,
                    runtime_options,
                    emit_message_delta=emit_message_delta,
                )
            self._accumulate_tool_results(
                state,
                final_response.tool_calls,
                agent_config,
                thread_id,
                recorder,
                runtime_options,
            )
            if state.required_inputs:
                self._advance_turn_state(state, agent_config, runtime_options)
                return self._finalize_completed_turn(
                    state,
                    state.latest_assistant_reply,
                    agent_config,
                    thread_id,
                    recorder,
                    runtime_options,
                    emit_message_delta=emit_message_delta,
                )
            repeated = await self._auto_repeat_conditional_tool_call(
                state,
                final_response.tool_calls,
                agent_config,
                thread_id,
                recorder,
                runtime_options,
            )
            if state.required_inputs:
                self._advance_turn_state(state, agent_config, runtime_options)
                return self._finalize_completed_turn(
                    state,
                    state.latest_assistant_reply,
                    agent_config,
                    thread_id,
                    recorder,
                    runtime_options,
                    emit_message_delta=emit_message_delta,
                )
            if repeated:
                policy = conditional_loop_policy(state.conversation, runtime_options)
                if policy is not None and not policy.should_continue(latest_tool_message_text(state.conversation)):
                    run_final_pass_after_conditional_repeat = True
                continue
            await self._wait_before_conditional_tool_retry(state, recorder, runtime_options)
        return self._finalize_exhausted_turn(
            state,
            final_response,
            agent_config,
            thread_id,
            recorder,
            runtime_options,
            emit_message_delta=emit_message_delta,
            max_rounds=max_rounds,
        )

    @staticmethod
    def _normalize_message(message: ChatHistoryMessage) -> dict[str, Any]:
        if isinstance(message, Message):
            return message.model_dump(mode="python")
        return dict(message)

    def _apply_context_compaction(
        self,
        state: ToolLoopState,
        agent_config: AgentConfig,
        recorder: EventRecorder,
        *,
        round_number: int | None,
    ) -> None:
        compaction = self._compact_context(agent_config, state.conversation)
        if not compaction.compacted:
            return
        state.context_compactions += 1
        state.last_context_event = compaction.event_payload()
        event_payload = (
            state.last_context_event
            if round_number is None
            else {**state.last_context_event, "round": round_number}
        )
        recorder.emit("context.compacted", event_payload)
        state.conversation = compaction.messages

    def _accumulate_tool_results(
        self,
        state: ToolLoopState,
        tool_calls: list[LlmToolCall],
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
    ) -> None:
        recorder.emit(
            "tool.calls.started",
            {
                "round": state.rounds,
                "tool_calls": [
                    {"id": tool_call.id, "name": tool_call.name}
                    for tool_call in tool_calls
                ],
            },
        )
        state.latest_tool_results = []
        for tool_call in tool_calls:
            state.tool_call_count += 1
            tool_result = self._execute_tool_call(
                tool_call,
                agent_config,
                thread_id,
                recorder,
                runtime_options=runtime_options,
            )
            state.latest_tool_results.append(tool_result.structured_content)
            self._extend_required_inputs(state.required_inputs, self._required_inputs_from_tool_result(tool_result))
            state.artifacts.extend(self._artifacts_from_tool_result(tool_result, seen=state.artifacts))
            state.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": self._tool_result_content(
                        tool_result,
                        max_chars=agent_config.runtime.max_tool_result_chars,
                    ),
                }
            )

    async def _wait_before_conditional_tool_retry(
        self,
        state: ToolLoopState,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
    ) -> None:
        policy = conditional_loop_policy(state.conversation, runtime_options)
        if policy is None or policy.delay_seconds <= 0:
            return
        if not policy.should_continue(latest_tool_message_text(state.conversation)):
            return
        recorder.emit(
            "tool.loop.waiting",
            {
                "round": state.rounds,
                "delay_seconds": policy.delay_seconds,
                "policy": policy.to_metadata(),
                "reason": "Latest tool result still matches the conditional retry policy.",
            },
        )
        await asyncio.sleep(policy.delay_seconds)

    async def _auto_repeat_conditional_tool_call(
        self,
        state: ToolLoopState,
        tool_calls: list[LlmToolCall],
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
    ) -> bool:
        if not self._can_auto_repeat_conditional_tool_call(state, tool_calls, runtime_options):
            return False
        original_call = tool_calls[0]
        repeated = False
        while True:
            policy = conditional_loop_policy(state.conversation, runtime_options)
            if policy is None or not policy.should_continue(latest_tool_message_text(state.conversation)):
                break
            if policy.delay_seconds > 0:
                recorder.emit(
                    "tool.loop.waiting",
                    {
                        "round": state.rounds,
                        "delay_seconds": policy.delay_seconds,
                        "policy": policy.to_metadata(),
                        "reason": "Latest tool result still matches the conditional retry policy.",
                    },
                )
                await asyncio.sleep(policy.delay_seconds)
            state.rounds += 1
            repeated = True
            repeat_call = LlmToolCall(
                id=f"{original_call.id}-conditional-repeat-{state.rounds}",
                name=original_call.name,
                arguments=original_call.arguments,
            )
            recorder.emit(
                "tool.loop.auto_repeating",
                {
                    "round": state.rounds,
                    "tool_call": {"id": repeat_call.id, "name": repeat_call.name},
                    "policy": policy.to_metadata(),
                    "reason": "Replaying the previous MCP tool call because the conditional loop still matches.",
                },
            )
            state.conversation.append(self._assistant_message(LlmChatResponse(tool_calls=[repeat_call])))
            self._accumulate_tool_results(
                state,
                [repeat_call],
                agent_config,
                thread_id,
                recorder,
                runtime_options,
            )
            if state.required_inputs:
                break
        return repeated

    def _can_auto_repeat_conditional_tool_call(
        self,
        state: ToolLoopState,
        tool_calls: list[LlmToolCall],
        runtime_options: RuntimeOptions | None,
    ) -> bool:
        policy = conditional_loop_policy(state.conversation, runtime_options)
        if policy is None or not policy.auto_repeat_tool_call:
            return False
        if len(tool_calls) != 1:
            return False
        tool_call = tool_calls[0]
        if policy.repeat_tool_name and policy.repeat_tool_name != tool_call.name:
            return False
        definition = self._tool_definition_by_name(state.tool_definitions, tool_call.name)
        if definition is None:
            return False
        source_type = str(definition.source.get("type") or "")
        if source_type not in {"mcp_streamable_http", "mcp_stdio"}:
            return False
        if source_type == "mcp_streamable_http" and str(definition.source.get("operation") or "call") != "call":
            return False
        return True

    def _advance_turn_state(
        self,
        state: ToolLoopState,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions | None,
    ) -> None:
        primary_skill_context = load_primary_skill_context(
            root_dir=self.tool_service.root_dir,
            artifact_store=self.artifact_store,
            runtime_options=runtime_options,
            thread_id=self._thread_id(runtime_options),
            available_artifacts=state.available_artifacts,
            current_artifacts=state.artifacts,
            required_inputs=state.required_inputs,
        )
        verification = verify_turn_completion(
            runtime_options=runtime_options,
            artifacts=self._completion_artifacts(state),
            required_inputs=state.required_inputs,
            latest_tool_results=state.latest_tool_results,
            tool_call_count=state.tool_call_count,
            primary_skill_context=primary_skill_context,
        )
        turn_policy = decide_turn_policy(
            TurnPolicyInput(
                messages=state.conversation,
                available_artifacts=self._completion_artifacts(state),
                current_phase=state.turn_phase,
                tool_call_count=state.tool_call_count,
                required_inputs=state.required_inputs,
                latest_tool_results=state.latest_tool_results,
                latest_assistant_reply=state.latest_assistant_reply,
                verification=verification,
                primary_skill_context=primary_skill_context,
                runtime_options=runtime_options,
            )
        )
        state.turn_phase = turn_policy.phase
        state.turn_policy_reason = turn_policy.reason
        state.primary_skill_context = primary_skill_context
        state.verification_verdict = verification.verdict
        state.verification_reason = verification.reason
        base_prompt, memory_context_count = build_system_prompt(
            agent_config,
            runtime_options,
            tool_service=self.tool_service,
            primary_skill_context=primary_skill_context,
        )
        state.system_prompt = self._prompt_with_turn_policy(base_prompt, state.turn_phase, state.turn_policy_reason)
        state.memory_context_count = memory_context_count
        tool_definitions = self._tool_definitions_for_run(agent_config, runtime_options)
        state.system_prompt = prompt_with_runtime_mcp_discovery_failures(state.system_prompt, runtime_options)
        exposure = select_tools_for_phase(
            tool_definitions,
            phase=state.turn_phase,
            primary_skill_context=primary_skill_context,
        )
        state.tool_definitions = exposure.visible_tools
        state.priority_tool_definitions = exposure.priority_tools
        state.secondary_tool_definitions = exposure.secondary_tools
        state.hidden_tool_definitions = exposure.hidden_tools
        if state.llm.tool_choice == "none":
            state.tools = []
        else:
            state.tools = [self._openai_tool(tool) for tool in exposure.visible_tools]

    def _finalize_completed_turn(
        self,
        state: ToolLoopState,
        reply: str | None,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
        *,
        emit_message_delta: bool,
    ) -> ToolLoopResult:
        final_reply = reply_with_required_inputs(
            reply or "",
            state.required_inputs,
            tool_call_count=state.tool_call_count,
            artifacts=state.artifacts,
        )
        final_reply, guard_metadata = guard_unverified_completion(
            final_reply,
            runtime_options,
            available_tool_count=len(state.tools),
            executed_tool_count=state.tool_call_count,
            required_inputs=state.required_inputs,
            artifacts=self._completion_artifacts(state),
        )
        return self._result(
            agent_config,
            thread_id,
            final_reply,
            state.rounds,
            state.tool_call_count,
            state.llm_metadata,
            recorder,
            state.conversation,
            context_compactions=state.context_compactions,
            last_context_event=state.last_context_event,
            emit_message_delta=emit_message_delta,
            run_id=recorder.run_id,
            mode=runtime_mode(runtime_options),
            extra_metadata={
                "turn_phase": state.turn_phase,
                "turn_policy_reason": state.turn_policy_reason,
                "verification_verdict": state.verification_verdict,
                "verification_reason": state.verification_reason,
                **(state.primary_skill_context.to_metadata() if state.primary_skill_context is not None else {}),
                **guard_metadata,
            },
            required_inputs=state.required_inputs,
            artifacts=state.artifacts,
        )

    def _finalize_exhausted_turn(
        self,
        state: ToolLoopState,
        final_response: LlmChatResponse,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
        *,
        emit_message_delta: bool,
        max_rounds: int,
    ) -> ToolLoopResult:
        reply = final_response.content or f"工具调用轮数已达到上限（{max_rounds} 轮），请缩小问题范围或继续追问。"
        return self._result(
            agent_config,
            thread_id,
            reply,
            state.rounds,
            state.tool_call_count,
            state.llm_metadata,
            recorder,
            state.conversation,
            status="failed" if final_response.tool_calls else "completed",
            completed_event=not final_response.tool_calls,
            context_compactions=state.context_compactions,
            last_context_event=state.last_context_event,
            emit_message_delta=emit_message_delta,
            run_id=recorder.run_id,
            mode=runtime_mode(runtime_options),
            extra_metadata={
                "turn_phase": state.turn_phase,
                "turn_policy_reason": state.turn_policy_reason,
                "verification_verdict": state.verification_verdict,
                "verification_reason": state.verification_reason,
                **(state.primary_skill_context.to_metadata() if state.primary_skill_context is not None else {}),
            },
            required_inputs=state.required_inputs,
            artifacts=state.artifacts,
        )

    def _finalize_empty_response_turn(
        self,
        state: ToolLoopState,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None,
        *,
        emit_message_delta: bool,
    ) -> ToolLoopResult:
        reply = "模型返回了空响应：没有生成内容，也没有调用任何工具。请重试，或检查当前模型的工具调用配置。"
        return self._result(
            agent_config,
            thread_id,
            reply,
            state.rounds,
            state.tool_call_count,
            state.llm_metadata,
            recorder,
            state.conversation,
            status="failed",
            completed_event=False,
            context_compactions=state.context_compactions,
            last_context_event=state.last_context_event,
            emit_message_delta=emit_message_delta,
            run_id=recorder.run_id,
            mode=runtime_mode(runtime_options),
            extra_metadata={
                "turn_phase": state.turn_phase,
                "turn_policy_reason": state.turn_policy_reason,
                "verification_verdict": state.verification_verdict,
                "verification_reason": state.verification_reason,
                "empty_llm_response": True,
                **(state.primary_skill_context.to_metadata() if state.primary_skill_context is not None else {}),
            },
            required_inputs=state.required_inputs,
            artifacts=state.artifacts,
        )

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

    def _tool_definitions_for_run(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions | None = None,
        recorder: EventRecorder | None = None,
    ) -> list[ToolDefinition]:
        mode = runtime_mode(runtime_options)
        if mode == "plan":
            return []
        allowed_names = self._allowed_tool_names(agent_config)
        selected_skill_names_for_run = selected_skill_names(runtime_options)
        selected_mcp_tool_names = self._selected_mcp_tool_names(runtime_options)
        runtime_client_tools = self._runtime_client_tools(runtime_options)
        if runtime_client_tools:
            selected_mcp_tool_names = selected_mcp_tool_names | {tool.name for tool in runtime_client_tools}
        if self._workspace_tools_enabled(runtime_options, mode):
            allowed_names = allowed_names | self._workspace_tool_names(mode)
        runtime_mcp_discovery = self.tool_service.discover_runtime_mcp_tools(self._runtime_mcp_servers(runtime_options))
        if runtime_options is not None:
            runtime_options.config_options["runtime_mcp_discovery_failures"] = [
                failure.to_event_data() for failure in runtime_mcp_discovery.failures
            ]
        for failure in runtime_mcp_discovery.failures:
            if recorder is not None:
                recorder.emit("mcp.discovery.failed", failure.to_event_data())
        runtime_mcp_tools = [*runtime_mcp_discovery.tools, *runtime_client_tools]
        if runtime_mcp_tools:
            if runtime_options is not None:
                runtime_options.config_options["runtime_mcp_tools"] = [tool.to_payload() for tool in runtime_mcp_tools]
            selected_mcp_tool_names = selected_mcp_tool_names | {tool.name for tool in runtime_mcp_tools}
        if not allowed_names and not selected_skill_names_for_run and not selected_mcp_tool_names:
            return []

        definitions = self.tool_service.list_tools()
        if runtime_mcp_tools:
            definitions = [*definitions, *runtime_mcp_tools]
        if not agent_config.memory.enabled:
            definitions = [
                tool for tool in definitions if tool.source.get("type") not in {"memory", "markdown_memory"}
            ]
        elif not agent_config.memory.markdown_enabled:
            definitions = [tool for tool in definitions if tool.source.get("type") != "markdown_memory"]
        definitions = [
            tool
            for tool in definitions
            if self._tool_selected_for_run(
                tool,
                allowed_names,
                selected_skill_names_for_run,
                selected_mcp_tool_names,
            )
        ]
        definitions = [tool for tool in definitions if self._tool_allowed_in_mode(tool, mode)]
        return definitions

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
    def _workspace_tools_enabled(runtime_options: RuntimeOptions | None, mode: str) -> bool:
        if runtime_options is None:
            return False
        raw_enabled = runtime_options.config_options.get("enableWorkspaceTools")
        if isinstance(raw_enabled, bool):
            return raw_enabled
        if isinstance(raw_enabled, str) and raw_enabled.strip().lower() in {"1", "true", "yes", "on"}:
            return True
        return mode in {"autonomous", "yolo"} and bool(
            str(runtime_options.config_options.get("session_cwd") or "").strip()
        )

    @staticmethod
    def _workspace_tool_names(mode: str) -> set[str]:
        names = {
            "present_files",
            "local_read_file",
            "local_search_text",
            "local_patch_file",
            "local_write_file",
            "local_todo",
            "extract_archive",
            "validate_yolo_training_inputs",
        }
        if mode in {"autonomous", "yolo"}:
            names.add("local_shell_command")
        return names

    @staticmethod
    def _selected_mcp_tool_names(runtime_options: RuntimeOptions | None) -> set[str]:
        if runtime_options is None:
            return set()
        return {name.strip() for name in runtime_options.selected_mcp_tools if name.strip()}

    @staticmethod
    def _runtime_mcp_servers(runtime_options: RuntimeOptions | None) -> list[dict[str, Any]]:
        if runtime_options is None:
            return []
        raw_servers = runtime_options.config_options.get("mcpServers") or runtime_options.config_options.get("mcp_servers")
        if not isinstance(raw_servers, list):
            return []
        return [dict(server) for server in raw_servers if isinstance(server, dict)]

    @staticmethod
    def _runtime_client_tools(runtime_options: RuntimeOptions | None) -> list[ToolDefinition]:
        if runtime_options is None:
            return []
        raw_tools = ToolCallingAgentLoop._runtime_client_tools_payload(runtime_options.config_options)
        items = ToolCallingAgentLoop._runtime_client_tool_items(raw_tools)
        definitions: list[ToolDefinition] = []
        seen: set[str] = set()
        for item in items:
            raw_name = str(item.get("name") or item.get("id") or item.get("tool") or item.get("toolName") or "").strip()
            if not raw_name:
                continue
            name = ToolCallingAgentLoop._runtime_client_tool_name(raw_name)
            if not name or name in seen:
                continue
            seen.add(name)
            source = dict(item.get("source") or {}) if isinstance(item.get("source"), dict) else {}
            response = str(
                source.get("response_template")
                or item.get("response_template")
                or item.get("responseTemplate")
                or item.get("response")
                or item.get("result")
                or item.get("output")
                or item.get("text")
                or f"Client tool {raw_name} completed."
            )
            source = {
                **source,
                "type": "manual",
                "response_template": response,
                "client_tool_name": raw_name,
            }
            definitions.append(
                ToolDefinition(
                    name=name,
                    title=str(item.get("title") or raw_name),
                    description=str(item.get("description") or f"Runtime client dynamic tool: {raw_name}"),
                    input_schema=ToolCallingAgentLoop._dict_or_default(
                        item.get("input_schema") or item.get("inputSchema")
                    ),
                    output_schema=ToolCallingAgentLoop._dict_or_default(
                        item.get("output_schema") or item.get("outputSchema")
                    ),
                    enabled=bool(item.get("enabled", True)),
                    source=source,
                    editable=False,
                )
            )
        return definitions

    @staticmethod
    def _runtime_client_tools_payload(config_options: dict[str, Any]) -> object:
        for key in (
            "session_init_tools",
            "sessionInitTools",
            "sessioninittools",
            "client_tools",
            "clientTools",
            "dynamic_tools",
            "dynamicTools",
        ):
            if key in config_options:
                return config_options[key]
        return None

    @staticmethod
    def _runtime_client_tool_items(raw_tools: object) -> list[dict[str, Any]]:
        if raw_tools is None:
            return []
        if isinstance(raw_tools, list):
            return [dict(item) for item in raw_tools if isinstance(item, dict)]
        if isinstance(raw_tools, dict):
            raw_list = raw_tools.get("tools")
            if isinstance(raw_list, list):
                return [dict(item) for item in raw_list if isinstance(item, dict)]
            return [
                {"name": name, "response": value}
                for name, value in raw_tools.items()
                if isinstance(name, str) and not name.startswith("_")
            ]
        if isinstance(raw_tools, str):
            items: list[dict[str, Any]] = []
            for line in raw_tools.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                raw_name, separator, response = stripped.partition("|")
                if not separator:
                    raw_name, separator, response = stripped.partition(":")
                if raw_name.strip():
                    items.append({"name": raw_name.strip(), "response": response.strip() if separator else ""})
            return items
        return []

    @staticmethod
    def _runtime_client_tool_name(raw_name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name.strip()).strip("_")
        if not normalized:
            return ""
        if normalized.startswith("jetlinks_session_"):
            return normalized[:128]
        return f"jetlinks_session_{normalized}"[:128]

    @staticmethod
    def _dict_or_default(value: object) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return {"type": "object", "additionalProperties": True}

    @staticmethod
    def _tool_definition_by_name(tools: list[ToolDefinition], name: str) -> ToolDefinition | None:
        for tool in tools:
            if tool.name == name:
                return tool
        return None

    @staticmethod
    def _thread_id(runtime_options: RuntimeOptions | None) -> str:
        raw = "" if runtime_options is None else str(runtime_options.thread_id or "")
        return raw.strip() or "thread-default"

    def _thread_artifacts(self, thread_id: str) -> list[dict[str, Any]]:
        manifest = self.artifact_store.read_manifest(thread_id)
        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list):
            return []
        return [dict(item) for item in raw_artifacts if isinstance(item, dict)]

    @staticmethod
    def _completion_artifacts(state: ToolLoopState) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for source in [state.available_artifacts, state.artifacts]:
            for item in source or []:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "")
                key = path or str(item.get("name") or "")
                if key and key in seen_paths:
                    continue
                merged.append(dict(item))
                if key:
                    seen_paths.add(key)
        return merged

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
            "mcp_streamable_http",
            "mcp_stdio",
        }

    @staticmethod
    def _max_tool_rounds(agent_config: AgentConfig, runtime_options: RuntimeOptions | None) -> int:
        base = ToolCallingAgentLoop._base_tool_rounds(agent_config, runtime_options)
        mode = runtime_mode(runtime_options)
        if mode == "plan":
            return 1
        if mode == "safe":
            return base
        if mode in {"autonomous", "yolo"}:
            return min(MAX_TOOL_ROUNDS_LIMIT, max(base, min(base * 2, 16)))
        return base

    @staticmethod
    def _base_tool_rounds(agent_config: AgentConfig, runtime_options: RuntimeOptions | None) -> int:
        configured = min(max(1, agent_config.runtime.max_tool_rounds), MAX_TOOL_ROUNDS_LIMIT)
        if runtime_options is None:
            return configured
        raw_value = runtime_options.config_options.get("max_tool_rounds")
        if not isinstance(raw_value, str | int | float):
            return configured
        try:
            requested = int(raw_value)
        except (TypeError, ValueError):
            return configured
        return min(max(1, requested), MAX_TOOL_ROUNDS_LIMIT)

    @staticmethod
    def _runtime_mode(runtime_options: RuntimeOptions | None) -> str:
        """Compatibility wrapper for callers that still reference the loop class."""

        return runtime_mode(runtime_options)

    @staticmethod
    def _selected_skill_names(runtime_options: RuntimeOptions | None) -> set[str]:
        """Compatibility wrapper for callers that still reference the loop class."""

        return selected_skill_names(runtime_options)

    @staticmethod
    def _tool_allowed_in_mode(tool: ToolDefinition, mode: str) -> bool:
        if mode in {"autonomous", "yolo"}:
            return True
        source_type = tool.source.get("type")
        operation = tool.source.get("operation")
        if mode == "safe":
            if source_type == "skill":
                return False
            if source_type == "local" and operation in {"write_file", "patch_file", "shell_command", "extract_archive"}:
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
    def _is_empty_final_response(response: LlmChatResponse) -> bool:
        return not response.tool_calls and not (response.content or "").strip()

    def _empty_response_retry_message(self, state: ToolLoopState) -> dict[str, Any]:
        tool_names = self._openai_tool_names(state.tools)
        tool_text = ", ".join(tool_names) if tool_names else "无"
        return {
            "role": "user",
            "content": "\n".join(
                [
                    "上一轮模型返回了空响应：没有生成内容，也没有调用工具。",
                    "请继续完成用户的原始请求。",
                    "如果需要生成文件、修改会话内容或返回结构化结果，请调用合适的可用工具。",
                    "如果无法完成，请直接输出明确的失败原因，不要再次返回空内容。",
                    f"当前可用工具：{tool_text}",
                ]
            ),
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
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None = None,
    ) -> ToolInvocationResult:
        outcome = self.tool_orchestrator.execute(
            tool_call,
            agent_config=agent_config,
            thread_id=thread_id,
            recorder=recorder,
            runtime_options=runtime_options,
        )
        return outcome.result

    @staticmethod
    def _required_inputs_from_tool_result(result: ToolInvocationResult) -> list[dict[str, Any]]:
        if result.structured_content.get("requires_input") is not True:
            return []
        raw_items = result.structured_content.get("required_inputs")
        if not isinstance(raw_items, list):
            return []
        return [dict(item) for item in raw_items if isinstance(item, dict)]

    @staticmethod
    def _extend_required_inputs(target: list[dict[str, Any]], items: list[dict[str, Any]]) -> None:
        seen = {
            (
                str(item.get("stage") or ""),
                str(item.get("type") or ""),
                str(item.get("reason") or ""),
            )
            for item in target
        }
        for item in items:
            key = (
                str(item.get("stage") or ""),
                str(item.get("type") or ""),
                str(item.get("reason") or ""),
            )
            if key in seen:
                continue
            target.append(item)
            seen.add(key)

    @classmethod
    def _guard_unverified_completion(
        cls,
        reply: str,
        runtime_options: RuntimeOptions | None,
        *,
        available_tool_count: int = 0,
        executed_tool_count: int = 0,
        required_inputs: list[dict[str, Any]] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Compatibility wrapper for tests and internal callers."""

        return guard_unverified_completion(
            reply,
            runtime_options,
            available_tool_count=available_tool_count,
            executed_tool_count=executed_tool_count,
            required_inputs=required_inputs,
            artifacts=artifacts,
        )

    @staticmethod
    def _reply_with_required_inputs(
        reply: str,
        required_inputs: list[dict[str, Any]],
        *,
        tool_call_count: int,
        artifacts: list[dict[str, Any]],
    ) -> str:
        """Compatibility wrapper for tests and internal callers."""

        return reply_with_required_inputs(
            reply,
            required_inputs,
            tool_call_count=tool_call_count,
            artifacts=artifacts,
        )

    @staticmethod
    def _prompt_with_turn_policy(prompt: str, phase: str, reason: str) -> str:
        if not phase:
            return prompt
        guidance = "\n".join(
            [
                "Turn policy is active.",
                f"Phase: {phase}",
                f"Why: {reason}",
            ]
        )
        return f"{prompt}\n\n{guidance}"

    @staticmethod
    def _artifacts_from_tool_result(
        result: ToolInvocationResult,
        *,
        seen: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        raw_artifacts = result.structured_content.get("artifacts")
        if not isinstance(raw_artifacts, list):
            return []
        seen_paths = {str(item.get("path") or "") for item in seen if isinstance(item, dict)}
        artifacts: list[dict[str, Any]] = []
        for item in raw_artifacts:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "")
            if path and path in seen_paths:
                continue
            artifacts.append(item)
            if path:
                seen_paths.add(path)
        return artifacts

    @staticmethod
    def _tool_arguments(raw_arguments: str) -> dict[str, Any]:
        return ToolOrchestrator.tool_arguments(raw_arguments)

    @staticmethod
    def _observable_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        return ToolOrchestrator.observable_arguments(arguments)

    @staticmethod
    def _tool_error_code(exc: Exception) -> str:
        return ToolOrchestrator.tool_error_code(exc)

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
        artifacts: list[dict[str, Any]] | None = None,
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
        artifact_refs = [ArtifactRef.model_validate(item) for item in artifacts or [] if isinstance(item, dict)]
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=thread_id,
            status="completed" if status == "completed" else "failed",
            reply=reply,
            artifacts=artifact_refs,
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
