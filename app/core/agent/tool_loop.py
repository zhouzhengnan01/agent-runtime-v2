from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.skills import SkillRegistry, SkillRunner
from app.core.tools import ToolDefinition, ToolInvocationResult, ToolInvocationService
from app.core.workflow.spec_builder import SpecBuilder
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
        self.artifact_store: ArtifactStore = self.tool_service.artifact_store
        self.skill_registry = SkillRegistry()
        self.spec_builder = SpecBuilder(self.skill_registry)
        self.skill_runner = SkillRunner(self.artifact_store)

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
        direct_skill = self._direct_selected_skill(agent_config, runtime_options)
        if direct_skill is not None:
            return self._run_direct_skill(
                agent_config=agent_config,
                messages=messages,
                thread_id=thread_id,
                recorder=recorder,
                llm_metadata=llm_metadata,
                conversation=conversation,
                skill_name=direct_skill,
                emit_message_delta=emit_message_delta,
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
            reply = await llm.complete(system_prompt, messages)
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
            final_response = await llm.complete_with_tools(system_prompt, conversation, tools)
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
                tool_result = self._execute_tool_call(tool_call, agent_config, thread_id, recorder)
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

    def _openai_tools(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions | None = None,
    ) -> list[dict[str, Any]]:
        allowed_names = self._allowed_tool_names(agent_config)
        selected_mcp_tools = self._normalized_selected_mcp_tools(runtime_options)
        if selected_mcp_tools:
            allowed_names = allowed_names | set(selected_mcp_tools)
        definitions = self.tool_service.list_tools()
        if not agent_config.memory.enabled:
            definitions = [tool for tool in definitions if tool.source.get("type") != "memory"]
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
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
    ) -> ToolInvocationResult:
        recorder.emit("tool.started", {"tool_name": tool_call.name, "tool_call_id": tool_call.id})
        try:
            arguments = self._tool_arguments(tool_call.arguments)
            arguments.setdefault("_thread_id", thread_id)
            arguments.setdefault("_agent_name", agent_config.name)
            arguments.setdefault("_memory_scope", agent_config.memory.scope)
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

    def _direct_selected_skill(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions | None,
    ) -> str | None:
        selected = self._normalized_selected_skills(runtime_options)
        if not selected:
            return None
        allowed_names = self._allowed_tool_names(agent_config)
        for skill_name in selected:
            if allowed_names and skill_name not in allowed_names:
                continue
            try:
                skill = self.skill_registry.get(skill_name)
            except KeyError:
                continue
            if skill.generation and skill.runner_path is not None:
                return skill_name
        return None

    @staticmethod
    def _normalized_selected_skills(runtime_options: RuntimeOptions | None) -> list[str]:
        if runtime_options is None:
            return []
        return [name.strip() for name in runtime_options.selected_skills if name.strip()]

    @staticmethod
    def _normalized_selected_mcp_tools(runtime_options: RuntimeOptions | None) -> list[str]:
        if runtime_options is None:
            return []
        return [name.strip() for name in runtime_options.selected_mcp_tools if name.strip()]

    def _run_direct_skill(
        self,
        *,
        agent_config: AgentConfig,
        messages: list[Message],
        thread_id: str,
        recorder: EventRecorder,
        llm_metadata: dict[str, Any],
        conversation: list[dict[str, Any]],
        skill_name: str,
        emit_message_delta: bool,
    ) -> ToolLoopResult:
        allowed_skills = self._allowed_skill_names(agent_config)
        recorder.emit(
            "direct_skill.started",
            {
                "skill_name": skill_name,
                "selected_skills": [skill_name],
                "mode": "explicit_runtime_options",
            },
        )
        spec = self.spec_builder.build(messages, [], allowed_skills)
        spec["skill_name"] = skill_name
        skill = self.skill_registry.get(skill_name)
        recorder.emit("skill.selected", {"skill": skill.to_event_payload(), "direct": True})
        recorder.emit("spec.completed", {"skill_name": skill_name, "spec": spec, "direct": True})
        paths = self.artifact_store.prepare_thread(thread_id)
        recorder.emit(
            "skill.started",
            {
                "skill_name": skill_name,
                "attempt": 0,
                "execution_mode": "local",
                "direct": True,
            },
        )
        run_result = self.skill_runner.run(skill_name, spec, paths)
        recorder.emit(
            "skill.completed",
            {
                "skill_name": skill_name,
                "attempt": 0,
                "execution_mode": run_result.data.get("execution_mode", "local"),
                "output_count": len(run_result.outputs),
                "data": run_result.data,
                "direct": True,
            },
        )
        for artifact in run_result.outputs:
            artifact_data = artifact.model_dump()
            recorder.emit("artifact.created", {"artifact": artifact_data, "attempt": 0})
            recorder.emit("preview.ready", {"artifact": artifact_data, "attempt": 0})
        artifact_lines = [
            f"- {artifact.name}（{artifact.kind}，{artifact.mime_type}，{artifact.path}）"
            for artifact in run_result.outputs
        ]
        artifact_text = "\n".join(artifact_lines) if artifact_lines else "- 无文件产物"
        reply = f"已按显式选择的 Skill 执行：{skill_name}。\n{artifact_text}\n可在右侧「文件」面板预览或下载。"
        if emit_message_delta:
            recorder.emit("agent.message.delta", {"text": reply})
        recorder.emit("agent.message", {"text": reply})
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=thread_id,
            reply=reply,
            artifacts=run_result.outputs,
            spec=spec,
            metadata={
                "workflow": "agent_loop",
                **llm_metadata,
                "tool_rounds": 0,
                "tool_call_count": 0,
                "direct_skill": True,
                "skill_name": skill_name,
            },
        )
        recorder.emit("run.completed", {"result": result.model_dump()})
        return ToolLoopResult(result=result, messages=conversation, rounds=0)

    def _allowed_skill_names(self, agent_config: AgentConfig) -> list[str]:
        if agent_config.skills:
            return agent_config.skills
        return [skill.name for skill in self.skill_registry.list(executable_only=True)]
