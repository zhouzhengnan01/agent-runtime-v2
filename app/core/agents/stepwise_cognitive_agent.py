"""
Stepwise cognitive agent.

Runs tool selection and execution one step at a time, using the latest tool
results to decide the next action.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, AsyncIterator

from langchain_core.messages import HumanMessage

from app.core.agents.cognitive_agent import LangChainCognitiveAgent
from app.services.session_markdown_logger import SessionMarkdownLogger


logger = logging.getLogger(__name__)


class StepwiseCognitiveAgent(LangChainCognitiveAgent):
    """Stepwise planning/execution engine."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default to 10 steps so multi-tool workflows (esp. external tools) can finish in one session.
        self.stepwise_max_steps = int(config.get("stepwise_max_steps", 10))
        self.stepwise_show_plan = bool(config.get("stepwise_show_plan", False))
        self.stepwise_planner_context = bool(config.get("stepwise_planner_context", True))
        self.stepwise_history_max = int(config.get("stepwise_history_max", 3))
        self.stepwise_result_max_chars = int(config.get("stepwise_result_max_chars", 300))
        self.session_md_enabled = bool(config.get("session_md_enabled", False))
        self.session_md_dir = config.get("session_md_dir", "storage/session_rd")
        self.session_md_logger = SessionMarkdownLogger(self.session_md_dir) if self.session_md_enabled else None

    def _build_stepwise_input(self, base_input: str, agent_context: Dict[str, Any]) -> str:
        if not self.stepwise_planner_context:
            return base_input
        history = agent_context.get("tool_call_history") or []
        if not history:
            return base_input

        recent = history[-self.stepwise_history_max:]
        lines = ["Tool results so far:"]
        for entry in recent:
            tool_name = self._display_tool_name(entry.get("tool_name", "unknown"))
            result = entry.get("result", "")
            summary = self._format_tool_result_for_prompt(result)
            if len(summary) > self.stepwise_result_max_chars:
                summary = summary[: self.stepwise_result_max_chars] + "..."
            lines.append(f"- {tool_name}: {summary}")

        lines.append("Decide the next action based on the available tools.")
        return f"{base_input}\n\n" + "\n".join(lines)

    def _md_append(self, title: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.session_md_logger or not self.session_id:
            return
        self.session_md_logger.append(
            session_id=self.session_id,
            title=title,
            content=content,
            meta=meta,
            agent_id=self.agent_id,
        )

    async def _run_stream(self, messages: List[Any], **kwargs) -> AsyncIterator[Any]:
        task_start_time = time.time()
        input_text = ""
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                input_text = last_message.content
            elif isinstance(last_message, dict):
                input_text = last_message.get("content", "")
            else:
                input_text = str(last_message)

        self._log_agent_execution_start(input_text, self.name)

        context_init = await self._initialize_execution_context(messages, **kwargs)
        system_prompt = context_init["system_prompt"]
        input_text = context_init["input_text"]
        tools = context_init["tools"]
        agent_context = context_init["agent_context"]

        base_input_text = input_text
        self._md_append("user", base_input_text)

        lc_messages: List[Any] = []
        if self.memory:
            try:
                memory_vars = self.memory.load_memory_variables({})
                lc_messages.extend(memory_vars.get("chat_history", []))
            except Exception as memory_error:
                logger.warning("Memory load failed; continue: %s", memory_error)
        lc_messages.append(HumanMessage(content=base_input_text))

        tools_executed: List[str] = []
        execution_success = True
        full_response = ""
        use_historical_plan = False
        last_plan: Optional[Dict[str, Any]] = None

        history_count = len(agent_context.get("tool_call_history", []))
        reached_limit = False

        for step in range(self.stepwise_max_steps):
            planner_input = self._build_stepwise_input(base_input_text, agent_context)
            plan, planned_tools, missing_tools, use_historical_plan, _ = await self._intelligent_tool_selection(
                planner_input, tools, agent_context
            )
            last_plan = plan

            if missing_tools:
                missing_list = ", ".join(missing_tools)
                message = f"未配置或不可用的工具: {missing_list}。请先注册/启用相关工具后重新初始化会话。"
                execution_success = False
                full_response = message
                self._md_append("assistant", message, {"missing_tools": missing_tools})
                yield message
                break

            if not planned_tools:
                break

            if len(planned_tools) > 1:
                planned_tools = [planned_tools[0]]
                if plan.get("tools_needed"):
                    plan["tools_needed"] = plan["tools_needed"][:1]
                if plan.get("steps"):
                    plan["steps"] = plan["steps"][:1]

            if self.stepwise_show_plan and plan.get("steps") and plan.get("tools_needed"):
                markdown_plan = self._format_plan_to_markdown(plan)
                if markdown_plan:
                    yield markdown_plan

            try:
                async for item in self._execute_tool_pipeline(planned_tools, planner_input, agent_context, lc_messages):
                    yield item
            except Exception as exc:
                # ToolPipelineAbort is raised from the base class for missing required params / external failures.
                execution_success = False
                full_response = str(exc)
                self._md_append(
                    "assistant",
                    full_response,
                    {"error": full_response, "stepwise": True},
                )
                yield full_response
                break

            for tool in planned_tools:
                tools_executed.append(self._display_tool_name(getattr(tool, "name", "")))

            new_history = agent_context.get("tool_call_history", [])[history_count:]
            if new_history:
                for entry in new_history:
                    tool_name = entry.get("tool_name", "unknown")
                    arguments = entry.get("arguments", {})
                    result = entry.get("result", "")
                    result_text = self._format_tool_result_for_prompt(result)
                    self._md_append("tool", result_text, {"tool": tool_name, "arguments": arguments})
            history_count = len(agent_context.get("tool_call_history", []))

            if step == self.stepwise_max_steps - 1:
                reached_limit = True

        if not full_response:
            if reached_limit:
                agent_context["stepwise_truncated"] = True
            async for chunk in self._generate_final_response(
                system_prompt, base_input_text, lc_messages, agent_context
            ):
                full_response += chunk
                yield chunk
            self._md_append("assistant", full_response, {"stepwise_truncated": reached_limit})

        task_duration = time.time() - task_start_time
        self._log_agent_execution_end(self.name, task_duration, tools_executed)
        await self._save_execution_to_memory(
            base_input_text,
            last_plan,
            tools_executed,
            full_response,
            use_historical_plan,
            execution_success,
            task_duration,
            agent_context,
        )
