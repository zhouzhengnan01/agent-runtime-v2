"""
Stepwise tool-calling template.

Runs tool selection and execution step-by-step and logs each session to Markdown.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

from app.core.agents.template_agent.tool_calling_template import ToolCallingAgent
from app.core.agents.stepwise_cognitive_agent import StepwiseCognitiveAgent

logger = logging.getLogger(__name__)


class StepwiseToolAgent(ToolCallingAgent):
    """Stepwise tool agent template."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}

        config.setdefault("name", "Stepwise工具智能体")
        config.setdefault("description", "逐步规划 + 工具调用 + 会话Markdown记录")
        # Default to 10 to support longer external tool chains.
        config.setdefault("stepwise_max_steps", 10)
        # Show tool name/step in the chat so users can see what gets called.
        config.setdefault("stepwise_show_plan", True)
        config.setdefault("stepwise_planner_context", True)
        config.setdefault("stepwise_history_max", 3)
        config.setdefault("stepwise_result_max_chars", 300)
        config.setdefault("session_md_enabled", True)
        config.setdefault("session_md_dir", "storage/session_rd")

        super().__init__(config)

    def _initialize(self):
        """Swap cognitive engine to stepwise implementation."""
        self.cognitive_engine = StepwiseCognitiveAgent(self.config)
        self.function_list = self.cognitive_engine.function_list
        logger.info("StepwiseToolAgent initialized: %s", self.name)

    def get_capabilities(self) -> List[str]:
        capabilities = [
            "stepwise_planning",
            "tools",
            "streaming",
            "confirmation",
            "session_markdown",
        ]
        if self.function_list:
            capabilities.append(f"supports_{len(self.function_list)}_tools")
        return capabilities
