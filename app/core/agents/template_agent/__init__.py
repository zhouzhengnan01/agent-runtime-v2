"""
Template Agent 模块

基于 LangChain 框架的智能体模板
"""

from app.core.agents.template_agent.base_template import BaseTemplateAgent
from app.core.agents.template_agent.video_inspection_template import VideoInspectionAgent
from app.core.agents.template_agent.tool_calling_template import ToolCallingAgent

__all__ = [
    "BaseTemplateAgent",  # LangChain 基类
    "VideoInspectionAgent",        # 视频巡检模板（LangChain）
    "ToolCallingAgent",            # 工具调用模板（LangChain）
]

__version__ = "3.0.0"  # 移除 qwen-agent 依赖