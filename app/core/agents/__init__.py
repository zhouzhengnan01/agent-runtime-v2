"""
JetLinks Agent 核心模块

提供基于 LangChain 的智能体模板和实现
"""

from app.core.agents.template_agent import (
    BaseTemplateAgent,
    VideoInspectionAgent,
    ToolCallingAgent
)

__all__ = [
    "BaseTemplateAgent",
    "VideoInspectionAgent",
    "ToolCallingAgent",
]

__version__ = "2.0.0"