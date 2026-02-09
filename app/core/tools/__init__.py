"""
工具系统 - 智能体能力扩展中心

功能说明:
- 提供统一的工具注册和管理机制
- 所有工具都继承自BaseTool（兼容qwen-agent）
- 支持内置工具、外部工具、MCP工具等多种类型
- 工具通过装饰器或注册函数动态注册到TOOL_REGISTRY

使用状态: ✅ 活跃使用 (工具注册中心)

调用链路:
1. 启动时: 自动导入并注册所有工具到TOOL_REGISTRY
2. 运行时: Agent → session_tool_manager → TOOL_REGISTRY → 具体工具
3. 执行: Agent.run() → 工具调用 → 返回结果

核心组件:
- TOOL_REGISTRY: 全局工具注册表
- BaseTool: 工具基类（qwen-agent兼容）
- register_tool(): 注册工具装饰器
- get_tool(): 获取工具实例
- list_tools(): 列出所有可用工具

主要工具类型:
1. 绘图工具: BoundingBoxDrawer
2. 下载工具: VideoDownloader
3. 数据分析: ChatBITool (Text-to-SQL, 数据可视化)
4. 图像分析: ImageDescriptionTool (图像描述工具)
"""

import logging

from app.core.tools.base import BaseTool, register_tool, get_tool, list_tools, TOOL_REGISTRY

logger = logging.getLogger(__name__)


def _safe_import(module_path: str, name: str):
    try:
        module = __import__(module_path, fromlist=[name])
        return getattr(module, name)
    except Exception as exc:
        logger.warning("⚠️ Skip tool import: %s (%s)", name, exc)
        return None

# 修复导入路径：agent_tools 子目录中的工具
# from app.core.tools.agent_tools.video_tools import (
#     VideoFrameExtractor,
#     VideoAnalyzer,
#     SafetyDetector,
#     ReportGenerator
# )  # 已删除视频分析工具
BoundingBoxDrawer = _safe_import("app.core.tools.agent_tools.bbox_drawer", "BoundingBoxDrawer")
from app.core.tools.agent_tools.video_downloader import VideoDownloader
ChatBITool = _safe_import("app.core.tools.agent_tools.chatbi_tool", "ChatBITool")
# from app.core.tools.agent_tools.weather import WeatherTool  # 已删除测试天气工具
from app.core.tools.agent_tools.image_description_tool import ImageDescriptionTool

# 新增的AI生成工具
from app.core.tools.agent_tools.video_generator_tool import VideoGeneratorTool
from app.core.tools.agent_tools.gemini_image_generator_tool import GeminiImageGeneratorTool
from app.core.tools.agent_tools.sora_video_generator_tool import SoraVideoGeneratorTool

# Google搜索工具
from app.core.tools.google_tool import GoogleSearchTool
from app.core.tools.baidu_tool  import BaiduSearchTool
# Skill tool
from app.core.tools.skill_tool import SkillTool
# 导入参数填充器
from app.core.tools.parameter_filler import (
    ParameterFiller,
    parameter_filler,
    init_parameter_filler,
    get_parameter_filler
)
from app.core.tools.session_tool_manager import SessionToolManager, session_tool_manager
from app.core.tools.internal_tool_executor import InternalToolExecutor, internal_tool_executor
from app.core.tools.internal_tool_registry import InternalToolRegistry, internal_tool_registry

__all__ = [
    "BaseTool",
    "register_tool",
    "get_tool",
    "list_tools",
    "TOOL_REGISTRY",
    # "VideoFrameExtractor",    # 已删除
    # "VideoAnalyzer",         # 已删除
    # "SafetyDetector",        # 已删除
    # "ReportGenerator",       # 已删除
    "BoundingBoxDrawer",
    "VideoDownloader",
    "ChatBITool",
    # "WeatherTool",  # 已删除测试天气工具
    "ImageDescriptionTool",
    # 新增的AI生成工具
    "VideoGeneratorTool",
    "GeminiImageGeneratorTool",
    "SoraVideoGeneratorTool",
    # Google搜索工具
    "GoogleSearchTool",
    "BaiduSearchTool",
    "SkillTool",
    "ParameterFiller",
    "parameter_filler",
    "init_parameter_filler",
    "get_parameter_filler",
    "SessionToolManager",
    "session_tool_manager",
    "InternalToolExecutor",
    "internal_tool_executor",
    "InternalToolRegistry",
    "internal_tool_registry"
]

_OPTIONAL_TOOLS = {
    "BoundingBoxDrawer": BoundingBoxDrawer,
    "ChatBITool": ChatBITool,
}

__all__ = [name for name in __all__ if name not in _OPTIONAL_TOOLS or _OPTIONAL_TOOLS[name] is not None]
