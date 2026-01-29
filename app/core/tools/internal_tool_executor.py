"""
内置工具执行器 - AI服务内部工具执行

功能说明:
- 判断是否是内置工具（检查是否在 TOOL_REGISTRY 中注册）
- 从 TOOL_REGISTRY 获取工具实例
- 执行工具并返回结果
- 统一的错误处理和日志记录

判断逻辑:
- 内置工具：在 TOOL_REGISTRY 中注册的工具（如 weather, ChatBITool）
- 外部工具：通过 WebSocket 传入的客户端工具（如 knowledgeService#Search）

使用示例:
    from app.core.tools.internal_tool_executor import internal_tool_executor

    # 判断是否是内置工具
    is_internal = internal_tool_executor.is_internal_tool("weather")  # True
    is_internal = internal_tool_executor.is_internal_tool("_ai_service_#weather")  # True（会去掉前缀）
    is_internal = internal_tool_executor.is_internal_tool("knowledgeService#Search")  # False

    # 执行内置工具
    result = await internal_tool_executor.execute(
        tool_name="_ai_service_#weather",  # 可带或不带前缀
        arguments={"city": "北京", "date": "2025-01-20"}
    )
"""

import logging
from typing import Dict, Any, Optional
import asyncio

from app.core.tools.base import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class InternalToolExecutor:
    """内置工具执行器"""

    # 内置工具前缀
    INTERNAL_TOOL_PREFIX = "_ai_service_#"

    def __init__(self):
        """初始化内置工具执行器"""
        self.tool_registry = TOOL_REGISTRY
        logger.info("✅ 内置工具执行器初始化完成")

    def is_internal_tool(self, tool_name: str) -> bool:
        """判断是否是内置工具（通过检查是否在 TOOL_REGISTRY 中注册）

        Args:
            tool_name: 工具名称，可能带有或不带有前缀

        Returns:
            bool: True 表示是内置工具，False 表示是外部工具

        Examples:
            >>> executor.is_internal_tool("_ai_service_#weather")
            True  # weather 在注册表中
            >>> executor.is_internal_tool("weather")
            True  # weather 在注册表中
            >>> executor.is_internal_tool("knowledgeService#Search")
            False  # 不在注册表中，是外部工具
        """
        if not tool_name:
            return False

        # 去掉前缀（如果有）
        real_name = self.get_real_tool_name(tool_name)

        # 检查是否在注册表中
        return real_name in self.tool_registry

    def get_real_tool_name(self, tool_name: str) -> str:
        """获取真实的工具名称（去掉前缀）

        Args:
            tool_name: 可能带前缀的工具名称

        Returns:
            str: 去掉前缀后的工具名称

        Examples:
            >>> executor.get_real_tool_name("_ai_service_#weather")
            "weather"
            >>> executor.get_real_tool_name("weather")
            "weather"
        """
        # 简单去掉前缀,避免递归
        if tool_name.startswith(self.INTERNAL_TOOL_PREFIX):
            return tool_name.replace(self.INTERNAL_TOOL_PREFIX, "", 1)
        return tool_name

    def get_tool_instance(self, tool_name: str) -> Optional[Any]:
        """从注册表获取工具实例

        Args:
            tool_name: 工具名称（可带或不带前缀）

        Returns:
            工具实例，如果不存在则返回 None
        """
        # 去掉前缀，获取真实名称
        real_name = self.get_real_tool_name(tool_name)

        # 从注册表获取工具类
        tool_class = self.tool_registry.get(real_name)

        if not tool_class:
            logger.warning(f"⚠️ 工具 {real_name} 未在 TOOL_REGISTRY 中注册")
            logger.debug(f"   可用工具: {list(self.tool_registry.keys())}")
            return None

        try:
            # 实例化工具
            tool_instance = tool_class()
            logger.info(f"✅ 成功获取工具实例: {real_name}")
            return tool_instance
        except Exception as e:
            logger.error(f"❌ 实例化工具失败: {real_name}, 错误: {e}", exc_info=True)
            return None

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = 120
    ) -> Dict[str, Any]:
        """执行内置工具

        Args:
            tool_name: 工具名称（可带或不带 _ai_service_# 前缀）
            arguments: 工具参数
            timeout: 超时时间（秒），默认120秒

        Returns:
            执行结果字典:
            {
                "success": True/False,
                "result": "工具返回结果" (成功时),
                "error": "错误信息" (失败时),
                "tool_name": "工具名称",
                "arguments": {...}
            }

        Examples:
            >>> result = await executor.execute(
            ...     tool_name="_ai_service_#weather",
            ...     arguments={"city": "北京"}
            ... )
            >>> print(result)
            {
                "success": True,
                "result": "📍 北京今天的天气预报：...",
                "tool_name": "weather",
                "arguments": {"city": "北京"}
            }
        """
        real_name = self.get_real_tool_name(tool_name)

        logger.info(f"🔧 [内置工具执行] 开始执行工具: {real_name}")
        logger.info(f"   参数: {arguments}")

        # 步骤1: 验证是否是内置工具（检查是否在注册表中）
        if not self.is_internal_tool(tool_name):
            error_msg = f"工具 {real_name} 不是内置工具（未在 TOOL_REGISTRY 中注册）"
            logger.error(f"❌ {error_msg}")
            logger.debug(f"   已注册的内置工具: {list(self.tool_registry.keys())}")
            return {
                "success": False,
                "error": error_msg,
                "tool_name": real_name,
                "arguments": arguments
            }

        # 步骤2: 获取工具实例
        tool_instance = self.get_tool_instance(tool_name)

        if not tool_instance:
            error_msg = f"工具 {real_name} 未注册或实例化失败"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "tool_name": real_name,
                "arguments": arguments
            }

        # 步骤3: 执行工具
        try:
            # 检查工具是否有 run 方法
            if not hasattr(tool_instance, 'run'):
                error_msg = f"工具 {real_name} 没有 run 方法"
                logger.error(f"❌ {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": real_name,
                    "arguments": arguments
                }

            # 执行工具（带超时控制）
            logger.info(f"⚙️ [内置工具执行] 调用 {real_name}.run() ...")

            try:
                # 尝试使用事件循环
                loop = asyncio.get_running_loop()

                # 在线程池中执行同步方法（避免阻塞事件循环）
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: tool_instance.run(**arguments)),
                    timeout=timeout
                )
            except RuntimeError:
                # 没有运行中的事件循环，直接同步调用
                result = tool_instance.run(**arguments)

            logger.info(f"✅ [内置工具执行] 工具 {real_name} 执行成功")

            return {
                "success": True,
                "result": result,
                "tool_name": real_name,
                "arguments": arguments
            }

        except asyncio.TimeoutError:
            error_msg = f"工具执行超时（{timeout}秒）"
            logger.error(f"⏱️ [内置工具执行] {error_msg}: {real_name}")
            return {
                "success": False,
                "error": error_msg,
                "tool_name": real_name,
                "arguments": arguments
            }

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            error_msg = str(e)

            logger.error(f"❌ [内置工具执行] 工具执行失败: {real_name}")
            logger.error(f"   错误: {error_msg}")
            logger.error(f"   详情: {error_detail}")

            return {
                "success": False,
                "error": error_msg,
                "detail": error_detail,
                "tool_name": real_name,
                "arguments": arguments
            }

    def execute_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = 120.0
    ) -> Dict[str, Any]:
        """同步执行内置工具（在没有事件循环的环境中使用）

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            timeout: 超时时间（秒）

        Returns:
            执行结果字典（格式同 execute）
        """
        real_name = self.get_real_tool_name(tool_name)

        logger.info(f"🔧 [内置工具执行-同步] 开始执行工具: {real_name}")

        # 验证是否是内置工具（检查是否在注册表中）
        if not self.is_internal_tool(tool_name):
            error_msg = f"工具 {real_name} 不是内置工具（未在 TOOL_REGISTRY 中注册）"
            logger.error(f"❌ {error_msg}")
            logger.debug(f"   已注册的内置工具: {list(self.tool_registry.keys())}")
            return {
                "success": False,
                "error": error_msg,
                "tool_name": real_name,
                "arguments": arguments
            }

        # 获取工具实例
        tool_instance = self.get_tool_instance(tool_name)

        if not tool_instance:
            error_msg = f"工具 {real_name} 未注册或实例化失败"
            return {
                "success": False,
                "error": error_msg,
                "tool_name": real_name,
                "arguments": arguments
            }

        # 执行工具
        try:
            result = tool_instance.run(**arguments)

            logger.info(f"✅ [内置工具执行-同步] 工具 {real_name} 执行成功")

            return {
                "success": True,
                "result": result,
                "tool_name": real_name,
                "arguments": arguments
            }

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()

            logger.error(f"❌ [内置工具执行-同步] 工具执行失败: {real_name}, 错误: {e}")

            return {
                "success": False,
                "error": str(e),
                "detail": error_detail,
                "tool_name": real_name,
                "arguments": arguments
            }

    def list_internal_tools(self) -> Dict[str, Dict[str, Any]]:
        """列出所有注册的内置工具

        Returns:
            工具字典: {tool_name: {description, parameters, ...}}
        """
        tools = {}

        for tool_name, tool_class in self.tool_registry.items():
            try:
                # 实例化获取详细信息
                tool_instance = tool_class()

                tools[tool_name] = {
                    "name": tool_name,
                    "full_name": f"{self.INTERNAL_TOOL_PREFIX}{tool_name}",
                    "description": getattr(tool_instance, 'description', ''),
                    "parameters": getattr(tool_instance, 'parameters', {}),
                    "require_confirmation": getattr(tool_instance, 'require_confirmation', True)
                }
            except Exception as e:
                logger.warning(f"⚠️ 无法加载工具信息: {tool_name}, 错误: {e}")
                continue

        return tools


# 全局单例
internal_tool_executor = InternalToolExecutor()


# 导出
__all__ = [
    'InternalToolExecutor',
    'internal_tool_executor'
]
