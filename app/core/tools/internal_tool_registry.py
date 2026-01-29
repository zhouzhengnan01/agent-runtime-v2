"""
内部工具注册表

用于区分内部工具和外部工具（来自 JetLinks 平台）
启动时从数据库加载所有内部工具到注册表
"""

import logging
from typing import Set, Dict, Any, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class InternalToolRegistry:
    """
    内部工具注册表

    功能：
    1. 启动时从数据库加载所有内部工具
    2. 提供快速查询接口判断工具是否为内部工具
    3. 用于 tool_router 区分内部/外部工具执行策略

    使用示例：
    ```python
    # 启动时加载
    await internal_tool_registry.load_from_database(db)

    # 判断工具类型
    if internal_tool_registry.is_internal_tool("knowledgeService"):
        # 内部工具：直接执行
    else:
        # 外部工具：需要 WebSocket 确认
    ```
    """

    def __init__(self):
        """初始化注册表"""
        # 存储内部工具的 tool_id
        self._internal_tool_ids: Set[str] = set()

        # 存储内部工具的详细信息（可选，用于调试）
        self._internal_tools: Dict[str, Dict[str, Any]] = {}

        # 标记是否已加载
        self._loaded: bool = False

    async def load_from_database(self, db: Session):
        """
        从数据库加载内部工具列表

        Args:
            db: 数据库会话
        """
        try:
            from app.models.tool import Tool

            # 查询所有内部工具（source='internal' 且 is_active=True）
            internal_tools = db.query(Tool).filter(
                Tool.source == "internal",
                Tool.is_active == True
            ).all()

            # 清空现有注册表
            self._internal_tool_ids.clear()
            self._internal_tools.clear()

            # 加载到注册表
            for tool in internal_tools:
                self._internal_tool_ids.add(tool.tool_id)
                self._internal_tools[tool.tool_id] = {
                    "id": tool.tool_id,
                    "name": tool.name,
                    "description": tool.description,
                    "is_async": tool.is_async
                }

            self._loaded = True

            logger.info(
                f"✅ 内部工具注册表加载完成 | "
                f"共 {len(self._internal_tool_ids)} 个内部工具: "
                f"{list(self._internal_tool_ids)}"
            )

        except Exception as e:
            logger.error(f"❌ 加载内部工具注册表失败: {e}", exc_info=True)
            # 即使加载失败，也标记为已加载（避免重复尝试）
            self._loaded = True

    def is_internal_tool(self, tool_id: str) -> bool:
        """
        判断工具是否为内部工具

        Args:
            tool_id: 工具ID（可以是 tool.name 或 tool.tool_id）
                    支持 _ai_service_# 前缀的工具名（如 _ai_service_#chatbi）

        Returns:
            True: 内部工具
            False: 外部工具

        示例：
        ```python
        if internal_tool_registry.is_internal_tool("knowledgeService"):
            print("这是内部工具")
        if internal_tool_registry.is_internal_tool("_ai_service_#chatbi"):
            print("这也是内部工具")
        ```
        """
        if not self._loaded:
            logger.warning(
                "⚠️ 内部工具注册表尚未加载，默认视为外部工具。"
                "请在应用启动时调用 load_from_database()"
            )
            return False

        # 1. 直接匹配 tool_id
        if tool_id in self._internal_tool_ids:
            return True

        # 2. 如果工具名以 _ai_service_# 开头，去掉前缀后再匹配
        #    例如: _ai_service_#chatbi -> chatbi
        if tool_id.startswith("_ai_service_#"):
            actual_tool_id = tool_id.replace("_ai_service_#", "", 1)
            return actual_tool_id in self._internal_tool_ids

        return False

    def get_internal_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        获取内部工具的详细信息

        Args:
            tool_id: 工具ID

        Returns:
            工具信息字典，如果不存在则返回 None
        """
        return self._internal_tools.get(tool_id)

    def get_all_internal_tools(self) -> Set[str]:
        """
        获取所有内部工具的ID列表

        Returns:
            内部工具ID集合
        """
        return self._internal_tool_ids.copy()

    def register_tool(self, tool_id: str, tool_info: Optional[Dict[str, Any]] = None):
        """
        手动注册内部工具（用于测试或动态注册）

        Args:
            tool_id: 工具ID
            tool_info: 工具信息（可选）
        """
        self._internal_tool_ids.add(tool_id)
        if tool_info:
            self._internal_tools[tool_id] = tool_info
        logger.info(f"📝 手动注册内部工具: {tool_id}")

    def unregister_tool(self, tool_id: str):
        """
        取消注册内部工具

        Args:
            tool_id: 工具ID
        """
        self._internal_tool_ids.discard(tool_id)
        self._internal_tools.pop(tool_id, None)
        logger.info(f"🗑️ 取消注册内部工具: {tool_id}")

    def is_loaded(self) -> bool:
        """
        检查注册表是否已加载

        Returns:
            True: 已加载
            False: 未加载
        """
        return self._loaded


# 创建全局单例
internal_tool_registry = InternalToolRegistry()
