"""
工具配置服务 - 从数据库读取工具执行配置
"""
from typing import Dict, Optional
from sqlalchemy.orm import Session
from app.models.tool import Tool
import logging

logger = logging.getLogger(__name__)


class ToolConfigService:
    """工具配置服务（单例）"""

    _instance = None
    _config_cache: Dict[str, Dict] = {}  # 内存缓存

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_execution_config(self, tool_id: str, db: Session = None) -> Dict:
        """获取工具执行配置

        Args:
            tool_id: 工具ID
            db: 数据库会话（可选，如果不传则使用缓存）

        Returns:
            {
                "execution_mode": "sync|async|background",
                "estimated_time": 3,
                "timeout": 30
            }
        """
        # 1. 尝试从缓存获取
        if tool_id in self._config_cache:
            return self._config_cache[tool_id]

        # 2. 如果有数据库会话，从数据库读取
        if db:
            try:
                tool = db.query(Tool).filter(Tool.tool_id == tool_id).first()
                if tool:
                    config = {
                        "execution_mode": tool.execution_mode or "sync",
                        "estimated_time": tool.estimated_time or 3,
                        "timeout": tool.timeout or 30
                    }
                    # 更新缓存
                    self._config_cache[tool_id] = config
                    return config
            except Exception as e:
                logger.warning(f"从数据库读取工具配置失败: {tool_id}, error: {e}")

        # 3. 默认配置（兜底）
        default_config = self._get_default_config(tool_id)
        self._config_cache[tool_id] = default_config
        return default_config

    def _get_default_config(self, tool_id: str) -> Dict:
        """根据工具名称推断默认配置"""

        # 已知的慢速工具
        slow_tools = {
            "QueryPropertyAgg": {"execution_mode": "async", "estimated_time": 10, "timeout": 30},
            "QueryDeviceData": {"execution_mode": "async", "estimated_time": 10, "timeout": 30},
            "KnowledgeRetrieval": {"execution_mode": "async", "estimated_time": 8, "timeout": 30},
            "SearchDocuments": {"execution_mode": "async", "estimated_time": 8, "timeout": 30},
        }

        # 已知的后台任务
        background_tools = {
            "VideoAnalysis": {"execution_mode": "background", "estimated_time": 60, "timeout": 300},
            "LargeDataExport": {"execution_mode": "background", "estimated_time": 120, "timeout": 600},
            "BatchProcessing": {"execution_mode": "background", "estimated_time": 90, "timeout": 300},
            "ImageProcessing": {"execution_mode": "background", "estimated_time": 30, "timeout": 120},
        }

        if tool_id in slow_tools:
            return slow_tools[tool_id]

        if tool_id in background_tools:
            return background_tools[tool_id]

        # 根据名称模式推断
        if any(pattern in tool_id for pattern in ["Query", "Search", "Analysis", "Retrieval"]):
            return {"execution_mode": "async", "estimated_time": 8, "timeout": 30}

        if any(pattern in tool_id for pattern in ["Batch", "Export", "Video", "Large"]):
            return {"execution_mode": "background", "estimated_time": 60, "timeout": 300}

        # 默认：快速同步工具
        return {"execution_mode": "sync", "estimated_time": 2, "timeout": 10}

    def preload_configs(self, db: Session):
        """预加载所有工具配置（启动时调用）"""
        try:
            tools = db.query(Tool).filter(Tool.is_active == True).all()
            for tool in tools:
                self._config_cache[tool.tool_id] = {
                    "execution_mode": tool.execution_mode or "sync",
                    "estimated_time": tool.estimated_time or 3,
                    "timeout": tool.timeout or 30
                }
            logger.info(f"✅ 已预加载 {len(self._config_cache)} 个工具配置")
        except Exception as e:
            logger.error(f"预加载工具配置失败: {e}")

    def get_tool_config(self, tool_id: str, db: Session = None) -> Optional['Tool']:
        """获取工具配置对象

        Args:
            tool_id: 工具ID
            db: 数据库会话（可选）

        Returns:
            Tool对象，如果不存在则返回None
        """
        # 优先使用传入的会话，否则内部创建一个临时会话
        own_session = False
        session = db
        if session is None:
            try:
                from app.db.session import SessionLocal
                session = SessionLocal()
                own_session = True
            except Exception as e:
                logger.error(f"初始化临时数据库会话失败: {e}")
                return None

        try:
            tool = session.query(Tool).filter(Tool.tool_id == tool_id).first()
            return tool
        except Exception as e:
            logger.error(f"从数据库获取工具配置失败: {tool_id}, 错误: {e}")
            return None
        finally:
            if own_session and session:
                session.close()

    def clear_cache(self):
        """清空缓存（用于配置更新后刷新）"""
        self._config_cache.clear()
        logger.info("✅ 工具配置缓存已清空")


# 全局单例
tool_config_service = ToolConfigService()
