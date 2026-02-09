"""
基于 LangChain 的模板智能体基类

所有 LangChain 模板都内置 LangChainCognitiveAgent 作为认知引擎
"""

# ═══════════════════════════════════════════════════════════════
# 标准库导入
# ═══════════════════════════════════════════════════════════════
from abc import abstractmethod
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import uuid
import logging

# ═══════════════════════════════════════════════════════════════
# 项目内部导入
# ═══════════════════════════════════════════════════════════════
from app.core.agents.cognitive_agent import LangChainCognitiveAgent
from app.services.session_markdown_logger import SessionMarkdownLogger

logger = logging.getLogger(__name__)


class BaseTemplateAgent:
    """
    LangChain 模板智能体基类

    特点:
    1. 完全基于 LangChain 框架
    2. 不依赖 qwen-agent (CognitiveAgent)
    3. 提供与 BaseTemplateAgent 兼容的接口
    4. 支持异步流式处理
    5. 内置认知能力支持（记忆、压缩、多模态）

    继承关系:
    BaseLangChainTemplateAgent (独立) → 具体的 LangChain 模板

    示例:
    - ToolCallingAgent (工具调用模板)
    - 未来的其他 LangChain 模板...

    子类必须实现:
    - _initialize(): 子类初始化
    - process_message(): 处理消息（非流式）
    - process_message_stream(): 处理消息（流式）
    - execute_task(): 执行特定任务
    - get_capabilities(): 返回能力列表
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 LangChain 模板智能体

        Args:
            name: 智能体名称
            description: 智能体描述
            version: 版本号
            config: 配置参数，包括:
                - model: 模型名称
                - system_prompt: 系统提示词
                - tools: 工具列表
                - handler: LangChainJAIPHandler 引用（用于工具确认）
                - temperature: 温度参数
                - max_tokens: 最大token数
                - enable_memory: 启用记忆
                - enable_compression: 启用压缩
                - enable_multimodal: 启用多模态
        """
        # ═══════════════════════════════════════════════════════════
        # 基本信息
        # ═══════════════════════════════════════════════════════════
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.version = version

        # ═══════════════════════════════════════════════════════════
        # 状态管理
        # ═══════════════════════════════════════════════════════════
        self.status = "initialized"
        self.created_at = datetime.now()
        self.last_active = datetime.now()

        # ═══════════════════════════════════════════════════════════
        # 会话管理
        # ═══════════════════════════════════════════════════════════
        self.session_history = []
        self.context = {}

        # ═══════════════════════════════════════════════════════════
        # 配置
        # ═══════════════════════════════════════════════════════════
        self.config = config or {}
        self.session_id = self.config.get("session_id")
        self.config_agent_id = self.config.get("agent_id")
        self.session_md_enabled = bool(self.config.get("session_md_enabled", False))
        self.session_md_dir = self.config.get("session_md_dir", "storage/session_rd")
        self.session_md_logger = (
            SessionMarkdownLogger(self.session_md_dir) if self.session_md_enabled else None
        )

        logger.info(f"初始化 BaseLangChainTemplateAgent: {self.name}")
        logger.info(f"  版本: {self.version}")
        logger.info(f"  Agent ID: {self.agent_id}")

        # ═══════════════════════════════════════════════════════════
        # 初始化 LangChain 认知引擎（核心组件）
        # ═══════════════════════════════════════════════════════════
        logger.info(f"  创建 LangChain 认知引擎...")

        self.cognitive_engine = LangChainCognitiveAgent(self.config)

        # 从认知引擎复制工具列表
        self.function_list = self.cognitive_engine.function_list

        logger.info(f"  ✅ 认知引擎初始化完成")
        logger.info(f"    工具数量: {len(self.function_list)}")
        logger.info(f"    记忆能力: {self.cognitive_engine.enable_memory}")
        logger.info(f"    压缩能力: {self.cognitive_engine.enable_compression}")
        logger.info(f"    多模态: {self.cognitive_engine.enable_multimodal}")

        # 调用子类初始化
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """
        子类初始化方法（必须实现）

        子类在此方法中进行特定的初始化操作，例如：
        - 创建 LangChain 组件
        - 初始化工具列表
        - 设置 handler 引用
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        处理用户消息（非流式）

        Args:
            message: 用户消息
            context: 上下文信息，可能包含:
                - session_id: 会话ID
                - user_id: 用户ID
                - file_id: 文件ID
                - metadata: 其他元数据

        Returns:
            str: 响应内容
        """
        pass

    @abstractmethod
    async def process_message_stream(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_params=None
    ) -> AsyncIterator[str]:
        """
        处理用户消息（流式）

        Args:
            message: 用户消息
            context: 上下文信息

        Yields:
            str: 响应片段
        """
        pass


    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        获取智能体能力列表

        Returns:
            List[str]: 能力列表，例如:
                - 'tool_calling': 工具调用
                - 'memory': 记忆管理
                - 'multimodal': 多模态处理
                - 'streaming': 流式输出
        """
        pass

    # ═══════════════════════════════════════════════════════════
    # 通用工具方法
    # ═══════════════════════════════════════════════════════════

    def update_status(self, status: str):
        """
        更新状态

        Args:
            status: 新状态 (initialized, running, idle, error, stopped)
        """
        self.status = status
        self.last_active = datetime.now()
        logger.info(f"[{self.name}] 状态更新: {status}")

    def add_to_history(self, message: str, response: str):
        """
        添加到会话历史

        Args:
            message: 用户消息
            response: 智能体响应
        """
        self.session_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        })

    def _md_append(self, title: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not self.session_md_logger or not self.session_id:
            return
        self.session_md_logger.append(
            session_id=self.session_id,
            title=title,
            content=content,
            meta=meta,
            agent_id=self.config_agent_id or self.agent_id,
        )

    def update_context(self, key: str, value: Any):
        """
        更新上下文

        Args:
            key: 键
            value: 值
        """
        self.context[key] = value

    def get_info(self) -> Dict[str, Any]:
        """
        获取智能体信息

        Returns:
            Dict: 智能体信息
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "capabilities": self.get_capabilities(),
            "function_count": len(self.function_list)
        }

    def __str__(self) -> str:
        """字符串表示"""
        return f"<BaseLangChainTemplateAgent: {self.name} (v{self.version})>"

    def __repr__(self) -> str:
        """详细表示"""
        return f"<BaseLangChainTemplateAgent(name='{self.name}', id='{self.agent_id}', status='{self.status}')>"
