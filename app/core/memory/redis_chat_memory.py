"""
Redis聊天记忆管理器

专门负责创建和管理基于Redis的聊天历史记忆
用于LangChain智能体的对话上下文存储

作者：JetLinks Team
创建时间：2025-01-25
"""

import logging
from typing import Optional
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
import json

logger = logging.getLogger(__name__)


class ConversationBufferWindowMemory:
    """
    简单的对话缓冲窗口记忆实现

    替代旧版本 langchain.memory.ConversationBufferWindowMemory
    保留最近k轮对话用于上下文管理
    """

    def __init__(self, chat_history: BaseChatMessageHistory = None, memory_key: str = "history", return_messages: bool = False, k: int = 10, **kwargs):
        """
        初始化对话缓冲窗口

        Args:
            chat_history: 聊天历史实例（兼容旧版本参数名 chat_memory）
            memory_key: 记忆键名（兼容旧版本）
            return_messages: 是否返回消息（兼容旧版本）
            k: 保留的对话轮数
            **kwargs: 其他兼容参数
        """
        # 处理参数兼容性
        if chat_history is None and 'chat_memory' in kwargs:
            chat_history = kwargs.get('chat_memory')

        self.chat_history = chat_history
        self.k = k
        self.buffer = []
        self.memory_key = memory_key
        self.return_messages = return_messages

    def save_context(self, inputs: dict, outputs: dict):
        """保存对话上下文"""
        # 提取用户输入和AI输出
        user_input = inputs.get("input", "") or inputs.get("human", "") or str(inputs)
        ai_output = outputs.get("output", "") or outputs.get("text", "") or str(outputs)

        # 添加到聊天历史
        self.chat_history.add_user_message(user_input)
        self.chat_history.add_ai_message(ai_output)

        # 更新缓冲区
        self._update_buffer()

    def _update_buffer(self):
        """更新缓冲区，保持k轮对话"""
        messages = self.chat_history.messages

        # 取最近的2*k条消息（k轮对话，每轮包含用户和AI）
        recent_messages = messages[-2*self.k:]
        self.buffer = recent_messages

    def load_memory_variables(self, variables: dict = None) -> dict:
        """加载内存变量"""
        return {
            "history": self.buffer,
            "chat_history": self.buffer
        }

    def clear(self):
        """清空记忆"""
        self.chat_history.clear()
        self.buffer = []

    @property
    def memory_variables(self) -> list:
        """内存变量名称"""
        return ["history", "chat_history"]


class RedisChatMemoryManager:
    """
    Redis聊天记忆管理器

    功能：
    1. 创建基于Redis的聊天历史实例
    2. 配置对话缓冲窗口记忆（保留最近N轮对话）
    3. 自动处理Redis连接参数（支持有/无密码）
    4. 生成唯一的session key（agent_id_session_id格式）

    使用示例：
    ```python
    from app.core.memory.redis_chat_memory import RedisChatMemoryManager

    memory_manager = RedisChatMemoryManager()
    memory = memory_manager.create_memory(
        session_id="user_session_123",
        agent_id="agent_456",
        window_size=10
    )

    # 在LangChain中使用
    chain = LLMChain(llm=llm, memory=memory)
    ```

    架构：
    - Redis存储：完整的对话历史
    - 窗口记忆：只保留最近N轮对话用于上下文
    - Session Key格式：{agent_id}_{session_id}
    """

    def __init__(self):
        """
        初始化Redis记忆管理器

        自动从app.config.settings读取Redis配置：
        - REDIS_HOST: Redis服务器地址
        - REDIS_PORT: Redis端口
        - REDIS_DB: Redis数据库编号
        - REDIS_PASSWORD: Redis密码（可选）
        """
        self.settings = None
        self._load_settings()

    def _load_settings(self):
        """
        加载Redis配置

        从app.config导入settings
        如果导入失败，记录警告但不抛出异常
        """
        try:
            from app.config import settings
            self.settings = settings
            logger.debug(f"📋 Redis配置加载成功: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"❌ 无法加载Redis配置: {e}")
            raise

    def _build_redis_url(self) -> str:
        """
        构建Redis连接URL

        格式：
        - 有密码：redis://:password@host:port/db
        - 无密码：redis://host:port/db

        Returns:
            str: Redis连接URL

        示例：
        - redis://:3676860@1.14.62.40:26739/1
        - redis://localhost:6379/0
        """
        if self.settings.REDIS_PASSWORD:
            redis_url = (
                f"redis://:{self.settings.REDIS_PASSWORD}"
                f"@{self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}"
                f"/{self.settings.REDIS_DB}"
            )
        else:
            redis_url = (
                f"redis://{self.settings.REDIS_HOST}:{self.settings.REDIS_PORT}"
                f"/{self.settings.REDIS_DB}"
            )
        return redis_url

    def _generate_session_key(
        self,
        session_id: str,
        agent_id: Optional[str] = None
    ) -> str:
        """
        生成唯一的session key

        格式：
        - 有agent_id: {agent_id}_{session_id}
        - 无agent_id: {session_id}

        Args:
            session_id: 会话ID（必填）
            agent_id: 智能体ID（可选）

        Returns:
            str: 唯一的session key

        示例：
        - _generate_session_key("session_123", "agent_456") → "agent_456_session_123"
        - _generate_session_key("session_123") → "session_123"
        """
        if agent_id:
            return f"{agent_id}_{session_id}"
        return session_id

    def create_memory(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
        window_size: int = 10,
        memory_key: str = "chat_history",
        return_messages: bool = True
    ) -> Optional[ConversationBufferWindowMemory]:
        """
        创建基于Redis的对话记忆实例

        工作流程：
        1. 构建Redis连接URL
        2. 生成唯一的session key
        3. 创建RedisChatMessageHistory实例
        4. 封装为ConversationBufferWindowMemory
        5. 配置窗口大小和返回格式

        Args:
            session_id: 会话ID（必填）
            agent_id: 智能体ID（可选，用于多智能体场景）
            window_size: 对话窗口大小（保留最近N轮对话，默认10）
            memory_key: 记忆键名（默认"chat_history"）
            return_messages: 是否返回消息对象（默认True，LangChain需要）

        Returns:
            ConversationBufferWindowMemory: 配置好的记忆实例
            None: 如果创建失败

        示例：
        ```python
        memory = manager.create_memory(
            session_id="user_123",
            agent_id="tool_calling_agent",
            window_size=20  # 保留最近20轮对话
        )
        ```

        注意：
        - window_size只影响传递给LLM的上下文长度
        - Redis中会保存完整的历史记录
        - 如果Redis连接失败，返回None（调用方需要处理）
        """
        try:
            # 步骤1: 构建Redis URL
            redis_url = self._build_redis_url()
            logger.debug(f"🔗 Redis URL: {redis_url.replace(self.settings.REDIS_PASSWORD or '', '***')}")

            # 步骤2: 生成session key
            session_key = self._generate_session_key(session_id, agent_id)
            logger.debug(f"🔑 Session Key: {session_key}")

            # 步骤3: 创建Redis聊天历史实例
            message_history = RedisChatMessageHistory(
                url=redis_url,
                session_id=session_key
            )
            logger.debug(f"📝 RedisChatMessageHistory创建成功")

            # 步骤4: 创建对话缓冲窗口记忆
            memory = ConversationBufferWindowMemory(
                chat_memory=message_history,
                memory_key=memory_key,
                return_messages=return_messages,
                k=window_size  # 保留最近k轮对话
            )

            logger.info(
                f"✅ Redis聊天记忆初始化成功 | "
                f"Session: {session_key} | "
                f"Window: {window_size}轮"
            )

            return memory

        except Exception as e:
            logger.error(f"❌ Redis聊天记忆初始化失败: {e}", exc_info=True)
            logger.warning(f"⚠️ 智能体将在无记忆模式下运行")
            return None

    def clear_memory(
        self,
        session_id: str,
        agent_id: Optional[str] = None
    ) -> bool:
        """
        清空指定会话的聊天历史

        Args:
            session_id: 会话ID
            agent_id: 智能体ID（可选）

        Returns:
            bool: 是否清空成功

        示例：
        ```python
        success = manager.clear_memory("user_123", "agent_456")
        ```
        """
        try:
            redis_url = self._build_redis_url()
            session_key = self._generate_session_key(session_id, agent_id)

            message_history = RedisChatMessageHistory(
                url=redis_url,
                session_id=session_key
            )
            message_history.clear()

            logger.info(f"🗑️ Redis聊天历史已清空: {session_key}")
            return True

        except Exception as e:
            logger.error(f"❌ 清空Redis历史失败: {e}")
            return False

    def create_chat_history(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
        key_prefix: str = "chat"
    ) -> Optional[RedisChatMessageHistory]:
        """
        创建 RedisChatMessageHistory 实例（用于自定义用途）

        与 create_memory() 的区别：
        - create_memory(): 返回 ConversationBufferWindowMemory（用于LangChain）
        - create_chat_history(): 返回 RedisChatMessageHistory（用于直接读写消息）

        Args:
            session_id: 会话ID
            agent_id: 智能体ID（可选）
            key_prefix: Redis key前缀（默认"chat"，可自定义如"tool_history"）

        Returns:
            RedisChatMessageHistory: Redis聊天历史实例
            None: 如果创建失败

        使用场景：
        - 工具调用历史存储（tool_selector）
        - 自定义消息存储
        - 不需要窗口记忆的场景

        示例：
        ```python
        # 工具历史存储
        history = manager.create_chat_history(
            session_id="session_123",
            agent_id="agent_456",
            key_prefix="tool_history"
        )
        history.add_message(HumanMessage(content="..."))
        messages = history.messages
        ```
        """
        try:
            redis_url = self._build_redis_url()

            # 生成带前缀的 session key
            base_key = self._generate_session_key(session_id, agent_id)
            session_key = f"{key_prefix}_{base_key}" if key_prefix else base_key

            message_history = RedisChatMessageHistory(
                url=redis_url,
                session_id=session_key
            )

            logger.debug(f"✅ RedisChatMessageHistory创建成功: {session_key}")
            return message_history

        except Exception as e:
            logger.error(f"❌ 创建RedisChatMessageHistory失败: {e}", exc_info=True)
            return None


# 创建全局单例实例（可选）
redis_memory_manager = RedisChatMemoryManager()
