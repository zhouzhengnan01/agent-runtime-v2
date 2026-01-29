"""
记忆管理模块

提供两种记忆管理器：
1. UnifiedMemoryManager: PostgreSQL + (Milvus/pgvector)（生产级，用于长期记忆和向量检索）
2. RedisChatMemoryManager: Redis聊天记忆管理器（用于LangChain对话上下文）
"""

from app.core.memory.unified_manager import UnifiedMemoryManager, unified_memory
from app.core.memory.redis_chat_memory import RedisChatMemoryManager, redis_memory_manager

__all__ = [
    "UnifiedMemoryManager",
    "unified_memory",
    "RedisChatMemoryManager",
    "redis_memory_manager"
]
