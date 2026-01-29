"""
LLM 调用模块
整合了灵活多服务商配置管理功能
"""

from app.core.llm.client import LLMClient
from app.core.llm.langchain_factory import LangChainLLMFactory, create_langchain_llm

# 多服务商配置管理
from .model_config_manager import (
    ModelConfigManager,
    ModelServiceConfig,
    MultiProviderConfig,
    get_model_config_manager
)
from app.core.llm.client_factory import (
    UnifiedClientFactory,
    create_llm_client,
    create_vlm_client,
    create_embedding_client,
    get_default_llm_model,
    get_default_vlm_model,
    get_default_embedding_model,
    create_unified_langchain_llm
)

__all__ = [
    # 核心功能
    "LLMClient",
    "LangChainLLMFactory",
    "create_langchain_llm",

    # 多服务商配置管理
    "ModelConfigManager",
    "ModelServiceConfig",
    "MultiProviderConfig",
    "get_model_config_manager",
    "UnifiedClientFactory",
    "create_llm_client",
    "create_vlm_client",
    "create_embedding_client",
    "get_default_llm_model",
    "get_default_vlm_model",
    "get_default_embedding_model",
    "create_unified_langchain_llm"
]