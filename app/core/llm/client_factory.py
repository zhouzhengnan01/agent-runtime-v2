"""
灵活多服务商客户端工厂
支持统一配置和分离配置两种模式，自动为每种模型选择最合适的服务商
"""

import logging
from typing import Any
from openai import OpenAI, AsyncOpenAI

from .model_config_manager import get_model_config_manager

logger = logging.getLogger(__name__)


class UnifiedClientFactory:
    """灵活多服务商客户端工厂 - 支持统一配置和分离配置模式"""

    @staticmethod
    def create_openai_client(model_type: str = "llm", async_client: bool = False) -> Any:
        """
        创建OpenAI兼容客户端 - 支持多服务商分离配置

        Args:
            model_type: 模型类型 (llm, vlm, embedding)
            async_client: 是否创建异步客户端

        Returns:
            OpenAI 客户端实例
        """
        config_manager = get_model_config_manager()
        model_config = config_manager.get_model_config(model_type)

        client_class = AsyncOpenAI if async_client else OpenAI

        client = client_class(
            api_key=model_config.api_key,
            base_url=model_config.base_url
        )

        logger.debug(f"创建 {'异步' if async_client else '同步'} {model_config.provider_name} 客户端: {model_type}")
        return client

    @staticmethod
    def get_default_model(model_type: str = "llm") -> str:
        """
        获取默认模型名称 - 支持多服务商分离配置

        Args:
            model_type: 模型类型 (llm, vlm, embedding)

        Returns:
            模型名称
        """
        config_manager = get_model_config_manager()
        model_config = config_manager.get_model_config(model_type)
        return model_config.model_name

    @staticmethod
    def create_langchain_llm(**kwargs):
        """创建LangChain LLM实例 - 支持多服务商分离配置"""
        from langchain_openai import ChatOpenAI

        config_manager = get_model_config_manager()

        # 获取指定模型类型的配置（默认LLM，可指定VLM）
        model_type = "vlm" if kwargs.get("use_vision", False) else "llm"
        model_config = config_manager.get_model_config(model_type)

        # 合并配置和用户参数，但不要让None值覆盖默认配置
        final_config = {
            "api_key": model_config.api_key,
            "base_url": model_config.base_url,
            "model": model_config.model_name,
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        # 只更新非None的用户参数
        for key, value in kwargs.items():
            if key != "use_vision" and value is not None:  # 排除use_vision参数
                final_config[key] = value

        logger.info(f"🤖 创建LangChain LLM: {model_config.provider_name} / {final_config['model']}")

        return ChatOpenAI(**final_config)


# 便捷函数 - 支持多服务商分离配置
def create_llm_client(async_client: bool = False):
    """创建LLM客户端"""
    return UnifiedClientFactory.create_openai_client("llm", async_client)

def create_vlm_client(async_client: bool = False):
    """创建VLM客户端"""
    return UnifiedClientFactory.create_openai_client("vlm", async_client)

def create_embedding_client(async_client: bool = False):
    """创建Embedding客户端"""
    return UnifiedClientFactory.create_openai_client("embedding", async_client)

def get_default_llm_model() -> str:
    """获取默认LLM模型"""
    return UnifiedClientFactory.get_default_model("llm")

def get_default_vlm_model() -> str:
    """获取默认VLM模型"""
    return UnifiedClientFactory.get_default_model("vlm")

def get_default_embedding_model() -> str:
    """获取默认Embedding模型"""
    return UnifiedClientFactory.get_default_model("embedding")

def create_unified_langchain_llm(**kwargs):
    """创建统一配置的LangChain LLM - 支持多服务商分离配置"""
    return UnifiedClientFactory.create_langchain_llm(**kwargs)