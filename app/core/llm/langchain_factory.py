"""
LangChain LLM 工厂

用于创建配置好的 LangChain LLM 实例
支持多种模型提供者（DashScope、OpenAI 等）
"""

import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LangChainLLMFactory:
    """
    LangChain LLM 工厂类

    功能：
    1. 创建配置好的 LangChain ChatOpenAI 实例
    2. 自动从环境变量或配置中获取 API Key
    3. 支持多种模型提供者（通过 base_url 切换）

    使用示例：
    ```python
    from app.core.llm.langchain_factory import create_langchain_llm

    # 使用 DashScope（默认）
    llm = create_langchain_llm(model="qwen-max", temperature=0.7)

    # 使用 OpenAI
    llm = create_langchain_llm(
        model="gpt-4",
        provider="openai",
        temperature=0.7
    )
    ```
    """

    # 支持的模型提供者配置
    PROVIDERS = {
        "dashscope": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "DASHSCOPE_API_KEY",
            "default_model": "qwen-max"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "default_model": "gpt-4"
        }
    }

    @classmethod
    def get_api_key(cls, provider: str = "dashscope") -> str:
        """
        获取 API Key

        优先级：
        1. 新模式：从分类配置获取（LLM_API_KEY, VLM_API_KEY, EMBEDDING_API_KEY）
        2. 传统模式：从环境变量获取（OPENAI_API_KEY, DASHSCOPE_API_KEY）
        3. 配置对象中的 API Key（从 app.config.settings）
        4. 返回 "dummy-key"（用于测试环境）

        Args:
            provider: 提供者名称（dashscope/openai）

        Returns:
            API Key 字符串

        Raises:
            ValueError: 如果 provider 不支持
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"不支持的 provider: {provider}。"
                f"可用的 provider: {list(cls.PROVIDERS.keys())}"
            )

        provider_config = cls.PROVIDERS[provider]
        api_key_env = provider_config["api_key_env"]

        # 1. 新模式：优先使用统一LLM配置（OpenAI兼容格式）
        llm_api_key = os.getenv("LLM_API_KEY")
        if llm_api_key:
            logger.debug(f"✅ 使用统一LLM配置: LLM_API_KEY")
            return llm_api_key

        # 2. 其他模式配置（向后兼容）
        vlm_api_key = os.getenv("VLM_API_KEY")
        if vlm_api_key:
            logger.debug(f"✅ 使用VLM配置: VLM_API_KEY")
            return vlm_api_key

        embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        if embedding_api_key:
            logger.debug(f"✅ 使用Embedding配置: EMBEDDING_API_KEY")
            return embedding_api_key

        # 3. 传统模式：从环境变量获取
        api_key = os.getenv(api_key_env)
        if api_key:
            logger.debug(f"✅ 使用传统模式获取 {api_key_env}")
            return api_key

        # 4. 从配置对象获取
        try:
            from app.config import settings
            llm_api_key = getattr(settings, "LLM_API_KEY", None)
            if llm_api_key:
                logger.debug("✅ 从配置对象获取 LLM_API_KEY")
                return llm_api_key
            vlm_api_key = getattr(settings, "VLM_API_KEY", None)
            if vlm_api_key:
                logger.debug("✅ 从配置对象获取 VLM_API_KEY")
                return vlm_api_key
            embedding_api_key = getattr(settings, "EMBEDDING_API_KEY", None)
            if embedding_api_key:
                logger.debug("✅ 从配置对象获取 EMBEDDING_API_KEY")
                return embedding_api_key

            if provider == "dashscope":
                api_key = settings.DASHSCOPE_API_KEY
            elif provider == "openai":
                api_key = settings.OPENAI_API_KEY

            if api_key:
                logger.debug(f"✅ 从配置对象获取 {api_key_env}")
                return api_key
        except Exception as e:
            logger.warning(f"⚠️ 无法从配置对象获取 API Key: {e}")

        # 5. 返回 dummy key（用于测试）
        logger.warning(f"⚠️ 未找到 {api_key_env}，使用 dummy-key")
        return "dummy-key"

    @classmethod
    def create_llm(
        cls,
        model: Optional[str] = None,
        provider: str = "dashscope",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> ChatOpenAI:
        """
        创建 LangChain ChatOpenAI 实例

        Args:
            model: 模型名称（如果不提供，从环境变量或使用默认模型）
            provider: 提供者名称（dashscope/openai）
            temperature: 温度参数（0.0-1.0）
            max_tokens: 最大 token 数
            api_key: API Key（如果不提供，自动从环境变量获取）
            base_url: API Base URL（如果不提供，自动从环境变量获取）
            **kwargs: 其他传递给 ChatOpenAI 的参数

        Returns:
            配置好的 ChatOpenAI 实例

        Raises:
            ValueError: 如果 provider 不支持

        示例：
        ```python
        # 使用环境变量配置（推荐）
        llm = LangChainLLMFactory.create_llm()

        # 指定模型
        llm = LangChainLLMFactory.create_llm(model="glm-4.6")

        # 创建传统模式 LLM
        llm = LangChainLLMFactory.create_llm(
            model="qwen-max",
            provider="dashscope"
        )
        ```
        """
        # 🚀 新模式：优先使用环境变量的分类配置
        env_model = os.getenv("LLM_MODEL") or os.getenv("LLM_MODEL_NEW")
        env_base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not env_model or not env_base_url:
            try:
                from app.config import settings
                if not env_model:
                    env_model = settings.LLM_MODEL or settings.LLM_MODEL_NEW
                if not env_base_url:
                    env_base_url = settings.LLM_BASE_URL or settings.OPENAI_BASE_URL or settings.OPENAI_API_BASE
            except Exception:
                pass

        if env_model and not model:
            model = env_model
            logger.info(f"✅ 使用环境变量模型: {model}")
        elif not model:
            # 降级到默认模型
            provider_config = cls.PROVIDERS[provider]
            model = provider_config["default_model"]
            logger.info(f"未指定模型，使用默认模型: {model}")

        # 🚀 新模式：优先使用环境变量的base_url
        if env_base_url and not base_url:
            base_url = env_base_url
            logger.info(f"✅ 使用环境变量API地址: {base_url}")
        elif not base_url:
            # 降级到provider配置
            provider_config = cls.PROVIDERS[provider]
            base_url = provider_config["base_url"]

        # 获取 API Key（会自动尝试新模式和传统模式）
        if not api_key:
            api_key = cls.get_api_key(provider)

        # 创建 LangChain ChatOpenAI 实例
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
                **kwargs
            )

            logger.info(
                f"✅ LangChain LLM 创建成功 | "
                f"Model: {model} | "
                f"Base URL: {base_url} | "
                f"Temperature: {temperature} | "
                f"Max Tokens: {max_tokens}"
            )

            return llm

        except Exception as e:
            logger.error(f"❌ LangChain LLM 创建失败: {e}", exc_info=True)
            raise


# 便捷函数（推荐使用）
def create_langchain_llm(
    model: Optional[str] = None,
    provider: str = "dashscope",
    temperature: float = 0.3,
    max_tokens: int = 2000,
    api_key: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    创建 LangChain LLM 实例（便捷函数）

    这是 LangChainLLMFactory.create_llm() 的快捷方式。

    Args:
        model: 模型名称（如果不提供，自动使用环境变量 LLM_MODEL）
        provider: 提供者（dashscope/openai，仅传统模式需要）
        temperature: 温度参数
        max_tokens: 最大 token 数
        api_key: API Key（可选，通常不需要手动指定）
        **kwargs: 其他参数

    Returns:
        配置好的 ChatOpenAI 实例

    示例：
    ```python
    from app.core.llm import create_langchain_llm

    # 🚀 新模式：使用环境变量配置（推荐）
    llm = create_langchain_llm()  # 自动读取 LLM_MODEL, LLM_API_KEY, LLM_BASE_URL

    # 🚀 新模式：指定模型，其他使用环境变量
    llm = create_langchain_llm(model="glm-4.6", temperature=0.1)

    # 📊 传统模式：指定服务商（向后兼容）
    llm = create_langchain_llm(model="qwen-max", provider="dashscope")
    ```
    """
    return LangChainLLMFactory.create_llm(
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        **kwargs
    )
