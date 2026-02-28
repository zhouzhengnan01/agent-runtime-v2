"""
灵活多服务商模型配置管理器
支持两种配置模式：
1. 统一配置模式：OPENAI_API_KEY + OPENAI_BASE_URL（所有模型来自同一服务商）
2. 分离配置模式：每种模型独立配置API Key和Base URL（混合多个服务商）
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_env_or_settings(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    try:
        from app.config import settings
        value = getattr(settings, name, None)
        if value:
            return str(value)
    except Exception:
        return None
    return None


@dataclass
class ModelServiceConfig:
    """单个模型服务配置"""
    api_key: str
    base_url: str
    model_name: str
    provider_name: str = ""


@dataclass
class MultiProviderConfig:
    """多服务商配置"""
    llm_config: ModelServiceConfig
    vlm_config: ModelServiceConfig
    embedding_config: ModelServiceConfig
    config_mode: str = "unified"  # unified | separated


class ModelConfigManager:
    """灵活多服务商模型配置管理器 - 支持统一配置和分离配置两种模式"""

    def __init__(self):
        self._config: Optional[MultiProviderConfig] = None

    def initialize(self):
        """
        启动时初始化模型配置 - 自动检测配置模式

        配置模式优先级：
        1. 分离配置模式：如果检测到LLM_API_KEY、VLM_API_KEY、EMBEDDING_API_KEY中任意一个
        2. 统一配置模式：使用OPENAI_API_KEY + OPENAI_BASE_URL
        3. Fallback：DashScope配置
        """
        logger.info("🚀 初始化灵活多服务商模型配置...")

        # 检测配置模式
        separated_keys = [
            _get_env_or_settings("LLM_API_KEY"),
            _get_env_or_settings("VLM_API_KEY"),
            _get_env_or_settings("EMBEDDING_API_KEY"),
        ]

        if any(separated_keys):
            # 分离配置模式
            self._config = self._init_separated_config()
            logger.info("📊 配置模式: 分离配置 (每种模型独立服务商)")
        else:
            # 统一配置模式 (向后兼容)
            self._config = self._init_unified_config()
            logger.info("📊 配置模式: 统一配置 (单一服务商)")

        self._log_config_summary()

    def _init_separated_config(self) -> MultiProviderConfig:
        """初始化分离配置模式 - 每种模型可以使用不同服务商"""

        # NOTE:
        # "separated config" is triggered when any of LLM_API_KEY/VLM_API_KEY/EMBEDDING_API_KEY is present.
        # In practice, many deployments only set **one** key (e.g. VLM) and expect it to work for all
        # model types. The previous implementation hard-required all three keys and failed fast with
        # "LLM_API_KEY not found", which breaks multimodal-only workloads (review, vision QA, etc.).
        #
        # Here we relax the constraint: missing keys fall back to any available key (LLM/VLM/OPENAI),
        # and missing base_url fields fall back across services as well. This keeps separated-config
        # flexible while still allowing explicit per-service overrides.

        # LLM配置
        llm_api_key = (
            _get_env_or_settings("LLM_API_KEY")
            or _get_env_or_settings("OPENAI_API_KEY")
            or _get_env_or_settings("VLM_API_KEY")
            or _get_env_or_settings("EMBEDDING_API_KEY")
        )
        llm_base_url = (
            _get_env_or_settings("LLM_BASE_URL")
            or _get_env_or_settings("OPENAI_BASE_URL")
            or _get_env_or_settings("OPENAI_API_BASE")
            or _get_env_or_settings("VLM_BASE_URL")
            or _get_env_or_settings("EMBEDDING_BASE_URL")
            or "https://api.openai.com/v1"
        )
        llm_model = (
            _get_env_or_settings("LLM_MODEL")
            or _get_env_or_settings("LLM_MODEL_NEW")
            or self._detect_default_model(llm_base_url, "llm")
        )
        llm_provider = self._detect_provider_name(llm_base_url)

        # VLM配置
        vlm_api_key = (
            _get_env_or_settings("VLM_API_KEY")
            or _get_env_or_settings("OPENAI_API_KEY")
            or _get_env_or_settings("LLM_API_KEY")
            or _get_env_or_settings("EMBEDDING_API_KEY")
        )
        vlm_base_url = (
            _get_env_or_settings("VLM_BASE_URL")
            or _get_env_or_settings("OPENAI_BASE_URL")
            or _get_env_or_settings("OPENAI_API_BASE")
            or _get_env_or_settings("LLM_BASE_URL")
            or _get_env_or_settings("EMBEDDING_BASE_URL")
            or "https://api.openai.com/v1"
        )
        vlm_model = (
            _get_env_or_settings("VLM_MODEL")
            or _get_env_or_settings("VLM_MODEL_NEW")
            or self._detect_default_model(vlm_base_url, "vlm")
        )
        vlm_provider = self._detect_provider_name(vlm_base_url)

        # Embedding配置
        embedding_api_key = (
            _get_env_or_settings("EMBEDDING_API_KEY")
            or _get_env_or_settings("OPENAI_API_KEY")
            or _get_env_or_settings("LLM_API_KEY")
            or _get_env_or_settings("VLM_API_KEY")
        )
        embedding_base_url = (
            _get_env_or_settings("EMBEDDING_BASE_URL")
            or _get_env_or_settings("OPENAI_BASE_URL")
            or _get_env_or_settings("OPENAI_API_BASE")
            or _get_env_or_settings("LLM_BASE_URL")
            or _get_env_or_settings("VLM_BASE_URL")
            or "https://api.openai.com/v1"
        )
        embedding_model = (
            _get_env_or_settings("EMBEDDING_MODEL")
            or _get_env_or_settings("EMBEDDING_MODEL_NEW")
            or self._detect_default_model(embedding_base_url, "embedding")
        )
        embedding_provider = self._detect_provider_name(embedding_base_url)

        # Final fallback: if some keys are still missing, reuse any available one.
        any_key = llm_api_key or vlm_api_key or embedding_api_key
        llm_api_key = llm_api_key or any_key
        vlm_api_key = vlm_api_key or any_key
        embedding_api_key = embedding_api_key or any_key

        if not (llm_api_key and vlm_api_key and embedding_api_key):
            raise ValueError("❌ 未配置API Key：请至少设置 OPENAI_API_KEY 或 任一(LLM_API_KEY/VLM_API_KEY/EMBEDDING_API_KEY)")

        return MultiProviderConfig(
            llm_config=ModelServiceConfig(
                api_key=llm_api_key,
                base_url=llm_base_url,
                model_name=llm_model,
                provider_name=llm_provider
            ),
            vlm_config=ModelServiceConfig(
                api_key=vlm_api_key,
                base_url=vlm_base_url,
                model_name=vlm_model,
                provider_name=vlm_provider
            ),
            embedding_config=ModelServiceConfig(
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                model_name=embedding_model,
                provider_name=embedding_provider
            ),
            config_mode="separated"
        )

    def _init_unified_config(self) -> MultiProviderConfig:
        """初始化统一配置模式 - 所有模型使用同一服务商（向后兼容）"""

        api_key = _get_env_or_settings("OPENAI_API_KEY")
        base_url = (
            _get_env_or_settings("OPENAI_BASE_URL")
            or _get_env_or_settings("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        if not api_key:
            # Fallback to DashScope if available
            if _get_env_or_settings("DASHSCOPE_API_KEY"):
                logger.info("🔄 OPENAI_API_KEY未配置，回退到DashScope")
                api_key = _get_env_or_settings("DASHSCOPE_API_KEY")
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            else:
                raise ValueError("❌ 未配置API Key：请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY")

        # 根据BASE_URL自动识别服务商和选择默认模型
        provider_name = self._detect_provider_name(base_url)
        llm_model = (
            _get_env_or_settings("LLM_MODEL")
            or _get_env_or_settings("LLM_MODEL_NEW")
            or self._detect_default_model(base_url, "llm")
        )
        vlm_model = (
            _get_env_or_settings("VLM_MODEL")
            or _get_env_or_settings("VLM_MODEL_NEW")
            or self._detect_default_model(base_url, "vlm")
        )
        embedding_model = (
            _get_env_or_settings("EMBEDDING_MODEL")
            or _get_env_or_settings("EMBEDDING_MODEL_NEW")
            or self._detect_default_model(base_url, "embedding")
        )

        # 所有模型使用相同的服务配置
        return MultiProviderConfig(
            llm_config=ModelServiceConfig(
                api_key=api_key,
                base_url=base_url,
                model_name=llm_model,
                provider_name=provider_name
            ),
            vlm_config=ModelServiceConfig(
                api_key=api_key,
                base_url=base_url,
                model_name=vlm_model,
                provider_name=provider_name
            ),
            embedding_config=ModelServiceConfig(
                api_key=api_key,
                base_url=base_url,
                model_name=embedding_model,
                provider_name=provider_name
            ),
            config_mode="unified"
        )

    def _detect_provider_name(self, base_url: str) -> str:
        """根据BASE_URL检测服务商名称"""
        if "bigmodel.cn" in base_url:
            return "智谱GLM"
        elif "deepseek.com" in base_url:
            return "DeepSeek"
        elif "dashscope.aliyuncs.com" in base_url:
            return "DashScope"
        elif "api.openai.com" in base_url:
            return "OpenAI"
        elif "api.anthropic.com" in base_url:
            return "Claude"
        elif "moonshot.cn" in base_url:
            return "月之暗面 Kimi"
        else:
            return "自定义OpenAI兼容服务"

    def _detect_default_model(self, base_url: str, model_type: str) -> str:
        """根据BASE_URL和模型类型检测默认模型"""
        provider_models = self._get_provider_models(base_url)
        return provider_models.get(f"{model_type}_model", "gpt-3.5-turbo")

    def _get_provider_models(self, base_url: str) -> Dict[str, str]:
        """获取服务商的默认模型配置"""
        if "bigmodel.cn" in base_url:
            return {
                "llm_model": "glm-4.6",
                "vlm_model": "glm-4.5v",
                "embedding_model": "text-embedding-ada-002"
            }
        elif "deepseek.com" in base_url:
            return {
                "llm_model": "deepseek-chat",
                "vlm_model": "deepseek-vl",
                "embedding_model": "text-embedding-ada-002"
            }
        elif "dashscope.aliyuncs.com" in base_url:
            return {
                "llm_model": "qwen-max",
                "vlm_model": "qwen-vl-max",
                "embedding_model": "text-embedding-v2"
            }
        elif "api.openai.com" in base_url:
            return {
                "llm_model": "gpt-4",
                "vlm_model": "gpt-4-vision-preview",
                "embedding_model": "text-embedding-ada-002"
            }
        elif "api.anthropic.com" in base_url:
            return {
                "llm_model": "claude-3-5-sonnet",
                "vlm_model": "claude-3-5-sonnet",
                "embedding_model": "text-embedding-ada-002"
            }
        elif "moonshot.cn" in base_url:
            return {
                "llm_model": "moonshot-v1-128k",
                "vlm_model": "moonshot-v1-128k",
                "embedding_model": "text-embedding-ada-002"
            }
        else:
            return {
                "llm_model": "gpt-3.5-turbo",
                "vlm_model": "gpt-4-vision-preview",
                "embedding_model": "text-embedding-ada-002"
            }

    def _log_config_summary(self):
        """记录配置摘要"""
        if self._config.config_mode == "separated":
            logger.info("📋 多服务商配置摘要:")
            logger.info(f"   💬 LLM: {self._config.llm_config.provider_name} / {self._config.llm_config.model_name}")
            logger.info(f"   👁️ VLM: {self._config.vlm_config.provider_name} / {self._config.vlm_config.model_name}")
            logger.info(f"   📊 Embedding: {self._config.embedding_config.provider_name} / {self._config.embedding_config.model_name}")
            logger.info("   🔀 每种模型可使用不同服务商，实现最优搭配")
        else:
            logger.info("📋 统一配置摘要:")
            logger.info(f"   🤖 服务商: {self._config.llm_config.provider_name}")
            logger.info(f"   💬 LLM: {self._config.llm_config.model_name}")
            logger.info(f"   👁️ VLM: {self._config.vlm_config.model_name}")
            logger.info(f"   📊 Embedding: {self._config.embedding_config.model_name}")
            logger.info(f"   🔗 API: {self._config.llm_config.base_url}")
            logger.info("   ✅ 一键切换：只需修改 OPENAI_API_KEY + OPENAI_BASE_URL")

    # 公共获取方法 - 保持向后兼容
    def get_config(self) -> MultiProviderConfig:
        """获取多服务商配置"""
        if not self._config:
            raise RuntimeError("模型配置未初始化，请先调用 initialize()")
        return self._config

    def get_model_config(self, model_type: str = "llm") -> ModelServiceConfig:
        """获取指定类型模型的配置"""
        if not self._config:
            raise RuntimeError("模型配置未初始化，请先调用 initialize()")

        if model_type == "llm":
            return self._config.llm_config
        elif model_type == "vlm":
            return self._config.vlm_config
        elif model_type == "embedding":
            return self._config.embedding_config
        else:
            return self._config.llm_config

    def get_openai_client_config(self, model_type: str = "llm") -> Dict[str, Any]:
        """获取OpenAI客户端配置 - 向后兼容方法"""
        model_config = self.get_model_config(model_type)

        return {
            "api_key": model_config.api_key,
            "base_url": model_config.base_url,
            "default_model": model_config.model_name
        }

    def get_switch_guide(self) -> str:
        """获取切换指南"""
        return """
🔄 灵活多服务商配置指南：

# ============ 模式1: 分离配置（推荐）- 每种模型独立服务商 ============

# LLM配置 - 智谱GLM（对中文友好，价格合理）
LLM_API_KEY=your_glm_api_key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4.6

# VLM配置 - OpenAI（视觉效果更好）
VLM_API_KEY=your_openai_key
VLM_BASE_URL=https://api.openai.com/v1
VLM_MODEL=gpt-4-vision-preview

# Embedding配置 - OpenAI（embedding专业）
EMBEDDING_API_KEY=your_openai_key
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-ada-002

# ============ 模式2: 统一配置（向后兼容）- 单一服务商 ============

# 智谱GLM
OPENAI_API_KEY=your_glm_api_key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_MODEL=glm-4.6
VLM_MODEL=glm-4.5v
EMBEDDING_MODEL=text-embedding-ada-002

# DeepSeek
OPENAI_API_KEY=your_deepseek_key
OPENAI_BASE_URL=https://api.deepseek.com/v1/
LLM_MODEL=deepseek-chat
VLM_MODEL=deepseek-vl

# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4
VLM_MODEL=gpt-4-vision-preview
EMBEDDING_MODEL=text-embedding-ada-002

💡 配置优势：
分离配置：
- 利用不同服务商的优势（如GLM的中文理解 + OpenAI的视觉能力）
- 降低成本（选择性价比最高的模型）
- 避免单点故障，提高可用性
- 灵活组合，实现最优性能

统一配置：
- 简单易用，只需两个变量
- 快速切换整体服务商
- 配置一致性好
        """


# 全局实例
model_config_manager = ModelConfigManager()


def get_model_config_manager() -> ModelConfigManager:
    """获取全局模型配置管理器"""
    return model_config_manager
