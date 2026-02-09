#!/usr/bin/env python3
"""
视频巡检智能体模板

功能：
1. 支持图片/视频的多模态分析
2. 专注于简单直接的分析模式（不支持工具调用）
3. 系统提示词完全由数据库配置
4. 支持流式和非流式输出

设计理念：
- 单一职责：只做视频/图片分析，不涉及复杂工具链
- 如需工具调用，请使用 ToolCallingAgent 模板

作者: JetLinks Team
版本: 3.2.0
"""

import logging
import base64
import json
import os
from typing import Dict, Any, Optional, List, AsyncIterator
from mimetypes import guess_type

import requests

from app.core.llm.client import LLMClient
from app.core.agents.template_agent.base_template import BaseTemplateAgent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# VideoInspectionAgent 主类
# ═══════════════════════════════════════════════════════════════

class VideoInspectionAgent(BaseTemplateAgent):
    """
    视频巡检智能体

    专注于视频/图片的简单分析，不支持工具调用模式。

    工作模式：
    - 直接调用LLM分析图片/视频内容
    - 支持多张图片的批量分析
    - 自动选择合适的视觉模型（VLM）

    系统提示词：
    - 完全由数据库配置控制
    - 支持通过 context.additional_prompt 动态补充
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化视频巡检智能体

        Args:
            config: 智能体配置（从数据库加载）
        """
        self._config = config

        super().__init__(
            name=config.get('name', 'VideoInspectionAgent'),
            description=config.get('description', '专业的视频巡检AI助手'),
            version=config.get('version', '3.1.0'),
            config=config
        )

    def _initialize(self):
        """从配置中提取参数"""
        self.system_prompt = self._config.get('system_prompt', '你是一个专业的视频巡检AI助手。')
        self.temperature = float(self._config.get('temperature', 0.3))
        self.max_tokens = int(self._config.get('max_tokens', 2000))
        self.model = self._config.get('model', 'qwen-plus')

        logger.info(f"✅ [VideoInspection] 初始化完成")
        logger.info(f"   系统提示词: {self.system_prompt[:50]}...")
        logger.info(f"   温度: {self.temperature}, 最大tokens: {self.max_tokens}")

    # ═══════════════════════════════════════════════════════════════
    # 公共接口：消息处理
    # ═══════════════════════════════════════════════════════════════

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        处理用户消息（非流式）

        VideoInspection专注于简单分析，不支持工具调用

        Args:
            message: 用户消息
            context: 上下文信息

        Returns:
            完整响应
        """
        try:
            logger.info("🔄 [VideoInspection] 开始简单分析")
            self._md_append("user", message)
            response = await self._simple_analysis(message, context)
            if response:
                self._md_append("assistant", response)
            return response

        except Exception as e:
            logger.error(f"❌ [VideoInspection] 处理失败: {e}", exc_info=True)
            return f"处理失败: {str(e)}"

    async def process_message_stream(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_params: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        处理用户消息（流式）

        VideoInspection专注于简单分析，不支持工具调用

        Args:
            message: 用户消息
            context: 上下文信息
            original_params: 原始参数（暂未使用）

        Yields:
            响应片段
        """
        try:
            logger.info("🔄 [VideoInspection] 开始流式简单分析")
            self._md_append("user", message)
            full_response = ""
            async for chunk in self._simple_analysis_stream(message, context):
                if chunk:
                    full_response += chunk
                yield chunk
            if full_response:
                self._md_append("assistant", full_response)

        except Exception as e:
            logger.error(f"❌ [VideoInspection] 流式处理失败: {e}", exc_info=True)
            yield f"处理失败: {str(e)}"

    # ═══════════════════════════════════════════════════════════════
    # 简单分析模式
    # ═══════════════════════════════════════════════════════════════

    async def _simple_analysis(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """简单分析（非流式）"""
        chunks = []
        async for chunk in self._simple_analysis_stream(message, context):
            chunks.append(chunk)
        return "".join(chunks)

    async def _simple_analysis_stream(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        简单分析（流式）

        核心流程：
        1. 提取图片URL
        2. 构建系统提示词（数据库基础 + 格式要求）
        3. 选择模型（有图片用VLM，无图片用LLM）
        4. 构建消息（多模态 or 纯文本）
        5. 调用LLM流式返回
        """
        try:
            context = context or {}

            # 1. 提取图片URL
            image_urls = self._extract_image_urls(context)
            has_image = bool(image_urls)

            logger.info(f"🔍 [SimpleAnalysis] 图片数量: {len(image_urls) if image_urls else 0}")

            # 2. 构建系统提示词
            system_prompt = self._build_system_prompt(has_image, context)

            # 3. 选择模型
            model = self._select_model(has_image)
            logger.info(f"🤖 [SimpleAnalysis] 选择模型: {model}")

            # 4. 构建消息
            messages = await self._build_messages(system_prompt, message, image_urls)

            # 5. 调用LLM
            llm_client = self._create_llm_client()
            async for chunk in llm_client.chat_stream(
                messages=messages,
                model=model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ):
                if chunk:
                    yield chunk

            logger.info("✅ [SimpleAnalysis] 完成")

        except Exception as e:
            logger.error(f"❌ [SimpleAnalysis] 失败: {e}", exc_info=True)
            yield f"分析失败: {str(e)}"

    # ═══════════════════════════════════════════════════════════════
    # 辅助方法：数据提取与构建
    # ═══════════════════════════════════════════════════════════════

    def _extract_image_urls(self, context: Dict[str, Any]) -> Optional[List[str]]:
        """
        从context中提取图片URL列表

        支持两种格式：
        - image_urls: 数组（推荐）
        - image_url: 单个字符串（兼容）

        Returns:
            图片URL列表，无图片返回None
        """
        if "image_urls" in context:
            urls = context["image_urls"]
            # 兼容单个字符串
            return [urls] if isinstance(urls, str) else urls

        if "image_url" in context:
            return [context["image_url"]]

        return None

    def _build_system_prompt(self, has_image: bool, context: Dict[str, Any]) -> str:
        """
        构建系统提示词

        来源优先级：
        1. 数据库配置的 system_prompt (基础)
        2. context 中的 additional_prompt (动态补充)
        3. 不再追加硬编码的格式要求

        Args:
            has_image: 是否有图片
            context: 上下文

        Returns:
            完整系统提示词
        """
        # 1. 从数据库加载的基础提示词
        prompt = self.system_prompt

        # 2. 如果 context 中有额外的提示词，追加上去
        if context and context.get("additional_prompt"):
            additional = context.get("additional_prompt")
            prompt += f"\n\n{additional}"
            logger.info(f"➕ [SystemPrompt] 追加额外提示词: {additional[:50]}...")

        logger.info(f"📝 [SystemPrompt] 最终提示词长度: {len(prompt)} 字符")

        return prompt

    def _select_model(self, has_image: bool) -> str:
        """
        选择合适的模型

        Args:
            has_image: 是否有图片

        Returns:
            模型名称
        """
        if has_image:
            return os.getenv("VLM_MODEL", "qwen-vl-max")
        else:
            return os.getenv("LLM_MODEL", "qwen-plus")

    async def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        image_urls: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        构建消息列表

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            image_urls: 图片URL列表

        Returns:
            消息列表 [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        messages = [{"role": "system", "content": system_prompt}]

        if image_urls:
            # 多模态消息
            user_content = await self._build_multimodal_content(image_urls, user_message)
        else:
            # 纯文本消息
            user_content = user_message

        messages.append({"role": "user", "content": user_content})
        return messages

    async def _build_multimodal_content(self, image_urls: List[str], text: str) -> List[Dict[str, Any]]:
        """
        构建多模态内容

        格式：[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}, ...]

        Args:
            image_urls: 图片URL列表
            text: 文本内容

        Returns:
            多模态内容列表
        """
        try:
            content = [{"type": "text", "text": text or "请分析这些图片"}]

            for idx, url in enumerate(image_urls, 1):
                logger.info(f"📥 [Multimodal] 下载图片 {idx}/{len(image_urls)}: {url[:100]}...")

                try:
                    # 下载图片
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    # 转换为base64
                    content_type = guess_type(url)[0] or 'image/jpeg'
                    image_base64 = base64.b64encode(response.content).decode('utf-8')

                    # 添加到content
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{image_base64}"
                        }
                    })

                    logger.info(f"✅ [Multimodal] 图片 {idx} 完成，大小: {len(response.content)} 字节")

                except requests.RequestException as e:
                    logger.warning(f"⚠️ [Multimodal] 图片 {idx} 下载失败: {e}")
                    content[0]["text"] += f"\n\n注意：图片 {idx} 加载失败 ({url[:50]}...)"

                except Exception as e:
                    logger.error(f"❌ [Multimodal] 图片 {idx} 处理失败: {e}")
                    content[0]["text"] += f"\n\n注意：图片 {idx} 处理失败"

            logger.info(f"✅ [Multimodal] 构建完成，共 {len(content)-1} 张图片")
            return content

        except Exception as e:
            logger.error(f"❌ [Multimodal] 构建失败: {e}", exc_info=True)
            return [{"type": "text", "text": f"{text}\n\n注意：图片加载失败 ({e})"}]

    def _create_llm_client(self) -> LLMClient:
        """创建LLM客户端"""
        try:
            return LLMClient(provider="unified")
        except Exception as e:
            logger.warning(f"⚠️ [VideoInspection] 统一客户端失败，回退到qwen: {e}")
            return LLMClient(provider="qwen")

    # ═══════════════════════════════════════════════════════════════
    # 元信息
    # ═══════════════════════════════════════════════════════════════

    def get_capabilities(self) -> List[str]:
        """返回智能体能力列表"""
        return [
            "视频分析",
            "图片分析",
            "安全检测",
            "巡检报告生成",
            "工具调用",
            "异步执行",
            "流式输出",
            "非流式输出",
            "自定义JSON格式输出",
            "Schema验证",
            "智能JSON修复",
            "图片下载处理",
            "多模态分析",
            "认知引擎集成"
        ]
