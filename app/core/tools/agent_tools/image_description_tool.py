"""
图片描述工具 - 使用 glm-4.5v 模型分析图片内容
"""
import logging
from typing import Any, Dict, Optional
import os

from app.core.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool("ImageDescription")
class ImageDescriptionTool(BaseTool):
    """图片描述工具 - 调用 glm-4.5v 模型分析图片"""

    name: str = "ImageDescription"
    description: str = "分析图片内容并生成详细描述。可以识别图片中的物体、场景、人物、文字等信息。"

    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "图片URL地址（支持 http/https 链接）"
            },
            "prompt": {
                "type": "string",
                "description": "分析提示（可选）。例如：'描述这张图片'、'图片中有什么物体'、'识别图片中的文字'",
                "default": "请详细描述这张图片的内容"
            }
        },
        "required": ["image_url"]
    }

    output: Dict[str, Any] = {
        "type": "object",
        "properties": [
            {
                "id": "status",
                "name": "执行状态",
                "valueType": {"type": "string"},
                "description": "执行状态（success/error）"
            },
            {
                "id": "description",
                "name": "图片描述",
                "valueType": {"type": "string"},
                "description": "图片内容描述"
            },
            {
                "id": "image_url",
                "name": "图片URL",
                "valueType": {"type": "string"},
                "description": "原始图片URL"
            },
            {
                "id": "model",
                "name": "模型名称",
                "valueType": {"type": "string"},
                "description": "使用的模型名称"
            },
            {
                "id": "prompt",
                "name": "分析提示",
                "valueType": {"type": "string"},
                "description": "使用的分析提示词"
            },
            {
                "id": "error",
                "name": "错误信息",
                "valueType": {"type": "string"},
                "description": "错误信息（如果失败）"
            }
        ]
    }

    require_confirmation: bool = False  # 不需要确认

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化工具"""
        super().__init__(config)

        # 延迟初始化配置，在运行时才加载
        self.api_key = None
        self.model_name = None
        self.base_url = None
        self.provider_name = None

        logger.info("🎯 [图片描述工具] 初始化完成，配置将在运行时加载")

    def _ensure_config(self):
        """确保配置已加载"""
        if not self.api_key:
            from app.core.llm import get_model_config_manager

            model_config_manager = get_model_config_manager()
            if not model_config_manager._config:
                model_config_manager.initialize()

            config = model_config_manager._config
            self.api_key = config.vlm_config.api_key
            self.model_name = config.vlm_config.model_name
            self.base_url = config.vlm_config.base_url
            self.provider_name = config.vlm_config.provider_name

            logger.info(f"🎯 [图片描述工具] 配置已加载: {self.model_name} ({self.provider_name})")

    def run(self, image_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        执行图片分析

        Args:
            image_url: 图片URL地址
            prompt: 分析提示（可选）
            **kwargs: 其他参数

        Returns:
            {
                "success": True/False,
                "description": "图片描述内容",
                "image_url": "原始图片URL",
                "model": "使用的模型名称",
                "error": "错误信息"（如果失败）
            }
        """
        try:
            # 确保配置已加载
            self._ensure_config()

            if not image_url:
                return {
                    "success": False,
                    "error": "图片URL不能为空"
                }

            # 验证URL格式
            if not image_url.startswith(('http://', 'https://')):
                return {
                    "success": False,
                    "error": "图片URL必须是 http:// 或 https:// 开头的有效链接"
                }

            # 默认提示
            if not prompt:
                prompt = "请详细描述这张图片的内容，包括场景、物体、人物、颜色、文字等信息。"

            logger.info(f"🖼️ [图片描述工具] 开始分析图片: {image_url[:100]}")
            logger.info(f"   提示: {prompt}")

            # 调用 VLM 模型
            description = self._call_vlm(image_url, prompt)

            return {
                "success": True,
                "description": description,
                "image_url": image_url,
                "model": self.model_name,
                "prompt": prompt
            }

        except Exception as e:
            logger.error(f"❌ [图片描述工具] 分析失败: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "image_url": image_url
            }

    def _call_vlm(self, image_url: str, prompt: str) -> str:
        """
        调用 VLM 模型进行图片分析

        Args:
            image_url: 图片URL
            prompt: 分析提示

        Returns:
            图片描述文本
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            import requests

            logger.info(f"🤖 [{self.provider_name}] 调用模型: {self.model_name}")

            # 创建LLM客户端
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.1,
                max_tokens=2000
            )

            # 下载图片并转换为base64
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            import base64
            from mimetypes import guess_type

            # 获取图片MIME类型
            content_type = guess_type(image_url)[0] or 'image/jpeg'
            image_base64 = base64.b64encode(response.content).decode('utf-8')

            # 构建消息内容
            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{image_base64}"
                    }
                }
            ]

            # 调用模型
            message = HumanMessage(content=content)
            result = llm.invoke([message])

            result_text = result.content
            logger.info(f"✅ [{self.provider_name}] 分析完成，结果长度: {len(result_text)} 字符")

            return result_text

        except requests.RequestException as e:
            error_msg = f"图片下载失败: {e}"
            logger.error(f"❌ [图片下载] 失败: {e}")
            raise Exception(error_msg)

        except Exception as e:
            logger.error(f"❌ [{self.provider_name}] 调用失败: {e}")
            raise

    async def run_async(self, image_url: str, prompt: str = None, **kwargs) -> Dict[str, Any]:
        """
        异步执行图片分析（目前是同步实现的包装）

        Args:
            image_url: 图片URL地址
            prompt: 分析提示（可选）
            **kwargs: 其他参数

        Returns:
            分析结果字典
        """
        # dashscope SDK 目前不支持异步，这里先用同步方式
        # 如果需要真正的异步，可以使用 asyncio.to_thread()
        import asyncio
        return await asyncio.to_thread(self.run, image_url, prompt, **kwargs)


# 工具实例（用于注册）
image_description_tool = ImageDescriptionTool()
