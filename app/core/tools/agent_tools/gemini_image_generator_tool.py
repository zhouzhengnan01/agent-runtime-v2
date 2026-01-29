"""
Gemini图片生成工具 - 基于Gemini AI的图片生成功能
集成到JetLinks Agent的工具系统中
"""
import os
import logging
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv

from app.core.tools.base import BaseTool, register_tool

# 导入图片生成模块
from .generators import GeminiImageGenerator

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


@register_tool("gemini_image_generator")
class GeminiImageGeneratorTool(BaseTool):
    """
    Gemini图片生成工具 - 基于Gemini AI的图片生成功能
    """

    name: str = "GeminiImageGenerator"
    description: str = "AI图片生成工具，支持文生图和图生图功能。可以将文本描述转换为高质量图片，或基于参考图片进行风格转换和内容修改。"

    # 参数定义
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "图片内容描述或修改要求。例如：'赛博朋克城市'、'将图片转换为油画风格'"
            },
            "mode": {
                "type": "string",
                "description": "生成模式",
                "enum": ["text_to_image", "image_to_image"],
                "default": "text_to_image"
            },
            "image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "参考图片URL列表（image_to_image模式时必需）"
            },
            "aspect_ratio": {
                "type": "string",
                "description": "图片宽高比",
                "enum": ["16:9", "9:16", "1:1", "4:3"],
                "default": "16:9"
            },
            "image_size": {
                "type": "string",
                "description": "图片分辨率",
                "enum": ["1k", "2k", "4k"],
                "default": "2k"
            },
            "style": {
                "type": "string",
                "description": "图片风格（可选）",
                "enum": ["realistic", "artistic", "anime", "oil_painting", "watercolor", "sketch"],
                "default": "realistic"
            }
        },
        "required": ["prompt"]
    }

    # 使用与chatbi一致的parameters格式
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "图片内容描述或修改要求。例如：'赛博朋克城市'、'将图片转换为油画风格'、'美丽的自然风光'"
            },
            "mode": {
                "type": "string",
                "description": "生成模式：text_to_image(文生图)、image_to_image(图生图)",
                "default": "text_to_image",
                "enum": ["text_to_image", "image_to_image"]
            },
            "image_urls": {
                "type": "string",
                "description": "参考图片URL列表（image_to_image模式时必需，多个URL用逗号分隔）",
                "default": ""
            },
            "aspect_ratio": {
                "type": "string",
                "description": "图片宽高比：16:9、9:16、1:1、4:3",
                "default": "16:9",
                "enum": ["16:9", "9:16", "1:1", "4:3"]
            },
            "image_size": {
                "type": "string",
                "description": "图片分辨率：1k、2k、4k",
                "default": "2k",
                "enum": ["1k", "2k", "4k"]
            },
            "style": {
                "type": "string",
                "description": "图片风格：realistic、artistic、anime、oil_painting、watercolor、sketch",
                "default": "realistic",
                "enum": ["realistic", "artistic", "anime", "oil_painting", "watercolor", "sketch"]
            }
        },
        "required": ["prompt"]
    }

    # 输出定义
    output: Dict[str, Any] = {
        "type": "object",
        "description": "图片生成结果",
        "properties": {
            "status": {"type": "string", "description": "生成状态"},
            "image_url": {"type": "string", "description": "生成的图片URL"},
            "mode": {"type": "string", "description": "使用的生成模式"},
            "message": {"type": "string", "description": "详细信息"}
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            self.image_generator = GeminiImageGenerator()
            logger.info("Gemini图片生成工具初始化成功")
        except Exception as e:
            logger.error(f"Gemini图片生成工具初始化失败: {e}")
            self.image_generator = None

    def _validate_params(self, prompt: str, mode: str = "text_to_image",
                         image_urls: Optional[List[str]] = None) -> bool:
        """验证参数"""
        if not prompt or not prompt.strip():
            return False

        if mode == "image_to_image" and (not image_urls or len(image_urls) == 0):
            return False

        if self.image_generator is None:
            return False

        return True

    def run(self, prompt: str, mode: str = "text_to_image", image_urls: str = "",
            aspect_ratio: str = "16:9", image_size: str = "2k", style: str = "realistic", **kwargs) -> Dict[str, Any]:
        """
        执行图片生成

        Args:
            prompt: 图片内容描述或修改要求
            mode: 生成模式
            image_urls: 参考图片URL列表（用逗号分隔）
            aspect_ratio: 图片宽高比
            image_size: 图片分辨率
            style: 图片风格
        """
        try:
            # 字符串参数处理
            if isinstance(image_urls, str):
                image_urls = [url.strip() for url in image_urls.split(',') if url.strip()] if image_urls else []

            logger.info(f"开始图片生成: prompt='{prompt[:50]}...', mode={mode}")

            logger.info(f"开始图片生成: prompt='{prompt[:50]}...', mode={mode}")

            # 参数验证
            if not self._validate_params(prompt, mode, image_urls):
                return {
                    "status": "error",
                    "message": "参数验证失败。请确保prompt不为空，且image_to_image模式需要提供图片URL",
                    "error_code": "INVALID_PARAMS"
                }

            # 执行图片生成
            if mode == "image_to_image":
                # 图生图
                image_url = self.image_generator.image_to_image(
                    prompt=prompt,
                    image_urls=image_urls,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    download=False  # 不下载，只返回URL
                )
            else:  # text_to_image
                # 文生图
                image_url = self.image_generator.text_to_image(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    download=False  # 不下载，只返回URL
                )

            if image_url:
                # 按照指定格式返回结果，包含 image 块
                formatted_result = f"""图片生成成功

:::image
```json
{image_url}
```
:::"""

                return {
                    "status": "success",
                    "message": formatted_result,
                    "image_url": image_url,
                    "mode": mode,
                    "image_size": image_size,
                    "aspect_ratio": aspect_ratio,
                    "style": style
                }
            else:
                return {
                    "status": "error",
                    "message": "图片生成失败",
                    "error_code": "IMAGE_GENERATION_FAILED"
                }

        except Exception as e:
            logger.error(f"图片生成过程出错: {e}")
            return {
                "status": "error",
                "message": f"图片生成失败: {str(e)}",
                "error_code": "GENERATION_FAILED"
            }