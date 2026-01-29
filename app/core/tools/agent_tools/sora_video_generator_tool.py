"""
Sora视频生成工具 - 基于Sora AI的视频生成功能
集成到JetLinks Agent的工具系统中
"""
import os
import logging
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv

from app.core.tools.base import BaseTool, register_tool

# 导入视频生成模块
from .generators import SoraVideoGenerator

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


@register_tool("SoraVideoGenerator")
class SoraVideoGeneratorTool(BaseTool):
    """
    Sora视频生成工具 - 基于Sora AI的视频生成功能
    """

    name: str = "SoraVideoGenerator"
    description: str = "AI视频生成工具，专门用于将静态图片转换为动态视频。支持多种视频风格、时长控制和质量设置。"

    # 参数定义
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "视频生成描述。例如：'让图片中的花朵绽放'、'动态的视频效果'"
            },
            "image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "源图片URL列表（必需）"
            },
            "duration": {
                "type": "integer",
                "description": "视频时长（秒）",
                "default": 15,
                "minimum": 5,
                "maximum": 60
            },
            "aspect_ratio": {
                "type": "string",
                "description": "视频宽高比",
                "enum": ["16:9", "9:16", "1:1", "4:3"],
                "default": "16:9"
            },
            "motion_strength": {
                "type": "string",
                "description": "动作强度",
                "enum": ["gentle", "moderate", "strong"],
                "default": "moderate"
            },
            "video_style": {
                "type": "string",
                "description": "视频风格",
                "enum": ["realistic", "cinematic", "animated", "dreamy", "dramatic"],
                "default": "cinematic"
            }
        },
        "required": ["prompt", "image_urls"]
    }

    # 输入参数定义 - 严格按照chatbi格式
    inputs: List[Dict[str, Any]] = [
        {
            "id": "prompt",
            "name": "视频生成描述",
            "expands": {},
            "required": True,
            "valueType": {
                "type": "string"
            },
            "description": "视频生成描述。例如：'让图片中的花朵绽放'、'动态的视频效果'、'流水潺潺的动画'"
        },
        {
            "id": "image_urls",
            "name": "源图片URL",
            "default": "",
            "expands": {},
            "required": True,
            "valueType": {
                "type": "string"
            },
            "description": "源图片URL列表（必需，多个URL用逗号分隔）"
        },
        {
            "id": "duration",
            "name": "视频时长",
            "default": 15,
            "expands": {},
            "required": False,
            "valueType": {
                "type": "int"
            },
            "description": "视频时长（秒），范围5-60秒"
        },
        {
            "id": "aspect_ratio",
            "name": "视频宽高比",
            "default": "16:9",
            "expands": {},
            "required": False,
            "valueType": {
                "type": "string"
            },
            "description": "视频宽高比：16:9、9:16、1:1、4:3"
        },
        {
            "id": "video_style",
            "name": "视频风格",
            "default": "cinematic",
            "expands": {},
            "required": False,
            "valueType": {
                "type": "string"
            },
            "description": "视频风格：realistic、cinematic、animated、dreamy、dramatic"
        },
        {
            "id": "motion_strength",
            "name": "动作强度",
            "default": "moderate",
            "expands": {},
            "required": False,
            "valueType": {
                "type": "string"
            },
            "description": "动作强度：gentle、moderate、strong"
        }
    ]

    # 输出定义
    output: Dict[str, Any] = {
        "type": "object",
        "description": "视频生成结果",
        "properties": {
            "status": {"type": "string", "description": "生成状态"},
            "video_url": {"type": "string", "description": "生成的视频URL"},
            "duration": {"type": "number", "description": "视频时长"},
            "aspect_ratio": {"type": "string", "description": "视频宽高比"},
            "message": {"type": "string", "description": "详细信息"}
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            self.video_generator = SoraVideoGenerator()
            logger.info("Sora视频生成工具初始化成功")
        except Exception as e:
            logger.error(f"Sora视频生成工具初始化失败: {e}")
            self.video_generator = None

    def _validate_params(self, prompt: str, image_urls: List[str], duration: int) -> bool:
        """验证参数"""
        if not prompt or not prompt.strip():
            return False

        if not image_urls or len(image_urls) == 0:
            return False

        if duration < 5 or duration > 60:
            return False

        if self.video_generator is None:
            return False

        return True

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行视频生成
        """
        try:
            # 获取参数
            prompt = kwargs.get('prompt', '').strip()
            image_urls = kwargs.get('image_urls', [])
            duration = kwargs.get('duration', 15)
            aspect_ratio = kwargs.get('aspect_ratio', '16:9')
            motion_strength = kwargs.get('motion_strength', 'moderate')
            video_style = kwargs.get('video_style', 'cinematic')

            logger.info(f"开始视频生成: prompt='{prompt[:30]}...', images={len(image_urls)}")

            # 参数验证
            if not self._validate_params(prompt, image_urls, duration):
                return {
                    "status": "error",
                    "message": "参数验证失败。请确保prompt和image_urls不为空，且duration在5-60秒之间",
                    "error_code": "INVALID_PARAMS"
                }

            # 构建增强的提示词
            enhanced_prompt = f"{prompt}，{video_style}风格，{motion_strength}动作强度的动态视频"

            # 生成视频
            video_url = self.video_generator.generate_and_wait(
                prompt=enhanced_prompt,
                image_urls=image_urls,
                aspect_ratio=aspect_ratio,
                duration=duration
            )

            if video_url:
                # 按照指定格式返回结果，包含 video 块
                formatted_result = f"""视频生成成功

:::video
```json
{video_url}
```
:::"""

                return {
                    "status": "success",
                    "message": formatted_result,
                    "video_url": video_url,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio,
                    "motion_strength": motion_strength,
                    "video_style": video_style
                }
            else:
                return {
                    "status": "error",
                    "message": "视频生成失败",
                    "error_code": "VIDEO_GENERATION_FAILED"
                }

        except Exception as e:
            logger.error(f"视频生成过程出错: {e}")
            return {
                "status": "error",
                "message": f"视频生成失败: {str(e)}",
                "error_code": "GENERATION_FAILED"
            }