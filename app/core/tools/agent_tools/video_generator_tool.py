"""
AI视频生成工具 - 基于Gemini图片生成和Sora视频生成
集成到JetLinks Agent的工具系统中
"""
import os
import logging
from typing import Any, Dict, Optional, List
from dotenv import load_dotenv

from app.core.tools.base import BaseTool, register_tool

# 导入生成器模块
from .generators import GeminiImageGenerator, SoraVideoGenerator

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


@register_tool("VideoGenerator")
class VideoGeneratorTool(BaseTool):
    """
    AI视频生成工具 - 支持文生图、图生图、图生视频功能
    """

    name: str = "VideoGenerator"
    description: str = "AI视频生成工具，支持将文本描述转换为视频，或将静态图片转换为动态视频。"

    # 参数定义 - OpenAPI格式（向后兼容）
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "视频内容描述或生成要求。例如：'未来科技城市夜景'、'让汽车变成机器人'"
            },
            "mode": {
                "type": "string",
                "description": "生成模式",
                "enum": ["text_to_video", "image_to_video", "image_only"],
                "default": "text_to_video"
            },
            "image_urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "参考图片URL列表（image_to_video模式时必需）"
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
            "image_size": {
                "type": "string",
                "description": "图片分辨率",
                "enum": ["1k", "2k", "4k"],
                "default": "2k"
            },
            "multi_keyframe": {
                "type": "boolean",
                "description": "是否启用多关键帧动画",
                "default": False
            },
            "keyframes": {
                "type": "integer",
                "description": "关键帧数量",
                "default": 5,
                "minimum": 3,
                "maximum": 10
            },
            "segment_duration": {
                "type": "integer",
                "description": "每个视频片段时长（秒）",
                "default": 3,
                "minimum": 1,
                "maximum": 10
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
                "description": "视频内容描述或生成要求。例如：'未来科技城市夜景'、'让汽车变成机器人'、'花朵逐渐绽放'"
            },
            "mode": {
                "type": "string",
                "description": "生成模式：text_to_video(文本到视频)、image_to_video(图片到视频)、image_only(仅生成图片)",
                "default": "text_to_video",
                "enum": ["text_to_video", "image_to_video", "image_only"]
            },
            "image_urls": {
                "type": "string",
                "description": "参考图片URL列表（image_to_video模式时必需，多个URL用逗号分隔）",
                "default": ""
            },
            "duration": {
                "type": "integer",
                "description": "视频时长（秒），范围5-60秒",
                "default": 15,
                "minimum": 5,
                "maximum": 60
            },
            "aspect_ratio": {
                "type": "string",
                "description": "视频宽高比：16:9、9:16、1:1、4:3",
                "default": "16:9",
                "enum": ["16:9", "9:16", "1:1", "4:3"]
            },
            "image_size": {
                "type": "string",
                "description": "图片分辨率：1k、2k、4k",
                "default": "2k",
                "enum": ["1k", "2k", "4k"]
            },
            "multi_keyframe": {
                "type": "boolean",
                "description": "是否启用多关键帧动画模式",
                "default": False
            },
            "keyframes": {
                "type": "integer",
                "description": "关键帧数量（多关键帧模式时生效），范围3-10",
                "default": 5,
                "minimum": 3,
                "maximum": 10
            },
            "segment_duration": {
                "type": "integer",
                "description": "每个视频片段时长（秒），范围1-10",
                "default": 3,
                "minimum": 1,
                "maximum": 10
            }
        },
        "required": ["prompt"]
    }

    # 输出定义
    output: Dict[str, Any] = {
        "type": "object",
        "description": "视频生成结果",
        "properties": {
            "status": {"type": "string", "description": "生成状态"},
            "image_url": {"type": "string", "description": "生成的图片URL"},
            "video_url": {"type": "string", "description": "生成的视频URL"},
            "duration": {"type": "number", "description": "视频时长"},
            "message": {"type": "string", "description": "详细信息"}
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = os.getenv('VIDEO_GENERATOR_API_KEY')
        if not self.api_key:
            raise ValueError("VIDEO_GENERATOR_API_KEY环境变量未设置")
        self.api_base = os.getenv('VIDEO_GENERATOR_API_BASE', 'duomiapi.com')

        # 初始化生成器
        try:
            self.image_generator = GeminiImageGenerator()
            self.video_generator = SoraVideoGenerator()
            logger.info("视频生成工具初始化成功")
        except Exception as e:
            logger.error(f"视频生成工具初始化失败: {e}")
            self.image_generator = None
            self.video_generator = None

    def _validate_params(self, prompt: str, mode: str = "text_to_video",
                         image_urls: Optional[List[str]] = None) -> bool:
        """验证参数"""
        if not prompt or not prompt.strip():
            return False

        if mode == "image_to_video" and (not image_urls or len(image_urls) == 0):
            return False

        if self.image_generator is None or self.video_generator is None:
            return False

        return True

    def run(self, prompt: str, mode: str = "text_to_video", image_urls: str = "",
            duration: int = 15, aspect_ratio: str = "16:9", image_size: str = "2k",
            multi_keyframe: bool = False, keyframes: int = 5, segment_duration: int = 3, **kwargs) -> Dict[str, Any]:
        """
        执行视频生成

        Args:
            prompt: 视频内容描述或生成要求
            mode: 生成模式
            image_urls: 参考图片URL列表（用逗号分隔）
            duration: 视频时长
            aspect_ratio: 视频宽高比
            image_size: 图片分辨率
            multi_keyframe: 是否启用多关键帧动画
            keyframes: 关键帧数量
            segment_duration: 片段时长
        """
        try:
            # 字符串参数处理
            if isinstance(image_urls, str):
                image_urls = [url.strip() for url in image_urls.split(',') if url.strip()] if image_urls else []

            logger.info(f"开始视频生成: prompt='{prompt[:50]}...', mode={mode}")

            logger.info(f"开始视频生成: prompt='{prompt[:50]}...', mode={mode}")

            # 参数验证
            if not self._validate_params(prompt, mode, image_urls):
                return {
                    "status": "error",
                    "message": "参数验证失败。请确保prompt不为空，且image_to_video模式需要提供图片URL",
                    "error_code": "INVALID_PARAMS"
                }

            # 根据模式执行不同的生成流程
            if mode == "image_only":
                return self._generate_image_only(prompt, image_urls, image_size, aspect_ratio)
            elif mode == "image_to_video":
                return self._generate_image_to_video(prompt, image_urls, duration, aspect_ratio, image_size)
            else:  # text_to_video
                return self._generate_text_to_video(prompt, duration, aspect_ratio, image_size,
                                                   multi_keyframe, keyframes, segment_duration)

        except Exception as e:
            logger.error(f"视频生成过程出错: {e}")
            return {
                "status": "error",
                "message": f"视频生成失败: {str(e)}",
                "error_code": "GENERATION_FAILED"
            }

    def _generate_image_only(self, prompt: str, image_urls: Optional[List[str]],
                           image_size: str, aspect_ratio: str) -> Dict[str, Any]:
        """仅生成图片"""
        try:
            logger.info(f"开始生成图片: prompt='{prompt[:50]}...'")

            if image_urls:
                # 图生图
                image_url = self.image_generator.image_to_image(
                    prompt=prompt,
                    image_urls=image_urls,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
            else:
                # 文生图
                image_url = self.image_generator.text_to_image(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
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
                    "mode": "image_only"
                }
            else:
                return {
                    "status": "error",
                    "message": "图片生成失败",
                    "error_code": "IMAGE_GENERATION_FAILED"
                }

        except Exception as e:
            logger.error(f"图片生成失败: {e}")
            return {
                "status": "error",
                "message": f"图片生成失败: {str(e)}",
                "error_code": "IMAGE_GENERATION_FAILED"
            }

    def _generate_image_to_video(self, prompt: str, image_urls: List[str],
                                duration: int, aspect_ratio: str, image_size: str) -> Dict[str, Any]:
        """图片到视频"""
        try:
            logger.info(f"开始图生视频: prompt='{prompt[:30]}...', images={len(image_urls)}")

            # 如果有图片URL，先进行图生图转换，然后生成视频
            if image_urls and prompt:
                # 图生图：将原图转换为符合要求的图片
                enhanced_image_url = self.image_generator.image_to_image(
                    prompt=prompt,
                    image_urls=image_urls,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )
                if enhanced_image_url:
                    image_urls = [enhanced_image_url]

            # 生成视频
            video_url = self.video_generator.generate_and_wait(
                prompt=f"{prompt}，动态视频",
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
                    "mode": "image_to_video"
                }
            else:
                return {
                    "status": "error",
                    "message": "视频生成失败",
                    "error_code": "VIDEO_GENERATION_FAILED"
                }

        except Exception as e:
            logger.error(f"图生视频失败: {e}")
            return {
                "status": "error",
                "message": f"视频生成失败: {str(e)}",
                "error_code": "VIDEO_GENERATION_FAILED"
            }

    def _generate_text_to_video(self, prompt: str, duration: int, aspect_ratio: str, image_size: str,
                                multi_keyframe: bool, keyframes: int, segment_duration: int) -> Dict[str, Any]:
        """文本到视频"""
        try:
            logger.info(f"开始文生视频: prompt='{prompt[:30]}...', multi_keyframe={multi_keyframe}")

            if multi_keyframe:
                # 多关键帧动画
                return self._generate_multi_keyframe_video(prompt, duration, aspect_ratio,
                                                         image_size, keyframes, segment_duration)
            else:
                # 标准文本到视频
                # 1. 先生成图片
                image_url = self.image_generator.text_to_image(
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size
                )

                if not image_url:
                    return {
                        "status": "error",
                        "message": "图片生成失败，无法继续生成视频",
                        "error_code": "IMAGE_GENERATION_FAILED"
                    }

                # 2. 基于图片生成视频
                video_url = self.video_generator.generate_and_wait(
                    prompt=f"{prompt}，动态视频效果",
                    image_urls=[image_url],
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
                        "image_url": image_url,
                        "video_url": video_url,
                        "duration": duration,
                        "mode": "text_to_video"
                    }
                else:
                    return {
                        "status": "partial",
                        "message": "图片生成成功，但视频生成失败",
                        "image_url": image_url,
                        "error_code": "VIDEO_GENERATION_FAILED"
                    }

        except Exception as e:
            logger.error(f"文生视频失败: {e}")
            return {
                "status": "error",
                "message": f"视频生成失败: {str(e)}",
                "error_code": "VIDEO_GENERATION_FAILED"
            }

    def _generate_multi_keyframe_video(self, prompt: str, duration: int, aspect_ratio: str,
                                      image_size: str, keyframes: int, segment_duration: int) -> Dict[str, Any]:
        """生成多关键帧视频"""
        try:
            logger.info(f"开始多关键帧视频生成: keyframes={keyframes}")

            # 简化版本：生成关键帧序列
            keyframe_images = []

            for i in range(keyframes):
                progress = int((i / (keyframes - 1)) * 100)
                keyframe_prompt = f"{prompt}（进度{progress}%）"

                if i == 0:
                    # 第一帧：文生图
                    image_url = self.image_generator.text_to_image(
                        prompt=keyframe_prompt,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
                else:
                    # 后续帧：基于前一帧的图生图
                    image_url = self.image_generator.image_to_image(
                        prompt=keyframe_prompt,
                        image_urls=[keyframe_images[-1]],
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )

                if image_url:
                    keyframe_images.append(image_url)
                else:
                    logger.error(f"关键帧{i+1}生成失败")
                    break

            if len(keyframe_images) < 2:
                return {
                    "status": "error",
                    "message": "关键帧生成失败，需要至少2帧",
                    "error_code": "KEYFRAME_GENERATION_FAILED"
                }

            # 基于关键帧生成视频片段（这里简化为使用第一张图片）
            video_url = self.video_generator.generate_and_wait(
                prompt=f"{prompt}，渐进式动画视频",
                image_urls=[keyframe_images[0]],
                aspect_ratio=aspect_ratio,
                duration=min(duration, 30)  # 限制长度
            )

            # 按照指定格式返回结果，包含 video 块
            formatted_result = f"""多关键帧视频生成成功，共生成{len(keyframe_images)}个关键帧

:::video
```json
{video_url}
```
:::"""

            return {
                "status": "success",
                "message": formatted_result,
                "video_url": video_url,
                "keyframes_count": len(keyframe_images),
                "duration": duration,
                "mode": "multi_keyframe"
            }

        except Exception as e:
            logger.error(f"多关键帧视频生成失败: {e}")
            return {
                "status": "error",
                "message": f"多关键帧视频生成失败: {str(e)}",
                "error_code": "MULTI_KEYFRAME_FAILED"
            }