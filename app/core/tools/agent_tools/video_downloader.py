"""
视频下载工具 - 支持从URL下载视频到本地
"""

import logging
import asyncio
import re
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse, unquote
import httpx
import aiofiles

from app.core.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# 配置
UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DOWNLOAD_TIMEOUT = 300  # 5分钟
CHUNK_SIZE = 8192  # 8KB chunks for streaming

# 支持的视频格式
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', 
    '.wmv', '.m4v', '.mpg', '.mpeg', '.3gp', '.f4v'
}


@register_tool("video_downloader")
class VideoDownloader(BaseTool):
    """从URL下载视频到本地的工具"""

    name = "video_downloader"
    description = "从URL下载视频文件到本地存储"
    parameters = {
        "type": "object",
        "properties": {
            "video_url": {
                "type": "string",
                "description": "视频文件的URL地址"
            },
            "custom_filename": {
                "type": "string",
                "description": "自定义文件名（可选）",
                "default": None
            }
        },
        "required": ["video_url"]
    }

    output = {
        "type": "object",
        "properties": [
            {
                "id": "status",
                "name": "执行状态",
                "valueType": {"type": "string"},
                "description": "执行状态（success/error）"
            },
            {
                "id": "file_path",
                "name": "文件路径",
                "valueType": {"type": "string"},
                "description": "下载后的文件绝对路径"
            },
            {
                "id": "filename",
                "name": "文件名",
                "valueType": {"type": "string"},
                "description": "文件名"
            },
            {
                "id": "original_url",
                "name": "原始URL",
                "valueType": {"type": "string"},
                "description": "原始URL"
            },
            {
                "id": "file_size",
                "name": "文件大小",
                "valueType": {"type": "int"},
                "description": "文件大小（字节）"
            },
            {
                "id": "file_size_mb",
                "name": "文件大小(MB)",
                "valueType": {"type": "float", "scale": 2},
                "description": "文件大小（MB）"
            },
            {
                "id": "error",
                "name": "错误信息",
                "valueType": {"type": "string"},
                "description": "错误信息（如果失败）"
            },
            {
                "id": "message",
                "name": "附加消息",
                "valueType": {"type": "string"},
                "description": "附加消息"
            }
        ]
    }

    # 启用交互式参数确认
    require_confirmation = True
    
    def run(self, video_url: str, custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """同步运行方法，调用异步下载"""
        try:
            # 在同步环境中运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._async_download(video_url, custom_filename))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"视频下载失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "视频下载过程中发生错误"
            }
    
    async def _async_download(self, video_url: str, custom_filename: Optional[str] = None) -> Dict[str, Any]:
        """异步下载视频文件"""
        try:
            # 验证URL
            if not self._is_valid_url(video_url):
                return {
                    "status": "error",
                    "error": "无效的URL格式",
                    "url": video_url
                }
            
            # 解析URL获取文件名
            parsed_url = urlparse(video_url)
            url_filename = Path(unquote(parsed_url.path)).name
            
            # 检查文件扩展名
            file_ext = Path(url_filename).suffix.lower()
            if not file_ext:
                file_ext = '.mp4'  # 默认扩展名
            elif file_ext not in VIDEO_EXTENSIONS:
                logger.warning(f"不常见的视频格式: {file_ext}")
            
            # 生成本地文件名
            if custom_filename:
                filename = custom_filename
                if not Path(filename).suffix:
                    filename += file_ext
            else:
                filename = f"{uuid.uuid4()}{file_ext}"
            
            filepath = UPLOAD_DIR / filename
            
            logger.info(f"开始下载视频: {video_url}")
            logger.info(f"保存到: {filepath}")
            
            # 使用httpx进行异步下载
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(DOWNLOAD_TIMEOUT),
                follow_redirects=True
            ) as client:
                # 先获取文件信息
                head_response = await client.head(video_url)
                content_length = head_response.headers.get('content-length')
                
                # 检查文件大小
                if content_length:
                    file_size = int(content_length)
                    if file_size > MAX_FILE_SIZE:
                        return {
                            "status": "error",
                            "error": f"文件太大: {file_size / 1024 / 1024:.2f}MB，超过限制 {MAX_FILE_SIZE / 1024 / 1024}MB",
                            "url": video_url
                        }
                    logger.info(f"文件大小: {file_size / 1024 / 1024:.2f}MB")
                
                # 流式下载
                async with client.stream('GET', video_url) as response:
                    response.raise_for_status()
                    
                    # 获取内容类型
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith(('video/', 'application/octet-stream')):
                        logger.warning(f"内容类型可能不是视频: {content_type}")
                    
                    # 写入文件
                    total_downloaded = 0
                    async with aiofiles.open(filepath, 'wb') as f:
                        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                            await f.write(chunk)
                            total_downloaded += len(chunk)
                            
                            # 进度日志（每10MB记录一次）
                            if total_downloaded % (10 * 1024 * 1024) == 0:
                                logger.info(f"已下载: {total_downloaded / 1024 / 1024:.2f}MB")
            
            # 验证文件
            if not filepath.exists():
                return {
                    "status": "error",
                    "error": "文件下载后未找到",
                    "url": video_url
                }
            
            file_size = filepath.stat().st_size
            if file_size == 0:
                filepath.unlink()  # 删除空文件
                return {
                    "status": "error",
                    "error": "下载的文件为空",
                    "url": video_url
                }
            
            logger.info(f"视频下载成功: {filepath}, 大小: {file_size / 1024 / 1024:.2f}MB")
            
            return {
                "status": "success",
                "file_path": str(filepath.absolute()),
                "filename": filename,
                "original_url": video_url,
                "file_size": file_size,
                "file_size_mb": round(file_size / 1024 / 1024, 2)
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP错误: {e}")
            return {
                "status": "error",
                "error": f"HTTP错误: {e.response.status_code}",
                "url": video_url
            }
        except httpx.TimeoutException:
            logger.error(f"下载超时: {video_url}")
            return {
                "status": "error",
                "error": f"下载超时（超过{DOWNLOAD_TIMEOUT}秒）",
                "url": video_url
            }
        except Exception as e:
            logger.error(f"下载视频时发生错误: {e}")
            return {
                "status": "error",
                "error": str(e),
                "url": video_url
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """验证URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except:
            return False


def extract_video_urls(text: str) -> list:
    """从文本中提取视频URL
    
    Args:
        text: 输入文本
        
    Returns:
        视频URL列表
    """
    # 多种URL模式
    patterns = [
        # 标准视频文件URL
        r'https?://[^\s<>"{}|\\^`\[\]]+\.(?:mp4|avi|mov|mkv|flv|webm|wmv|m4v|mpg|mpeg|3gp|f4v)',
        # 可能的视频流URL
        r'https?://[^\s<>"{}|\\^`\[\]]+/video/[^\s<>"{}|\\^`\[\]]+',
        r'https?://[^\s<>"{}|\\^`\[\]]+/stream/[^\s<>"{}|\\^`\[\]]+',
        # YouTube等平台
        r'https?://(?:www\.)?youtube\.com/watch\?v=[^\s]+',
        r'https?://youtu\.be/[^\s]+',
        # 通用模式（需要进一步验证）
        r'https?://[^\s<>"{}|\\^`\[\]]+(?:/[^\s<>"{}|\\^`\[\]]*)+'
    ]
    
    urls = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        urls.extend(matches)
    
    # 去重并保持顺序
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def is_video_url(url: str) -> bool:
    """判断URL是否可能是视频URL
    
    Args:
        url: URL字符串
        
    Returns:
        是否可能是视频URL
    """
    # 检查文件扩展名
    parsed = urlparse(url.lower())
    path = unquote(parsed.path)
    
    # 检查常见视频扩展名
    if any(path.endswith(ext) for ext in VIDEO_EXTENSIONS):
        return True
    
    # 检查常见视频平台
    video_domains = [
        'youtube.com', 'youtu.be', 'vimeo.com', 
        'dailymotion.com', 'bilibili.com'
    ]
    if any(domain in parsed.netloc for domain in video_domains):
        return True
    
    # 检查路径中的视频相关关键词
    video_keywords = ['video', 'stream', 'media', 'mp4', 'play']
    if any(keyword in path for keyword in video_keywords):
        return True
    
    return False