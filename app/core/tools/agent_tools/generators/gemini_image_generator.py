import http.client
import json
import logging
import time
import requests
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiImageGenerator:
    def __init__(self):
        self.base_url = os.getenv('VIDEO_GENERATOR_API_BASE', 'duomiapi.com')
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('VIDEO_GENERATOR_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY或VIDEO_GENERATOR_API_KEY环境变量未设置")

        self.headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }
        self.download_dir = "downloads/images"

        # 创建下载目录
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def generate_image(self, prompt: str, image_urls: Optional[List[str]] = None,
                     aspect_ratio: str = "16:9", image_size: str = "1k",
                     model: str = "gemini-3-pro-image-preview") -> Optional[str]:
        """
        生成图片

        Args:
            prompt: 图片描述文本
            image_urls: 参考图片URL列表（图生图时使用）
            aspect_ratio: 宽高比
            image_size: 图片尺寸 (1k, 2k, 4k)
            model: 使用的模型

        Returns:
            生成的图片URL，失败返回None
        """
        conn = http.client.HTTPSConnection(self.base_url)

        # 构建基础payload，duration对这个API是必需的
        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": 5,  # 这个API必须要有duration参数
            "image_size": image_size
        }

        # 根据是否有图片URL决定是文生图还是图生图
        if image_urls:
            payload["image_urls"] = image_urls
            logger.info("图生图模式")
        else:
            logger.info("文生图模式")

        try:
            logger.info("正在提交图片生成请求...")
            logger.debug("请求数据: %s", json.dumps(payload, ensure_ascii=False))

            # 发送请求到Gemini图片生成API
            # 文生图使用nano-banana，图生图使用nano-banana-edit
            endpoint = "/api/gemini/nano-banana" if not image_urls else "/api/gemini/nano-banana-edit"
            conn.request("POST", endpoint, json.dumps(payload), self.headers)
            res = conn.getresponse()

            logger.debug("响应状态码: %s", res.status)

            data = res.read()
            response_text = data.decode("utf-8")
            logger.debug("原始响应: %s", response_text)

            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error("JSON解析错误: %s", e)
                return None

            # 处理API响应格式
            if response_data.get("code") == 200:  # 成功状态
                task_id = response_data.get("data", {}).get("task_id")
                if task_id:
                    logger.info("图片生成任务已提交，任务ID: %s", task_id)
                    return self.check_image_task_status(task_id)
                else:
                    logger.warning("响应中未找到task_id")
                    return None
            else:
                logger.error("API返回错误: %s", response_data)
                return None

        except Exception as e:
            logger.error("提交请求时出错: %s", e, exc_info=True)
            return None
        finally:
            conn.close()

    def check_image_task_status(self, task_id: str) -> Optional[str]:
        """
        查询图片生成任务状态

        Args:
            task_id: 任务ID

        Returns:
            图片URL，失败返回None
        """
        conn = http.client.HTTPSConnection(self.base_url)

        try:
            headers = self.headers  # 使用初始化时设置的headers
            conn.request("GET", f"/api/gemini/nano-banana/{task_id}", '', headers)
            res = conn.getresponse()
            data = res.read()
            response_data = json.loads(data.decode("utf-8"))

            if response_data.get("code") == 200:
                task_data = response_data.get("data", {})
                state = task_data.get("state")

                if state == "succeeded":
                    # 获取图片URL
                    images = task_data.get("data", {}).get("images", [])
                    if images:
                        image_url = images[0].get("url")
                        logger.info("图片生成完成: %s", image_url)
                        return image_url
                    else:
                        logger.warning("图片已完成但未获取到下载链接")
                        return None
                elif state == "failed":
                    logger.error("图片生成失败")
                    return None
                else:
                    # pending, running等状态
                    logger.info("图片处理中，状态: %s", state)
                    # 等待10秒后再次查询
                    time.sleep(10)
                    return self.check_image_task_status(task_id)
            else:
                logger.warning("查询状态失败: %s", response_data)
                return None

        except Exception as e:
            logger.error("查询状态时出错: %s", e, exc_info=True)
            return None
        finally:
            conn.close()

    def download_image(self, image_url: str, filename: Optional[str] = None) -> Optional[str]:
        """
        下载图片

        Args:
            image_url: 图片URL
            filename: 保存的文件名

        Returns:
            本地文件路径，失败返回None
        """
        try:
            if not filename:
                # 根据URL或时间戳生成文件名
                timestamp = int(time.time())
                if image_url.endswith('.png'):
                    filename = f"generated_image_{timestamp}.png"
                else:
                    filename = f"generated_image_{timestamp}.jpg"

            filepath = os.path.join(self.download_dir, filename)

            logger.info("开始下载图片: %s", image_url)
            response = requests.get(image_url, stream=True, timeout=(10, 30))  # 连接超时10秒，读取超时30秒
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("图片下载完成: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("下载图片时出错: %s", e, exc_info=True)
            return None

    def text_to_image(self, prompt: str, image_size: str = "2k",
                     aspect_ratio: str = "16:9", download: bool = True) -> Optional[str]:
        """
        文生图

        Args:
            prompt: 图片描述
            image_size: 图片尺寸
            aspect_ratio: 宽高比
            download: 是否下载到本地

        Returns:
            图片URL或本地路径
        """
        logger.info("=== 文生图开始 ===")

        image_url = self.generate_image(
            prompt=prompt,
            image_size=image_size,
            aspect_ratio=aspect_ratio
        )

        if image_url and download:
            return self.download_image(image_url)

        return image_url

    def image_to_image(self, prompt: str, image_urls: List[str],
                      image_size: str = "2k", download: bool = True) -> Optional[str]:
        """
        图生图

        Args:
            prompt: 修改描述
            image_urls: 参考图片URL列表
            image_size: 图片尺寸
            download: 是否下载到本地

        Returns:
            图片URL或本地路径
        """
        logger.info("=== 图生图开始 ===")

        image_url = self.generate_image(
            prompt=prompt,
            image_urls=image_urls,
            image_size=image_size
        )

        if image_url and download:
            return self.download_image(image_url)

        return image_url

# # 使用示例
# if __name__ == "__main__":
#     generator = GeminiImageGenerator()

#     # 文生图示例
#     reference_image = ""

#     prompt = "生成一个agent架构图"
#     result = generator.text_to_image(prompt, image_size="2k")
#     print(f"文生图结果: {result}")

#     # 图生图示例
#     reference_image = "https://img4.bitautoimg.com/autoalbum/files/20230208/686/202302084997585641968630018_3000x2000_w1.png"
#     prompt = "Turn the car in this picture into a robot, must be a picture"
#     result = generator.image_to_image(prompt, [reference_image])
#     print(f"图生图结果: {result}")
