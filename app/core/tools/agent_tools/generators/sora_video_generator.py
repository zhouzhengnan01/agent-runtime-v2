import http.client
import json
import logging
import time
import requests
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

class SoraVideoGenerator:
    def __init__(self):
        self.base_url = os.getenv('VIDEO_GENERATOR_API_BASE', 'duomiapi.com')
        api_key = os.getenv('VIDEO_GENERATOR_API_KEY')
        if not api_key:
            raise ValueError("VIDEO_GENERATOR_API_KEY环境变量未设置")

        self.headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }
        self.download_dir = "downloads/videos"

        # 创建下载目录
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def generate_video(self, prompt, image_urls=None, aspect_ratio="16:9", duration=10, model="sora-2"):
        """提交视频生成请求"""
        conn = http.client.HTTPSConnection(self.base_url)

        payload = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration
        }

        if image_urls:
            payload["image_urls"] = image_urls

        try:
            logger.info("正在提交视频生成请求...")
            logger.debug("请求数据: %s", json.dumps(payload, ensure_ascii=False))

            # 发送请求
            conn.request("POST", "/v1/videos/generations", json.dumps(payload), self.headers)
            res = conn.getresponse()

            logger.debug("响应状态码: %s", res.status)
            logger.debug("响应头: %s", dict(res.getheaders()))

            data = res.read()
            response_text = data.decode("utf-8")
            logger.debug("原始响应: %s", response_text)

            # 尝试解析JSON
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error("JSON解析错误: %s", e)
                logger.error("响应内容不是有效的JSON格式")
                return None

            # 检查响应格式
            if "success" in response_data:
                if response_data.get("success"):
                    task_id = response_data.get("data", {}).get("task_id")
                    if task_id:
                        logger.info("视频生成任务已提交，任务ID: %s", task_id)
                        return task_id
                    else:
                        logger.warning("响应中未找到task_id")
                        return None
                else:
                    logger.error("API返回失败状态: %s", response_data)
                    # 如果有错误信息，详细打印
                    if "error" in response_data:
                        logger.error("错误信息: %s", response_data.get("error"))
                    if "message" in response_data:
                        logger.error("错误消息: %s", response_data.get("message"))
                    return None
            else:
                # 可能API格式不同，尝试直接查找task_id
                if "task_id" in response_data:
                    task_id = response_data.get("task_id")
                    logger.info("视频生成任务已提交（直接格式），任务ID: %s", task_id)
                    return task_id
                elif "data" in response_data and "task_id" in response_data["data"]:
                    task_id = response_data["data"]["task_id"]
                    logger.info("视频生成任务已提交（data格式），任务ID: %s", task_id)
                    return task_id
                elif "id" in response_data:
                    # API只返回了id，这可能就是task_id
                    task_id = response_data.get("id")
                    logger.info("视频生成任务已提交（ID格式），任务ID: %s", task_id)
                    return task_id
                else:
                    logger.warning("未知的响应格式: %s", response_data)
                    return None

        except Exception as e:
            logger.error("提交请求时出错: %s", e, exc_info=True)
            return None
        finally:
            conn.close()

    def check_task_status(self, task_id):
        """查询任务状态"""
        conn = http.client.HTTPSConnection(self.base_url)

        try:
            headers = self.headers  # 使用初始化时设置的headers
            conn.request("GET", f"/v1/videos/tasks/{task_id}", '', headers)
            res = conn.getresponse()
            data = res.read()
            response_text = data.decode("utf-8")
            logger.debug("状态查询响应: %s", response_text)
            response_data = json.loads(response_text)

            # 处理实际API响应格式 - 从根级别直接获取状态信息
            if "state" in response_data:
                state = response_data.get("state")  # 从根级别获取state
                progress = response_data.get("progress", 0)  # 从根级别获取progress
                video_url = response_data.get("url")  # 从根级别获取url
                data = response_data.get("data")  # 获取data字段（可能为None）

                # 将state映射为status以保持兼容性
                status = state
                if state == "succeeded":
                    status = "completed"
                elif state == "failed":
                    status = "failed"

                logger.info("任务状态: %s, 进度: %s%%", state, progress)

                # 构造返回数据
                task_data = data if data else {}  # 如果data为None，使用空字典

                return {
                    "status": status,
                    "progress": progress,
                    "video_url": video_url,
                    "task_data": task_data,
                    "raw_response": response_data  # 保留原始响应用于调试
                }
            else:
                logger.warning("查询状态失败: 未找到state字段, 响应: %s", response_data)
                return None

        except Exception as e:
            logger.error("查询状态时出错: %s", e, exc_info=True)
            return None
        finally:
            conn.close()

    def download_video(self, video_url, filename=None):
        """下载视频"""
        try:
            if not filename:
                # 从URL中提取文件名或使用时间戳
                filename = f"video_{int(time.time())}.mp4"

            filepath = os.path.join(self.download_dir, filename)

            logger.info("开始下载视频: %s", video_url)
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("视频下载完成: %s", filepath)
            return filepath

        except Exception as e:
            logger.error("下载视频时出错: %s", e, exc_info=True)
            return None

    def generate_and_wait(self, prompt, image_urls=None, aspect_ratio="16:9", duration=10, model="sora-2"):
        """生成视频并等待完成，然后自动下载"""
        logger.info("=== 视频生成流程开始 ===")

        # 1. 提交生成任务
        task_id = self.generate_video(prompt, image_urls, aspect_ratio, duration, model)
        if not task_id:
            logger.error("任务提交失败，退出")
            return False

        # 2. 轮询检查任务状态
        logger.info("开始监控任务进度，每60秒查询一次...")

        while True:
            status_info = self.check_task_status(task_id)

            if not status_info:
                logger.warning("状态查询失败，30秒后重试...")
                time.sleep(30)
                continue

            status = status_info["status"]
            progress = status_info["progress"]

            # 检查是否完成（状态已在check_task_status中映射）
            if status == "completed" or status == "succeeded":
                video_url = status_info["video_url"]
                if video_url:
                    logger.info("视频生成完成！")
                    logger.info("视频URL: %s", video_url)
                    # 3. 下载视频
                    downloaded_path = self.download_video(video_url)
                    if downloaded_path:
                        logger.info("视频已下载到: %s", downloaded_path)
                        return True
                    else:
                        logger.error("视频下载失败")
                        return False
                else:
                    logger.warning("视频已完成但未获取到下载链接")
                    return False

            elif status == "failed":
                logger.error("视频生成失败")
                return False

            elif status in ["processing", "pending", "queued", "running"]:
                logger.info("任务处理中，进度: %s%%，60秒后再次查询...", progress)
                time.sleep(60)

            else:
                logger.warning("未知状态: %s，60秒后再次查询...", status)
                time.sleep(60)

# 使用示例
if __name__ == "__main__":
    generator = SoraVideoGenerator()

    # 视频生成参数
    prompt = "一辆汽车如何变成图片中的机器人"
    # image_urls = ["https://crea-img.xinzhiyun.net/upload/202502251100509684.jpg"]
    image_urls = ["https://th.bing.com/th/id/R.9e3c4cab6acffe64efa0c5a4f584cf87?rik=NdbjvtUjs7wZcg&riu=http%3a%2f%2fpic36.photophoto.cn%2f20150802%2f0005018322215790_b.jpg&ehk=kEMqTnoGWp8XJsE8fZpW7Ku7FHpOMGWnn%2bnkP1n7x8E%3d&risl=&pid=ImgRaw&r=0"]

    # 生成视频并等待完成
    generator.generate_and_wait(
        prompt=prompt,
        image_urls=image_urls,
        aspect_ratio="16:9",
        duration=10,
        model="sora-2"
    )
