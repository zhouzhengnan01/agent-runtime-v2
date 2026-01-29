# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json
import base64
from typing import Dict, List, Optional, Tuple, Iterator, Any
from queue import Queue
from pathlib import Path

import requests

from app.shared.jetlinks_video.utils import logger_utils
from app.shared.jetlinks_video.configs.vlm_config import VlmConfig
from app.shared.jetlinks_video.utils.file_utils import read_env_kv

logger = logger_utils.get_logger(__name__)

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent
ENV_PATH = os.path.join(BASE_PATH, ".env")

VLM_HTTP_MAX_WAIT = float(read_env_kv(ENV_PATH, "VLM_HTTP_MAX_WAIT", "240") or 240)
VLM_HTTP_MAX_RETRY = int(read_env_kv(ENV_PATH, "VLM_HTTP_MAX_RETRY", "1") or 1)


def _strip_file_scheme(p: str) -> str:
    if isinstance(p, str) and p.lower().startswith("file://"):
        return p[len("file://"):]
    return p


def _image_to_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"[CloudVLMCurlClient] 图片转 base64 失败: {e}")
        return ""


def _poll_ctrl_heartbeat(q_ctrl: Optional[Queue], stop: Optional[object]) -> bool:
    """用于流式过程中感知 STOP 信号。"""
    if not q_ctrl or stop is None:
        return False
    try:
        msg = q_ctrl.get_nowait()
    except Exception:
        return False

    if msg is stop:
        return True
    if isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN"):
        return True

    # 放回队列，避免消费掉别人的指令
    try:
        q_ctrl.put_nowait(msg)
    except Exception:
        pass
    return False


class VlmCurlClient:
    """
    通用的 VLM HTTP 客户端（OpenAI 兼容协议）：
    - 支持 Qwen / GLM / 其他兼容模型
    - 只接收“高层语义参数”：system_prompt / user_prompt / text / images / videos
    - 内部构造 OpenAI chat.completions 所需的 messages
    """

    def __init__(
        self,
        vlm_config: VlmConfig,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        text: Optional[str] = None,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        q_ctrl: Optional[Queue] = None,
        stop: Optional[object] = None,
    ) -> None:
        # 从配置里取基础信息
        self.api_key: Optional[str] = vlm_config.vlm_api_key
        self.base_url: str = vlm_config.vlm_base_url
        self.model_name: str = vlm_config.vlm_model_name
        self.want_streaming: bool = vlm_config.vlm_streaming
        self.temperature: float = vlm_config.vlm_temperature
        self.backup_api_key: Optional[str] = vlm_config.vlm_api_key_backup
        self.backup_base_url: Optional[str] = vlm_config.vlm_base_url_backup
        self.backup_model_name: Optional[str] = vlm_config.vlm_model_name_backup

        # 上游提供的“高层”参数
        self.system_prompt = (system_prompt or "").strip()
        self.user_prompt = (user_prompt or "").strip()
        self.text = (text or "").strip()
        self.images: List[str] = images or []
        self.videos: List[str] = videos or []

        self.q_ctrl = q_ctrl
        self.stop = stop

    # ----------------- 消息构造 -----------------
    def _build_messages(self, api_key: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool]:
        """
        构造 OpenAI 兼容 messages：
        - system: 单条（如有）
        - user: 一条，包含 text + image_url + video（如果目标模型支持）
        返回: (messages, has_image)
        """
        active_key = api_key if api_key is not None else self.api_key
        if not active_key:
            raise ValueError("VLM_API_KEY 未配置，请检查 .env 文件")

        messages: List[Dict[str, Any]] = []
        has_image = False

        # 1) system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # 2) user 内容：统一放在一个 user message 内
        user_contents: List[Dict[str, Any]] = []

        # 2.1 文本部分（user_prompt + text 合并）
        text_parts: List[str] = []
        if self.user_prompt:
            text_parts.append(self.user_prompt)
        if self.text:
            text_parts.append(self.text)
        if text_parts:
            user_contents.append({"type": "text", "text": "\n".join(text_parts)})

        # 2.2 图片：转为 base64 data URL
        for img_path in self.images:
            if not img_path:
                continue
            # http(s) 直接透传
            if img_path.lower().startswith(("http://", "https://")):
                user_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": img_path},
                    }
                )
                has_image = True
                continue

            # 本地 / file://
            local_path = _strip_file_scheme(img_path)
            if not os.path.exists(local_path):
                logger.warning("[CloudVLMCurlClient] 图片文件不存在: %s", local_path)
                continue
            b64 = _image_to_base64(local_path)
            if not b64:
                continue
            user_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
            has_image = True

        # 2.3 视频（可选）：如果你后面需要，可以按模型要求去构造。
        for v in self.videos:
            logger.warning("[CloudVLMCurlClient] 当前未对视频进行特殊处理: %s", v)

        if user_contents:
            messages.append({"role": "user", "content": user_contents})

        if not messages:
            raise ValueError("CloudVLMCurlClient: 构造出的 messages 为空，请检查上游参数")

        return messages, has_image

    # ----------------- payload 构造 -----------------
    def _build_payload(
        self,
        *,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        messages, has_image = self._build_messages(api_key=api_key)

        payload: Dict[str, Any] = {
            "model": model_name or self.model_name,
            "messages": messages,
            "stream": bool(self.want_streaming),
            "temperature": self.temperature,
            "max_tokens": 2048,
        }

        return payload, has_image

    # ----------------- 日志：调用参数（中文便于排查） -----------------
    def _log_call_info(
        self,
        *,
        stream: bool,
        payload: Dict[str, Any],
        base_url: Optional[str] = None,
        label: str = "cloud",
    ) -> None:
        """
        每次发起调用前打印关键信息（中文）：
        model / stream / temperature / max_tokens / base_url / timeout / retry
        """
        model = payload.get("model") or self.model_name
        temperature = payload.get("temperature", self.temperature)
        max_tokens = payload.get("max_tokens", 0)

        # 兜底类型
        try:
            temperature_f = float(temperature) if temperature is not None else 0.0
        except Exception:
            temperature_f = 0.0
        try:
            max_tokens_i = int(max_tokens) if max_tokens is not None else 0
        except Exception:
            max_tokens_i = 0

        base_url = (base_url or self.base_url or "").strip().rstrip("/")
        timeout_s = VLM_HTTP_MAX_WAIT
        retry_n = VLM_HTTP_MAX_RETRY

        logger.info(
            "【VLM调用/%s】model=%s｜stream=%s｜temperature=%.3f｜max_tokens=%d｜base_url=%s｜timeout=%.1fs｜retry=%d",
            label,
            str(model),
            "是" if stream else "否",
            temperature_f,
            max_tokens_i,
            base_url,
            float(timeout_s),
            int(retry_n),
        )

    # ----------------- 非流式调用 -----------------
    def _call_api_nonstream_with(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        label: str,
    ) -> Tuple[str, Dict[str, Any]]:
        payload, has_image = self._build_payload(model_name=model_name, api_key=api_key)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{base_url.rstrip('/')}/chat/completions"

        # ✅ 中文日志（替代旧英文日志）
        self._log_call_info(stream=False, payload=payload, base_url=base_url, label=label)

        resp = requests.post(url, json=payload, headers=headers, timeout=VLM_HTTP_MAX_WAIT)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"VLM API 错误 {resp.status_code}: {err}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("VLM API 返回的 choices 为空")

        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content", "")

        # 兼容 content 为 list 的情况
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    parts.append(str(c["text"]))
            content = "".join(parts)

        text = (content or "").strip()
        if not text:
            text = "模型未生成有效回复。"

        usage = data.get("usage") or {}
        usage = {
            "backend": label,
            "model": model_name,
            "status": "ok",
            "raw_usage": usage,
        }
        return text, usage

    def _call_api_nonstream(self) -> Tuple[str, Dict[str, Any]]:
        return self._call_api_nonstream_with(
            base_url=self.base_url,
            api_key=self.api_key or "",
            model_name=self.model_name,
            label="cloud",
        )

    # ----------------- SSE 工具 -----------------
    @staticmethod
    def _iter_sse_data_lines(resp: requests.Response) -> Iterator[str]:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line:
                continue
            if line.startswith("data:"):
                yield line[len("data:"):].strip()

    # ----------------- 流式调用 -----------------
    def _call_api_stream_with(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        label: str,
    ) -> Iterator[Tuple[Optional[str], Optional[dict]]]:
        payload, has_image = self._build_payload(model_name=model_name, api_key=api_key)
        payload["stream"] = True  # 强制流式

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{base_url.rstrip('/')}/chat/completions"

        # ✅ 中文日志（替代旧英文日志）
        self._log_call_info(stream=True, payload=payload, base_url=base_url, label=label)

        t0 = time.time()
        usage_final: Optional[dict] = None

        with requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=VLM_HTTP_MAX_WAIT,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                try:
                    err = resp.json()
                except Exception:
                    err = resp.text
                raise RuntimeError(f"VLM API 错误 {resp.status_code}: {err}")

            for data_str in self._iter_sse_data_lines(resp):
                # STOP 检测
                if _poll_ctrl_heartbeat(self.q_ctrl, self.stop):
                    logger.warning("[CloudVLMCurlClient] 检测到 STOP 信号，中断流式读取")
                    break

                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except Exception:
                    continue

                # usage
                u = chunk.get("usage")
                if isinstance(u, dict) and u:
                    usage_final = u

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                c0 = choices[0] or {}
                delta_obj = c0.get("delta") or {}

                delta_text = delta_obj.get("content") or ""
                if delta_text:
                    yield str(delta_text), None

        if not usage_final:
            usage_final = {}

        usage_final = {
            "backend": label,
            "model": model_name,
            "status": "ok",
            "elapsed_sec": time.time() - t0,
            "raw_usage": usage_final,
        }

        # 与之前协议保持一致：最后用 (None, usage_final) 结尾
        yield None, usage_final

    def _call_api_stream(self) -> Iterator[Tuple[Optional[str], Optional[dict]]]:
        return self._call_api_stream_with(
            base_url=self.base_url,
            api_key=self.api_key or "",
            model_name=self.model_name,
            label="cloud",
        )

    def _has_backup(self) -> bool:
        return bool(self.backup_base_url and self.backup_api_key)

    @staticmethod
    def _is_connection_error(error: Exception) -> bool:
        if isinstance(
            error,
            (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ),
        ):
            return True

        msg = str(error).lower()
        if any(
            key in msg
            for key in (
                "connection error",
                "connect timeout",
                "timed out",
                "timeout",
                "connection refused",
                "network is unreachable",
            )
        ):
            return True

        if "vlm api 错误" in msg and any(code in msg for code in (" 502", " 503", " 504", " 524")):
            return True

        return False

    def _call_backup_nonstream(self) -> Tuple[str, Dict[str, Any]]:
        backup_model = self.backup_model_name or self.model_name
        return self._call_api_nonstream_with(
            base_url=self.backup_base_url or "",
            api_key=self.backup_api_key or "",
            model_name=backup_model,
            label="backup",
        )

    # ----------------- 对外 infer -----------------
    def infer(
        self,
    ) -> Tuple[
        str,
        Optional[Iterator[Tuple[Optional[str], Optional[dict]]]],
        Optional[Tuple[str, Optional[dict]]],
    ]:
        """
        对外统一接口：
        - want_streaming=True：先尝试流式，失败则本次回退非流式
            -> ("stream", iterator, None)
        - want_streaming=False 或回退：
            -> ("nonstream", None, (full_text, usage))
        """
        # 优先走流式
        if self.want_streaming:
            try:
                return "stream", self._call_api_stream(), None
            except Exception as e:
                logger.warning(
                    "[CloudVLMCurlClient] 流式调用失败，本次回退非流式: %s",
                    e,
                )
                if self._has_backup() and self._is_connection_error(e):
                    logger.warning("[CloudVLMCurlClient] 主服务不可达，切换到备用服务（非流式）")
                    try:
                        text, usage = self._call_backup_nonstream()
                        return "nonstream", None, (text, usage)
                    except Exception as fallback_err:
                        logger.error("[CloudVLMCurlClient] 备用服务调用失败: %s", fallback_err)

        t0 = time.time()
        attempt = 0
        last_err: Optional[Exception] = None
        final_text: str = ""

        while attempt <= VLM_HTTP_MAX_RETRY:
            attempt += 1
            try:
                final_text, usage = self._call_api_nonstream()
                # 补充统一字段
                usage.setdefault("backend", "cloud")
                usage.setdefault("model", self.model_name)
                usage.setdefault("status", "ok")
                usage.setdefault("attempts", attempt)
                usage.setdefault("elapsed_sec", time.time() - t0)
                return "nonstream", None, (final_text, usage)
            except Exception as e:
                last_err = e
                logger.error(
                    "[CloudVLMCurlClient] 非流式调用失败 attempt=%d/%d, error=%s",
                    attempt,
                    VLM_HTTP_MAX_RETRY + 1,
                    e,
                )
                if self._has_backup() and self._is_connection_error(e):
                    logger.warning("[CloudVLMCurlClient] 主服务不可达，切换到备用服务（非流式）")
                    try:
                        text, usage = self._call_backup_nonstream()
                        usage.setdefault("fallback_from", "cloud")
                        return "nonstream", None, (text, usage)
                    except Exception as fallback_err:
                        last_err = fallback_err
                if attempt > VLM_HTTP_MAX_RETRY:
                    break

        elapsed = time.time() - t0
        usage_err = {
            "backend": "cloud",
            "model": self.model_name,
            "status": "api_error",
            "attempts": attempt,
            "elapsed_sec": elapsed,
            "error": str(last_err) if last_err else "unknown_error",
        }
        # 出错时仍然返回 nonstream，交给上游处理
        return "nonstream", None, ("", usage_err)
