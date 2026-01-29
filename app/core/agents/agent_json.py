"""
JSON 智能体代理 - 简化版

专门用于处理JSON格式的智能体对话请求

特点：
1. 极简架构，直接调用LLM
2. 支持parameter替换系统提示词
3. 支持response_format一次性生成JSON
4. 无工具调用，纯结构化输出
"""

from dataclasses import dataclass
import logging
import re
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse, urlunparse

from app.exceptions import DatabaseUnavailableError
from app.core.llm.client import LLMClient
from app.core.llm.model_config_manager import get_model_config_manager
from app.services.agent_service import AgentService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfigSnapshot:
    id: str
    name: str
    type: str
    config: Dict[str, Any]


_AGENT_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def _now_s() -> float:
    return time.monotonic()


def _get_agent_cache_ttl_s() -> int:
    # Default 5 minutes; allow env override per deployment.
    import os

    raw = (os.getenv("AGENT_CONFIG_CACHE_TTL") or "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except Exception:
            return 300
    return 300


def _get_agent_stale_ttl_s() -> int:
    # Serve stale cache for a while when DB is temporarily unavailable.
    import os

    raw = (os.getenv("AGENT_CONFIG_CACHE_STALE_TTL") or "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except Exception:
            return 3600
    return 3600


def _cache_get_agent(agent_id: str) -> Optional[AgentConfigSnapshot]:
    entry = _AGENT_CONFIG_CACHE.get(agent_id)
    if not entry:
        return None
    expires_at = entry.get("expires_at") or 0.0
    if expires_at and expires_at > _now_s():
        return entry.get("value")
    return None


def _cache_get_agent_stale(agent_id: str) -> Optional[AgentConfigSnapshot]:
    entry = _AGENT_CONFIG_CACHE.get(agent_id)
    if not entry:
        return None
    fetched_at = entry.get("fetched_at") or 0.0
    if not fetched_at:
        return None
    if fetched_at + _get_agent_stale_ttl_s() > _now_s():
        return entry.get("value")
    return None


def _cache_set_agent(agent_id: str, value: AgentConfigSnapshot) -> None:
    ttl = _get_agent_cache_ttl_s()
    now = _now_s()
    _AGENT_CONFIG_CACHE[agent_id] = {
        "value": value,
        "fetched_at": now,
        "expires_at": now + ttl if ttl > 0 else 0.0,
    }


class JSONAgentProcessor:
    """JSON智能体代理 - 极简版"""

    def _build_json_output_instruction(self, json_schema: Optional[Dict[str, Any]]) -> str:
        """
        Build a strict "JSON only" instruction block for providers that don't
        support OpenAI `response_format` (or ignore it).
        """
        import json

        def _default_for(spec: Any, key: str) -> Any:
            if not isinstance(spec, dict):
                return None
            if "default" in spec:
                return spec.get("default")
            enum = spec.get("enum")
            if isinstance(enum, list) and enum:
                return enum[0]
            t = spec.get("type")
            if isinstance(t, list) and t:
                t = t[0]
            t = str(t or "").strip().lower()
            if t == "boolean":
                return False
            if t == "integer":
                return 0
            if t == "number":
                return 0
            if t == "string":
                return ""
            if t == "array":
                return []
            if t == "object":
                return {}
            return None

        lines = [
            "你必须只输出一个有效的 JSON 对象（仅 JSON，不要任何解释/前后缀/markdown/代码块）。",
            "要求：必须是 { ... } 形式；键和值字符串必须使用双引号；布尔值用 true/false；空值用 null；禁止尾随逗号。",
        ]

        allowed_keys: list[str] = []
        required_keys: list[str] = []
        example_obj: dict[str, Any] = {}

        if isinstance(json_schema, dict) and json_schema:
            props = json_schema.get("properties")
            if isinstance(props, dict) and props:
                allowed_keys = [str(k) for k in props.keys()]
                for k, spec in props.items():
                    example_obj[str(k)] = _default_for(spec, str(k))
            required = json_schema.get("required")
            if isinstance(required, list):
                required_keys = [str(k) for k in required]

            if allowed_keys:
                lines.append("只允许输出以下字段（不要输出其它字段）： " + ", ".join(allowed_keys))
            if required_keys:
                lines.append("必须包含字段： " + ", ".join(required_keys))
            if example_obj:
                lines.append("输出示例（仅示例，最终必须只输出 JSON 对象本体）：")
                lines.append(json.dumps(example_obj, ensure_ascii=False))

            schema_text = json.dumps(json_schema, ensure_ascii=False)
            lines.extend(
                [
                    "输出必须严格符合下面的 JSON Schema：",
                    schema_text,
                ]
            )

        lines.append("现在只输出 JSON 对象本体。")
        return "\n".join(lines)

    def _looks_like_multimodal_model(self, model: Optional[str]) -> bool:
        """
        Heuristic: determine whether a model name likely supports multimodal (image/video).

        We need this because some agents may be configured with a text-only model name,
        while the request contains `files` with `image`/`video`. In that case, we should
        fall back to the configured VLM model to avoid silently ignoring visual inputs.
        """
        m = (model or "").strip().lower()
        if not m:
            return False

        # Common multimodal/vision identifiers
        # (Keep conservative: only match well-known substrings.)
        multimodal_markers = [
            "-vl-",
            "vl-",
            "-vl",
            "vision",
            "4v",
            "4o",  # e.g. gpt-4o
            "omni",
        ]
        return any(marker in m for marker in multimodal_markers)

    def _has_visual_inputs(self, context: Dict[str, Any]) -> bool:
        """Return True if request context includes image/video inputs."""
        try:
            files = context.get("files") or []
            for f in files:
                if isinstance(f, dict):
                    media_type = (f.get("media_type") or "").strip().lower()
                else:
                    media_type = (getattr(f, "media_type", "") or "").strip().lower()
                if media_type in {"image", "video"}:
                    return True

            # Backward compatible fields
            if context.get("image_urls") or context.get("image_url"):
                return True
        except Exception:
            # If we cannot reliably determine, do not force multimodal.
            return False
        return False

    def _get_int_env(self, key: str, default: int) -> int:
        import os
        try:
            return int(os.getenv(key, default))
        except Exception:
            return default

    def _get_float_env(self, key: str, default: float) -> float:
        import os
        try:
            return float(os.getenv(key, default))
        except Exception:
            return default

    def _coerce_int(self, value: Any, default: int) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _get_media_tuning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        从请求 context 中读取多媒体调参，优先级：context > env > default
        """
        default_max_frames = self._get_int_env("VIDEO_MAX_FRAMES", 12)
        default_sample_fps = self._get_float_env("VIDEO_SAMPLE_FPS", 2.0)
        default_max_side = self._get_int_env("VIDEO_FRAME_MAX_SIDE", 768)
        default_jpeg_quality = self._get_int_env("VIDEO_JPEG_QUALITY", 75)

        max_frames = self._coerce_int(context.get("video_max_frames"), default_max_frames)
        sample_fps = self._coerce_float(context.get("video_sample_fps"), default_sample_fps)
        max_side = self._coerce_int(context.get("image_max_side"), default_max_side)
        jpeg_quality = self._coerce_int(context.get("image_jpeg_quality"), default_jpeg_quality)

        max_frames = max(1, min(32, max_frames))
        sample_fps = max(0.1, min(10.0, sample_fps))
        max_side = max(256, min(2048, max_side))
        jpeg_quality = max(30, min(95, jpeg_quality))

        return {
            "max_frames": max_frames,
            "sample_fps": sample_fps,
            "max_side": max_side,
            "jpeg_quality": jpeg_quality,
        }

    def _parse_roi_rect(self, roi: Any) -> Optional[Dict[str, Any]]:
        if roi is None:
            return None

        rect = None
        normalized = None

        if isinstance(roi, dict):
            normalized = roi.get("normalized")
            rect = roi.get("rect")
            if rect is None and all(k in roi for k in ("x1", "y1", "x2", "y2")):
                rect = [roi.get("x1"), roi.get("y1"), roi.get("x2"), roi.get("y2")]
            if rect is None and all(k in roi for k in ("x", "y", "w", "h")):
                x = roi.get("x")
                y = roi.get("y")
                w = roi.get("w")
                h = roi.get("h")
                rect = [x, y, None if w is None else x + w, None if h is None else y + h]
            if rect is None and all(k in roi for k in ("left", "top", "right", "bottom")):
                rect = [roi.get("left"), roi.get("top"), roi.get("right"), roi.get("bottom")]
        elif isinstance(roi, (list, tuple)) and len(roi) >= 4:
            rect = roi

        if not rect:
            return None

        try:
            x1 = float(rect[0])
            y1 = float(rect[1])
            x2 = float(rect[2])
            y2 = float(rect[3])
        except Exception:
            return None

        if normalized is None:
            vals = [x1, y1, x2, y2]
            normalized = min(vals) >= 0.0 and max(vals) <= 1.0
        elif isinstance(normalized, str):
            normalized = normalized.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            normalized = bool(normalized)

        return {"rect": (x1, y1, x2, y2), "normalized": normalized}

    def _apply_roi_to_frame(self, frame: Any, roi: Optional[Dict[str, Any]]) -> Any:
        if frame is None or not roi:
            return frame

        rect = roi.get("rect") if isinstance(roi, dict) else None
        if not rect:
            return frame

        try:
            x1, y1, x2, y2 = rect
        except Exception:
            return frame

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

        if roi.get("normalized"):
            x1 *= w
            x2 *= w
            y1 *= h
            y2 *= h

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1_i = max(0, min(int(round(x1)), w - 1))
        y1_i = max(0, min(int(round(y1)), h - 1))
        x2_i = max(0, min(int(round(x2)), w))
        y2_i = max(0, min(int(round(y2)), h))

        if x2_i <= x1_i or y2_i <= y1_i:
            logger.warning("⚠️ [ROI] Invalid rect after clamp, skip crop: %s", rect)
            return frame

        return frame[y1_i:y2_i, x1_i:x2_i]

    def _looks_like_video_stream_url(self, url: str) -> bool:
        """
        判断一个 URL 是否更像“视频流”（如 rtsp/rtmp/hls m3u8）。

        用于在多模态视频处理时选择更合适的抓帧方式（ffmpeg 直抓帧）。
        """
        try:
            parsed = urlparse(url)
            scheme = (parsed.scheme or "").lower().strip()
            if scheme in {"rtsp", "rtsps", "rtmp", "rtmps"}:
                return True
            lower = url.lower()
            if ".m3u8" in lower:
                return True
        except Exception:
            return False
        return False

    def _download_to_tempfile(
        self,
        url: str,
        *,
        timeout: int,
        suffix: str,
        max_size_mb: Optional[int] = None
    ) -> str:
        import tempfile
        import requests
        import time
        import os

        # 默认不使用环境变量里的代理（HTTP_PROXY/HTTPS_PROXY），避免内网代理对大文件/二进制做截断
        # 如确实需要走代理，可设置 VIDEO_DOWNLOAD_TRUST_ENV=1
        session = requests.Session()
        session.trust_env = os.getenv("VIDEO_DOWNLOAD_TRUST_ENV", "").strip().lower() in {"1", "true", "yes", "y"}

        headers = {
            # 尽量模拟浏览器下载行为，避免某些内网网关/代理对非浏览器请求做截断
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "*/*",
            # 避免对二进制内容做压缩/解压造成中间层异常
            "Accept-Encoding": "identity",
            # 有些代理对 keep-alive 长连接更容易提前断开
            "Connection": "close",
        }
        referer = os.getenv("VIDEO_DOWNLOAD_REFERER", "").strip()
        if referer:
            headers["Referer"] = referer

        max_bytes = None
        if max_size_mb and max_size_mb > 0:
            max_bytes = max_size_mb * 1024 * 1024

        attempts = max(1, self._get_int_env("VIDEO_DOWNLOAD_RETRIES", 3))
        connect_timeout = max(1, self._get_int_env("VIDEO_DOWNLOAD_CONNECT_TIMEOUT", 10))
        range_chunk_mb = max(1, self._get_int_env("VIDEO_DOWNLOAD_CHUNK_MB", 4))
        range_chunk_bytes = range_chunk_mb * 1024 * 1024
        range_threshold_mb = max(1, self._get_int_env("VIDEO_DOWNLOAD_RANGE_THRESHOLD_MB", 2))
        range_threshold_bytes = range_threshold_mb * 1024 * 1024
        force_range = os.getenv("VIDEO_DOWNLOAD_FORCE_RANGE", "").strip().lower() in {"1", "true", "yes", "y"}

        download_url = url
        expected_bytes: Optional[int] = None
        accept_ranges = ""

        # 先用 HEAD 获取真实大小/是否支持 Range（失败则降级）
        try:
            head_resp = session.head(
                url,
                headers=headers,
                timeout=(connect_timeout, timeout),
                verify=False,
                allow_redirects=True,
            )
            if head_resp.ok:
                download_url = head_resp.url or url
                accept_ranges = (head_resp.headers.get("Accept-Ranges") or "").lower()
                cl = head_resp.headers.get("Content-Length")
                if cl:
                    try:
                        expected_bytes = int(cl)
                    except Exception:
                        expected_bytes = None
        except Exception:
            pass

        # HEAD 没拿到大小，尝试用 0-0 的 Range 探测（很多服务器会返回 Content-Range: bytes 0-0/total）
        if expected_bytes is None:
            try:
                probe_headers = dict(headers)
                probe_headers["Range"] = "bytes=0-0"
                with session.get(
                    download_url,
                    headers=probe_headers,
                    timeout=(connect_timeout, timeout),
                    verify=False,
                    stream=True,
                    allow_redirects=True,
                ) as probe:
                    if probe.status_code == 206:
                        accept_ranges = (probe.headers.get("Accept-Ranges") or "bytes").lower()
                        cr = probe.headers.get("Content-Range") or ""
                        m = re.search(r"/(\d+)$", cr)
                        if m:
                            expected_bytes = int(m.group(1))
            except Exception:
                pass

        if expected_bytes is not None and max_bytes and expected_bytes > max_bytes:
            raise ValueError(f"视频超过大小限制: {expected_bytes} bytes > {max_bytes} bytes")

        last_err: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    downloaded = 0

                    should_use_range = (
                        expected_bytes is not None
                        and ("bytes" in (accept_ranges or "" ) or force_range)
                        and (force_range or expected_bytes >= range_threshold_bytes)
                    )

                    if should_use_range:
                        pos = 0
                        while expected_bytes is not None and pos < expected_bytes:
                            end = min(pos + range_chunk_bytes - 1, expected_bytes - 1)
                            part_headers = dict(headers)
                            part_headers["Range"] = f"bytes={pos}-{end}"

                            with session.get(
                                download_url,
                                headers=part_headers,
                                timeout=(connect_timeout, timeout),
                                verify=False,
                                stream=True,
                                allow_redirects=True,
                            ) as resp:
                                # 206 = Partial Content (range ok), 200 = server ignored range (兜底)
                                if resp.status_code not in (200, 206):
                                    resp.raise_for_status()

                                if pos > 0 and resp.status_code == 200:
                                    raise IOError("服务器忽略 Range，无法继续分段下载")

                                content_type = (resp.headers.get("Content-Type") or "").lower()
                                if "text/html" in content_type or "application/json" in content_type:
                                    raise ValueError(f"下载内容类型异常: {content_type}")

                                wrote = 0
                                first_chunk = (pos == 0)
                                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                    if not chunk:
                                        continue
                                    if first_chunk:
                                        first_chunk = False
                                        # mp4 快速签名检查：4 bytes size + "ftyp"
                                        if suffix == ".mp4" and len(chunk) >= 12 and chunk[4:8] != b"ftyp":
                                            raise ValueError("下载内容不像 mp4（缺少 ftyp header）")

                                    wrote += len(chunk)
                                    downloaded += len(chunk)
                                    if max_bytes and downloaded > max_bytes:
                                        raise ValueError(f"视频超过大小限制: >{max_size_mb}MB")
                                    tmp_file.write(chunk)

                                if resp.status_code == 206:
                                    expected_part = end - pos + 1
                                    if wrote != expected_part:
                                        raise IOError(
                                            f"下载不完整(range): {pos}-{end} got={wrote} expected={expected_part}"
                                        )

                            pos = end + 1
                    else:
                        with session.get(
                            download_url,
                            headers=headers,
                            timeout=(connect_timeout, timeout),
                            verify=False,
                            stream=True,
                            allow_redirects=True,
                        ) as resp:
                            resp.raise_for_status()

                            content_type = (resp.headers.get("Content-Type") or "").lower()
                            if "text/html" in content_type or "application/json" in content_type:
                                raise ValueError(f"下载内容类型异常: {content_type}")

                            first_chunk = True
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                if first_chunk:
                                    first_chunk = False
                                    # mp4 快速签名检查：4 bytes size + "ftyp"
                                    if suffix == ".mp4" and len(chunk) >= 12 and chunk[4:8] != b"ftyp":
                                        raise ValueError("下载内容不像 mp4（缺少 ftyp header）")

                                downloaded += len(chunk)
                                if max_bytes and downloaded > max_bytes:
                                    raise ValueError(f"视频超过大小限制: >{max_size_mb}MB")
                                tmp_file.write(chunk)

                if expected_bytes is not None and downloaded != expected_bytes:
                    raise IOError(f"下载不完整: {downloaded}/{expected_bytes} bytes")

                # 过小文件基本不可能是有效视频（常见为错误页/截断）
                if suffix == ".mp4" and downloaded < 64 * 1024:
                    raise IOError(f"下载视频过小: {downloaded} bytes")

                return tmp_path
            except Exception as e:
                last_err = e
                if tmp_path:
                    try:
                        import os
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                if attempt < attempts:
                    time.sleep(min(5.0, 0.5 * attempt))
                    continue
                break

        raise last_err or RuntimeError("视频下载失败")

    def _encode_frame_to_jpeg_base64(
        self,
        frame,
        *,
        max_side: int,
        quality: int,
        roi: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            import base64
            import cv2

            if frame is None:
                return None

            if roi:
                frame = self._apply_roi_to_frame(frame, roi)
                if frame is None:
                    return None

            h, w = frame.shape[:2]
            if max_side and max(h, w) > max_side:
                if h >= w:
                    new_h = max_side
                    new_w = int(w * (max_side / h))
                else:
                    new_w = max_side
                    new_h = int(h * (max_side / w))
                frame = cv2.resize(frame, (max(1, new_w), max(1, new_h)), interpolation=cv2.INTER_AREA)

            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if not ok:
                return None
            return base64.b64encode(buffer).decode('utf-8')
        except Exception:
            return None

    def _compress_image_bytes_to_jpeg_base64(
        self,
        image_bytes: bytes,
        *,
        max_side: int,
        quality: int,
        roi: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            import numpy as np
            import cv2

            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            return self._encode_frame_to_jpeg_base64(img, max_side=max_side, quality=quality, roi=roi)
        except Exception:
            return None

    async def process_message(
        self,
        message: str,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
        parameter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        处理JSON格式的消息请求

        Args:
            message: 用户消息
            agent_id: 智能体ID
            context: 上下文信息（运行时参数：image_urls, json_schema等）
            parameter: 系统提示词参数（用于{{variable}}替换）

        Returns:
            完整的AI响应文本（JSON格式）
        """
        try:
            logger.info(f"🚀 [JSON代理] 开始处理消息")
            logger.info(f"  智能体ID: {agent_id}")
            logger.info(f"  用户消息: '{message}'")
            logger.info(f"  消息长度: {len(message)}")
            logger.info(f"  上下文: {context}")
            if parameter:
                logger.info(f"  📝 系统参数: {parameter}")

            # 1. 获取智能体配置
            agent_config = await self._get_agent_config(agent_id)

            # 2. 获取系统提示词并应用parameter替换
            system_prompt = self._build_system_prompt(
                agent_config=agent_config,
                parameter=parameter
            )
            logger.info(f"📋 [JSON代理] 系统提示词长度: {len(system_prompt)}")
            if len(system_prompt) > 200:
                logger.info(f"📋 [JSON代理] 系统提示词预览: {system_prompt[:200]}...")
            else:
                logger.info(f"📋 [JSON代理] 系统提示词: {system_prompt}")

            # 3. 构建消息
            messages = await self._build_messages(
                system_prompt=system_prompt,
                message=message,
                context=context or {}
            )
            logger.info(f"📨 [JSON代理] 构建的消息数量: {len(messages)}")
            for i, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if isinstance(content, str):
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                elif isinstance(content, list):
                    content_preview = f"[多模态内容，{len(content)}项]"
                else:
                    content_preview = f"[{type(content).__name__}]"
                logger.info(f"📨 [JSON代理] 消息{i+1} ({role}): {content_preview}")

            # 4. 准备response_format（如果有json_schema）
            response_format = self._build_response_format(context or {})
            logger.info(f"🎯 [JSON代理] Response格式: {response_format}")

            # 5. 选择模型
            model = self._select_model(agent_config, context or {})
            logger.info(f"🤖 [JSON代理] 选择模型: {model}")
            logger.info(f"🌡️ [JSON代理] 温度: {agent_config.config.get('temperature', 0.3)}")
            logger.info(f"📏 [JSON代理] 最大tokens: {agent_config.config.get('max_tokens', 2048)}")

            # 5.1 记录当前模型配置来源（便于排查“模型不存在”类错误）
            cfg_manager = get_model_config_manager()
            cfg = getattr(cfg_manager, "_config", None)
            if cfg:
                logger.info(
                    "[JSON代理] 当前模型配置 | "
                    f"LLM: {cfg.llm_config.provider_name} {cfg.llm_config.base_url} {cfg.llm_config.model_name} | "
                    f"VLM: {cfg.vlm_config.provider_name} {cfg.vlm_config.base_url} {cfg.vlm_config.model_name} | "
                    f"Embedding: {cfg.embedding_config.provider_name} {cfg.embedding_config.base_url} {cfg.embedding_config.model_name}"
                )
            else:
                logger.warning("[JSON代理] ModelConfigManager 尚未初始化，可能未加载 .env")

            # 6. 调用LLM
            llm_client = LLMClient(provider="unified")
            logger.info(f"📡 [JSON代理] 开始调用LLM...")
            response = await llm_client.chat(
                messages=messages,
                model=model,
                temperature=agent_config.config.get('temperature', 0.3),
                max_tokens=agent_config.config.get('max_tokens', 2048),
                response_format=response_format
            )

            logger.info(f"✅ [JSON代理] 消息处理完成，响应长度: {len(response)}")
            logger.info(f"📄 [JSON代理] 完整响应: {response}")
            return response

        except Exception as e:
            logger.error(f"❌ [JSON代理] 处理失败: {e}", exc_info=True)
            raise

    async def _get_agent_config(self, agent_id: str) -> AgentConfigSnapshot:
        """获取智能体配置"""
        cached = _cache_get_agent(agent_id)
        if cached:
            logger.debug("📦 [配置] 命中缓存: agent_id=%s", agent_id)
            return cached

        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            agent_service = AgentService(db)
            agent = agent_service.get_agent(agent_id)
            if not agent:
                raise ValueError(f"智能体 {agent_id} 不存在")

            cfg = getattr(agent, "config", None) or {}
            if isinstance(cfg, str):
                try:
                    import json

                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {}

            snapshot = AgentConfigSnapshot(
                id=str(getattr(agent, "id", "") or agent_id),
                name=str(getattr(agent, "name", "") or agent_id),
                type=str(getattr(agent, "type", "") or "unknown"),
                config=cfg if isinstance(cfg, dict) else {},
            )
            _cache_set_agent(agent_id, snapshot)
            logger.info("📋 [配置] 智能体: %s (类型: %s)", snapshot.name, snapshot.type)
            return snapshot
        except DatabaseUnavailableError:
            stale = _cache_get_agent_stale(agent_id)
            if stale:
                logger.warning("⚠️ [配置] DB不可用，使用缓存(可能过期): agent_id=%s", agent_id)
                return stale
            raise
        finally:
            try:
                db.close()
            except Exception:
                pass

    def _build_system_prompt(
        self,
        agent_config: Any,
        parameter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建系统提示词并应用parameter替换

        Args:
            agent_config: 智能体配置
            parameter: 参数字典，用于替换{{variable}}

        Returns:
            最终的系统提示词
        """
        config = agent_config.config or {}

        # 修复：使用正确的字段名，支持多种可能的字段名
        system_prompt = (
            config.get('system_prompt') or
            config.get('prompt') or
            config.get('description') or
            ''
        )

        # 调试：打印完整的配置信息
        logger.info(f"🔧 [配置调试] 完整的agent配置: {config}")
        logger.info(f"🔧 [配置调试] 配置中的键: {list(config.keys())}")
        logger.info(f"🔧 [配置调试] 使用的系统提示词字段: {system_prompt[:100] if len(system_prompt) > 100 else system_prompt}")

        # 应用parameter替换
        if parameter:
            system_prompt = self._apply_parameter_replacement(system_prompt, parameter)

        return system_prompt

    def _apply_parameter_replacement(
        self,
        prompt: str,
        parameter: Dict[str, Any]
    ) -> str:
        """
        使用parameter替换系统提示词中的模板变量

        支持的占位符格式：{{variable_name}}

        Args:
            prompt: 原始系统提示词（包含{{}}占位符）
            parameter: 参数字典，用于替换占位符

        Returns:
            替换后的系统提示词
        """
        enhanced_prompt = prompt
        replaced_vars = []

        for key, value in parameter.items():
            pattern = r'\{\{' + re.escape(key) + r'\}\}'
            if re.search(pattern, enhanced_prompt):
                enhanced_prompt = re.sub(pattern, str(value), enhanced_prompt)
                replaced_vars.append(key)

        if replaced_vars:
            logger.info(f"📝 [提示词增强] 替换了 {len(replaced_vars)} 个变量: {replaced_vars}")
        else:
            logger.info(f"📝 [提示词增强] 未找到可替换的变量占位符")

        return enhanced_prompt

    async def _build_messages(
        self,
        system_prompt: str,
        message: str,
        context: Dict[str, Any]
    ) -> list:
        """
        构建消息列表

        Args:
            system_prompt: 系统提示词
            message: 用户消息
            context: 上下文（可能包含files、image_urls等）

        Returns:
            消息列表
        """
        # Some OpenAI-compatible backends may not handle multiple system messages properly.
        # Merge JSON-only instruction into the *first* system prompt to maximize compliance.
        json_schema = context.get("json_schema")
        output_format = str(context.get("output_format") or "").strip().lower()
        if output_format == "json" or json_schema:
            system_prompt = (system_prompt or "").rstrip() + "\n\n" + self._build_json_output_instruction(
                json_schema if isinstance(json_schema, dict) else None
            )

        messages = [{"role": "system", "content": system_prompt}]

        # 优先使用files字段（新格式），兼容旧的image_urls
        files = context.get('files')
        image_urls = context.get('image_urls') or context.get('image_url')

        logger.info(f"🔍 [MessageBuilder] context包含: files={bool(files)}, image_urls={bool(image_urls)}")
        if files:
            logger.info(f"  📁 files数量: {len(files)}, 内容: {files}")
        if image_urls:
            logger.info(f"  🖼️ image_urls数量: {len(image_urls) if isinstance(image_urls, list) else 1}, 内容: {image_urls}")

        if files:
            # 使用新的files格式（支持视频和图片）
            logger.info(f"📥 [MessageBuilder] 使用files字段构建多模态内容")
            user_content = await self._build_multimodal_content_from_files(
                files=files,
                text=message,
                context=context,
            )
        elif image_urls:
            # 兼容旧的image_urls格式
            logger.info(f"📥 [MessageBuilder] 使用image_urls字段构建多模态内容")
            if isinstance(image_urls, str):
                image_urls = [image_urls]
            if context.get("roi"):
                roi_files = [{"url": u, "media_type": "image"} for u in image_urls]
                user_content = await self._build_multimodal_content_from_files(
                    files=roi_files,
                    text=message,
                    context=context,
                )
            else:
                user_content = await self._build_multimodal_content(image_urls, message)
        else:
            # 纯文本消息
            logger.info(f"📝 [MessageBuilder] 纯文本消息")
            user_content = message

        messages.append({"role": "user", "content": user_content})
        logger.info(f"✅ [MessageBuilder] 消息构建完成，content类型: {type(user_content).__name__}")
        return messages

    async def _build_multimodal_content_from_files(
        self,
        *,
        files: list,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        从files列表构建多模态内容（支持图片和视频）

        Args:
            files: 文件列表，每个文件包含url和media_type
                  例如: [{"url": "http://...", "media_type": "image"},
                        {"url": "http://...", "media_type": "video"}]
            text: 文本内容

        Returns:
            多模态内容列表
        """
        import asyncio
        import base64
        import os
        import requests
        from mimetypes import guess_type
        from urllib.parse import unquote, urlparse

        content = [{"type": "text", "text": text or "请分析这些内容"}]
        roi = self._parse_roi_rect((context or {}).get("roi"))
        if roi:
            logger.info("🔍 [Multimodal] ROI enabled: %s (normalized=%s)", roi.get("rect"), roi.get("normalized"))

        def _resolve_local_path(raw_url: str) -> Optional[str]:
            if not isinstance(raw_url, str):
                return None
            s = raw_url.strip()
            if not s:
                return None
            if s.startswith("file://"):
                parsed = urlparse(s)
                path_str = unquote(parsed.path or "")
                if os.name == "nt" and path_str.startswith("/"):
                    path_str = path_str.lstrip("/")
                candidate = path_str
            else:
                candidate = os.path.expanduser(s)
            if os.path.isabs(candidate) and os.path.isfile(candidate):
                return candidate
            return None

        for idx, file_item in enumerate(files, 1):
            try:
                url = file_item.get('url') if isinstance(file_item, dict) else file_item.url
                media_type = file_item.get('media_type') if isinstance(file_item, dict) else file_item.media_type

                logger.info(f"📥 [Multimodal] 处理 {media_type} {idx}/{len(files)}: {url[:100]}...")

                if isinstance(media_type, str):
                    media_type = media_type.strip().lower()
                if not media_type and isinstance(url, str):
                    guessed = guess_type(url)[0] or ""
                    if guessed.startswith("video"):
                        media_type = "video"
                    elif guessed.startswith("image"):
                        media_type = "image"

                tuning = self._get_media_tuning(context or {})
                max_frames = tuning["max_frames"]
                sample_fps = tuning["sample_fps"]
                max_side = tuning["max_side"]
                jpeg_quality = tuning["jpeg_quality"]
                max_video_size_mb = self._get_int_env("VIDEO_MAX_SIZE_MB", 500)
                is_stream_video = (media_type == 'video') and self._looks_like_video_stream_url(url)
                local_path = _resolve_local_path(url) if isinstance(url, str) else None
                tmp_video_path = None
                tmp_video_is_temp = False

                # 处理 data URL（base64编码的数据）
                if url.startswith('data:'):
                    logger.info(f"🔄 [Multimodal] 检测到 data: 协议，解析base64数据")
                    try:
                        # 解析 data URL: data:image/jpeg;base64,xxxxx
                        import re
                        match = re.match(r'data:([^;]+);base64,(.+)', url)
                        if not match:
                            logger.error(f"❌ [Multimodal] 无效的 data URL 格式")
                            content[0]["text"] += f"\n\n注意：文件 {idx} data URL格式无效"
                            continue

                        mime_type = match.group(1)
                        base64_data = match.group(2)
                        file_content = base64.b64decode(base64_data)

                        file_size_mb = len(file_content) / (1024 * 1024)
                        logger.info(f"✅ [Multimodal] data URL解析完成，文件大小: {file_size_mb:.2f} MB")
                    except Exception as e:
                        logger.error(f"❌ [Multimodal] data URL解析失败: {e}")
                        content[0]["text"] += f"\n\n注意：文件 {idx} data URL解析失败"
                        continue

                # 处理本地文件路径
                elif local_path:
                    if media_type == 'video':
                        try:
                            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                        except Exception:
                            file_size_mb = -1
                        tmp_video_path = local_path
                        logger.info(f"✅ [Multimodal] 本地视频加载完成，文件大小: {file_size_mb:.2f} MB")
                    else:
                        try:
                            with open(local_path, "rb") as f:
                                file_content = f.read()
                            file_size_mb = len(file_content) / (1024 * 1024)
                            logger.info(f"✅ [Multimodal] 本地图片加载完成，文件大小: {file_size_mb:.2f} MB")
                        except Exception as e:
                            logger.error(f"❌ [Multimodal] 本地文件读取失败: {e}")
                            content[0]["text"] += f"\n\n注意：文件 {idx} 读取失败"
                            continue

                # 处理 HTTP/HTTPS URL
                elif url.startswith('http://') or url.startswith('https://'):
                    # 允许通过环境变量覆盖下载超时（默认视频120s、图片30s）
                    video_timeout = self._get_int_env("VIDEO_DOWNLOAD_TIMEOUT", 120)
                    image_timeout = self._get_int_env("IMAGE_DOWNLOAD_TIMEOUT", 30)
                    timeout = video_timeout if media_type == 'video' else image_timeout
                    if media_type == 'video':
                        if is_stream_video:
                            logger.info(f"⏱️ [Multimodal] 检测到视频流URL，改用 ffmpeg 抓帧: {url[:120]}")
                        else:
                            logger.info(f"⏱️ [Multimodal] 开始流式下载视频，超时时间: {timeout}秒")
                            tmp_video_path = None
                            try:
                                tmp_video_path = await asyncio.to_thread(
                                    self._download_to_tempfile,
                                    url,
                                    timeout=timeout,
                                    suffix=".mp4",
                                    max_size_mb=max_video_size_mb,
                                )
                                tmp_video_is_temp = True
                                try:
                                    import os
                                    file_size_mb = os.path.getsize(tmp_video_path) / (1024 * 1024)
                                except Exception:
                                    file_size_mb = -1
                                logger.info(f"✅ [Multimodal] 视频下载完成，文件大小: {file_size_mb:.2f} MB")
                            except Exception as e:
                                logger.error(f"❌ [Multimodal] 视频下载失败: {e}")
                                # 下载失败时兜底尝试 ffmpeg 直抓帧（尤其是 HLS/流媒体）
                                if self._looks_like_video_stream_url(url):
                                    logger.warning(f"⚠️ [Multimodal] 视频下载失败，尝试按视频流处理: {url[:120]}")
                                    is_stream_video = True
                                else:
                                    content[0]["text"] += f"\n\n注意：视频 {idx} 下载失败"
                                    continue
                    else:
                        logger.info(f"⏱️ [Multimodal] 开始下载图片，超时时间: {timeout}秒")
                        try:
                            response = await asyncio.to_thread(
                                requests.get, url, timeout=timeout, verify=False
                            )
                            response.raise_for_status()
                        except requests.exceptions.ConnectionError as e:
                            proxy_url = self._rewrite_to_proxy(url)
                            if proxy_url:
                                logger.warning(f"⚠️ [Multimodal] 原地址不可达，尝试代理地址: {proxy_url}")
                                try:
                                    response = await asyncio.to_thread(
                                        requests.get,
                                        proxy_url,
                                        timeout=timeout,
                                        verify=False,
                                    )
                                    response.raise_for_status()
                                    url = proxy_url  # 记录使用的代理地址
                                except Exception as e2:
                                    logger.error(f"❌ [Multimodal] 代理地址仍不可达: {e2}")
                                    raise e
                            else:
                                raise e

                        file_content = response.content
                        file_size_mb = len(file_content) / (1024 * 1024)
                        logger.info(f"✅ [Multimodal] 下载完成，文件大小: {file_size_mb:.2f} MB")

                else:
                    if media_type == 'video' and is_stream_video:
                        logger.info(f"⏱️ [Multimodal] 视频流URL，准备抓帧: {url[:120]}")
                    else:
                        logger.error(f"❌ [Multimodal] 不支持的URL格式: {url[:50]}")
                        content[0]["text"] += f"\n\n注意：文件 {idx} URL格式不支持"
                        continue

                # 根据media_type处理
                if media_type == 'video':
                    tmp_path = None
                    cleanup_tmp = False
                    try:
                        if url.startswith('data:'):
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                                tmp_file.write(file_content)
                                tmp_path = tmp_file.name
                            cleanup_tmp = True
                        elif is_stream_video:
                            video_frames = await self._extract_video_frames_from_stream(
                                url=url,
                                max_frames=max_frames,
                                sample_fps=sample_fps,
                                max_side=max_side,
                                jpeg_quality=jpeg_quality,
                                roi=roi,
                            )
                            if video_frames:
                                logger.info(f"🎬 [Video] (stream) 提取了 {len(video_frames)} 帧")
                                for frame_base64 in video_frames:
                                    content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{frame_base64}"
                                        }
                                    })
                                content[0]["text"] += f"\n\n[视频 {idx}: 已从视频流提取 {len(video_frames)} 个关键帧进行分析]"
                            else:
                                logger.warning(f"⚠️ [Video] 视频流 {idx} 抓帧失败")
                                content[0]["text"] += f"\n\n注意：视频流 {idx} 拉取/抓帧失败"
                            continue
                        else:
                            tmp_path = tmp_video_path  # type: ignore[name-defined]
                            cleanup_tmp = bool(tmp_video_is_temp)  # type: ignore[name-defined]
                            if not tmp_path:
                                logger.warning("⚠️ [Video] 未获取到可用的视频路径")
                                content[0]["text"] += f"\n\n注意：视频 {idx} 文件不可用"
                                continue

                        video_frames = await self._extract_video_frames(
                            video_path=tmp_path,
                            url=url,
                            max_frames=max_frames,
                            sample_fps=sample_fps,
                            max_side=max_side,
                            jpeg_quality=jpeg_quality,
                            roi=roi,
                        )

                        if video_frames:
                            logger.info(f"🎬 [Video] 提取了 {len(video_frames)} 帧")
                            for frame_base64 in video_frames:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame_base64}"
                                    }
                                })
                            content[0]["text"] += f"\n\n[视频 {idx}: 已提取 {len(video_frames)} 个关键帧进行分析]"
                        else:
                            logger.warning(f"⚠️ [Video] 视频 {idx} 帧提取失败")
                            content[0]["text"] += f"\n\n注意：视频 {idx} 处理失败"
                    finally:
                        if tmp_path and cleanup_tmp:
                            try:
                                import os
                                os.unlink(tmp_path)
                            except Exception:
                                pass

                elif media_type == 'image':
                    # 图片处理：压缩以提速与降低传输体积（无法解码时fallback为原始base64）
                    compressed = self._compress_image_bytes_to_jpeg_base64(
                        file_content,
                        max_side=max_side,
                        quality=jpeg_quality,
                        roi=roi,
                    )
                    if compressed:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{compressed}"
                            }
                        })
                    else:
                        if url.startswith('data:'):
                            content_type = mime_type or 'image/jpeg'
                        else:
                            content_type = guess_type(url)[0] or 'image/jpeg'
                        image_base64 = base64.b64encode(file_content).decode('utf-8')
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content_type};base64,{image_base64}"
                            }
                        })
                    logger.info(f"✅ [Image] 图片 {idx} 完成(压缩={bool(compressed)})")
                else:
                    logger.warning(f"⚠️ [Multimodal] 不支持的媒体类型: {media_type}")
                    content[0]["text"] += f"\n\n注意：文件 {idx} 类型不支持({media_type})"

            except Exception as e:
                logger.error(f"❌ [Multimodal] 文件 {idx} 加载失败: {type(e).__name__}: {str(e)}")
                logger.error(f"   URL: {url if 'url' in locals() else 'N/A'}")
                logger.error(f"   Media Type: {media_type if 'media_type' in locals() else 'N/A'}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                content[0]["text"] += f"\n\n注意：文件 {idx} 加载失败 ({type(e).__name__})"

        return content

    async def _extract_video_frames_from_stream(
        self,
        *,
        url: str,
        max_frames: int = 12,
        sample_fps: float = 2.0,
        max_side: int = 768,
        jpeg_quality: int = 75,
        roi: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        使用 ffmpeg 从视频流/网络视频中直接抓帧（不落整段视频）。

        适用场景：
        - rtsp/rtmp 等实时流
        - hls(m3u8) 等分片流
        - 部分 OpenCV 无法解码的视频 URL
        """
        import asyncio
        import base64
        import os
        import shutil
        import tempfile
        from pathlib import Path

        def _resolve_local_hls_playlist(stream_url: str) -> Optional[str]:
            try:
                parsed = urlparse(stream_url)
                scheme = (parsed.scheme or "").lower()
                if scheme not in {"rtsp", "rtsps", "rtmp", "rtmps"}:
                    return None
                from hashlib import sha1

                project_root = Path(__file__).resolve().parents[3]
                hls_root = project_root / "storage" / "static" / "hls"
                stream_id = f"stream_{sha1(stream_url.encode('utf-8')).hexdigest()[:12]}"
                playlist_path = hls_root / stream_id / "index.m3u8"
                if playlist_path.exists() and playlist_path.stat().st_size > 0:
                    return str(playlist_path)
            except Exception:
                return None
            return None

        ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
        if not shutil.which(ffmpeg_bin):
            logger.warning("⚠️ [Video] ffmpeg 未安装或不可用，无法抓帧：%s", ffmpeg_bin)
            return []

        max_frames = max(1, int(max_frames))
        sample_fps = max(0.1, float(sample_fps))

        # 抓帧总时长兜底（避免流一直不出帧导致等待过久）
        default_capture_seconds = int(max(4.0, min(20.0, (max_frames / sample_fps) + 2.0)))
        capture_seconds = max(2, self._get_int_env("VIDEO_STREAM_CAPTURE_SECONDS", default_capture_seconds))
        timeout = max(5, self._get_int_env("VIDEO_STREAM_FFMPEG_TIMEOUT", max(20, capture_seconds + 10)))

        tmp_dir = Path(tempfile.mkdtemp(prefix="jetlinks_stream_frames_"))
        try:
            out_pattern = str(tmp_dir / "frame_%03d.jpg")

            capture_url = url
            hls_path = _resolve_local_hls_playlist(url)
            if hls_path:
                logger.info("🎞️ [Video] 使用本地HLS抓帧: %s", hls_path)
                capture_url = hls_path

            # RTSP 默认走 TCP，避免 UDP 被防火墙拦截导致拉流失败
            input_args = []
            try:
                scheme = urlparse(capture_url).scheme.lower()
            except Exception:
                scheme = ""
            if scheme in {"rtsp", "rtsps"}:
                input_args += ["-rtsp_transport", "tcp"]

            cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                *input_args,
                "-i",
                capture_url,
                "-an",
                "-t",
                str(capture_seconds),
                "-vf",
                f"fps={sample_fps}",
                "-frames:v",
                str(max_frames),
                out_pattern,
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                logger.error("❌ [Video] ffmpeg 抓帧超时：%ss url=%s", timeout, url)
                return []

            if proc.returncode != 0:
                err_text = ((stderr or b"") + (stdout or b"")).decode("utf-8", errors="replace")
                logger.error(
                    "❌ [Video] ffmpeg 抓帧失败(rc=%s): %s",
                    proc.returncode,
                    err_text[:4000],
                )
                if capture_url != url:
                    logger.error("❌ [Video] 抓帧失败源: %s", url)
                return []

            frames = sorted(tmp_dir.glob("frame_*.jpg"))
            if not frames:
                logger.warning("⚠️ [Video] ffmpeg 未输出任何帧：url=%s", capture_url)
                return []

            frames_base64: list[str] = []
            for p in frames:
                try:
                    img_bytes = p.read_bytes()
                    compressed = self._compress_image_bytes_to_jpeg_base64(
                        img_bytes,
                        max_side=max_side,
                        quality=jpeg_quality,
                        roi=roi,
                    )
                    if compressed:
                        frames_base64.append(compressed)
                    else:
                        frames_base64.append(base64.b64encode(img_bytes).decode("utf-8"))
                except Exception:
                    continue

            return frames_base64[:max_frames]
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    async def _extract_video_frames(
        self,
        *,
        video_path: str,
        url: str,
        max_frames: int = 12,
        sample_fps: float = 2.0,
        max_side: int = 768,
        jpeg_quality: int = 75,
        roi: Optional[Dict[str, Any]] = None,
    ) -> list:
        """
        从视频内容提取关键帧

        Args:
            video_path: 视频文件路径
            url: 视频URL（用于日志）
            max_frames: 最大提取帧数

        Returns:
            帧的base64编码列表
        """
        import os

        try:
            # 尝试导入cv2
            try:
                import cv2
                import numpy as np
            except ImportError:
                logger.warning("⚠️ [Video] OpenCV未安装，无法处理视频")
                return []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("❌ [Video] 无法打开视频文件")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            duration = (total_frames / fps) if (fps > 0 and total_frames > 0) else 0.0
            logger.info(f"🎬 [Video] 视频信息: {total_frames}帧, {fps:.2f}fps, {duration:.2f}秒")

            if total_frames <= 0:
                cap.release()
                logger.error("❌ [Video] 无法读取视频总帧数")
                return []

            max_frames = max(1, int(max_frames))
            duration_ms = max(1.0, duration * 1000.0) if duration > 0 else max(1.0, (total_frames / max(fps, 1.0)) * 1000.0)
            duration_s = max(0.001, duration_ms / 1000.0)
            sample_fps = max(0.1, float(sample_fps))

            # 候选帧：按时间采样（基于 sample_fps），并做上限约束避免过慢
            requested_candidates = int(duration_s * sample_fps) + 1
            candidate_count = min(total_frames, max(6, requested_candidates))
            candidate_count = min(candidate_count, max_frames * 6)

            candidate_times_ms = np.linspace(
                0,
                max(0.0, duration_ms - 1.0),
                num=int(candidate_count),
                dtype=np.float64
            )

            candidates: list[dict] = []
            for t_ms in candidate_times_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t_ms))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                cv2.normalize(hist, hist)

                candidates.append({"t_ms": float(t_ms), "frame": frame, "hist": hist})

            if len(candidates) < max(3, min(max_frames, 6)):
                logger.warning("⚠️ [Video] Seek抽帧结果偏少，尝试顺序读取兜底")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                candidates = []
                step = max(1, total_frames // max(1, candidate_count))
                frame_index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % step == 0:
                        t_ms = (frame_index / max(fps, 1.0)) * 1000.0
                        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
                        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
                        cv2.normalize(hist, hist)
                        candidates.append({"t_ms": float(t_ms), "frame": frame, "hist": hist})
                        if len(candidates) >= candidate_count:
                            break
                    frame_index += 1

            cap.release()

            if not candidates:
                return []

            # 计算相邻候选帧差异分数（越大越“有变化”）
            scores = [0.0]
            for i in range(1, len(candidates)):
                prev = candidates[i - 1]["hist"]
                curr = candidates[i]["hist"]
                corr = float(cv2.compareHist(prev, curr, cv2.HISTCMP_CORREL))
                scores.append(1.0 - corr)

            # 选择：首尾必选 + 变化优先，且避免时间过近的重复帧
            selected = set()
            if len(candidates) >= 1:
                selected.add(0)
            if len(candidates) >= 2:
                selected.add(len(candidates) - 1)

            min_gap_ms = max(250.0, (duration_ms / max_frames) * 0.6)
            ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            for i in ranked:
                if len(selected) >= max_frames:
                    break
                t_ms = candidates[i]["t_ms"]
                if all(abs(t_ms - candidates[j]["t_ms"]) >= min_gap_ms for j in selected):
                    selected.add(i)

            # 不足则用均匀补齐
            if len(selected) < max_frames:
                fill_indices = np.linspace(0, len(candidates) - 1, num=max_frames, dtype=int).tolist()
                for i in fill_indices:
                    if len(selected) >= max_frames:
                        break
                    selected.add(int(i))

            selected_list = sorted(selected, key=lambda i: candidates[i]["t_ms"])[:max_frames]
            frames_base64: list[str] = []
            for i in selected_list:
                b64 = self._encode_frame_to_jpeg_base64(
                    candidates[i]["frame"],
                    max_side=max_side,
                    quality=jpeg_quality,
                    roi=roi,
                )
                if b64:
                    frames_base64.append(b64)

            logger.info(f"✅ [Video] 成功提取 {len(frames_base64)} 帧(上限{max_frames})")
            return frames_base64

        except Exception as e:
            logger.error(f"❌ [Video] 视频帧提取失败: {e}", exc_info=True)
            return []

    async def _build_multimodal_content(
        self,
        image_urls: list,
        text: str
    ) -> list:
        """
        构建多模态内容

        Args:
            image_urls: 图片URL列表
            text: 文本内容

        Returns:
            多模态内容列表
        """
        import base64
        import requests
        from mimetypes import guess_type

        content = [{"type": "text", "text": text or "请分析这些图片"}]

        for idx, url in enumerate(image_urls, 1):
            try:
                logger.info(f"📥 [Multimodal] 下载图片 {idx}/{len(image_urls)}: {url[:100]}...")

                response = requests.get(url, timeout=30)
                response.raise_for_status()

                content_type = guess_type(url)[0] or 'image/jpeg'
                image_base64 = base64.b64encode(response.content).decode('utf-8')

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{content_type};base64,{image_base64}"
                    }
                })

                logger.info(f"✅ [Multimodal] 图片 {idx} 完成")

            except Exception as e:
                logger.warning(f"⚠️ [Multimodal] 图片 {idx} 加载失败: {e}")
                content[0]["text"] += f"\n\n注意：图片 {idx} 加载失败"

        return content

    def _build_response_format(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        构建response_format参数

        Args:
            context: 上下文（可能包含json_schema）

        Returns:
            response_format配置，如果没有json_schema则返回None
        """
        import os

        mode = (os.getenv("JSON_RESPONSE_FORMAT") or "auto").strip().lower()
        output_format = str(context.get("output_format") or "").strip().lower()
        json_schema = context.get("json_schema")

        if mode in {"none", "off", "disable", "disabled"}:
            return None

        # Even when user doesn't pass a schema, this endpoint expects JSON.
        if not isinstance(json_schema, dict) or not json_schema:
            if output_format == "json":
                return {"type": "json_object"}
            return None

        if mode in {"json_object", "object"}:
            return {"type": "json_object"}

        if mode in {"json_schema", "schema"}:
            logger.info("🎯 [JSON代理] 启用 Structured Outputs（json_schema）")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "AnalysisResult",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        # auto: prefer `json_schema` only for official OpenAI endpoint; fall back to `json_object` elsewhere.
        try:
            cfg_manager = get_model_config_manager()
            if not getattr(cfg_manager, "_config", None):
                cfg_manager.initialize()
            model_type = "vlm" if self._has_visual_inputs(context) else "llm"
            model_cfg = cfg_manager.get_model_config(model_type)
            base_url = (getattr(model_cfg, "base_url", "") or "").lower()
        except Exception:
            base_url = ""

        if "api.openai.com" in base_url:
            logger.info("🎯 [JSON代理] auto: OpenAI base_url detected, using json_schema")
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "AnalysisResult",
                    "schema": json_schema,
                    "strict": True,
                },
            }

        logger.info("🎯 [JSON代理] auto: non-OpenAI base_url detected, using json_object")
        return {"type": "json_object"}

    def _select_model(self, agent_config: Any, context: Dict[str, Any]) -> str:
        """
        选择合适的模型

        Args:
            agent_config: 智能体配置
            context: 上下文

        Returns:
            模型名称
        """
        import os

        # 优先使用agent配置的模型
        config = agent_config.config or {}
        model = config.get('model')

        has_visual = self._has_visual_inputs(context)

        # If visual inputs exist, prefer a multimodal-capable model.
        # Some agents may be configured with a text-only model (e.g. qwen-max),
        # which would cause image/video inputs to be ignored by the provider.
        if has_visual:
            if model and self._looks_like_multimodal_model(model):
                return model
            vlm_model = os.getenv("VLM_MODEL", "qwen-vl-max")
            if model and model != vlm_model:
                logger.warning(
                    "⚠️ [JSON代理] 检测到图片/视频输入，但agent配置模型看起来非多模态(%s)，已自动切换为VLM_MODEL=%s",
                    model,
                    vlm_model,
                )
            return vlm_model

        # No visual inputs: keep agent configured model if provided.
        if model:
            # 如果 agent 配置里固定了云端模型名（如 qwen-vl-max），但当前服务通过 VLM_BASE_URL 指向本地 OpenAI 兼容服务，
            # 则将其映射到环境变量 VLM_MODEL，避免本地服务返回 "model does not exist"。
            env_vlm_model = os.getenv("VLM_MODEL")
            vlm_base_url = os.getenv("VLM_BASE_URL", "")
            is_multimodal = bool(context.get("files") or context.get("image_urls") or context.get("image_url"))
            if (
                is_multimodal
                and env_vlm_model
                and env_vlm_model != model
                and vlm_base_url
                and "dashscope.aliyuncs.com" not in vlm_base_url
                and model.startswith("qwen-vl-")
            ):
                logger.info(f"🔁 [JSON代理] 将配置模型 {model} 映射为环境 VLM_MODEL={env_vlm_model}")
                return env_vlm_model
            return model


        # 检查是否有多模态内容（图片或视频）
        has_multimodal = False

        # 检查新格式的files字段
        files = context.get('files')
        if files:
            has_multimodal = True
        # 兼容旧格式的image_urls
        elif context.get('image_urls') or context.get('image_url'):
            has_multimodal = True

        # 根据是否有多模态内容选择模型
        if has_multimodal:
            return os.getenv("VLM_MODEL", "qwen-vl-max")
        else:
            return os.getenv("LLM_MODEL", "qwen-plus")

    def _rewrite_to_proxy(self, url: str) -> Optional[str]:
        """
        将 192.168.x.x:port 的内网地址重写为 proxy 域名：
        192.168.A.B:PORT -> http://168-A-B-PORT.proxy.jetlinks.cn<path>
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            port = parsed.port or 80
            if not host.startswith("192.168."):
                return None
            parts = host.split(".")
            if len(parts) != 4:
                return None
            proxy_host = f"{parts[1]}-{parts[2]}-{parts[3]}-{port}.proxy.jetlinks.cn"
            new_netloc = proxy_host
            proxy_url = urlunparse((
                "http",
                new_netloc,
                parsed.path or "",
                parsed.params or "",
                parsed.query or "",
                parsed.fragment or "",
            ))
            return proxy_url
        except Exception:
            return None


# 便捷函数
async def process_json_message(
    message: str,
    agent_id: str,
    context: Optional[Dict[str, Any]] = None,
    parameter: Optional[Dict[str, Any]] = None
) -> str:
    """
    便捷的JSON消息处理函数

    Args:
        message: 用户消息
        agent_id: 智能体ID
        context: 上下文信息（运行时参数：image_urls, json_schema等）
        parameter: 系统提示词参数（用于{{variable}}替换）

    Returns:
        AI响应文本（JSON格式）
    """
    processor = JSONAgentProcessor()
    return await processor.process_message(
        message=message,
        agent_id=agent_id,
        context=context,
        parameter=parameter
    )
