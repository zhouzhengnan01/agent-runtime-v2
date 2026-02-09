"""
JSON 智能体对话接口 - 简化版

提供JSON格式的智能体对话功能，支持结构化输出

功能：
1. JSON格式的智能体对话
2. 支持response_format一次性生成JSON
3. 自动结构化数据提取
4. 完整的错误处理和日志
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import time
import logging
import json
import base64
import os
import re
import copy
from urllib.parse import urlparse

from app.db.session import get_db
from app.core.agents.agent_json import process_json_message, process_json_message_with_usage
from app.core.llm.usage import TokenUsageTracker
from app.services.review_record_service import get_review_record_service
from app.services.review_kb_sync_service import enqueue_review_kb_sync

logger = logging.getLogger(__name__)

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _parse_json_with_fallback(raw_response: str) -> Optional[Dict[str, Any]]:
    """
    智能JSON解析，支持多种格式fallback

    尝试顺序：
    1. 直接json.loads()
    2. 提取markdown代码块中的JSON
    3. 提取文本中任意JSON对象
    4. 返回None（解析失败）

    Args:
        raw_response: LLM返回的原始文本

    Returns:
        解析出的JSON对象，失败返回None
    """
    # 1. 尝试直接解析
    try:
        data = json.loads(raw_response)
        logger.info("✅ [JSON解析] 直接解析成功")
        return data
    except json.JSONDecodeError:
        logger.debug("⚠️ [JSON解析] 直接解析失败，尝试提取")

    # 2. 尝试从markdown代码块提取
    markdown_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # ```json {...} ```
        r'```\s*(\{.*?\})\s*```',       # ``` {...} ```
    ]

    for pattern in markdown_patterns:
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                logger.info("✅ [JSON解析] 从markdown代码块提取成功")
                return data
            except json.JSONDecodeError:
                continue

    # 3. 尝试提取任意JSON对象（处理文本+JSON混合情况）
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    match = re.search(json_pattern, raw_response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            logger.info("✅ [JSON解析] 从文本中提取JSON对象成功")
            return data
        except json.JSONDecodeError:
            pass

    # 4. 所有方法都失败
    logger.error("❌ [JSON解析] 无法解析JSON，所有fallback方法均失败")
    return None


def _merge_usage_meta(meta: Dict[str, Any], usage_tracker: Optional[TokenUsageTracker]) -> None:
    if not usage_tracker:
        return
    usage_meta = usage_tracker.build_meta()
    if usage_meta:
        meta.update(usage_meta)


def _default_value_for_schema_spec(spec: Any) -> Any:
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


def _fill_missing_fields_by_schema(
    data: Dict[str, Any],
    json_schema: Optional[Dict[str, Any]],
    *,
    fallback_summary: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    if not isinstance(json_schema, dict):
        return data
    if str(json_schema.get("type") or "").strip().lower() != "object":
        return data
    props = json_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return data

    summary_keys = {
        "summary",
        "desc",
        "description",
        "reason",
        "detail",
        "message",
        "msg",
        "analysis",
        "explain",
        "explanation",
    }

    out = dict(data)
    for key, spec in props.items():
        if key in out:
            continue
        k = str(key)
        default = _default_value_for_schema_spec(spec)
        if (
            isinstance(spec, dict)
            and str(spec.get("type") or "").strip().lower() == "string"
            and fallback_summary
            and k.strip().lower() in summary_keys
        ):
            out[k] = fallback_summary[:2000]
            continue
        out[k] = default
    return out


def _build_json_repair_system_prompt(json_schema: Dict[str, Any]) -> str:
    import json

    props = json_schema.get("properties") if isinstance(json_schema, dict) else None
    keys = list(props.keys()) if isinstance(props, dict) else []
    required = json_schema.get("required") if isinstance(json_schema, dict) else None
    required_keys = [str(k) for k in required] if isinstance(required, list) else []

    example: Dict[str, Any] = {}
    if isinstance(props, dict):
        for k, spec in props.items():
            example[str(k)] = _default_value_for_schema_spec(spec)

    lines = [
        "你是一个 JSON 修复/格式化器。",
        "你必须只输出一个 JSON 对象（不要解释、不要 markdown、不要代码块、不要任何前后缀文字）。",
    ]
    if keys:
        lines.append("只允许输出以下字段（不要输出其它字段）： " + ", ".join([str(k) for k in keys]))
    if required_keys:
        lines.append("必须包含字段： " + ", ".join(required_keys))
    if example:
        lines.append("输出示例（仅示例，最终必须只输出 JSON 对象本体）：")
        lines.append(json.dumps(example, ensure_ascii=False))
    lines.append("输出必须严格符合下面的 JSON Schema：")
    lines.append(json.dumps(json_schema, ensure_ascii=False))
    lines.append("现在只输出 JSON 对象本体。")
    return "\n".join(lines)


async def _repair_json_with_llm(
    *,
    raw_response: str,
    user_message: str,
    json_schema: Dict[str, Any],
    usage_tracker: Optional[TokenUsageTracker] = None,
) -> Optional[Dict[str, Any]]:
    try:
        from app.core.llm.client import LLMClient
        from app.core.llm.model_config_manager import get_model_config_manager
    except Exception:
        return None

    try:
        cfg_manager = get_model_config_manager()
        if not getattr(cfg_manager, "_config", None):
            cfg_manager.initialize()
        model_cfg = cfg_manager.get_model_config("llm")
        model = getattr(model_cfg, "model_name", None)
    except Exception:
        model = None

    if not model:
        import os
        model = (os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo").strip()

    llm_client = LLMClient(provider="unified")
    messages = [
        {"role": "system", "content": _build_json_repair_system_prompt(json_schema)},
        {
            "role": "user",
            "content": (
                "请将下面内容转换为符合 schema 的 JSON，并只输出 JSON。\n\n"
                f"[用户消息]\n{user_message}\n\n"
                f"[LLM原始回答]\n{raw_response}\n"
            ),
        },
    ]

    try:
        repaired_text = await llm_client.chat(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        if usage_tracker:
            usage_tracker.add(llm_client.last_usage, source="json_repair")
    except Exception as e:
        logger.warning("⚠️ [JSON修复] 二次修复调用失败: %s", e, exc_info=True)
        return None

    parsed = _parse_json_with_fallback(repaired_text)
    if parsed is None:
        logger.warning("⚠️ [JSON修复] 二次修复仍未得到可解析JSON，raw=%s", _safe_text(repaired_text, 2000))
        return None

    parsed["_schema_source"] = "llm_repair"
    return parsed


def _coerce_hit(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "hit", "match", "matched"}:
            return 1
        if s in {"0", "false", "no", "n", "miss", "unmatch", "unmatched"}:
            return 0
    return None


def _augment_json_schema_with_hit(json_schema: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], bool]:
    """
    Add a universal `hit` field into a user-provided JSON Schema (strict mode).

    We only augment object schemas with `properties`. The returned schema is a deep copy.
    """
    if not isinstance(json_schema, dict):
        return json_schema, False
    if str(json_schema.get("type") or "").strip().lower() != "object":
        return json_schema, False
    properties = json_schema.get("properties")
    if isinstance(properties, dict) and "hit" in properties:
        return json_schema, False

    schema = copy.deepcopy(json_schema)
    props = schema.setdefault("properties", {})
    if not isinstance(props, dict):
        props = {}
        schema["properties"] = props
    props["hit"] = {
        "type": "integer",
        "enum": [0, 1],
        "description": "通用命中标记：0=未命中，1=命中。命中=符合用户消息中需要判断/检测的条件；若用户消息非二分类判断/命中型问题，则返回0。",
    }
    required = schema.get("required")
    if isinstance(required, list):
        if "hit" not in required:
            required.append("hit")
    else:
        schema["required"] = ["hit"]
    return schema, True


def _extract_hit_and_strip(
    structured_data: Optional[Dict[str, Any]],
    *,
    hit_added_to_schema: bool,
    original_schema: Optional[Dict[str, Any]],
) -> tuple[int, Optional[Dict[str, Any]]]:
    """
    Extract `hit` (0/1) from parsed JSON, and optionally strip it from data payload.
    """
    if not isinstance(structured_data, dict):
        return 0, structured_data

    hit = _coerce_hit(structured_data.get("hit"))
    if hit is None:
        # Fallback: common binary flags in business payloads.
        for k in (
            "is_hit",
            "isHit",
            "matched",
            "match",
            "flagged",
            "is_flagged",
            "isFlagged",
            "illegal",
            "is_illegal",
            "isIllegal",
            "violation",
            "is_violation",
            "isViolation",
        ):
            if k in structured_data:
                hit = _coerce_hit(structured_data.get(k))
                if hit is not None:
                    break

    # Fallback: if schema has a single boolean field, treat it as the decision field.
    if hit is None and isinstance(original_schema, dict):
        props = original_schema.get("properties")
        if isinstance(props, dict):
            bool_keys = [
                key
                for key, spec in props.items()
                if isinstance(spec, dict) and str(spec.get("type") or "").strip().lower() == "boolean"
            ]
            if len(bool_keys) == 1:
                hit = _coerce_hit(structured_data.get(bool_keys[0]))

    if hit is None:
        hit = 0

    if hit_added_to_schema:
        data = dict(structured_data)
        data.pop("hit", None)
        return hit, data
    return hit, structured_data


def _create_sentence_summary(json_data: Dict[str, Any]) -> Optional[str]:
    """
    生成一句话摘要

    Args:
        json_data: 结构化JSON数据

    Returns:
        一句话摘要，如果无法生成则返回None
    """
    try:
        if not isinstance(json_data, dict):
            return None

        parts = []

        def _coerce_bool(value: Any) -> Optional[bool]:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                s = value.strip().lower()
                if s in {"true", "yes", "y", "1", "是"}:
                    return True
                if s in {"false", "no", "n", "0", "否"}:
                    return False
            return None

        def _safe_json(value: Any) -> str:
            try:
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return str(value)

        def _format_simple(value: Any) -> str:
            bool_value = _coerce_bool(value)
            if bool_value is not None:
                return "是" if bool_value else "否"
            return str(value)

        def _format_value(key: str, value: Any) -> Optional[str]:
            if value is None:
                return None
            if key == "timestamp" and isinstance(value, (int, float)):
                return None
            if isinstance(value, str) and not value.strip():
                return None
            if isinstance(value, (list, dict)) and not value:
                return None

            if key == "confidence" and isinstance(value, (int, float)):
                return f"{int(value * 100)}%"
            if key == "safety_level":
                safety_map = {"safe": "安全", "warning": "警告", "danger": "危险"}
                return str(safety_map.get(value, value))
            if key == "key_objects" and isinstance(value, list):
                items = [_format_simple(v) for v in value if v is not None and str(v).strip()]
                return ",".join(items) if items else None

            bool_value = _coerce_bool(value)
            if bool_value is not None:
                return "是" if bool_value else "否"

            if isinstance(value, list):
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    items = [_format_simple(v) for v in value if v is not None and str(v).strip()]
                    return ",".join(items) if items else None
                return _safe_json(value)
            if isinstance(value, dict):
                return _safe_json(value)
            if isinstance(value, str):
                return value.strip()
            return str(value)

        def _append_kv(key: str, value: Any) -> None:
            formatted = _format_value(key, value)
            if formatted is None:
                return
            parts.append(f"{key}:{formatted}")

        # 字段优先级顺序（重要字段在前）
        priority_keys = ["summary", "content_type", "safety_level", "confidence", "status", "key_objects"]

        # 先处理优先字段
        for key in priority_keys:
            if key in json_data:
                _append_kv(key, json_data.get(key))

        # 再处理其他字段
        for key, value in json_data.items():
            if key in priority_keys or key in ["_schema_source", "_agent_id"]:
                continue
            _append_kv(key, value)

        # 拼接成一句话
        if parts:
            return "，".join(parts) + "。"
        return None

    except Exception as e:
        logger.warning(f"⚠️ [Summary] 生成一句话摘要失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 请求和响应模型
# ═══════════════════════════════════════════════════════════════

class FileItem(BaseModel):
    """文件项"""
    url: str = Field(..., description="文件URL")
    media_type: str = Field(..., description="媒体类型：image, video等")


class ContextData(BaseModel):
    """上下文数据"""
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="系统提示词参数（{{variable}}替换）")
    task: Optional[str] = Field(default=None, description="任务名称（可选，用于页面展示/筛选）")
    files: Optional[List[FileItem]] = Field(default=None, description="文件列表")
    roi: Optional[Union[List[float], Dict[str, Any]]] = Field(
        default=None,
        description="ROI rect (pixel or normalized). Example: {\"rect\":[x1,y1,x2,y2],\"normalized\":true}",
    )
    output_format: Optional[str] = Field(default="json", description="输出格式")
    json_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON Schema定义")
    video_max_frames: Optional[int] = Field(default=None, description="视频最大抽帧数（默认12）")
    video_sample_fps: Optional[float] = Field(default=None, description="视频抽帧候选采样频率(帧/秒)，越大越准但越慢（默认2.0）")
    image_max_side: Optional[int] = Field(default=None, description="图片/视频帧压缩最大边（默认768）")
    image_jpeg_quality: Optional[int] = Field(default=None, description="图片/视频帧JPEG质量1-100（默认75）")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息内容", min_length=1, max_length=10000)
    context: ContextData = Field(..., description="上下文信息")

    class Config:
        schema_extra = {
            "example": {
                "message": "根据图片判断是否有人",
                "context": {
                    "parameters": {},
                    "files": [
                        {
                            "url": "http://example.com/image.png",
                            "media_type": "image"
                        }
                    ],
                    "output_format": "json",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "number": {
                                "description": "人数",
                                "type": "integer"
                            },
                            "someone": {
                                "description": "是否有人",
                                "type": "boolean"
                            }
                        }
                    }
                }
            }
        }


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool = Field(..., description="处理是否成功")
    hit: int = Field(default=0, description="通用命中标记（0=未命中，1=命中）")
    data: Optional[Dict[str, Any]] = Field(default=None, description="结构化数据")
    result: Optional[Dict[str, Any]] = Field(default=None, description="结构化数据（同data）")
    raw_response: Optional[str] = Field(default=None, description="响应摘要（由结构化数据拼接生成）")
    summary_sentence: Optional[str] = Field(default=None, description="一句话摘要（由JSON数据拼接而成）")
    error: Optional[str] = Field(default=None, description="错误信息")
    error_type: Optional[str] = Field(default=None, description="错误类型")
    meta: Dict[str, Any] = Field(..., description="元数据信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: float


def _mask_data_uri(url: str, max_chars: int = 256) -> str:
    if not isinstance(url, str):
        return ""
    if not url.startswith("data:"):
        return url
    if len(url) <= max_chars:
        return url
    return url[:max_chars] + f"\n…(truncated, original_length={len(url)})"


def _safe_text(text: str, max_chars: int = 20000) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…(truncated, original_length={len(text)})"


def _enqueue_review_kb_sync(record_id: str) -> None:
    try:
        enqueue_review_kb_sync(record_id)
    except Exception:
        pass


def _build_human_readable_response(
    *,
    success: bool,
    structured_data: Optional[Dict[str, Any]],
    summary_sentence: Optional[str],
    error: Optional[str] = None,
) -> str:
    """
    Build a human readable `raw_response` for clients.

    We intentionally avoid returning the original raw JSON text (LLM output) here,
    and instead return a summary paragraph derived from the parsed result.
    """
    if success:
        if isinstance(summary_sentence, str) and summary_sentence.strip():
            return summary_sentence.strip()
        if isinstance(structured_data, dict):
            fallback = _create_sentence_summary(structured_data)
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
        return "请求成功。"

    if isinstance(error, str) and error.strip():
        return error.strip()
    return "请求失败。"


def _extract_first_video_url(files: Optional[List[FileItem]]) -> Optional[str]:
    if not files:
        return None
    for file_item in files:
        try:
            if (file_item.media_type or "").lower() == "video" and (file_item.url or "").strip():
                return file_item.url.strip()
        except Exception:
            continue
    return None


def _build_media_frames(files: Optional[List[FileItem]]) -> List[Dict[str, Any]]:
    """
    Build a stable `meta.media_frames` list for clients.

    - Always returns at least 2 items so clients can safely access [0] (image) and [1] (video).
    - Prefers the first image and first video found in the request `context.files`.
    """
    image_item: Optional[FileItem] = None
    video_item: Optional[FileItem] = None
    rest: List[FileItem] = []

    for f in files or []:
        try:
            media_type = (f.media_type or "").lower().strip()
            if media_type == "image" and image_item is None:
                image_item = f
                continue
            if media_type == "video" and video_item is None:
                video_item = f
                continue
            rest.append(f)
        except Exception:
            continue

    def _to_frame(file_item: Optional[FileItem], media_type_fallback: str) -> Dict[str, Any]:
        if not file_item:
            return {"url": "", "media_type": media_type_fallback}
        return {
            "url": _mask_data_uri(file_item.url),
            "media_type": (file_item.media_type or media_type_fallback),
        }

    frames: List[Dict[str, Any]] = [
        _to_frame(image_item, "image"),
        _to_frame(video_item, "video"),
    ]

    for f in rest:
        try:
            frames.append({"url": _mask_data_uri(f.url), "media_type": f.media_type})
        except Exception:
            continue

    return frames


def _get_int_env(name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(name, "")).strip())
        if value > 0:
            return value
    except Exception:
        pass
    return default


def _decode_edge_history_url(url: str) -> Optional[Dict[str, str]]:
    if not isinstance(url, str) or "/ai/task/history/_read/" not in url or "/zlmedia/" not in url:
        return None
    try:
        parsed = urlparse(url)
        encoded = parsed.path.split("/zlmedia/", 1)[1]
        encoded = encoded.rsplit(".", 1)[0]
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.b64decode(encoded + padding).decode("utf-8", errors="ignore")
        parts = decoded.split("|")
        if len(parts) < 4:
            return None
        _, stream_id, rel_path, filename = parts[:4]
        return {"stream_id": stream_id, "rel_path": rel_path, "filename": filename}
    except Exception:
        return None


def _select_edge_history_images(
    *,
    directory: str,
    target_filename: str,
    max_images: int,
    window_seconds: int,
) -> List[str]:
    if max_images <= 1:
        return []
    try:
        entries = [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    except Exception:
        return []

    def _extract_ts(name: str) -> Optional[int]:
        base = os.path.basename(name)
        match = re.search(r"_(\\d{10,13})\\.", base)
        if not match:
            return None
        try:
            ts = int(match.group(1))
        except Exception:
            return None
        if ts > 10_000_000_000:
            ts = ts // 1000
        return ts

    target_ts = _extract_ts(target_filename)
    candidates = []
    for path in entries:
        ts = _extract_ts(path)
        candidates.append((ts, path))

    if target_ts is not None:
        windowed = [item for item in candidates if item[0] is not None and abs(item[0] - target_ts) <= window_seconds]
        if windowed:
            candidates = windowed
        candidates.sort(key=lambda item: (item[0] or 0, item[1]))
    else:
        candidates.sort(key=lambda item: os.path.basename(item[1]))

    result = []
    for _, path in candidates:
        if os.path.basename(path) == target_filename:
            continue
        result.append(path)
        if len(result) >= max_images - 1:
            break
    return result


def _expand_edge_history_files(
    files: Optional[List[FileItem]],
    *,
    max_images: int,
    window_seconds: int,
) -> Optional[List[FileItem]]:
    if not files or len(files) != 1:
        return files
    file_item = files[0]
    media_type = (getattr(file_item, "media_type", None) or "").lower()
    if media_type != "image":
        return files
    url = getattr(file_item, "url", None)
    if not isinstance(url, str):
        return files
    decoded = _decode_edge_history_url(url)
    if not decoded:
        return files

    zlmedia_root = os.getenv("EDGE_ZLMEDIA_ROOT", "/data/jetlinks-edge/data/zlmedia")
    vhost = os.getenv("EDGE_ZLMEDIA_VHOST", "__defaultVhost__")
    app_name = os.getenv("EDGE_ZLMEDIA_APP")
    if not app_name:
        try:
            candidates = [d for d in os.listdir(zlmedia_root) if os.path.isdir(os.path.join(zlmedia_root, d))]
            app_name = candidates[0] if candidates else ""
        except Exception:
            app_name = ""
    if not app_name:
        return files

    stream_id = decoded["stream_id"]
    rel_path = decoded["rel_path"]
    filename = decoded["filename"]
    local_dir = os.path.join(
        zlmedia_root,
        app_name,
        stream_id,
        "data",
        "snapshot",
        vhost,
        "ai",
        rel_path,
    )
    local_path = os.path.join(local_dir, filename)
    if not os.path.isfile(local_path):
        return files

    extra_paths = _select_edge_history_images(
        directory=local_dir,
        target_filename=filename,
        max_images=max_images,
        window_seconds=window_seconds,
    )
    if not extra_paths:
        return files

    expanded = [file_item]
    for path in extra_paths:
        expanded.append(FileItem(url=path, media_type="image"))
    return expanded


# ═══════════════════════════════════════════════════════════════
# API 端点
# ═══════════════════════════════════════════════════════════════

@router.post("/agents/{agent_id}/chat/json", response_model=ChatResponse)
async def chat_with_json_response(
    agent_id: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    智能体JSON格式对话接口（简化版）

    **功能特点:**
    - 🚀 极简架构，直接LLM调用
    - 📋 支持response_format一次性生成JSON
    - ⚡ 无工具调用开销，高性能
    - 🎯 纯结构化输出

    **使用场景:**
    - 数据分析和结构化输出
    - 自动化集成和API调用
    - 批量处理和数据提取
    """
    start_time = time.time()
    usage_tracker = TokenUsageTracker()

    try:
        review_store = get_review_record_service()

        logger.info(f"🚀 [JSON API] 收到请求")
        logger.info(f"  智能体ID: {agent_id}")
        logger.info(f"  用户消息: '{request.message}'")
        logger.info(f"  消息长度: {len(request.message)}")
        logger.info(f"  文件数量: {len(request.context.files) if request.context.files else 0}")
        logger.info(f"  系统参数: {bool(request.context.parameters)}")
        logger.info(f"  JSON Schema: {bool(request.context.json_schema)}")
        if request.context.json_schema:
            logger.info(f"📋 [JSON API] Schema: {request.context.json_schema}")
        if request.context.files:
            for i, file_item in enumerate(request.context.files):
                logger.info(f"📁 [JSON API] 文件{i+1}: {file_item.url} (类型: {file_item.media_type})")
        if request.context.parameters:
            logger.info(f"🔧 [JSON API] 参数: {request.context.parameters}")

        max_images = _get_int_env("EDGE_HISTORY_MAX_IMAGES", 16)
        window_seconds = _get_int_env("EDGE_HISTORY_WINDOW_SEC", 10)
        expanded_files = _expand_edge_history_files(
            request.context.files,
            max_images=max_images,
            window_seconds=window_seconds,
        )
        if expanded_files is not request.context.files:
            logger.info(
                "🧩 [EdgeHistory] 已扩展图片数量: %s -> %s",
                len(request.context.files) if request.context.files else 0,
                len(expanded_files) if expanded_files else 0,
            )

        # 1. 转换参数格式（适配内部处理器）
        original_json_schema = request.context.json_schema
        augmented_json_schema, hit_added_to_schema = _augment_json_schema_with_hit(original_json_schema)
        context_dict = {
            "json_schema": augmented_json_schema,
            "output_format": request.context.output_format,
            "video_max_frames": request.context.video_max_frames,
            "video_sample_fps": request.context.video_sample_fps,
            "image_max_side": request.context.image_max_side,
            "image_jpeg_quality": request.context.image_jpeg_quality,
            "roi": request.context.roi,
        }

        # 直接传递files字段（支持图片和视频）
        if expanded_files:
            # 转换为字典格式
            context_dict["files"] = [
                {"url": f.url, "media_type": f.media_type}
                for f in expanded_files
            ]
            logger.info(f"  📷 文件数量: {len(expanded_files)}")
            for f in expanded_files:
                logger.info(f"    - {f.media_type}: {f.url}")

        # 2. 处理消息（一次性生成JSON）
        raw_response, usage = await process_json_message_with_usage(
            message=request.message,
            agent_id=agent_id,
            context=context_dict,
            parameter=request.context.parameters or {}
        )
        usage_tracker.add(usage, source="json_response")

        logger.info(f"💬 [JSON API] 获得原始响应，长度: {len(raw_response)}")
        logger.info(f"📄 [JSON API] 原始响应内容: {raw_response}")

        # 3. 智能JSON解析（支持多种格式fallback）
        structured_data = _parse_json_with_fallback(raw_response)
        logger.info(f"🔍 [JSON API] 解析后的结构化数据: {structured_data}")

        repaired = False
        if structured_data is None and isinstance(augmented_json_schema, dict) and augmented_json_schema:
            logger.warning("⚠️ [JSON API] 首次解析失败，尝试二次修复为JSON...")
            repaired_data = await _repair_json_with_llm(
                raw_response=raw_response,
                user_message=request.message,
                json_schema=augmented_json_schema,
                usage_tracker=usage_tracker,
            )
            if repaired_data is not None:
                structured_data = repaired_data
                repaired = True
                logger.info("✅ [JSON API] 二次修复成功")

        if isinstance(structured_data, dict):
            structured_data = _fill_missing_fields_by_schema(
                structured_data,
                augmented_json_schema if isinstance(augmented_json_schema, dict) else None,
                fallback_summary=raw_response if repaired else None,
            )

        if structured_data is None:
            # 解析失败，返回错误响应
            logger.error(f"❌ [JSON API] JSON解析失败，response_format未生效")
            response_time = round(time.time() - start_time, 3)
            human_readable = _build_human_readable_response(
                success=False,
                structured_data=None,
                summary_sentence=None,
                error="LLM未返回有效JSON格式",
            )
            response = ChatResponse(
                success=False,
                hit=0,
                result=None,
                data=None,
                error="LLM未返回有效JSON格式",
                error_type="JSONParseError",
                raw_response=human_readable,
                meta={
                    "agent_id": agent_id,
                    "response_time": response_time,
                    "error_category": "json_parse_failed",
                    "raw_response_length": len(raw_response),
                    "timestamp": int(time.time() * 1000),
                    "media_frames": _build_media_frames(request.context.files),
                }
            )
            _merge_usage_meta(response.meta, usage_tracker)
            try:
                video_url = _extract_first_video_url(request.context.files)
                request_payload = {
                    "message": request.message,
                    "context": {
                        "parameters": request.context.parameters or {},
                        "task": request.context.task,
                        "output_format": request.context.output_format,
                        "json_schema": request.context.json_schema,
                        "video_max_frames": request.context.video_max_frames,
                        "video_sample_fps": request.context.video_sample_fps,
                        "image_max_side": request.context.image_max_side,
                        "image_jpeg_quality": request.context.image_jpeg_quality,
                        "roi": request.context.roi,
                        "files": [
                            {"url": _mask_data_uri(f.url), "media_type": f.media_type}
                            for f in (request.context.files or [])
                        ],
                    },
                }
                response_payload = {
                    "success": False,
                    "data": None,
                    "result": None,
                    "raw_response": human_readable,
                    "summary_sentence": None,
                    "error": "LLM未返回有效JSON格式",
                    "error_type": "JSONParseError",
                }
                record_meta = {
                    "http_status": 200,
                    "cost_ms": int(response_time * 1000),
                    "summary": "JSONParseError",
                    "llm_raw_response": _safe_text(raw_response),
                    "llm_raw_response_length": len(raw_response),
                    "llm_json_repair_attempted": True,
                }
                _merge_usage_meta(record_meta, usage_tracker)
                record_paths = review_store.create_record(
                    agent_id=agent_id,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    meta=record_meta,
                )
                review_store.save_image_preview_from_files(record_paths.record_id, request.context.files)
                response.meta["review_record_id"] = record_paths.record_id
                response.meta["review_record_detail"] = f"/api/v1/review-records/{record_paths.record_id}"

                if video_url:
                    review_store.update_record(
                        record_paths.record_id,
                        {
                            "video": {
                                "source_url": video_url,
                                "clip_seconds": getattr(review_store, "clip_seconds", 30),
                                "status": "queued",
                                "static_url": None,
                                "file_path": None,
                                "error": None,
                                "recorded_at": None,
                            }
                        },
                    )
                    background_tasks.add_task(
                        review_store.record_video_for_record,
                        record_id=record_paths.record_id,
                        source_url=video_url,
                    )
                if getattr(review_store, "max_records", 0) > 0:
                    background_tasks.add_task(review_store.enforce_retention_async)
                _enqueue_review_kb_sync(record_paths.record_id)
            except Exception:
                pass

            return response

        logger.info(f"✅ [JSON API] JSON解析成功")

        hit, data_payload = _extract_hit_and_strip(
            structured_data,
            hit_added_to_schema=hit_added_to_schema,
            original_schema=original_json_schema,
        )

        # 4. 生成一句话摘要
        sentence_summary = _create_sentence_summary(data_payload or {})
        human_readable = _build_human_readable_response(
            success=True,
            structured_data=data_payload,
            summary_sentence=sentence_summary,
        )

        # 5. 构建响应
        response_time = round(time.time() - start_time, 3)

        response = ChatResponse(
            success=True,
            hit=hit,
            data=data_payload,
            result=data_payload,
            raw_response=human_readable,
            summary_sentence=sentence_summary,
            meta={
                "agent_id": agent_id,
                "response_time": response_time,
                "schema_source": (data_payload or {}).get("_schema_source", "unknown"),
                "message_length": len(request.message),
                "response_length": len(raw_response),
                "timestamp": int(time.time() * 1000),
                "media_frames": _build_media_frames(request.context.files),
            }
        )
        _merge_usage_meta(response.meta, usage_tracker)

        # ───────────────────────────────────────────────────────────
        # 落盘调用记录（全局最近1000条）+ 可选录制30秒视频
        # ───────────────────────────────────────────────────────────
        video_url = _extract_first_video_url(request.context.files)
        request_payload = {
            "message": request.message,
            "context": {
                "parameters": request.context.parameters or {},
                "task": request.context.task,
                "output_format": request.context.output_format,
                "json_schema": request.context.json_schema,
                "video_max_frames": request.context.video_max_frames,
                "video_sample_fps": request.context.video_sample_fps,
                "image_max_side": request.context.image_max_side,
                "image_jpeg_quality": request.context.image_jpeg_quality,
                "roi": request.context.roi,
                "files": [
                    {"url": _mask_data_uri(f.url), "media_type": f.media_type}
                    for f in (request.context.files or [])
                ],
            },
        }
        response_payload = {
            "success": True,
            "hit": hit,
            "data": data_payload,
            "result": data_payload,
            "raw_response": human_readable,
            "summary_sentence": sentence_summary,
            "error": None,
            "error_type": None,
        }
        record_meta = {
            "http_status": 200,
            "cost_ms": int(response_time * 1000),
            "summary": sentence_summary,
            "llm_raw_response": _safe_text(raw_response),
            "llm_raw_response_length": len(raw_response),
            "llm_json_repair_used": bool(repaired),
        }
        _merge_usage_meta(record_meta, usage_tracker)
        record_paths = review_store.create_record(
            agent_id=agent_id,
            request_payload=request_payload,
            response_payload=response_payload,
            meta=record_meta,
        )
        review_store.save_image_preview_from_files(record_paths.record_id, request.context.files)

        response.meta["review_record_id"] = record_paths.record_id
        response.meta["review_record_detail"] = f"/api/v1/review-records/{record_paths.record_id}"
        response.meta["review_record_video"] = {
            "status": "queued" if video_url else "not_requested",
            "static_url": None,
        }

        if video_url:
            review_store.update_record(
                record_paths.record_id,
                {
                    "video": {
                        "source_url": video_url,
                        "clip_seconds": getattr(review_store, "clip_seconds", 30),
                        "status": "queued",
                        "static_url": None,
                        "file_path": None,
                        "error": None,
                        "recorded_at": None,
                    }
                },
            )
            background_tasks.add_task(
                review_store.record_video_for_record,
                record_id=record_paths.record_id,
                source_url=video_url,
            )
            response.meta["review_record_video"]["status"] = "queued"

        if getattr(review_store, "max_records", 0) > 0:
            background_tasks.add_task(review_store.enforce_retention_async)
        _enqueue_review_kb_sync(record_paths.record_id)

        return response

    except ValueError as e:
        # 业务逻辑错误（如智能体不存在）
        response_time = round(time.time() - start_time, 3)
        logger.warning(f"⚠️ [JSON API] 业务错误: {e}")

        human_readable = _build_human_readable_response(
            success=False,
            structured_data=None,
            summary_sentence=None,
            error=str(e),
        )
        response = ChatResponse(
            success=False,
            hit=0,
            result=None,
            data=None,
            error=str(e),
            error_type="ValidationError",
            raw_response=human_readable,
            meta={
                "agent_id": agent_id,
                "response_time": response_time,
                "error_category": "business_logic",
                "media_frames": _build_media_frames(request.context.files),
            }
        )
        _merge_usage_meta(response.meta, usage_tracker)
        try:
            review_store = get_review_record_service()
            video_url = _extract_first_video_url(request.context.files)
            record_meta = {
                "http_status": 200,
                "cost_ms": int(response_time * 1000),
                "summary": str(e),
                "llm_raw_response": None,
            }
            _merge_usage_meta(record_meta, usage_tracker)
            record_paths = review_store.create_record(
                agent_id=agent_id,
                request_payload={
                    "message": request.message,
                    "context": {
                        "parameters": request.context.parameters or {},
                        "task": request.context.task,
                        "output_format": request.context.output_format,
                        "json_schema": request.context.json_schema,
                        "roi": request.context.roi,
                        "files": [
                            {"url": _mask_data_uri(f.url), "media_type": f.media_type}
                            for f in (request.context.files or [])
                        ],
                    },
                },
                response_payload={
                    "success": False,
                    "data": None,
                    "result": None,
                    "raw_response": human_readable,
                    "summary_sentence": None,
                    "error": str(e),
                    "error_type": "ValidationError",
                },
                meta=record_meta,
            )
            review_store.save_image_preview_from_files(record_paths.record_id, request.context.files)
            response.meta["review_record_id"] = record_paths.record_id
            response.meta["review_record_detail"] = f"/api/v1/review-records/{record_paths.record_id}"
            if video_url:
                review_store.update_record(
                    record_paths.record_id,
                    {
                        "video": {
                            "source_url": video_url,
                            "clip_seconds": getattr(review_store, "clip_seconds", 30),
                            "status": "queued",
                            "static_url": None,
                            "file_path": None,
                            "error": None,
                            "recorded_at": None,
                        }
                    },
                )
                background_tasks.add_task(
                    review_store.record_video_for_record,
                    record_id=record_paths.record_id,
                    source_url=video_url,
                )
            if getattr(review_store, "max_records", 0) > 0:
                background_tasks.add_task(review_store.enforce_retention_async)
            _enqueue_review_kb_sync(record_paths.record_id)
        except Exception:
            pass

        return response

    except Exception as e:
        # 系统错误
        response_time = round(time.time() - start_time, 3)
        logger.error(f"❌ [JSON API] 系统错误: {e}", exc_info=True)

        error_message = f"系统内部错误: {str(e)}"
        human_readable = _build_human_readable_response(
            success=False,
            structured_data=None,
            summary_sentence=None,
            error=error_message,
        )
        response = ChatResponse(
            success=False,
            hit=0,
            result=None,
            data=None,
            error=error_message,
            error_type=type(e).__name__,
            raw_response=human_readable,
            meta={
                "agent_id": agent_id,
                "response_time": response_time,
                "error_category": "system_error",
                "media_frames": _build_media_frames(request.context.files),
            }
        )
        _merge_usage_meta(response.meta, usage_tracker)
        try:
            review_store = get_review_record_service()
            video_url = _extract_first_video_url(request.context.files)
            record_meta = {
                "http_status": 200,
                "cost_ms": int(response_time * 1000),
                "summary": error_message,
                "llm_raw_response": None,
            }
            _merge_usage_meta(record_meta, usage_tracker)
            record_paths = review_store.create_record(
                agent_id=agent_id,
                request_payload={
                    "message": request.message,
                    "context": {
                        "parameters": request.context.parameters or {},
                        "task": request.context.task,
                        "output_format": request.context.output_format,
                        "json_schema": request.context.json_schema,
                        "roi": request.context.roi,
                        "files": [
                            {"url": _mask_data_uri(f.url), "media_type": f.media_type}
                            for f in (request.context.files or [])
                        ],
                    },
                },
                response_payload={
                    "success": False,
                    "data": None,
                    "result": None,
                    "raw_response": human_readable,
                    "summary_sentence": None,
                    "error": error_message,
                    "error_type": type(e).__name__,
                },
                meta=record_meta,
            )
            review_store.save_image_preview_from_files(record_paths.record_id, request.context.files)
            response.meta["review_record_id"] = record_paths.record_id
            response.meta["review_record_detail"] = f"/api/v1/review-records/{record_paths.record_id}"
            if video_url:
                review_store.update_record(
                    record_paths.record_id,
                    {
                        "video": {
                            "source_url": video_url,
                            "clip_seconds": getattr(review_store, "clip_seconds", 30),
                            "status": "queued",
                            "static_url": None,
                            "file_path": None,
                            "error": None,
                            "recorded_at": None,
                        }
                    },
                )
                background_tasks.add_task(
                    review_store.record_video_for_record,
                    record_id=record_paths.record_id,
                    source_url=video_url,
                )
            if getattr(review_store, "max_records", 0) > 0:
                background_tasks.add_task(review_store.enforce_retention_async)
            _enqueue_review_kb_sync(record_paths.record_id)
        except Exception:
            pass

        return response


@router.get("/agents/json/health", response_model=HealthResponse)
async def health_check():
    """
    JSON处理器健康检查

    返回JSON处理器的运行状态
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )


@router.get("/agents/{agent_id}/json/info")
async def get_agent_json_info(
    agent_id: str,
    db: Session = Depends(get_db)
):
    """
    获取智能体的JSON输出配置信息

    检查智能体是否支持JSON输出，以及相关配置
    """
    try:
        from app.services.agent_service import AgentService

        agent_service = AgentService(db)
        agent_config = agent_service.get_agent(agent_id)

        if not agent_config:
            raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

        config = agent_config.config or {}
        output_format = config.get("output_format", "text")
        json_schema = config.get("json_schema")

        return {
            "agent_id": agent_id,
            "name": agent_config.name,
            "type": agent_config.type,
            "output_format": output_format,
            "supports_json": output_format == "json",
            "has_json_schema": bool(json_schema),
            "json_schema": json_schema
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [智能体信息] 获取失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取智能体信息失败: {str(e)}")


# ═══════════════════════════════════════════════════════════════
# 便捷的测试端点（可选，开发环境使用）
# ═══════════════════════════════════════════════════════════════

@router.post("/agents/{agent_id}/chat/json/test")
async def test_json_chat(
    agent_id: str,
    message: str,
    background_tasks: BackgroundTasks
):
    """
    JSON对话测试端点（简化版）

    用于快速测试，接受查询参数而非JSON body
    """
    request = ChatRequest(message=message, context=ContextData())
    return await chat_with_json_response(agent_id, request, background_tasks)
