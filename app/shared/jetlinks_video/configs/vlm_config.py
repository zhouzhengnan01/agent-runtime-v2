# -*- coding: utf-8 -*-
"""
VlmConfig

配置读取优先级：
1) 项目根目录 `.env`（DEFAULT_BASE_PATH/.env）——使用 python-dotenv 的 `dotenv_values` 读取（不会注入 os.environ）
2) OS 环境变量（docker / k8s / systemd / 手动 export）

说明：
- `main.py` 会尝试 `load_dotenv` 把 `.env` 注入环境变量；本模块也支持“只靠环境变量”运行。
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.shared.jetlinks_video.utils.logger_utils import get_logger

logger = get_logger(__name__)

# =========================
# 1) 定位项目根目录 & .env
# =========================
# 当前文件: app/shared/jetlinks_video/configs/vlm_config.py
# 向上 4 层: configs -> jetlinks_video -> shared -> app -> 项目根目录
DEFAULT_BASE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

ENV_FILE_PATH = os.path.join(DEFAULT_BASE_PATH, ".env")

_ENV_MAP: Dict[str, str] = {}

try:
    from dotenv import dotenv_values  # type: ignore
except Exception as e:  # noqa: BLE001
    dotenv_values = None  # type: ignore[assignment]
    logger.warning("[VLM] python-dotenv 不可用，将仅使用 OS 环境变量读取配置: %s", e)

if dotenv_values and os.path.exists(ENV_FILE_PATH):
    try:
        _ENV_MAP = {
            str(k): str(v)
            for k, v in (dotenv_values(ENV_FILE_PATH) or {}).items()
            if k is not None and v is not None
        }
        logger.info("[VLM] Loaded .env values from %s", ENV_FILE_PATH)
    except Exception as e:  # noqa: BLE001
        logger.warning("[VLM] 读取 .env 失败（将回退 OS 环境变量）：%s", e)
else:
    if os.path.exists(ENV_FILE_PATH):
        logger.info("[VLM] .env exists but not loaded (python-dotenv unavailable): %s", ENV_FILE_PATH)
    else:
        logger.info("[VLM] .env not found at %s; using os.environ only", ENV_FILE_PATH)


def _env_get_optional(key: str) -> Optional[str]:
    raw = _ENV_MAP.get(key)
    if raw is None:
        raw = os.getenv(key)
    if raw is None:
        return None
    val = str(raw).strip()
    return val if val else None


def _env_get_first(keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    for k in keys:
        v = _env_get_optional(k)
        if v:
            return v, k
    return None, None


def _require_first(keys: Tuple[str, ...], *, label: str) -> Tuple[str, str]:
    v, used = _env_get_first(keys)
    if not v or not used:
        raise ValueError(
            f"[VLM] {label} 未配置：请在 {ENV_FILE_PATH} 或环境变量中设置 "
            + " / ".join(keys)
        )
    return v, used


def _load_vlm_model_name() -> str:
    v, used = _require_first(
        ("VLM_MODEL_NAME", "VLM_MODEL_NEW", "VLM_MODEL", "VLM_CLOUD_MODEL_NAME"),
        label="VLM_MODEL_NAME",
    )
    logger.info("[VLM] Using model=%s (from %s)", v, used)
    return v


def _load_vlm_api_key() -> str:
    v, used = _require_first(
        ("VLM_API_KEY", "VLM_CLOUD_API_KEY"),
        label="VLM_API_KEY",
    )
    logger.info("[VLM] Loaded API key (masked) from %s", used)
    return v


def _load_vlm_base_url() -> str:
    v, used = _require_first(
        ("VLM_BASE_URL", "VLM_CLOUD_BASE_URL"),
        label="VLM_BASE_URL",
    )
    logger.info("[VLM] Using base_url=%s (from %s)", v, used)
    return v


def _load_vlm_model_name_backup() -> Optional[str]:
    v, used = _env_get_first(
        ("VLM_MODEL_NAME_BACKUP", "VLM_MODEL_BACKUP", "VLM_CLOUD_MODEL_NAME_BACKUP"),
    )
    if v:
        logger.info("[VLM] Using backup model=%s (from %s)", v, used)
    return v


def _load_vlm_api_key_backup() -> Optional[str]:
    v, used = _env_get_first(
        ("VLM_API_KEY_BACKUP", "VLM_CLOUD_API_KEY_BACKUP", "OPENAI_API_KEY_BACKUP"),
    )
    if v:
        logger.info("[VLM] Loaded backup API key (masked) from %s", used)
    return v


def _load_vlm_base_url_backup() -> Optional[str]:
    v, used = _env_get_first(
        ("VLM_BASE_URL_BACKUP", "VLM_CLOUD_BASE_URL_BACKUP", "OPENAI_BASE_URL_BACKUP"),
    )
    if v:
        logger.info("[VLM] Using backup base_url=%s (from %s)", v, used)
    return v


class VlmConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # offline 离线音视频解析的 VLM 侧系统提示词只能由上层传入, 且没有预设
    offline_system_prompt: Optional[str] = ""

    # ========= 核心 VLM 相关字段（全部由 .env 驱动） =========
    vlm_model_name: str = Field(default_factory=_load_vlm_model_name)
    vlm_api_key: str = Field(default_factory=_load_vlm_api_key)
    vlm_base_url: str = Field(default_factory=_load_vlm_base_url)
    vlm_model_name_backup: Optional[str] = Field(default_factory=_load_vlm_model_name_backup)
    vlm_api_key_backup: Optional[str] = Field(default_factory=_load_vlm_api_key_backup)
    vlm_base_url_backup: Optional[str] = Field(default_factory=_load_vlm_base_url_backup)

    # ========= 其他 VLM 控制项 =========
    vlm_streaming: bool = False

    # 是否严格 JSON 格式输出
    is_json_format: bool = True

    # 证据帧静态导出（主控侧会根据 task_id 覆盖此路径）
    vlm_static_evidence_images_dir: str = os.path.join(DEFAULT_BASE_PATH, "storage", "evidence_images")
    vlm_static_evidence_images_url_prefix: str = "/storage/evidence_images"

    # 模型温度参数，模型每次输出都倾向于概率最大的token
    vlm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # 视觉输入最多使用的帧数（配合 cut_config.topk_frames）
    vlm_max_frames: int = Field(default=8, ge=1)
