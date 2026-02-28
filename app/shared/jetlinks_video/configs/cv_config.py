from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.shared.jetlinks_video.utils.file_utils import read_env_kv


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_ENV_PATH = _PROJECT_ROOT / ".env"


def _env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    if val is not None:
        return str(val).strip() or default
    return read_env_kv(_ENV_PATH, key, default)


def _env_get_float(key: str, default: float) -> float:
    raw = _env_get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_get_float_optional(key: str) -> Optional[float]:
    raw = _env_get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _env_get_int(key: str, default: int) -> int:
    raw = _env_get(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except Exception:
        return default


def _env_get_list_int(key: str) -> Optional[List[int]]:
    raw = _env_get(key)
    if not raw:
        return None
    items = []
    for part in str(raw).replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p))
        except Exception:
            continue
    return items or None


CV_MODEL_REGISTRY: Dict[str, str] = {
    "yolov8n.pt": "yolov8n.pt",
    "yolov8n-pose.pt": "yolov8n-pose.pt",
}

def _default_device() -> str:
    """
    Ultralytics fails hard when forcing `device=0` on CPU-only installs.
    Prefer env override, else auto-detect CUDA and fall back to CPU.
    """
    explicit = _env_get("CV_DEVICE") or _env_get("YOLO_DEVICE")
    if explicit:
        return explicit
    try:
        import torch  # type: ignore

        return "0" if bool(getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
    except Exception:
        return "cpu"


@dataclass
class CvPoseConfig:
    # Recall-friendly defaults (may raise false positives).
    min_kpt_conf: float = 0.2
    fall_ratio: float = 0.9
    fall_angle_ratio: float = 0.9
    fall_low_y: float = 0.6
    smoke_dist_ratio: float = 0.7
    smoke_wrist_y_margin: float = 0.25
    fight_center_ratio: float = 0.9
    fight_hand_ratio: float = 0.8
    fight_min_score: float = 0.15


@dataclass
class CvRoiConfig:
    rect: Tuple[float, float, float, float]
    normalized: bool = True
    mode: str = "center"
    draw: bool = False
    padding: float = 0.0


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value


def resolve_model_path(
    model_hint: str,
    *,
    model_dir: Optional[str] = None,
    model_registry: Optional[Dict[str, str]] = None,
) -> str:
    if not model_hint:
        return model_hint
    hint = model_hint.strip()
    if not hint:
        return hint

    if model_registry and hint in model_registry:
        hint = model_registry[hint]

    if os.path.isabs(hint) or _looks_like_path(hint):
        return hint

    if model_dir:
        return str(Path(model_dir) / hint)

    return hint


@dataclass
class CvConfig:
    model_path: str
    device: str
    conf: float
    iou: float
    imgsz: int
    max_det: int
    max_boxes: int
    classes: Optional[List[int]] = None
    model_dir: Optional[str] = None
    model_registry: Dict[str, str] = field(default_factory=lambda: dict(CV_MODEL_REGISTRY))
    task: Optional[str] = None
    pose: CvPoseConfig = field(default_factory=CvPoseConfig)
    roi: Optional[CvRoiConfig] = None
    roi_by_id: Dict[str, CvRoiConfig] = field(default_factory=dict)


def build_cv_config(model_hint: Optional[str] = None) -> CvConfig:
    model_dir = _env_get("CV_MODEL_DIR") or _env_get("YOLO_MODEL_DIR")
    if not model_dir:
        model_dir = str(_PROJECT_ROOT / "storage" / "models" / "cv")
    else:
        model_dir_path = Path(model_dir)
        if not model_dir_path.is_absolute():
            model_dir = str((_PROJECT_ROOT / model_dir_path).resolve())

    default_model = (
        _env_get("CV_MODEL_PATH")
        or _env_get("YOLO_MODEL_PATH")
        or _env_get("CV_MODEL_NAME")
        or _env_get("YOLO_MODEL_NAME")
        or "yolov8n.pt"
    )

    selected_model = (model_hint or "").strip() or default_model
    resolved_model = resolve_model_path(
        selected_model,
        model_dir=model_dir,
        model_registry=CV_MODEL_REGISTRY,
    )

    is_om_model = str(resolved_model).lower().endswith(".om")
    conf_default = 0.05 if is_om_model else 0.25
    conf = None
    if is_om_model:
        conf = _env_get_float_optional("CV_OM_CONF")
    if conf is None:
        conf = _env_get_float("CV_CONF", conf_default)

    cfg = CvConfig(
        model_path=resolved_model,
        device=_default_device(),
        conf=float(conf),
        iou=_env_get_float("CV_IOU", 0.45),
        imgsz=_env_get_int("CV_IMGSZ", 640),
        max_det=_env_get_int("CV_MAX_DET", 300),
        max_boxes=_env_get_int("CV_MAX_BOXES", 12),
        classes=_env_get_list_int("CV_CLASSES"),
        model_dir=model_dir,
        model_registry=dict(CV_MODEL_REGISTRY),
    )

    return cfg
