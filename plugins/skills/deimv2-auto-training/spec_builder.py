from __future__ import annotations

import re
from typing import Any


DEFAULT_MODEL_VARIANT = "deimv2-dinov3-s"
MODEL_VARIANTS = {"deimv2-dinov3-s", "deimv2-dinov3-m", "deimv2-dinov3-l", "deimv2-dinov3-x"}


KEYWORDS = (
    "deimv2",
    "deim",
    "dinov3",
    "dino",
    "目标检测训练",
    "自动训练",
    "5090",
    "npu",
    "gpu",
)


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if skill_name != "deimv2-auto-training" or skill_name not in allowed_skills:
        return 0
    text = (routing_text or "").lower()
    score = sum(32 for keyword in KEYWORDS if keyword.lower() in text)
    if attachments:
        score += 8
    return score


def build_spec(
    skill_name: str,
    user_text: str,
    routing_text: str,
    attachments: list[Any],
    base_spec: dict[str, Any],
) -> dict[str, Any]:
    del attachments
    text = user_text or routing_text
    spec = dict(base_spec)
    spec["skill_name"] = skill_name
    spec.setdefault("training", {})
    training = spec["training"] if isinstance(spec["training"], dict) else {}
    training.setdefault("epochs", _int_after_key(text, "epochs", 10))
    training.setdefault("img_size", _int_after_key(text, "imgsz", _int_after_key(text, "图片尺寸", 640)))
    training.setdefault("batch", _int_after_key(text, "batch", 1))
    training.setdefault("device", "auto")
    training.setdefault("model_variant", _model_variant_from_text(text))
    spec["training"] = training
    return spec


def _model_variant_from_text(text: str) -> str:
    lowered = (text or "").lower().replace("_", "-")
    for size in ("s", "m", "l", "x"):
        if re.search(rf"\b(?:deimv2[- ]*)?(?:dinov3[- ]*)?{size}\b", lowered):
            variant = f"deimv2-dinov3-{size}"
            if variant in MODEL_VARIANTS:
                return variant
    return DEFAULT_MODEL_VARIANT


def _int_after_key(text: str, key: str, default: int) -> int:
    match = re.search(rf"{re.escape(key)}\s*[=:：]?\s*(\d+)", text or "", flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default
