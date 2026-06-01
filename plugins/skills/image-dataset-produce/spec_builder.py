from __future__ import annotations

import re
from typing import Any


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if skill_name != "image-dataset-produce" or skill_name not in allowed_skills:
        return 0
    text = routing_text.lower()
    keywords = (
        "合成图片",
        "生成图片",
        "数据增强",
        "扩充数据集",
        "训练数据生成",
        "参考图生成",
        "image generation",
        "synthetic dataset",
        "augmentation",
        "flux",
    )
    score = 80 if any(keyword in text for keyword in keywords) else 0
    if _first_image_path(attachments):
        score += 30
    return score


def build_spec(
    skill_name: str,
    user_text: str,
    routing_text: str,
    attachments: list[Any],
    base_spec: dict[str, Any],
) -> dict[str, Any]:
    spec = dict(base_spec)
    text = user_text or routing_text
    spec["skill_name"] = skill_name
    spec.setdefault("prompt", _prompt_from_text(text))
    image_path = _first_image_path(attachments)
    if image_path and not spec.get("input_image"):
        spec["input_image"] = image_path
    spec.setdefault("api_url", "http://218.67.242.10:58801/flux2/generate")
    spec.setdefault("timeout", 120)
    spec["attachments"] = [_attachment_payload(item) for item in attachments]
    return spec


def _prompt_from_text(text: str) -> str:
    match = re.search(r"(?:prompt|提示词)\s*[:：=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _first_image_path(attachments: list[Any]) -> str:
    for attachment in attachments:
        payload = _attachment_payload(attachment)
        mime_type = str(payload.get("mime_type") or payload.get("content_type") or "").lower()
        name = str(payload.get("name") or payload.get("filename") or "").lower()
        path = str(payload.get("path") or payload.get("local_path") or payload.get("file_path") or "").strip()
        if path and (mime_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS) or path.lower().endswith(IMAGE_EXTENSIONS)):
            return path
    return ""


def _attachment_payload(attachment: Any) -> dict[str, Any]:
    if isinstance(attachment, dict):
        return dict(attachment)
    model_dump = getattr(attachment, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, dict) else {}
    return {}
