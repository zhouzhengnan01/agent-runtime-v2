from __future__ import annotations

import re
from typing import Any


IMAGE_MIME_PREFIX = "image/"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_LABELS = ["person", "monitor"]


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if skill_name != "data-auto-annotation" or skill_name not in allowed_skills:
        return 0
    text = routing_text.lower()
    keywords = (
        "自动标注",
        "数据标注",
        "预标注",
        "目标检测",
        "coco",
        "sam3",
        "annotation",
        "labeling",
        "dataset",
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
    spec["skill_name"] = skill_name
    spec.setdefault("format", "coco")
    spec.setdefault("title", "数据自动标注 COCO 导出")

    image_path = _first_image_path(attachments)
    if image_path and not spec.get("image_path"):
        spec["image_path"] = image_path

    labels = _labels_from_text(user_text or routing_text)
    if labels and not spec.get("labels"):
        spec["labels"] = labels
    elif not spec.get("labels"):
        spec["labels"] = DEFAULT_LABELS

    spec["attachments"] = [_attachment_payload(item) for item in attachments]
    spec["quality_requirements"] = ["coco_schema", "image_dimensions", "bbox_xywh", "category_mapping"]
    spec["verification_rules"] = list(spec["quality_requirements"])
    return spec


def _first_image_path(attachments: list[Any]) -> str:
    for attachment in attachments:
        payload = _attachment_payload(attachment)
        mime_type = str(payload.get("mime_type") or payload.get("content_type") or "").lower()
        name = str(payload.get("name") or payload.get("filename") or "").lower()
        path = str(payload.get("path") or payload.get("local_path") or payload.get("file_path") or "").strip()
        if path and (mime_type.startswith(IMAGE_MIME_PREFIX) or name.endswith(IMAGE_EXTENSIONS) or path.lower().endswith(IMAGE_EXTENSIONS)):
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


def _labels_from_text(text: str) -> list[str]:
    explicit = re.search(r"(?:labels?|类别|标签|目标)\s*[:：=]\s*([^\n，。；;]+)", text, flags=re.IGNORECASE)
    if explicit:
        return _split_labels(explicit.group(1))
    quoted = re.findall(r"`([^`]+)`", text)
    if quoted:
        labels = []
        for item in quoted:
            labels.extend(_split_labels(item))
        if labels:
            return labels
    return []


def _split_labels(value: str) -> list[str]:
    labels = [item.strip() for item in re.split(r"[,，/、\s]+", value) if item.strip()]
    return labels[:20]
