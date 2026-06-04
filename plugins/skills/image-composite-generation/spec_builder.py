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
    if skill_name != "image-composite-generation" or skill_name not in allowed_skills:
        return 0
    text = routing_text.lower()
    keywords = (
        "图片合成",
        "图像合成",
        "合成图片",
        "合成图像",
        "二图合成",
        "背景图",
        "前景图",
        "数据增强",
        "扩充数据集",
        "训练数据生成",
        "composite",
        "image composition",
        "synthetic dataset",
        "augmentation",
    )
    score = 85 if any(keyword in text for keyword in keywords) else 0
    if len(_image_paths(attachments)) >= 2:
        score += 35
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
    image_paths = _image_paths(attachments)
    if image_paths and not spec.get("image1"):
        spec["image1"] = image_paths[0]
    if len(image_paths) > 1 and not spec.get("image2"):
        spec["image2"] = image_paths[1]
    spec.setdefault("api_url", "https://www.tokencloud.yun/v1/images/generations")
    spec.setdefault("token", "sk-6qbqlTAZV7qVszLplZhxZH2yvQwSMRotrOT8wr9aVPII8BYo")
    spec.setdefault("model", "wan2.7-image-pro")
    spec.setdefault("timeout", 180)
    spec.setdefault("size", "768*768")
    spec.setdefault("count", 1)
    spec["attachments"] = [_attachment_payload(item) for item in attachments]
    return spec


def _prompt_from_text(text: str) -> str:
    match = re.search(r"(?:prompt|提示词|composite_prompt|合成提示词)\s*[:：=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _image_paths(attachments: list[Any]) -> list[str]:
    paths: list[str] = []
    for attachment in attachments:
        payload = _attachment_payload(attachment)
        mime_type = str(payload.get("mime_type") or payload.get("content_type") or "").lower()
        name = str(payload.get("name") or payload.get("filename") or "").lower()
        path = str(payload.get("path") or payload.get("local_path") or payload.get("file_path") or "").strip()
        path_lower = path.lower()
        if path and (mime_type.startswith("image/") or name.endswith(IMAGE_EXTENSIONS) or path_lower.endswith(IMAGE_EXTENSIONS)):
            paths.append(path)
    return paths


def _attachment_payload(attachment: Any) -> dict[str, Any]:
    if isinstance(attachment, dict):
        return dict(attachment)
    model_dump = getattr(attachment, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, dict) else {}
    return {}
