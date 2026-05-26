from __future__ import annotations

import re
from typing import Any


IMAGE_MIME_PREFIX = "image/"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DATASET_EXTENSIONS = (".zip", ".tar", ".tar.gz")


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
    if _first_data_path(attachments):
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
    spec.setdefault("format", "coco")
    spec.setdefault("title", "数据自动标注 COCO 导出")

    data_path = _first_data_path(attachments) or _path_from_uploaded_files_text(text)
    if data_path:
        if _looks_like_archive(data_path):
            spec.setdefault("dataset_archive", data_path)
            spec.setdefault("dataset_root", data_path)
        else:
            spec.setdefault("image_path", data_path)
            spec.setdefault("dataset_root", data_path)

    labels = _labels_from_text(text)
    if labels and not spec.get("labels"):
        spec["labels"] = labels
    if "skip_generation" not in spec:
        spec["skip_generation"] = False
    spec.setdefault("split_requested", _wants_split(text))
    spec.setdefault(
        "planner_llm",
        {
            "base_url": "http://124.132.152.75:62092/v1",
            "api_key": "abc@123",
            "model": "Qwen3.6-35B-A3B",
            "temperature": 0.2,
            "max_tokens": 1024,
            "timeout": 120,
        },
    )

    spec["attachments"] = [_attachment_payload(item) for item in attachments]
    spec["quality_requirements"] = ["coco_schema", "image_dimensions", "bbox_xywh", "category_mapping"]
    spec["verification_rules"] = list(spec["quality_requirements"])
    return spec


def _first_data_path(attachments: list[Any]) -> str:
    for attachment in attachments:
        payload = _attachment_payload(attachment)
        mime_type = str(payload.get("mime_type") or payload.get("content_type") or "").lower()
        name = str(payload.get("name") or payload.get("filename") or "").lower()
        path = str(payload.get("path") or payload.get("local_path") or payload.get("file_path") or "").strip()
        path_lower = path.lower()
        if path and (
            mime_type.startswith(IMAGE_MIME_PREFIX)
            or name.endswith(IMAGE_EXTENSIONS)
            or path_lower.endswith(IMAGE_EXTENSIONS)
            or name.endswith(DATASET_EXTENSIONS)
            or path_lower.endswith(DATASET_EXTENSIONS)
        ):
            return path
    return ""


def _path_from_uploaded_files_text(text: str) -> str:
    for match in re.finditer(r"path=([^,\n]+)", text or "", flags=re.IGNORECASE):
        value = match.group(1).strip()
        lower = value.lower()
        if lower.endswith(IMAGE_EXTENSIONS) or lower.endswith(DATASET_EXTENSIONS):
            return value
    return ""


def _looks_like_archive(path: str) -> bool:
    lower = (path or "").lower()
    return lower.endswith(DATASET_EXTENSIONS)


def _attachment_payload(attachment: Any) -> dict[str, Any]:
    if isinstance(attachment, dict):
        return dict(attachment)
    model_dump = getattr(attachment, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _labels_from_text(text: str) -> list[str]:
    explicit = re.search(r"(?:labels?|class_names|类别|标签|目标|检测类别)\s*[:：=]\s*([^\n。；;]+)", text, flags=re.IGNORECASE)
    if explicit:
        return _split_labels(explicit.group(1))
    quoted = re.findall(r"`([^`]+)`", text)
    labels: list[str] = []
    for item in quoted:
        labels.extend(_split_labels(item))
    return labels[:20]


def _split_labels(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，、\s]+", value) if item.strip()][:20]


def _wants_generation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        keyword in lowered
        for keyword in (
            "\u5408\u6210",
            "\u751f\u6210",
            "\u751f\u56fe",
            "\u6570\u636e\u589e\u5f3a",
            "\u6269\u5145",
            "augmentation",
            "synthetic",
            "composite",
            "generate",
        )
    )


def _wants_split(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        keyword in lowered
        for keyword in (
            "\u5212\u5206",
            "\u8bad\u7ec3\u96c6",
            "\u9a8c\u8bc1\u96c6",
            "\u6d4b\u8bd5\u96c6",
            "split",
            "train",
            "val",
            "valid",
            "test",
            "dataset.split",
        )
    )
