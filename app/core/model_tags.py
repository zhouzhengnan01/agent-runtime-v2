from __future__ import annotations

from typing import Iterable


MODEL_TAGS: tuple[str, ...] = (
    "chat",
    "reasoning",
    "vision",
    "embedding",
    "tool_call",
    "image_generation",
    "video_generation",
    "audio_generation",
    "text_to_speech",
    "speech_to_text",
    "vision_segmentation",
    "rerank",
)

_ALIASES = {
    "tool_calling": "tool_call",
    "tool-calling": "tool_call",
    "toolcall": "tool_call",
    "tts": "text_to_speech",
    "stt": "speech_to_text",
    "segmentation": "vision_segmentation",
    "vision-segmentation": "vision_segmentation",
    "reranking": "rerank",
}


def normalize_model_tags(value: object) -> list[str]:
    """Return known model capability tags in stable order without duplicates."""

    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    allowed = set(MODEL_TAGS)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = item.strip().lower().replace("-", "_")
        tag = _ALIASES.get(tag, tag)
        if tag not in allowed or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized
