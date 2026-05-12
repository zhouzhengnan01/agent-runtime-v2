from __future__ import annotations

from typing import Any


KEYWORDS = (
    "论文",
    "调研",
    "算法对比",
    "rt-detr",
    "rtdetr",
    "deim",
    "d-fine",
    "rf-detr",
    "damo-yolo",
    "比 yolo",
    "更好的算法",
    "候选算法",
)


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    del attachments
    if skill_name != "algorithm-research-scout" or skill_name not in allowed_skills:
        return 0
    text = routing_text.lower().replace("rt detr", "rt-detr").replace("d fine", "d-fine")
    score = sum(24 for keyword in KEYWORDS if keyword in text)
    if score and ("棕榈" in text or "palm" in text):
        score += 20
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
    spec["task_type"] = "object_detection"
    spec["objective"] = "palm fruit detection" if ("棕榈" in text or "palm" in text.lower()) else "object detection"
    spec.setdefault("baseline", "production YOLO baseline")
    spec.setdefault("max_results", 8)
    spec.setdefault("mode", "online-research")
    spec["constraints"] = ["AGX Orin deployment", "RTX 5090 training", "license risk check", "training script maturity"]
    return spec
