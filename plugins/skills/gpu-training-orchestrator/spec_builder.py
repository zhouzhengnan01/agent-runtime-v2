from __future__ import annotations

import re
from typing import Any


KEYWORDS = (
    "gpu 训练",
    "训练编排",
    "benchmark",
    "正式训练",
    "epochs",
    "batch",
    "断点恢复",
    "results.csv",
    "best.pt",
    "last.pt",
    "5090",
    "agx",
)


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if skill_name != "gpu-training-orchestrator" or skill_name not in allowed_skills:
        return 0
    text = routing_text.lower()
    score = sum(24 for keyword in KEYWORDS if keyword in text)
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
    spec.setdefault("model", _model(text))
    spec.setdefault("data", _data_path(text))
    spec.setdefault("epochs", _int_after_key(text, "epochs", 1))
    spec.setdefault("batch", _int_after_key(text, "batch", 1))
    spec.setdefault("benchmark_first", True)
    spec.setdefault("machines", _machines(text))
    return spec


def _model(text: str) -> str:
    lowered = text.lower()
    for item in ("yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt", "rtdetr-l.pt"):
        if item in lowered:
            return item
    return "yolo11n.pt"


def _data_path(text: str) -> str:
    match = re.search(r"(?:data|data_yaml)\s*[=:：]\s*([^\s'\"，。；;]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(/[^\s'\"，。；;]+?\.ya?ml)", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _int_after_key(text: str, key: str, default: int) -> int:
    match = re.search(rf"{re.escape(key)}\s*[=:：]\s*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _machines(text: str) -> list[str]:
    lowered = text.lower()
    machines = []
    if "agx" in lowered:
        machines.append("AGX Orin")
    if "5090" in lowered:
        machines.append("RTX 5090")
    return machines or ["AGX Orin", "RTX 5090"]
