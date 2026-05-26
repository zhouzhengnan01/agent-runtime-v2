from __future__ import annotations

import re
from typing import Any


KEYWORDS = ("train", "training", "yolo", "epochs", "batch", "imgsz", "best.pt", "训练")


def score_skill(skill_name: str, routing_text: str, attachments: list[Any], allowed_skills: list[str]) -> int:
    del attachments
    if skill_name != "gpu-training-orchestrator" or skill_name not in allowed_skills:
        return 0
    text = routing_text.lower()
    return sum(20 for keyword in KEYWORDS if keyword.lower() in text)


def build_spec(skill_name: str, user_text: str, routing_text: str, attachments: list[Any], base_spec: dict[str, Any]) -> dict[str, Any]:
    del attachments
    text = user_text or routing_text
    spec = dict(base_spec)
    spec["skill_name"] = skill_name
    spec["overrides_text"] = text
    data_yaml = _value_after_keys(text, ("data_yaml", "dataset_yaml"))
    if data_yaml:
        spec["data_yaml"] = data_yaml
    spec["training"] = {
        "task": _value_after_keys(text, ("training_task", "task")) or "detect",
        "model": _value_after_keys(text, ("model",)) or _model_guess(text),
        "epochs": _int_after_keys(text, ("epochs", "epoch"), 50),
        "imgsz": _int_after_keys(text, ("imgsz",), 640),
        "batch": _int_after_keys(text, ("batch", "batch_size"), 16),
        "device": _value_after_keys(text, ("device",)) or "0",
        "workers": _int_after_keys(text, ("workers",), 4),
        "patience": _int_after_keys(text, ("patience",), 8),
    }
    spec["runtime"] = {
        "conda_env_name": _value_after_keys(text, ("conda_env_name", "conda_env")) or "yolo",
        "enforce_conda_env": _has_flag(text, ("enforce_conda_env",)),
    }
    output_project = _value_after_keys(text, ("project_dir",))
    run_name = _value_after_keys(text, ("run_name",))
    if output_project or run_name:
        spec["output"] = {"project_dir": output_project, "run_name": run_name or "."}
    return spec


def _value_after_keys(text: str, keys: tuple[str, ...]) -> str:
    for key in keys:
        match = re.search(rf"{re.escape(key)}\s*[:=：]\s*([^\n,，;；\s]+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'`")
    return ""


def _int_after_keys(text: str, keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        match = re.search(rf"{re.escape(key)}\s*[:=：]\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return default


def _has_flag(text: str, keys: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(key.lower() in lowered for key in keys)


def _model_guess(text: str) -> str:
    lowered = text.lower()
    for item in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "rtdetr-l.pt"):
        if item in lowered:
            return item
    return "yolo11n.pt"
