from __future__ import annotations

import re
from typing import Any


SKILL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "algorithm-engineer": (
        "完整算法工程师应用",
        "算法工程师应用",
        "完整应用",
        "一站式",
        "工作台",
        "全流程搞定",
        "整合",
        "算法工程师",
        "全流程",
        "算法迭代",
        "训练模型",
        "检测模型",
        "模型上线",
        "端到端",
        "full cycle",
    ),
    "algorithm-research-scout": (
        "论文",
        "调研",
        "算法对比",
        "rtdetr",
        "rt-detr",
        "deim",
        "d-fine",
        "rf-detr",
        "damo-yolo",
        "比 yolo",
        "更好的算法",
    ),
    "dataset-curator": (
        "数据集",
        "数据治理",
        "去重",
        "标签冲突",
        "yolo 格式",
        "划分训练集",
        "缺标签",
    ),
    "model-candidate-selector": (
        "候选模型",
        "选择模型",
        "模型选择",
        "baseline",
        "显存",
        "部署难度",
    ),
    "remote-gpu-ops": (
        "gpu 机器",
        "远端",
        "同步数据",
        "nvidia-smi",
        "rsync",
        "agx",
        "5090",
    ),
    "gpu-training-orchestrator": (
        "训练",
        "benchmark",
        "epochs",
        "batch",
        "断点恢复",
        "results.csv",
    ),
    "cpu-training-runner": (
        "cpu 训练",
        "CPU 训练",
        "沙盒训练",
        "启动沙盒",
        "best.pt",
        "device=cpu",
        "yolo train",
    ),
    "detector-evaluator": (
        "评估",
        "map50",
        "precision",
        "recall",
        "计数误差",
        "ab",
        "a/b",
    ),
    "deployment-candidate-reviewer": (
        "上线",
        "部署",
        "替换线上",
        "回滚",
        "候选模型",
        "推理服务",
    ),
    "experiment-ledger": (
        "实验记录",
        "ledger",
        "复盘",
        "记录实验",
        "对比实验",
    ),
}


DEFAULT_SKILL = "algorithm-engineer"


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if skill_name not in allowed_skills or skill_name not in SKILL_KEYWORDS:
        return 0
    text = _normalize(routing_text)
    score = 0
    for keyword in SKILL_KEYWORDS[skill_name]:
        if keyword in text:
            score += 24
    if skill_name == "algorithm-engineer" and any(
        word in text for word in ("完整", "应用", "整合", "工作台", "全流程搞定", "全流程", "端到端", "搞定", "算法工程师")
    ):
        score += 90
    if attachments:
        score += 6
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
    spec["task"] = routing_text or user_text
    spec["objective"] = _objective(user_text or routing_text)
    spec["domain"] = _domain(user_text or routing_text)
    spec["baseline"] = _baseline(user_text or routing_text)
    spec["dataset_path"] = _extract_path(user_text or routing_text)
    spec["data_yaml"] = _extract_data_yaml(user_text or routing_text)
    spec["model"] = _model(user_text or routing_text)
    spec["epochs"] = _int_after_key(user_text or routing_text, "epochs", default=1)
    spec["imgsz"] = _int_after_key(user_text or routing_text, "imgsz", default=320)
    spec["batch"] = _int_after_key(user_text or routing_text, "batch", default=1)
    spec["mock"] = "mock" in _normalize(user_text or routing_text) or "模拟" in _normalize(user_text or routing_text)
    spec["machines"] = _machines(user_text or routing_text)
    spec["candidate_algorithms"] = _candidate_algorithms(user_text or routing_text)
    spec["business_metrics"] = ["totalAccuracy", "netTotalAccuracy", "absDiffAvg", "per-tree count error"]
    spec["detection_metrics"] = ["precision", "recall", "mAP50", "mAP50-95"]
    spec["attachments"] = [_attachment_payload(item) for item in attachments]
    spec["safety_rules"] = [
        "不停止已有训练，除非用户明确要求。",
        "不停止线上推理服务，除非用户明确同意。",
        "不在小系统盘放大数据集。",
        "长训练前必须先跑小 benchmark。",
    ]
    return spec


def _normalize(value: str) -> str:
    return value.lower().replace("rt detr", "rt-detr").replace("d fine", "d-fine")


def _objective(text: str) -> str:
    normalized = _normalize(text)
    if "计数" in normalized:
        return "果串检测与计数误差优化"
    if "标注" in normalized:
        return "自动预标注与训练数据闭环"
    if "上线" in normalized or "部署" in normalized:
        return "候选检测模型上线评审"
    return "目标检测算法迭代"


def _domain(text: str) -> str:
    normalized = _normalize(text)
    if "棕榈" in normalized or "palm" in normalized:
        return "palm fruit detection"
    return "computer vision object detection"


def _baseline(text: str) -> str:
    normalized = _normalize(text)
    for item in ("yolo11", "yolov11", "yolov8", "rt-detr", "rtdetr", "sam3"):
        if item in normalized:
            return item
    return "production YOLO baseline"


def _extract_path(text: str) -> str:
    for match in re.finditer(r"(?<![A-Za-z0-9])(/[A-Za-z0-9._/\\-]+)", text):
        path = match.group(1)
        if _looks_like_filesystem_path(path):
            return path
    return "/data/palm_fruit_datasets/organized/latest_integrated_dedup"


def _extract_data_yaml(text: str) -> str:
    keyed = re.search(r"(?:data_yaml|data)\s*[=:：]\s*([^\s'\"，。；;]+?\.ya?ml)", text, flags=re.IGNORECASE)
    if keyed:
        return keyed.group(1)
    for match in re.finditer(r"(?<![A-Za-z0-9])(/[^\s'\"，。；;]+?\.ya?ml)", text):
        return match.group(1)
    return ""


def _model(text: str) -> str:
    normalized = _normalize(text)
    for item in ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt", "yolov8s.pt", "rtdetr-l.pt"):
        if item.lower() in normalized:
            return item
    return "yolo11n.pt"


def _int_after_key(text: str, key: str, *, default: int) -> int:
    normalized = _normalize(text)
    patterns = [
        rf"{re.escape(key)}\s*[=:：]\s*(\d+)",
        rf"{re.escape(key)}\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return default
    return default


def _looks_like_filesystem_path(path: str) -> bool:
    lowered = path.lower()
    if any(part in lowered for part in ("/data/", "/mnt/", "/home/", "/Users/".lower(), "/workspace/", "/datasets/")):
        return True
    if lowered.endswith((".yaml", ".yml", ".json", ".csv", ".txt", ".jpg", ".jpeg", ".png", ".pt")):
        return True
    return len(path.split("/")) >= 3


def _machines(text: str) -> list[str]:
    normalized = _normalize(text)
    machines = []
    if "agx" in normalized:
        machines.append("AGX Orin")
    if "5090" in normalized:
        machines.append("5090 GPU server")
    return machines or ["AGX Orin", "5090 GPU server"]


def _candidate_algorithms(text: str) -> list[str]:
    normalized = _normalize(text)
    known = [
        ("YOLO baseline", ("yolo",)),
        ("RT-DETR-L/X", ("rt-detr", "rtdetr")),
        ("DEIM / D-FINE", ("deim", "d-fine")),
        ("RF-DETR", ("rf-detr",)),
        ("DAMO-YOLO", ("damo",)),
    ]
    selected = [name for name, keys in known if any(key in normalized for key in keys)]
    return selected or ["YOLO baseline", "RT-DETR-L/X", "DEIM / D-FINE", "RF-DETR", "DAMO-YOLO"]


def _attachment_payload(attachment: Any) -> dict[str, Any]:
    if isinstance(attachment, dict):
        return dict(attachment)
    model_dump = getattr(attachment, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, dict) else {}
    return {}
