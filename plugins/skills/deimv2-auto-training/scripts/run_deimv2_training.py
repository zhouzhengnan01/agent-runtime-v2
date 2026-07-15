from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SKILL_ROOT.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.hardware.device_selector import reserve_training_device

PROJECT_MODELS_ROOT = PROJECT_ROOT / "models"
VENDOR_DEIMV2_ROOT = SKILL_ROOT / "vendor" / "deimv2"
DEFAULT_MODEL_VARIANT = "deimv2-dinov3-s"
DEFAULT_BACKBONE_CHECKPOINT = Path("ckpts/vitt_distill.pt")
MODEL_VARIANTS = {
    "deimv2-dinov3-s": {
        "template_config": Path("configs/deimv2/deimv2_dinov3_s_coco.yml"),
        "tuning_checkpoint": Path("ckpts/deimv2_dinov3_s_coco.pth"),
    },
    "deimv2-dinov3-m": {
        "template_config": Path("configs/deimv2/deimv2_dinov3_m_coco.yml"),
        "tuning_checkpoint": Path("ckpts/deimv2_dinov3_m_coco.pth"),
    },
    "deimv2-dinov3-l": {
        "template_config": Path("configs/deimv2/deimv2_dinov3_l_coco.yml"),
        "tuning_checkpoint": Path("ckpts/deimv2_dinov3_l_coco.pth"),
    },
    "deimv2-dinov3-x": {
        "template_config": Path("configs/deimv2/deimv2_dinov3_x_coco.yml"),
        "tuning_checkpoint": Path("ckpts/deimv2_dinov3_x_coco.pth"),
    },
}


def normalize_model_variant(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return DEFAULT_MODEL_VARIANT
    normalized = raw.replace("_", "-").replace(" ", "")
    normalized = re.sub(r"-coco$", "", normalized)
    if normalized in {"s", "m", "l", "x"}:
        return f"deimv2-dinov3-{normalized}"
    if normalized in {"dinov3-s", "dinov3-m", "dinov3-l", "dinov3-x"}:
        return f"deimv2-{normalized}"
    return normalized if normalized in MODEL_VARIANTS else DEFAULT_MODEL_VARIANT


def model_spec(training: dict[str, Any]) -> dict[str, Path]:
    return MODEL_VARIANTS[normalize_model_variant(training.get("model_variant"))]


def load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Input spec is not an object: {path}")
    return payload


def resolve_deimv2_root(spec: dict[str, Any]) -> Path:
    candidates = [
        os.environ.get("DEIMV2_ROOT", ""),
        str(spec.get("deimv2_root") or ""),
        str(VENDOR_DEIMV2_ROOT),
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser().resolve()
        if (path / "train.py").is_file() and (path / "engine").is_dir():
            return path
    raise FileNotFoundError("DEIMv2 root not found. Set DEIMV2_ROOT or bundle vendor/deimv2.")


def python_prefix(training: dict[str, Any]) -> list[str]:
    explicit = str(training.get("python") or "").strip()
    if explicit:
        return [explicit]
    conda_env = str(training.get("conda_env") or training.get("conda_env_name") or "").strip()
    if conda_env and os.environ.get("CONDA_DEFAULT_ENV") != conda_env:
        return [str(training.get("conda_exe") or "conda"), "run", "--no-capture-output", "-n", conda_env, "python"]
    return [sys.executable]


def detect_target_hardware(prefix: list[str]) -> dict[str, Any]:
    cmd = prefix + [str(SKILL_ROOT / "scripts" / "detect_hardware.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Hardware detection failed:\n{proc.stdout}\n{proc.stderr}")
    start = proc.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"Hardware detector returned no JSON: {proc.stdout}")
    payload, _ = json.JSONDecoder().raw_decode(proc.stdout[start:])
    return payload


def choose_template(deim_root: Path, training: dict[str, Any]) -> Path:
    requested = str(training.get("template_config") or "").strip()
    path = Path(requested) if requested else model_spec(training)["template_config"]
    path = path if path.is_absolute() else deim_root / path
    if not path.is_file():
        raise FileNotFoundError(f"DEIMv2 template not found: {path}")
    if "dinov3" not in path.name.lower():
        raise ValueError(f"Only DEIMv2 DINOv3 templates are allowed here: {path}")
    return path.resolve()


def resolve_checkpoint(
    deim_root: Path,
    value: str,
    default: Path | None = None,
    *,
    env_name: str = "",
) -> Path | None:
    candidates: list[str] = []
    env_value = os.environ.get(env_name, "").strip() if env_name else ""
    if env_value:
        candidates.append(env_value)
    requested = Path(value).expanduser() if value else None
    if requested is not None and requested.is_absolute():
        candidates.append(value)
    elif requested is not None:
        candidates.extend(
            [
                str(PROJECT_ROOT / requested),
                str(PROJECT_MODELS_ROOT / requested.name),
            ]
        )
        if requested.parts and requested.parts[0].lower() != "models":
            candidates.append(value)
    if default is not None:
        project_model_candidates = [
            PROJECT_MODELS_ROOT / "deimv2" / default.name,
            PROJECT_MODELS_ROOT / default.name,
        ]
        candidates.extend([
            *(str(path) for path in project_model_candidates),
            *([] if requested is None or requested.is_absolute() or value in candidates else [value]),
            str(deim_root / default),
            str(VENDOR_DEIMV2_ROOT / default),
            str(Path("/models/deimv2") / default.name),
        ])
    for raw in candidates:
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else deim_root / path
        if path.is_file():
            return path.resolve()
    if value or default == DEFAULT_BACKBONE_CHECKPOINT:
        raise FileNotFoundError(f"DEIMv2 checkpoint not found. Tried: {candidates}")
    return None


def resolve_tuning_checkpoint(deim_root: Path, training: dict[str, Any]) -> Path | None:
    if training.get("disable_tuning_checkpoint"):
        return None
    explicit = str(training.get("tuning_checkpoint") or "").strip()
    if explicit or os.environ.get("DEIMV2_TUNING_CHECKPOINT"):
        return resolve_checkpoint(deim_root, explicit, env_name="DEIMV2_TUNING_CHECKPOINT")
    default_tuning_checkpoint = model_spec(training)["tuning_checkpoint"]
    for raw in (
        PROJECT_MODELS_ROOT / "deimv2" / default_tuning_checkpoint.name,
        PROJECT_MODELS_ROOT / default_tuning_checkpoint.name,
        deim_root / default_tuning_checkpoint,
        VENDOR_DEIMV2_ROOT / default_tuning_checkpoint,
        Path("/models/deimv2") / default_tuning_checkpoint.name,
    ):
        if raw.is_file():
            return raw.resolve()
    return None


def merge_yaml_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if key == "__include__":
            continue
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            merge_yaml_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.Loader) or {}
    return payload if isinstance(payload, dict) else {}


def load_resolved_yaml(path: Path) -> dict[str, Any]:
    # DEIMv2 模板通过 __include__ 分层组合。这里解析成一个完整字典，
    # 让生成的 train.yml 更适合审核和部署排障。
    payload = load_yaml_payload(path)
    resolved: dict[str, Any] = {}
    for include_path in resolve_yaml_include_paths(path, payload):
        if include_path.is_file():
            merge_yaml_dict(resolved, load_resolved_yaml(include_path.resolve()))
    merge_yaml_dict(resolved, payload)
    return resolved


def resolve_yaml_include_paths(path: Path, payload: dict[str, Any] | None = None) -> list[Path]:
    payload = payload if payload is not None else load_yaml_payload(path)
    includes = payload.get("__include__") or []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    result: list[Path] = []
    for raw_include in includes:
        include_path = Path(str(raw_include)).expanduser()
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        result.append(include_path.resolve())
    return result


def visible_include_paths(train_yml: Path) -> list[str]:
    # 在生成的 YAML 中保留叶子 include 路径。下面写入的是最终生效配置，
    # include 记录用于追溯官方 vendor 模板来源。
    payload = load_yaml_payload(train_yml)
    visible: list[Path] = []
    for include_path in resolve_yaml_include_paths(train_yml, payload):
        include_payload = load_yaml_payload(include_path) if include_path.is_file() else {}
        child_includes = resolve_yaml_include_paths(include_path, include_payload)
        if child_includes:
            visible.extend(child_includes)
        else:
            visible.append(include_path)
    unique: list[str] = []
    seen: set[str] = set()
    for path in visible:
        rel = os.path.relpath(path, train_yml.parent).replace("\\", "/")
        if rel not in seen:
            seen.add(rel)
            unique.append(rel)
    return unique


def _float_training_value(training: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = training.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def apply_model_training_overrides(resolved: dict[str, Any], training: dict[str, Any]) -> None:
    optimizer_override = training.get("optimizer") if isinstance(training.get("optimizer"), dict) else {}
    if optimizer_override:
        optimizer = resolved.get("optimizer") if isinstance(resolved.get("optimizer"), dict) else {}
        merge_yaml_dict(optimizer, optimizer_override)
        resolved["optimizer"] = optimizer
    lr = _float_training_value(training, "lr", "learning_rate")
    if lr is not None:
        resolved.setdefault("optimizer", {})["lr"] = lr
    weight_decay = _float_training_value(training, "weight_decay")
    if weight_decay is not None:
        resolved.setdefault("optimizer", {})["weight_decay"] = weight_decay
    betas = training.get("betas")
    if isinstance(betas, list) and betas:
        resolved.setdefault("optimizer", {})["betas"] = betas


def write_resolved_train_yaml(train_yml: Path, training: dict[str, Any]) -> None:
    include_paths = visible_include_paths(train_yml.resolve())
    resolved = load_resolved_yaml(train_yml.resolve())
    apply_model_training_overrides(resolved, training)
    # 最终 train.yml 同时包含 __include__ 来源和合并后的生效字段。
    # 运维/算法同事只看一个文件就能确认完整训练配置。
    output: dict[str, Any] = {"__include__": include_paths}
    output.update(resolved)
    train_yml.write_text(yaml.dump(output, allow_unicode=True, sort_keys=False), encoding="utf-8")


def ypath(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_categories(dataset_root: Path) -> list[dict[str, Any]]:
    train_json = dataset_root / "annotations" / "instances_train.json"
    payload = json.loads(train_json.read_text(encoding="utf-8-sig"))
    categories = payload.get("categories") if isinstance(payload, dict) else []
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"No categories in {train_json}")
    return [cat for cat in categories if isinstance(cat, dict)]


def write_runtime_configs(
    *,
    configs_dir: Path,
    template: Path,
    dataset_root: Path,
    output_dir: Path,
    training: dict[str, Any],
    hardware: str,
    epochs: int,
    backbone_checkpoint: Path,
    stage: str,
) -> Path:
    configs_dir.mkdir(parents=True, exist_ok=True)
    categories = read_categories(dataset_root)
    class_names = [str(cat.get("name") or "") for cat in categories]
    img_size = int(training.get("img_size") or training.get("imgsz") or (640 if hardware == "cuda" else 320))
    default_batch = 8 if hardware == "cuda" else (4 if hardware == "npu" else 1)
    batch = max(1, int(training.get("batch") or default_batch))
    workers = max(0, int(training.get("workers") if training.get("workers") is not None else (4 if hardware == "cuda" else 0)))
    flat_epoch = int(training.get("flat_epoch") or max(1, epochs // 2))
    no_aug_epoch = int(training.get("no_aug_epoch") if training.get("no_aug_epoch") is not None else 0)
    policy_end = max(1, epochs)
    warmup_iter = int(training.get("warmup_iter") or max(1, min(100, epochs)))
    checkpoint_freq = int(training.get("checkpoint_freq") or max(1, epochs // 2))
    dataset_yml = configs_dir / "dataset.yml"
    train_yml = configs_dir / f"{stage}.yml"
    annotations = dataset_root / "annotations"
    # 数据集路径来自已准备好的 COCO 划分。上游数据处理阶段负责保证
    # val/test 只包含真实图；这里仅把这些文件绑定到 DEIMv2 dataloader 结构。
    dataset_yml.write_text(
        "\n".join(
            [
                f"num_classes: {len(categories)}",
                "remap_mscoco_category: False",
                "train_dataloader:",
                f"  total_batch_size: {batch}",
                f"  num_workers: {workers}",
                "  drop_last: False",
                "  dataset:",
                f"    img_folder: '{ypath(dataset_root / 'images' / 'train')}'",
                f"    ann_file: '{ypath(annotations / 'instances_train.json')}'",
                "val_dataloader:",
                "  total_batch_size: 1",
                f"  num_workers: {workers}",
                "  dataset:",
                f"    img_folder: '{ypath(dataset_root / 'images' / 'val')}'",
                f"    ann_file: '{ypath(annotations / 'instances_val.json')}'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rel_template = os.path.relpath(template, configs_dir).replace("\\", "/")
    rel_dataset = os.path.relpath(dataset_yml, configs_dir).replace("\\", "/")
    train_yml.write_text(
        f"""__include__: ['{rel_template}', '{rel_dataset}']

output_dir: '{ypath(output_dir)}'
epoches: {epochs}
flat_epoch: {flat_epoch}
no_aug_epoch: {no_aug_epoch}
warmup_iter: {warmup_iter}
checkpoint_freq: {checkpoint_freq}
use_ema: False
eval_spatial_size: [{img_size}, {img_size}]
class_names: {json.dumps(class_names, ensure_ascii=False)}

DINOv3STAs:
  weights_path: '{ypath(backbone_checkpoint)}'

PostProcessor:
  num_top_queries: {int(training.get("num_top_queries") or 100)}

train_dataloader:
  dataset:
    transforms:
      ops:
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [{img_size}, {img_size}]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        epoch: [1, {flat_epoch}, {policy_end}]
  collate_fn:
    base_size: {img_size}
    base_size_repeat: 1
    stop_epoch: {epochs}
    mixup_prob: 0.0
    mixup_epochs: [1, {flat_epoch}]
    copyblend_epochs: [1, 0]
    ema_restart_decay: 0.9999

val_dataloader:
  dataset:
    transforms:
      ops:
        - {{type: Resize, size: [{img_size}, {img_size}]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
""",
        encoding="utf-8",
    )
    write_resolved_train_yaml(train_yml, training)
    return train_yml


def training_command(
    prefix: list[str],
    deim_root: Path,
    config: Path,
    hardware: str,
    tuning_checkpoint: Path | None,
    runtime_device: str | None = None,
) -> list[str]:
    train_py = deim_root / "train.py"
    if hardware == "npu":
        # NPU 启动放在 wrapper 中处理，让 torch_npu 环境适配逻辑
        # 不侵入官方 DEIMv2 train.py 源码。
        cmd = prefix + [
            str(SKILL_ROOT / "scripts" / "npu_train_launcher.py"),
            "--train-py",
            str(train_py),
            "--",
            "-c",
            str(config),
            "-d",
            str(runtime_device or "npu:0"),
            "--seed",
            "0",
        ]
    else:
        cmd = prefix + [str(train_py), "-c", str(config), "--seed", "0"]
        if hardware == "cuda":
            cmd.append("--use-amp")
        else:
            cmd.extend(["-d", "cpu"])
    if tuning_checkpoint:
        # -t 表示检测器 checkpoint 微调。DINOv3 backbone 通过 train.yml 中的
        # DINOv3STAs.weights_path 配置，不作为 -t 传入。
        cmd.extend(["-t", str(tuning_checkpoint)])
    return cmd


def choose_hardware(training: dict[str, Any], detected: dict[str, Any]) -> str:
    selected = str(detected.get("selected") or "cpu")
    requested = str(training.get("device") or "auto").strip().lower()
    aliases = {"gpu": "cuda", "cuda": "cuda", "0": "cuda", "npu": "npu", "cpu": "cpu", "auto": "auto", "": "auto"}
    wanted = aliases.get(requested, requested)
    if wanted in {"", "auto"}:
        return selected
    if wanted == "cuda" and not detected.get("cuda_available"):
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if wanted == "npu" and not detected.get("npu_available"):
        raise RuntimeError("NPU was explicitly requested but is unavailable")
    return wanted


def find_checkpoint(run_dir: Path) -> str:
    preferred_names = ("best_stg2.pth", "best_stg1.pth", "last.pth")
    for name in preferred_names:
        matches = sorted(run_dir.rglob(name))
        if matches:
            return str(matches[-1])
    matches = sorted(run_dir.rglob("*.pth"))
    return str(matches[-1]) if matches else ""


def bool_from_any(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def export_onnx_after_training(
    *,
    prefix: list[str],
    deim_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    logs_dir: Path,
    training: dict[str, Any],
) -> dict[str, Any]:
    output_path = checkpoint_path.with_suffix(".onnx")
    opset = int(training.get("onnx_opset") or 17)
    check = bool_from_any(training.get("onnx_check"), True)
    simplify = bool_from_any(training.get("onnx_simplify"), True)
    official_exporter = deim_root / "tools" / "deployment" / "export_onnx.py"
    if not official_exporter.is_file():
        raise FileNotFoundError(f"DEIMv2 official ONNX exporter not found: {official_exporter}")
    cmd = prefix + [
        str(official_exporter),
        "-c",
        str(config_path),
        "-r",
        str(checkpoint_path),
        "--opset",
        str(opset),
    ]
    if check:
        cmd.append("--check")
    if simplify:
        cmd.append("--simplify")
    exporter = str(official_exporter)

    log_path = logs_dir / "export_onnx.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(deim_root), stdout=log, stderr=subprocess.STDOUT, text=True)
    result = {
        "enabled": True,
        "status": "completed" if proc.returncode == 0 and output_path.is_file() else "failed",
        "returncode": proc.returncode,
        "exporter": exporter,
        "command": cmd,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path) if output_path.is_file() else "",
        "log": str(log_path),
        "opset": opset,
        "check": check,
        "simplify": simplify,
    }
    if result["status"] != "completed":
        raise RuntimeError(f"DEIMv2 ONNX export failed. See {log_path}")
    return result


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_deimv2_log_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    ap_pattern = re.compile(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=([0-9.:]+)\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*(\d+)\s*\]\s*=\s*([-+0-9.]+)"
    )
    ar_pattern = re.compile(
        r"Average Recall\s+\(AR\)\s+@\[\s*IoU=([0-9.:]+)\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*(\d+)\s*\]\s*=\s*([-+0-9.]+)"
    )
    metrics: dict[str, Any] = {}
    metric_lines: list[str] = []
    for line in text.splitlines():
        if "Average Precision" in line or "Average Recall" in line or "best_stat:" in line:
            metric_lines.append(line.strip())
        ap_match = ap_pattern.search(line)
        if ap_match:
            iou, max_dets, raw_value = ap_match.groups()
            value = _parse_float(raw_value)
            if value is None:
                continue
            if iou == "0.50:0.95" and max_dets == "100":
                metrics["mAP50_95"] = value
            elif iou == "0.50" and max_dets == "100":
                metrics["mAP50"] = value
            elif iou == "0.75" and max_dets == "100":
                metrics["mAP75"] = value
            continue
        ar_match = ar_pattern.search(line)
        if ar_match:
            iou, max_dets, raw_value = ar_match.groups()
            value = _parse_float(raw_value)
            if value is None or iou != "0.50:0.95":
                continue
            if max_dets == "1":
                metrics["AR1"] = value
            elif max_dets == "10":
                metrics["AR10"] = value
            elif max_dets == "100":
                metrics["AR100"] = value
            continue
        if "best_stat:" in line:
            raw_best = line.split("best_stat:", 1)[1].strip()
            try:
                best_stat = ast.literal_eval(raw_best)
            except (SyntaxError, ValueError):
                best_stat = {}
            if isinstance(best_stat, dict):
                metrics["best_epoch"] = best_stat.get("epoch")
                best_bbox = best_stat.get("coco_eval_bbox")
                if best_bbox is not None:
                    metrics["best_coco_eval_bbox"] = best_bbox
                    metrics["fitness"] = best_bbox
    if not metrics:
        return {}
    results_dict: dict[str, Any] = {}
    if "mAP50_95" in metrics:
        results_dict["metrics/mAP50-95(B)"] = metrics["mAP50_95"]
    if "mAP50" in metrics:
        results_dict["metrics/mAP50(B)"] = metrics["mAP50"]
    if "AR100" in metrics:
        results_dict["metrics/recall(B)"] = metrics["AR100"]
    if "fitness" in metrics:
        results_dict["fitness"] = metrics["fitness"]
    return {
        "metrics": metrics,
        "results_dict": results_dict,
        "eval_results": "\n".join(metric_lines[-16:]),
    }


YOLO_COMPAT_RESULTS_COLUMNS = [
    "epoch",
    "time",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
    "metrics/mAP75(B)",
    "metrics/AR1(B)",
    "metrics/AR10(B)",
    "metrics/AR100(B)",
    "fitness",
    "train/loss",
    "train/giou_loss",
    "train/fgl_loss",
    "deimv2/best_coco_eval_bbox",
]


def _coco_eval_bbox_metrics(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        return {}
    result: dict[str, Any] = {}
    keys = {
        0: "mAP50_95",
        1: "mAP50",
        2: "mAP75",
        6: "AR1",
        7: "AR10",
        8: "AR100",
    }
    for index, key in keys.items():
        if index >= len(values):
            continue
        try:
            result[key] = float(values[index])
        except (TypeError, ValueError):
            continue
    return result


def _result_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _yolo_compat_csv_row_from_deimv2_json(record: dict[str, Any]) -> dict[str, Any]:
    metrics = _coco_eval_bbox_metrics(record.get("test_coco_eval_bbox"))
    raw_epoch = record.get("epoch")
    try:
        epoch = int(raw_epoch) + 1
    except (TypeError, ValueError):
        epoch = raw_epoch
    lr = record.get("train_lr")
    row = {
        "epoch": epoch,
        "time": "",
        "train/box_loss": record.get("train_loss_bbox"),
        "train/cls_loss": record.get("train_loss_mal"),
        "train/dfl_loss": record.get("train_loss_fgl"),
        "metrics/precision(B)": None,
        "metrics/recall(B)": metrics.get("AR100"),
        "metrics/mAP50(B)": metrics.get("mAP50"),
        "metrics/mAP50-95(B)": metrics.get("mAP50_95"),
        "val/box_loss": "",
        "val/cls_loss": "",
        "val/dfl_loss": "",
        "lr/pg0": lr,
        "lr/pg1": lr,
        "lr/pg2": lr,
        "metrics/mAP75(B)": metrics.get("mAP75"),
        "metrics/AR1(B)": metrics.get("AR1"),
        "metrics/AR10(B)": metrics.get("AR10"),
        "metrics/AR100(B)": metrics.get("AR100"),
        "fitness": metrics.get("mAP50_95"),
        "train/loss": record.get("train_loss"),
        "train/giou_loss": record.get("train_loss_giou"),
        "train/fgl_loss": record.get("train_loss_fgl"),
        "deimv2/best_coco_eval_bbox": "",
    }
    # 保留 DEIMv2 原生日志字段。前端如果自行渲染 CSV，可以看到完整的
    # train_loss_*、aux/dn/pre 等专属 loss；与 YOLO 兼容列重名时加前缀保留。
    for key, value in record.items():
        native_key = key if key not in row else f"deimv2/{key}"
        row[native_key] = value
    return row


def parse_deimv2_json_log_rows(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("epoch") is not None:
            rows.append(_yolo_compat_csv_row_from_deimv2_json(record))
    return rows


def fallback_yolo_compat_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    best_epoch = metrics.get("best_epoch")
    try:
        epoch = int(best_epoch) + 1
    except (TypeError, ValueError):
        epoch = best_epoch if best_epoch is not None else ""
    return {
        "epoch": epoch,
        "time": "",
        "train/box_loss": "",
        "train/cls_loss": "",
        "train/dfl_loss": "",
        "metrics/precision(B)": None,
        "metrics/recall(B)": metrics.get("AR100"),
        "metrics/mAP50(B)": metrics.get("mAP50"),
        "metrics/mAP50-95(B)": metrics.get("mAP50_95"),
        "val/box_loss": "",
        "val/cls_loss": "",
        "val/dfl_loss": "",
        "lr/pg0": "",
        "lr/pg1": "",
        "lr/pg2": "",
        "metrics/mAP75(B)": metrics.get("mAP75"),
        "metrics/AR1(B)": metrics.get("AR1"),
        "metrics/AR10(B)": metrics.get("AR10"),
        "metrics/AR100(B)": metrics.get("AR100"),
        "fitness": metrics.get("fitness") if metrics.get("fitness") is not None else metrics.get("best_coco_eval_bbox"),
        "train/loss": "",
        "train/giou_loss": "",
        "train/fgl_loss": "",
        "deimv2/best_coco_eval_bbox": metrics.get("best_coco_eval_bbox"),
    }


def _results_csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames = list(YOLO_COMPAT_RESULTS_COLUMNS)
    seen = set(fieldnames)
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    return fieldnames


def write_yolo_compatible_results_csv(summary: dict[str, Any], work_dir: Path, run_dir: Path) -> dict[str, str]:
    rows = parse_deimv2_json_log_rows(run_dir / "train" / "log.txt")
    if not rows:
        rows = [fallback_yolo_compat_csv_row(summary)]
    fieldnames = _results_csv_fieldnames(rows)
    output_paths = {
        "results_csv": work_dir / "results.csv",
        "train_results_csv": run_dir / "train" / "results.csv",
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _result_csv_value(row.get(key)) for key in fieldnames})
    return {key: str(path) for key, path in output_paths.items()}


def summarize_coco_split(dataset_root: Path, split: str) -> dict[str, Any]:
    ann_path = dataset_root / "annotations" / f"instances_{split}.json"
    if not ann_path.is_file():
        return {"split": split, "images": 0, "instances": 0, "source_counts": {}}
    payload = json.loads(ann_path.read_text(encoding="utf-8-sig"))
    images = payload.get("images") if isinstance(payload, dict) else []
    annotations = payload.get("annotations") if isinstance(payload, dict) else []
    source_counts: dict[str, int] = {}
    if isinstance(images, list):
        for image in images:
            if not isinstance(image, dict):
                continue
            source = str(image.get("source") or ("synthetic" if image.get("is_synthetic") else "real"))
            source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "split": split,
        "images": len(images) if isinstance(images, list) else 0,
        "instances": len(annotations) if isinstance(annotations, list) else 0,
        "source_counts": source_counts,
    }


def build_yolo_compatible_run_summary(summary: dict[str, Any], dataset_root: Path, work_dir: Path, run_dir: Path) -> dict[str, Any]:
    categories = read_categories(dataset_root)
    class_names = [str(cat.get("name") or "") for cat in categories if str(cat.get("name") or "").strip()]
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    yolo_results_dict = {
        "metrics/precision(B)": None,
        "metrics/recall(B)": metrics.get("AR100"),
        "metrics/mAP50(B)": metrics.get("mAP50"),
        "metrics/mAP50-95(B)": metrics.get("mAP50_95"),
        "fitness": metrics.get("fitness") if metrics.get("fitness") is not None else metrics.get("best_coco_eval_bbox"),
    }
    yolo_results_dict = {key: value for key, value in yolo_results_dict.items() if value is not None or key == "metrics/precision(B)"}
    yolo_metrics = dict(metrics)
    yolo_metrics.setdefault("precision", None)
    if "recall" not in yolo_metrics and metrics.get("AR100") is not None:
        yolo_metrics["recall"] = metrics.get("AR100")

    split_counts: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_summary = summarize_coco_split(dataset_root, split)
        split_counts[split] = int(split_summary.get("images") or 0)
        source_counts[split] = split_summary.get("source_counts") if isinstance(split_summary.get("source_counts"), dict) else {}
    eval_split = "test" if split_counts.get("test") else "val"
    test_set_summary = summarize_coco_split(dataset_root, eval_split)

    # 前端沿用 YOLO run_summary.json 解析，因此这里输出兼容字段；
    # DEIMv2 原始字段仍放在 training_summary.json 和 deimv2_summary 中。
    return {
        "config_path": str(summary.get("config") or ""),
        "conda_env_name": "",
        "accelerator": str(summary.get("selected") or ""),
        "device": str(summary.get("selected") or ""),
        "device_selection": summary.get("device_selection"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "npu_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "dataset_yaml": str(summary.get("dataset_config") or ""),
        "dataset_root": str(dataset_root),
        "task": "detect",
        "requested_model": str(summary.get("model_variant") or ""),
        "model": str(summary.get("model_variant") or ""),
        "model_source": str(summary.get("model_source") or "deimv2"),
        "strict_model": False,
        "model_sha256": str(summary.get("model_sha256") or ""),
        "model_original_name": str(summary.get("model_original_name") or ""),
        "model_id": str(summary.get("model_id") or ""),
        "model_fallback_used": False,
        "model_load_error": "",
        "num_images": sum(split_counts.values()),
        "num_categories": len(class_names),
        "class_names": class_names,
        "split_counts": split_counts,
        "source_counts": source_counts,
        "test_set_summary": test_set_summary,
        "run_root": str(work_dir),
        "runs_dir": str(run_dir),
        "train_save_dir": str(run_dir / "train"),
        "results_csv": str(summary.get("results_csv") or ""),
        "train_results_csv": str(summary.get("train_results_csv") or ""),
        "best_pt": str(summary.get("best_checkpoint") or ""),
        "best_checkpoint": str(summary.get("best_checkpoint") or ""),
        "last_checkpoint": str(summary.get("last_checkpoint") or ""),
        "onnx_model": str(summary.get("onnx_path") or ""),
        "onnx_path": str(summary.get("onnx_path") or ""),
        "onnx_export": summary.get("onnx_export") or {"enabled": False},
        "eval_results": str(summary.get("eval_results") or ""),
        "results_dict": yolo_results_dict,
        "metrics": yolo_metrics,
        "per_class_metrics": [
            {
                "class_name": class_name,
                "precision": None,
                "recall": None,
                "mAP50": None,
                "mAP50-95": None,
                "analysis": "DEIMv2 当前日志未输出类别级 Precision/Recall/mAP，前端兼容字段保留为空。",
            }
            for class_name in class_names
        ],
        "class_performance_analysis": {
            "summary": "DEIMv2 当前解析的是 COCO 全局 AP/AR 指标，未生成类别级 YOLO 指标。",
            "classes": [],
        },
        "eval_error": "",
        "status": summary.get("status") or "completed",
        "training_backend": "deimv2",
        "deimv2_summary": summary,
    }


def run_training(spec: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    deim_root = resolve_deimv2_root(spec)
    dataset_root = Path(str(spec["dataset_root"])).resolve()
    work_dir = Path(str(spec["work_dir"])).resolve()
    training = spec.get("training") if isinstance(spec.get("training"), dict) else {}
    prefix = python_prefix(training)
    detected = detect_target_hardware(prefix)
    # 先选设备再生成检测器配置，因为 CUDA/NPU/CPU 的 batch、workers
    # 等默认值不同。
    device_reservation = reserve_training_device(
        requested=training.get("device"),
        backend="deimv2",
        model_variant=training.get("model_variant"),
        batch=training.get("batch"),
        img_size=training.get("img_size") or training.get("imgsz"),
        project_root=PROJECT_ROOT,
        min_free_memory_mb=training.get("min_free_memory_mb") or training.get("required_free_memory_mb"),
        max_gpu_utilization=training.get("max_gpu_utilization"),
    )
    hardware = device_reservation.accelerator
    detected["device_selection"] = device_reservation.to_dict()
    detected["selected"] = hardware
    template = choose_template(deim_root, training)
    backbone = resolve_checkpoint(
        deim_root,
        str(training.get("backbone_checkpoint") or ""),
        DEFAULT_BACKBONE_CHECKPOINT,
        env_name="DEIMV2_BACKBONE_CHECKPOINT",
    )
    assert backbone is not None
    tuning = resolve_tuning_checkpoint(deim_root, training)
    epochs = int(training.get("epochs") or 10)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha1(str(work_dir).encode("utf-8")).hexdigest()[:12]
    configs_dir = work_dir / "configs" if work_dir.drive == deim_root.drive else deim_root / "outputs" / "codex-generated-configs" / config_hash
    run_dir = work_dir / "runs"
    logs_dir = work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stages: list[dict[str, Any]] = []

    for stage, count in [("train", epochs)]:
        output_dir = run_dir / stage
        config = write_runtime_configs(
            configs_dir=configs_dir,
            template=template,
            dataset_root=dataset_root,
            output_dir=output_dir,
            training=training,
            hardware=hardware,
            epochs=count,
            backbone_checkpoint=backbone,
            stage=stage,
        )
        cmd = training_command(prefix, deim_root, config, hardware, tuning, device_reservation.runtime_device)
        stage_result: dict[str, Any] = {
            "stage": stage,
            "epochs": count,
            "config": str(config),
            "output_dir": str(output_dir),
            "command": cmd,
            "device_selection": device_reservation.to_dict(),
        }
        stages.append(stage_result)
        if dry_run:
            stage_result["returncode"] = 0
            continue
        log_path = logs_dir / f"{stage}.log"
        env = os.environ.copy()
        # CUDA 下只把选中的物理 GPU 暴露给 DEIMv2。
        # 官方 train.py 仍可按默认 cuda:0 逻辑运行。
        env.update(device_reservation.env)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(deim_root), stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
        stage_result["returncode"] = proc.returncode
        stage_result["log"] = str(log_path)
        if proc.returncode != 0:
            raise RuntimeError(f"DEIMv2 {stage} failed. See {log_path}")

    best_checkpoint = find_checkpoint(run_dir)
    onnx_export: dict[str, Any] = {"enabled": bool_from_any(training.get("export_onnx"), False), "status": "skipped"}
    if onnx_export["enabled"] and not dry_run:
        if not best_checkpoint:
            raise RuntimeError("DEIMv2 ONNX export requested but no checkpoint was generated.")
        onnx_export = export_onnx_after_training(
            prefix=prefix,
            deim_root=deim_root,
            config_path=configs_dir / "train.yml",
            checkpoint_path=Path(best_checkpoint).resolve(),
            logs_dir=logs_dir,
            training=training,
        )
    metric_payload = parse_deimv2_log_metrics(logs_dir / "train.log")
    summary = {
        "status": "completed",
        "training_backend": "deimv2",
        "model_variant": normalize_model_variant(training.get("model_variant")),
        "deimv2_root": str(deim_root),
        "dataset_root": str(dataset_root),
        "hardware": detected,
        "selected": hardware,
        "device_selection": device_reservation.to_dict(),
        "template": str(template),
        "backbone_checkpoint": str(backbone),
        "tuning_checkpoint": str(tuning) if tuning else "",
        "disable_tuning_checkpoint": bool(training.get("disable_tuning_checkpoint")),
        "model_source": str(training.get("model_source") or ""),
        "model_id": str(training.get("model_id") or ""),
        "model_sha256": str(training.get("model_sha256") or ""),
        "model_original_name": str(training.get("model_original_name") or ""),
        "model_extension": str(training.get("model_extension") or ""),
        "deimv2_upload_model_usage": str(training.get("deimv2_upload_model_usage") or ""),
        "config": str(configs_dir / "train.yml"),
        "dataset_config": str(configs_dir / "dataset.yml"),
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": str((run_dir / "train" / "last.pth")) if (run_dir / "train" / "last.pth").is_file() else "",
        "onnx_export": onnx_export,
        "onnx_path": str(onnx_export.get("onnx_path") or ""),
        "stages": stages,
        "dry_run": dry_run,
    }
    summary.update(metric_payload)
    summary.update(write_yolo_compatible_results_csv(summary, work_dir, run_dir))
    (work_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    compatible_summary = build_yolo_compatible_run_summary(summary, dataset_root, work_dir, run_dir)
    (work_dir / "run_summary.json").write_text(json.dumps(compatible_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    device_reservation.release()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DEIMv2 runtime YAML and train.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_training(load_spec(Path(args.input).resolve()), dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
