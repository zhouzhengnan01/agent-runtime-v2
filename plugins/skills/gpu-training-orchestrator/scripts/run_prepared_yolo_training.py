import argparse
import ast
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_log_file = None
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "app" / "core").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.hardware.device_selector import reserve_training_device


class _TeeWriter:
    def __init__(self, original, log_path: Path):
        self._original = original
        self._log_fh = log_path.open("a", encoding="utf-8")

    def write(self, data):
        self._original.write(data)
        if self._log_fh and data:
            self._log_fh.write(data)
            self._log_fh.flush()
        return len(data) if data else 0

    def flush(self):
        self._original.flush()
        if self._log_fh:
            self._log_fh.flush()

    def fileno(self):
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_tee(log_path: Path) -> None:
    global _log_file
    _log_file = log_path
    sys.stdout = _TeeWriter(sys.stdout, log_path)
    sys.stderr = _TeeWriter(sys.stderr, log_path)


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _safe_results_dict(results: object) -> Dict[str, object]:
    raw = getattr(results, "results_dict", None)
    if not isinstance(raw, dict):
        return {}
    safe: Dict[str, object] = {}
    for key, value in raw.items():
        try:
            safe[str(key)] = float(value)
        except (TypeError, ValueError):
            safe[str(key)] = str(value)
    return safe


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float_list(value: object) -> List[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError:
            return []
    result: List[float] = []
    for item in values:
        number = _safe_float(item)
        if number is not None:
            result.append(number)
    return result


def _safe_int_list(value: object) -> List[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        try:
            values = list(value)  # type: ignore[arg-type]
        except TypeError:
            return []
    result: List[int] = []
    for item in values:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _metric_value_for_class(values: List[float], class_index: int, class_count: int, ap_class_indices: List[int]) -> float | None:
    if not values:
        return None
    if len(values) == class_count and 0 <= class_index < len(values):
        return values[class_index]
    if ap_class_indices and len(ap_class_indices) == len(values):
        for pos, metric_class_index in enumerate(ap_class_indices):
            if metric_class_index == class_index:
                return values[pos]
        return None
    if 0 <= class_index < len(values):
        return values[class_index]
    return None


def _class_metric_analysis(class_name: str, precision: float | None, recall: float | None, map50: float | None, map50_95: float | None) -> str:
    if precision is None and recall is None and map50 is None and map50_95 is None:
        return "未获取到该类别的评估指标，建议检查 test split 是否包含该类别实例或当前 Ultralytics 版本是否返回类别级指标。"
    if recall == 0:
        return f"{class_name} 在当前测试集上召回率为 0，模型没有成功检出该类别；若精确率较高，通常表示误报少但漏检严重。"
    if recall is not None and recall >= 0.95 and map50 is not None and map50 >= 0.9:
        return f"{class_name} 表现优异，召回率和 mAP50 都很高，当前测试集上识别稳定。"
    if map50 is not None and map50 < 0.3:
        return f"{class_name} 的 mAP50 较低，当前测试集上的定位或识别效果较弱，建议补充更多该类别样本并检查标注质量。"
    if recall is not None and recall < 0.5:
        return f"{class_name} 召回率偏低，存在明显漏检风险，建议增加该类别训练样本、困难样本和小目标样本。"
    if precision is not None and precision < 0.5:
        return f"{class_name} 精确率偏低，误报较多，建议加入更多负样本或相似干扰类别。"
    return f"{class_name} 指标处于可用但仍需关注的水平，建议结合更多测试图像继续验证泛化表现。"


def _extract_per_class_metrics(results: object, class_names: List[str], test_set_summary: Dict[str, Any]) -> List[Dict[str, object]]:
    if results is None or not class_names:
        return []
    box = getattr(results, "box", None)
    if box is None:
        return []
    ap_class_indices = _safe_int_list(
        getattr(box, "ap_class_index", None)
        if getattr(box, "ap_class_index", None) is not None
        else getattr(box, "ap_class", None)
    )
    metric_values = {
        "precision": _safe_float_list(getattr(box, "p", None)),
        "recall": _safe_float_list(getattr(box, "r", None)),
        "mAP50": _safe_float_list(getattr(box, "ap50", None)),
        "mAP50-95": _safe_float_list(getattr(box, "maps", None) if getattr(box, "maps", None) is not None else getattr(box, "ap", None)),
    }
    instance_counts = test_set_summary.get("instances_per_class", {})
    if not isinstance(instance_counts, dict):
        instance_counts = {}

    metrics: List[Dict[str, object]] = []
    class_count = len(class_names)
    for class_index, class_name in enumerate(class_names):
        precision = _metric_value_for_class(metric_values["precision"], class_index, class_count, ap_class_indices)
        recall = _metric_value_for_class(metric_values["recall"], class_index, class_count, ap_class_indices)
        map50 = _metric_value_for_class(metric_values["mAP50"], class_index, class_count, ap_class_indices)
        map50_95 = _metric_value_for_class(metric_values["mAP50-95"], class_index, class_count, ap_class_indices)
        metrics.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "instances": int(instance_counts.get(class_name, 0) or 0),
                "precision": precision,
                "recall": recall,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "analysis": _class_metric_analysis(class_name, precision, recall, map50, map50_95),
            }
        )
    return metrics


def _build_class_performance_analysis(test_set_summary: Dict[str, Any], per_class_metrics: List[Dict[str, object]]) -> Dict[str, object]:
    if not per_class_metrics:
        return {
            "summary": "未获取到类别级指标；请确认 test split 中存在标注实例，且当前 Ultralytics 版本返回 per-class metrics。",
            "classes": [],
        }
    image_count = int(test_set_summary.get("images", 0) or 0)
    instance_count = int(test_set_summary.get("instances", 0) or 0)
    return {
        "summary": f"从测试集（{image_count} 张图像，{instance_count} 个实例）的细粒度结果来看，各类别表现如下。",
        "classes": [
            {
                "class_name": item.get("class_name"),
                "precision": item.get("precision"),
                "recall": item.get("recall"),
                "mAP50": item.get("mAP50"),
                "mAP50-95": item.get("mAP50-95"),
                "analysis": item.get("analysis"),
            }
            for item in per_class_metrics
        ],
    }


def _parse_simple_dataset_yaml(dataset_yaml: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for raw_line in dataset_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "names":
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    data[key] = [str(item) for item in parsed]
                elif isinstance(parsed, dict):
                    data[key] = [str(parsed[item]) for item in sorted(parsed)]
            except Exception:
                data[key] = []
        else:
            data[key] = value
    return data


def _class_names_from_dataset_yaml(dataset_yaml: Path) -> List[str]:
    data = _parse_simple_dataset_yaml(dataset_yaml)
    names = data.get("names")
    return [str(item) for item in names] if isinstance(names, list) else []


def _summarize_dataset_yaml_split(dataset_yaml: Path, split_name: str, class_names: List[str]) -> Dict[str, object]:
    data = _parse_simple_dataset_yaml(dataset_yaml)
    root = Path(str(data.get("path") or dataset_yaml.parent)).expanduser()
    split_value = data.get(split_name)
    images_dir = Path(str(split_value)).expanduser() if split_value else root / "images" / split_name
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    if "images" in images_dir.parts:
        parts = list(images_dir.parts)
        parts[parts.index("images")] = "labels"
        labels_dir = Path(*parts)
    else:
        labels_dir = root / "labels" / split_name

    image_count = len([path for path in images_dir.glob("*") if path.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0
    instances_per_class = {name: 0 for name in class_names}
    instance_count = 0
    if labels_dir.exists():
        for label_path in labels_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_index = int(float(parts[0]))
                except (TypeError, ValueError):
                    continue
                if 0 <= class_index < len(class_names):
                    instances_per_class[class_names[class_index]] += 1
                    instance_count += 1
    return {
        "split": split_name,
        "images": image_count,
        "instances": instance_count,
        "instances_per_class": instances_per_class,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_model_name(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_absolute():
        return str(model_path)
    if model_path.suffix.lower() != ".pt":
        return model_name
    project_candidate = (_project_root() / "models" / model_path.name).resolve()
    if project_candidate.exists() and project_candidate.is_file():
        return str(project_candidate)
    package_candidate = (Path(__file__).resolve().parent.parent / model_path).resolve()
    if package_candidate.exists() and package_candidate.is_file():
        return str(package_candidate)
    return model_name


def _load_yolo_model(
    yolo_cls: object,
    model_name: str,
    task: str,
    *,
    strict_model: bool = False,
    expected_sha256: str = "",
):
    resolved_model = _resolve_model_name(model_name)
    if strict_model:
        candidate = Path(resolved_model).expanduser()
        if not candidate.is_absolute():
            raise RuntimeError("User-uploaded model must use an absolute path.")
        if not candidate.is_file() or candidate.suffix.lower() != ".pt":
            raise RuntimeError(f"User-uploaded YOLO model is missing or invalid: {candidate}")
        expected = expected_sha256.strip().lower()
        if expected and _file_sha256(candidate) != expected:
            raise RuntimeError(f"User-uploaded YOLO model SHA256 mismatch: {candidate}")
        # 用户上传模型走严格加载：如果文件无法加载，要明确失败，
        # 不能静默切换到内置模型。
        try:
            return yolo_cls(str(candidate), task=task), str(candidate), False, ""
        except Exception as exc:
            raise RuntimeError(f"Failed to load user-uploaded YOLO model '{candidate}': {exc}") from exc
    try:
        return yolo_cls(resolved_model, task=task), resolved_model, False, ""
    except RuntimeError as exc:
        message = str(exc)
        if "PytorchStreamReader failed reading zip archive" not in message and "failed finding central directory" not in message:
            return _load_builtin_fallback_model(yolo_cls, model_name, resolved_model, task, exc)
        candidate = Path(resolved_model)
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pt":
            if _is_builtin_fallback_candidate(candidate):
                log_warn(f"Detected broken bundled weight file, trying another fallback without deleting: {candidate}")
                return _load_builtin_fallback_model(yolo_cls, model_name, resolved_model, task, exc)
            log_warn(f"Detected broken local weight file, deleting and retrying download: {candidate}")
            candidate.unlink()
            try:
                return yolo_cls(model_name, task=task), model_name, False, ""
            except Exception as retry_exc:
                return _load_builtin_fallback_model(yolo_cls, model_name, resolved_model, task, retry_exc)
        return _load_builtin_fallback_model(yolo_cls, model_name, resolved_model, task, exc)
    except Exception as exc:
        return _load_builtin_fallback_model(yolo_cls, model_name, resolved_model, task, exc)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_builtin_fallback_model(yolo_cls: object, requested_model: str, resolved_model: str, task: str, original_exc: Exception):
    fallback = _find_builtin_fallback_model(resolved_model)
    if fallback is None:
        raise original_exc
    log_warn(
        "Failed to load requested YOLO model "
        f"'{requested_model}' ({original_exc}); using bundled fallback model: {fallback}"
    )
    try:
        return yolo_cls(str(fallback), task=task), str(fallback), True, str(original_exc)
    except Exception as fallback_exc:
        raise RuntimeError(
            f"Failed to load requested YOLO model '{requested_model}' and fallback model '{fallback}': {fallback_exc}"
        ) from original_exc


def _find_builtin_fallback_model(resolved_model: str = "") -> Path | None:
    resolved_requested = _safe_resolved_path(resolved_model)
    for candidate in _builtin_fallback_model_candidates():
        if not candidate.is_file():
            continue
        if resolved_requested is not None and _safe_resolved_path(candidate) == resolved_requested:
            continue
        return candidate.resolve()
    return None


def _builtin_fallback_model_candidates() -> list[Path]:
    project_models = _project_root() / "models"
    package_root = Path(__file__).resolve().parent.parent
    return [
        project_models / "yolov11n.pt",
        project_models / "yolo11n.pt",
        package_root / "yolov11n.pt",
        package_root / "yolo11n.pt",
    ]


def _is_builtin_fallback_candidate(path: Path) -> bool:
    resolved = _safe_resolved_path(path)
    if resolved is None:
        return False
    return any(_safe_resolved_path(candidate) == resolved for candidate in _builtin_fallback_model_candidates())


def _safe_resolved_path(value: object) -> Path | None:
    try:
        return Path(str(value)).expanduser().resolve()
    except Exception:
        return None


def _check_conda_runtime(cfg: Dict) -> str:
    if _preferred_accelerator() == "npu":
        return "current-npu"
    runtime_cfg = cfg.get("runtime", {})
    conda_env_name = str(runtime_cfg.get("conda_env_name", "")).strip()
    enforce = bool(runtime_cfg.get("enforce_conda_env", True))
    if not conda_env_name:
        return "current"
    current_env = str(os.environ.get("CONDA_DEFAULT_ENV", "")).strip()
    if current_env != conda_env_name:
        msg = f"Current conda env is '{current_env or 'N/A'}', but input requires '{conda_env_name}'."
        if enforce:
            raise EnvironmentError(msg)
        log_warn(msg)
    return conda_env_name


def _is_npu_runtime_available() -> bool:
    if _truthy_env("JETLINKS_FORCE_CURRENT_PYTHON_ON_NPU"):
        return True
    if not _has_ascend_runtime_hint():
        return False
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return _torch_npu_available(torch)


def _preferred_accelerator(torch_module: object | None = None) -> str:
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except Exception:
            return "cpu"
    try:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and bool(cuda.is_available()) and int(cuda.device_count()) > 0:
            return "cuda"
    except Exception:
        pass
    if _is_npu_runtime_available():
        return "npu"
    return "cpu"


def _has_ascend_runtime_hint() -> bool:
    env_names = (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
    )
    if any(str(os.environ.get(name) or "").strip() for name in env_names):
        return True
    return any(Path(path).exists() for path in ("/usr/local/Ascend", "/dev/davinci_manager"))


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _torch_npu_available(torch_module: object) -> bool:
    npu = getattr(torch_module, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


def _torch_npu_device_count(torch_module: object) -> int:
    npu = getattr(torch_module, "npu", None)
    if npu is None:
        return 0
    try:
        return int(npu.device_count())
    except Exception:
        return 0


def _normalize_device_for_runtime(raw_device: object, *, has_npu: bool, has_cuda: bool) -> str:
    requested = str(raw_device or "").strip()
    if has_cuda:
        if requested.isdigit():
            return requested
        if requested.lower().startswith("cuda:"):
            return requested
        return "0"
    if has_npu:
        if not requested or requested == "cpu":
            return "npu:0"
        if requested.lower().startswith("npu:"):
            return requested
        if requested.isdigit():
            visible = [item.strip() for item in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",") if item.strip()]
            if len(visible) == 1 and requested == visible[0]:
                return "npu:0"
            return f"npu:{requested}"
        return "npu:0"
    if requested and requested != "cpu":
        log_warn(f"Requested device='{requested}' but no CUDA/NPU accelerator is available; falling back to CPU")
    return "cpu"


def _configure_npu_runtime(torch_module: object) -> None:
    importlib.import_module("torch_npu")
    importlib.import_module("torch_npu.contrib.transfer_to_npu")
    npu = getattr(torch_module, "npu", None)
    if npu is None:
        raise RuntimeError("torch_npu is installed but torch.npu is unavailable")
    config = getattr(npu, "config", None)
    if config is not None:
        config.allow_internal_format = False
    set_compile_mode = getattr(npu, "set_compile_mode", None)
    if callable(set_compile_mode):
        set_compile_mode(jit_compile=False)


def run(config_path: Path) -> None:
    cfg = _load_json(config_path)
    conda_env_name = _check_conda_runtime(cfg)
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    output_cfg = cfg.get("output", {})

    dataset_yaml = Path(str(dataset_cfg.get("data_yaml") or dataset_cfg.get("dataset_yaml") or "")).expanduser()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Prepared dataset yaml not found: {dataset_yaml}")
    class_names = [str(item) for item in dataset_cfg.get("class_names", [])] if isinstance(dataset_cfg.get("class_names"), list) else []
    if not class_names:
        class_names = _class_names_from_dataset_yaml(dataset_yaml)
    test_set_summary = _summarize_dataset_yaml_split(dataset_yaml, "test", class_names)

    project_dir = Path(str(output_cfg.get("project_dir") or "")).expanduser()
    run_name = str(output_cfg.get("run_name", "exp"))
    if not str(project_dir):
        raise ValueError("output.project_dir must not be empty")
    run_root = project_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    task = str(training_cfg.get("task", "detect")).strip().lower()
    if task not in {"detect", "segment"}:
        raise ValueError(f"training.task only supports detect/segment, got: {task}")

    requested_model_name = str(training_cfg.get("model", "yolo11n.pt"))
    model_source = str(training_cfg.get("model_source") or "")
    strict_model = bool(training_cfg.get("strict_model", False))
    model_sha256 = str(training_cfg.get("model_sha256") or "")
    model_original_name = str(training_cfg.get("model_original_name") or "")
    model_id = str(training_cfg.get("model_id") or "")
    epochs = int(training_cfg.get("epochs", 100))
    imgsz = int(training_cfg.get("imgsz", 640))
    batch = int(training_cfg.get("batch", 16))
    # 在初始化重型训练运行时前先预约设备。CUDA 会通过 CUDA_VISIBLE_DEVICES
    # 将选中的物理卡映射为 Ultralytics 进程内的本地 device 0。
    device_reservation = reserve_training_device(
        requested=training_cfg.get("device"),
        backend="yolo",
        model_variant=requested_model_name,
        batch=batch,
        img_size=imgsz,
        project_root=PROJECT_ROOT,
        min_free_memory_mb=training_cfg.get("min_free_memory_mb") or training_cfg.get("required_free_memory_mb"),
        max_gpu_utilization=training_cfg.get("max_gpu_utilization"),
    )
    os.environ.update(device_reservation.env)

    import torch

    device = device_reservation.runtime_device
    accelerator = device_reservation.accelerator
    _has_cuda = accelerator == "cuda"
    _has_npu = accelerator == "npu"
    if accelerator == "npu":
        _configure_npu_runtime(torch)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Please install ultralytics in the current runtime environment") from exc
    workers = int(training_cfg.get("workers", 8))
    patience = int(training_cfg.get("patience", 50))

    train_overrides = {}
    val_overrides = {}
    if device == "cpu":
        # CPU 训练主要用于冒烟测试。降低线程和 worker 压力，
        # 避免开发机在小规模测试时卡死。
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        if batch > 4:
            log_warn(f"CPU mode batch={batch} is high; lowering to 4")
            batch = 4
        if workers != 0:
            log_warn(f"CPU mode workers={workers}; lowering to 0")
            workers = 0
        train_overrides.update({"amp": False, "cache": False, "deterministic": False, "plots": True})
        val_overrides.update({"plots": True})
    elif accelerator == "npu":
        # NPU 路径保持保守，因为 plotting 和多进程能力会随 torch_npu、
        # Ascend toolkit 版本不同而表现不一致。
        if workers != 0:
            log_warn(f"NPU mode workers={workers}; lowering to 0")
            workers = 0
        train_overrides.update({"amp": False, "cache": False, "plots": False})
        val_overrides.update({"plots": False})

    log_info(
        "Training runtime="
        f"{conda_env_name}, data={dataset_yaml}, device={device}, "
        f"accelerator={accelerator}, physical_device={device_reservation.physical_index}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}, "
        f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}, "
        f"NPU={_has_npu}, npu_count={_torch_npu_device_count(torch) if _has_npu else 0}"
    )
    model, model_name, model_fallback_used, model_load_error = _load_yolo_model(
        YOLO,
        requested_model_name,
        task,
        strict_model=strict_model,
        expected_sha256=model_sha256,
    )
    train_results = model.train(
        data=str(dataset_yaml),
        task=task,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        project=str(run_root),
        name="train",
        exist_ok=True,
        **train_overrides,
    )

    eval_results = None
    eval_error = ""
    try:
        eval_results = model.val(
            data=str(dataset_yaml),
            split="test",
            task=task,
            device=device,
            project=str(run_root),
            name="test_eval",
            exist_ok=True,
            **val_overrides,
        )
    except Exception as exc:
        eval_error = str(exc)
        log_warn(f"test split evaluation failed, training outputs are preserved: {eval_error}")

    eval_results_dict = _safe_results_dict(eval_results)
    per_class_metrics = _extract_per_class_metrics(eval_results, class_names, test_set_summary)
    class_performance_analysis = _build_class_performance_analysis(test_set_summary, per_class_metrics)
    summary = {
        "config_path": str(config_path),
        "conda_env_name": conda_env_name,
        "accelerator": accelerator,
        "device": device,
        "device_selection": device_reservation.to_dict(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "npu_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "dataset_yaml": str(dataset_yaml),
        "task": task,
        "requested_model": requested_model_name,
        "model": model_name,
        "model_source": model_source,
        "strict_model": strict_model,
        "model_sha256": model_sha256,
        "model_original_name": model_original_name,
        "model_id": model_id,
        "model_fallback_used": model_fallback_used,
        "model_load_error": model_load_error,
        "num_categories": len(class_names),
        "class_names": class_names,
        "test_set_summary": test_set_summary,
        "run_root": str(run_root),
        "runs_dir": str(run_root),
        "train_save_dir": str(getattr(train_results, "save_dir", "")),
        "eval_results": str(eval_results) if eval_results is not None else "",
        "results_dict": eval_results_dict,
        "per_class_metrics": per_class_metrics,
        "class_performance_analysis": class_performance_analysis,
        "eval_error": eval_error,
    }
    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    device_reservation.release()
    log_info(f"Training completed. Summary written to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO from a prepared data.yaml only.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()
    if args.log_file:
        log_path = Path(args.log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _install_tee(log_path)
        log_info(f"log_file: {log_path}")
    try:
        run(Path(args.input).resolve())
    finally:
        if _log_file is not None:
            sys.stdout.write("=== TRAINING_PROCESS_DONE ===\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
