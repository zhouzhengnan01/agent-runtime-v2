import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

_log_file = None


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

    project_dir = Path(str(output_cfg.get("project_dir") or "")).expanduser()
    run_name = str(output_cfg.get("run_name", "exp"))
    if not str(project_dir):
        raise ValueError("output.project_dir must not be empty")
    run_root = project_dir / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    task = str(training_cfg.get("task", "detect")).strip().lower()
    if task not in {"detect", "segment"}:
        raise ValueError(f"training.task only supports detect/segment, got: {task}")

    import torch

    requested_model_name = str(training_cfg.get("model", "yolo11n.pt"))
    model_source = str(training_cfg.get("model_source") or "")
    strict_model = bool(training_cfg.get("strict_model", False))
    model_sha256 = str(training_cfg.get("model_sha256") or "")
    model_original_name = str(training_cfg.get("model_original_name") or "")
    model_id = str(training_cfg.get("model_id") or "")
    epochs = int(training_cfg.get("epochs", 100))
    imgsz = int(training_cfg.get("imgsz", 640))
    batch = int(training_cfg.get("batch", 16))
    _has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    _has_npu = False if _has_cuda else _is_npu_runtime_available()
    device = _normalize_device_for_runtime(training_cfg.get("device"), has_npu=_has_npu, has_cuda=_has_cuda)
    accelerator = "cuda" if _has_cuda else ("npu" if _has_npu else "cpu")
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
        if workers != 0:
            log_warn(f"NPU mode workers={workers}; lowering to 0")
            workers = 0
        train_overrides.update({"amp": False, "cache": False, "plots": False})
        val_overrides.update({"plots": False})

    log_info(
        "Training runtime="
        f"{conda_env_name}, data={dataset_yaml}, device={device}, "
        f"CUDA={_has_cuda}, NPU={_has_npu}, npu_count={_torch_npu_device_count(torch) if _has_npu else 0}"
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

    summary = {
        "config_path": str(config_path),
        "conda_env_name": conda_env_name,
        "accelerator": accelerator,
        "device": device,
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
        "run_root": str(run_root),
        "runs_dir": str(run_root),
        "train_save_dir": str(getattr(train_results, "save_dir", "")),
        "eval_results": str(eval_results) if eval_results is not None else "",
        "results_dict": _safe_results_dict(eval_results),
        "eval_error": eval_error,
    }
    summary_path = run_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
