import argparse
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


def _resolve_model_name(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_absolute():
        return str(model_path)
    if model_path.suffix.lower() != ".pt":
        return model_name
    package_candidate = (Path(__file__).resolve().parent.parent / model_path).resolve()
    if package_candidate.exists() and package_candidate.is_file():
        return str(package_candidate)
    return model_name


def _load_yolo_model(yolo_cls: object, model_name: str, task: str):
    resolved_model = _resolve_model_name(model_name)
    try:
        return yolo_cls(resolved_model, task=task), resolved_model
    except RuntimeError as exc:
        message = str(exc)
        if "PytorchStreamReader failed reading zip archive" not in message and "failed finding central directory" not in message:
            raise
        candidate = Path(resolved_model)
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pt":
            log_warn(f"Detected broken local weight file, deleting and retrying download: {candidate}")
            candidate.unlink()
            return yolo_cls(model_name, task=task), model_name
        raise


def _check_conda_runtime(cfg: Dict) -> str:
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
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Please install ultralytics in the current runtime environment") from exc

    model_name = str(training_cfg.get("model", "yolo11n.pt"))
    epochs = int(training_cfg.get("epochs", 100))
    imgsz = int(training_cfg.get("imgsz", 640))
    batch = int(training_cfg.get("batch", 16))
    _has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    _default_device = "0" if _has_cuda else "cpu"
    device = str(training_cfg.get("device", _default_device))
    if device != "cpu" and not _has_cuda:
        log_warn(f"Requested device='{device}' but CUDA is unavailable; falling back to CPU")
        device = "cpu"
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

    log_info(f"Training runtime={conda_env_name}, data={dataset_yaml}, device={device}, CUDA={_has_cuda}")
    model, model_name = _load_yolo_model(YOLO, model_name, task)
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
        "dataset_yaml": str(dataset_yaml),
        "task": task,
        "model": model_name,
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
