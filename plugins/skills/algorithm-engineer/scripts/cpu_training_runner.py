#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrainingRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingConfig:
    data_yaml: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    experiment_name: str
    timeout_seconds: int
    mock: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded CPU YOLO training job.")
    parser.add_argument("--request", required=True, help="Path to skill-run.v1 request JSON.")
    parser.add_argument("--outputs", required=True, help="Directory where artifacts must be written.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        request = _load_request(Path(args.request))
        config = _training_config(request)
        workspace_dir = Path(str(request.get("workspace_dir") or outputs_dir / "workspace")).resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        result = run_training(config, workspace_dir, outputs_dir)
    except Exception as exc:
        error = {
            "ok": False,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (outputs_dir / "training-summary.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
        (outputs_dir / "training-summary.md").write_text(_error_markdown(error), encoding="utf-8")
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **result}, ensure_ascii=False))
    return 0


def run_training(config: TrainingConfig, workspace_dir: Path, outputs_dir: Path) -> dict[str, Any]:
    run_dir = workspace_dir / "runs" / "detect" / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs_dir / "train.log"

    if config.mock:
        _write_mock_training_outputs(config, run_dir, log_path)
    else:
        data_yaml = Path(config.data_yaml).expanduser()
        if not data_yaml.is_file():
            raise TrainingRunnerError(
                "data_yaml is required for real CPU training. Provide a valid YOLO data.yaml path, "
                "or set mock=true only for pipeline smoke tests."
            )
        _require_module("ultralytics", "Install ultralytics in the sandbox environment before real CPU training.")
        _run_ultralytics_training(config, data_yaml, workspace_dir, log_path)

    artifacts = _collect_outputs(config, outputs_dir, run_dir)
    summary = _summary_payload(config, artifacts, run_dir)
    (outputs_dir / "training-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (outputs_dir / "training-summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _run_ultralytics_training(config: TrainingConfig, data_yaml: Path, workspace_dir: Path, log_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "ultralytics",
        "detect",
        "train",
        f"model={config.model}",
        f"data={data_yaml}",
        f"epochs={config.epochs}",
        f"imgsz={config.imgsz}",
        f"batch={config.batch}",
        "device=cpu",
        "workers=0",
        f"project={workspace_dir / 'runs'}",
        f"name={config.experiment_name}",
        "exist_ok=True",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=workspace_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=config.timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        raise TrainingRunnerError(f"YOLO CPU training failed with exit code {completed.returncode}. See train.log.")


def _write_mock_training_outputs(config: TrainingConfig, run_dir: Path, log_path: Path) -> None:
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mock": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "data_yaml": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
    }
    (weights_dir / "best.pt").write_bytes(b"mock best.pt for cpu-training-runner\n" + json.dumps(payload).encode("utf-8"))
    (weights_dir / "last.pt").write_bytes(b"mock last.pt for cpu-training-runner\n" + json.dumps(payload).encode("utf-8"))
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"])
        writer.writerow([1, 0.01, 0.001, 0.02, 0.03])
    (run_dir / "args.yaml").write_text(
        "\n".join(
            [
                f"model: {config.model}",
                f"data: {config.data_yaml}",
                f"epochs: {config.epochs}",
                f"imgsz: {config.imgsz}",
                f"batch: {config.batch}",
                "device: cpu",
                "mock: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.write_text("Mock CPU training completed. This is not a real model.\n", encoding="utf-8")


def _collect_outputs(config: TrainingConfig, outputs_dir: Path, run_dir: Path) -> dict[str, str]:
    mapping = {
        "best_pt": run_dir / "weights" / "best.pt",
        "last_pt": run_dir / "weights" / "last.pt",
        "results_csv": run_dir / "results.csv",
        "args_yaml": run_dir / "args.yaml",
        "train_log": outputs_dir / "train.log",
    }
    artifacts: dict[str, str] = {}
    for key, source in mapping.items():
        if not source.is_file():
            if key == "best_pt":
                raise TrainingRunnerError(f"Training completed without best.pt: expected {source}")
            continue
        target_name = {
            "best_pt": "best.pt",
            "last_pt": "last.pt",
            "results_csv": "results.csv",
            "args_yaml": "args.yaml",
            "train_log": "train.log",
        }[key]
        target = outputs_dir / target_name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        artifacts[key] = target.name
    artifacts["run_dir"] = str(run_dir)
    artifacts["experiment_name"] = config.experiment_name
    return artifacts


def _summary_payload(config: TrainingConfig, artifacts: dict[str, str], run_dir: Path) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "completed",
        "execution_mode": "cpu_mock" if config.mock else "cpu_training",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "data_yaml": config.data_yaml,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "device": "cpu",
        "mock": config.mock,
        "run_dir": str(run_dir),
        "artifacts": artifacts,
        "best_pt": artifacts.get("best_pt", ""),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CPU Training Summary",
        "",
        f"- Status: {summary['status']}",
        f"- Execution: {summary['execution_mode']}",
        f"- Model: `{summary['model']}`",
        f"- Data: `{summary['data_yaml'] or 'not provided'}`",
        f"- Device: `{summary['device']}`",
        f"- Epochs: {summary['epochs']}",
        f"- Image size: {summary['imgsz']}",
        f"- Batch: {summary['batch']}",
        f"- Run dir: `{summary['run_dir']}`",
        "",
        "## Artifacts",
        "",
    ]
    artifacts = summary.get("artifacts", {})
    if isinstance(artifacts, dict):
        for key, value in artifacts.items():
            lines.append(f"- {key}: `{value}`")
    if summary.get("mock"):
        lines.extend(["", "> This was a mock smoke run. The generated best.pt is not a real trained model."])
    return "\n".join(lines) + "\n"


def _error_markdown(error: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CPU Training Failed",
            "",
            f"- Error type: `{error.get('error_type', '')}`",
            f"- Error: {error.get('error', '')}",
            "",
        ]
    )


def _training_config(request: dict[str, Any]) -> TrainingConfig:
    spec = request.get("spec", {})
    if not isinstance(spec, dict):
        raise TrainingRunnerError("request.spec must be an object.")
    data_yaml = _first_string(spec, "data_yaml", "data", "dataset_path")
    return TrainingConfig(
        data_yaml=data_yaml,
        model=_first_string(spec, "model", default="yolo11n.pt"),
        epochs=_bounded_int(spec.get("epochs"), default=1, minimum=1, maximum=5),
        imgsz=_bounded_int(spec.get("imgsz"), default=320, minimum=128, maximum=960),
        batch=_bounded_int(spec.get("batch"), default=1, minimum=1, maximum=8),
        experiment_name=_safe_name(_first_string(spec, "experiment_name", "name", default="cpu_train_smoke")),
        timeout_seconds=_bounded_int(spec.get("timeout_seconds"), default=7200, minimum=30, maximum=86400),
        mock=bool(spec.get("mock", False)),
    )


def _load_request(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingRunnerError(f"Unable to read request JSON: {path}") from exc
    if not isinstance(data, dict):
        raise TrainingRunnerError("request JSON must be an object.")
    return data


def _first_string(data: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
    return cleaned.strip(".-") or "cpu_train_smoke"


def _require_module(module_name: str, message: str) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        raise TrainingRunnerError(message) from exc


if __name__ == "__main__":
    raise SystemExit(main())
