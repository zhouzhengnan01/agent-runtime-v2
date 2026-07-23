from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: str):
    module_path = (ROOT / path).resolve()
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deimv2_failure_reply_contains_train_log_root_cause(tmp_path: Path) -> None:
    runner = _load_module("deimv2_auto_training_runner_failure_test", "plugins/skills/deimv2-auto-training/runner.py")
    workflow = _load_module("yolo_training_flow_failure_test", "plugins/workflows/builtin-artifact-workflows/yolo_training_flow.py")
    project_dir = tmp_path / "deimv2_training_run"
    log_path = project_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        """Epoch: [0] [ 1/10] loss: 3.2
Traceback (most recent call last):
  File \"torch/utils/data/_utils/signal_handling.py\", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 80566) is killed by signal: Killed.

The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File \"tools/train.py\", line 88, in <module>
    main(args)
RuntimeError: DataLoader worker (pid(s) 80566) exited unexpectedly
""",
        encoding="utf-8",
    )

    fallback = runner._build_reply(
        project_dir,
        1,
        {},
        "",
        f"RuntimeError: DEIMv2 train failed. See {log_path}",
    )
    payload = json.loads(fallback)
    reply = workflow._training_failed_reply(
        summary={},
        fallback_reply=fallback,
        best_pt="",
        data_preparation_summary={},
        training_backend="deimv2",
    )

    assert payload["training_error"]["message"] == "RuntimeError: DataLoader worker (pid 80566) is killed by signal: Killed."
    assert payload["training_error"]["log_path"] == str(log_path)
    assert "内存不足（OOM）" in payload["training_error"]["hint"]
    assert "RuntimeError: DataLoader worker (pid 80566) is killed by signal: Killed." in reply
    assert "DataLoader worker (pid(s) 80566) exited unexpectedly" in reply
    assert str(log_path) in reply


def test_deimv2_failure_uses_log_tail_when_process_has_no_python_exception(tmp_path: Path) -> None:
    runner = _load_module("deimv2_auto_training_runner_native_failure_test", "plugins/skills/deimv2-auto-training/runner.py")
    project_dir = tmp_path / "deimv2_training_run"
    log_path = project_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("loading dataset\nnative process stopped unexpectedly\n", encoding="utf-8")

    error = runner._read_training_error(project_dir)

    assert error is not None
    assert "native process stopped unexpectedly" in error["message"]
    assert error["log_path"] == str(log_path)


def test_deimv2_failure_does_not_report_training_progress_as_error(tmp_path: Path) -> None:
    runner = _load_module("deimv2_auto_training_runner_progress_test", "plugins/skills/deimv2-auto-training/runner.py")
    project_dir = tmp_path / "deimv2_training_run"
    log_path = project_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "Epoch: [0] Total time: 0:00:55\n"
        "Average Precision (AP) = 0.000\n"
        "Epoch: [1] [0/3] loss: 51.5388 time: 38.9895\n",
        encoding="utf-8",
    )

    error = runner._read_training_error(project_dir)

    assert error is not None
    assert error["message"] == "训练进程异常退出，但 train.log 中未检测到明确的报错信息。"
    assert "details" not in error
    assert "Epoch:" not in error["message"]
    assert "loss:" not in error["message"]
    assert error["log_path"] == str(log_path)


def test_deimv2_failure_detects_unknown_exception_type(tmp_path: Path) -> None:
    runner = _load_module("deimv2_auto_training_runner_unknown_error_test", "plugins/skills/deimv2-auto-training/runner.py")
    project_dir = tmp_path / "deimv2_training_run"
    log_path = project_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "Epoch: [3] [2/10] loss: 4.2\n"
        "Traceback (most recent call last):\n"
        "  File \"vendor/runtime.py\", line 42, in execute\n"
        "    client.send(batch)\n"
        "vendor.transport.RemoteProtocolException: peer closed the stream\n",
        encoding="utf-8",
    )

    error = runner._read_training_error(project_dir)

    assert error is not None
    assert error["message"] == "vendor.transport.RemoteProtocolException: peer closed the stream"
    assert error["details"].startswith("Traceback (most recent call last):")
    assert "loss: 4.2" not in error["details"]


def test_deimv2_failure_detects_generic_error_severity(tmp_path: Path) -> None:
    runner = _load_module("deimv2_auto_training_runner_severity_test", "plugins/skills/deimv2-auto-training/runner.py")
    project_dir = tmp_path / "deimv2_training_run"
    log_path = project_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "Epoch: [1] [0/3] loss: 51.5\n"
        "[2026-07-22 12:00:00] CRITICAL worker pool became unavailable\n",
        encoding="utf-8",
    )

    error = runner._read_training_error(project_dir)

    assert error is not None
    assert "CRITICAL worker pool became unavailable" in error["message"]
    assert "loss: 51.5" not in error["details"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_deimv2_failure_reply_contains_train_log_root_cause(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_deimv2_failure_uses_log_tail_when_process_has_no_python_exception(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_deimv2_failure_does_not_report_training_progress_as_error(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_deimv2_failure_detects_unknown_exception_type(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_deimv2_failure_detects_generic_error_severity(Path(directory))
    print("direct assertion passed")
