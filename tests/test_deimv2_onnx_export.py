from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_runner() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "skills"
        / "deimv2-auto-training"
        / "scripts"
        / "run_deimv2_training.py"
    )
    spec = importlib.util.spec_from_file_location("run_deimv2_training_onnx_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner() -> ModuleType:
    return _load_runner()


def _export_fixture(tmp_path: Path) -> dict[str, Path]:
    deim_root = tmp_path / "deimv2"
    exporter = deim_root / "tools" / "deployment" / "export_onnx.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# test exporter\n", encoding="utf-8")
    config = tmp_path / "train.yml"
    checkpoint = tmp_path / "best_stg1.pth"
    config.write_text("task: detection\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    logs = tmp_path / "logs"
    logs.mkdir()
    return {
        "deim_root": deim_root,
        "config": config,
        "checkpoint": checkpoint,
        "logs": logs,
        "output": checkpoint.with_suffix(".onnx"),
    }


@pytest.mark.parametrize(
    ("accelerator", "visibility"),
    [
        ("gpu", {"CUDA_VISIBLE_DEVICES": "1"}),
        (
            "npu",
            {
                "ASCEND_RT_VISIBLE_DEVICES": "3",
                "ASCEND_VISIBLE_DEVICES": "3",
                "NPU_VISIBLE_DEVICES": "3",
            },
        ),
    ],
)
def test_onnx_export_isolates_gpu_and_npu_runtime(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accelerator: str,
    visibility: dict[str, str],
) -> None:
    paths = _export_fixture(tmp_path)
    for name, value in visibility.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "128")

    calls: list[dict[str, Any]] = []

    def fake_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if kwargs["name"] == "export":
            paths["output"].write_bytes(b"base-onnx")
        elif kwargs["name"] == "simplify":
            Path(kwargs["cmd"][-2]).write_bytes(b"simplified-onnx")
        return {
            "name": kwargs["name"],
            "status": "completed",
            "returncode": 0,
            "timeout_seconds": kwargs["timeout_seconds"],
            "elapsed_seconds": 0.01,
            "command": kwargs["cmd"],
            "log": str(kwargs["log_path"]),
            "error": "",
        }

    monkeypatch.setattr(runner, "_run_onnx_stage", fake_stage)
    result = runner.export_onnx_after_training(
        prefix=["python"],
        deim_root=paths["deim_root"],
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        logs_dir=paths["logs"],
        training={
            "onnx_check": True,
            "onnx_simplify": True,
            "onnx_export_batch_size": 1,
            "onnx_export_threads": 4,
            "onnx_export_timeout_seconds": 11,
            "onnx_simplify_timeout_seconds": 12,
            "onnx_check_timeout_seconds": 13,
        },
    )

    assert result["status"] == "completed"
    assert paths["output"].read_bytes() == b"simplified-onnx"
    assert [call["name"] for call in calls] == ["export", "simplify", "check"]
    assert [call["timeout_seconds"] for call in calls] == [11, 12, 13]
    assert "--check" not in calls[0]["cmd"]
    assert "--simplify" not in calls[0]["cmd"]
    assert calls[0]["cmd"][calls[0]["cmd"].index("--batch-size") + 1] == "1"
    assert result["batch_size"] == 1

    thread_variables = {
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
    for call in calls:
        env = call["env"]
        assert env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        assert all(env[name] == "4" for name in thread_variables)
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "ASCEND_RT_VISIBLE_DEVICES" not in env
        assert "ASCEND_VISIBLE_DEVICES" not in env
        assert "NPU_VISIBLE_DEVICES" not in env

    assert accelerator in {"gpu", "npu"}


def test_onnx_stage_timeout_terminates_process_group(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimedOutProcess:
        pid = 4321

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired(["python", "export.py"], timeout)

        def poll(self) -> None:
            return None

    terminated: list[int] = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: TimedOutProcess())
    monkeypatch.setattr(
        runner,
        "_terminate_process_group",
        lambda proc: terminated.append(proc.pid),
    )

    result = runner._run_onnx_stage(
        name="export",
        cmd=["python", "export.py"],
        cwd=tmp_path,
        log_path=tmp_path / "export.log",
        env=os.environ.copy(),
        timeout_seconds=3,
    )

    assert result["status"] == "timeout"
    assert result["returncode"] is None
    assert terminated == [4321]
    assert "process group terminated" in (tmp_path / "export.log").read_text(encoding="utf-8")


def test_simplify_timeout_keeps_base_onnx_and_skips_check(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _export_fixture(tmp_path)
    calls: list[str] = []

    def fake_stage(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["name"])
        if kwargs["name"] == "export":
            paths["output"].write_bytes(b"base-onnx")
            status = "completed"
            returncode = 0
            error = ""
        else:
            status = "timeout"
            returncode = None
            error = "ONNX simplify timed out after 2 seconds."
        return {
            "name": kwargs["name"],
            "status": status,
            "returncode": returncode,
            "timeout_seconds": kwargs["timeout_seconds"],
            "elapsed_seconds": 0.01,
            "command": kwargs["cmd"],
            "log": str(kwargs["log_path"]),
            "error": error,
        }

    monkeypatch.setattr(runner, "_run_onnx_stage", fake_stage)
    result = runner.export_onnx_after_training(
        prefix=["python"],
        deim_root=paths["deim_root"],
        config_path=paths["config"],
        checkpoint_path=paths["checkpoint"],
        logs_dir=paths["logs"],
        training={
            "onnx_check": True,
            "onnx_simplify": True,
            "onnx_simplify_timeout_seconds": 2,
        },
    )

    assert result["status"] == "timeout"
    assert result["failed_stage"] == "simplify"
    assert result["onnx_path"] == str(paths["output"])
    assert paths["output"].read_bytes() == b"base-onnx"
    assert calls == ["export", "simplify"]
