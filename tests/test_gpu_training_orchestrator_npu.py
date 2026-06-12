from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner_module():
    path = Path("plugins/skills/gpu-training-orchestrator/runner.py").resolve()
    spec = importlib.util.spec_from_file_location("gpu_training_orchestrator_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepared_training_module():
    path = Path("plugins/skills/gpu-training-orchestrator/scripts/run_prepared_yolo_training.py").resolve()
    spec = importlib.util.spec_from_file_location("run_prepared_yolo_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_npu_runtime_uses_current_python_even_when_conda_requested(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setenv("JETLINKS_FORCE_CURRENT_PYTHON_ON_NPU", "true")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)),
    )
    script = Path("/opt/app/train.py")
    request = Path("/tmp/input.json")

    command = module._training_command(
        {"runtime": {"conda_env_name": "yolo_jetson"}},
        script,
        request,
    )

    assert command == [sys.executable, str(script), "--input", str(request)]


def test_cuda_runtime_keeps_conda_priority_over_npu_hint(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.setenv("JETLINKS_FORCE_CURRENT_PYTHON_ON_NPU", "true")
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)),
    )
    script = Path("/opt/app/train.py")
    request = Path("/tmp/input.json")

    command = module._training_command(
        {"runtime": {"conda_env_name": "cuda-train"}},
        script,
        request,
    )

    assert command[:6] == ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "cuda-train", "python"]


def test_non_npu_runtime_keeps_existing_conda_command(monkeypatch) -> None:
    module = _load_runner_module()
    monkeypatch.delenv("JETLINKS_FORCE_CURRENT_PYTHON_ON_NPU", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    script = Path("/opt/app/train.py")
    request = Path("/tmp/input.json")

    command = module._training_command(
        {"runtime": {"conda_env_name": "yolo_jetson"}},
        script,
        request,
    )

    assert command == [
        "/opt/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "yolo_jetson",
        "python",
        str(script),
        "--input",
        str(request),
    ]


def test_physical_card_seven_maps_to_container_npu_zero(monkeypatch) -> None:
    module = _load_prepared_training_module()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "7")

    assert module._normalize_device_for_runtime("7", has_npu=True, has_cuda=False) == "npu:0"
    assert module._normalize_device_for_runtime("0", has_npu=True, has_cuda=False) == "npu:0"


def test_cuda_has_priority_over_npu() -> None:
    module = _load_prepared_training_module()

    assert module._normalize_device_for_runtime("npu:0", has_npu=True, has_cuda=True) == "0"


def test_npu_has_priority_over_cpu() -> None:
    module = _load_prepared_training_module()

    assert module._normalize_device_for_runtime("cpu", has_npu=True, has_cuda=False) == "npu:0"


def test_cpu_is_used_when_no_accelerator_is_available() -> None:
    module = _load_prepared_training_module()

    assert module._normalize_device_for_runtime("0", has_npu=False, has_cuda=False) == "cpu"
