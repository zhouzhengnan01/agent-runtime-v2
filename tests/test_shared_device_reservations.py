from __future__ import annotations

from pathlib import Path

import pytest

from app.core.hardware import device_selector


def _candidate(accelerator: str) -> device_selector.DeviceCandidate:
    return device_selector.DeviceCandidate(
        accelerator=accelerator,
        index=0,
        name=f"{accelerator}-0",
        total_mb=24576,
        used_mb=0,
        free_mb=24576,
        utilization=0,
    )


@pytest.mark.parametrize(
    ("accelerator", "env_name", "runtime_device"),
    [
        ("cuda", "CUDA_VISIBLE_DEVICES", "0"),
        ("npu", "ASCEND_RT_VISIBLE_DEVICES", "npu:0"),
    ],
)
def test_same_accelerator_accepts_multiple_shared_memory_reservations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accelerator: str,
    env_name: str,
    runtime_device: str,
) -> None:
    candidate = _candidate(accelerator)
    monkeypatch.setattr(
        device_selector,
        "_query_cuda_candidates",
        lambda: [candidate] if accelerator == "cuda" else [],
    )
    monkeypatch.setattr(
        device_selector,
        "_query_npu_candidates",
        lambda **_kwargs: [candidate] if accelerator == "npu" else [],
    )

    first = device_selector.reserve_training_device(
        requested="auto",
        backend="deimv2",
        project_root=tmp_path,
        min_free_memory_mb=6000,
        allow_cpu=False,
    )
    second = device_selector.reserve_training_device(
        requested="auto",
        backend="deimv2",
        project_root=tmp_path,
        min_free_memory_mb=6000,
        allow_cpu=False,
    )

    assert first.physical_index == second.physical_index == 0
    assert first.runtime_device == second.runtime_device == runtime_device
    assert first.env[env_name] == second.env[env_name] == "0"
    assert first.lock_path != second.lock_path
    assert Path(first.lock_path).is_file()
    assert Path(second.lock_path).is_file()

    with pytest.raises(RuntimeError):
        device_selector.reserve_training_device(
            requested="auto",
            backend="deimv2",
            project_root=tmp_path,
            min_free_memory_mb=12000,
            allow_cpu=False,
        )

    first.release()
    second.release()
    assert not Path(first.lock_path).exists()
    assert not Path(second.lock_path).exists()
