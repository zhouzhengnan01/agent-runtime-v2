from __future__ import annotations

import atexit
import json
import os
import platform
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LOCK_STALE_SECONDS = int(os.environ.get("JETLINKS_DEVICE_LOCK_STALE_SECONDS", str(24 * 60 * 60)))


@dataclass
class DeviceCandidate:
    accelerator: str
    index: int
    name: str = ""
    total_mb: int | None = None
    used_mb: int | None = None
    free_mb: int | None = None
    utilization: int | None = None
    available: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator": self.accelerator,
            "index": self.index,
            "name": self.name,
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "utilization": self.utilization,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass
class TrainingDeviceReservation:
    accelerator: str
    runtime_device: str
    physical_index: int | None = None
    env: dict[str, str] = field(default_factory=dict)
    lock_path: str = ""
    lock_token: str = ""
    reason: str = ""
    requested: str = ""
    required_free_mb: int | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accelerator": self.accelerator,
            "runtime_device": self.runtime_device,
            "physical_index": self.physical_index,
            "env": self.env,
            "lock_path": self.lock_path,
            "reason": self.reason,
            "requested": self.requested,
            "required_free_mb": self.required_free_mb,
            "candidates": self.candidates,
            "explicit": self.explicit,
        }

    def release(self) -> None:
        if not self.lock_path:
            return
        path = Path(self.lock_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if str(payload.get("token") or "") != self.lock_token:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def reserve_training_device(
    *,
    requested: object = "auto",
    backend: str,
    model_variant: object = "",
    batch: object = None,
    img_size: object = None,
    project_root: Path,
    min_free_memory_mb: object = None,
    max_gpu_utilization: object = None,
    allow_cpu: bool = True,
) -> TrainingDeviceReservation:
    """预约一个训练设备，并返回训练进程实际使用的设备映射。

    生产调度建议传入 device=auto。显式请求如 cuda:6 或 npu:0
    会严格执行，设备不可用时直接失败。
    """
    raw_requested = str(requested or "auto").strip()
    requested_norm = raw_requested.lower()
    required_mb = _coerce_positive_int(min_free_memory_mb) or _estimate_required_mb(
        backend=backend,
        model_variant=str(model_variant or ""),
        batch=_coerce_positive_int(batch),
        img_size=_coerce_positive_int(img_size),
    )
    max_util = _coerce_positive_int(max_gpu_utilization)
    if max_util is None:
        max_util = 95
    lock_dir = project_root / ".runtime" / "device_locks"
    explicit = _is_explicit_device(requested_norm)

    if requested_norm in {"cpu", "none"}:
        return TrainingDeviceReservation(
            accelerator="cpu",
            runtime_device="cpu",
            reason="CPU was explicitly requested.",
            requested=raw_requested,
            explicit=True,
        )

    cuda_candidates = _query_cuda_candidates()
    npu_candidates = _query_npu_candidates()
    all_candidates = [candidate.to_dict() for candidate in [*cuda_candidates, *npu_candidates]]

    if _wants_cuda(requested_norm):
        selected = _select_candidate(
            candidates=cuda_candidates,
            requested=requested_norm,
            required_free_mb=required_mb,
            max_utilization=max_util,
            lock_dir=lock_dir,
            explicit=explicit,
        )
        if selected is not None:
            return selected
        if explicit:
            raise RuntimeError(_selection_error("CUDA", raw_requested, required_mb, cuda_candidates))

    if _wants_npu(requested_norm):
        selected = _select_candidate(
            candidates=npu_candidates,
            requested=requested_norm,
            required_free_mb=required_mb,
            max_utilization=max_util,
            lock_dir=lock_dir,
            explicit=explicit,
        )
        if selected is not None:
            return selected
        if explicit:
            raise RuntimeError(_selection_error("NPU", raw_requested, required_mb, npu_candidates))

    if requested_norm in {"", "auto", "gpu", "cuda", "cuda:auto"}:
        selected = _select_candidate(
            candidates=cuda_candidates,
            requested="auto",
            required_free_mb=required_mb,
            max_utilization=max_util,
            lock_dir=lock_dir,
            explicit=False,
        )
        if selected is not None:
            selected.candidates = all_candidates
            return selected

        selected = _select_candidate(
            candidates=npu_candidates,
            requested="auto",
            required_free_mb=required_mb,
            max_utilization=max_util,
            lock_dir=lock_dir,
            explicit=False,
        )
        if selected is not None:
            selected.candidates = all_candidates
            return selected

    if allow_cpu:
        return TrainingDeviceReservation(
            accelerator="cpu",
            runtime_device="cpu",
            reason="No accelerator with enough free resources was selected; falling back to CPU.",
            requested=raw_requested,
            required_free_mb=required_mb,
            candidates=all_candidates,
            explicit=False,
        )

    raise RuntimeError(_selection_error("accelerator", raw_requested, required_mb, [*cuda_candidates, *npu_candidates]))


def _select_candidate(
    *,
    candidates: list[DeviceCandidate],
    requested: str,
    required_free_mb: int,
    max_utilization: int,
    lock_dir: Path,
    explicit: bool,
) -> TrainingDeviceReservation | None:
    if not candidates:
        return None
    filtered = _filter_requested(candidates, requested)
    if not filtered:
        return None
    evaluated: list[DeviceCandidate] = []
    for candidate in filtered:
        candidate = DeviceCandidate(**candidate.to_dict())
        lock_path = _lock_path(lock_dir, candidate.accelerator, candidate.index)
        # 这里使用进程级锁，而不是依赖深度学习框架内部状态。
        # 这样可以避免两个 ACP 训练请求同时选中同一张看起来空闲的卡；
        # 过期锁会在下面自动清理。
        if _lock_is_live(lock_path):
            candidate.available = False
            candidate.reason = f"device is reserved by {lock_path}"
        elif candidate.free_mb is not None and candidate.free_mb < required_free_mb:
            candidate.available = False
            candidate.reason = f"free memory {candidate.free_mb} MB is below required {required_free_mb} MB"
        elif (
            not explicit
            and candidate.utilization is not None
            and candidate.utilization > max_utilization
        ):
            candidate.available = False
            candidate.reason = f"utilization {candidate.utilization}% is above limit {max_utilization}%"
        else:
            candidate.available = True
            candidate.reason = "available"
        evaluated.append(candidate)

    available = [candidate for candidate in evaluated if candidate.available]
    available.sort(
        key=lambda item: (
            item.free_mb if item.free_mb is not None else -1,
            -(item.utilization if item.utilization is not None else 0),
            -item.index,
        ),
        reverse=True,
    )
    for candidate in available:
        reservation = _try_reserve(candidate, lock_dir, requested, required_free_mb, explicit, evaluated)
        if reservation is not None:
            return reservation
    return None


def _try_reserve(
    candidate: DeviceCandidate,
    lock_dir: Path,
    requested: str,
    required_free_mb: int,
    explicit: bool,
    candidates: list[DeviceCandidate],
) -> TrainingDeviceReservation | None:
    token = uuid.uuid4().hex
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = _lock_path(lock_dir, candidate.accelerator, candidate.index)
    payload = {
        "token": token,
        "pid": os.getpid(),
        "created_at": time.time(),
        "accelerator": candidate.accelerator,
        "index": candidate.index,
        "hostname": platform.node(),
    }
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    if candidate.accelerator == "cuda":
        # 将选中的物理 GPU 映射为进程内的本地 device 0。
        # 这样 YOLO 和 DEIMv2 命令保持简单，同时避开繁忙物理卡。
        env = {"CUDA_VISIBLE_DEVICES": str(candidate.index)}
        runtime_device = "0"
    else:
        # Ascend 工具链在 ASCEND_RT_VISIBLE_DEVICES 过滤后，
        # 通常从 0 开始寻址可见 NPU，因此训练命令使用 npu:0。
        env = {
            "ASCEND_RT_VISIBLE_DEVICES": str(candidate.index),
            "ASCEND_VISIBLE_DEVICES": str(candidate.index),
            "NPU_VISIBLE_DEVICES": str(candidate.index),
        }
        runtime_device = "npu:0"
    reservation = TrainingDeviceReservation(
        accelerator=candidate.accelerator,
        runtime_device=runtime_device,
        physical_index=candidate.index,
        env=env,
        lock_path=str(path),
        lock_token=token,
        reason=f"selected {candidate.accelerator}:{candidate.index} with free memory {candidate.free_mb if candidate.free_mb is not None else 'unknown'} MB",
        requested=requested,
        required_free_mb=required_free_mb,
        candidates=[item.to_dict() for item in candidates],
        explicit=explicit,
    )
    atexit.register(reservation.release)
    return reservation


def _query_cuda_candidates() -> list[DeviceCandidate]:
    candidates = _query_cuda_with_nvidia_smi()
    if candidates:
        return candidates
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            return []
        count = int(cuda.device_count())
        result: list[DeviceCandidate] = []
        for index in range(count):
            name = ""
            try:
                name = str(cuda.get_device_name(index))
            except Exception:
                name = f"CUDA GPU {index}"
            result.append(DeviceCandidate(accelerator="cuda", index=index, name=name))
        return result
    except Exception:
        return []


def _query_cuda_with_nvidia_smi() -> list[DeviceCandidate]:
    # 调度时优先使用 nvidia-smi，因为 torch 通常只能拿到卡数量和名称，
    # 拿不到空闲显存和利用率。
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    result: list[DeviceCandidate] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        index = _coerce_positive_int(parts[0])
        if index is None:
            continue
        result.append(
            DeviceCandidate(
                accelerator="cuda",
                index=index,
                name=parts[1],
                total_mb=_coerce_positive_int(parts[2]),
                used_mb=_coerce_positive_int(parts[3]),
                free_mb=_coerce_positive_int(parts[4]),
                utilization=_coerce_positive_int(parts[5]),
            )
        )
    return result


def _query_npu_candidates() -> list[DeviceCandidate]:
    # NPU 空闲显存查询会受 Ascend 驱动和 toolkit 版本影响。
    # 当前先按索引做保守预约，至少避免并发请求撞到同一张 NPU。
    if not _has_ascend_runtime_hint():
        return []
    count = _torch_npu_device_count()
    if count <= 0:
        count = _count_visible_npu_devices()
    if count <= 0:
        return []
    return [DeviceCandidate(accelerator="npu", index=index, name=f"Ascend NPU {index}") for index in range(count)]


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


def _torch_npu_device_count() -> int:
    try:
        import torch
        import torch_npu  # noqa: F401

        npu = getattr(torch, "npu", None)
        if npu is None or not bool(npu.is_available()):
            return 0
        return int(npu.device_count())
    except Exception:
        return 0


def _count_visible_npu_devices() -> int:
    for name in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"):
        raw = str(os.environ.get(name) or "").strip()
        if raw:
            return len([item for item in raw.split(",") if item.strip()])
    return 1


def _filter_requested(candidates: list[DeviceCandidate], requested: str) -> list[DeviceCandidate]:
    index = _requested_index(requested)
    if index is None:
        return candidates
    return [candidate for candidate in candidates if candidate.index == index]


def _requested_index(requested: str) -> int | None:
    if not requested or requested in {"auto", "gpu", "cuda", "npu", "cuda:auto", "npu:auto"}:
        return None
    if requested.isdigit():
        return int(requested)
    match = re.match(r"^(?:cuda|npu):(\d+)$", requested)
    if match:
        return int(match.group(1))
    return None


def _wants_cuda(requested: str) -> bool:
    return requested in {"", "auto", "gpu", "cuda", "cuda:auto"} or requested.startswith("cuda:") or requested.isdigit()


def _wants_npu(requested: str) -> bool:
    return requested in {"", "auto", "npu", "npu:auto"} or requested.startswith("npu:")


def _is_explicit_device(requested: str) -> bool:
    return bool(requested and requested not in {"auto", "gpu", "cuda:auto", "npu:auto"})


def _lock_path(lock_dir: Path, accelerator: str, index: int) -> Path:
    return lock_dir / f"{accelerator}-{index}.lock"


def _lock_is_live(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _unlink_quietly(path)
        return False
    created_at = float(payload.get("created_at") or 0)
    pid = int(payload.get("pid") or 0)
    if created_at and time.time() - created_at > LOCK_STALE_SECONDS:
        _unlink_quietly(path)
        return False
    if pid and not _pid_alive(pid):
        _unlink_quietly(path)
        return False
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _estimate_required_mb(*, backend: str, model_variant: str, batch: int | None, img_size: int | None) -> int:
    # 这里是保守调度阈值，不是精确显存模型。
    # 它只决定某张卡是否值得尝试；真正训练时框架自身的 OOM
    # 仍然是最后一道保护。
    backend_key = backend.lower()
    variant = model_variant.lower()
    batch = batch or (8 if backend_key == "deimv2" else 16)
    img_size = img_size or 640
    scale = max(0.25, (float(img_size) / 640.0) ** 2)

    if backend_key == "deimv2":
        if variant.endswith("-x") or "_x" in variant:
            base = 32768
            default_batch = 2
        elif variant.endswith("-l") or "_l" in variant:
            base = 24576
            default_batch = 2
        elif variant.endswith("-m") or "_m" in variant:
            base = 16384
            default_batch = 4
        else:
            base = 8192
            default_batch = 8
        return int(max(4096, base * (float(batch) / float(default_batch)) * scale * 1.15))

    variant_text = variant or ""
    if re.search(r"(?:^|[^a-z])x(?:\.pt)?$", variant_text):
        base = 24576
        default_batch = 16
    elif re.search(r"(?:^|[^a-z])l(?:\.pt)?$", variant_text):
        base = 16384
        default_batch = 16
    elif re.search(r"(?:^|[^a-z])m(?:\.pt)?$", variant_text):
        base = 10240
        default_batch = 16
    elif re.search(r"(?:^|[^a-z])s(?:\.pt)?$", variant_text):
        base = 6144
        default_batch = 16
    else:
        base = 4096
        default_batch = 16
    return int(max(3072, base * (float(batch) / float(default_batch)) * scale * 1.15))


def _coerce_positive_int(value: object) -> int | None:
    try:
        number = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _selection_error(
    accelerator: str,
    requested: str,
    required_free_mb: int,
    candidates: list[DeviceCandidate],
) -> str:
    details = "; ".join(
        f"{item.accelerator}:{item.index} free={item.free_mb if item.free_mb is not None else 'unknown'}MB util={item.utilization if item.utilization is not None else 'unknown'}% reason={item.reason or 'not selected'}"
        for item in candidates
    )
    return (
        f"Requested {accelerator} device '{requested}' could not be reserved. "
        f"Required free memory is about {required_free_mb} MB. "
        f"Candidates: {details or 'none'}"
    )
