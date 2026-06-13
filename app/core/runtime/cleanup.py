from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class RuntimeCleanupConfig:
    root_dir: Path = Path(".runtime")
    retention_seconds: int = 24 * 60 * 60
    interval_seconds: int = 60 * 60
    initial_delay_seconds: int = 5 * 60
    max_delete_per_run: int = 5000
    enabled: bool = True


def runtime_cleanup_config_from_env() -> RuntimeCleanupConfig:
    return RuntimeCleanupConfig(
        root_dir=Path(os.getenv("JETLINKS_RUNTIME_CLEANUP_ROOT", ".runtime")),
        retention_seconds=_positive_int_env("JETLINKS_RUNTIME_CLEANUP_RETENTION_SECONDS", 24 * 60 * 60),
        interval_seconds=_positive_int_env("JETLINKS_RUNTIME_CLEANUP_INTERVAL_SECONDS", 60 * 60),
        initial_delay_seconds=_non_negative_int_env("JETLINKS_RUNTIME_CLEANUP_INITIAL_DELAY_SECONDS", 5 * 60),
        max_delete_per_run=_positive_int_env("JETLINKS_RUNTIME_CLEANUP_MAX_DELETE_PER_RUN", 5000),
        enabled=_bool_env("JETLINKS_RUNTIME_CLEANUP_ENABLED", True),
    )


def start_runtime_cleanup_task(config: RuntimeCleanupConfig | None = None) -> asyncio.Task[None] | None:
    active_config = config or runtime_cleanup_config_from_env()
    if not active_config.enabled:
        logger.info("runtime cleanup disabled")
        return None
    return asyncio.create_task(_runtime_cleanup_loop(active_config), name="runtime-cleanup")


async def stop_runtime_cleanup_task(task: asyncio.Task[None] | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _runtime_cleanup_loop(config: RuntimeCleanupConfig) -> None:
    logger.info(
        "runtime cleanup scheduled root=%s retention_seconds=%s interval_seconds=%s initial_delay_seconds=%s "
        "max_delete_per_run=%s",
        config.root_dir,
        config.retention_seconds,
        config.interval_seconds,
        config.initial_delay_seconds,
        config.max_delete_per_run,
    )
    if config.initial_delay_seconds:
        await asyncio.sleep(config.initial_delay_seconds)
    while True:
        try:
            result = await asyncio.to_thread(run_runtime_cleanup_once, config)
            if result.get("deleted", 0) or result.get("remaining_expired", 0):
                logger.info("runtime cleanup completed result=%s", result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("runtime cleanup failed error=%s", exc, exc_info=True)
        await asyncio.sleep(config.interval_seconds)


def run_runtime_cleanup_once(config: RuntimeCleanupConfig | None = None) -> dict[str, int | str]:
    active_config = config or runtime_cleanup_config_from_env()
    threads_dir = active_config.root_dir / "threads"
    if not threads_dir.is_dir():
        return {"status": "threads_dir_missing", "deleted": 0, "remaining_expired": 0}
    lock_path = active_config.root_dir / ".cleanup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked", "deleted": 0, "remaining_expired": 0}
        cutoff = time.time() - active_config.retention_seconds
        candidates = _expired_thread_dirs(threads_dir, cutoff)
        deleted = 0
        for thread_dir in candidates[: active_config.max_delete_per_run]:
            shutil.rmtree(thread_dir, ignore_errors=True)
            deleted += 1
        remaining = max(0, len(candidates) - deleted)
        return {"status": "ok", "deleted": deleted, "remaining_expired": remaining}


def _expired_thread_dirs(threads_dir: Path, cutoff: float) -> list[Path]:
    expired: list[tuple[float, Path]] = []
    for child in threads_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime < cutoff:
            expired.append((mtime, child))
    expired.sort(key=lambda item: item[0])
    return [path for _mtime, path in expired]


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _positive_int_env(name: str, default: int) -> int:
    value = _int_env(name, default)
    return value if value > 0 else default


def _non_negative_int_env(name: str, default: int) -> int:
    value = _int_env(name, default)
    return value if value >= 0 else default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
