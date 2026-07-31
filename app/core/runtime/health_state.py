from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl


MAX_REVIEW_CONCURRENCY = max(1, int(os.getenv("JETLINKS_REVIEW_MAX_CONCURRENCY", "2") or "2"))
READY_MAX_QUEUE_BACKLOG = max(0, int(os.getenv("JETLINKS_READY_MAX_QUEUE_BACKLOG", "32") or "32"))
READY_CPU_LOAD_THRESHOLD = max(0.0, float(os.getenv("JETLINKS_READY_CPU_LOAD_THRESHOLD", "3.5") or "3.5"))
REVIEW_STATE_DIR = Path(os.getenv("JETLINKS_REVIEW_STATE_DIR", "/tmp/jetlinks-agent-runtime"))
REVIEW_SLOT_WAIT_SECONDS = max(0.05, float(os.getenv("JETLINKS_REVIEW_SLOT_WAIT_SECONDS", "0.2") or "0.2"))
REVIEW_SLOT_STALE_SECONDS = max(30.0, float(os.getenv("JETLINKS_REVIEW_SLOT_STALE_SECONDS", "600") or "600"))

_STARTED_AT = time.time()
_PROCESS_HELD_TOKENS: set[str] = set()


def uptime_seconds() -> float:
    return max(0.0, time.time() - _STARTED_AT)


async def review_state() -> dict[str, Any]:
    return await asyncio.to_thread(_review_state_sync)


def ready_from_state(state: dict[str, Any]) -> bool:
    return not readiness_issues(state)


def readiness_issues(state: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if int(state.get("running", 0)) >= MAX_REVIEW_CONCURRENCY:
        issues.append("review_concurrency_full")
    if READY_MAX_QUEUE_BACKLOG > 0 and int(state.get("pending", 0)) > READY_MAX_QUEUE_BACKLOG:
        issues.append("review_queue_backlog_full")
    load1 = state.get("cpu_load_1m")
    if READY_CPU_LOAD_THRESHOLD > 0 and isinstance(load1, int | float) and load1 >= READY_CPU_LOAD_THRESHOLD:
        issues.append("cpu_high")
    return issues


class ReviewSlot:
    def __init__(self) -> None:
        self._token = f"{os.getpid()}:{time.monotonic_ns()}"
        self._cancel_event = threading.Event()

    async def __aenter__(self) -> "ReviewSlot":
        try:
            acquired = await asyncio.to_thread(_acquire_slot_sync, self._token, self._cancel_event)
        except BaseException:
            self._cancel_event.set()
            _release_slot_sync(self._token)
            raise
        if not acquired:
            _release_slot_sync(self._token)
            raise asyncio.CancelledError
        _PROCESS_HELD_TOKENS.add(self._token)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.cancel()

    def cancel(self) -> None:
        self._cancel_event.set()
        _PROCESS_HELD_TOKENS.discard(self._token)
        # Keep cleanup synchronous: an already-cancelled task must not interrupt
        # removal of its cross-worker slot token.
        _release_slot_sync(self._token)


def _review_state_sync() -> dict[str, Any]:
    with _locked_state() as state:
        _cleanup_stale_locked(state)
        _write_state_locked(state)
        return _public_state(state)


def _acquire_slot_sync(token: str, cancel_event: threading.Event | None = None) -> bool:
    cancel_event = cancel_event or threading.Event()
    registered = False
    try:
        while not cancel_event.is_set():
            with _locked_state() as state:
                _cleanup_stale_locked(state)
                if cancel_event.is_set():
                    state.setdefault("running", {}).pop(token, None)
                    state.setdefault("pending", {}).pop(token, None)
                    _write_state_locked(state)
                    return False
                if not registered:
                    state.setdefault("pending", {})
                    state["pending"][token] = _entry()
                    registered = True
                running = state.setdefault("running", {})
                if token in running:
                    _write_state_locked(state)
                    return True
                if len(running) < MAX_REVIEW_CONCURRENCY:
                    state.setdefault("pending", {}).pop(token, None)
                    running[token] = _entry()
                    _write_state_locked(state)
                    return True
                _write_state_locked(state)
            cancel_event.wait(REVIEW_SLOT_WAIT_SECONDS)
        return False
    finally:
        if cancel_event.is_set():
            _release_slot_sync(token)


def _release_slot_sync(token: str) -> None:
    with _locked_state() as state:
        state.setdefault("running", {}).pop(token, None)
        state.setdefault("pending", {}).pop(token, None)
        _cleanup_stale_locked(state)
        _write_state_locked(state)


class _locked_state:
    def __enter__(self) -> dict[str, Any]:
        REVIEW_STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._lock_fp = (REVIEW_STATE_DIR / "review-state.lock").open("a+b")
        _lock_file(self._lock_fp)
        self._state_path = REVIEW_STATE_DIR / "review-state.json"
        try:
            self._state = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            self._state = {}
        if not isinstance(self._state, dict):
            self._state = {}
        self._state.setdefault("running", {})
        self._state.setdefault("pending", {})
        return self._state

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            _unlock_file(self._lock_fp)
        finally:
            self._lock_fp.close()


def _lock_file(lock_fp: Any) -> None:
    if os.name != "nt":
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        return
    lock_fp.seek(0)
    if lock_fp.read(1) == b"":
        lock_fp.write(b"\0")
        lock_fp.flush()
    while True:
        try:
            lock_fp.seek(0)
            msvcrt.locking(lock_fp.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            time.sleep(REVIEW_SLOT_WAIT_SECONDS)


def _unlock_file(lock_fp: Any) -> None:
    if os.name != "nt":
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
        return
    lock_fp.seek(0)
    msvcrt.locking(lock_fp.fileno(), msvcrt.LK_UNLCK, 1)


def _entry() -> dict[str, Any]:
    now = time.time()
    return {"pid": os.getpid(), "updated_at": now}


def _cleanup_stale_locked(state: dict[str, Any]) -> None:
    now = time.time()
    current_pid = os.getpid()
    for bucket_name in ("running", "pending"):
        bucket = state.setdefault(bucket_name, {})
        for token, entry in list(bucket.items()):
            if not isinstance(entry, dict):
                bucket.pop(token, None)
                continue
            pid = int(entry.get("pid") or 0)
            updated_at = float(entry.get("updated_at") or 0)
            if pid == current_pid and token in _PROCESS_HELD_TOKENS:
                entry["updated_at"] = now
                continue
            if updated_at and now - updated_at <= REVIEW_SLOT_STALE_SECONDS and _pid_alive(pid):
                continue
            bucket.pop(token, None)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_state_locked(state: dict[str, Any]) -> None:
    state["max_concurrency"] = MAX_REVIEW_CONCURRENCY
    state["ready_max_queue_backlog"] = READY_MAX_QUEUE_BACKLOG
    state["updated_at"] = time.time()
    path = REVIEW_STATE_DIR / "review-state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_concurrency": MAX_REVIEW_CONCURRENCY,
        "running": len(state.get("running") or {}),
        "pending": len(state.get("pending") or {}),
        "ready_max_queue_backlog": READY_MAX_QUEUE_BACKLOG,
        "ready_cpu_load_threshold": READY_CPU_LOAD_THRESHOLD,
        "cpu_load_1m": _cpu_load_1m(),
    }


def _cpu_load_1m() -> float | None:
    try:
        return round(os.getloadavg()[0], 3)
    except (AttributeError, OSError):
        return None
