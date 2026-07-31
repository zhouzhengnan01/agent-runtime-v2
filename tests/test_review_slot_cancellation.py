from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import ModuleType


def _load_health_state() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "app" / "core" / "runtime" / "health_state.py"
    spec = importlib.util.spec_from_file_location("health_state_review_slot_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _use_memory_state(health_state: ModuleType, monkeypatch) -> tuple[dict, threading.Lock]:
    monkeypatch.setattr(health_state, "MAX_REVIEW_CONCURRENCY", 1)
    monkeypatch.setattr(health_state, "REVIEW_SLOT_WAIT_SECONDS", 0.01)
    health_state._PROCESS_HELD_TOKENS.clear()
    state = {"running": {}, "pending": {}}
    state_lock = threading.Lock()

    class MemoryStateLock:
        def __enter__(self):
            state_lock.acquire()
            return state

        def __exit__(self, exc_type, exc, tb):
            state_lock.release()

    monkeypatch.setattr(health_state, "_locked_state", MemoryStateLock)
    monkeypatch.setattr(health_state, "_write_state_locked", lambda current: None)
    monkeypatch.setattr(health_state, "_cleanup_stale_locked", lambda current: None)
    return state, state_lock


def test_cancelled_waiter_does_not_leak_review_slot(monkeypatch) -> None:
    health_state = _load_health_state()
    state, state_lock = _use_memory_state(health_state, monkeypatch)

    holder_cancel = threading.Event()
    waiter_cancel = threading.Event()
    holder_token = "holder"
    waiter_token = "waiter"
    assert health_state._acquire_slot_sync(holder_token, holder_cancel) is True

    waiter_result: list[bool] = []
    waiter = threading.Thread(
        target=lambda: waiter_result.append(health_state._acquire_slot_sync(waiter_token, waiter_cancel)),
        daemon=True,
    )
    waiter.start()

    for _ in range(200):
        with state_lock:
            if waiter_token in state["pending"]:
                break
        time.sleep(0.01)
    else:
        raise AssertionError("review slot waiter was not registered")

    waiter_cancel.set()
    waiter.join(timeout=2)
    health_state._release_slot_sync(holder_token)

    assert waiter_result == [False]
    assert state["running"] == {}
    assert state["pending"] == {}


def test_cancelled_async_review_slot_cleans_background_waiter(monkeypatch) -> None:
    health_state = _load_health_state()
    state, state_lock = _use_memory_state(health_state, monkeypatch)

    async def scenario() -> None:
        holder = health_state.ReviewSlot()
        await holder.__aenter__()
        waiter = asyncio.create_task(health_state.ReviewSlot().__aenter__())
        for _ in range(200):
            with state_lock:
                if len(state["pending"]) == 1:
                    break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("async review slot waiter was not registered")

        waiter.cancel()
        with suppress(asyncio.CancelledError):
            await waiter
        await holder.__aexit__(None, None, None)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert state["running"] == {}
    assert state["pending"] == {}
