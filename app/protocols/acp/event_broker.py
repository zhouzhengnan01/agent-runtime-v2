from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid


@dataclass(frozen=True)
class AcpPublishedEvent:
    event: str
    payload: dict[str, Any]
    event_id: str = ""


@dataclass
class _SubscriberState:
    keys: set[str]
    queue: asyncio.Queue[AcpPublishedEvent]
    offsets: dict[str, int]
    seen_event_ids: set[str]
    poll_task: asyncio.Task[None] | None = None


class AcpEventBroker:
    def __init__(self, *, shared_dir: Path | None = None, poll_seconds: float = 0.2) -> None:
        self._subscribers: dict[str, dict[str, _SubscriberState]] = defaultdict(dict)
        self._subscriber_states: dict[str, _SubscriberState] = {}
        self._lock = asyncio.Lock()
        self._shared_dir = shared_dir or Path(".runtime") / "acp_event_broker"
        self._poll_seconds = max(0.05, poll_seconds)

    async def subscribe(self, keys: set[str]) -> tuple[str, asyncio.Queue[AcpPublishedEvent]]:
        subscriber_id = uuid.uuid4().hex
        queue: asyncio.Queue[AcpPublishedEvent] = asyncio.Queue()
        normalized = _normalized_keys(keys)
        state = _SubscriberState(
            keys=normalized,
            queue=queue,
            offsets={key: self._event_log_size(key) for key in normalized},
            seen_event_ids=set(),
        )
        async with self._lock:
            self._subscriber_states[subscriber_id] = state
            for key in normalized:
                self._subscribers[key][subscriber_id] = state
        state.poll_task = asyncio.create_task(self._poll_shared_events(subscriber_id))
        return subscriber_id, queue

    async def unsubscribe(self, subscriber_id: str, keys: set[str]) -> None:
        state: _SubscriberState | None = None
        async with self._lock:
            state = self._subscriber_states.pop(subscriber_id, None)
            unsubscribe_keys = state.keys if state is not None else _normalized_keys(keys)
            for key in unsubscribe_keys:
                subscribers = self._subscribers.get(key)
                if subscribers is None:
                    continue
                subscribers.pop(subscriber_id, None)
                if not subscribers:
                    self._subscribers.pop(key, None)
        if state is not None and state.poll_task is not None:
            state.poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.poll_task

    async def publish(self, keys: set[str], event: str, payload: dict[str, Any], *, shared: bool = False) -> None:
        event_id = uuid.uuid4().hex
        states: dict[str, _SubscriberState] = {}
        normalized = _normalized_keys(keys)
        if shared:
            self._append_shared_event(normalized, event, payload, event_id)
        async with self._lock:
            for key in normalized:
                for subscriber_id, state in self._subscribers.get(key, {}).items():
                    states[subscriber_id] = state
        published = AcpPublishedEvent(event=event, payload=payload, event_id=event_id)
        for state in states.values():
            if event_id in state.seen_event_ids:
                continue
            state.seen_event_ids.add(event_id)
            state.queue.put_nowait(published)

    async def _poll_shared_events(self, subscriber_id: str) -> None:
        while True:
            await asyncio.sleep(self._poll_seconds)
            async with self._lock:
                state = self._subscriber_states.get(subscriber_id)
            if state is None:
                return
            for key in list(state.keys):
                for published in self._read_shared_events(state, key):
                    if published.event_id in state.seen_event_ids:
                        continue
                    state.seen_event_ids.add(published.event_id)
                    state.queue.put_nowait(published)

    def _read_shared_events(self, state: _SubscriberState, key: str) -> list[AcpPublishedEvent]:
        path = self._event_log_path(key)
        if not path.is_file():
            state.offsets[key] = 0
            return []
        offset = state.offsets.get(key, 0)
        events: list[AcpPublishedEvent] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                while True:
                    line_start = handle.tell()
                    line = handle.readline()
                    if not line:
                        state.offsets[key] = handle.tell()
                        break
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        state.offsets[key] = line_start
                        break
                    state.offsets[key] = handle.tell()
                    event = row.get("event")
                    payload = row.get("payload")
                    event_id = row.get("event_id")
                    if isinstance(event, str) and isinstance(payload, dict) and isinstance(event_id, str):
                        events.append(AcpPublishedEvent(event=event, payload=payload, event_id=event_id))
        except OSError:
            return []
        return events

    def _append_shared_event(self, keys: set[str], event: str, payload: dict[str, Any], event_id: str) -> None:
        if not keys:
            return
        row = json.dumps(
            {
                "event_id": event_id,
                "event": event,
                "payload": payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            self._shared_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        for key in keys:
            try:
                with self._event_log_path(key).open("a", encoding="utf-8") as handle:
                    handle.write(row + "\n")
            except OSError:
                continue

    def _event_log_size(self, key: str) -> int:
        path = self._event_log_path(key)
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _event_log_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self._shared_dir / f"{digest}.jsonl"


def _normalized_keys(keys: set[str]) -> set[str]:
    return {str(key).strip() for key in keys if str(key).strip()}


acp_event_broker = AcpEventBroker()
