from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import uuid


@dataclass(frozen=True)
class AcpPublishedEvent:
    event: str
    payload: dict[str, Any]


class AcpEventBroker:
    def __init__(self) -> None:
        self._subscribers: dict[str, dict[str, asyncio.Queue[AcpPublishedEvent]]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def subscribe(self, keys: set[str]) -> tuple[str, asyncio.Queue[AcpPublishedEvent]]:
        subscriber_id = uuid.uuid4().hex
        queue: asyncio.Queue[AcpPublishedEvent] = asyncio.Queue()
        async with self._lock:
            for key in _normalized_keys(keys):
                self._subscribers[key][subscriber_id] = queue
        return subscriber_id, queue

    async def unsubscribe(self, subscriber_id: str, keys: set[str]) -> None:
        async with self._lock:
            for key in _normalized_keys(keys):
                subscribers = self._subscribers.get(key)
                if subscribers is None:
                    continue
                subscribers.pop(subscriber_id, None)
                if not subscribers:
                    self._subscribers.pop(key, None)

    async def publish(self, keys: set[str], event: str, payload: dict[str, Any]) -> None:
        queues: dict[int, asyncio.Queue[AcpPublishedEvent]] = {}
        async with self._lock:
            for key in _normalized_keys(keys):
                for queue in self._subscribers.get(key, {}).values():
                    queues[id(queue)] = queue
        published = AcpPublishedEvent(event=event, payload=payload)
        for queue in queues.values():
            queue.put_nowait(published)


def _normalized_keys(keys: set[str]) -> set[str]:
    return {str(key).strip() for key in keys if str(key).strip()}


acp_event_broker = AcpEventBroker()
