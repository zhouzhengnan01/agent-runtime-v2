from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.schemas import ChatEvent


@dataclass
class EventRecorder:
    """Collect typed run events for HTTP, CLI, and UI surfaces."""

    agent: str
    thread_id: str
    on_emit: Callable[[ChatEvent], None] | None = None
    _sequence: int = 0
    events: list[ChatEvent] = field(default_factory=list)

    def emit(self, event_type: str, data: dict[str, Any] | None = None, message: str | None = None) -> ChatEvent:
        payload: dict[str, Any] = {
            "agent": self.agent,
            "thread_id": self.thread_id,
            "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if data:
            payload.update(data)
        if message is not None:
            payload["message"] = message

        event = ChatEvent(type=event_type, data=payload)
        self.events.append(event)
        if self.on_emit is not None:
            self.on_emit(event)
        self._sequence += 1
        return event
