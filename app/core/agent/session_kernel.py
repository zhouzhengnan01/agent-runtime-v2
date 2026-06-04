from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event, RLock
from typing import Literal
from uuid import uuid4


TurnStatus = Literal["running", "completed", "failed", "cancelled"]


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class AgentTurn:
    """Internal lifecycle state for one runtime turn.

    The current HTTP/ACP contracts remain request-driven, so this object is not
    serialized into public responses. It gives the runtime a stable place to
    track active turns, cancellation, and future queued input without changing
    the existing event schema.
    """

    turn_id: str
    run_id: str
    thread_id: str
    agent: str
    workflow: str
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    status: TurnStatus = "running"
    cancel_requested: bool = False
    _cancel_event: Event = field(default_factory=Event, repr=False)

    def request_cancel(self) -> None:
        self.cancel_requested = True
        self.status = "cancelled"
        self._cancel_event.set()

    def finish(self, status: TurnStatus = "completed") -> None:
        if self.status == "cancelled":
            return
        self.status = status
        self.finished_at = _now()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()


@dataclass
class AgentSession:
    thread_id: str
    active_turn: AgentTurn | None = None
    pending_inputs: list[dict[str, object]] = field(default_factory=list)
    last_turn: AgentTurn | None = None

    def begin_turn(self, *, run_id: str, agent: str, workflow: str) -> AgentTurn:
        turn = AgentTurn(
            turn_id=f"turn-{uuid4().hex[:12]}",
            run_id=run_id,
            thread_id=self.thread_id,
            agent=agent,
            workflow=workflow,
        )
        self.active_turn = turn
        self.last_turn = turn
        return turn

    def finish_turn(self, run_id: str, status: TurnStatus = "completed") -> None:
        turn = self.active_turn
        if turn is None or turn.run_id != run_id:
            return
        turn.finish(status)
        self.active_turn = None
        self.last_turn = turn

    def cancel_active_turn(self) -> bool:
        if self.active_turn is None:
            return False
        self.active_turn.request_cancel()
        self.last_turn = self.active_turn
        self.active_turn = None
        return True


class AgentSessionManager:
    """Small in-memory session kernel shared by runtime entrypoints.

    It is deliberately conservative: it does not alter public events, persisted
    conversation layout, or ACP semantics. The goal is to create a runtime-owned
    lifecycle boundary that can later grow pending input, interruption, and
    resume behavior behind the existing protocol surface.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, AgentSession] = {}

    def get_or_create(self, thread_id: str) -> AgentSession:
        normalized = thread_id.strip() or "thread-default"
        with self._lock:
            session = self._sessions.get(normalized)
            if session is None:
                session = AgentSession(thread_id=normalized)
                self._sessions[normalized] = session
            return session

    def begin_turn(self, *, thread_id: str, run_id: str, agent: str, workflow: str) -> AgentTurn:
        with self._lock:
            session = self.get_or_create(thread_id)
            return session.begin_turn(run_id=run_id, agent=agent, workflow=workflow)

    def finish_turn(self, *, thread_id: str, run_id: str, status: TurnStatus = "completed") -> None:
        with self._lock:
            self.get_or_create(thread_id).finish_turn(run_id, status=status)

    def cancel_active_turn(self, thread_id: str) -> bool:
        with self._lock:
            return self.get_or_create(thread_id).cancel_active_turn()

    def snapshot(self, thread_id: str) -> dict[str, object]:
        with self._lock:
            session = self.get_or_create(thread_id)
            active = session.active_turn
            last = session.last_turn
            return {
                "thread_id": session.thread_id,
                "active_turn": _turn_snapshot(active),
                "last_turn": _turn_snapshot(last),
                "pending_input_count": len(session.pending_inputs),
            }


def _turn_snapshot(turn: AgentTurn | None) -> dict[str, object] | None:
    if turn is None:
        return None
    return {
        "turn_id": turn.turn_id,
        "run_id": turn.run_id,
        "thread_id": turn.thread_id,
        "agent": turn.agent,
        "workflow": turn.workflow,
        "started_at": turn.started_at,
        "finished_at": turn.finished_at,
        "status": turn.status,
        "cancel_requested": turn.cancel_requested,
    }

