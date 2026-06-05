from __future__ import annotations

from app.core.agent.session_kernel import AgentSessionManager


def test_session_kernel_tracks_turn_lifecycle_without_public_events() -> None:
    manager = AgentSessionManager()

    turn = manager.begin_turn(
        thread_id="thread-1",
        run_id="run-1",
        agent="default",
        workflow="agent_loop",
    )
    running = manager.snapshot("thread-1")

    assert turn.status == "running"
    assert running["active_turn"]["run_id"] == "run-1"  # type: ignore[index]
    assert running["pending_input_count"] == 0

    manager.finish_turn(thread_id="thread-1", run_id="run-1", status="completed")
    finished = manager.snapshot("thread-1")

    assert finished["active_turn"] is None
    assert finished["last_turn"]["status"] == "completed"  # type: ignore[index]


def test_session_kernel_can_cancel_active_turn() -> None:
    manager = AgentSessionManager()
    manager.begin_turn(thread_id="thread-2", run_id="run-2", agent="default", workflow="agent_loop")

    assert manager.cancel_active_turn("thread-2") is True
    snapshot = manager.snapshot("thread-2")

    assert snapshot["active_turn"] is None
    assert snapshot["last_turn"]["status"] == "cancelled"  # type: ignore[index]
