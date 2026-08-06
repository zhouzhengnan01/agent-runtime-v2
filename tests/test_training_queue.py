from __future__ import annotations

import asyncio

from app.core.training_queue import TrainingQueueManager


def _record(index: int) -> dict[str, object]:
    return {
        "job_id": f"job-{index}",
        "thread_id": f"thread-{index}",
    }


def test_training_queue_admits_ten_runs_and_ten_waiters_then_rejects() -> None:
    queue = TrainingQueueManager(max_running=10, max_waiting=10)
    records = [_record(index) for index in range(21)]

    states = [queue.admit(record) for record in records]

    assert states == ["running"] * 10 + ["waiting"] * 10 + ["queue_rejected"]
    assert [records[index]["queue_position"] for index in range(10, 20)] == list(range(1, 11))
    snapshot = queue.snapshot()
    assert snapshot["running"] == 10
    assert snapshot["waiting"] == 10
    assert snapshot["queue_full"] is True


def test_training_queue_promotes_waiters_in_fifo_order() -> None:
    queue = TrainingQueueManager(max_running=2, max_waiting=3)
    records = [_record(index) for index in range(5)]
    for record in records:
        queue.admit(record)

    promoted = queue.finish("job-0")

    assert [record["job_id"] for record in promoted] == ["job-2"]
    assert records[2]["queue_status"] == "running"
    assert records[3]["queue_position"] == 1
    assert records[4]["queue_position"] == 2


def test_training_queue_can_cancel_waiting_job_without_using_running_slot() -> None:
    queue = TrainingQueueManager(max_running=1, max_waiting=2)
    records = [_record(index) for index in range(3)]
    for record in records:
        queue.admit(record)

    assert queue.cancel_waiting("job-1") is True
    assert records[2]["queue_position"] == 1
    assert queue.snapshot()["running"] == 1
    assert queue.snapshot()["waiting"] == 1


def test_training_job_snapshot_exposes_admission_and_capacity() -> None:
    queue = TrainingQueueManager(max_running=1, max_waiting=1)
    running = {"job_id": "job-running", "thread_id": "running-thread", "error": None}
    waiting = {"job_id": "job-waiting", "thread_id": "waiting-thread", "error": None}
    rejected = {"job_id": "job-rejected", "thread_id": "rejected-thread", "error": None}

    queue.admit(running)
    queue.admit(waiting)
    queue.admit(rejected)
    rejected["error"] = "Training queue is full."

    running_snapshot = queue.job_snapshot(running)
    waiting_snapshot = queue.job_snapshot(waiting)
    rejected_snapshot = queue.job_snapshot(rejected)

    assert running_snapshot == {
        "schema": "jetlinks-training-job-queue.v1",
        "status": "running",
        "accepted": True,
        "starts_immediately": True,
        "position": None,
        "queued_at": None,
        "max_running": 1,
        "max_waiting": 1,
        "running": 1,
        "waiting": 1,
        "available_running_slots": 0,
        "available_waiting_slots": 0,
        "queue_full": True,
        "reason": None,
        "estimate_type": "completion",
        "estimated_seconds": 3600,
        "updated_at": running_snapshot["updated_at"],
    }
    assert waiting_snapshot["status"] == "waiting"
    assert waiting_snapshot["accepted"] is True
    assert waiting_snapshot["starts_immediately"] is False
    assert waiting_snapshot["position"] == 1
    assert waiting_snapshot["estimate_type"] == "start"
    assert waiting_snapshot["estimated_seconds"] == 3600
    assert rejected_snapshot["status"] == "queue_rejected"
    assert rejected_snapshot["accepted"] is False
    assert rejected_snapshot["starts_immediately"] is False
    assert rejected_snapshot["reason"] == "Training queue is full."
    assert rejected_snapshot["estimate_type"] == "retry"
    assert rejected_snapshot["estimated_seconds"] == 3600


def test_training_queue_uses_persisted_median_after_three_successes(tmp_path) -> None:
    history_path = tmp_path / "training_duration_history.json"
    queue = TrainingQueueManager(
        max_running=1,
        max_waiting=1,
        default_duration_seconds=3600,
        history_path=history_path,
    )

    queue.record_successful_duration(100)
    queue.record_successful_duration(300)
    before_threshold = queue.job_snapshot({"job_id": "pending", "queue_status": None})
    queue.record_successful_duration(200)
    after_threshold = queue.job_snapshot({"job_id": "pending", "queue_status": None})

    assert before_threshold["estimated_seconds"] == 3600
    assert after_threshold["estimated_seconds"] == 200

    restored = TrainingQueueManager(
        max_running=1,
        max_waiting=1,
        default_duration_seconds=3600,
        history_path=history_path,
    )
    restored_snapshot = restored.job_snapshot({"job_id": "restored", "queue_status": None})

    assert restored_snapshot["estimated_seconds"] == 200


def test_training_queue_collects_successful_job_duration_on_finish(tmp_path) -> None:
    queue = TrainingQueueManager(
        max_running=1,
        max_waiting=1,
        default_duration_seconds=3600,
        history_path=tmp_path / "training_duration_history.json",
    )
    durations = [100, 200, 300]
    for index, duration in enumerate(durations):
        record = {
            "job_id": f"job-{index}",
            "thread_id": f"thread-{index}",
            "status": "queued",
        }
        queue.admit(record)
        record.update(
            {
                "status": "completed",
                "started_at": "2026-08-04T00:00:00Z",
                "completed_at": f"2026-08-04T00:{duration // 60:02d}:{duration % 60:02d}Z",
            }
        )
        queue.finish(record["job_id"], promote=False)

    snapshot = queue.job_snapshot({"job_id": "next", "queue_status": None})

    assert snapshot["estimated_seconds"] == 200


def test_waiting_estimate_uses_fifo_execution_waves() -> None:
    queue = TrainingQueueManager(
        max_running=2,
        max_waiting=3,
        default_duration_seconds=100,
    )
    records = [_record(index) for index in range(6)]
    for record in records:
        queue.admit(record)

    assert queue.job_snapshot(records[2])["estimated_seconds"] == 100
    assert queue.job_snapshot(records[3])["estimated_seconds"] == 100
    assert queue.job_snapshot(records[4])["estimated_seconds"] == 200
    assert queue.job_snapshot(records[5])["estimated_seconds"] == 100
    assert queue.job_snapshot(records[5])["estimate_type"] == "retry"


def test_public_training_job_response_contains_queue_snapshot(monkeypatch) -> None:
    from app.api import training

    queue = TrainingQueueManager(max_running=1, max_waiting=1)
    record = {
        "job_id": "job-running",
        "thread_id": "thread-running",
        "agent_name": "default",
        "status": "queued",
        "created_at": "2026-08-04T00:00:00Z",
        "started_at": None,
        "completed_at": None,
        "run_id": None,
        "error": None,
        "status_url": "/api/training/status/thread-running",
        "artifacts_url": "/api/artifacts/thread-running",
        "result": None,
    }
    queue.admit(record)
    monkeypatch.setattr(training, "training_queue", queue)

    payload = training._public_job_record(record)

    assert payload["queue_status"] == "running"
    assert payload["queue"]["status"] == "running"
    assert payload["queue"]["accepted"] is True
    assert payload["queue"]["starts_immediately"] is True
    assert payload["queue"]["max_running"] == 1
    assert payload["queue"]["available_running_slots"] == 0
    assert payload["queue"]["estimate_type"] == "completion"
    assert payload["queue"]["estimated_seconds"] == 3600
    assert "estimated_at" not in payload["queue"]
    assert "estimate_basis" not in payload["queue"]
    assert "estimate_sample_size" not in payload["queue"]
    assert "estimated_task_duration_seconds" not in payload["queue"]


def test_duplicate_active_training_request_returns_existing_job_before_validating_new_parameters(
    monkeypatch,
) -> None:
    from app.api import training

    queue = TrainingQueueManager(max_running=1, max_waiting=1)
    existing = {
        "job_id": "job-existing",
        "thread_id": "same-thread",
        "agent_name": "default",
        "status": "queued",
        "created_at": "2026-08-04T00:00:00Z",
        "started_at": None,
        "completed_at": None,
        "run_id": None,
        "error": None,
        "status_url": "/api/training/status/same-thread",
        "artifacts_url": "/api/artifacts/same-thread",
        "result": None,
    }
    queue.admit(existing)
    audit: dict[str, object] = {}
    monkeypatch.setattr(training, "training_queue", queue)
    monkeypatch.setattr(training, "_jobs_by_thread", {"same-thread": existing})
    monkeypatch.setattr(training, "_jobs_by_id", {"job-existing": existing})
    monkeypatch.setattr(training, "record_http_training_job_request", lambda _thread_id: "request-duplicate")
    monkeypatch.setattr(
        training,
        "mark_http_training_job_request_duplicate",
        lambda thread_id, request_id, **kwargs: audit.update(
            {"thread_id": thread_id, "request_id": request_id, **kwargs}
        ),
    )

    async def invoke() -> dict[str, object]:
        monkeypatch.setattr(training, "_jobs_lock", asyncio.Lock())
        return await training.create_training_job(
            training.TrainingJobRequest(
                threadId="same-thread",
                agentName="agent-that-does-not-exist",
                content="",
                runtimeOptions={"modelName": "ignored-model"},
            )
        )

    payload = asyncio.run(invoke())

    assert payload["job_id"] == "job-existing"
    assert payload["existing_job"] is True
    assert payload["request_action"] == "returned_existing"
    assert payload["request_parameters_applied"] is False
    assert payload["message"] == "Training job already exists for this thread."
    assert payload["queue"]["status"] == "running"
    assert audit == {
        "thread_id": "same-thread",
        "request_id": "request-duplicate",
        "existing_job_id": "job-existing",
        "existing_run_id": None,
    }
