from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.core.http_training_jobs import (
    mark_http_training_job_request_duplicate,
    record_http_training_job_request,
    reject_http_training_job_request,
    update_http_training_job_marker,
    write_http_training_job_marker,
)


def _marker(tmp_path: Path, thread_id: str) -> dict:
    path = tmp_path / ".runtime" / "http_training_jobs" / f"{thread_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_http_training_job_marker_audits_accepted_and_rejected_requests(tmp_path: Path, monkeypatch) -> None:
    thread_id = "audit-thread"
    monkeypatch.chdir(tmp_path)

    accepted_request = record_http_training_job_request(thread_id)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-accepted",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "queued",
            "created_at": "2026-07-24T09:00:00Z",
        },
        request_id=accepted_request,
    )
    update_http_training_job_marker(
        thread_id,
        status="completed",
        started_at="2026-07-24T09:00:01Z",
        completed_at="2026-07-24T09:10:00Z",
        run_id="run-deimv2-audit",
    )

    rejected_request = record_http_training_job_request(thread_id)
    reject_http_training_job_request(
        thread_id,
        rejected_request,
        http_status=409,
        reason="Training job already active.",
    )

    marker = _marker(tmp_path, thread_id)
    assert marker["request_stats"]["received_count"] == 2
    assert marker["request_stats"]["accepted_count"] == 1
    assert marker["request_stats"]["rejected_count"] == 1
    assert marker["request_stats"]["pending_count"] == 0
    assert marker["request_stats"]["last_received_at"]
    assert len(marker["job_history"]) == 2

    accepted, rejected = marker["job_history"]
    assert accepted["request_id"] == accepted_request
    assert accepted["accepted"] is True
    assert accepted["http_status"] == 200
    assert accepted["job_id"] == "job-accepted"
    assert accepted["run_id"] == "run-deimv2-audit"
    assert accepted["status"] == "completed"
    assert accepted["completed_at"] == "2026-07-24T09:10:00Z"
    assert rejected["request_id"] == rejected_request
    assert rejected["accepted"] is False
    assert rejected["http_status"] == 409
    assert rejected["reason"] == "Training job already active."
    assert rejected["status"] == "rejected"


def test_http_training_request_counter_is_safe_across_threads(tmp_path: Path, monkeypatch) -> None:
    thread_id = "concurrent-audit-thread"
    monkeypatch.chdir(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        request_ids = list(executor.map(lambda _index: record_http_training_job_request(thread_id), range(24)))

    marker = _marker(tmp_path, thread_id)
    assert len(set(request_ids)) == 24
    assert marker["request_stats"]["received_count"] == 24
    assert marker["request_stats"]["accepted_count"] == 0
    assert marker["request_stats"]["rejected_count"] == 0
    assert marker["request_stats"]["pending_count"] == 24
    assert len(marker["job_history"]) == 24


def test_new_job_preserves_previous_request_audit(tmp_path: Path, monkeypatch) -> None:
    thread_id = "multi-round-audit-thread"
    monkeypatch.chdir(tmp_path)

    for index in (1, 2):
        request_id = record_http_training_job_request(thread_id)
        write_http_training_job_marker(
            thread_id,
            {
                "job_id": f"job-{index}",
                "thread_id": thread_id,
                "agent_name": "default",
                "status": "queued",
                "created_at": f"2026-07-24T09:0{index}:00Z",
            },
            request_id=request_id,
        )
        update_http_training_job_marker(
            thread_id,
            status="completed",
            completed_at=f"2026-07-24T09:1{index}:00Z",
            run_id=f"run-deimv2-{index}",
        )

    marker = _marker(tmp_path, thread_id)
    assert marker["job_id"] == "job-2"
    assert marker["run_id"] == "run-deimv2-2"
    assert marker["request_stats"]["received_count"] == 2
    assert marker["request_stats"]["accepted_count"] == 2
    assert marker["request_stats"]["rejected_count"] == 0
    assert [item["job_id"] for item in marker["job_history"]] == ["job-1", "job-2"]
    assert [item["run_id"] for item in marker["job_history"]] == ["run-deimv2-1", "run-deimv2-2"]


def test_duplicate_request_is_counted_without_accepting_or_rejecting_new_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    thread_id = "duplicate-audit-thread"
    monkeypatch.chdir(tmp_path)

    first_request = record_http_training_job_request(thread_id)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-existing",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "running",
            "created_at": "2026-08-04T00:00:00Z",
            "run_id": "run-existing",
        },
        request_id=first_request,
    )
    duplicate_request = record_http_training_job_request(thread_id)

    mark_http_training_job_request_duplicate(
        thread_id,
        duplicate_request,
        existing_job_id="job-existing",
        existing_run_id="run-existing",
    )

    marker = _marker(tmp_path, thread_id)
    stats = marker["request_stats"]
    duplicate = marker["job_history"][-1]
    assert stats == {
        "received_count": 2,
        "accepted_count": 1,
        "rejected_count": 0,
        "duplicate_count": 1,
        "pending_count": 0,
        "last_received_at": stats["last_received_at"],
    }
    assert duplicate["accepted"] is True
    assert duplicate["duplicate"] is True
    assert duplicate["created_job"] is False
    assert duplicate["http_status"] == 200
    assert duplicate["status"] == "returned_existing"
    assert duplicate["existing_job_id"] == "job-existing"
    assert duplicate["run_id"] == "run-existing"

    update_http_training_job_marker(
        thread_id,
        status="completed",
        completed_at="2026-08-04T01:00:00Z",
        run_id="run-existing",
    )
    updated_history = _marker(tmp_path, thread_id)["job_history"]

    assert updated_history[0]["status"] == "completed"
    assert updated_history[1]["status"] == "returned_existing"
    assert updated_history[1]["duplicate"] is True
