from __future__ import annotations

import json
from pathlib import Path

from app.core.training_status import build_training_stream_status


def test_deimv2_openblas_fatal_log_marks_stream_failed(tmp_path: Path, monkeypatch) -> None:
    thread_id = "fatal-log-thread"
    run_id = "run-deimv2-fatal"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    log_dir = run_dir / "deimv2_training_run" / "logs"
    log_dir.mkdir(parents=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "training",
                "training": {"status": "running", "started_at": "2026-07-15T08:00:00Z"},
            }
        ),
        encoding="utf-8",
    )
    (log_dir / "train.log").write_text(
        "Start training\nOpenBLAS error: Memory allocation still failed after 10 retries, giving up.\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    status = build_training_stream_status(thread_id, run_id=run_id)

    assert status["status"] == "failed"
    assert status["phase"] == "failed"
    assert status["info"]["status"] == "failed"
    assert "OpenBLAS error" in status["errors"][0]["message"]


def test_deimv2_python_exception_log_marks_stream_failed(tmp_path: Path, monkeypatch) -> None:
    thread_id = "exception-log-thread"
    run_id = "run-deimv2-exception"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    log_dir = run_dir / "deimv2_training_run" / "logs"
    log_dir.mkdir(parents=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "training",
                "training": {"status": "running", "started_at": "2026-07-15T08:00:00Z"},
            }
        ),
        encoding="utf-8",
    )
    (log_dir / "train.log").write_text(
        "Traceback (most recent call last):\n  File \"train.py\", line 1, in <module>\nValueError: invalid dataset annotation\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    status = build_training_stream_status(thread_id, run_id=run_id)

    assert status["status"] == "failed"
    assert status["phase"] == "failed"
    assert status["info"]["status"] == "failed"
    assert "ValueError" in status["errors"][0]["message"] or "Traceback" in status["errors"][0]["message"]
