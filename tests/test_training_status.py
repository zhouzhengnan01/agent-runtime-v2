from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.core.training_status import TrainingProgressWriter, build_training_stream_status, chain_event_hooks


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


def test_progress_writer_uses_unique_temp_file_and_hook_errors_do_not_abort(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    writer = TrainingProgressWriter(
        thread_id="thread-a",
        run_id="run-a",
        run_dir=run_dir,
        workflow="yolo_training_flow",
        backend="deimv2",
        app_template_name="app",
        selected_skills=[],
        user_text="train",
    )
    writer.initialize()

    event = SimpleNamespace(type="data_preparation.image_annotated", data={})
    chain_event_hooks(lambda _event: (_ for _ in ()).throw(PermissionError("locked")), writer.handle_event)(event)

    assert (run_dir / "progress_state.json").exists()
    assert not (run_dir / "progress_state.json.tmp").exists()


def test_stream_annotation_metrics_include_synthetic_annotations(tmp_path: Path, monkeypatch) -> None:
    thread_id = "synthetic-annotation-thread"
    run_id = "run-deimv2-synthetic-annotation"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    uploaded = run_dir / "uploaded_dataset" / "datasets"
    synthetic_images = run_dir / "pipeline_work" / "synthetic_images"
    uploaded.mkdir(parents=True)
    synthetic_images.mkdir(parents=True)
    for name in ["real1.jpg", "real2.jpg"]:
        (uploaded / name).write_bytes(b"image")
    for name in ["syn1.png", "syn2.png"]:
        (synthetic_images / name).write_bytes(b"image")
    (run_dir / "pipeline_work").mkdir(parents=True, exist_ok=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "annotation",
                "annotation": {"status": "running", "started_at": "2026-07-16T02:00:00Z"},
                "generation": {"status": "completed", "started_at": "2026-07-16T02:01:00Z", "completed": 2},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "pipeline_work" / "real_coco.json").write_text(
        json.dumps({"images": [{"id": 1}, {"id": 2}]}),
        encoding="utf-8",
    )
    (run_dir / "pipeline_work" / "synthetic_coco.json").write_text(
        json.dumps({"images": [{"id": 3}, {"id": 4}]}),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    status = build_training_stream_status(thread_id, run_id=run_id)

    metrics = status["info"]["metrics"]
    assert metrics["annotated"] == 4
    assert metrics["total"] == 4
    assert metrics["completed"] is True
    assert metrics["real"]["annotated"] == 2
    assert metrics["synthetic"]["annotated"] == 2
