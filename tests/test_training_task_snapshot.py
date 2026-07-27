from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from app.core import training_status


def _write_run(
    thread_dir: Path,
    run_id: str,
    *,
    status: str,
    phase: str,
    timestamp: float,
) -> Path:
    run_dir = thread_dir / "outputs" / "yolo_training_flow" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_dir.name,
                "run_id": run_id,
                "workflow": "yolo_training_flow",
                "backend": "deimv2",
                "status": status,
                "phase": phase,
                "phase_label": f"{phase} label",
                "created_at": f"2026-07-23T00:00:0{int(timestamp)}Z",
                "started_at": f"2026-07-23T00:00:0{int(timestamp)}Z",
                "updated_at": f"2026-07-23T00:01:0{int(timestamp)}Z",
                "completed_at": f"2026-07-23T00:02:0{int(timestamp)}Z" if status == "completed" else None,
            }
        ),
        encoding="utf-8",
    )
    os.utime(run_dir, (timestamp, timestamp))
    return run_dir


def test_training_status_includes_thread_task_snapshot(tmp_path: Path, monkeypatch) -> None:
    threads_root = tmp_path / "threads"
    thread_dir = threads_root / "face"
    first = _write_run(
        thread_dir,
        "run-deimv2-first",
        status="completed",
        phase="completed",
        timestamp=1,
    )
    second = _write_run(
        thread_dir,
        "run-deimv2-second",
        status="running",
        phase="annotation",
        timestamp=2,
    )
    (first / "deimv2_training_run").mkdir()
    (first / "deimv2_training_run" / "old.onnx").write_bytes(b"old")
    real_coco = second / "pipeline_work" / "real_coco.json"
    real_coco.parent.mkdir(parents=True)
    real_image = real_coco.parent / "face.jpg"
    Image.new("RGB", (1920, 1080), color="white").save(real_image)
    real_coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "face.jpg"}],
                "annotations": [],
                "categories": [{"id": 1, "name": "face"}],
            }
        ),
        encoding="utf-8",
    )
    uploaded_dir = second / "uploaded_dataset"
    uploaded_dir.mkdir()
    uploaded_image = uploaded_dir / "uploaded_face.jpg"
    Image.new("RGB", (1024, 768), color="white").save(uploaded_image)
    uploaded_coco = uploaded_dir / "uploaded_face_coco.json"
    uploaded_coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": uploaded_image.name}],
                "annotations": [],
                "categories": [{"id": 1, "name": "face"}],
            }
        ),
        encoding="utf-8",
    )
    synthetic_dir = second / "pipeline_work" / "synthetic_images"
    synthetic_dir.mkdir(parents=True)
    synthetic_image = synthetic_dir / "synthetic_001.jpg"
    Image.new("RGB", (1280, 720), color="gray").save(synthetic_image)
    synthetic_coco = synthetic_dir / "synthetic_001_coco.json"
    synthetic_coco.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": synthetic_image.name}],
                "annotations": [],
                "categories": [{"id": 1, "name": "face"}],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = second / "deimv2_training_run" / "runs" / "train" / "best_stg1.onnx"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"onnx")
    os.utime(first, (1, 1))
    os.utime(second, (2, 2))

    monkeypatch.setattr(training_status, "runtime_root", lambda: threads_root)
    monkeypatch.setattr(training_status, "_resource_status", lambda *_args: {})

    status = training_status.build_training_task_status("face")
    snapshot = status["task_snapshot"]

    assert status["schema"] == "jetlinks-training-task-status.v1"
    assert status["status"] == "running"
    assert status["phase"] == "annotation"
    assert "run_id" not in status
    assert "annotation" not in status
    assert "generation" not in status
    assert "training" not in status
    assert snapshot["total_runs"] == 2
    assert snapshot["latest_run_id"] == "run-deimv2-second"
    assert snapshot["current_run_id"] == "run-deimv2-second"
    assert snapshot["selected_run_id"] == "run-deimv2-second"
    assert snapshot["current_run"]["status"] == "running"
    assert snapshot["current_run"]["phase"] == "annotation"
    assert "annotation" in snapshot["current_run"]
    assert "generation" in snapshot["current_run"]
    assert "training" in snapshot["current_run"]
    assert snapshot["current_run"]["has_artifacts"] is True
    assert snapshot["current_run"]["artifact_count"] == 5
    artifacts = {artifact["name"]: artifact for artifact in snapshot["current_run"]["artifacts"]}
    assert artifacts["real_coco.json"]["preview_url"].startswith("/api/artifacts/face/")
    assert artifacts["real_coco.json"]["run_id"] == "run-deimv2-second"
    assert artifacts["real_coco.json"]["run_sequence"] == 2
    assert artifacts["real_coco.json"]["phase"] == "real_annotation"
    assert artifacts["real_coco.json"]["phase_label"] == "real image annotation"
    assert "image_download_url" not in artifacts["real_coco.json"]
    assert "annotationPreview" not in artifacts["uploaded_face_coco.json"]
    assert artifacts["uploaded_face_coco.json"]["image_download_url"].startswith("/api/artifacts/face/")
    assert artifacts["uploaded_face_coco.json"]["image_download_url"].endswith("?download=true")
    assert artifacts["synthetic_001.jpg"]["phase"] == "synthetic_generation"
    assert artifacts["synthetic_001_coco.json"]["phase"] == "synthetic_annotation"
    assert "annotationPreview" not in artifacts["synthetic_001_coco.json"]
    assert artifacts["synthetic_001_coco.json"]["image_download_url"].startswith("/api/artifacts/face/")
    assert artifacts["synthetic_001_coco.json"]["image_download_url"].endswith("?download=true")
    assert artifacts["best_stg1.onnx"]["phase"] == "training"
    assert "runs" not in snapshot
    assert [run["run_sequence"] for run in snapshot["history_runs"]] == [1, 2]
    assert all(set(run) == set(snapshot["current_run"]) for run in snapshot["history_runs"])
    historical = snapshot["history_runs"][0]
    assert historical["run_id"] == "run-deimv2-first"
    assert historical["has_artifacts"] is True
    assert historical["artifact_count"] == 1
    assert historical["artifacts"][0]["name"] == "old.onnx"
    assert historical["artifacts"][0]["run_id"] == "run-deimv2-first"
    assert historical["artifacts"][0]["run_sequence"] == 1
    assert snapshot["history_runs"][1] == snapshot["current_run"]
    assert snapshot["status_counts"] == {"completed": 1, "running": 1}


def test_historical_run_query_keeps_snapshot_on_latest_run(tmp_path: Path, monkeypatch) -> None:
    threads_root = tmp_path / "threads"
    thread_dir = threads_root / "face"
    _write_run(
        thread_dir,
        "run-deimv2-first",
        status="completed",
        phase="completed",
        timestamp=1,
    )
    _write_run(
        thread_dir,
        "run-deimv2-second",
        status="running",
        phase="training",
        timestamp=2,
    )
    monkeypatch.setattr(training_status, "runtime_root", lambda: threads_root)
    monkeypatch.setattr(training_status, "_resource_status", lambda *_args: {})

    status = training_status.build_training_task_status("face", run_id="run-deimv2-first")

    assert status["task_snapshot"]["selected_run_id"] == "run-deimv2-first"
    assert status["task_snapshot"]["current_run_id"] == "run-deimv2-second"
    assert status["task_snapshot"]["current_run"]["run_id"] == "run-deimv2-second"
    assert status["task_snapshot"]["current_run"]["status"] == "running"
    assert status["task_snapshot"]["current_run"]["has_artifacts"] is False
    assert status["task_snapshot"]["current_run"]["artifacts"] == []
    assert all(
        set(run) == set(status["task_snapshot"]["current_run"])
        for run in status["task_snapshot"]["history_runs"]
    )
