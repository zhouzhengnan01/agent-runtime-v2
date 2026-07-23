from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.training_status import TrainingProgressWriter, build_training_stream_status
from PIL import Image


def _load_annotation_runner():
    path = ROOT / "plugins" / "skills" / "data-auto-annotation" / "runner.py"
    spec = importlib.util.spec_from_file_location("data_auto_annotation_generation_status_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generation_progress_line_is_converted_to_runtime_event() -> None:
    runner = _load_annotation_runner()

    started = runner._synthetic_generation_progress_event(
        '[data-prep-event] synthetic_generation {"status":"running","planned":5,"completed":0}'
    )
    timeout = runner._synthetic_generation_progress_event(
        '[data-prep-event] synthetic_generation {"status":"timeout","planned":5,"error":"ReadTimeout"}'
    )

    assert started == (
        "data_preparation.synthetic_generation.started",
        {"status": "running", "planned": 5, "completed": 0},
    )
    assert timeout == (
        "data_preparation.synthetic_generation.finished",
        {"status": "timeout", "planned": 5, "error": "ReadTimeout"},
    )


def test_stream_status_reports_generation_phase_and_preserves_timeout(tmp_path: Path) -> None:
    thread_id = "generation-timeout-thread"
    run_id = "run-deimv2-generation-timeout"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    writer = TrainingProgressWriter(
        thread_id=thread_id,
        run_id=run_id,
        run_dir=run_dir,
        workflow="yolo_training_flow",
        backend="deimv2",
        app_template_name="app",
        selected_skills=[],
        user_text="train",
    )
    writer.initialize()
    writer.handle_event(
        SimpleNamespace(
            type="data_preparation.synthetic_generation.started",
            data={"status": "running", "planned": 5, "completed": 0},
        )
    )

    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        running = build_training_stream_status(thread_id, run_id=run_id)
        assert running["phase"] == "synthetic_generation"
        assert running["info"]["stage"] == "generation"
        assert running["info"]["status"] == "running"

        writer.handle_event(
            SimpleNamespace(
                type="data_preparation.synthetic_generation.finished",
                data={
                    "status": "timeout",
                    "planned": 5,
                    "completed": 0,
                    "success": 0,
                    "failed": 5,
                    "error": "ReadTimeout: image model request timed out",
                },
            )
        )
        timed_out = build_training_stream_status(thread_id, run_id=run_id)
        assert timed_out["phase"] == "synthetic_generation"
        assert timed_out["phase_label"] == "synthetic generation timeout"
        assert timed_out["info"]["stage"] == "generation"
        assert timed_out["info"]["status"] == "timeout"
        assert timed_out["generation"]["status"] == "timeout"
        assert timed_out["generation"]["error"].startswith("ReadTimeout")

        writer.handle_event(SimpleNamespace(type="skill.completed", data={"skill_name": "data-auto-annotation"}))
        after_data_prep = build_training_stream_status(thread_id, run_id=run_id)
        assert after_data_prep["generation"]["status"] == "timeout"
    finally:
        os.chdir(previous)


def test_stream_status_reports_sufficient_data_as_completed_skip(tmp_path: Path) -> None:
    thread_id = "generation-skipped-thread"
    run_id = "run-deimv2-generation-skipped"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    writer = TrainingProgressWriter(
        thread_id=thread_id,
        run_id=run_id,
        run_dir=run_dir,
        workflow="yolo_training_flow",
        backend="deimv2",
        app_template_name="app",
        selected_skills=[],
        user_text="train",
    )
    writer.initialize()
    writer.handle_event(
        SimpleNamespace(
            type="data_preparation.synthetic_generation.finished",
            data={
                "status": "skipped",
                "planned": 0,
                "completed": 0,
                "success": 0,
                "failed": 0,
                "reason_code": "sufficient_data_samples",
                "reason": "Data samples are sufficient.",
            },
        )
    )

    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        skipped = build_training_stream_status(thread_id, run_id=run_id)
        assert skipped["phase"] == "synthetic_generation"
        assert skipped["generation"]["status"] == "skipped"
        assert skipped["generation"]["reason_code"] == "sufficient_data_samples"
        assert skipped["generation"]["reason"] == "Data samples are sufficient."
        assert skipped["info"]["status"] == "skipped"
        assert skipped["info"]["metrics"]["completed"] is True
        assert skipped["info"]["metrics"]["reason_code"] == "sufficient_data_samples"
        assert skipped["info"]["metrics"]["reason"] == "Data samples are sufficient."
    finally:
        os.chdir(previous)


def test_data_preparation_skips_generation_without_complete_pair(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "images" / "real.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), color="white").save(image_path)
    coco_path = dataset_root / "annotations" / "coco.json"
    coco_path.parent.mkdir(parents=True)
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "real.jpg", "width": 32, "height": 24}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 8, 9], "area": 72}],
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "coco_json": str(coco_path),
                "labels": ["person"],
                "work_dir": str(tmp_path / "work"),
                "output_dir": str(output_dir),
                "skip_generation": True,
                "generation_skip_reason": "sufficient_data_samples",
                "split_requested": False,
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "plugins" / "skills" / "data-auto-annotation" / "scripts" / "run_data_preparation_pipeline.py"),
            "--input-json",
            str(request_path),
        ],
        cwd=str(ROOT),
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 0, stderr or stdout
    assert '"status": "skipped"' in stdout
    summary = json.loads((output_dir / "data_preparation_summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic_generation_status"] == "skipped_sufficient_data"
    assert summary["synthetic_generation_skip_reason"] == "sufficient_data_samples"
    assert summary["synthetic_generation_skip_description"] == "Data samples are sufficient."
    assert summary["synthetic_pending_inputs"] is False
    assert Path(summary["training_coco"]).resolve() == coco_path.resolve()
    assert not (tmp_path / "work" / "synthetic_images").exists()


if __name__ == "__main__":
    test_generation_progress_line_is_converted_to_runtime_event()
    with tempfile.TemporaryDirectory() as directory:
        test_stream_status_reports_generation_phase_and_preserves_timeout(Path(directory))
    print("synthetic generation status assertions passed")
