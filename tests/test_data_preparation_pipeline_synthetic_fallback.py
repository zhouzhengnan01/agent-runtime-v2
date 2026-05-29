from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_continues_with_real_dataset_when_synthetic_generation_fails(tmp_path: Path, monkeypatch) -> None:
    module = _load_pipeline_module()
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    work_dir = tmp_path / "pipeline_work"
    output_dir = tmp_path / "prepared_data"
    image1 = tmp_path / "image1"
    image2 = tmp_path / "image2"
    images_dir.mkdir(parents=True)
    image1.mkdir()
    image2.mkdir()
    real_coco = work_dir / "real_coco.json"
    plan_path = work_dir / "synthetic_plan.json"

    def fake_inspect(dataset_root_arg: Path, work_dir_arg: Path, dry_run: bool) -> dict:
        del dataset_root_arg, work_dir_arg, dry_run
        return {"status": "ok", "format": "unlabeled", "images_dir": str(images_dir)}

    def fake_prepare_real_coco(inspection: dict, work_dir_arg: Path, task_labels: list[str], dry_run: bool) -> Path:
        del inspection, task_labels, dry_run
        work_dir_arg.mkdir(parents=True, exist_ok=True)
        real_coco.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
        return real_coco

    def fake_plan(*args, **kwargs) -> Path:
        del args, kwargs
        plan_path.write_text('{"recommended_synthetic_count":1,"generation_plan":[{"count":1,"prompt":"p"}]}', encoding="utf-8")
        return plan_path

    def fake_generate(*args, **kwargs):
        del args, kwargs
        raise subprocess.CalledProcessError(
            1,
            ["run_composite.py"],
            output=b'{"error":{"message":"API-key is blocked."}}',
            stderr=b"",
        )

    def fake_export(coco_path: Path, image_root: Path, output_dir_arg: Path, dry_run: bool) -> Path:
        del dry_run
        assert coco_path == real_coco
        assert image_root == images_dir
        combined = output_dir_arg / "combined_dataset"
        combined.mkdir(parents=True)
        return combined

    monkeypatch.setattr(module, "_inspect", fake_inspect)
    monkeypatch.setattr(module, "_prepare_real_coco", fake_prepare_real_coco)
    monkeypatch.setattr(module, "_plan_synthetic", fake_plan)
    monkeypatch.setattr(module, "_read_recommended_synthetic_count", lambda path: 1)
    monkeypatch.setattr(module, "_generate_and_annotate_synthetic", fake_generate)
    monkeypatch.setattr(module, "_export_combined_dataset", fake_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_data_preparation_pipeline.py",
            "--dataset-root",
            str(dataset_root),
            "--labels",
            "person",
            "--work-dir",
            str(work_dir),
            "--output-dir",
            str(output_dir),
            "--image1",
            str(image1),
            "--image2",
            str(image2),
        ],
    )

    module.main()

    summary = json.loads((output_dir / "data_preparation_summary.json").read_text(encoding="utf-8"))
    assert summary["training_coco"] == str(real_coco)
    assert summary["training_root"] == str(images_dir)
    assert summary["synthetic_generation_status"] == "failed"
    assert summary["synthetic_generation_fallback"] == "real_dataset_only"
    assert "API-key is blocked" in summary["synthetic_generation_error"]


def _load_pipeline_module() -> object:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "plugins" / "skills" / "data-auto-annotation" / "scripts" / "run_data_preparation_pipeline.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("run_data_preparation_pipeline_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
