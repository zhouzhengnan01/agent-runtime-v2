from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_continues_with_real_dataset_when_composite_and_produce_fail(tmp_path: Path, monkeypatch) -> None:
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
    assert "API-key is blocked" in summary["synthetic_generation_primary_error"]


def test_pipeline_falls_back_to_image_dataset_produce_when_composite_quota_fails(tmp_path: Path, monkeypatch) -> None:
    module = _load_pipeline_module()
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    work_dir = tmp_path / "pipeline_work"
    output_dir = tmp_path / "prepared_data"
    image1 = tmp_path / "image1"
    image2 = tmp_path / "image2"
    synthetic_root = work_dir / "synthetic_images"
    merged_root = work_dir / "merged_images"
    images_dir.mkdir(parents=True)
    image1.mkdir()
    image2.mkdir()
    (image1 / "ref.png").write_bytes(b"fake")
    real_coco = work_dir / "real_coco.json"
    synthetic_coco = work_dir / "synthetic_coco.json"
    merged_coco = work_dir / "merged_coco.json"
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
            output=b'{"error":{"message":"quota insufficient"}}',
            stderr=b"",
        )

    def fake_produce(
        plan_arg: Path,
        image1_arg: Path,
        labels: list[str],
        output_dir_arg: Path,
        dry_run: bool,
        planner_llm: dict,
        produce_count_button: bool,
        fixed_produce_synthetic_count: int,
    ):
        assert plan_arg == plan_path
        assert image1_arg == image1.resolve()
        assert labels == ["person"]
        assert output_dir_arg == work_dir.resolve()
        assert dry_run is False
        assert planner_llm == {}
        assert produce_count_button is True
        assert fixed_produce_synthetic_count == 5
        synthetic_root.mkdir(parents=True)
        synthetic_coco.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
        return synthetic_root, synthetic_coco

    def fake_merge(real_coco_arg: Path, synthetic_coco_arg: Path, real_root: Path, synthetic_root_arg: Path, output_dir_arg: Path, dry_run: bool) -> Path:
        assert real_coco_arg == real_coco
        assert synthetic_coco_arg == synthetic_coco
        assert real_root == images_dir.resolve()
        assert synthetic_root_arg == synthetic_root
        assert output_dir_arg == work_dir.resolve()
        assert dry_run is False
        merged_root.mkdir(parents=True)
        merged_coco.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
        return merged_coco

    def fake_export(coco_path: Path, image_root: Path, output_dir_arg: Path, dry_run: bool) -> Path:
        del dry_run
        assert coco_path == merged_coco
        assert image_root == merged_root
        combined = output_dir_arg / "combined_dataset"
        combined.mkdir(parents=True)
        return combined

    monkeypatch.setattr(module, "_inspect", fake_inspect)
    monkeypatch.setattr(module, "_prepare_real_coco", fake_prepare_real_coco)
    monkeypatch.setattr(module, "_plan_synthetic", fake_plan)
    monkeypatch.setattr(module, "_read_recommended_synthetic_count", lambda path: 1)
    monkeypatch.setattr(module, "_generate_and_annotate_synthetic", fake_generate)
    monkeypatch.setattr(module, "_generate_and_annotate_synthetic_with_produce", fake_produce)
    monkeypatch.setattr(module, "_merge", fake_merge)
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
    assert summary["training_coco"] == str(merged_coco)
    assert summary["training_root"] == str(merged_root)
    assert summary["synthetic_generation_status"] == "merged"
    assert summary["synthetic_generation_fallback"] == "image-dataset-produce"
    assert "quota insufficient" in summary["synthetic_generation_primary_error"]


def test_image_dataset_produce_count_uses_fixed_value_when_button_is_false(tmp_path: Path) -> None:
    module = _load_pipeline_module()
    plan = {"recommended_synthetic_count": 100, "generation_plan": [{"count": 100, "prompt": "p"}]}

    count = module._produce_synthetic_count(plan, reference_count=2, produce_count_button=False, fixed_produce_synthetic_count=3)

    assert count == 3


def test_image_dataset_produce_count_uses_model_plan_when_button_is_true() -> None:
    module = _load_pipeline_module()
    plan = {"recommended_synthetic_count": 7, "generation_plan": [{"count": 7, "prompt": "p"}]}

    count = module._produce_synthetic_count(plan, reference_count=2, produce_count_button=True, fixed_produce_synthetic_count=3)

    assert count == 7


def test_image_dataset_produce_fallback_randomly_samples_image1_references(tmp_path: Path, monkeypatch) -> None:
    module = _load_pipeline_module()
    image1 = tmp_path / "image1"
    image1.mkdir()
    refs = [image1 / "a.jpg", image1 / "b.jpg", image1 / "c.jpg"]
    for ref in refs:
        ref.write_bytes(b"fake")
    work_dir = tmp_path / "work"
    plan_path = work_dir / "synthetic_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(
            {
                "task_description": "Train face detector",
                "class_names": ["face"],
                "recommended_synthetic_count": 10,
                "generation_plan": [{"count": 10, "prompt": "p"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    generated_images: list[Path] = []

    def fake_run(cmd: list[str], dry_run: bool = False, *, retries: int = 1, retry_sleep: float = 5.0) -> str:
        del cmd, dry_run, retries, retry_sleep
        out = work_dir / "synthetic_images" / f"generated_{len(generated_images) + 1}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"image")
        generated_images.append(out)
        return f"result_image_path: {out}\n"

    def fake_annotate(image_path: Path, labels: list[str], output_json: Path, dry_run: bool) -> Path:
        del image_path, labels, dry_run
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
        return output_json

    def fake_combine(coco_files: list[Path], output_path: Path) -> Path:
        output_path.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
        return output_path

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_annotate_one_synthetic", fake_annotate)
    monkeypatch.setattr(module, "_combine_coco_files", fake_combine)

    module._generate_and_annotate_synthetic_with_produce(
        plan_path,
        image1,
        ["face"],
        work_dir,
        False,
        {},
        False,
        4,
        random_seed=2,
    )

    input_files = sorted((work_dir / "generation_inputs").glob("produce_*.json"))
    input_images = [json.loads(path.read_text(encoding="utf-8"))["task"]["input_image"] for path in input_files]

    assert len(input_files) == 4
    assert all(Path(value).name in {"a.jpg", "b.jpg", "c.jpg"} for value in input_images)
    assert len(set(input_images)) > 1


def test_image_dataset_produce_fallback_rewrites_composite_prompts() -> None:
    module = _load_pipeline_module()
    plan = {
        "task_description": "Train a YOLO face detection model for surveillance images.",
        "class_names": ["face"],
        "recommended_synthetic_count": 2,
        "generation_plan": [
            {
                "count": 2,
                "prompt": (
                    "Take the images from image2.zip and naturally composite them into image1.zip. "
                    "Ensure realistic integration by matching lighting."
                ),
            }
        ],
    }

    prompts = module._fallback_prompt_sequence(plan, ["face"], 2, {})

    assert len(prompts) >= 2
    assert all("image2" not in prompt.lower() for prompt in prompts)
    assert all("composite" not in prompt.lower() for prompt in prompts)
    assert all("Generate a new realistic image" in prompt for prompt in prompts)
    assert all("not an identical copy" in prompt for prompt in prompts)
    assert all("face" in prompt for prompt in prompts)


def test_image_dataset_produce_runner_accepts_task_only_payload(tmp_path: Path) -> None:
    module = _load_produce_runner_module()
    image_path = tmp_path / "ref.png"
    output_dir = tmp_path / "out"
    image_path.write_bytes(b"fake")

    normalized = module._normalize_runtime_payload(
        {
            "task": {
                "input_image": str(image_path),
                "prompt": "generate detection sample",
                "output_dir": str(output_dir),
            }
        }
    )

    assert normalized["task"]["input_image"] == str(image_path)
    assert normalized["task"]["prompt"] == "generate detection sample"
    assert normalized["task"]["output_dir"] == str(output_dir)
    assert normalized["model"]["api_url"]


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


def _load_produce_runner_module() -> object:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "plugins" / "skills" / "image-dataset-produce" / "scripts" / "run_generation.py"
    spec = importlib.util.spec_from_file_location("image_dataset_produce_run_generation_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
