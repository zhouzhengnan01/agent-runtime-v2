from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.artifacts import ArtifactStore
from app.core.training_artifact_updates import build_training_artifact_session_updates
from app.core.training_status import TrainingProgressWriter, _stream_info
from app.schemas import ChatEvent
from PIL import Image


def _load_flow():
    path = ROOT / "plugins" / "workflows" / "builtin-artifact-workflows" / "yolo_training_flow.py"
    spec = importlib.util.spec_from_file_location("yolo_training_flow_data_reuse_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_coco(path: Path, images: list[dict], annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "person"}],
            }
        ),
        encoding="utf-8",
    )


def _build_source_run(flow, store: ArtifactStore, run_id: str, fingerprint: dict):
    thread = store.prepare_thread("reuse-thread")
    run = flow.TrainingRunPaths(
        thread=thread,
        run_id=run_id,
        workspace=thread.workspace / "runs" / run_id,
        outputs=thread.outputs / "yolo_training_flow" / "runs" / run_id,
    )
    run.workspace.mkdir(parents=True)
    real_root = run.outputs / "uploaded_dataset"
    synthetic_root = run.outputs / "pipeline_work" / "synthetic_images"
    merged_root = run.outputs / "pipeline_work" / "merged_images"
    synthetic_annotations = run.outputs / "pipeline_work" / "synthetic_annotations"
    for directory in (real_root, synthetic_root, merged_root / "real", merged_root / "synthetic", synthetic_annotations):
        directory.mkdir(parents=True, exist_ok=True)
    for path, color in (
        (real_root / "real.jpg", "red"),
        (synthetic_root / "synthetic.jpg", "blue"),
        (merged_root / "real" / "real.jpg", "red"),
        (merged_root / "synthetic" / "synthetic.jpg", "blue"),
    ):
        Image.new("RGB", (32, 32), color=color).save(path)
    real_image = {"id": 1, "file_name": "real.jpg", "width": 32, "height": 32, "source": "real"}
    synthetic_image = {"id": 1, "file_name": "synthetic.jpg", "width": 32, "height": 32, "source": "synthetic"}
    _write_coco(real_root / "real_coco.json", [real_image], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}])
    _write_coco(synthetic_annotations / "synthetic_coco.json", [synthetic_image], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}])
    _write_coco(run.outputs / "pipeline_work" / "real_coco.json", [real_image], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}])
    _write_coco(run.outputs / "pipeline_work" / "synthetic_coco.json", [synthetic_image], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}])
    _write_coco(
        run.outputs / "pipeline_work" / "merged_coco.json",
        [
            {**real_image, "id": 1, "file_name": "real/real.jpg"},
            {**synthetic_image, "id": 2, "file_name": "synthetic/synthetic.jpg"},
        ],
        [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 1, 2, 2]},
        ],
    )
    (run.outputs / "pipeline_work" / "synthetic_plan.json").write_text("{}", encoding="utf-8")
    flow._write_data_preparation_reuse_manifest(run, fingerprint=fingerprint, source_run_id="")
    return run


def _build_real_only_source_run(flow, store: ArtifactStore, run_id: str, fingerprint: dict):
    thread = store.prepare_thread("reuse-thread")
    run = flow.TrainingRunPaths(
        thread=thread,
        run_id=run_id,
        workspace=thread.workspace / "runs" / run_id,
        outputs=thread.outputs / "yolo_training_flow" / "runs" / run_id,
    )
    run.workspace.mkdir(parents=True)
    real_root = run.outputs / "uploaded_dataset"
    real_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color="red").save(real_root / "real.jpg")
    real_image = {"id": 1, "file_name": "real.jpg", "width": 32, "height": 32, "source": "real"}
    annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}]
    _write_coco(real_root / "real_coco.json", [real_image], annotations)
    _write_coco(run.outputs / "pipeline_work" / "real_coco.json", [real_image], annotations)
    flow._write_data_preparation_reuse_manifest(run, fingerprint=fingerprint, source_run_id="")
    return run


def test_complete_previous_data_preparation_is_copied_into_new_run(tmp_path: Path) -> None:
    flow = _load_flow()
    store = ArtifactStore(root_dir=tmp_path / "threads")
    fingerprint = {"digest": "same-input", "inputs": {"labels": ["person"], "generation_enabled": True}}
    source = _build_source_run(flow, store, "run-deimv2-source", fingerprint)
    thread = source.thread
    target = flow.TrainingRunPaths(
        thread=thread,
        run_id="run-deimv2-target",
        workspace=thread.workspace / "runs" / "run-deimv2-target",
        outputs=thread.outputs / "yolo_training_flow" / "runs" / "run-deimv2-target",
    )
    target.workspace.mkdir(parents=True)
    (target.outputs / "uploaded_dataset").mkdir(parents=True)
    (target.outputs / "uploaded_dataset" / "stale.jpg").write_bytes(b"stale")

    reused, reason = flow._reuse_previous_data_preparation(
        source=source,
        target=target,
        fingerprint=fingerprint,
        mode="auto",
    )

    assert reason == "reused"
    assert reused is not None
    assert reused.source_run_id == source.run_id
    assert reused.real_images == 1
    assert reused.synthetic_images == 1
    assert reused.reuse_level == "real_and_synthetic"
    assert reused.synthetic_reused is True
    assert reused.coco_json == (target.outputs / "pipeline_work" / "merged_coco.json").resolve()
    assert (target.outputs / "uploaded_dataset" / "real.jpg").is_file()
    assert not (target.outputs / "uploaded_dataset" / "stale.jpg").exists()
    assert (source.outputs / "uploaded_dataset" / "real.jpg").is_file()

    status = {
        "annotation": {
            "real": {"output_coco": str(target.outputs / "pipeline_work" / "real_coco.json")},
            "synthetic": {"output_coco": str(target.outputs / "pipeline_work" / "synthetic_coco.json")},
        },
        "generation": {"output_dir": str(target.outputs / "pipeline_work" / "synthetic_images")},
        "paths": {"run_dir": str(target.outputs)},
    }
    updates = build_training_artifact_session_updates(
        thread_id=thread.thread_id,
        status=status,
        artifact_store=store,
        published_artifacts=set(),
    )
    assert updates
    assert all("run-deimv2-target" in update["artifact"]["path"] for update in updates)
    assert any(update["artifact"]["name"] == "real_coco.json" for update in updates)
    assert any(update["artifact"]["name"] == "synthetic.jpg" for update in updates)

    pipeline_input = target.workspace / "reuse-pipeline-input.json"
    pipeline_input.write_text(
        json.dumps(
            {
                "dataset_root": str(reused.dataset_root),
                "coco_json": str(reused.coco_json),
                "labels": ["person"],
                "work_dir": str(target.outputs / "pipeline_work"),
                "output_dir": str(target.outputs / "prepared_data"),
                "skip_generation": True,
                "reuse_synthetic_data": True,
                "split_requested": True,
                "split": {"train": 0.7, "val": 0.2, "test": 0.1},
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "plugins" / "skills" / "data-auto-annotation" / "scripts" / "run_data_preparation_pipeline.py"),
            "--input-json",
            str(pipeline_input),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    summary = json.loads((target.outputs / "prepared_data" / "data_preparation_summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic_policy"]["enabled"] is True
    assert Path(summary["training_coco"]).resolve() == reused.coco_json
    assert Path(summary["prepared_dataset"]).is_dir()


def test_fingerprint_mismatch_falls_back_without_copying(tmp_path: Path) -> None:
    flow = _load_flow()
    store = ArtifactStore(root_dir=tmp_path / "threads")
    source = _build_source_run(
        flow,
        store,
        "run-deimv2-source",
        {"digest": "old", "inputs": {"dataset_sha256": "old", "labels": ["person"]}},
    )
    target = flow.TrainingRunPaths(
        thread=source.thread,
        run_id="run-deimv2-target",
        workspace=source.thread.workspace / "runs" / "run-deimv2-target",
        outputs=source.thread.outputs / "yolo_training_flow" / "runs" / "run-deimv2-target",
    )
    target.workspace.mkdir(parents=True)

    reused, reason = flow._reuse_previous_data_preparation(
        source=source,
        target=target,
        fingerprint={"digest": "new", "inputs": {"dataset_sha256": "new", "labels": ["person"]}},
        mode="auto",
    )

    assert reused is None
    assert reason == "real_annotation_fingerprint_mismatch"
    assert not (target.outputs / "pipeline_work" / "merged_coco.json").exists()


def test_real_only_previous_run_is_reused_when_generation_is_skipped(tmp_path: Path) -> None:
    flow = _load_flow()
    store = ArtifactStore(root_dir=tmp_path / "threads")
    fingerprint = {"digest": "real-only", "inputs": {"labels": ["person"], "generation_enabled": False}}
    source = _build_real_only_source_run(flow, store, "run-deimv2-source", fingerprint)
    target = flow.TrainingRunPaths(
        thread=source.thread,
        run_id="run-deimv2-target",
        workspace=source.thread.workspace / "runs" / "run-deimv2-target",
        outputs=source.thread.outputs / "yolo_training_flow" / "runs" / "run-deimv2-target",
    )
    target.workspace.mkdir(parents=True)

    reused, reason = flow._reuse_previous_data_preparation(
        source=source,
        target=target,
        fingerprint=fingerprint,
        mode="auto",
    )

    assert reason == "reused"
    assert reused is not None
    assert reused.reuse_level == "real_only"
    assert reused.synthetic_reused is False
    assert reused.synthetic_images == 0
    assert reused.coco_json == (target.outputs / "pipeline_work" / "real_coco.json").resolve()
    assert reused.dataset_root == (target.outputs / "uploaded_dataset").resolve()
    assert not (target.outputs / "pipeline_work" / "merged_coco.json").exists()


def test_failed_synthetic_run_reuses_real_annotations_and_retries_generation(tmp_path: Path) -> None:
    flow = _load_flow()
    store = ArtifactStore(root_dir=tmp_path / "threads")
    fingerprint = {"digest": "generation-failed", "inputs": {"labels": ["person"], "generation_enabled": True}}
    source = _build_real_only_source_run(flow, store, "run-deimv2-source", fingerprint)
    manifest_path = source.outputs / flow.DATA_REUSE_MANIFEST_NAME
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest.update({"reusable": False, "reason": "real_or_synthetic_artifacts_incomplete"})
    manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
    target = flow.TrainingRunPaths(
        thread=source.thread,
        run_id="run-deimv2-target",
        workspace=source.thread.workspace / "runs" / "run-deimv2-target",
        outputs=source.thread.outputs / "yolo_training_flow" / "runs" / "run-deimv2-target",
    )
    target.workspace.mkdir(parents=True)

    reused, reason = flow._reuse_previous_data_preparation(
        source=source,
        target=target,
        fingerprint=fingerprint,
        mode="auto",
    )

    assert reason == "reused"
    assert reused is not None
    assert reused.reuse_level == "real_only"
    assert reused.synthetic_reused is False
    generation_enabled = True
    skip_generation = reused.synthetic_reused or not generation_enabled
    assert skip_generation is False


def test_real_annotations_are_reused_when_generation_becomes_enabled(tmp_path: Path) -> None:
    flow = _load_flow()
    store = ArtifactStore(root_dir=tmp_path / "threads")
    previous_fingerprint = {
        "digest": "legacy-no-generation",
        "inputs": {
            "dataset_sha256": "same-dataset",
            "labels": ["person"],
            "annotation_provider": "locate_sam3",
            "generation_enabled": False,
            "image1_sha256": "",
            "image2_sha256": "",
            "max_synthetic_images": 5,
        },
    }
    current_fingerprint = {
        "digest": "generation-enabled",
        "inputs": {
            "dataset_sha256": "same-dataset",
            "labels": ["person"],
            "annotation_provider": "locate_sam3",
            "generation_enabled": True,
            "image1_sha256": "image-one",
            "image2_sha256": "image-two",
            "max_synthetic_images": 5,
        },
    }
    source = _build_real_only_source_run(flow, store, "run-deimv2-source", previous_fingerprint)
    target = flow.TrainingRunPaths(
        thread=source.thread,
        run_id="run-deimv2-target",
        workspace=source.thread.workspace / "runs" / "run-deimv2-target",
        outputs=source.thread.outputs / "yolo_training_flow" / "runs" / "run-deimv2-target",
    )
    target.workspace.mkdir(parents=True)

    reused, reason = flow._reuse_previous_data_preparation(
        source=source,
        target=target,
        fingerprint=current_fingerprint,
        mode="auto",
    )

    assert reason == "reused"
    assert reused is not None
    assert reused.reuse_level == "real_only"
    assert reused.synthetic_reused is False
    assert reused.coco_json == (target.outputs / "pipeline_work" / "real_coco.json").resolve()
    assert not (target.outputs / "pipeline_work" / "synthetic_coco.json").exists()
    provenance = json.loads((target.outputs / "data_reuse_provenance.json").read_text(encoding="utf-8"))
    assert provenance["real_fingerprint_matched"] is True
    assert provenance["synthetic_fingerprint_matched"] is False
    assert provenance["synthetic_reuse_reason"] == "synthetic_artifacts_incomplete"


def test_real_and_synthetic_fingerprints_are_independent(tmp_path: Path) -> None:
    flow = _load_flow()
    dataset = tmp_path / "datasets.zip"
    image1 = tmp_path / "image1.zip"
    image2 = tmp_path / "image2.zip"
    dataset.write_bytes(b"dataset")
    image1.write_bytes(b"image-one")
    image2.write_bytes(b"image-two")

    without_generation = flow._data_preparation_fingerprint(
        dataset_package=dataset,
        image1=None,
        image2=None,
        labels=["face"],
        annotation_provider="locate_sam3",
        generation_enabled=False,
        max_synthetic_images=5,
    )
    with_generation = flow._data_preparation_fingerprint(
        dataset_package=dataset,
        image1=image1,
        image2=image2,
        labels=["face"],
        annotation_provider="locate_sam3",
        generation_enabled=True,
        max_synthetic_images=5,
    )

    assert without_generation["real"]["digest"] == with_generation["real"]["digest"]
    assert without_generation["synthetic"]["digest"] != with_generation["synthetic"]["digest"]


def test_reuse_progress_is_exposed_as_stream_status(tmp_path: Path) -> None:
    writer = TrainingProgressWriter(
        thread_id="reuse-thread",
        run_id="run-target",
        run_dir=tmp_path / "run-target",
        workflow="yolo_training_flow",
        backend="deimv2",
        app_template_name="app",
        selected_skills=[],
        user_text="train",
    )
    writer.initialize()
    writer.handle_event(
        ChatEvent(
            type="data_preparation.artifacts_reused",
            data={
                "source_run_id": "run-source",
                "target_run_id": "run-target",
                "real_images": 14,
                "synthetic_images": 5,
                "annotations": 42,
                "copied_files": 38,
                "reuse_level": "real_and_synthetic",
                "synthetic_reused": True,
            },
        )
    )
    state = json.loads((tmp_path / "run-target" / "progress_state.json").read_text(encoding="utf-8"))
    info = _stream_info(state)

    assert state["phase"] == "data_reuse"
    assert state["generation"]["status"] == "completed"
    assert state["data_reuse"]["real_images"] + state["data_reuse"]["synthetic_images"] == 19
    assert "completed" not in state["annotation"]
    assert info["stage"] == "data_reuse"
    assert info["metrics"]["source_run_id"] == "run-source"


def test_real_only_reuse_does_not_mark_generation_completed(tmp_path: Path) -> None:
    writer = TrainingProgressWriter(
        thread_id="reuse-thread",
        run_id="run-target",
        run_dir=tmp_path / "run-target",
        workflow="yolo_training_flow",
        backend="deimv2",
        app_template_name="app",
        selected_skills=[],
        user_text="train",
    )
    writer.initialize()
    writer.handle_event(
        ChatEvent(
            type="data_preparation.artifacts_reused",
            data={
                "source_run_id": "run-source",
                "target_run_id": "run-target",
                "real_images": 14,
                "synthetic_images": 0,
                "annotations": 20,
                "copied_files": 29,
                "reuse_level": "real_only",
                "synthetic_reused": False,
            },
        )
    )
    state = json.loads((tmp_path / "run-target" / "progress_state.json").read_text(encoding="utf-8"))

    assert state["data_reuse"]["reuse_level"] == "real_only"
    assert state["annotation"]["status"] == "completed"
    assert state.get("generation", {}).get("status") not in {"completed", "reused"}


def test_http_and_acp_runtime_option_aliases_support_data_reuse() -> None:
    from app.api.training import _normalize_runtime_options
    from app.protocols.acp.adapter import _runtime_options_from_source

    http_options = _normalize_runtime_options(
        {"reusePreviousDataPreparation": "auto", "reuseFromRunId": "run-source"}
    )
    acp_options = _runtime_options_from_source(
        {"reusePreviousDataPreparation": "required", "reuseFromRunId": "run-source"}
    )

    assert http_options["reuse_previous_data_preparation"] == "auto"
    assert http_options["reuse_from_run_id"] == "run-source"
    assert acp_options["reuse_previous_data_preparation"] == "required"
    assert acp_options["reuse_from_run_id"] == "run-source"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_complete_previous_data_preparation_is_copied_into_new_run(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_fingerprint_mismatch_falls_back_without_copying(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_reuse_progress_is_exposed_as_stream_status(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_real_only_previous_run_is_reused_when_generation_is_skipped(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_failed_synthetic_run_reuses_real_annotations_and_retries_generation(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_real_annotations_are_reused_when_generation_becomes_enabled(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_real_and_synthetic_fingerprints_are_independent(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_real_only_reuse_does_not_mark_generation_completed(Path(directory))
    test_http_and_acp_runtime_option_aliases_support_data_reuse()
    print("training data reuse assertions passed")
