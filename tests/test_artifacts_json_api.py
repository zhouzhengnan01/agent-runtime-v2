from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import artifacts_json as artifacts_json_api
from app.main import create_app


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_prepare_deimv2_dataset_module():
    path = Path(
        "plugins/skills/deimv2-auto-training/scripts/prepare_deimv2_dataset.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location("prepare_deimv2_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coco_payload(*, synthetic: bool = False) -> dict[str, object]:
    source = "synthetic" if synthetic else "real"
    return {
        "images": [
            {
                "id": 10,
                "file_name": f"{source}/sample.jpg",
                "width": 640,
                "height": 480,
                "source": source,
                "is_synthetic": synthetic,
            }
        ],
        "annotations": [
            {
                "id": 20,
                "image_id": 10,
                "category_id": 30,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 30, "name": "person", "supercategory": "object"}],
    }


def test_artifacts_json_returns_all_rounds_with_images_and_merged_coco(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = artifacts_json_api.ArtifactStore(tmp_path)
    paths = store.prepare_thread("face")
    run_id = "run-deimv2-001"
    run_dir = paths.outputs / "yolo_training_flow" / "runs" / run_id
    merged = {
        "images": [
            *_coco_payload()["images"],
            *_coco_payload(synthetic=True)["images"],
        ],
        "annotations": [
            *_coco_payload()["annotations"],
            {
                **_coco_payload(synthetic=True)["annotations"][0],
                "id": 21,
                "image_id": 11,
            },
        ],
        "categories": _coco_payload()["categories"],
    }
    merged["images"][1]["id"] = 11
    _write_json(run_dir / "pipeline_work" / "merged_coco.json", merged)
    real_image = run_dir / "uploaded_dataset" / "real" / "sample.jpg"
    real_image.parent.mkdir(parents=True, exist_ok=True)
    real_image.write_bytes(b"real")
    synthetic_image = run_dir / "pipeline_work" / "synthetic_images" / "synthetic" / "sample.jpg"
    synthetic_image.parent.mkdir(parents=True, exist_ok=True)
    synthetic_image.write_bytes(b"synthetic")
    _write_json(run_dir / "uploaded_dataset" / "real" / "sample_coco.json", _coco_payload())
    (run_dir / "run_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_summary.json").write_text('{"status":"completed"}', encoding="utf-8")
    yolo_label = (
        run_dir
        / "prepared_data"
        / "prepared_dataset"
        / "labels"
        / "val"
        / "sample.txt"
    )
    yolo_label.parent.mkdir(parents=True, exist_ok=True)
    yolo_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (run_dir / "training-notes.txt").write_text("keep this text artifact", encoding="utf-8")
    model_path = run_dir / "deimv2_training_run" / "runs" / "train" / "best.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")

    second_run = paths.outputs / "yolo_training_flow" / "runs" / "run-deimv2-002"
    (second_run / "failed-run.txt").parent.mkdir(parents=True, exist_ok=True)
    (second_run / "failed-run.txt").write_text("failed before annotation", encoding="utf-8")

    monkeypatch.setattr(artifacts_json_api, "store", store)
    client = TestClient(create_app())
    response = client.get("/api/artifacts_json/face")

    assert response.status_code == 200
    payload = response.json()
    assert payload["threadId"] == "face"
    assert payload["totalRounds"] == 2
    assert [item["runId"] for item in payload["rounds"]] == [
        "run-deimv2-001",
        "run-deimv2-002",
    ]

    first_round = payload["rounds"][0]
    assert first_round["round"] == 1
    names = [item["name"] for item in first_round["artifacts"]]
    assert "sample_coco.json" not in names
    assert "merged_coco.json" not in names
    assert "run_summary.json" in names
    assert "sample.txt" not in names
    assert "training-notes.txt" in names
    assert "instances_all.json" not in names
    assert names.count("sample.jpg") == 2
    assert first_round["merged_coco"]["real_image_count"] == 1
    assert first_round["merged_coco"]["synthetic_image_count"] == 1
    assert first_round["merged_coco"]["image_count"] == 2
    assert first_round["merged_coco"]["annotation_count"] == 2

    real_artifact = next(
        item for item in first_round["artifacts"] if "/uploaded_dataset/" in item["path"]
    )
    synthetic_artifact = next(
        item for item in first_round["artifacts"] if "/synthetic_images/" in item["path"]
    )
    model_artifact = next(item for item in first_round["artifacts"] if item["name"] == "best.onnx")
    summary_artifact = next(
        item for item in first_round["artifacts"] if item["name"] == "run_summary.json"
    )
    assert real_artifact["stage"] == "annotation"
    assert real_artifact["artifact_type"] == "image"
    assert synthetic_artifact["stage"] == "generation"
    assert synthetic_artifact["artifact_type"] == "image"
    assert model_artifact["stage"] == "training"
    assert model_artifact["artifact_type"] == "model"
    assert summary_artifact["stage"] == "summary"
    assert summary_artifact["artifact_type"] == "summary"
    for artifact in first_round["artifacts"]:
        assert artifact["round"] == 1
        assert artifact["runId"] == run_id
        assert artifact["threadId"] == "face"
        assert artifact["downloadUrl"].startswith("http://testserver/api/artifacts_json/face/")

    download = client.get(first_round["merged_coco"]["download_url"])
    assert download.status_code == 200
    exported = download.json()
    assert [item["id"] for item in exported["images"]] == [1, 2]
    assert [item["id"] for item in exported["annotations"]] == [1, 2]
    assert exported["categories"] == [{"id": 1, "name": "person", "supercategory": "object"}]
    assert all("jetlinks_artifact" not in item for item in exported["images"])
    assert client.get(real_artifact["downloadUrl"]).content == b"real"
    assert client.get(synthetic_artifact["downloadUrl"]).content == b"synthetic"

    failed_round = payload["rounds"][1]
    assert failed_round["round"] == 2
    assert failed_round["merged_coco"] is None
    assert failed_round["source_coco_files"] == []
    assert "No COCO annotations found" in failed_round["coco_error"]
    assert [item["name"] for item in failed_round["artifacts"]] == ["failed-run.txt"]


def test_artifacts_json_requires_a_training_run_but_allows_a_run_without_coco(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = artifacts_json_api.ArtifactStore(tmp_path)
    paths = store.prepare_thread("face")
    monkeypatch.setattr(artifacts_json_api, "store", store)
    client = TestClient(create_app())

    missing = client.get("/api/artifacts_json/face")
    assert missing.status_code == 404
    assert "No training runs found" in missing.json()["detail"]

    empty_run = paths.outputs / "yolo_training_flow" / "runs" / "run-deimv2-empty"
    empty_run.mkdir(parents=True)
    response = client.get("/api/artifacts_json/face")
    assert response.status_code == 200
    round_payload = response.json()["rounds"][0]
    assert round_payload["merged_coco"] is None
    assert "No COCO annotations found" in round_payload["coco_error"]


def test_artifacts_json_merges_real_and_synthetic_when_merged_coco_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = artifacts_json_api.ArtifactStore(tmp_path)
    paths = store.prepare_thread("face")
    run_id = "run-deimv2-sources"
    pipeline_work = paths.outputs / "yolo_training_flow" / "runs" / run_id / "pipeline_work"
    _write_json(pipeline_work / "real_coco.json", _coco_payload())
    _write_json(pipeline_work / "synthetic_coco.json", _coco_payload(synthetic=True))

    monkeypatch.setattr(artifacts_json_api, "store", store)
    response = TestClient(create_app()).get("/api/artifacts_json/face")

    assert response.status_code == 200
    payload = response.json()["rounds"][0]
    assert payload["source_coco_files"] == [
        "pipeline_work/real_coco.json",
        "pipeline_work/synthetic_coco.json",
    ]
    assert payload["merged_coco"]["real_image_count"] == 1
    assert payload["merged_coco"]["synthetic_image_count"] == 1


def test_deimv2_preparation_ignores_jetlinks_image_urls(tmp_path: Path) -> None:
    module = _load_prepare_deimv2_dataset_module()
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    coco_path = dataset_root / "annotations" / "instances_all.json"
    _write_json(
        coco_path,
        {
            "images": [
                {
                    "id": 1,
                    "file_name": "images/sample.jpg",
                    "width": 640,
                    "height": 480,
                    "jetlinks_artifact": {
                        "preview_url": "/api/artifacts_json/face/sample.jpg",
                        "download_url": "/api/artifacts_json/face/sample.jpg?download=true",
                    },
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 20, 30, 40],
                }
            ],
            "categories": [{"id": 1, "name": "person"}],
        },
    )
    output_dir = tmp_path / "prepared"

    summary = module.prepare_dataset(
        {
            "dataset_root": str(dataset_root),
            "coco_json": str(coco_path),
            "output_dir": str(output_dir),
            "class_names": ["person"],
            "split": {"train": 1.0, "val": 0.0, "test": 0.0},
        }
    )

    assert summary["split_counts"]["train"]["images"] == 1
    training_coco = json.loads(
        (output_dir / "annotations" / "instances_train.json").read_text(encoding="utf-8")
    )
    assert training_coco["images"][0]["file_name"].endswith("sample.jpg")
    assert "jetlinks_artifact" not in training_coco["images"][0]
