from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_sam3_module(monkeypatch):
    requests_stub = types.ModuleType("requests")
    requests_stub.exceptions = types.SimpleNamespace(
        ConnectTimeout=type("ConnectTimeout", (Exception,), {}),
        ReadTimeout=type("ReadTimeout", (Exception,), {}),
        ConnectionError=type("ConnectionError", (Exception,), {}),
        HTTPError=type("HTTPError", (Exception,), {}),
        RequestException=type("RequestException", (Exception,), {}),
    )
    pil_stub = types.ModuleType("PIL")
    image_stub = types.ModuleType("PIL.Image")
    pil_stub.Image = image_stub
    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    monkeypatch.setitem(sys.modules, "PIL", pil_stub)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_stub)

    script_path = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "skills"
        / "data-auto-annotation"
        / "scripts"
        / "sam3-predict.py"
    )
    spec = importlib.util.spec_from_file_location("sam3_predict", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sam3_predict_url_can_be_overridden_by_environment(monkeypatch) -> None:
    module = _load_sam3_module(monkeypatch)

    monkeypatch.setenv("SAM3_PREDICT_URL", "https://sam3.example.test/sam3/predict")

    assert module.effective_url("$SAM3_PREDICT_URL") == "https://sam3.example.test/sam3/predict"
    assert module.effective_url("${SAM3_PREDICT_URL}") == "https://sam3.example.test/sam3/predict"


def test_sam3_predict_url_falls_back_to_default(monkeypatch) -> None:
    module = _load_sam3_module(monkeypatch)
    monkeypatch.delenv("SAM3_PREDICT_URL", raising=False)
    monkeypatch.delenv("SAM3_URL", raising=False)

    assert module.effective_url("$SAM3_PREDICT_URL") == module.DEFAULT_URL


def test_sam3_predict_explicit_url_is_not_overridden(monkeypatch) -> None:
    module = _load_sam3_module(monkeypatch)
    monkeypatch.setenv("SAM3_PREDICT_URL", "https://sam3.example.test/sam3/predict")

    assert module.effective_url("https://explicit.example.test/predict") == "https://explicit.example.test/predict"


def test_sam3_predict_streams_per_image_coco_sidecars(tmp_path: Path, monkeypatch) -> None:
    module = _load_sam3_module(monkeypatch)
    images_dir = tmp_path / "uploaded_dataset" / "images"
    images_dir.mkdir(parents=True)
    first = images_dir / "000_first.jpg"
    second = images_dir / "001_second.jpg"
    first.write_bytes(b"fake")
    second.write_bytes(b"fake")
    output = tmp_path / "pipeline_work" / "real_coco.json"
    first_sidecar = images_dir / "000_first.jpg.coco.json"
    saw_first_sidecar_before_second_request = False

    def fake_image_size(image_path: Path) -> tuple[int, int]:
        del image_path
        return (64, 64)

    def fake_post_image(args, headers, data, image_path: Path) -> dict:
        del args, headers, data
        nonlocal saw_first_sidecar_before_second_request
        if image_path == second:
            saw_first_sidecar_before_second_request = first_sidecar.is_file()
        return {"boxes": []}

    def fake_response_to_coco(
        payload,
        image_path: Path,
        image_id: int,
        category_map,
        categories,
        source: str = "real",
        is_synthetic: bool = False,
        annotation_start_id: int = 1,
    ) -> tuple[dict, list[dict]]:
        del payload, category_map, categories
        return (
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": 64,
                "height": 64,
                "source": source,
                "is_synthetic": is_synthetic,
            },
            [
                {
                    "id": annotation_start_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [1, 2, 3, 4],
                    "area": 12,
                    "iscrowd": 0,
                    "segmentation": [],
                    "score": 0.9,
                }
            ],
        )

    monkeypatch.setattr(module, "image_size", fake_image_size)
    monkeypatch.setattr(module, "post_image", fake_post_image)
    monkeypatch.setattr(module, "response_to_coco", fake_response_to_coco)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sam3-predict.py",
            "--input-dir",
            str(images_dir),
            "--text-prompts",
            "person",
            "--source",
            "real",
            "--per-image-output-dir",
            str(images_dir),
            "--per-image-base-dir",
            str(images_dir),
            "--output",
            str(output),
        ],
    )

    assert module.main() == 0

    assert saw_first_sidecar_before_second_request is True
    assert first_sidecar.is_file()
    assert (images_dir / "001_second.jpg.coco.json").is_file()
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    first_payload = json.loads(first_sidecar.read_text(encoding="utf-8"))
    assert len(aggregate["images"]) == 2
    assert len(aggregate["annotations"]) == 2
    assert first_payload["images"][0]["id"] == 1
    assert first_payload["annotations"][0]["image_id"] == 1
