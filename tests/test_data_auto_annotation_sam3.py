from __future__ import annotations

import importlib.util
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


def test_sam3_predict_builds_v1_json_payload(monkeypatch, tmp_path: Path) -> None:
    module = _load_sam3_module(monkeypatch)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-bytes")

    payload = module.build_sam3_payload(image_path, ["person", "monitor"], 0.35, 0.5)

    assert module.DEFAULT_URL == "http://218.67.242.10:58800/v1/sam3/predict"
    assert payload["model"] == "sam3"
    assert payload["input"]["image"] == "data:image/png;base64,cG5nLWJ5dGVz"
    assert payload["input"]["text_prompts"] == ["person", "monitor"]
    assert payload["parameters"] == {"conf": 0.35, "iou": 0.5}


def test_locate_sam3_provider_uses_locateanything_payload_and_default_url(monkeypatch, tmp_path: Path) -> None:
    module = _load_sam3_module(monkeypatch)
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png-bytes")

    payload = module.build_annotation_payload(
        "locate_sam3",
        "",
        image_path,
        ["cat", "laptop"],
        0.35,
        0.5,
        "hybrid",
        1024,
        True,
    )

    assert module.effective_url(module.DEFAULT_URL, "locate_sam3") == module.LOCATE_SAM3_DEFAULT_URL
    assert payload["model"] == "locateanything"
    assert payload["input"]["text_prompts"] == ["cat", "laptop"]
    assert payload["parameters"] == {
        "iou": 0.5,
        "mode": "hybrid",
        "max_new_tokens": 1024,
        "save_visualization": True,
    }


def test_sam3_predict_extracts_nested_locate_sam3_boxes(monkeypatch) -> None:
    module = _load_sam3_module(monkeypatch)
    payload = {
        "data": {
            "boxes": [
                {
                    "text": "cat",
                    "confidence": 0.8,
                    "bbox_2d": [10, 20, 30, 50],
                }
            ]
        }
    }

    boxes = module._extract_boxes(payload)
    normalized = module._normalize_box(boxes[0])

    assert normalized == {"label": "cat", "score": 0.8, "bbox": [10.0, 20.0, 20.0, 30.0]}
