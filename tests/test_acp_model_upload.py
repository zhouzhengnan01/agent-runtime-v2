from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import acp_http_stream
from app.core.artifacts import ArtifactStore
from app.main import create_app


def test_acp_model_upload_accepts_pt_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/acp/models/model-upload-thread",
        files={"file": ("yolov8n-test.pt", b"model-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["modelId"].startswith("model-")
    assert payload["thread_id"] == "model-upload-thread"
    assert payload["name"] == "yolov8n-test.pt"
    assert payload["path"] == "/mnt/user-data/uploads/models/yolov8n-test.pt"
    assert payload["attachment"] == {
        "name": "yolov8n-test.pt",
        "path": "/mnt/user-data/uploads/models/yolov8n-test.pt",
        "mime_type": "application/octet-stream",
        "metadata": {
            "modelId": payload["modelId"],
            "role": "training_model",
            "source": "user_upload",
            "sha256": payload["sha256"],
            "size": len(b"model-bytes"),
        },
    }
    model_path = tmp_path / "model-upload-thread" / "uploads" / "models" / "yolov8n-test.pt"
    assert model_path.read_bytes() == b"model-bytes"
    registry = json.loads(
        (tmp_path / "model-upload-thread" / "workspace" / "training_models.json").read_text(encoding="utf-8")
    )
    model_record = registry["models"][payload["modelId"]]
    assert model_record["modelId"] == payload["modelId"]
    assert model_record["source"] == "user_upload"
    assert model_record["path"] == "/mnt/user-data/uploads/models/yolov8n-test.pt"
    assert model_record["local_path"] == str(model_path.resolve())
    assert model_record["sha256"] == payload["sha256"]


def test_acp_model_upload_accepts_pth_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/acp/models/model-upload-thread",
        files={"file": ("deimv2-finetune.pth", b"checkpoint-bytes", "application/octet-stream")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "deimv2-finetune.pth"
    assert payload["path"] == "/mnt/user-data/uploads/models/deimv2-finetune.pth"

    model_path = tmp_path / "model-upload-thread" / "uploads" / "models" / "deimv2-finetune.pth"
    assert model_path.read_bytes() == b"checkpoint-bytes"
    registry = json.loads(
        (tmp_path / "model-upload-thread" / "workspace" / "training_models.json").read_text(encoding="utf-8")
    )
    model_record = registry["models"][payload["modelId"]]
    assert model_record["path"] == "/mnt/user-data/uploads/models/deimv2-finetune.pth"
    assert model_record["local_path"] == str(model_path.resolve())


def test_acp_model_upload_registers_each_upload_with_unique_model_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(tmp_path))
    client = TestClient(create_app())

    first = client.post(
        "/api/acp/models/model-upload-thread",
        files={"file": ("custom.pt", b"first-model", "application/octet-stream")},
    ).json()
    second = client.post(
        "/api/acp/models/model-upload-thread",
        files={"file": ("custom.pt", b"second-model", "application/octet-stream")},
    ).json()

    assert first["modelId"] != second["modelId"]
    registry = json.loads(
        (tmp_path / "model-upload-thread" / "workspace" / "training_models.json").read_text(encoding="utf-8")
    )
    assert set(registry["models"]) == {first["modelId"], second["modelId"]}


def test_acp_model_upload_rejects_non_pt_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/acp/models/model-upload-thread",
        files={"file": ("model.onnx", b"onnx", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .pt and .pth model files are supported."
