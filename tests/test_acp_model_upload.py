from __future__ import annotations

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
    assert payload["thread_id"] == "model-upload-thread"
    assert payload["name"] == "yolov8n-test.pt"
    assert payload["path"] == "/mnt/user-data/uploads/models/yolov8n-test.pt"
    assert payload["attachment"] == {
        "name": "yolov8n-test.pt",
        "path": "/mnt/user-data/uploads/models/yolov8n-test.pt",
        "mime_type": "application/octet-stream",
    }
    assert (tmp_path / "model-upload-thread" / "uploads" / "models" / "yolov8n-test.pt").read_bytes() == b"model-bytes"


def test_acp_model_upload_rejects_non_pt_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/acp/models/model-upload-thread",
        files={"file": ("model.onnx", b"onnx", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .pt model files are supported."
