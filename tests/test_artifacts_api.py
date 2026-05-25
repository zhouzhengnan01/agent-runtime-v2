from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import artifacts as artifacts_api
from app.main import create_app


def test_artifact_download_route_returns_attachment_for_text_files(tmp_path: Path, monkeypatch) -> None:
    store = artifacts_api.ArtifactStore(tmp_path)
    paths = store.prepare_thread("download-thread")
    store.write_text_artifact(paths, "result.md", "# Hello\n")

    monkeypatch.setattr(artifacts_api, "store", store)
    client = TestClient(create_app())

    response = client.get("/api/artifacts/download-thread/mnt/user-data/outputs/result.md?download=true")

    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith('attachment; filename="result.md"')
    assert response.text == "# Hello\n"


def test_artifact_route_serves_encoded_image_paths(tmp_path: Path, monkeypatch) -> None:
    store = artifacts_api.ArtifactStore(tmp_path)
    paths = store.prepare_thread("preview-thread")
    artifact = store.write_bytes_artifact(paths, "中文 预览/report #1?.png", b"png-bytes")

    monkeypatch.setattr(artifacts_api, "store", store)
    client = TestClient(create_app())

    response = client.get(artifact.preview_url)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png-bytes"
