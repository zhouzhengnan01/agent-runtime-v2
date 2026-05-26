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
