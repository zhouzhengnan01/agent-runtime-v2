from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.api import uploads
from app.core.artifacts import ArtifactStore
from app.main import create_app


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(uploads, "store", ArtifactStore(tmp_path))
    return TestClient(create_app())


def test_training_dataset_upload_uses_canonical_name_and_replaces_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)

    first = client.post(
        "/api/uploads/training-input-thread",
        files={"file": ("dataset.zip", b"first-dataset", "application/zip")},
    )
    second = client.post(
        "/api/uploads/training-input-thread",
        files={"file": ("datasets.zip", b"second-dataset", "application/zip")},
    )

    assert first.status_code == 200
    assert first.json()["name"] == "datasets.zip"
    assert first.json()["original_name"] == "dataset.zip"
    assert first.json()["replaced"] is False
    assert second.status_code == 200
    assert second.json()["name"] == "datasets.zip"
    assert second.json()["replaced"] is True

    uploads_dir = tmp_path / "training-input-thread" / "uploads"
    assert (uploads_dir / "datasets.zip").read_bytes() == b"second-dataset"
    assert not (uploads_dir / "dataset.zip").exists()
    assert [path.name for path in uploads_dir.iterdir()] == ["datasets.zip"]


def test_training_image_upload_replaces_existing_file(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    first = client.post(
        "/api/uploads/training-input-thread",
        files={"file": ("image1.zip", b"first-image-set", "application/zip")},
    )
    second = client.post(
        "/api/uploads/training-input-thread",
        files={"file": ("image1.zip", b"second-image-set", "application/zip")},
    )

    assert first.status_code == 200
    assert first.json()["replaced"] is False
    assert second.status_code == 200
    assert second.json()["replaced"] is True
    assert (
        tmp_path / "training-input-thread" / "uploads" / "image1.zip"
    ).read_bytes() == b"second-image-set"


def test_failed_training_input_upload_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    first = client.post(
        "/api/uploads/training-input-thread",
        files={"file": ("image2.zip", b"old", "application/zip")},
    )
    assert first.status_code == 200

    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 4)
    failed = client.post(
        "/api/uploads/training-input-thread",
        files={"file": ("image2.zip", b"too-large", "application/zip")},
    )

    assert failed.status_code == 413
    uploads_dir = tmp_path / "training-input-thread" / "uploads"
    assert (uploads_dir / "image2.zip").read_bytes() == b"old"
    assert not list(uploads_dir.glob("*.uploading"))


def test_non_training_upload_keeps_unique_versions(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    first = client.post(
        "/api/uploads/general-upload-thread",
        files={"file": ("notes.zip", b"first", "application/zip")},
    ).json()
    second = client.post(
        "/api/uploads/general-upload-thread",
        files={"file": ("notes.zip", b"second", "application/zip")},
    ).json()

    assert first["name"] == "notes.zip"
    assert first["replaced"] is False
    assert second["name"].startswith("notes-")
    assert second["name"].endswith(".zip")
    assert second["replaced"] is False
    uploads_dir = tmp_path / "general-upload-thread" / "uploads"
    assert (uploads_dir / first["name"]).read_bytes() == b"first"
    assert (uploads_dir / second["name"]).read_bytes() == b"second"
