from pathlib import Path

import pytest

from app.core.artifacts import ArtifactStore


def test_artifact_store_blocks_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("t1")
    ref = store.write_text_artifact(paths, "ok.txt", "hello")
    assert ref.path.endswith("/ok.txt")

    with pytest.raises(ValueError):
        store.resolve_virtual_path("t1", "/mnt/user-data/outputs/../../secret.txt")


def test_artifact_list(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("t1")
    store.write_text_artifact(paths, "a/result.md", "# hi")
    refs = store.list_artifacts("t1")
    assert len(refs) == 1
    assert refs[0].name == "result.md"
    manifest = store.read_manifest("t1")
    assert manifest["primary_artifact"] == "outputs/a/result.md"
    assert manifest["artifacts"][0]["path"] == "outputs/a/result.md"


def test_artifact_store_accepts_virtual_output_paths(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("t1")

    store.write_text_artifact(paths, "/mnt/user-data/outputs/reports/result.md", "# hi")

    assert (paths.outputs / "reports" / "result.md").read_text() == "# hi"
    assert store.output_path(paths, "/mnt/user-data/outputs/reports/result.md") == paths.outputs / "reports" / "result.md"
    assert store.output_path(paths, "mnt/user-data/outputs/reports/result.md") == paths.outputs / "reports" / "result.md"


def test_artifact_store_prepares_thread_workspace_dirs(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("thread/one")

    assert paths.root == tmp_path / "thread-one"
    assert paths.workspace == tmp_path / "thread-one" / "workspace"
    assert paths.uploads == tmp_path / "thread-one" / "uploads"
    assert paths.outputs == tmp_path / "thread-one" / "outputs"
    assert paths.previews == tmp_path / "thread-one" / "previews"
    assert paths.versions == tmp_path / "thread-one" / "versions"
    assert paths.memory == tmp_path / "thread-one" / "memory"
    assert paths.manifest == tmp_path / "thread-one" / "manifest.json"
    assert paths.manifest.is_file()
