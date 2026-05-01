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

