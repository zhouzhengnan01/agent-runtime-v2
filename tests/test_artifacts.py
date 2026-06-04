from pathlib import Path

import pytest

from app.core.artifacts import ArtifactStore
from app.schemas import agent_result_content, artifact_content_block, artifact_relative_path


def test_artifact_store_blocks_traversal(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("t1")
    ref = store.write_text_artifact(paths, "ok.txt", "hello")
    assert ref.path.endswith("/ok.txt")

    with pytest.raises(ValueError):
        store.resolve_virtual_path("t1", "/mnt/user-data/outputs/../../secret.txt")


def test_artifact_store_resolves_relative_output_resource_links(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("t1")
    store.write_text_artifact(paths, "reports/result.md", "# hi")

    assert store.resolve_virtual_path("t1", "outputs/reports/result.md") == (
        paths.outputs / "reports" / "result.md"
    ).resolve()


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


def test_artifact_ref_urls_encode_path_segments(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("thread with spaces")

    ref = store.write_bytes_artifact(paths, "中文 预览/report #1?.png", b"image")

    assert ref.path == "/mnt/user-data/outputs/中文 预览/report #1?.png"
    assert ref.preview_url == (
        "/api/artifacts/thread-with-spaces/"
        "mnt/user-data/outputs/%E4%B8%AD%E6%96%87%20%E9%A2%84%E8%A7%88/report%20%231%3F.png"
    )
    assert ref.download_url == f"{ref.preview_url}?download=true"


def test_artifact_content_blocks_use_relative_output_paths(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("content-thread")
    ref = store.write_text_artifact(paths, "reports/result.md", "# hi")

    assert artifact_relative_path(ref.path) == "outputs/reports/result.md"
    assert artifact_content_block(ref) == {
        "type": "resource_link",
        "uri": "outputs/reports/result.md",
        "path": "outputs/reports/result.md",
        "name": "result.md",
        "mimeType": "text/markdown",
        "size": 4,
        "title": "result.md",
    }
    assert agent_result_content("done", [ref]) == [
        {"type": "text", "text": "done"},
        {
            "type": "resource_link",
            "uri": "outputs/reports/result.md",
            "path": "outputs/reports/result.md",
            "name": "result.md",
            "mimeType": "text/markdown",
            "size": 4,
            "title": "result.md",
        },
    ]


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
