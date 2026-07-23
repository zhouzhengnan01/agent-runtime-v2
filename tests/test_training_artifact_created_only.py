from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.artifacts import ArtifactStore
from app.core.training_artifact_updates import build_training_artifact_session_updates


def _load_annotation_runner():
    path = ROOT / "plugins" / "skills" / "data-auto-annotation" / "runner.py"
    spec = importlib.util.spec_from_file_location("data_auto_annotation_created_only_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_polled_training_artifact_emits_created_only(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("created-only-thread")
    coco = paths.outputs / "yolo_training_flow" / "runs" / "run-1" / "pipeline_work" / "real_coco.json"
    coco.parent.mkdir(parents=True)
    coco.write_text('{"images":[],"annotations":[],"categories":[]}', encoding="utf-8")
    status = {"annotation": {"real": {"output_coco": str(coco)}}}

    updates = build_training_artifact_session_updates(
        thread_id=paths.thread_id,
        status=status,
        artifact_store=store,
        published_artifacts=set(),
    )

    assert len(updates) == 1
    update = updates[0]
    assert update["_meta"]["jetlinksRuntimeEvent"]["type"] == "artifact.created"
    assert update["artifact"]["preview_url"]
    assert update["artifact"]["download_url"]


def test_annotation_runner_emits_created_without_preview() -> None:
    runner = _load_annotation_runner()
    events: list[str] = []
    artifact = SimpleNamespace(model_dump=lambda: {"name": "real_coco.json"})

    runner._emit_artifacts([artifact], lambda event_type, _payload: events.append(event_type))

    assert events == ["artifact.created"]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        test_polled_training_artifact_emits_created_only(Path(directory))
    test_annotation_runner_emits_created_without_preview()
    print("training artifact created-only assertions passed")
