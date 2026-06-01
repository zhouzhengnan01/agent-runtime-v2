from __future__ import annotations

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


class FakeArtifactStore:
    def __init__(self, outputs_root: Path) -> None:
        self.outputs_root = outputs_root.resolve()
        self.upserts: list[Path] = []

    def upsert_artifact(self, paths: object, file_path: Path) -> None:
        del paths
        self.upserts.append(file_path.resolve())

    def to_artifact_ref(self, thread_id: str, file_path: Path) -> object:
        resolved = file_path.resolve()
        try:
            rel = resolved.relative_to(self.outputs_root).as_posix()
        except ValueError:
            rel = resolved.as_posix()
        return SimpleNamespace(
            thread_id=thread_id,
            path=f"/mnt/user-data/outputs/{rel}",
            name=file_path.name,
            kind=file_path.suffix.lstrip(".") or "file",
            mime_type="application/octet-stream",
        )


def test_gpu_training_runner_collects_only_key_artifacts_from_large_run(tmp_path: Path) -> None:
    module = _load_runner("plugins/skills/gpu-training-orchestrator/runner.py", "gpu_training_runner_for_test")
    paths = _thread_paths(tmp_path)
    flow_root = paths.outputs / "yolo_training_flow"
    run_root = flow_root / "training_run"
    train_dir = run_root / "train"
    eval_dir = run_root / "test_eval"
    log_dir = flow_root / "logs"

    _touch(run_root / "run_summary.json")
    _touch(train_dir / "weights" / "best.pt")
    _touch(train_dir / "weights" / "last.pt")
    for name in [
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
    ]:
        _touch(train_dir / name)
    _touch(eval_dir / "confusion_matrix.png")
    for index in range(20):
        _touch(train_dir / f"train_batch{index}.jpg")
    for name in module.LOG_FILENAMES:
        _touch(log_dir / name)

    for index in range(800):
        _touch(run_root / "synthetic_images" / f"synth_{index:05d}.png")
    for index in range(1000):
        _touch(run_root / "labels" / f"synth_{index:05d}.txt")
    for index in range(800):
        _touch(run_root / "intermediate_json" / f"synth_{index:05d}.json")
    for index in range(400):
        _touch(run_root / "crops" / f"crop_{index:05d}.jpg")

    spec = {"output": {"project_dir": str(run_root), "run_name": "."}}
    store = FakeArtifactStore(paths.outputs)

    outputs = module._collect_user_visible_outputs(spec, paths, store)
    artifact_paths = _artifact_paths(outputs)

    assert _file_count(run_root) >= 3000
    assert len(store.upserts) <= 30
    assert len(outputs) == len(store.upserts)
    assert any(path.endswith("/weights/best.pt") for path in artifact_paths)
    assert any(path.endswith("/weights/last.pt") for path in artifact_paths)
    assert any(path.endswith("/run_summary.json") for path in artifact_paths)
    assert any(path.endswith("/results.csv") for path in artifact_paths)
    assert any(path.endswith("/confusion_matrix.png") for path in artifact_paths)
    assert sum(PurePosixPath(path).name.startswith("train_batch") for path in artifact_paths) == module.MAX_BATCH_VISUALS
    assert not any("/synthetic_images/" in path for path in artifact_paths)
    assert not any("/labels/" in path for path in artifact_paths)
    assert not any("/intermediate_json/" in path for path in artifact_paths)
    assert not any("/crops/" in path for path in artifact_paths)


def test_data_auto_annotation_runner_skips_bulk_prepared_and_synthetic_files(tmp_path: Path) -> None:
    module = _load_runner("plugins/skills/data-auto-annotation/runner.py", "data_auto_annotation_runner_for_test")
    paths = _thread_paths(tmp_path)
    flow_root = paths.outputs / "yolo_training_flow"
    prepared_root = flow_root / "prepared_data"
    work_dir = flow_root / "pipeline_work"
    log_dir = flow_root / "logs"

    for name in module.KEY_PREPARED_ARTIFACT_NAMES:
        _touch(prepared_root / name)
    for name in module.KEY_WORK_ARTIFACT_NAMES:
        _touch(work_dir / name)
    for name in module.LOG_FILENAMES:
        _touch(log_dir / name)

    for index in range(900):
        _touch(prepared_root / "prepared_dataset" / "images" / "train" / f"image_{index:05d}.jpg")
    for index in range(900):
        _touch(prepared_root / "prepared_dataset" / "labels" / "train" / f"image_{index:05d}.txt")
    for index in range(800):
        _touch(work_dir / "synthetic_images" / f"synth_{index:05d}.png")
    for index in range(800):
        _touch(work_dir / "synthetic_annotations" / f"synth_{index:05d}_coco.json")
    for index in range(800):
        _touch(work_dir / "generation_inputs" / f"scene_{index:05d}.json")
    for index in range(600):
        _touch(flow_root / "uploaded_dataset" / "images" / f"real_{index:05d}.jpg.coco.json")

    spec = {"output_dir": str(prepared_root), "work_dir": str(work_dir)}
    store = FakeArtifactStore(paths.outputs)

    outputs = module._collect_outputs(spec, paths, store)
    artifact_paths = _artifact_paths(outputs)

    assert _file_count(flow_root) >= 3000
    assert len(store.upserts) <= 20
    assert len(outputs) == len(store.upserts)
    assert any(path.endswith("/dataset.yaml") for path in artifact_paths)
    assert any(path.endswith("/data_preparation_summary.json") for path in artifact_paths)
    assert any(path.endswith("/synthetic_plan.json") for path in artifact_paths)
    assert not any("/prepared_dataset/" in path for path in artifact_paths)
    assert not any("/uploaded_dataset/" in path for path in artifact_paths)
    assert not any("/synthetic_images/" in path for path in artifact_paths)
    assert not any("/synthetic_annotations/" in path for path in artifact_paths)
    assert not any("/generation_inputs/" in path for path in artifact_paths)


def test_gpu_training_runner_writes_logs_while_subprocess_is_running(tmp_path: Path) -> None:
    module = _load_runner("plugins/skills/gpu-training-orchestrator/runner.py", "gpu_training_runner_live_logs_for_test")
    paths = _thread_paths(tmp_path)
    flow_root = paths.outputs / "yolo_training_flow"
    run_root = flow_root / "training_run"
    log_dir = flow_root / "logs"
    stdout_log = log_dir / "yolo-training-stdout.txt"
    stderr_log = log_dir / "yolo-training-stderr.txt"
    done_file = tmp_path / "done.txt"
    script_path = tmp_path / "emit_logs.py"
    script_path.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "import time",
                "print('live-start', flush=True)",
                "time.sleep(0.8)",
                "print('live-error', file=sys.stderr, flush=True)",
                f"Path({str(done_file)!r}).write_text('done', encoding='utf-8')",
                "print('live-end', flush=True)",
            ]
        ),
        encoding="utf-8",
    )
    spec = {"output": {"project_dir": str(run_root), "run_name": "."}}
    saw_log_before_done = threading.Event()

    def watch_stdout() -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            if stdout_log.is_file() and "live-start" in stdout_log.read_text(encoding="utf-8"):
                if not done_file.exists():
                    saw_log_before_done.set()
                return
            time.sleep(0.02)

    watcher = threading.Thread(target=watch_stdout)
    watcher.start()
    completed = module._run_command_with_live_logs(
        [sys.executable, str(script_path)],
        tmp_path,
        os.environ.copy(),
        spec,
    )
    watcher.join(timeout=1)

    assert completed.returncode == 0
    assert saw_log_before_done.is_set()
    assert "live-start" in stdout_log.read_text(encoding="utf-8")
    assert "live-error" in stderr_log.read_text(encoding="utf-8")


def test_gpu_training_runner_prepares_ultralytics_font_assets(tmp_path: Path) -> None:
    module = _load_runner("plugins/skills/gpu-training-orchestrator/runner.py", "gpu_training_runner_fonts_for_test")
    source = tmp_path / "source-font.ttf"
    source.write_bytes(b"fake-font")
    module.ULTRALYTICS_FONT_CANDIDATES = [source]

    config_dir = tmp_path / "ultralytics_config"
    module._prepare_ultralytics_offline_assets(config_dir)

    for base in (config_dir, config_dir / "Ultralytics"):
        assert (base / "Arial.ttf").read_bytes() == b"fake-font"
        assert (base / "Arial.Unicode.ttf").read_bytes() == b"fake-font"


def _load_runner(relative_path: str, module_name: str) -> object:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / relative_path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _thread_paths(tmp_path: Path) -> object:
    root = tmp_path / "thread"
    return SimpleNamespace(
        thread_id="local-mock-800",
        root=root,
        workspace=root / "workspace",
        uploads=root / "uploads",
        outputs=root / "outputs",
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _artifact_paths(outputs: list[object]) -> list[str]:
    return [str(getattr(output, "path")).replace("\\", "/") for output in outputs]


def _file_count(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())
