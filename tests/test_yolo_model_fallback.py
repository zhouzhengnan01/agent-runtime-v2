from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def test_load_yolo_model_falls_back_to_builtin_model_when_requested_model_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_training_script()
    fallback_model = tmp_path / "models" / "yolo11n.pt"
    fallback_model.parent.mkdir(parents=True, exist_ok=True)
    fallback_model.write_bytes(b"weights")
    monkeypatch.setattr(module, "_builtin_fallback_model_candidates", lambda: [fallback_model])

    calls: list[str] = []

    class FakeYOLO:
        def __init__(self, model: str, task: str) -> None:
            assert task == "detect"
            calls.append(str(model))
            if str(model) != str(fallback_model.resolve()):
                raise OSError("network unavailable")

    model, resolved_model, fallback_used, load_error = module._load_yolo_model(FakeYOLO, "yolo11s.pt", "detect")

    assert isinstance(model, FakeYOLO)
    assert calls == ["yolo11s.pt", str(fallback_model.resolve())]
    assert resolved_model == str(fallback_model.resolve())
    assert fallback_used is True
    assert "network unavailable" in load_error


def test_resolve_model_name_prefers_project_models_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_training_script()
    project_root = tmp_path / "repo"
    bundled_model = project_root / "models" / "yolov11n.pt"
    bundled_model.parent.mkdir(parents=True, exist_ok=True)
    bundled_model.write_bytes(b"weights")
    monkeypatch.setattr(module, "_project_root", lambda: project_root)

    assert module._resolve_model_name("yolov11n.pt") == str(bundled_model.resolve())


def _load_training_script() -> object:
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "plugins" / "skills" / "gpu-training-orchestrator" / "scripts" / "run_prepared_yolo_training.py"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("run_prepared_yolo_training_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
