from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_training_script():
    path = Path("plugins/skills/gpu-training-orchestrator/scripts/run_yolo_training.py").resolve()
    spec = importlib.util.spec_from_file_location("run_yolo_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_summary_includes_per_class_metrics_and_analysis(tmp_path: Path) -> None:
    module = _load_training_script()
    prepared_root = tmp_path / "prepared_dataset"
    (prepared_root / "images" / "test").mkdir(parents=True)
    (prepared_root / "labels" / "test").mkdir(parents=True)
    (prepared_root / "images" / "test" / "sample.jpg").write_bytes(b"")
    (prepared_root / "labels" / "test" / "sample.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n"
        "0 0.6 0.5 0.1 0.1\n"
        "1 0.3 0.3 0.05 0.05\n",
        encoding="utf-8",
    )

    class FakeBox:
        p = [0.99, 1.0]
        r = [1.0, 0.0]
        ap50 = [0.995, 0.111]
        maps = [0.91, 0.022]

    class FakeResults:
        box = FakeBox()

    test_set_summary = module._summarize_yolo_split(prepared_root, "test", ["person", "cigarette"])
    per_class_metrics = module._extract_per_class_metrics(FakeResults(), ["person", "cigarette"], test_set_summary)
    analysis = module._build_class_performance_analysis(test_set_summary, per_class_metrics)

    assert test_set_summary["images"] == 1
    assert test_set_summary["instances"] == 3
    assert per_class_metrics[0]["class_name"] == "person"
    assert per_class_metrics[0]["precision"] == 0.99
    assert "表现优异" in per_class_metrics[0]["analysis"]
    assert per_class_metrics[1]["class_name"] == "cigarette"
    assert per_class_metrics[1]["recall"] == 0.0
    assert "召回率为 0" in per_class_metrics[1]["analysis"]
    assert "1 张图像，3 个实例" in analysis["summary"]
