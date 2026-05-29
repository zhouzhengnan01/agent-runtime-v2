from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path


def _load_yolo_training_flow_module():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    path = Path("plugins/workflows/builtin-artifact-workflows/yolo_training_flow.py").resolve()
    spec = importlib.util.spec_from_file_location("yolo_training_flow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yolo_training_flow_extracts_one_message_training_spec() -> None:
    module = _load_yolo_training_flow_module()
    text = (
        "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b\uff0c"
        "\u5408\u6210\u63d0\u793a\u8bcd\u4e3a\uff1a"
        "\u628a image2 \u4e2d\u7684\u76ee\u6807\u81ea\u7136\u5408\u6210\u5230 image1 "
        "\u7684\u573a\u666f\u4e2d\uff0c\u5f62\u6210\u771f\u5b9e\u76d1\u63a7\u753b\u9762\uff0c"
        " \u6807\u6ce8\u7c7b\u522b\u4e3a\uff1aperson\uff0ccigarette\uff0c"
        "\u8bad\u7ec3\u53c2\u6570\u4e3a\uff1a"
        "conda_env_name=yolo_jetson model=yolo11n.pt epochs=10 imgsz=640 batch=8 "
        "device=0 workers=4 patience=20 dataset.split.train=0.7 "
        "dataset.split.val=0.2 dataset.split.test=0.1"
    )

    assert module._extract_generation_prompt(text) == "把 image2 中的目标自然合成到 image1 的场景中，形成真实监控画面"
    assert module._extract_annotation_labels(text) == ["person", "cigarette"]
    assert module._has_explicit_training_config(text) is True
    training_cfg = module._extract_training_config(text)
    assert training_cfg["runtime"]["conda_env_name"] == "yolo_jetson"
    assert training_cfg["training"]["epochs"] == 10
    assert training_cfg["split"] == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_yolo_training_flow_normalizes_llm_extracted_spec() -> None:
    module = _load_yolo_training_flow_module()
    spec = {
        "training": {
            "task": "detection",
            "model": "yolo11n.pt",
            "epochs": "10",
            "imgsz": "640",
            "batch": "8",
            "device": "0",
            "workers": "4",
            "patience": "20",
        },
        "runtime": {"conda_env_name": "yolo_jetson"},
        "split": {"train": "0.7", "val": "0.2", "test": "0.1"},
    }

    training_cfg = module._spec_training_config(spec)

    assert training_cfg["runtime"]["conda_env_name"] == "yolo_jetson"
    assert training_cfg["training"]["task"] == "detect"
    assert training_cfg["training"]["batch"] == 8
    assert training_cfg["split"]["test"] == 0.1


def test_yolo_training_flow_strips_runtime_attachment_context_from_user_text() -> None:
    module = _load_yolo_training_flow_module()
    messages = [
        module.Message(
            role="user",
            content=(
                "hello\n\n"
                "Uploaded files available to tools:\n"
                "1. name=best.pt, path=/mnt/user-data/outputs/yolo_training_flow/training_run/train/weights/best.pt\n"
                "2. name=dataset.yaml, path=/mnt/user-data/outputs/yolo_training_flow/prepared_data/dataset.yaml"
            ),
        )
    ]

    assert module._last_user_text(messages) == "hello"
    assert module._looks_like_yolo_training_request(module._last_user_text(messages)) is False


def test_yolo_training_flow_strips_workbench_capability_context_from_user_text() -> None:
    module = _load_yolo_training_flow_module()
    messages = [
        module.Message(
            role="user",
            content=(
                "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b\n\n"
                "[Workbench selected capabilities]\n"
                "Selected Skills: data-auto-annotation, image-dataset-generation, gpu-training-orchestrator"
            ),
        )
    ]

    assert module._last_user_text(messages) == "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b"


def test_yolo_training_flow_ignores_thread_file_dataset_as_new_upload() -> None:
    module = _load_yolo_training_flow_module()
    attachments = [
        module.Attachment(
            name="dataset.zip",
            path="/mnt/user-data/uploads/dataset.zip",
            metadata={"thread_file": True},
        )
    ]

    assert module._has_dataset_attachment(attachments, include_thread_files=False) is False
    assert module._find_dataset_package_attachment(attachments, include_thread_files=False) is None


def test_yolo_training_flow_replaces_generic_model_label_from_vehicle_intent() -> None:
    module = _load_yolo_training_flow_module()
    spec = {"labels": ["object"]}

    fixed = module._ensure_intent_labels(spec, "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u8f66\u8f86\u68c0\u6d4b\u6a21\u578b")

    assert fixed["labels"] == ["car"]


def test_yolo_training_flow_preserves_explicit_label_over_model_object() -> None:
    module = _load_yolo_training_flow_module()
    spec = {"labels": ["object"]}

    fixed = module._ensure_intent_labels(
        spec,
        "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u8f66\u8f86\u68c0\u6d4b\u6a21\u578b\uff0c\u6807\u6ce8\u7c7b\u522b\u4e3a: car",
    )

    assert fixed["labels"] == ["car"]


def test_yolo_training_flow_infers_labels_from_bottle_and_steel_intents() -> None:
    module = _load_yolo_training_flow_module()

    bottle = module._ensure_intent_labels(
        {"labels": ["target"]},
        "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u74f6\u5b50\u68c0\u6d4b\u6a21\u578b",
    )
    steel = module._ensure_intent_labels(
        {"labels": ["\u76ee\u6807"]},
        "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u94a2\u6750\u68c0\u6d4b\u6a21\u578b",
    )

    assert bottle["labels"] == ["bottle"]
    assert steel["labels"] == ["steel"]


def test_yolo_training_flow_overrides_defect_label_from_smoking_intent() -> None:
    module = _load_yolo_training_flow_module()
    spec = {
        "task_description": "\u9ed8\u8ba4\u89e3\u6790\u4e3a\u7f3a\u9677/\u5f02\u7269\u68c0\u6d4b\uff0c\u6807\u7b7e\u8bbe\u4e3a defect\u3002",
        "generation_prompt": "\u5c06 image2.zip \u4e2d\u7684\u76ee\u6807\u7269\uff08defect\uff09\u81ea\u7136\u5408\u6210\u5230 image1.zip \u63d0\u4f9b\u7684\u80cc\u666f\u573a\u666f\u4e2d\u3002",
        "labels": ["defect"],
    }

    fixed = module._ensure_intent_labels(spec, "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b")

    assert fixed["labels"] == ["person", "cigarette"]
    assert "defect" not in fixed["generation_prompt"]
    assert "person, cigarette" in fixed["generation_prompt"]


def test_yolo_training_flow_normalizes_chinese_face_label_to_yolo_safe_english() -> None:
    module = _load_yolo_training_flow_module()
    spec = {
        "task_description": "\u57fa\u4e8e\u5408\u6210\u6570\u636e\u8bad\u7ec3YOLO\u4eba\u8138\u68c0\u6d4b\u6a21\u578b",
        "generation_prompt": "\u8bf7\u5c06 image2.zip \u4e2d\u7684\u4eba\u8138\u56fe\u50cf\u4f5c\u4e3a\u524d\u666f\u76ee\u6807\u81ea\u7136\u5408\u6210\u5230 image1.zip \u63d0\u4f9b\u7684\u80cc\u666f\u573a\u666f\u4e2d\u3002",
        "labels": ["\u4eba\u8138"],
    }

    fixed = module._ensure_intent_labels(spec, "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u4eba\u8138\u68c0\u6d4b\u6a21\u578b")

    assert fixed["labels"] == ["face"]
    assert module._normalize_detection_labels(["\u4eba\u8138", "\u672a\u77e5\u4e2d\u6587\u7c7b\u522b"]) == ["face"]


def test_yolo_training_flow_saves_annotation_labels_as_yolo_safe_english(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = SimpleNamespace(workspace=workspace)

    module._save_annotation_labels(paths, ["\u4eba\u8138"])

    assert module._load_annotation_labels(paths) == ["face"]
    assert (workspace / "annotation_labels.json").read_text(encoding="utf-8") == '[\n  "face"\n]'


def test_yolo_training_flow_reuses_training_objective_for_continue_text() -> None:
    module = _load_yolo_training_flow_module()

    combined = module._combine_spec_user_text(
        "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b",
        "\u7ee7\u7eed\u5904\u7406\u521a\u4e0a\u4f20\u7684\u6587\u4ef6\u3002",
    )

    assert combined == "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b"
    assert module._infer_labels_from_training_intent(combined) == ["person", "cigarette"]


def test_yolo_training_flow_loads_training_objective_from_thread_memory(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    root = tmp_path
    workspace = root / "workspace"
    memory = root / "memory"
    workspace.mkdir()
    memory.mkdir()
    (memory / "conversation.jsonl").write_text(
        (
            '{"message":{"role":"user","content":"'
            '\\u5e2e\\u6211\\u8bad\\u7ec3\\u4e00\\u4e2aYOLO\\u62bd\\u70df\\u68c0\\u6d4b\\u6a21\\u578b\\n\\n'
            '[Workbench selected capabilities]\\nSelected Skills: data-auto-annotation"}}\n'
            '{"message":{"role":"user","content":"\\u7ee7\\u7eed\\u5904\\u7406\\u521a\\u4e0a\\u4f20\\u7684\\u6587\\u4ef6\\u3002"}}\n'
        ),
        encoding="utf-8",
    )
    paths = SimpleNamespace(root=root, workspace=workspace)

    objective = module._load_training_objective(paths)

    assert objective == "\u5e2e\u6211\u8bad\u7ec3\u4e00\u4e2aYOLO\u62bd\u70df\u68c0\u6d4b\u6a21\u578b"
    assert (workspace / "training_objective.txt").read_text(encoding="utf-8") == objective
