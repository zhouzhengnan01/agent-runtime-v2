from __future__ import annotations

import importlib.util
import sys
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
