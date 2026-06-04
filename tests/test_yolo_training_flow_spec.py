from __future__ import annotations

import importlib.util
from pathlib import Path

from app.core.agent.input_required import required_inputs_for_request
from app.core.artifacts import ArtifactStore
from app.core.config.agent_config import AgentConfig
from app.schemas import ChatRequest, Message, RuntimeOptions


def _load_yolo_training_flow_module():
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


def test_yolo_training_flow_owns_three_zip_input_contract(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    request = ChatRequest(
        messages=[Message(role="user", content="帮我训练一个YOLO抽烟检测模型")],
        runtime_options=RuntimeOptions(
            thread_id="postman-yolo-test-inputs",
            workflow="yolo_training_flow",
            selected_skills=[
                "data-auto-annotation",
                "image-dataset-generation",
                "image-dataset-produce",
                "gpu-training-orchestrator",
            ],
            skill_parameters={"yolo_training_flow": {"auto_generate_missing_spec": True}},
        ),
    )

    assert required_inputs_for_request(request) == []

    workflow = module.YoloTrainingWorkflow(ArtifactStore(root_dir=tmp_path))
    result, _events = workflow.run_with_events(
        agent_config=AgentConfig(name="default", display_name="Default"),
        messages=request.messages,
        attachments=[],
        thread_id=request.runtime_options.thread_id or "postman-yolo-test-inputs",
        runtime_options=request.runtime_options,
        workflow_name="yolo_training_flow",
    )

    required_inputs = result.metadata["required_inputs"]
    assert result.metadata["requires_input"] is True
    assert [item["type"] for item in required_inputs] == ["dataset", "image", "image"]
    assert "数据集" in required_inputs[0]["reason"]
    assert "image1" in required_inputs[1]["reason"]
    assert "image2" in required_inputs[2]["reason"]
