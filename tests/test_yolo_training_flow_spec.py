from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from app.core.agent.input_required import required_inputs_for_request
from app.core.artifacts import ArtifactStore
from app.core.config.agent_config import AgentConfig
from app.schemas import Attachment, ChatRequest, Message, RuntimeOptions


def _load_yolo_training_flow_module():
    path = Path("plugins/workflows/builtin-artifact-workflows/yolo_training_flow.py").resolve()
    spec = importlib.util.spec_from_file_location("yolo_training_flow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_data_preparation_pipeline_module():
    path = Path("plugins/skills/data-auto-annotation/scripts/run_data_preparation_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("run_data_preparation_pipeline", path)
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


def test_yolo_training_flow_rejects_zero_test_split_from_model_spec() -> None:
    module = _load_yolo_training_flow_module()
    spec = {
        "training": {
            "task": "detect",
            "model": "yolov8n.pt",
            "epochs": 100,
            "imgsz": 640,
            "batch": 4,
            "device": "0",
            "workers": 4,
            "patience": 20,
        },
        "runtime": {"conda_env_name": "", "enforce_conda_env": False},
        "split": {"train": 0.8, "val": 0.2, "test": 0.0},
    }

    training_cfg = module._spec_training_config(spec)

    assert training_cfg["split"] == {"train": 0.7, "val": 0.2, "test": 0.1}


def test_yolo_training_flow_fallback_keeps_test_split_for_small_dataset() -> None:
    module = _load_yolo_training_flow_module()

    training_cfg = module._fallback_training_from_dataset_facts({"image_count": 14})

    assert training_cfg["split"]["test"] == 0.1


def test_yolo_training_flow_reads_max_synthetic_images_from_runtime_options() -> None:
    module = _load_yolo_training_flow_module()

    assert module._max_synthetic_images(RuntimeOptions(max_synthetic_images=8)) == 8
    assert (
        module._max_synthetic_images(
            RuntimeOptions(max_synthetic_images=module.DEFAULT_MAX_SYNTHETIC_IMAGES + 1)
        )
        == module.DEFAULT_MAX_SYNTHETIC_IMAGES
    )
    assert (
        module._max_synthetic_images(
            RuntimeOptions(
                skill_parameters={"yolo_training_flow": {"maxSyntheticImages": 8}},
            )
        )
        == 8
    )


def test_image_dataset_produce_count_respects_max_synthetic_cap() -> None:
    module = _load_data_preparation_pipeline_module()
    plan = {
        "recommended_synthetic_count": 20,
        "generation_plan": [{"count": 20}],
    }

    assert module._produce_synthetic_count(plan, 100, True, 5, 12) == 12
    assert module._produce_synthetic_count(plan, 100, True, 5, 0) == 0


def test_yolo_training_flow_epochs_button_keeps_model_value_when_enabled() -> None:
    module = _load_yolo_training_flow_module()
    training_cfg = {"training": {"epochs": 80}}
    runtime_options = RuntimeOptions(
        skill_parameters={"yolo_training_flow": {"button_epochs": True}},
    )

    module._apply_epochs_policy(training_cfg, runtime_options)

    assert training_cfg["training"]["epochs"] == 80


def test_yolo_training_flow_epochs_button_uses_fixed_value_when_disabled() -> None:
    module = _load_yolo_training_flow_module()
    training_cfg = {"training": {"epochs": 80}}
    runtime_options = RuntimeOptions(
        skill_parameters={"yolo_training_flow": {"button_epochs": False}},
    )

    module._apply_epochs_policy(training_cfg, runtime_options)

    assert training_cfg["training"]["epochs"] == 50


def test_yolo_training_flow_test_app_disables_model_epochs() -> None:
    app_config = Path("config/apps/algorithm-engineer-full-cycle-test.json").read_text(encoding="utf-8")

    assert '"button_epochs": false' in app_config


def test_yolo_training_flow_selects_uploaded_model_by_model_id(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("uploaded-model-thread")
    model_path = paths.uploads / "models" / "custom.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"custom-model")
    sha256 = hashlib.sha256(b"custom-model").hexdigest()
    model_id = "model-custom"
    registry = {
        "models": {
            model_id: {
                "modelId": model_id,
                "name": "custom.pt",
                "path": "/mnt/user-data/uploads/models/custom.pt",
                "local_path": str(model_path.resolve()),
                "source": "user_upload",
                "sha256": sha256,
            }
        }
    }
    (paths.workspace / "training_models.json").write_text(
        json.dumps(registry, ensure_ascii=False),
        encoding="utf-8",
    )

    selected, error = module._resolve_user_training_model(paths, model_id)

    assert error == ""
    assert selected["modelId"] == model_id
    assert selected["local_path"] == str(model_path.resolve())
    assert selected["sha256"] == sha256
    training_cfg = {"training": {"model": "yolo11n.pt"}}
    module._apply_user_training_model(training_cfg, selected)
    assert training_cfg["training"]["model"] == str(model_path.resolve())
    assert training_cfg["training"]["model_source"] == "user_upload"
    assert training_cfg["training"]["strict_model"] is True
    assert training_cfg["training"]["model_id"] == model_id

    automatic, automatic_error = module._resolve_user_training_model(paths, "")
    assert automatic_error == ""
    assert automatic is None


def test_yolo_training_flow_new_dataset_attachment_overrides_stale_marker(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("stale-dataset-thread")
    stale_path = paths.uploads / "dataset.zip"
    current_path = paths.uploads / "datasets.zip"
    current_path.write_bytes(b"current")
    module._save_dataset_package_path(paths, str(stale_path))

    attachment = Attachment(
        name="uploads/dataset.zip",
        path=".runtime/threads/stale-dataset-thread/uploads/datasets.zip",
        mime_type="application/zip",
        metadata={"others": {"role": "dataset"}},
    )
    dataset_attachment = module._find_dataset_package_attachment(
        [attachment],
        role_hints={},
        include_thread_files=False,
    )
    assert dataset_attachment is not None

    resolved = module._resolve_uploaded_local_path(paths.root, dataset_attachment.path)
    module._save_dataset_package_path(paths, str(resolved))

    assert module._load_dataset_package_path(paths) == str(current_path.resolve())


def test_yolo_training_flow_rejects_model_id_outside_thread_models(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("uploaded-model-thread")
    outside_model = tmp_path / "outside.pt"
    outside_model.write_bytes(b"outside-model")
    model_id = "model-outside"
    registry = {
        "models": {
            model_id: {
                "modelId": model_id,
                "name": "outside.pt",
                "path": str(outside_model),
                "source": "user_upload",
            }
        }
    }
    (paths.workspace / "training_models.json").write_text(
        json.dumps(registry, ensure_ascii=False),
        encoding="utf-8",
    )

    selected, error = module._resolve_user_training_model(paths, model_id)

    assert selected is None
    assert "uploads/models" in error


def test_yolo_training_flow_rejects_unknown_model_id(tmp_path: Path) -> None:
    module = _load_yolo_training_flow_module()
    paths = ArtifactStore(root_dir=tmp_path).prepare_thread("uploaded-model-thread")

    selected, error = module._resolve_user_training_model(paths, "model-missing")

    assert selected is None
    assert "model-missing" in error


def test_yolo_training_flow_reads_training_model_id_from_runtime_options() -> None:
    module = _load_yolo_training_flow_module()

    assert (
        module._requested_training_model_id(RuntimeOptions(training_model_id="model-direct"))
        == "model-direct"
    )
    assert (
        module._requested_training_model_id(
            RuntimeOptions(
                skill_parameters={"yolo_training_flow": {"modelId": "model-nested"}},
            )
        )
        == "model-nested"
    )


def test_yolo_training_flow_without_model_id_clears_previous_uploaded_model() -> None:
    module = _load_yolo_training_flow_module()
    training_cfg = {
        "training": {
            "model": "D:/runtime/uploads/models/previous.pt",
            "model_source": "user_upload",
            "strict_model": True,
            "model_sha256": "old-sha",
            "model_original_name": "previous.pt",
            "model_id": "model-previous",
        }
    }
    request_spec = {"training": {"model": "yolo11s.pt"}}

    module._clear_user_training_model_selection(training_cfg, request_spec)

    assert training_cfg["training"] == {"model": "yolo11s.pt"}


def test_acp_runtime_options_keep_llm_model_id_separate_from_training_model_id() -> None:
    from app.protocols.acp.adapter import _runtime_options_from_source

    payload = _runtime_options_from_source(
        {
            "modelId": "llm-model",
            "trainingModelId": "model-uploaded",
            "maxSyntheticImages": 50,
            "skillParameters": {"yolo_training_flow": {"modelId": "model-nested"}},
        }
    )

    assert payload["model_name"] == "llm-model"
    assert payload["training_model_id"] == "model-uploaded"
    assert payload["max_synthetic_images"] == 50
    assert payload["skill_parameters"]["yolo_training_flow"]["modelId"] == "model-nested"


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
