from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from app.core.agent.input_required import required_inputs_for_request
from app.core.artifacts import ArtifactStore
from app.core.config.agent_config import AgentConfig
from app.schemas import ChatRequest, Message, RuntimeOptions


def _load_yolo_training_flow_module():
    path = Path("plugins/workflows/builtin-artifact-workflows/yolo_training_flow.py").resolve()
    spec = importlib.util.spec_from_file_location("yolo_training_flow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_data_preparation_pipeline_module():
    path = Path("plugins/skills/data-auto-annotation/scripts/run_data_preparation_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("run_data_preparation_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_intent_uses_training_label_when_business_label_is_not_ascii() -> None:
    module = _load_yolo_training_flow_module()
    spec = {
        "labels": ["security_guard"],
        "intent_items": [
            {
                "label": "安保人员",
                "business_label": "安保人员",
                "type": "person_attribute_detection",
                "training_labels": ["security_guard"],
                "base_entity": "person",
                "observable_entities": ["person", "uniform"],
                "annotation_strategy": "constrained_target",
                "primary_sam3_prompt": "person wearing security uniform",
                "sam3_prompts": ["person wearing security uniform"],
                "sam3_prompt_map": {"person wearing security uniform": "security_guard"},
                "forbidden_direct_prompts": ["person"],
            }
        ],
    }

    items = module._spec_intent_items(spec)

    assert items[0]["label"] == "security_guard"
    assert items[0]["business_label"] == "安保人员"
    assert items[0]["sam3_prompts"] == ["person wearing security uniform"]


def test_semantic_intent_keeps_llm_plan_without_merging_behavior_rule_prompts() -> None:
    flow = _load_yolo_training_flow_module()
    preparation = _load_data_preparation_pipeline_module()
    spec = {
        "task_type": "state_detection",
        "labels": ["fallen_person"],
        "intent_items": [
            {
                "label": "person_fall",
                "type": "state_detection",
                "training_labels": ["fallen_person"],
                "base_entity": "person",
                "required_states": ["fallen"],
                "observable_entities": ["person", "ground_surface"],
                "annotation_strategy": "constrained_target",
                "primary_sam3_prompt": "person falling or lying on ground",
                "sam3_prompts": ["person falling or lying on ground"],
                "sam3_prompt_map": {"person falling or lying on ground": "fallen_person"},
                "forbidden_direct_prompts": ["person"],
            }
        ],
    }

    ensured = flow._ensure_intent_labels(spec, "帮我训练一个人员跌倒检测模型")
    items = flow._spec_intent_items(ensured)
    prompt_map = flow._annotation_prompt_map_from_spec(ensured, ["fallen_person"])
    prompts = preparation._annotation_prompts_for_sam3(
        ["fallen_person"], list(prompt_map), prompt_map, items
    )

    assert prompts == ["person falling or lying on ground"]
    assert "person" not in prompt_map


def test_semantic_intent_keeps_one_sam3_prompt_for_each_training_label() -> None:
    flow = _load_yolo_training_flow_module()
    preparation = _load_data_preparation_pipeline_module()
    labels = ["garbage", "abandoned_item", "debris"]
    spec = {
        "labels": labels,
        "intent_items": [
            {
                "label": "garbage_item_anomaly",
                "type": "anomaly_detection",
                "training_labels": labels,
                "base_entity": "object",
                "required_states": ["discarded", "misplaced"],
                "annotation_strategy": "constrained_target",
                "primary_sam3_prompt": "discarded garbage on ground",
                "sam3_prompts": ["discarded garbage on ground"],
                "sam3_prompt_map": {
                    "discarded garbage on ground": "garbage",
                    "left-behind or misplaced personal items": "abandoned_item",
                    "scattered debris or broken objects": "debris",
                },
                "forbidden_direct_prompts": ["object", "item"],
            }
        ],
    }

    items = flow._spec_intent_items(spec)
    prompt_map = flow._annotation_prompt_map_from_spec(spec, labels)
    prompts = preparation._annotation_prompts_for_sam3(labels, list(prompt_map), prompt_map, items)
    effective_map = preparation._prompt_label_map_for_sam3(labels, prompts, prompt_map)

    assert {prompt: effective_map[prompt] for prompt in prompts} == {
        "discarded garbage on ground": "garbage",
        "left-behind or misplaced personal items": "abandoned_item",
        "scattered debris or broken objects": "debris",
    }


def test_smoking_intent_forces_entity_annotation_policy() -> None:
    flow = _load_yolo_training_flow_module()
    preparation = _load_data_preparation_pipeline_module()
    spec = {
        "task_type": "behavior_detection",
        "labels": ["smoking_person"],
        "intent_items": [
            {
                "label": "smoking_person",
                "type": "behavior_detection",
                "training_labels": ["smoking_person"],
                "base_entity": "person",
                "behavior": "smoking",
                "annotation_strategy": "constrained_target",
                "primary_sam3_prompt": "smoking person",
                "sam3_prompts": ["smoking person"],
                "sam3_prompt_map": {"smoking person": "smoking_person"},
            }
        ],
    }

    ensured = flow._ensure_intent_labels(spec, "train a smoking detection model")
    labels = flow._spec_string_list(ensured, "labels")
    items = flow._spec_intent_items(ensured)
    prompt_map = flow._annotation_prompt_map_from_spec(ensured, labels)
    prompts = preparation._annotation_prompts_for_sam3(labels, list(prompt_map), prompt_map, items)

    assert labels == ["person", "cigarette"]
    assert prompt_map == {"person": "person", "cigarette": "cigarette"}
    assert prompts == ["person", "cigarette"]


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

    assert module._max_synthetic_images(RuntimeOptions(max_synthetic_images=12)) == 12
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


def test_yolo_training_flow_ignores_stale_http_cancel_marker(tmp_path: Path, monkeypatch) -> None:
    module = _load_yolo_training_flow_module()
    monkeypatch.chdir(tmp_path)

    from app.core.http_training_jobs import write_http_training_job_marker

    thread_id = "acp-after-http-cancel"
    paths = SimpleNamespace(thread_id=thread_id)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-old",
            "status": "cancelled",
            "run_id": "run-deimv2-old",
            "error": "old cancel",
        },
    )

    assert module._http_training_cancel_requested(paths, SimpleNamespace(run_id="run-deimv2-new")) is False
    assert module._http_training_cancel_requested(paths, SimpleNamespace(run_id="run-deimv2-old")) is True


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


def test_yolo_training_flow_only_requires_dataset_input(tmp_path: Path) -> None:
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
    assert [item["type"] for item in required_inputs] == ["dataset"]
    assert "数据集" in required_inputs[0]["reason"]


def test_synthetic_generation_requires_complete_image_pair() -> None:
    module = _load_yolo_training_flow_module()

    assert module._effective_synthetic_generation(True, None, None) == (False, "sufficient_data_samples")
    assert module._effective_synthetic_generation(True, "image1.zip", None) == (False, "sufficient_data_samples")
    assert module._effective_synthetic_generation(True, None, "image2.zip") == (False, "sufficient_data_samples")
    assert module._effective_synthetic_generation(True, "image1.zip", "image2.zip") == (True, "")
    assert module._effective_synthetic_generation(False, "image1.zip", "image2.zip") == (False, "not_requested")
