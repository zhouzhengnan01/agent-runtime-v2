from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.apps import AppTemplateRegistry
from app.core.apps.models import AppModelOption, select_app_model
from app.core.apps.runtime_options import merge_runtime_options_with_template
from app.core.agent import AgentRuntime
from app.core.skills import SkillRegistry
from app.schemas import ChatRequest, Message, RuntimeOptions
from app.main import create_app
from tools.app_smoke_matrix import expected_artifact_patterns, template_expects_artifacts


def test_app_template_registry_lists_and_gets_templates(tmp_path: Path) -> None:
    templates_path = tmp_path / "config" / "apps" / "templates.json"
    templates_path.parent.mkdir(parents=True)
    templates_path.write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "name": "demo",
                        "title": "Demo App",
                        "agent_name": "default",
                        "selected_skills": ["markdown-rendering"],
                        "model_tags": ["chat", "tool_calling", "unknown"],
                        "models": [
                            {
                                "name": "gpt-5.5",
                                "features": ["vision", "reasoning", "chat"],
                                "priority_features": ["chat"],
                                "priority": 0,
                                "provider": "openai-compatible",
                                "model": "gpt-5.5",
                                "base_url": "http://127.0.0.1:9100/api/llm/openai/v1/providers/test/",
                                "default_model": "gpt-5.5",
                                "api_key_enc": "enc.fernet.v1.test",
                                "tool_choice": "auto",
                                "temperature": 0.4,
                                "max_tokens": 2048,
                            }
                        ],
                        "prompt_examples": ["hello"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = AppTemplateRegistry(root_dir=tmp_path)

    templates = registry.list()

    assert [template.name for template in templates] == ["demo"]
    assert registry.get("demo").selected_skills == ["markdown-rendering"]
    assert registry.get("demo").model_tags == ["chat", "tool_call"]
    assert [model.model_dump(mode="json", exclude_none=True) for model in registry.get("demo").models] == [
        {
            "name": "gpt-5.5",
            "features": ["vision", "reasoning", "chat"],
            "priority_features": ["chat"],
            "priority": 0,
            "provider": "openai-compatible",
            "model": "gpt-5.5",
            "base_url": "http://127.0.0.1:9100/api/llm/openai/v1/providers/test/",
            "default_model": "gpt-5.5",
            "api_key_enc": "enc.fernet.v1.test",
            "tool_choice": "auto",
            "temperature": 0.4,
            "max_tokens": 2048,
        }
    ]
    payload = registry.get("demo").to_payload()
    assert "api_key" not in payload["models"][0]
    assert "api_key_enc" not in payload["models"][0]


def test_app_template_payload_strips_runtime_option_secrets(tmp_path: Path) -> None:
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "demo.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "title": "Demo App",
                "agent_name": "default",
                "runtime_options": {
                    "model_name": "demo-model",
                    "base_url": "http://demo.local/v1",
                    "api_key": "plain-key",
                    "api_key_enc": "enc.fernet.v1.test",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = AppTemplateRegistry(root_dir=tmp_path).get("demo").to_payload()

    assert payload["runtime_options"] == {
        "model_name": "demo-model",
        "base_url": "http://demo.local/v1",
    }


def test_app_template_registry_deduplicates_collection_and_file_templates(tmp_path: Path) -> None:
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "templates.json").write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "name": "demo",
                        "title": "Old Demo App",
                        "agent_name": "default",
                        "selected_skills": ["markdown-rendering"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (apps_dir / "demo.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "title": "New Demo App",
                "agent_name": "default",
                "selected_skills": ["drawio-generation"],
            }
        ),
        encoding="utf-8",
    )
    (apps_dir / "._templates.json").write_bytes(b"\x00\xffnot-json")
    registry = AppTemplateRegistry(root_dir=tmp_path)

    templates = registry.list()

    assert [template.name for template in templates] == ["demo"]
    assert templates[0].title == "New Demo App"
    assert registry.get("demo").selected_skills == ["drawio-generation"]


def test_app_template_registry_rejects_unknown_template(tmp_path: Path) -> None:
    registry = AppTemplateRegistry(root_dir=tmp_path)

    with pytest.raises(KeyError):
        registry.get("missing")


def test_app_runtime_options_expand_platform_skill_aliases() -> None:
    template = AppTemplateRegistry().get("general-jetlinks-assistant")

    options = merge_runtime_options_with_template(
        RuntimeOptions(app_template_name="general-jetlinks-assistant", selected_skills=["1778483741456a5glxkmk"]),
        template,
    )

    assert options.selected_skills == [
        "algorithm-engineer",
        "dataset-curator",
        "data-auto-annotation",
        "image-dataset-generation",
        "algorithm-research-scout",
        "model-candidate-selector",
        "remote-gpu-ops",
        "gpu-training-orchestrator",
        "cpu-training-runner",
        "detector-evaluator",
        "deployment-candidate-reviewer",
        "experiment-ledger",
    ]


def test_request_selected_skills_expand_platform_skill_aliases(tmp_path: Path) -> None:
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "demo.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "title": "Demo App",
                "agent_name": "default",
                "selected_skills": ["markdown-rendering"],
            }
        ),
        encoding="utf-8",
    )
    template = AppTemplateRegistry(root_dir=tmp_path).get("demo")

    options = merge_runtime_options_with_template(
        RuntimeOptions(
            app_template_name="demo",
            selected_skills=["1778483741456a5glxkmk", "experiment-ledger"],
        ),
        template,
    )

    assert options.selected_skills[:2] == ["algorithm-engineer", "dataset-curator"]
    assert options.selected_skills[-1] == "experiment-ledger"
    assert options.selected_skills.count("experiment-ledger") == 1


def test_app_templates_api_lists_preconfigured_templates() -> None:
    client = TestClient(create_app())

    response = client.get("/api/apps/templates")

    assert response.status_code == 200
    templates = response.json()["templates"]
    names = {template["name"] for template in templates}
    assert "general-jetlinks-assistant" in names
    assert "behavior-safety-detector" in names
    assert all(template["agent_name"] for template in templates)


def test_preconfigured_app_templates_reference_existing_capabilities() -> None:
    registry = AppTemplateRegistry()

    assert registry.validate_references() == []


def test_preconfigured_app_template_skills_have_local_entities() -> None:
    registry = AppTemplateRegistry()
    config_skills = {path.stem for path in (registry.root_dir / "config" / "skills").glob("*.json")}

    missing = sorted(
        {
            skill_name
            for template in registry.list()
            for skill_name in template.selected_skills
            if skill_name not in config_skills
        }
    )

    assert missing == []


def test_effective_app_template_skills_have_local_entities_after_runtime_expansion() -> None:
    registry = AppTemplateRegistry()
    runtime = AgentRuntime(app_template_registry=registry)
    skill_names = {skill.name for skill in SkillRegistry(registry.root_dir).list()}

    missing_by_template: dict[str, list[str]] = {}
    for template in registry.list():
        runtime_options = merge_runtime_options_with_template(
            RuntimeOptions(app_template_name=template.name, thread_id=f"skill-audit-{template.name}"),
            template,
        )
        effective = runtime._effective_request(
            ChatRequest(
                messages=[Message(role="user", content=template.prompt_examples[0] if template.prompt_examples else "测试")],
                runtime_options=runtime_options,
            )
        )
        missing = sorted(
            {
                skill_name
                for skill_name in effective.runtime_options.selected_skills
                if skill_name not in skill_names
            }
        )
        if missing:
            missing_by_template[template.name] = missing

    assert missing_by_template == {}


def test_preconfigured_app_template_workflows_have_local_entities() -> None:
    registry = AppTemplateRegistry()
    config_workflows = {"agent_loop"} | {path.stem for path in (registry.root_dir / "config" / "workflows").glob("*.json")}

    missing = sorted(
        {
            template.workflow
            for template in registry.list()
            if template.workflow and template.workflow not in config_workflows
        }
    )

    assert missing == []


def test_component_development_app_uses_executable_auto_skill() -> None:
    registry = AppTemplateRegistry()
    template = registry.get("zujiankaifa")
    options = merge_runtime_options_with_template(
        RuntimeOptions(app_template_name=template.name),
        template,
    )
    skill = SkillRegistry(registry.root_dir).get(options.selected_skills[0])

    assert options.selected_skills == ["jetlinks-ai-component"]
    assert skill.executable is True
    assert skill.output_kind == "component"
    assert options.config_options["auto_execute_primary_skill"] is True


def test_preconfigured_app_templates_carry_model_defaults() -> None:
    registry = AppTemplateRegistry()

    for template in registry.list():
        if template.name == "algorithm-engineer-full-cycle":
            assert template.runtime_options["mode"] == "yolo"
            assert template.runtime_options["config_options"]["max_tool_rounds"] == 1000
        elif template.name == "behavior-review":
            assert template.runtime_options["mode"] == "yolo"
            assert template.runtime_options["config_options"]["max_tool_rounds"] == 1000
        elif template.name == "algorithm-engineer-workbench":
            assert template.runtime_options["mode"] == "yolo"
            assert template.runtime_options["config_options"]["max_tool_rounds"] == 16
        elif template.name == "reference-image-yolo-training":
            assert template.runtime_options == {"mode": "yolo", "config_options": {"max_tool_rounds": 16}}
        elif template.name in {
            "70aaee52-99c2-49f5-a9c7-fb746821d3df",
            "0bb9536b-8a36-40fd-8c5b-ea22804b55ab",
            "737d9452-99c6-470e-a77a-20f2f7d73eff",
            "305fb466-1f7f-442b-861e-02e3246f8563",
        }:
            assert template.runtime_options == {
                "mode": "safe",
                "config_options": {
                    "max_tool_rounds": 3,
                    "max_empty_response_retries": 0,
                    "auto_execute_primary_skill": True,
                },
            }
        elif template.name == "algorithm-engineer-full-cycle-test":
            assert template.runtime_options["model_env"] == "LLM_MODEL"
            assert template.runtime_options["base_url_env"] == "LLM_BASE_URL"
            assert template.runtime_options["api_key"] == "abc@123"
            assert template.models == []
            continue
        elif template.name == "mcp-test-users-loop":
            assert template.runtime_options["mode"] == "safe"
            assert template.runtime_options["config_options"]["mcpServers"][0]["name"] == "db"
        else:
            assert template.runtime_options == {}, template.name
        assert len(template.models) == 1, template.name
        model = template.models[0]
        assert model.name, template.name
        assert model.model, template.name
        assert model.default_model, template.name
        assert model.base_url, template.name
        if template.name == "305fb466-1f7f-442b-861e-02e3246f8563":
            assert model.api_key is None, template.name
            assert model.api_key_env == "JETLINKS_APP_305FB466_GPT54_API_KEY", template.name
        else:
            assert model.api_key == "abc@123", template.name
            assert model.api_key_env is None, template.name
        assert model.api_key_enc is None, template.name
        assert model.temperature == 0.4, template.name
        assert model.max_tokens == 2048, template.name


def test_preconfigured_app_templates_carry_model_tags() -> None:
    registry = AppTemplateRegistry()
    allowed = {
        "chat",
        "reasoning",
        "vision",
        "embedding",
        "tool_call",
        "image_generation",
        "video_generation",
        "audio_generation",
        "text_to_speech",
        "speech_to_text",
        "vision_segmentation",
        "rerank",
    }

    for template in registry.list():
        if template.model_tags:
            assert set(template.model_tags) <= allowed, template.name
        assert "tool_calling" not in template.model_tags, template.name

    assert AppTemplateRegistry().get("data-auto-annotation").model_tags == [
        "chat",
        "vision_segmentation",
        "tool_call",
    ]


def test_app_model_selection_uses_priority_features_and_priority() -> None:
    models = [
        AppModelOption(name="vision-low", model="vision-low", features=["vision", "chat"], priority=1),
        AppModelOption(name="chat-high", model="chat-high", features=["chat"], priority_features=["chat"], priority=10),
        AppModelOption(name="chat-low", model="chat-low", features=["chat"], priority_features=["chat"], priority=0),
    ]

    selected = select_app_model(models, model_type="chat")

    assert selected is not None
    assert selected.model == "chat-low"


def test_data_auto_annotation_template_does_not_preselect_mcp_tools() -> None:
    template = AppTemplateRegistry().get("data-auto-annotation")

    assert template.selected_skills == ["data-auto-annotation"]
    assert template.selected_mcp_tools == []


def test_preconfigured_app_templates_do_not_preselect_mcp_tools() -> None:
    registry = AppTemplateRegistry()

    assert {template.name: template.selected_mcp_tools for template in registry.list()} == {
        template.name: [] for template in registry.list()
    }


def test_preconfigured_app_templates_do_not_preselect_artifact_workflow() -> None:
    registry = AppTemplateRegistry()

    assert {
        template.name: template.workflow
        for template in registry.list()
        if template.workflow == "artifact_workflow"
    } == {}


def test_behavior_review_template_enables_continuous_second_pass_policy() -> None:
    template = AppTemplateRegistry().get("behavior-review")

    assert template.workflow is None
    assert template.title == "行为识别连续复判"
    assert template.runtime_options["mode"] == "yolo"
    assert template.runtime_options["config_options"]["max_tool_rounds"] == 1000
    policy = template.runtime_options["config_options"]["agent_execution_policy"]
    assert policy["mode"] == "continuous_review"
    assert policy["stage_order"] == [
        "alert_ingest",
        "evidence_integrity_check",
        "second_pass_judgement",
        "false_positive_suppression",
        "risk_grade",
        "action_recommendation",
        "review_ledger",
    ]
    conditional = template.runtime_options["config_options"]["conditional_tool_loop"]
    assert conditional["continue_while"] == {
        "json_path": "$.pending_count",
        "operator": "gt",
        "expected": 0,
    }
    assert conditional["poll_interval_seconds"] == 3
    assert "tool_call" in template.models[0].features


def test_algorithm_engineer_full_cycle_selects_full_stage_skill_chain() -> None:
    template = AppTemplateRegistry().get("algorithm-engineer-full-cycle")

    assert template.workflow is None
    assert template.title == "算法工程师正式版"
    assert template.runtime_options["mode"] == "yolo"
    assert template.runtime_options["config_options"]["max_tool_rounds"] == 1000
    policy = template.runtime_options["config_options"]["agent_execution_policy"]
    assert policy["mode"] == "full_cycle_autonomous"
    assert policy["stage_order"][0] == "algorithm-engineer"
    assert policy["stage_order"][-1] == "experiment-ledger"
    assert template.selected_skills == [
        "algorithm-engineer",
        "algorithm-research-scout",
        "dataset-curator",
        "data-auto-annotation",
        "image-dataset-generation",
        "model-candidate-selector",
        "remote-gpu-ops",
        "gpu-training-orchestrator",
        "cpu-training-runner",
        "detector-evaluator",
        "deployment-candidate-reviewer",
        "experiment-ledger",
    ]


def test_algorithm_cpu_training_sandbox_template_selects_training_runner() -> None:
    template = AppTemplateRegistry().get("algorithm-cpu-training-sandbox")

    assert template.workflow is None
    assert template.category == "algorithm-training"
    assert template.selected_skills == ["cpu-training-runner"]


def test_algorithm_training_related_templates_use_dedicated_category() -> None:
    registry = AppTemplateRegistry()
    training_templates = {
        "algorithm-cpu-training-sandbox",
        "algorithm-dataset-curation",
        "algorithm-engineer-full-cycle",
        "algorithm-evaluation-deployment",
        "algorithm-research-benchmark",
        "algorithm-training-orchestration",
    }

    categories = {name: registry.get(name).category for name in training_templates}
    workflows = {name: registry.get(name).workflow for name in training_templates}

    assert categories == {name: "algorithm-training" for name in training_templates}
    assert workflows["algorithm-training-orchestration"] == "agent_loop"
    assert {name: workflow for name, workflow in workflows.items() if name != "algorithm-training-orchestration"} == {
        name: None for name in training_templates if name != "algorithm-training-orchestration"
    }


def test_tianjin_park_templates_use_dedicated_category_and_real_skills() -> None:
    registry = AppTemplateRegistry()
    park_templates = {
        "tianjin-business-docs",
        "tianjin-content-ops",
        "tianjin-data-analyst",
        "tianjin-meeting-efficiency",
        "tianjin-park-assistant",
        "tianjin-public-opinion",
        "tianjin-rpa-ops",
    }

    categories = {name: registry.get(name).category for name in park_templates}
    workflows = {name: registry.get(name).workflow for name in park_templates}
    selected_skills = {
        skill_name
        for name in park_templates
        for skill_name in registry.get(name).selected_skills
    }
    configured_skills = {skill.name for skill in SkillRegistry(registry.root_dir).list(executable_only=True)}

    assert categories == {name: "park-operations" for name in park_templates}
    assert workflows["tianjin-business-docs"] is None
    assert {name: workflow for name, workflow in workflows.items() if name != "tianjin-business-docs"} == {
        name: "agent_loop" for name in park_templates if name != "tianjin-business-docs"
    }
    assert {
        "tianjin-chatbi-analyst",
        "tianjin-document-generator",
        "tianjin-sentiment-monitor",
        "tianjin-rpa-operator",
    }.issubset(selected_skills)
    assert selected_skills <= configured_skills


def test_app_template_reference_validation_uses_installed_workflow_plugins(tmp_path: Path) -> None:
    (tmp_path / "config" / "agents").mkdir(parents=True)
    (tmp_path / "config" / "agents" / "default.json").write_text(
        json.dumps({"name": "default", "display_name": "Default Agent"}),
        encoding="utf-8",
    )
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "custom.json").write_text(
        json.dumps(
            {
                "name": "custom",
                "title": "Custom Workflow App",
                "agent_name": "default",
                "workflow": "custom_workflow",
            }
        ),
        encoding="utf-8",
    )
    plugin_root = tmp_path / "plugins" / "workflows" / "custom-workflow"
    workflow_dir = plugin_root / "workflows" / "custom"
    workflow_dir.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        json.dumps({"id": "custom-workflow", "name": "Custom Workflow", "workflows": ["workflows/*/workflow.json"]}),
        encoding="utf-8",
    )
    (workflow_dir / "workflow.json").write_text(
        json.dumps({"name": "custom_workflow", "display_name": "Custom Workflow", "handler": "module.py:CustomWorkflow"}),
        encoding="utf-8",
    )
    config_workflows = tmp_path / "config" / "workflows"
    config_workflows.mkdir(parents=True)
    (config_workflows / "custom_workflow.json").write_text(
        json.dumps({"name": "custom_workflow", "display_name": "Custom Workflow", "handler": "module.py:CustomWorkflow"}),
        encoding="utf-8",
    )
    (plugin_root / "module.py").write_text(
        """
class CustomWorkflow:
    def run_with_events(self, *args, **kwargs):
        raise NotImplementedError
""".strip(),
        encoding="utf-8",
    )

    assert AppTemplateRegistry(root_dir=tmp_path).validate_references() == []

    (apps_dir / "custom.json").write_text(
        json.dumps(
            {
                "name": "custom",
                "title": "Custom Workflow App",
                "agent_name": "default",
                "workflow": "missing_workflow",
            }
        ),
        encoding="utf-8",
    )

    assert AppTemplateRegistry(root_dir=tmp_path).validate_references() == [
        "custom: unknown workflow missing_workflow"
    ]


def test_preconfigured_app_templates_have_no_duplicate_names_or_titles() -> None:
    registry = AppTemplateRegistry()
    templates = registry.list()
    app_files = [path for path in (registry.root_dir / "config" / "apps").glob("*.json") if path.name != "templates.json"]
    names = [template.name for template in templates]

    assert len(templates) == len(app_files)
    assert len(names) == len(set(names))


def test_preconfigured_app_templates_have_smoke_artifact_expectations() -> None:
    registry = AppTemplateRegistry()

    missing = [
        template.name
        for template in registry.list()
        if template_expects_artifacts(template.model_dump())
        and len(template.selected_skills) <= 1
        and not expected_artifact_patterns(template.model_dump())
    ]

    assert missing == []
