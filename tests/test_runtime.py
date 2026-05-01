import asyncio
from collections.abc import AsyncIterator
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig, AgentConfigLoader
from app.core.llm import OpenAICompatibleClient
from app.core.routing import WorkflowRouter
from app.core.skills import SkillRegistry
from app.core.workflow import WorkflowRegistry
from app.schemas import Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


async def _collect_events(source: AsyncIterator[ChatEvent]) -> list[ChatEvent]:
    return [event async for event in source]


def test_artifact_generator_creates_verified_markdown(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("artifact-generator")
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 markdown 架构说明")],
        runtime_options=RuntimeOptions(thread_id="t1"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.status == "completed"
    assert result.artifacts
    assert result.verification is not None
    assert result.verification.passed is True


def test_agent_with_no_configured_skills_uses_installed_skill_plugins(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="plugin-only-agent",
        display_name="Plugin Only Agent",
        tools=["artifact_workflow"],
        skills=[],
        workflows={"default": "artifact_workflow"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 markdown 插件说明")],
        runtime_options=RuntimeOptions(thread_id="plugin-only-skills"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["skill_name"] == "markdown-rendering"
    assert result.artifacts[0].name == "result.md"


def test_artifact_generator_emits_coded_events(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("artifact-generator")
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 drawio 架构图")],
        runtime_options=RuntimeOptions(thread_id="events"),
    )
    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]
    assert result.status == "completed"
    assert event_types[0] == "run.started"
    assert "spec.completed" in event_types
    assert "skill.started" in event_types
    assert "sandbox.policy" in event_types
    assert "artifact.created" in event_types
    assert "verifier.completed" in event_types
    assert event_types[-1] == "run.completed"
    sandbox_event = next(event for event in events if event.type == "sandbox.policy")
    assert sandbox_event.data["profile_name"] == "drawio"
    assert sandbox_event.data["eligible"] is True
    assert sandbox_event.data["use_sandbox"] is False
    assert "architecture.drawio" in result.reply


def test_default_agent_stream_emits_text_delta(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    agent.model.base_url = None
    agent.model.api_key = None
    request = ChatRequest(
        messages=[Message(role="user", content="你好")],
        runtime_options=RuntimeOptions(thread_id="stream-default"),
    )

    events = asyncio.run(_collect_events(runtime.iter_events(agent, request)))
    event_types = [event.type for event in events]

    assert event_types[:2] == ["run.started", "llm.started"]
    assert "agent.message.delta" in event_types
    assert events[-1].type == "run.completed"


def test_unregistered_workflow_config_falls_back_to_agent_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path),
        workflow_registry=WorkflowRegistry(),
    )
    agent = AgentConfigLoader().load("artifact-generator")
    agent.model.base_url = None
    agent.model.api_key = None
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 markdown 架构说明")],
        runtime_options=RuntimeOptions(thread_id="workflow-plugin-disabled"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.metadata["workflow"] == "agent_loop"
    assert result.artifacts == []
    assert "spec.started" not in [event.type for event in events]


def test_workflow_router_uses_skill_manifest_routing_metadata(monkeypatch) -> None:
    monkeypatch.setenv("LLM_WORKFLOW_ROUTER", "0")
    router = WorkflowRouter(available_workflows={"artifact_workflow", "evidence_first_detection"})
    agent = AgentConfigLoader().load("default")
    skill = SkillRegistry().get("drawio-generation")
    assert skill.routing is not None
    assert "原型图" in skill.routing["keywords"]

    request = ChatRequest(messages=[Message(role="user", content="可以帮我画一个原型图")])

    selection = router.select_with_details(agent, request)

    assert selection.workflow_name == "artifact_workflow"
    assert selection.skill_name == "drawio-generation"
    assert selection.reason == "plugin_score_match"


def test_workflow_router_llm_can_select_from_skill_descriptions(monkeypatch) -> None:
    seen_clients: list[tuple[str, str]] = []

    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        seen_clients.append((self.model, self.base_url))
        assert "Available skills" in messages[0].content
        assert "drawio-generation" in messages[0].content
        return '{"skill_name": "drawio-generation"}'

    monkeypatch.setenv("LLM_WORKFLOW_ROUTER", "1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    router = WorkflowRouter(available_workflows={"artifact_workflow", "evidence_first_detection"})
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(messages=[Message(role="user", content="给我做一个系统蓝图")])

    selection = router.select_with_details(agent, request)

    assert selection.workflow_name == "artifact_workflow"
    assert selection.skill_name == "drawio-generation"
    assert selection.mode == "llm"
    assert seen_clients == [(agent.model.model, agent.model.base_url)]


def test_workflow_router_respects_agent_json_routing_switch(monkeypatch) -> None:
    calls: list[str] = []

    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        calls.append(messages[0].content)
        return '{"skill_name": "drawio-generation"}'

    monkeypatch.delenv("LLM_WORKFLOW_ROUTER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    router = WorkflowRouter()
    agent = AgentConfigLoader().load("default")
    agent.routing.llm_workflow_router = False
    request = ChatRequest(messages=[Message(role="user", content="给我做一个系统蓝图")])

    selection = router.select_with_details(agent, request)

    assert calls == []
    assert selection.workflow_name is None


def test_capability_question_ignores_previous_behavior_intent(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    agent.model.base_url = None
    agent.model.api_key = None
    request = ChatRequest(
        messages=[
            Message(role="user", content="老人摔倒"),
            Message(role="assistant", content="文本规则判断：疑似命中行为规则；待上传图片、视频或结构化视觉证据后才能给出视觉识别结果。"),
            Message(role="user", content="你能干啥呢"),
        ],
        runtime_options=RuntimeOptions(thread_id="capability-after-behavior"),
    )

    events = asyncio.run(_collect_events(runtime.iter_events(agent, request)))
    event_types = [event.type for event in events]

    assert "llm.started" in event_types
    assert "spec.started" not in event_types
    assert "agent.message.delta" in event_types


def test_behavior_detector_capability_question_bypasses_detection_workflow(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("behavior-detector")
    agent.model.base_url = None
    agent.model.api_key = None
    request = ChatRequest(
        messages=[
            Message(role="user", content="老人摔倒"),
            Message(role="assistant", content="文本规则判断：疑似命中行为规则；待上传图片、视频或结构化视觉证据后才能给出视觉识别结果。"),
            Message(role="user", content="你能干啥呢"),
        ],
        runtime_options=RuntimeOptions(thread_id="behavior-capability"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.metadata["workflow"] == "agent_loop"
    assert "待上传" not in result.reply


def test_prototype_request_routes_to_drawio(tmp_path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    runtime = AgentRuntime(artifact_store=store)
    agent = AgentConfigLoader().load("artifact-generator")
    request = ChatRequest(
        messages=[Message(role="user", content="可以帮我画一个原型图")],
        runtime_options=RuntimeOptions(thread_id="prototype"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.artifacts[0].name == "prototype.drawio"
    assert result.artifacts[1].name == "prototype.png"
    assert result.artifacts[1].kind == "image"
    assert result.artifacts[1].mime_type == "image/png"
    png_path = store.resolve_virtual_path(result.thread_id, result.artifacts[1].path)
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "prototype.drawio" in result.reply
    assert "prototype.png" in result.reply


def test_drawio_dot_prompt_routes_to_drawio_with_png_preview(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="生成一份 JetLinks IoT 平台架构 Draw.io 图，包含接入层、规则引擎、数据存储、告警和运维监控",
            )
        ],
        runtime_options=RuntimeOptions(thread_id="drawio-dot-architecture"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.metadata["workflow"] == "artifact_workflow"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.spec["diagram_type"] == "layered_architecture"
    assert [artifact.name for artifact in result.artifacts] == ["architecture.drawio", "architecture.png"]
    assert result.artifacts[1].kind == "image"
    assert result.artifacts[1].mime_type == "image/png"


def test_drawio_refinement_followup_routes_to_real_artifacts(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_SPEC_PLANNER", "0")
    store = ArtifactStore(root_dir=tmp_path)
    runtime = AgentRuntime(artifact_store=store)
    agent = AgentConfigLoader().load("default")
    initial_user = Message(
        role="user",
        content="生成一份 JetLinks IoT 平台架构 Draw.io 图，包含接入层、规则引擎、数据存储、告警和运维监控",
    )
    initial_request = ChatRequest(
        messages=[initial_user],
        runtime_options=RuntimeOptions(thread_id="drawio-refinement"),
    )
    initial = asyncio.run(runtime.run(agent, initial_request))

    request = ChatRequest(
        messages=[
            initial_user,
            Message(role="assistant", content=initial.reply),
            Message(role="user", content="这个不是我想要的呢，太丑啦，能不能美化下呢"),
        ],
        runtime_options=RuntimeOptions(thread_id="drawio-refinement"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]

    assert result.status == "completed"
    assert event_types[0] == "run.started"
    assert events[0].data["workflow"] == "artifact_workflow"
    assert "artifact.created" in event_types
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.spec["visual_style"] == "polished"
    assert result.spec["refinement_requested"] is True
    assert [artifact.name for artifact in result.artifacts] == ["architecture_v2.drawio", "architecture_v2.png"]
    assert result.artifacts[1].kind == "image"
    assert "architecture_v2.png" in result.reply


def test_drawio_additive_followup_keeps_artifact_workflow_and_versions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_SPEC_PLANNER", "0")
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    thread_id = "drawio-additive-versioned"
    first_user = Message(
        role="user",
        content="生成一份 JetLinks IoT 平台架构 Draw.io 图，包含接入层、规则引擎、数据存储、告警和运维监控",
    )
    first = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(messages=[first_user], runtime_options=RuntimeOptions(thread_id=thread_id)),
        )
    )
    second_user = Message(role="user", content="不好看呢，再改改呢")
    second = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[first_user, Message(role="assistant", content=first.reply), second_user],
                runtime_options=RuntimeOptions(thread_id=thread_id),
            ),
        )
    )
    third_user = Message(role="user", content="再加上ai复判的逻辑呢")

    result, events = asyncio.run(
        runtime.run_with_events(
            agent,
            ChatRequest(
                messages=[
                    first_user,
                    Message(role="assistant", content=first.reply),
                    second_user,
                    Message(role="assistant", content=second.reply),
                    third_user,
                ],
                runtime_options=RuntimeOptions(thread_id=thread_id),
            ),
        )
    )
    event_types = [event.type for event in events]

    assert first.artifacts[0].name == "architecture.drawio"
    assert second.artifacts[0].name == "architecture_v2.drawio"
    assert result.metadata["workflow"] == "artifact_workflow"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.spec["refinement_requested"] is True
    assert [artifact.name for artifact in result.artifacts] == ["architecture_v3.drawio", "architecture_v3.png"]
    assert "agent.message" in event_types
    assert event_types[-1] == "run.completed"


@pytest.mark.parametrize(
    ("prompt", "expected_skill"),
    [
        ("生成一份 JetLinks IoT 平台架构 Draw.io 图", "drawio-generation"),
        ("生成一份 markdown 架构说明", "markdown-rendering"),
        ("生成一份 ppt 方案", "pptx-generation"),
        ("生成一份 excel 设备模板", "excel-generation"),
        ("生成一份 xmind 思维导图", "xmind-generation"),
    ],
)
def test_generation_skills_use_llm_spec_planner_when_enabled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    expected_skill: str,
) -> None:
    calls: list[str] = []

    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        calls.append(messages[0].content)
        return "{}"

    monkeypatch.setenv("LLM_SPEC_PLANNER", "1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content=prompt)],
        runtime_options=RuntimeOptions(thread_id=f"llm-planner-{expected_skill}"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]

    assert calls
    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["skill_name"] == expected_skill
    assert result.spec["planner"]["mode"] == "llm"
    assert result.spec["planner"]["enabled"] is True
    assert result.artifacts
    assert "spec.planner.started" in event_types
    assert "spec.planner.completed" in event_types


def test_drawio_llm_planner_can_enrich_architecture_spec(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_complete_sync(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        return """
        {
          "visual_style": "polished",
          "swimlanes": ["接入层", "规则计算", "数据服务", "运维闭环"],
          "nodes": ["MQTT 接入", "CoAP 接入", "设备网关", "规则引擎", "数据过滤", "时序存储", "告警中心", "通知服务", "运维监控", "可视化看板"],
          "lane_nodes": {
            "接入层": ["MQTT 接入", "CoAP 接入", "设备网关"],
            "规则计算": ["规则引擎", "数据过滤"],
            "数据服务": ["时序存储"],
            "运维闭环": ["告警中心", "通知服务", "运维监控", "可视化看板"]
          },
          "edges": [
            ["MQTT 接入", "设备网关"],
            ["CoAP 接入", "设备网关"],
            ["设备网关", "规则引擎"],
            ["规则引擎", "数据过滤"],
            ["数据过滤", "时序存储"],
            ["规则引擎", "告警中心"],
            ["告警中心", "通知服务"],
            ["时序存储", "可视化看板"],
            ["运维监控", "可视化看板"]
          ]
        }
        """

    monkeypatch.setenv("LLM_SPEC_PLANNER", "1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    store = ArtifactStore(root_dir=tmp_path)
    runtime = AgentRuntime(artifact_store=store)
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="生成一份 JetLinks IoT 平台架构 Draw.io 图，包含接入层、规则引擎、数据存储、告警和运维监控",
            )
        ],
        runtime_options=RuntimeOptions(thread_id="llm-planner-drawio-enriched"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["planner"]["mode"] == "llm"
    assert result.spec["visual_style"] == "polished"
    assert result.spec["swimlanes"] == ["接入层", "规则计算", "数据服务", "运维闭环"]
    assert "MQTT 接入" in result.spec["nodes"]
    assert result.spec["lane_nodes"]["运维闭环"] == ["告警中心", "通知服务", "运维监控", "可视化看板"]
    assert [artifact.name for artifact in result.artifacts] == ["architecture.drawio", "architecture.png"]

    drawio_path = store.resolve_virtual_path(result.thread_id, result.artifacts[0].path)
    root = ET.fromstring(drawio_path.read_text(encoding="utf-8"))
    cells = root.findall(".//mxCell")
    lane_by_name = {
        str(cell.get("value")): str(cell.get("id"))
        for cell in cells
        if "swimlane" in str(cell.get("style"))
    }
    node_parent = {
        str(cell.get("value")): str(cell.get("parent"))
        for cell in cells
        if cell.get("vertex") == "1" and "swimlane" not in str(cell.get("style"))
    }
    assert node_parent["MQTT 接入"] == lane_by_name["接入层"]
    assert node_parent["时序存储"] == lane_by_name["数据服务"]
    assert node_parent["告警中心"] == lane_by_name["运维闭环"]
    assert "edgeStyle=orthogonalEdgeStyle" in drawio_path.read_text(encoding="utf-8")

    png_path = store.resolve_virtual_path(result.thread_id, result.artifacts[1].path)
    with Image.open(png_path) as image:
        assert image.size[0] >= 900
        assert image.size[1] >= 500
        assert len(image.getcolors(maxcolors=1000000) or []) > 50


def test_default_agent_can_route_generation_skills(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="可以帮我画一个原型图")],
        runtime_options=RuntimeOptions(thread_id="default-prototype"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.metadata["workflow"] == "artifact_workflow"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.artifacts[0].name == "prototype.drawio"
    assert result.artifacts[1].name == "prototype.png"


def test_drawio_followup_keeps_previous_prototype_intent(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("artifact-generator")
    request = ChatRequest(
        messages=[
            Message(role="user", content="可以帮我画一个原型图"),
            Message(role="assistant", content="已生成文件"),
            Message(role="user", content="能不能是drawio格式的呢"),
        ],
        runtime_options=RuntimeOptions(thread_id="prototype-followup"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.spec["diagram_type"] == "prototype_wireframe"
    assert result.artifacts[0].name == "prototype.drawio"
    assert result.artifacts[1].name == "prototype.png"


def test_behavior_detector_text_only_is_honest(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("behavior-detector")
    request = ChatRequest(
        messages=[Message(role="user", content="人员翻越围栏进入禁区")],
        runtime_options=RuntimeOptions(thread_id="t2"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert "待上传" in result.reply
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.metadata["skill_name"] == "behavior-detection"


def test_behavior_detector_with_attachment_can_emit_visual_score(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("behavior-detector")
    request = ChatRequest(
        messages=[Message(role="user", content="人员翻越围栏进入禁区")],
        attachments=[Attachment(name="evidence.jpg", mime_type="image/jpeg", data_base64="ZmFrZQ==")],
        runtime_options=RuntimeOptions(thread_id="t3"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.spec is not None
    assert result.spec["evidence_mode"] == "visual_or_structured"
    assert "critical" in result.reply


def test_generation_skills_create_verified_binary_artifacts(tmp_path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("artifact-generator")

    for thread_id, prompt, expected_suffix in [
        ("ppt", "生成一份 ppt 方案", ".pptx"),
        ("excel", "生成一份 excel 设备模板", ".xlsx"),
        ("xmind", "生成一份 xmind 思维导图", ".xmind"),
    ]:
        request = ChatRequest(
            messages=[Message(role="user", content=prompt)],
            runtime_options=RuntimeOptions(thread_id=thread_id),
        )
        result = asyncio.run(runtime.run(agent, request))
        assert result.status == "completed"
        assert result.verification is not None
        assert result.verification.passed is True
        assert result.artifacts[0].name.endswith(expected_suffix)
