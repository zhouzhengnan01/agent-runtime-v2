import asyncio
from pathlib import Path
from collections.abc import AsyncIterator
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from app.core.agent import AgentRuntime
from app.core.apps import AppTemplateRegistry
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig, AgentConfigLoader
from app.core.config.secrets import SecretCodec
from app.core.agent.turn_verifier import verify_turn_completion
from app.core.llm import LlmChatResponse, OpenAICompatibleClient
from app.core.routing import WorkflowRouter
from app.core.skills import SkillRegistry
from app.core.workflow import WorkflowRegistry
from app.schemas import AgentRunResult, Attachment, ChatEvent, ChatRequest, Message, RuntimeOptions


@pytest.fixture(autouse=True)
def _disable_runtime_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_PROVIDER", "local")
    monkeypatch.setenv("SANDBOX_EXECUTOR_ENABLED", "false")
    monkeypatch.delenv("SANDBOX_SKILLS", raising=False)


async def _collect_events(source: AsyncIterator[ChatEvent]) -> list[ChatEvent]:
    return [event async for event in source]


def test_artifact_generator_creates_verified_markdown(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 markdown 架构说明")],
        runtime_options=RuntimeOptions(thread_id="t1", workflow="artifact_workflow"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.status == "completed"
    assert result.artifacts
    assert result.verification is not None
    assert result.verification.passed is True


def test_agent_with_no_configured_skills_uses_installed_skill_plugins(tmp_path: Path) -> None:
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
        runtime_options=RuntimeOptions(thread_id="plugin-only-skills", workflow="artifact_workflow"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["skill_name"] == "markdown-rendering"
    assert result.artifacts[0].name == "result.md"


def test_artifact_generator_emits_coded_events(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 drawio 架构图")],
        runtime_options=RuntimeOptions(thread_id="events", workflow="artifact_workflow"),
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
    assert "sandbox.failed" not in event_types
    assert "sandbox.fallback" not in event_types
    assert "architecture.drawio" in result.reply


def test_default_agent_stream_emits_text_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_runtime_init_model_config_overrides_env_and_agent_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del system_prompt, messages
        seen.update(
            {
                "model": self.model,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        return "ok"

    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://env.local/v1")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path),
        model_config={
            "model": "runtime-model",
            "base_url": "http://runtime.local/v1",
            "api_key": "runtime-key",
            "temperature": 0.2,
            "max_tokens": 123,
        },
    )
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        model={"model": "json-model", "base_url": "http://json.local/v1", "api_key": "json-key"},
        tools=[],
        skills=[],
    )
    request = ChatRequest(
        messages=[Message(role="user", content="hi")],
        runtime_options=RuntimeOptions(thread_id="runtime-model-config"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.reply == "ok"
    assert seen == {
        "model": "runtime-model",
        "base_url": "http://runtime.local/v1",
        "api_key": "runtime-key",
        "temperature": 0.2,
        "max_tokens": 123,
    }


def test_request_model_options_override_runtime_init_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del system_prompt, messages
        seen.update({"model": self.model, "base_url": self.base_url})
        return "ok"

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path),
        model_config={"model": "runtime-model", "base_url": "http://runtime.local/v1"},
    )
    agent = AgentConfig(name="json-model", display_name="JSON Model", tools=[], skills=[])
    request = ChatRequest(
        messages=[Message(role="user", content="hi")],
        runtime_options=RuntimeOptions(
            thread_id="request-model-config",
            model_name="request-model",
            base_url="http://request.local/v1",
        ),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.reply == "ok"
    assert seen == {"model": "request-model", "base_url": "http://request.local/v1"}


def test_app_template_models_supply_default_chat_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "demo-app.json").write_text(
        """
        {
          "name": "demo-app",
          "title": "Demo App",
          "agent_name": "default",
          "models": [
            {
              "name": "vision-model",
              "model_type": "vision",
              "features": ["vision", "chat"],
              "priority": 0,
              "model": "vision-model",
              "base_url": "http://vision.local/v1"
            },
            {
              "name": "chat-model",
              "features": ["chat"],
              "priority_features": ["chat"],
              "priority": 0,
              "model": "chat-model",
              "base_url": "http://chat.local/v1",
              "api_key": "chat-key",
              "temperature": 0.2,
              "max_tokens": 1024
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del system_prompt, messages
        seen.update(
            {
                "model": self.model,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        return "ok"

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"), app_template_registry=AppTemplateRegistry(tmp_path))
    agent = AgentConfig(
        name="default",
        display_name="Default",
        model={"model": "agent-model", "base_url": "http://agent.local/v1", "api_key": "agent-key"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        runtime_options=RuntimeOptions(thread_id="app-model", app_template_name="demo-app"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.reply == "ok"
    assert seen == {
        "model": "chat-model",
        "base_url": "http://chat.local/v1",
        "api_key": "chat-key",
        "temperature": 0.2,
        "max_tokens": 1024,
    }


def test_app_template_runtime_options_decrypt_api_key_enc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    encrypted = SecretCodec(tmp_path).encrypt(
        "runtime-template-key",
        purpose="app:demo-app:model:chat-model:api_key",
    )
    (apps_dir / "demo-app.json").write_text(
        f"""
        {{
          "name": "demo-app",
          "title": "Demo App",
          "agent_name": "default",
          "runtime_options": {{
            "model_name": "chat-model",
            "base_url": "http://chat.local/v1",
            "api_key_enc": "{encrypted}"
          }}
        }}
        """,
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del system_prompt, messages
        seen.update({"model": self.model, "base_url": self.base_url, "api_key": self.api_key})
        return "ok"

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"), app_template_registry=AppTemplateRegistry(tmp_path))
    agent = AgentConfig(
        name="default",
        display_name="Default",
        model={"model": "agent-model", "base_url": "http://agent.local/v1", "api_key": "agent-key"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="hello")],
        runtime_options=RuntimeOptions(
            thread_id="app-runtime-key",
            app_template_name="demo-app",
            model_name="chat-model",
            base_url="http://chat.local/v1",
        ),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.reply == "ok"
    assert seen == {
        "model": "chat-model",
        "base_url": "http://chat.local/v1",
        "api_key": "runtime-template-key",
    }


def test_agent_loop_persists_and_restores_thread_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_messages: list[list[dict[str, object]]] = []
    replies = iter(["第一轮回复", "第二轮回复"])

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del self, system_prompt
        seen_messages.append([dict(message) for message in messages])
        return next(replies)

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    agent = AgentConfig(
        name="session-agent",
        display_name="Session Agent",
        model={"model": "session-model", "base_url": "http://llm.local/v1", "api_key": "key"},
        tools=[],
        skills=[],
    )

    first = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="第一轮问题")],
                runtime_options=RuntimeOptions(thread_id="session-1"),
            ),
        )
    )
    second = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="第二轮问题")],
                runtime_options=RuntimeOptions(thread_id="session-1"),
            ),
        )
    )

    assert first.reply == "第一轮回复"
    assert second.reply == "第二轮回复"
    assert seen_messages[0] == [{"role": "user", "content": "第一轮问题"}]
    assert seen_messages[1] == [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回复"},
        {"role": "user", "content": "第二轮问题"},
    ]
    history_path = tmp_path / "threads" / "session-1" / "memory" / "conversation.jsonl"
    transcript_path = tmp_path / "threads" / "session-1" / "memory" / "conversation.md"
    assert history_path.is_file()
    assert transcript_path.is_file()
    assert "第二轮回复" in transcript_path.read_text(encoding="utf-8")


def test_agent_loop_does_not_duplicate_client_supplied_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_messages: list[list[dict[str, object]]] = []
    replies = iter(["第一轮回复", "第二轮回复"])

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del self, system_prompt
        seen_messages.append([dict(message) for message in messages])
        return next(replies)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    agent = AgentConfig(
        name="session-agent",
        display_name="Session Agent",
        model={"model": "session-model", "base_url": "http://llm.local/v1", "api_key": "key"},
        tools=[],
        skills=[],
    )

    asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="第一轮问题")],
                runtime_options=RuntimeOptions(thread_id="session-2"),
            ),
        )
    )
    asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[
                    Message(role="user", content="第一轮问题"),
                    Message(role="assistant", content="第一轮回复"),
                    Message(role="user", content="第二轮问题"),
                ],
                runtime_options=RuntimeOptions(thread_id="session-2"),
            ),
        )
    )

    assert seen_messages[1] == [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回复"},
        {"role": "user", "content": "第二轮问题"},
    ]


def test_session_history_filters_local_placeholder_replies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_messages: list[list[dict[str, object]]] = []
    replies = iter(["第一轮回复", "第二轮回复"])
    placeholder = "v2 无状态 Agent 已收到请求。当前 agent JSON、运行时参数或环境变量未配置 LLM base_url/model，因此返回本地占位响应。"

    async def fake_complete(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
    ) -> str:
        del self, system_prompt
        seen_messages.append([dict(message) for message in messages])
        return next(replies)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", fake_complete)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    agent = AgentConfig(
        name="session-agent",
        display_name="Session Agent",
        model={"model": "session-model", "base_url": "http://llm.local/v1", "api_key": "key"},
        tools=[],
        skills=[],
    )

    thread_id = "session-placeholder"
    history_path = tmp_path / "threads" / thread_id / "memory" / "conversation.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "\n".join(
            [
                '{"timestamp":"2026-05-13T00:00:00Z","run_id":"old-1","message":{"role":"user","content":"旧问题"}}',
                '{"timestamp":"2026-05-13T00:00:01Z","run_id":"old-1","message":{"role":"assistant","content":"%s"}}'
                % placeholder,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="新问题")],
                runtime_options=RuntimeOptions(thread_id=thread_id),
            ),
        )
    )

    assert result.reply == "第一轮回复"
    assert seen_messages[0] == [{"role": "user", "content": "新问题"}]
    persisted = history_path.read_text(encoding="utf-8")
    assert placeholder not in persisted
    assert "旧问题" not in persisted
    assert "新问题" in persisted
    assert "第一轮回复" in persisted


def test_message_selected_skills_do_not_block_app_template_model_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del system_prompt, messages
        seen.update(
            {
                "configured": self.configured,
                "model": self.model,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "tool_count": len(tools),
            }
        )
        return LlmChatResponse(content="ok")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path / "threads"))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content=(
                    "围绕 JetLinks IoT 平台生成一组交付物\n\n"
                    "[Workbench selected capabilities]\n"
                    "Selected Skills: drawio-generation, pptx-generation, excel-generation"
                ),
            )
        ],
        runtime_options=RuntimeOptions(
            thread_id="message-selected-skills-app-template",
            app_template_name="artifact-suite",
            model_type="chat",
        ),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.reply == "ok"
    assert seen["configured"] is True
    assert seen["model"] == "Qwen3.6-35B-A3B"
    assert seen["base_url"] == "http://124.132.152.75:62092/v1"
    assert seen["api_key"] == "abc@123"
    assert seen["tool_count"] >= 1


def test_artifact_suite_verifier_passes_when_required_deliverables_exist() -> None:
    verification = verify_turn_completion(
        runtime_options=RuntimeOptions(
            app_template_name="artifact-suite",
            selected_skills=[
                "drawio-generation",
                "pptx-generation",
                "excel-generation",
                "xmind-generation",
                "markdown-rendering",
                "deliverables-export",
            ],
        ),
        artifacts=[
            {"name": "architecture.drawio"},
            {"name": "architecture.png"},
            {"name": "deck.pptx"},
            {"name": "device-template.xlsx"},
            {"name": "mindmap.xmind"},
            {"name": "result.md"},
            {"name": "commands.txt"},
            {"name": "steps.docx"},
        ],
        required_inputs=[],
        latest_tool_results=[],
        tool_call_count=6,
        primary_skill_context=None,
    )

    assert verification.verdict == "passed"


def test_artifact_suite_verifier_stays_pending_when_deliverables_missing() -> None:
    verification = verify_turn_completion(
        runtime_options=RuntimeOptions(
            app_template_name="artifact-suite",
            selected_skills=[
                "drawio-generation",
                "pptx-generation",
                "excel-generation",
                "xmind-generation",
                "markdown-rendering",
                "deliverables-export",
            ],
        ),
        artifacts=[
            {"name": "architecture.drawio"},
            {"name": "architecture.png"},
            {"name": "deck.pptx"},
        ],
        required_inputs=[],
        latest_tool_results=[],
        tool_call_count=3,
        primary_skill_context=None,
    )

    assert verification.verdict == "pending"
    assert "device-template.xlsx" in verification.missing_evidence


def test_workflow_persists_and_restores_thread_conversation(tmp_path: Path) -> None:
    seen_messages: list[list[Message]] = []
    replies = iter(["工作流第一轮回复", "工作流第二轮回复"])

    class CapturingWorkflow:
        def run_with_events(
            self,
            agent_config: AgentConfig,
            messages: list[Message],
            attachments: list[Attachment],
            thread_id: str | None,
            on_event=None,
            workflow_name: str | None = None,
            runtime_options: RuntimeOptions | None = None,
        ) -> tuple[AgentRunResult, list[ChatEvent]]:
            del attachments, on_event, runtime_options
            seen_messages.append(list(messages))
            reply = next(replies)
            event = ChatEvent(
                type="run.started",
                data={
                    "run_id": f"run-{len(seen_messages)}",
                    "agent": agent_config.name,
                    "thread_id": thread_id or "",
                    "workflow": workflow_name or "capturing_workflow",
                },
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=thread_id or "",
                reply=reply,
                metadata={"workflow": workflow_name or "capturing_workflow"},
            )
            return result, [event]

    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path / "threads"),
        workflow_registry=WorkflowRegistry({"capturing_workflow": CapturingWorkflow()}),
    )
    agent = AgentConfig(
        name="workflow-session-agent",
        display_name="Workflow Session Agent",
        tools=[],
        skills=[],
    )

    first = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="第一轮生成")],
                runtime_options=RuntimeOptions(thread_id="workflow-session", workflow="capturing_workflow"),
            ),
        )
    )
    second = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="第二轮修改")],
                runtime_options=RuntimeOptions(thread_id="workflow-session", workflow="capturing_workflow"),
            ),
        )
    )

    assert first.reply == "工作流第一轮回复"
    assert second.reply == "工作流第二轮回复"
    assert [(message.role, message.content) for message in seen_messages[1]] == [
        ("user", "第一轮生成"),
        ("assistant", "工作流第一轮回复"),
        ("user", "第二轮修改"),
    ]
    transcript_path = tmp_path / "threads" / "workflow-session" / "memory" / "conversation.md"
    assert "工作流第二轮回复" in transcript_path.read_text(encoding="utf-8")


def test_runtime_init_skills_replace_agent_json_default_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(str(tool["function"]["name"]) for tool in tools)
        return LlmChatResponse(content="ok")

    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path),
        model_config={"model": "runtime-model", "base_url": "http://runtime.local/v1"},
        skills=["markdown-rendering"],
    )
    agent = AgentConfig(
        name="json-model",
        display_name="JSON Model",
        tools=[],
        skills=["drawio-generation"],
    )
    request = ChatRequest(
        messages=[Message(role="user", content="hi")],
        runtime_options=RuntimeOptions(thread_id="runtime-skills"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.reply == "ok"
    assert "markdown-rendering" in seen_tools
    assert "drawio-generation" not in seen_tools


def test_agent_workflow_config_does_not_auto_run_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
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


def test_agent_runtime_enriches_missing_image_input_for_all_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        return LlmChatResponse(content="我现在仍然没有收到可用的图片附件，请重新上传图片。")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="missing-image-agent",
        display_name="Missing Image Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        tools=["jetlinks_runtime_status"],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="请把我上传的图片自动标注成 COCO JSON")],
        runtime_options=RuntimeOptions(thread_id="missing-image"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.metadata["requires_input"] is True
    assert result.metadata["required_inputs"][0]["type"] == "image"
    completed = next(event for event in reversed(events) if event.type == "run.completed")
    event_result = completed.data["result"]
    assert event_result["metadata"]["requires_input"] is True
    assert event_result["metadata"]["required_inputs"][0]["accept"] == "image/*"


def test_agent_runtime_preflights_data_auto_annotation_without_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        raise AssertionError("LLM should not be called when data auto annotation has no image attachment.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fail_if_called)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="data-auto-preflight-agent",
        display_name="Data Auto Preflight Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="请把我上传的图片自动标注成 COCO JSON，类别 labels: person car helmet")],
        runtime_options=RuntimeOptions(thread_id="data-auto-preflight", selected_skills=["data-auto-annotation"]),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.reply == "请先上传图片后继续。"
    assert result.metadata["requires_input"] is True
    assert result.metadata["required_inputs"][0]["type"] == "image"
    assert [event.type for event in events] == ["run.started", "agent.message", "run.completed"]
    completed = events[-1].data["result"]
    assert completed["metadata"]["required_inputs"][0]["accept"] == "image/*"


def test_agent_runtime_yolo_mode_still_requires_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        raise AssertionError("LLM should not be called when required image input is missing.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fail_if_called)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="data-auto-yolo-preflight-agent",
        display_name="Data Auto YOLO Preflight Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="请把图片自动标注成 COCO JSON")],
        runtime_options=RuntimeOptions(
            thread_id="data-auto-yolo-preflight",
            mode="yolo",
            selected_skills=["data-auto-annotation"],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.metadata["mode"] == "yolo"
    assert result.metadata["requires_input"] is True
    assert result.metadata["required_inputs"][0]["type"] == "image"
    assert [event.type for event in events] == ["run.started", "agent.message", "run.completed"]


def test_agent_runtime_preflights_data_auto_annotation_intent_without_selected_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        raise AssertionError("LLM should not be called for an image annotation request without an image.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fail_if_called)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="data-auto-intent-preflight-agent",
        display_name="Data Auto Intent Preflight Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="请把图片自动标注成 COCO JSON，类别 labels: person car helmet")],
        runtime_options=RuntimeOptions(thread_id="data-auto-intent-preflight"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.reply == "请先上传图片后继续。"
    assert result.metadata["requires_input"] is True
    assert result.metadata["required_inputs"][0]["type"] == "image"
    assert [event.type for event in events] == ["run.started", "agent.message", "run.completed"]


def test_agent_runtime_preflights_algorithm_training_without_base_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        raise AssertionError("LLM should not be called when algorithm training has no base data.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fail_if_called)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="algorithm-training-preflight-agent",
        display_name="Algorithm Training Preflight Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "artifact_workflow"},
    )
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="帮我把棕榈果检测算法工程师全流程跑起来：先盘点数据和 baseline，再安排 CPU/GPU 训练。",
            )
        ],
        runtime_options=RuntimeOptions(
            thread_id="algorithm-training-preflight",
            workflow="artifact_workflow",
            selected_skills=[
                "algorithm-engineer",
                "dataset-curator",
                "data-auto-annotation",
                "cpu-training-runner",
                "experiment-ledger",
            ],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.reply == "请先上传数据集后继续。"
    assert result.metadata["requires_input"] is True
    assert result.metadata["required_inputs"][0]["type"] == "dataset"
    assert "data.yaml" in result.metadata["required_inputs"][0]["reason"]
    assert [event.type for event in events] == ["run.started", "agent.message", "run.completed"]


def test_agent_runtime_preflights_algorithm_training_followup_choice_without_base_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail_if_called(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        raise AssertionError("LLM should not answer numbered follow-ups before algorithm data is provided.")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fail_if_called)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="algorithm-training-followup-preflight-agent",
        display_name="Algorithm Training Follow-up Preflight Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[
            Message(role="user", content="帮我把棕榈果检测算法工程师全流程跑起来。"),
            Message(role="assistant", content="请选择下一步动作。"),
            Message(role="user", content="1"),
        ],
        runtime_options=RuntimeOptions(
            thread_id="algorithm-training-followup-preflight",
            workflow="agent_loop",
            selected_skills=[
                "algorithm-engineer",
                "algorithm-research-scout",
                "dataset-curator",
                "data-auto-annotation",
                "model-candidate-selector",
                "cpu-training-runner",
                "detector-evaluator",
                "experiment-ledger",
            ],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert result.reply == "请先上传数据集后继续。"
    assert result.metadata["requires_input"] is True
    assert result.metadata["required_inputs"][0]["type"] == "dataset"
    assert [event.type for event in events] == ["run.started", "agent.message", "run.completed"]


def test_agent_runtime_research_only_algorithm_flow_can_answer_without_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        return LlmChatResponse(content=f"已进入调研工具链，tools={len(tools)}。")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="algorithm-research-only-agent",
        display_name="Algorithm Research Only Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="只做棕榈果检测候选算法调研。")],
        runtime_options=RuntimeOptions(
            thread_id="algorithm-research-only",
            workflow="agent_loop",
            selected_skills=["algorithm-research-scout", "model-candidate-selector"],
        ),
    )

    result, _events = asyncio.run(runtime.run_with_events(agent, request))

    assert "requires_input" not in result.metadata
    assert result.reply.startswith("已进入调研工具链")


def test_agent_runtime_algorithm_training_flow_does_not_require_image_just_because_annotation_skill_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="已进入算法训练流程。")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="algorithm-training-data-yaml-agent",
        display_name="Algorithm Training Data YAML Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="用 /data/palm/dataset/data.yaml 跑棕榈果检测训练，默认 CPU 优先。",
            )
        ],
        runtime_options=RuntimeOptions(
            thread_id="algorithm-training-data-yaml",
            selected_skills=["algorithm-engineer", "data-auto-annotation", "cpu-training-runner"],
        ),
    )

    result, _events = asyncio.run(runtime.run_with_events(agent, request))

    assert "requires_input" not in result.metadata
    assert result.reply == "已进入算法训练流程。"
    assert "data-auto-annotation" in seen_tools


def test_agent_runtime_yolo_training_zip_text_does_not_trigger_image_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="已进入 YOLO 训练 dry-run。")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="yolo-zip-preflight-agent",
        display_name="YOLO Zip Preflight Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content=(
                    "处理上传的 generated_images.zip，labels=fallen_person，"
                    "调用 gpu-training-orchestrator 做 dry_run。"
                ),
            )
        ],
        runtime_options=RuntimeOptions(
            thread_id="yolo-zip-preflight",
            mode="yolo",
            selected_skills=["gpu-training-orchestrator"],
        ),
    )

    result, _events = asyncio.run(runtime.run_with_events(agent, request))

    assert "requires_input" not in result.metadata
    assert result.reply == "已进入 YOLO 训练 dry-run。"
    assert "gpu-training-orchestrator" in seen_tools


def test_agent_runtime_uses_existing_thread_upload_for_selected_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_messages: list[list[dict[str, object]]] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, tools
        seen_messages.append(messages)
        return LlmChatResponse(content="已使用当前会话里的图片继续处理。")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("existing-upload-skill")
    image_path = paths.uploads / "scene.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    runtime = AgentRuntime(artifact_store=store)
    agent = AgentConfig(
        name="existing-upload-skill-agent",
        display_name="Existing Upload Skill Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="继续用刚才上传的图片做 COCO 标注")],
        runtime_options=RuntimeOptions(
            thread_id="existing-upload-skill",
            selected_skills=["data-auto-annotation"],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert "requires_input" not in result.metadata
    assert result.reply == "已使用当前会话里的图片继续处理。"
    assert "run.completed" in [event.type for event in events]
    assert seen_messages
    assert "/mnt/user-data/uploads/scene.png" in str(seen_messages[-1])


def test_agent_runtime_recovers_selected_skill_from_workbench_message_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_tools: list[str] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages
        seen_tools.extend(tool["function"]["name"] for tool in tools)
        return LlmChatResponse(content="已调用自动标注工具。")

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("message-selected-skill")
    (paths.uploads / "scene.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    runtime = AgentRuntime(artifact_store=store)
    agent = AgentConfig(
        name="message-selected-skill-agent",
        display_name="Message Selected Skill Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        skills=[],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content=(
                    "继续处理刚上传的文件。\n\n"
                    "[Workbench selected capabilities]\n"
                    "Selected Skills: data-auto-annotation"
                ),
            )
        ],
        runtime_options=RuntimeOptions(thread_id="message-selected-skill"),
    )

    result, _events = asyncio.run(runtime.run_with_events(agent, request))

    assert "requires_input" not in result.metadata
    assert "data-auto-annotation" in seen_tools


def test_agent_runtime_merges_runtime_selected_skill_into_workflow_allowlist(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="selected-skill-agent",
        display_name="Selected Skill Agent",
        skills=["behavior-detection"],
    )
    request = ChatRequest(
        messages=[Message(role="user", content="请把图片自动标注成 COCO JSON")],
        attachments=[Attachment(name="scene.png", path="/mnt/user-data/uploads/scene.png", mime_type="image/png")],
        runtime_options=RuntimeOptions(
            thread_id="selected-skill-allowlist",
            workflow="artifact_workflow",
            selected_skills=["data-auto-annotation"],
        ),
    )

    effective_agent, effective_request = runtime._prepare_execution(agent, request)

    assert effective_request.runtime_options.selected_skills == ["data-auto-annotation"]
    assert effective_agent.skills == ["behavior-detection", "data-auto-annotation"]


def test_agent_runtime_does_not_infer_input_required_from_successful_upload_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> LlmChatResponse:
        del self, system_prompt, messages, tools
        return LlmChatResponse(
            content=(
                "已读取上传文件路径 /mnt/user-data/uploads/input.txt，"
                "并创建 outputs/e2e-summary.md。"
            )
        )

    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="upload-reference-agent",
        display_name="Upload Reference Agent",
        model={"base_url": "http://llm.local/v1", "api_key": "key", "model": "tool-model"},
        tools=["present_files"],
        workflows={"default": "agent_loop"},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="继续同一个会话，确认上传文件和生成文件都能看到。")],
        runtime_options=RuntimeOptions(thread_id="upload-reference"),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))

    assert "requires_input" not in result.metadata
    completed = events[-1].data["result"]
    assert "requires_input" not in completed["metadata"]


def test_unregistered_explicit_workflow_fails_fast(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        artifact_store=ArtifactStore(root_dir=tmp_path),
        workflow_registry=WorkflowRegistry(),
    )
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 markdown 架构说明")],
        runtime_options=RuntimeOptions(thread_id="workflow-plugin-disabled", workflow="artifact_workflow"),
    )

    with pytest.raises(ValueError, match="Workflow is not registered: artifact_workflow"):
        asyncio.run(runtime.run_with_events(agent, request))


def test_workflow_router_uses_skill_manifest_routing_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_workflow_router_llm_can_select_from_skill_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(OpenAICompatibleClient, "complete_sync", fake_complete_sync)
    monkeypatch.setattr(OpenAICompatibleClient, "configured", property(lambda self: True))
    router = WorkflowRouter(available_workflows={"artifact_workflow", "evidence_first_detection"})
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(messages=[Message(role="user", content="给我做一个系统蓝图")])

    selection = router.select_with_details(agent, request)

    assert selection.workflow_name == "artifact_workflow"
    assert selection.skill_name == "drawio-generation"
    assert selection.mode == "llm"
    assert seen_clients == [(agent.model.model or agent.model.default_model, agent.model.base_url or "")]


def test_workflow_router_respects_agent_json_routing_switch(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_capability_question_ignores_previous_behavior_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_behavior_detector_capability_question_bypasses_detection_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        runtime_options=RuntimeOptions(thread_id="behavior-capability"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.metadata["workflow"] == "agent_loop"
    assert "待上传" not in result.reply


def test_prototype_request_routes_to_drawio(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    runtime = AgentRuntime(artifact_store=store)
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="可以帮我画一个原型图")],
        runtime_options=RuntimeOptions(thread_id="prototype", workflow="artifact_workflow"),
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


def test_drawio_dot_prompt_routes_to_drawio_with_png_preview(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[
            Message(
                role="user",
                content="生成一份 JetLinks IoT 平台架构 Draw.io 图，包含接入层、规则引擎、数据存储、告警和运维监控",
            )
        ],
        runtime_options=RuntimeOptions(thread_id="drawio-dot-architecture", workflow="artifact_workflow"),
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


def test_drawio_refinement_followup_routes_to_real_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        runtime_options=RuntimeOptions(thread_id="drawio-refinement", workflow="artifact_workflow"),
    )
    initial = asyncio.run(runtime.run(agent, initial_request))

    request = ChatRequest(
        messages=[
            initial_user,
            Message(role="assistant", content=initial.reply),
            Message(role="user", content="这个不是我想要的呢，太丑啦，能不能美化下呢"),
        ],
        runtime_options=RuntimeOptions(thread_id="drawio-refinement", workflow="artifact_workflow"),
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
    tmp_path: Path,
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
            ChatRequest(
                messages=[first_user],
                runtime_options=RuntimeOptions(thread_id=thread_id, workflow="artifact_workflow"),
            ),
        )
    )
    second_user = Message(role="user", content="不好看呢，再改改呢")
    second = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[first_user, Message(role="assistant", content=first.reply), second_user],
                runtime_options=RuntimeOptions(thread_id=thread_id, workflow="artifact_workflow"),
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
                runtime_options=RuntimeOptions(thread_id=thread_id, workflow="artifact_workflow"),
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
    tmp_path: Path,
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
    monkeypatch.setattr(OpenAICompatibleClient, "configured", property(lambda self: True))
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content=prompt)],
        runtime_options=RuntimeOptions(thread_id=f"llm-planner-{expected_skill}", workflow="artifact_workflow"),
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


def test_drawio_llm_planner_can_enrich_architecture_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(OpenAICompatibleClient, "configured", property(lambda self: True))
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
        runtime_options=RuntimeOptions(thread_id="llm-planner-drawio-enriched", workflow="artifact_workflow"),
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


def test_default_agent_does_not_auto_route_generation_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    agent.model.base_url = None
    agent.model.api_key = None
    request = ChatRequest(
        messages=[Message(role="user", content="可以帮我画一个原型图")],
        runtime_options=RuntimeOptions(thread_id="default-prototype"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.metadata["workflow"] == "agent_loop"
    assert result.spec is None
    assert result.artifacts == []


def test_default_agent_runs_explicit_generation_workflow(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="可以帮我画一个原型图")],
        runtime_options=RuntimeOptions(thread_id="default-prototype-explicit", workflow="artifact_workflow"),
    )

    result = asyncio.run(runtime.run(agent, request))

    assert result.status == "completed"
    assert result.metadata["workflow"] == "artifact_workflow"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.artifacts[0].name == "prototype.drawio"
    assert result.artifacts[1].name == "prototype.png"


def test_selected_generation_skill_does_not_auto_route_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    agent.model.base_url = None
    agent.model.api_key = None
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份项目汇报材料")],
        runtime_options=RuntimeOptions(
            thread_id="selected-skill-workflow",
            selected_skills=["pptx-generation"],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]

    assert result.status == "completed"
    assert result.metadata["workflow"] == "agent_loop"
    assert result.spec is None
    assert result.artifacts == []
    assert "direct_skill.started" not in event_types
    assert "verifier.completed" not in event_types


def test_selected_generation_skill_uses_explicit_artifact_workflow_without_agent_mapping(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="minimal-skill-agent",
        display_name="Minimal Skill Agent",
        skills=["markdown-rendering"],
        workflows={},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="生成一份 markdown 项目说明")],
        runtime_options=RuntimeOptions(
            thread_id="selected-skill-default-workflow",
            workflow="artifact_workflow",
            selected_skills=["markdown-rendering"],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]

    assert result.status == "completed"
    assert result.metadata["workflow"] == "artifact_workflow"
    assert result.spec is not None
    assert result.spec["skill_name"] == "markdown-rendering"
    assert result.verification is not None
    assert result.verification.passed is True
    assert "direct_skill.started" not in event_types
    assert "verifier.completed" in event_types


def test_selected_generic_execution_skill_uses_explicit_artifact_workflow(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="generic-execution-agent",
        display_name="Generic Execution Agent",
        skills=["public-skill-demo"],
        workflows={},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="运行 public demo skill")],
        runtime_options=RuntimeOptions(
            thread_id="selected-generic-execution-skill",
            workflow="artifact_workflow",
            selected_skills=["public-skill-demo"],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]

    assert result.status == "completed"
    assert result.metadata["workflow"] == "artifact_workflow"
    assert result.spec is not None
    assert result.spec["skill_name"] == "public-skill-demo"
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.artifacts
    assert result.artifacts[0].name.endswith(".md")
    assert "direct_skill.started" not in event_types
    assert "verifier.completed" in event_types


def test_selected_behavior_skill_uses_explicit_evidence_workflow_without_agent_mapping(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfig(
        name="minimal-behavior-agent",
        display_name="Minimal Behavior Agent",
        skills=["behavior-detection"],
        workflows={},
    )
    request = ChatRequest(
        messages=[Message(role="user", content="人员翻越围栏进入禁区")],
        runtime_options=RuntimeOptions(
            thread_id="selected-behavior-default-workflow",
            workflow="evidence_first_detection",
            selected_skills=["behavior-detection"],
        ),
    )

    result, events = asyncio.run(runtime.run_with_events(agent, request))
    event_types = [event.type for event in events]

    assert result.status == "completed"
    assert result.metadata["workflow"] == "evidence_first_detection"
    assert result.metadata["skill_name"] == "behavior-detection"
    assert result.verification is not None
    assert result.verification.passed is True
    assert "direct_skill.started" not in event_types
    assert "verifier.completed" in event_types


def test_drawio_followup_keeps_previous_prototype_intent(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[
            Message(role="user", content="可以帮我画一个原型图"),
            Message(role="assistant", content="已生成文件"),
            Message(role="user", content="能不能是drawio格式的呢"),
        ],
        runtime_options=RuntimeOptions(thread_id="prototype-followup", workflow="artifact_workflow"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.status == "completed"
    assert result.spec is not None
    assert result.spec["skill_name"] == "drawio-generation"
    assert result.spec["diagram_type"] == "prototype_wireframe"
    assert result.artifacts[0].name == "prototype.drawio"
    assert result.artifacts[1].name == "prototype.png"


def test_behavior_detector_text_only_is_honest(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="人员翻越围栏进入禁区")],
        runtime_options=RuntimeOptions(thread_id="t2", workflow="evidence_first_detection"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert "待上传" in result.reply
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.metadata["skill_name"] == "behavior-detection"
    assert [artifact.name for artifact in result.artifacts] == ["behavior-detection.md", "behavior-detection.json"]


def test_behavior_detector_with_attachment_can_emit_visual_score(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")
    request = ChatRequest(
        messages=[Message(role="user", content="人员翻越围栏进入禁区")],
        attachments=[Attachment(name="evidence.jpg", mime_type="image/jpeg", data_base64="ZmFrZQ==")],
        runtime_options=RuntimeOptions(thread_id="t3", workflow="evidence_first_detection"),
    )
    result = asyncio.run(runtime.run(agent, request))
    assert result.verification is not None
    assert result.verification.passed is True
    assert result.spec is not None
    assert result.spec["evidence_mode"] == "visual_or_structured"
    assert "critical" in result.reply
    assert [artifact.name for artifact in result.artifacts] == ["behavior-detection.md", "behavior-detection.json"]


def test_generation_skills_create_verified_binary_artifacts(tmp_path: Path) -> None:
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))
    agent = AgentConfigLoader().load("default")

    for thread_id, prompt, expected_suffix in [
        ("ppt", "生成一份 ppt 方案", ".pptx"),
        ("excel", "生成一份 excel 设备模板", ".xlsx"),
        ("xmind", "生成一份 xmind 思维导图", ".xmind"),
    ]:
        request = ChatRequest(
            messages=[Message(role="user", content=prompt)],
            runtime_options=RuntimeOptions(thread_id=thread_id, workflow="artifact_workflow"),
        )
        result = asyncio.run(runtime.run(agent, request))
        assert result.status == "completed"
        assert result.verification is not None
        assert result.verification.passed is True
        assert result.artifacts[0].name.endswith(expected_suffix)
