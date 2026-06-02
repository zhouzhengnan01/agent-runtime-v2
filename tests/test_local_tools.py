from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from typing import Any

import httpx
from pytest import MonkeyPatch

from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.config.agent_config import ModelConfig
from app.core.llm.openai_compatible import LlmChatResponse, LlmToolCall, OpenAICompatibleClient
from app.core.tools import ToolInvocationService
from app.schemas import ChatRequest, Message, RuntimeOptions


def test_local_tools_write_read_search_and_todo(tmp_path: Path) -> None:
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))
    thread_args = {"_thread_id": "local-tools"}

    write_result = service.call_tool(
        "local_write_file",
        {**thread_args, "path": "notes/plan.md", "content": "JetLinks agent runtime\n工具闭环\n"},
    )
    assert write_result.is_error is False
    assert write_result.structured_content["path"] == "notes/plan.md"

    read_result = service.call_tool("local_read_file", {**thread_args, "path": "notes/plan.md"})
    assert read_result.is_error is False
    assert "工具闭环" in read_result.content[0]["text"]

    virtual_read_result = service.call_tool(
        "local_read_file",
        {**thread_args, "path": "/mnt/user-data/workspace/notes/plan.md"},
    )
    assert virtual_read_result.is_error is False
    assert "工具闭环" in virtual_read_result.content[0]["text"]

    search_result = service.call_tool("local_search_text", {**thread_args, "pattern": "agent", "path": "."})
    assert search_result.is_error is False
    assert search_result.structured_content["matches"][0]["path"] == "notes/plan.md"

    add_result = service.call_tool("local_todo", {**thread_args, "action": "add", "text": "补齐本地工具"})
    assert add_result.is_error is False
    assert add_result.structured_content["todos"][0]["text"] == "补齐本地工具"

    complete_result = service.call_tool("local_todo", {**thread_args, "action": "complete", "id": 1})
    assert complete_result.is_error is False
    assert complete_result.structured_content["todos"][0]["done"] is True


def test_present_files_lists_thread_outputs_and_optional_workspace(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("present-files")
    store.write_text_artifact(paths, "result.md", "# Result")
    service = ToolInvocationService(artifact_store=store)

    outputs_only = service.call_tool("present_files", {"_thread_id": "present-files"})
    with_workspace = service.call_tool(
        "present_files",
        {"_thread_id": "present-files", "include_workspace": True},
    )

    assert outputs_only.is_error is False
    assert outputs_only.structured_content["files"][0]["scope"] == "outputs"
    assert outputs_only.structured_content["files"][0]["path"] == "result.md"
    assert with_workspace.is_error is False


def test_local_file_to_base64_encodes_svg_from_outputs(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("base64-tools")
    svg = "<svg viewBox=\"0 0 10 10\" xmlns=\"http://www.w3.org/2000/svg\"></svg>"
    (paths.outputs / "generated-bigscreen").mkdir(parents=True)
    (paths.outputs / "generated-bigscreen" / "background.svg").write_text(svg, encoding="utf-8")
    service = ToolInvocationService(artifact_store=store)

    result = service.call_tool(
        "local_file_to_base64",
        {
            "_thread_id": "base64-tools",
            "path": "/mnt/user-data/outputs/generated-bigscreen/background.svg",
        },
    )

    assert result.is_error is False
    assert result.structured_content["path"] == "/mnt/user-data/outputs/generated-bigscreen/background.svg"
    assert result.structured_content["mime_type"] == "image/svg+xml;charset=UTF-8"
    assert result.structured_content["base64"].startswith("PHN2ZyB2aWV3Qm94")
    assert result.structured_content["data_uri"].startswith("data:image/svg+xml;charset=UTF-8;base64,")


def test_local_file_to_base64_rejects_oversized_files(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("base64-too-large")
    (paths.workspace / "large.png").write_bytes(b"x" * 8)
    service = ToolInvocationService(artifact_store=store)

    result = service.call_tool(
        "local_file_to_base64",
        {"_thread_id": "base64-too-large", "path": "large.png", "max_bytes": 4},
    )

    assert result.is_error is True
    assert "too large" in result.content[0]["text"]


def test_local_download_url_downloads_image_to_uploads(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    original_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.test/assets/scene.jpeg?token=abc"
        return httpx.Response(
            200,
            headers={"content-type": "image/jpeg", "content-length": "8"},
            content=b"jpegdata",
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    store = ArtifactStore(root_dir=tmp_path)
    service = ToolInvocationService(artifact_store=store)

    result = service.call_tool(
        "local_download_url",
        {
            "_thread_id": "download-image",
            "url": "https://example.test/assets/scene.jpeg?token=abc",
        },
    )

    assert result.is_error is False
    assert result.structured_content["path"] == "/mnt/user-data/uploads/scene.jpeg"
    assert result.structured_content["mime_type"] == "image/jpeg"
    assert result.structured_content["kind"] == "image"
    assert (tmp_path / "download-image" / "uploads" / "scene.jpeg").read_bytes() == b"jpegdata"


def test_local_download_url_accepts_video_and_sanitizes_filename(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    original_client = httpx.Client

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "video/mp4"},
            content=b"mp4data",
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    store = ArtifactStore(root_dir=tmp_path)
    service = ToolInvocationService(artifact_store=store)

    result = service.call_tool(
        "local_download_url",
        {
            "_thread_id": "download-video",
            "url": "https://example.test/video/download",
            "filename": "../camera 01.mp4",
        },
    )

    assert result.is_error is False
    assert result.structured_content["path"] == "/mnt/user-data/uploads/camera_01.mp4"
    assert result.structured_content["kind"] == "video"
    assert (tmp_path / "download-video" / "uploads" / "camera_01.mp4").read_bytes() == b"mp4data"


def test_local_download_url_rejects_non_media_response(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    original_client = httpx.Client

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html></html>")

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = service.call_tool(
        "local_download_url",
        {"_thread_id": "download-html", "url": "https://example.test/index.html"},
    )

    assert result.is_error is True
    assert "not allowed" in result.content[0]["text"]
    assert not (tmp_path / "download-html" / "uploads" / "index.html").exists()


def test_local_download_url_rejects_oversized_response(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    original_client = httpx.Client

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png", "content-length": "9"},
            content=b"123456789",
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = service.call_tool(
        "local_download_url",
        {
            "_thread_id": "download-too-large",
            "url": "https://example.test/image.png",
            "max_bytes": 8,
        },
    )

    assert result.is_error is True
    assert "too large" in result.content[0]["text"]
    assert not (tmp_path / "download-too-large" / "uploads" / "image.png").exists()


def test_extract_archive_and_validate_yolo_training_inputs(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("yolo-tools")
    archive = paths.uploads / "generated_images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("images/scene1.jpg", b"fake-jpg-1")
        handle.writestr("images/scene2.jpg", b"fake-jpg-2")
    ref_image = paths.uploads / "reference.jpg"
    ref_image.write_bytes(b"fake-ref")
    service = ToolInvocationService(artifact_store=store)

    extracted = service.call_tool(
        "extract_archive",
        {
            "_thread_id": "yolo-tools",
            "path": "/mnt/user-data/uploads/generated_images.zip",
            "output_dir": "/mnt/user-data/workspace/dataset_root",
        },
    )
    dataset_root = extracted.structured_content["dataset_root"]
    validated = service.call_tool(
        "validate_yolo_training_inputs",
        {
            "_thread_id": "yolo-tools",
            "dataset_root": dataset_root,
            "ref_image": "/mnt/user-data/uploads/reference.jpg",
            "labels": ["person"],
            "training": {
                "model": "yolo11n.pt",
                "epochs": 1,
                "imgsz": 320,
                "batch": 1,
                "device": "cpu",
                "amp": False,
            },
            "runtime": {"conda_env_name": "cv_train"},
            "split": {"train": 0.7, "val": 0.2, "test": 0.1},
        },
    )

    assert extracted.is_error is False
    assert extracted.structured_content["image_count"] == 2
    assert dataset_root == "dataset_root/images"
    assert (paths.workspace / "dataset_root" / "images" / "scene1.jpg").is_file()
    assert validated.is_error is False
    assert validated.structured_content["ready"] is True
    assert validated.structured_content["normalized"]["image_count"] == 2
    assert validated.structured_content["normalized"]["training"]["amp"] is False


def test_local_tools_block_path_traversal(tmp_path: Path) -> None:
    service = ToolInvocationService(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = service.call_tool(
        "local_write_file",
        {"_thread_id": "local-tools", "path": "../escape.txt", "content": "blocked"},
    )

    assert result.is_error is True
    assert "path traversal" in result.content[0]["text"].lower()


def test_agent_loop_can_use_local_workspace_tools(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls: list[list[str]] = []

    async def fake_complete_with_tools(
        self: OpenAICompatibleClient,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
    ) -> LlmChatResponse:
        calls.append([tool["function"]["name"] for tool in tools])
        if len(calls) == 1:
            return LlmChatResponse(
                tool_calls=[
                    LlmToolCall(
                        id="call_write",
                        name="local_write_file",
                        arguments='{"path":"result.txt","content":"hello from local tool"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        tool_message = next(message for message in messages if message.get("role") == "tool")
        payload = json.loads(tool_message["content"])
        assert payload["isError"] is False
        assert payload["structuredContent"]["path"] == "result.txt"
        return LlmChatResponse(content="已写入。", finish_reason="stop")

    monkeypatch.delenv("LOCAL_SHELL_TOOL_ENABLED", raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, "complete_with_tools", fake_complete_with_tools)
    agent = AgentConfig(
        name="local-tool-agent",
        display_name="Local Tool Agent",
        model=ModelConfig(base_url="http://llm.local/v1", api_key="key", model="tool-model"),
        tools=["local_write_file", "local_shell_command"],
        workflows={"default": "agent_loop"},
    )
    runtime = AgentRuntime(artifact_store=ArtifactStore(root_dir=tmp_path))

    result = asyncio.run(
        runtime.run(
            agent,
            ChatRequest(
                messages=[Message(role="user", content="写入文件")],
                runtime_options=RuntimeOptions(thread_id="agent-local-tools"),
            ),
        )
    )

    assert result.status == "completed"
    assert result.reply == "已写入。"
    assert calls[0] == ["local_write_file"]
    assert (tmp_path / "agent-local-tools" / "workspace" / "result.txt").read_text() == (
        "hello from local tool"
    )
