from __future__ import annotations

import io
import importlib.util
import json
import threading
import subprocess
import sys
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import pytest
from fastapi.testclient import TestClient

from app.api import skills as skills_api
from app.core.artifacts import ArtifactStore
from app.core.skills import SkillRegistry, SkillRunner
from app.core.skills.plugins import SkillPluginManager
from app.main import create_app


def test_skill_plugin_upload_registers_and_executes_uploaded_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/skills/plugins",
        files={"file": ("summary-plugin.zip", _plugin_zip(), "application/zip")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["plugin"]["id"] == "summary-plugin"
    assert data["plugin"]["skills"] == ["uploaded-summary"]

    plugins = client.get("/api/skills/plugins")
    assert plugins.status_code == 200
    assert "summary-plugin" in {plugin["id"] for plugin in plugins.json()["plugins"]}

    listed = client.get("/api/skills")
    assert listed.status_code == 200
    skills = {skill["name"]: skill for skill in listed.json()["skills"]}
    assert skills["uploaded-summary"]["executable"] is True
    assert skills["uploaded-summary"]["source"]["type"] == "plugin"
    assert skills["uploaded-summary"]["input_schema"]["properties"]["model"]["x_param_kind"] == "cv_model"
    assert skills["uploaded-summary"]["input_schema"]["properties"]["legacy"]["x_param_kind"] == "other"

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("uploaded-plugin")
    result = SkillRunner(store, root_dir=tmp_path).run(
        "uploaded-summary",
        {"text": "hello plugin"},
        paths,
    )

    assert result.skill_name == "uploaded-summary"
    assert result.outputs[0].name == "summary.txt"
    assert (paths.outputs / "summary.txt").read_text(encoding="utf-8") == "hello plugin"

    saved_manifest = json.loads(
        (tmp_path / "plugins" / "skills" / "summary-plugin" / "skills" / "uploaded-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_manifest["input_schema"]["properties"]["model"]["x_param_kind"] == "cv_model"
    assert saved_manifest["input_schema"]["properties"]["legacy"]["x_param_kind"] == "other"
    assert "x-param-kind" not in saved_manifest["input_schema"]["properties"]["legacy"]


def test_skill_plugin_upload_can_replace_existing_plugin(tmp_path: Path) -> None:
    manager = SkillPluginManager(tmp_path)

    first = manager.install_zip(_plugin_zip())
    second = manager.install_zip(_plugin_zip())

    assert first.plugin_id == "summary-plugin"
    assert second.plugin_id == "summary-plugin"
    assert "uploaded-summary" in manager.load_skills()


def test_skill_plugin_local_path_install_registers_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())
    zip_path = tmp_path / "summary-plugin.zip"
    zip_path.write_bytes(_plugin_zip())

    response = client.post("/api/skills/plugins/local-path", json={"path": str(zip_path)})

    assert response.status_code == 200
    assert response.json()["plugin"]["id"] == "summary-plugin"
    listed = client.get("/api/skills")
    assert listed.status_code == 200
    skills = {skill["name"]: skill for skill in listed.json()["skills"]}
    assert "uploaded-summary" in skills


def test_builtin_skill_plugin_uses_complete_skill_packages() -> None:
    root = Path(__file__).resolve().parents[1] / "plugins" / "skills" / "builtin-artifact-skills" / "skills"
    package_names = {package.name for package in root.iterdir() if package.is_dir()}
    assert "deliverables-export" in package_names
    complete_package_names = package_names - {"behavior-review"}
    for package in root.iterdir():
        if not package.is_dir() or package.name not in complete_package_names:
            continue
        assert (package / "SKILL.md").is_file()
        assert (package / "manifest.json").is_file()
        manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
        assert isinstance(manifest.get("input_schema"), dict)
        assert isinstance(manifest.get("output_schema"), dict)
        assert (package / "requirements.txt").is_file()
        assert (package / "sandbox.yml").is_file()
        assert (package / "runner.py").is_file()
        assert (package / "spec_builder.py").is_file()


def test_tianjin_park_skill_defaults_to_local_generation_without_backend(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("tianjin-local")

    result = SkillRunner(store).run(
        "tianjin-chatbi-analyst",
        {"action": "query", "query": "统计各单位火警数量", "chart_type": "bar"},
        paths,
    )

    assert result.data["mode"] == "local"
    assert result.data["success"] is True
    assert result.outputs[0].name == "tianjin-chatbi-analyst-query.md"
    report = (paths.outputs / "tianjin-chatbi-analyst-query.md").read_text(encoding="utf-8")
    assert "智能问数分析" in report
    assert "统计各单位火警数量" in report


def test_tianjin_park_skill_uses_runtime_llm_when_available(tmp_path: Path) -> None:
    server = _OpenAICompatibleTestServer("## 模型生成结果\n\n天津园区会议纪要已由大模型生成。")
    server.start()
    try:
        store = ArtifactStore(root_dir=tmp_path / "runtime")
        paths = store.prepare_thread("tianjin-llm")

        result = SkillRunner(store).run(
            "tianjin-meeting-minutes",
            {
                "action": "analyze",
                "title": "天津园区例会",
                "transcriptLines": [{"speakerName": "张工", "time": "09:30", "text": "张工负责完成园区大屏接入。"}],
                "_llm_base_url": server.base_url,
                "_llm_model": "test-model",
                "_llm_api_key": "test-key",
                "_llm_max_tokens": 512,
            },
            paths,
        )
    finally:
        server.stop()

    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["headers"]["authorization"] == "Bearer test-key"
    assert server.requests[0]["body"]["model"] == "test-model"
    assert result.data["mode"] == "local"
    assert result.data["llm_configured"] is True
    report = (paths.outputs / "tianjin-meeting-minutes-analyze.md").read_text(encoding="utf-8")
    assert "模型生成结果" in report
    assert "天津园区会议纪要已由大模型生成" in report


def test_tianjin_document_skill_local_mode_uses_defaults_for_model_generation(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("tianjin-document-local")

    result = SkillRunner(store).run(
        "tianjin-document-generator",
        {
            "action": "generate",
            "projectName": "天津园区智能运营平台",
            "clientName": "天津示例客户",
            "keyInfo": "模型原生生成商务文档",
        },
        paths,
    )

    names = [artifact.name for artifact in result.outputs]
    assert result.data["mode"] == "local"
    assert result.data["success"] is True
    assert "天津园区智能运营平台-方案建议书.md" in names
    report = (paths.outputs / "tianjin-document-generator-generate.md").read_text(encoding="utf-8")
    assert "天津园区智能运营平台" in report
    assert "模型原生生成商务文档" in report


def test_tianjin_park_skill_can_call_backend_when_explicitly_requested(tmp_path: Path) -> None:
    server = _TianjinTestServer()
    server.start()
    try:
        store = ArtifactStore(root_dir=tmp_path / "runtime")
        paths = store.prepare_thread("tianjin-backend")

        result = SkillRunner(store).run(
            "tianjin-chatbi-analyst",
            {
                "action": "query",
                "mode": "backend",
                "query": "统计各单位火警数量",
                "chart_type": "bar",
                "api_base_url": server.api_base_url,
            },
            paths,
        )
    finally:
        server.stop()

    assert server.requests[0]["method"] == "POST"
    assert server.requests[0]["path"] == "/api/v1/chatbi/query"
    assert server.requests[0]["body"]["query"] == "统计各单位火警数量"
    assert result.data["ok"] is True
    assert result.data["success"] is True
    assert result.outputs[0].name == "tianjin-chatbi-analyst-query.md"
    assert result.outputs[1].name == "tianjin-chatbi-analyst-query.json"
    report = (paths.outputs / "tianjin-chatbi-analyst-query.md").read_text(encoding="utf-8")
    assert "天津园区智能问数结果" in report
    assert "火警数量" in report


def test_tianjin_document_skill_downloads_generated_document(tmp_path: Path) -> None:
    server = _TianjinTestServer()
    server.start()
    try:
        store = ArtifactStore(root_dir=tmp_path / "runtime")
        paths = store.prepare_thread("tianjin-document")

        result = SkillRunner(store).run(
            "tianjin-document-generator",
            {
                "action": "generate",
                "document_type": "price_file",
                "mode": "backend",
                "template_type": "software",
                "project_name": "智慧园区平台",
                "client_name": "天津示例客户",
                "key_info": "建设智慧园区运营平台",
                "api_base_url": server.api_base_url,
            },
            paths,
        )
    finally:
        server.stop()

    names = [artifact.name for artifact in result.outputs]
    assert server.requests[0]["path"] == "/api/v1/document/generate"
    assert server.requests[1]["path"] == "/api/v1/document/download/doc-1"
    assert "智慧园区平台报价.txt" in names
    assert (paths.outputs / "智慧园区平台报价.txt").read_text(encoding="utf-8") == "报价文档正文"


def test_tianjin_meeting_skill_normalizes_transcript_aliases(tmp_path: Path) -> None:
    server = _TianjinTestServer()
    server.start()
    try:
        store = ArtifactStore(root_dir=tmp_path / "runtime")
        paths = store.prepare_thread("tianjin-meeting")

        result = SkillRunner(store).run(
            "tianjin-meeting-minutes",
            {
                "action": "analyze",
                "mode": "backend",
                "title": "天津园区例会",
                "transcriptLines": [
                    {
                        "speakerName": "张工",
                        "time": "09:30",
                        "text": "张工负责完成消防告警联动，2026-05-20日交付。",
                    }
                ],
                "api_base_url": server.api_base_url,
            },
            paths,
        )
    finally:
        server.stop()

    request = server.requests[0]
    assert request["path"] == "/api/v1/meeting/analyze"
    assert request["body"]["name"] == "天津园区例会"
    assert request["body"]["transcriptLines"] == [
        {
            "speaker": "张工",
            "timestamp": "09:30",
            "content": "张工负责完成消防告警联动，2026-05-20日交付。",
        }
    ]
    assert result.data["success"] is True
    report = (paths.outputs / "tianjin-meeting-minutes-analyze.md").read_text(encoding="utf-8")
    assert "消防告警联动" in report


def test_tianjin_calendar_skill_normalizes_natural_text_aliases(tmp_path: Path) -> None:
    server = _TianjinTestServer()
    server.start()
    try:
        store = ArtifactStore(root_dir=tmp_path / "runtime")
        paths = store.prepare_thread("tianjin-calendar")

        result = SkillRunner(store).run(
            "tianjin-calendar-reminder",
            {
                "action": "parse_event",
                "mode": "backend",
                "text": "明天下午三点在会议室开天津园区例会",
                "api_base_url": server.api_base_url,
            },
            paths,
        )
    finally:
        server.stop()

    request = server.requests[0]
    assert request["path"] == "/api/v1/calendar/events/parse"
    assert request["body"]["natural_text"] == "明天下午3点在会议室开天津园区例会"
    assert request["body"]["user_timezone"] == "Asia/Shanghai"
    assert result.data["success"] is True
    report = (paths.outputs / "tianjin-calendar-reminder-parse_event.md").read_text(encoding="utf-8")
    assert "明天下午3点" in report


def test_tianjin_dangerous_skill_requires_confirmation(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("tianjin-rpa")

    result = SkillRunner(store).run(
        "tianjin-rpa-operator",
        {"action": "start_process", "process_id": 1},
        paths,
    )

    assert result.data["requires_input"] is True
    assert result.data["required_inputs"][0]["type"] == "confirmation"


class _TianjinTestServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def api_base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/api/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer._handle(self)

            def do_POST(self) -> None:
                outer._handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or "0")
        raw_body = handler.rfile.read(length) if length else b""
        body: Any = {}
        if raw_body:
            body = json.loads(raw_body.decode("utf-8"))
        parsed = urlparse(handler.path)
        self.requests.append({"method": handler.command, "path": parsed.path, "query": parsed.query, "body": body})
        if parsed.path == "/api/v1/chatbi/query":
            self._json(
                handler,
                {
                    "success": True,
                    "message": "查询成功",
                    "data": {
                        "query": body["query"],
                        "sql": "select unit, count(*) as 火警数量 from alarms group by unit",
                        "data": [{"联网单位": "A园区", "火警数量": 3}],
                        "analysis": "A园区火警数量为3",
                        "chart": {"type": "bar"},
                    },
                },
            )
            return
        if parsed.path == "/api/v1/document/generate":
            self._json(
                handler,
                {
                    "success": True,
                    "message": "文档生成成功",
                    "data": {
                        "document_id": "doc-1",
                        "title": "智慧园区平台报价",
                        "content": "报价文档正文",
                        "download_url": "/api/v1/document/download/doc-1?format=txt",
                    },
                },
            )
            return
        if parsed.path == "/api/v1/meeting/analyze":
            line = body["transcriptLines"][0]
            self._json(
                handler,
                {
                    "success": True,
                    "message": "会议分析完成",
                    "data": {
                        "summaryPoints": [
                            {"id": 1, "time": line["timestamp"], "type": "key", "content": line["content"]}
                        ],
                        "tasks": [
                            {
                                "id": 1,
                                "priority": "medium",
                                "extractTime": line["timestamp"],
                                "description": "完成消防告警联动",
                                "assignee": line["speaker"],
                                "deadline": "2026-05-20",
                            }
                        ],
                        "keywords": [{"word": "消防告警联动", "count": 1}],
                        "topics": [{"name": "园区运营", "frequency": 80}],
                    },
                },
            )
            return
        if parsed.path == "/api/v1/calendar/events/parse":
            self._json(
                handler,
                {
                    "success": True,
                    "message": "事件解析成功",
                    "data": {
                        "title": "天津园区例会",
                        "start_time": "2026-05-20T15:00:00+08:00",
                        "end_time": "2026-05-20T16:00:00+08:00",
                        "priority": "medium",
                        "original_text": body["natural_text"],
                    },
                },
            )
            return
        if parsed.path == "/api/v1/document/download/doc-1":
            payload = "报价文档正文".encode("utf-8")
            handler.send_response(200)
            handler.send_header("Content-Type", "text/plain; charset=utf-8")
            handler.send_header("Content-Disposition", "attachment; filename*=UTF-8''%E6%99%BA%E6%85%A7%E5%9B%AD%E5%8C%BA%E5%B9%B3%E5%8F%B0%E6%8A%A5%E4%BB%B7.txt")
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
            return
        self._json(handler, {"success": False, "message": "not found"}, status=404)

    @staticmethod
    def _json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)


class _OpenAICompatibleTestServer:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(length) if length else b""
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                parsed = urlparse(self.path)
                outer.requests.append(
                    {
                        "path": parsed.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": body,
                    }
                )
                payload = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": outer.content},
                            "finish_reason": "stop",
                        }
                    ]
                }
                response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler


def test_skill_registry_exposes_model_tags_as_metadata() -> None:
    registry = SkillRegistry()

    data_auto = registry.get("data-auto-annotation")
    algorithm_research = registry.get("algorithm-research-scout")
    image_dataset_generation = registry.get("image-dataset-generation")
    image_composite_generation = registry.get("image-composite-generation")
    payload = data_auto.to_event_payload()

    assert data_auto.model_tags == ("vision_segmentation",)
    assert algorithm_research.model_tags == ("chat", "reasoning", "rerank")
    assert image_dataset_generation.executable is True
    assert image_dataset_generation.model_tags == ("image_generation", "vision_segmentation")
    assert image_composite_generation.executable is True
    assert image_composite_generation.model_tags == ("image_generation", "vision_segmentation")
    assert payload["model_tags"] == ["vision_segmentation"]
    assert "model_tags" not in (data_auto.input_schema or {}).get("properties", {})


def test_image_composite_spec_builder_uses_two_image_attachments() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "skills"
        / "image-composite-generation"
        / "spec_builder.py"
    )
    spec = importlib.util.spec_from_file_location("image_composite_spec_builder_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    attachments = [
        {"name": "background.jpg", "path": "/tmp/background.jpg", "mime_type": "image/jpeg"},
        {"name": "object.png", "path": "/tmp/object.png", "mime_type": "image/png"},
    ]

    score = module.score_skill(
        "image-composite-generation",
        "请做图片合成 prompt: 目标自然出现在监控画面中",
        attachments,
        ["image-composite-generation"],
    )
    built = module.build_spec(
        "image-composite-generation",
        "请做图片合成 prompt: 目标自然出现在监控画面中",
        "",
        attachments,
        {},
    )

    assert score >= 120
    assert built["image1"] == "/tmp/background.jpg"
    assert built["image2"] == "/tmp/object.png"
    assert built["prompt"] == "目标自然出现在监控画面中"
    assert built["api_url"] == "https://www.tokencloud.yun/v1/images/generations"
    assert built["token"].startswith("sk-")
    assert built["model"] == "wan2.7-image-pro"


def test_behavior_review_outputs_second_pass_logic_and_continuous_state(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("behavior-review-continuous")

    result = SkillRunner(store).run(
        "behavior-review",
        {
            "event_id": "alarm-1",
            "batch_id": "batch-1",
            "review_round": 3,
            "text_rule_candidates": ["翻越进入禁区"],
            "has_visual_evidence": True,
            "detector_confidence": 0.91,
            "rule_confidence": 0.8,
            "continuous_review": True,
            "poll_interval_seconds": 3,
        },
        paths,
    )

    assert result.data["review_decision"] == "confirm_incident"
    assert result.data["risk_level"] == "high"
    assert result.data["risk_score"] >= 0.87
    assert result.data["continuous_review"]["enabled"] is True
    assert result.data["continuous_review"]["next_action"] == "poll_next_batch"
    assert [step["stage"] for step in result.data["second_review_logic"]] == [
        "文本规则初筛",
        "证据完整性检查",
        "模型/规则一致性",
        "误报抑制",
        "二次研判结论",
    ]
    report = (paths.outputs / "behavior-review.md").read_text(encoding="utf-8")
    assert "## 二次研判链路" in report
    assert "## 连续复判" in report


def test_behavior_review_suppresses_visual_confirmation_without_evidence(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("behavior-review-no-evidence")

    result = SkillRunner(store).run(
        "behavior-review",
        {
            "text_rule_candidates": ["老人摔倒"],
            "has_visual_evidence": False,
            "continuous_review": True,
        },
        paths,
    )

    assert result.data["review_decision"] == "need_more_evidence"
    assert result.data["risk_score"] == 0.0
    assert result.data["manual_review_required"] is True
    assert result.data["second_review_logic"][1]["decision"] == "evidence_missing"


def test_drawio_generation_escapes_attribute_quotes(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("drawio-quotes")

    result = SkillRunner(store).run(
        "drawio-generation",
        {
            "title": 'JetLinks "Smoke" 架构',
            "nodes": ['入口 "A"', 'Runtime "B"'],
            "edges": [['入口 "A"', 'Runtime "B"']],
            "swimlanes": ['交互 "层"'],
        },
        paths,
    )

    drawio_artifact = next(artifact for artifact in result.outputs if artifact.name.endswith(".drawio"))
    ElementTree.fromstring((paths.outputs / drawio_artifact.name).read_text(encoding="utf-8"))


def test_skill_registry_uses_config_skills_as_entity_catalog() -> None:
    registry = SkillRegistry()
    skills = registry.list()
    names = {skill.name for skill in skills}
    config_names = {path.stem for path in (registry.root_dir / "config" / "skills").glob("*.json")}

    assert names == config_names
    assert "public-skill-demo" not in names
    assert "algorithm-engineer-app" not in names
    assert "behavior-review" in names
    assert "deliverables-export" in names
    for skill in skills:
        assert skill.manifest_path is not None
        assert "config/skills" in str(skill.manifest_path)


def test_formal_skills_have_top_level_plugin_entities() -> None:
    manager = SkillPluginManager()
    root = manager.root_dir
    config_names = {path.stem for path in (root / "config" / "skills").glob("*.json")}
    plugin_roots = {
        path.name
        for path in (root / "plugins" / "skills").iterdir()
        if path.is_dir() and (path / "plugin.json").is_file()
    }
    loaded = manager.load_skills()

    assert config_names - plugin_roots == set()
    for skill_name in config_names:
        loaded_skill = loaded[skill_name]
        assert loaded_skill.plugin is not None, skill_name
        assert loaded_skill.plugin.plugin_id == skill_name
        assert loaded_skill.plugin.root.name == skill_name


def test_uploaded_plugin_is_materialized_as_local_skill_entity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/skills/plugins",
        files={"file": ("summary-plugin.zip", _plugin_zip(), "application/zip")},
    )

    assert response.status_code == 200
    entity_path = tmp_path / "config" / "skills" / "uploaded-summary.json"
    assert entity_path.is_file()
    assert json.loads(entity_path.read_text(encoding="utf-8"))["name"] == "uploaded-summary"

    listed = client.get("/api/skills")
    assert listed.status_code == 200
    skills = {skill["name"]: skill for skill in listed.json()["skills"]}
    assert skills["uploaded-summary"]["source_path"] == str(entity_path)
    assert skills["uploaded-summary"]["source"]["plugin_id"] == "summary-plugin"


def test_algorithm_engineer_spec_builder_does_not_treat_agx_5090_as_dataset_path() -> None:
    spec_builder = _load_algorithm_engineer_spec_builder()

    spec = spec_builder.build_spec(
        "algorithm-engineer",
        "围绕通用目标检测，打通 AGX/5090 训练编排和业务评估。",
        "围绕通用目标检测，打通 AGX/5090 训练编排和业务评估。",
        [],
        {},
    )

    assert spec["dataset_path"] == ""
    assert spec["domain"] == "multi-domain computer vision object detection"
    assert spec["machines"] == ["AGX Orin", "5090 GPU server"]


def test_algorithm_engineer_spec_builder_extracts_user_named_cv_task() -> None:
    spec_builder = _load_algorithm_engineer_spec_builder()

    smoking = spec_builder.build_spec(
        "algorithm-engineer",
        "帮我训练一个抽烟 CV 检测模型。",
        "帮我训练一个抽烟 CV 检测模型。",
        [],
        {},
    )
    helmet = spec_builder.build_spec(
        "algorithm-engineer",
        "帮我训练一个安全帽佩戴检测模型。",
        "帮我训练一个安全帽佩戴检测模型。",
        [],
        {},
    )
    defect = spec_builder.build_spec(
        "algorithm-engineer",
        "帮我训练一个工业缺陷检测模型。",
        "帮我训练一个工业缺陷检测模型。",
        [],
        {},
    )

    assert smoking["objective"] == "抽烟检测模型训练"
    assert smoking["domain"] == "抽烟视觉检测"
    assert helmet["objective"] == "安全帽佩戴检测模型训练"
    assert helmet["domain"] == "安全帽佩戴视觉检测"
    assert defect["objective"] == "工业缺陷检测模型训练"
    assert defect["domain"] == "工业缺陷视觉检测"


def test_algorithm_engineer_spec_builder_extracts_cpu_training_options() -> None:
    spec_builder = _load_algorithm_engineer_spec_builder()

    spec = spec_builder.build_spec(
        "cpu-training-runner",
        "用 CPU 沙盒训练 data_yaml=/tmp/palm/data.yaml model=yolo11n.pt epochs=2 imgsz=320 batch=1 mock=true",
        "用 CPU 沙盒训练 data_yaml=/tmp/palm/data.yaml model=yolo11n.pt epochs=2 imgsz=320 batch=1 mock=true",
        [],
        {},
    )

    assert spec["data_yaml"] == "/tmp/palm/data.yaml"
    assert spec["model"] == "yolo11n.pt"
    assert spec["epochs"] == 2
    assert spec["imgsz"] == 320
    assert spec["batch"] == 1
    assert spec["mock"] is True


def test_cpu_training_runner_mock_generates_best_pt(tmp_path: Path) -> None:
    request = {
        "request_schema_version": "skill-run.v1",
        "skill_name": "cpu-training-runner",
        "thread_id": "cpu-train-test",
        "spec": {
            "data_yaml": "",
            "model": "yolo11n.pt",
            "epochs": 1,
            "imgsz": 320,
            "batch": 1,
            "experiment_name": "mock_cpu_train",
            "mock": True,
        },
        "workspace_dir": str(tmp_path / "workspace"),
        "outputs_dir": str(tmp_path / "outputs"),
    }
    request_path = tmp_path / "request.json"
    outputs_dir = tmp_path / "outputs"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    script = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "skills"
        / "algorithm-engineer"
        / "scripts"
        / "cpu_training_runner.py"
    )

    completed = subprocess.run(
        [sys.executable, str(script), "--request", str(request_path), "--outputs", str(outputs_dir)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert (outputs_dir / "best.pt").is_file()
    assert (outputs_dir / "last.pt").is_file()
    assert (outputs_dir / "results.csv").is_file()
    assert (outputs_dir / "training-summary.json").is_file()
    summary = json.loads((outputs_dir / "training-summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["best_pt"] == "best.pt"
    assert summary["mock"] is True


def test_cpu_training_runner_skill_mock_generates_best_pt(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("cpu-train-skill-test")

    result = SkillRunner(store).run(
        "cpu-training-runner",
        {
            "data_yaml": "",
            "model": "yolo11n.pt",
            "epochs": 1,
            "imgsz": 320,
            "batch": 1,
            "mock": True,
        },
        paths,
    )

    names = [artifact.name for artifact in result.outputs]
    assert "best.pt" in names
    assert "last.pt" in names
    assert "results.csv" in names
    assert "training-summary.json" in names
    assert result.data["status"] == "completed"
    assert result.data["best_pt"] == "best.pt"
    assert result.data["mock"] is True


def test_algorithm_engineer_sequence_reply_is_professional_status_card() -> None:
    runner = _load_algorithm_engineer_runner()
    run_result = type(
        "RunResult",
        (),
        {
            "data": {
                "sequence": [
                    {"skill_name": "algorithm-engineer"},
                    {"skill_name": "dataset-curator"},
                    {"skill_name": "algorithm-research-scout"},
                    {"skill_name": "model-candidate-selector"},
                    {"skill_name": "remote-gpu-ops"},
                    {"skill_name": "gpu-training-orchestrator"},
                    {"skill_name": "detector-evaluator"},
                    {"skill_name": "deployment-candidate-reviewer"},
                    {"skill_name": "experiment-ledger"},
                ]
            },
            "outputs": [
                type("Artifact", (), {"name": "algorithm-engineer.md"})(),
                type("Artifact", (), {"name": "dataset-curator.md"})(),
                type("Artifact", (), {"name": "gpu-training-orchestrator.md"})(),
                type("Artifact", (), {"name": "detector-evaluator.md"})(),
                type("Artifact", (), {"name": "experiment-ledger.md"})(),
            ],
        },
    )()

    reply = runner.format_reply("algorithm-engineer", object(), run_result)

    assert reply.startswith("算法工程师全流程工作台已搭好。")
    assert "当前看板：" in reply
    assert "数据治理：已建立" in reply
    assert "CPU 训练沙盒" in reply
    assert "algorithm-engineer, dataset-curator" not in reply


def _load_algorithm_engineer_runner() -> object:
    path = Path(__file__).resolve().parents[1] / "plugins" / "skills" / "algorithm-engineer" / "runner.py"
    spec = importlib.util.spec_from_file_location("algorithm_engineer_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_algorithm_engineer_spec_builder() -> object:
    path = Path(__file__).resolve().parents[1] / "plugins" / "skills" / "algorithm-engineer" / "spec_builder.py"
    spec = importlib.util.spec_from_file_location("algorithm_engineer_spec_builder_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builtin_deliverables_export_skill_executes(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("deliverables")
    result = SkillRunner(store).run(
        "deliverables-export",
        {
            "commands": "uv run pytest",
            "steps": "# 验证步骤\n\n- 运行测试\n- 检查输出",
            "commands_name": "commands.txt",
            "steps_name": "steps.docx",
        },
        paths,
    )

    assert result.skill_name == "deliverables-export"
    assert [artifact.name for artifact in result.outputs] == ["commands.txt", "steps.docx"]
    assert (paths.outputs / "commands.txt").read_text(encoding="utf-8") == "uv run pytest"
    assert zipfile.is_zipfile(paths.outputs / "steps.docx")


def test_skill_package_file_api_reads_and_updates_package_assets(tmp_path: Path) -> None:
    client = TestClient(create_app())

    files_response = client.get("/api/skills/deliverables-export/files")
    assert files_response.status_code == 200
    files = {item["id"]: item for item in files_response.json()["files"]}
    assert {"manifest", "skill-md", "requirements", "runner", "script-runner"} <= set(files)
    assert "input-schema" not in files
    assert "output-schema" not in files
    assert Path(files["skill-md"]["path"]).parts[-6:] == ("plugins", "skills", "deliverables-export", "skills", "deliverables-export", "SKILL.md")

    read_response = client.get("/api/skills/deliverables-export/files/skill-md")
    assert read_response.status_code == 200
    original = read_response.json()["content"]
    assert "name: deliverables-export" in original

    custom_root = tmp_path / "custom"
    custom_root.mkdir()
    plugin_root = custom_root / "plugins" / "skills" / "editable-plugin"
    skill_root = plugin_root / "skills" / "editable-skill"
    skill_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        """
{
  "id": "editable-plugin",
  "name": "Editable Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "editable-skill",
  "description": "Editable",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": [],
  "input_schema": {
    "type": "object",
    "properties": {
      "model": {"type": "string", "x_param_kind": "cv_model"},
      "legacy": {"type": "string", "x-param-kind": "custom-legacy-kind"}
    }
  },
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )
    (skill_root / "runner.py").write_text("def run(skill_name, spec, paths, artifact_store):\n    return {}\n", encoding="utf-8")

    manager = SkillPluginManager(custom_root)
    manifest_file, saved = manager.write_package_file(
        "editable-skill",
        "manifest",
        """
{
  "name": "editable-skill",
  "description": "Editable",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": [],
  "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
    )

    assert manifest_file.exists is True
    assert '"title"' in saved
    assert "title" in manager.read_manifest("editable-skill")["input_schema"]["properties"]


def test_skill_manifest_config_api_generates_manifest_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())

    response = client.put(
        "/api/skills/custom-report/manifest-config",
        json={
            "config": {
                "description": "Custom report skill",
                "output_kind": "markdown",
                "generation": True,
                "quality_template": ["summary", "table"],
                "routing": {"keywords": ["report"]},
                "execution": {"type": "template", "filename": "report.md", "template": "# $title"},
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "x_param_kind": "model"},
                        "legacy": {"type": "string", "x_param_kind": "custom-old-kind"},
                    },
                },
                "output_schema": {"type": "object", "properties": {"artifacts": {"type": "array"}}},
                "sandbox": {
                    "enabled": False,
                    "profile": "",
                    "request_schema_version": "skill-run.v1",
                    "adapter_command": "",
                    "fallback_to_local": True,
                },
            }
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "custom-report"
    assert data["source_exists"] is True
    manifest_path = tmp_path / "config" / "skills" / "custom-report.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "custom-report"
    assert manifest["routing"]["keywords"] == ["report"]
    assert manifest["execution"]["type"] == "template"
    assert manifest["input_schema"]["properties"]["title"]["type"] == "string"
    assert manifest["input_schema"]["properties"]["title"]["x_param_kind"] == "model"
    assert manifest["input_schema"]["properties"]["legacy"]["x_param_kind"] == "other"
    assert manifest["sandbox"]["fallback_to_local"] is True


def test_generic_template_skill_executes_without_runner(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "skills" / "generic-template-plugin"
    skill_root = plugin_root / "skills" / "generic-template"
    skill_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        """
{
  "id": "generic-template-plugin",
  "name": "Generic Template Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "generic-template",
  "description": "Generic template",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": [],
  "execution": {
    "type": "template",
    "filename": "$title.md",
    "template": "# $title\\n\\n$body"
  },
  "input_schema": {
    "type": "object",
    "properties": {
      "model": {"type": "string", "x_param_kind": "cv_model"},
      "legacy": {"type": "string", "x-param-kind": "custom-legacy-kind"}
    }
  },
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("generic-template")
    result = SkillRunner(store, root_dir=tmp_path).run(
        "generic-template",
        {"title": "daily-report", "body": "done"},
        paths,
    )

    assert result.skill_name == "generic-template"
    assert result.outputs[0].name == "daily-report.md"
    assert (paths.outputs / "daily-report.md").read_text(encoding="utf-8") == "# daily-report\n\ndone"
    executable_names = {skill.name for skill in SkillRegistry(tmp_path).list(executable_only=True)}
    assert "generic-template" not in executable_names


def test_public_template_skill_is_discoverable_and_executes(tmp_path: Path) -> None:
    registry = SkillRegistry()

    with pytest.raises(KeyError):
        registry.get("public-skill-demo")
    assert "public-skill-demo" not in {item.name for item in registry.list(executable_only=True)}

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("public-skill-demo")
    result = SkillRunner(store).run(
        "public-skill-demo",
        {"title": "public-demo-output", "message": "public demo smoke"},
        paths,
    )

    assert result.skill_name == "public-skill-demo"
    assert result.outputs[0].name == "public-demo-output.md"
    assert (paths.outputs / "public-demo-output.md").read_text(encoding="utf-8") == "# public-demo-output\n\npublic demo smoke"


def test_python_script_skill_requires_only_script_config(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "skills" / "python-script-plugin"
    skill_root = plugin_root / "skills" / "python-script-skill"
    script_root = skill_root / "scripts"
    script_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        """
{
  "id": "python-script-plugin",
  "name": "Python Script Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "python-script-skill",
  "description": "Python script skill",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": [],
  "execution": {
    "type": "python_script",
    "script": "scripts/run_skill.py"
  },
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )
    (script_root / "run_skill.py").write_text(
        """
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
spec = payload["spec"]
outputs = Path(payload["outputs_dir"])
(outputs / "result.md").write_text(f"# {spec['title']}\\n", encoding="utf-8")
print(json.dumps({"message": "ok"}, ensure_ascii=False))
""",
        encoding="utf-8",
    )

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("python-script-skill")
    result = SkillRunner(store, root_dir=tmp_path).run(
        "python-script-skill",
        {"title": "Python Skill"},
        paths,
    )

    assert result.skill_name == "python-script-skill"
    assert result.data["message"] == "ok"
    assert result.outputs[0].name == "result.md"
    assert (paths.outputs / "result.md").read_text(encoding="utf-8") == "# Python Skill\n"


def test_python_script_skill_expands_thread_virtual_paths_in_spec(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "skills" / "virtual-path-plugin"
    skill_root = plugin_root / "skills" / "virtual-path-skill"
    script_root = skill_root / "scripts"
    script_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        """
{
  "id": "virtual-path-plugin",
  "name": "Virtual Path Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "virtual-path-skill",
  "description": "Virtual path skill",
  "output_kind": "markdown",
  "generation": true,
  "quality_template": [],
  "execution": {
    "type": "python_script",
    "script": "scripts/run_skill.py"
  },
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )
    (script_root / "run_skill.py").write_text(
        """
import json
import sys
from pathlib import Path

payload = json.load(sys.stdin)
image_path = Path(payload["spec"]["image_path"])
outputs = Path(payload["outputs_dir"])
content = image_path.read_text(encoding="utf-8")
(outputs / "result.md").write_text(str(image_path) + "\\n" + content, encoding="utf-8")
print(json.dumps({"image_path": str(image_path)}, ensure_ascii=False))
""",
        encoding="utf-8",
    )

    store = ArtifactStore(root_dir=tmp_path / "runtime")
    paths = store.prepare_thread("virtual-path-skill")
    (paths.uploads / "input.txt").write_text("uploaded content", encoding="utf-8")
    result = SkillRunner(store, root_dir=tmp_path).run(
        "virtual-path-skill",
        {"image_path": "/mnt/user-data/uploads/input.txt"},
        paths,
    )

    output = (paths.outputs / "result.md").read_text(encoding="utf-8")
    assert str(paths.uploads / "input.txt") in result.data["image_path"]
    assert str(paths.uploads / "input.txt") in output
    assert "uploaded content" in output


def test_image_composite_runner_normalizes_runtime_payload_and_collects_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "skills"
        / "image-composite-generation"
        / "scripts"
        / "run_composite.py"
    )
    image1 = tmp_path / "background.jpg"
    image2 = tmp_path / "object.png"
    output_dir = tmp_path / "outputs"
    child = tmp_path / "fake_child.py"
    image1.write_bytes(b"image1")
    image2.write_bytes(b"image2")
    child.write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--url")
parser.add_argument("--token")
parser.add_argument("--model")
parser.add_argument("--image1")
parser.add_argument("--image2")
parser.add_argument("--prompt")
parser.add_argument("--output-dir")
parser.add_argument("--size")
parser.add_argument("--count")
parser.add_argument("--timeout")
parser.add_argument("--sleep")
parser.add_argument("--retries")
parser.add_argument("--retry-sleep")
args = parser.parse_args()
target = Path(args.output_dir) / "composite_001.png"
target.write_bytes(b"png")
print(f"Composite success. saved: {target}")
""",
        encoding="utf-8",
    )
    payload = {
        "spec": {
            "attachments": [
                {"name": "background.jpg", "path": str(image1), "mime_type": "image/jpeg"},
                {"name": "object.png", "path": str(image2), "mime_type": "image/png"},
            ],
            "prompt": "自然合成到监控画面",
            "count": 1,
            "timeout": 1,
        },
        "outputs_dir": str(output_dir),
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("IMAGE_COMPOSITE_SCRIPT", str(child))

    completed = subprocess.run(
        [sys.executable, str(script), "--input", str(input_path)],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "composite_001.png").read_bytes() == b"png"
    assert "result_image_path:" in completed.stdout
    assert "status: success" in completed.stdout


def test_python_script_manifest_rejects_missing_script_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "plugins" / "skills" / "missing-script-plugin"
    skill_root = package_root / "skills" / "missing-script"
    skill_root.mkdir(parents=True)
    (package_root / "plugin.json").write_text(
        """
{
  "id": "missing-script-plugin",
  "name": "Missing Script Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "missing-script",
  "description": "Missing script",
  "output_kind": "json",
  "generation": true,
  "quality_template": [],
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())

    response = client.put(
        "/api/skills/missing-script/manifest-config",
        json={
            "config": {
                "description": "Missing script",
                "output_kind": "json",
                "generation": True,
                "quality_template": [],
                "execution": {"type": "python_script", "script": "scripts/run_skill.py"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "sandbox": {"enabled": False, "profile": None, "request_schema_version": "skill-run.v1"},
            }
        },
    )

    assert response.status_code == 400
    assert "Python script not found" in response.json()["detail"]


def test_python_script_config_override_uses_plugin_root_for_single_skill_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_root = tmp_path / "plugins" / "skills" / "data-auto-annotation"
    script_root = plugin_root / "scripts"
    script_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        """
{
  "id": "data-auto-annotation",
  "name": "data-auto-annotation",
  "version": "1.0.0",
  "skills": ["SKILL.md"]
}
""",
        encoding="utf-8",
    )
    (plugin_root / "SKILL.md").write_text(
        """
---
name: data-auto-annotation
description: Data annotation
---
""",
        encoding="utf-8",
    )
    (script_root / "sam3-predict.py").write_text("print('{}')\n", encoding="utf-8")
    config_dir = tmp_path / "config" / "skills"
    config_dir.mkdir(parents=True)
    (config_dir / "data-auto-annotation.json").write_text(
        """
{
  "name": "data-auto-annotation",
  "description": "Data annotation",
  "output_kind": "json",
  "generation": true,
  "execution": {
    "type": "python_script",
    "script": "scripts/sam3-predict.py"
  },
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())

    detail = client.get("/api/skills/data-auto-annotation")
    assert detail.status_code == 200
    file_paths = {Path(item["path"]).name for item in detail.json()["package_files"]}
    assert "SKILL.md" in file_paths
    assert "requirements.txt" in file_paths
    assert "sam3-predict.py" in file_paths

    response = client.put(
        "/api/skills/data-auto-annotation/manifest-config",
        json={
            "config": {
                "description": "Data annotation updated",
                "output_kind": "json",
                "generation": True,
                "quality_template": [],
                "execution": {"type": "python_script", "script": "scripts/sam3-predict.py"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "sandbox": {"enabled": True, "profile": "python-skill", "request_schema_version": "skill-run.v1"},
            }
        },
    )

    assert response.status_code == 200
    assert (plugin_root / "sandbox.yml").is_file()


def test_skill_manifest_config_api_syncs_sandbox_yaml_for_package_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "plugins" / "skills" / "manifest-sync-plugin"
    skill_root = package_root / "skills" / "report-sync"
    skill_root.mkdir(parents=True)
    (package_root / "plugin.json").write_text(
        """
{
  "id": "manifest-sync-plugin",
  "name": "Manifest Sync Plugin",
  "version": "1.0.0",
  "skills": ["skills/*/manifest.json"]
}
""",
        encoding="utf-8",
    )
    (skill_root / "manifest.json").write_text(
        """
{
  "name": "report-sync",
  "description": "Report Sync",
  "output_kind": "json",
  "generation": true,
  "quality_template": [],
  "input_schema": {"type": "object"},
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": "", "request_schema_version": "skill-run.v1"}
}
""",
        encoding="utf-8",
    )
    (skill_root / "runner.py").write_text("def run(skill_name, spec, paths, artifact_store):\n    return {}\n", encoding="utf-8")

    manager = SkillPluginManager(tmp_path)
    registry = SkillRegistry(tmp_path)
    monkeypatch.setattr(skills_api, "registry", registry)
    monkeypatch.setattr(skills_api, "plugin_manager", manager)
    client = TestClient(create_app())

    response = client.put(
        "/api/skills/report-sync/manifest-config",
        json={
            "config": {
                "description": "Report sync skill",
                "output_kind": "markdown",
                "generation": True,
                "quality_template": ["summary"],
                "routing": {"keywords": ["report"]},
                "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"artifacts": {"type": "array"}}},
                "sandbox": {
                    "enabled": True,
                    "profile": "report-profile",
                    "request_schema_version": "skill-run.v1",
                    "adapter_command": "python runner.py",
                    "fallback_to_local": False,
                },
            }
        },
    )

    assert response.status_code == 200
    sandbox_path = skill_root / "sandbox.yml"
    assert sandbox_path.is_file()
    sandbox_text = sandbox_path.read_text(encoding="utf-8")
    assert "sandbox:" in sandbox_text
    assert "enabled: true" in sandbox_text
    assert 'profile: "report-profile"' in sandbox_text
    assert 'adapter_command: "python runner.py"' in sandbox_text
    assert "fallback_to_local: false" in sandbox_text


def test_single_skill_markdown_package_upload_registers_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skills_api, "registry", SkillRegistry(tmp_path))
    monkeypatch.setattr(skills_api, "plugin_manager", SkillPluginManager(tmp_path))
    client = TestClient(create_app())

    response = client.post(
        "/api/skills/plugins",
        files={"file": ("markdown-skill.zip", _single_skill_markdown_zip(), "application/zip")},
    )

    assert response.status_code == 200
    assert response.json()["plugin"]["id"] == "markdown-skill"

    listed = client.get("/api/skills")
    assert listed.status_code == 200
    skills = {skill["name"]: skill for skill in listed.json()["skills"]}
    assert skills["markdown-skill"]["source"]["type"] == "plugin"
    assert skills["markdown-skill"]["input_schema"]["type"] == "object"
    assert skills["markdown-skill"]["executable"] is True


def _plugin_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "plugin.json",
            """
{
  "id": "summary-plugin",
  "name": "Summary Plugin",
  "version": "1.0.0",
  "runner": "runner.py",
  "skills": ["skills/*.json"]
}
""",
        )
        archive.writestr(
            "skills/uploaded-summary.json",
            """
{
  "name": "uploaded-summary",
  "description": "Uploaded summary skill.",
  "output_kind": "text",
  "generation": true,
  "quality_template": ["summary"],
  "input_schema": {
    "type": "object",
    "properties": {
      "model": {"type": "string", "x_param_kind": "cv_model"},
      "legacy": {"type": "string", "x-param-kind": "custom-legacy-kind"}
    }
  },
  "output_schema": {"type": "object"},
  "sandbox": {"enabled": false, "profile": null, "request_schema_version": "skill-run.v1"}
}
""",
        )
        archive.writestr(
            "runner.py",
            """
from __future__ import annotations


def run(skill_name, spec, paths, artifact_store):
    text = str(spec.get("text") or "empty")
    artifact = artifact_store.write_text_artifact(paths, "summary.txt", text)
    return {
        "skill_name": skill_name,
        "outputs": [artifact.model_dump()],
        "data": {"length": len(text)}
    }
""",
        )
    return buffer.getvalue()


def _single_skill_markdown_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "SKILL.md",
            """
---
name: markdown-skill
description: Uploaded SKILL.md package.
tags:
  - markdown
  - summary
---

# Uploaded Skill
""",
        )
        archive.writestr("requirements.txt", "")
        archive.writestr(
            "runner.py",
            """
from __future__ import annotations


def run(skill_name, spec, paths, artifact_store):
    artifact = artifact_store.write_text_artifact(paths, "uploaded.md", "# Uploaded")
    return {
        "skill_name": skill_name,
        "outputs": [artifact.model_dump()],
        "data": {"ok": True}
    }
""",
        )
    return buffer.getvalue()
