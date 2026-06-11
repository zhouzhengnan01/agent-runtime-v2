import io
import json

from tools.app_smoke_matrix import (
    auth_headers,
    continuation_spec,
    continuation_runtime_options,
    config_app_names,
    expected_artifact_patterns,
    ensure_smoke_image,
    get_json,
    is_retryable_failure,
    missing_expected_apps,
    missing_artifact_patterns,
    needs_smoke_image,
    post_json,
    render_markdown_report,
    runtime_options,
    should_check_continuation,
    smoke_template,
    template_expects_artifacts,
    unexpected_skill_artifacts,
    validate_artifact_content,
    write_smoke_outputs,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return io.BytesIO(self.payload)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_smoke_matrix_attaches_image_only_for_image_required_skills() -> None:
    assert needs_smoke_image({"category": "vision", "selected_skills": []}) is False
    assert needs_smoke_image({"category": "generation", "selected_skills": ["data-auto-annotation"]}) is True
    assert needs_smoke_image({"category": "algorithm-training", "selected_skills": ["reference-image-yolo-trainer"]}) is True
    assert needs_smoke_image({"category": "vision", "selected_skills": ["behavior-detection"]}) is False
    assert needs_smoke_image({"category": "algorithm", "selected_skills": ["dataset-curator"]}) is False


def test_smoke_matrix_behavior_detection_can_pass_without_artifacts() -> None:
    assert template_expects_artifacts({"workflow": "evidence_first_detection", "selected_skills": ["behavior-detection"]}) is False
    assert expected_artifact_patterns({"workflow": "evidence_first_detection", "selected_skills": ["behavior-detection"]}) == []


def test_smoke_matrix_expected_artifact_patterns_cover_known_skills() -> None:
    multi_skill_template = {
        "selected_skills": [
            "data-auto-annotation",
            "drawio-generation",
            "pptx-generation",
            "dataset-curator",
            "cpu-training-runner",
        ]
    }

    assert expected_artifact_patterns(multi_skill_template) == []
    assert expected_artifact_patterns({"selected_skills": ["cpu-training-runner"]}) == [
        "training-summary.md",
        "training-summary.json",
        "best.pt",
        "last.pt",
        "results.csv",
        "args.yaml",
    ]
    assert expected_artifact_patterns({"selected_skills": ["reference-image-yolo-trainer"]}) == [
        "training-summary.md",
        "best.pt",
        "results.csv",
    ]


def test_smoke_matrix_reports_missing_expected_artifacts() -> None:
    names = ["annotations.coco.json", "architecture.drawio", "deck.pptx"]

    assert missing_artifact_patterns(names, ["annotations.coco.json", "*.drawio", "*.png", "*.pptx"]) == ["*.png"]


def test_smoke_matrix_reports_unexpected_skill_artifacts() -> None:
    template = {"selected_skills": ["data-auto-annotation"]}
    names = [
        "annotations.coco.json",
        "data-auto-annotation-stderr.txt",
        "deck.pptx",
    ]

    assert unexpected_skill_artifacts(template, names) == ["deck.pptx"]
    assert unexpected_skill_artifacts({"selected_skills": ["pptx-generation"]}, ["deck.pptx"]) == []
    assert unexpected_skill_artifacts({"selected_skills": ["markdown-rendering"]}, ["result.md"]) == []
    assert unexpected_skill_artifacts(
        {"selected_skills": ["reference-image-yolo-trainer"]},
        [
            "annotations.coco.json",
            "data-auto-annotation-stderr.txt",
            "dataset-curator.json",
            "dataset-curator.md",
            "image-dataset-generation-stderr.txt",
            "training-summary.json",
            "last.pt",
            "args.yaml",
            "best.pt",
            "results.csv",
            "training-summary.md",
        ],
    ) == []
    assert unexpected_skill_artifacts({"selected_skills": []}, ["deck.pptx", "device-template.xlsx", "mindmap.xmind"]) == [
        "deck.pptx",
        "device-template.xlsx",
        "mindmap.xmind",
    ]


def test_smoke_matrix_runtime_options_force_autonomous_mode() -> None:
    options = runtime_options(
        {
            "workflow": "artifact_workflow",
            "selected_skills": ["drawio-generation"],
            "selected_mcp_tools": ["artifact_read"],
            "runtime_options": {"skill_parameters": {"drawio-generation": {"style": "layered"}}},
        },
        "thread-1",
        12,
    )

    assert options["thread_id"] == "thread-1"
    assert options["workflow"] == "artifact_workflow"
    assert options["selected_skills"] == ["drawio-generation"]
    assert options["selected_mcp_tools"] == ["artifact_read"]
    assert options["mode"] == "autonomous"
    assert options["config_options"] == {"max_tool_rounds": 12}
    assert options["skill_parameters"] == {"drawio-generation": {"style": "layered"}}


def test_smoke_matrix_continuation_options_reuse_thread_outputs_without_workflow() -> None:
    template = {
        "workflow": "artifact_workflow",
        "selected_skills": ["data-auto-annotation"],
        "selected_mcp_tools": ["artifact_read"],
    }

    assert should_check_continuation(template) is False

    options = continuation_runtime_options(template, "thread-1", 12)

    assert options["thread_id"] == "thread-1"
    assert "workflow" not in options
    assert options["mode"] == "autonomous"
    assert options["selected_skills"] == ["data-auto-annotation"]
    assert options["selected_mcp_tools"] == [
        "artifact_read",
        "artifact_list",
        "artifact_write",
        "present_files",
        "local_read_file",
    ]


def test_smoke_matrix_continuation_specs_cover_image_and_markdown_apps() -> None:
    image_spec = continuation_spec({"selected_skills": ["data-auto-annotation"]})
    markdown_spec = continuation_spec({"selected_skills": ["markdown-rendering"]})

    assert should_check_continuation({"selected_skills": ["data-auto-annotation"]}) is False
    assert image_spec is None
    assert should_check_continuation({"selected_skills": ["markdown-rendering"]}) is False
    assert markdown_spec is None
    assert should_check_continuation({"selected_skills": ["pptx-generation"]}) is False


def test_smoke_matrix_renders_human_readable_markdown_report() -> None:
    report = render_markdown_report(
        {"total": 1, "ok": 1, "failed": []},
        [
            {
                "name": "data-auto-annotation",
                "agent_name": "default",
                "ok": True,
                "artifact_count": 2,
                "missing_expected_artifact_patterns": [],
                "continuation_check": {"ok": True},
                "thread_id": "thread-1",
            }
        ],
        {"base_url": "http://runtime", "auth": "none"},
    )

    assert "# JetLinks App Smoke Matrix" in report
    assert "- Total templates: 1" in report
    assert "- base_url: `http://runtime`" in report
    assert "| data-auto-annotation | default | yes | 2 | - | ok | `thread-1` |" in report


def test_smoke_matrix_report_explains_retry_instability_failure() -> None:
    report = render_markdown_report(
        {"total": 1, "ok": 0, "failed": [{"name": "artifact-suite"}]},
        [
            {
                "name": "artifact-suite",
                "ok": False,
                "error": "Template passed only after retrying a transient failure.",
                "retry_instability": True,
                "previous_attempts": [
                    {
                        "attempt": 1,
                        "thread_id": "thread-timeout",
                        "error": "TimeoutError('timed out')",
                        "duration_ms": 813242.7,
                    }
                ],
                "checks": {"no_retry_required": False},
            }
        ],
        {"fail_on_retry": True},
    )

    assert "- fail_on_retry: `True`" in report
    assert "Retry instability" in report
    assert "TimeoutError('timed out')" in report
    assert "no_retry_required: failed" in report


def test_smoke_matrix_validates_known_artifact_content_formats() -> None:
    assert (
        validate_artifact_content(
            "annotations.coco.json",
            b'{"images":[],"annotations":[],"categories":[]}',
        )
        is None
    )
    assert validate_artifact_content("preview.png", b"\x89PNG\r\n\x1a\npayload") is None
    assert validate_artifact_content("result.md", "# Result\n".encode()) is None
    assert validate_artifact_content("architecture.drawio", b"<mxfile><diagram /></mxfile>") is None

    assert validate_artifact_content("annotations.coco.json", b'{"images":[]}') == (
        "annotations.coco.json: COCO `annotations` must be a list"
    )
    assert validate_artifact_content("preview.png", b"not-png") == "preview.png: invalid PNG signature"


def test_smoke_matrix_retries_only_transient_failures() -> None:
    assert is_retryable_failure({"error": "TimeoutError('timed out')"}) is True
    assert is_retryable_failure({"error": "ConnectionRefusedError('down')"}) is True
    assert is_retryable_failure({"artifact_content_errors": ["remote-gpu-ops.md: unable to read artifact (TimeoutError('timed out'))"]}) is True
    assert is_retryable_failure({"artifact_content_errors": ["invalid JSON"]}) is False
    assert is_retryable_failure({"error": ""}) is False


def test_smoke_matrix_generates_default_image_when_missing(tmp_path) -> None:
    image_path = tmp_path / "smoke.jpg"

    ensured = ensure_smoke_image(image_path, explicit=False)

    assert ensured == image_path
    assert image_path.read_bytes().startswith(b"\xff\xd8")


def test_smoke_matrix_requires_explicit_image_to_exist(tmp_path) -> None:
    image_path = tmp_path / "missing.jpg"

    try:
        ensure_smoke_image(image_path, explicit=True)
    except FileNotFoundError as exc:
        assert str(image_path) in str(exc)
    else:  # pragma: no cover - defensive assertion path
        raise AssertionError("explicit missing image should fail")


def test_smoke_matrix_parse_args_defaults_require_at_least_one_template(monkeypatch) -> None:
    from tools import app_smoke_matrix

    monkeypatch.setattr("sys.argv", ["app_smoke_matrix.py"])

    args = app_smoke_matrix.parse_args()

    assert args.min_templates == 1
    assert args.fail_on_retry is False


def test_smoke_matrix_parse_args_supports_fail_on_retry(monkeypatch) -> None:
    from tools import app_smoke_matrix

    monkeypatch.setattr("sys.argv", ["app_smoke_matrix.py", "--fail-on-retry"])

    args = app_smoke_matrix.parse_args()

    assert args.fail_on_retry is True


def test_smoke_matrix_builds_bearer_auth_headers() -> None:
    assert auth_headers("") == {}
    assert auth_headers("secret") == {"Authorization": "Bearer secret"}


def test_smoke_matrix_http_helpers_send_bearer_token(monkeypatch) -> None:
    from tools import app_smoke_matrix

    seen = []

    def fake_urlopen(request, timeout):
        seen.append(
            {
                "url": request.full_url,
                "headers": dict(request.header_items()),
                "data": request.data,
                "timeout": timeout,
            }
        )
        return _FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(app_smoke_matrix.urllib.request, "urlopen", fake_urlopen)

    assert get_json("http://runtime", "/api/apps/templates", 9, "secret") == {"ok": True}
    assert post_json("http://runtime", "/api/agents/default/runs", {"hello": "world"}, 7, "secret") == {"ok": True}

    assert seen[0]["url"] == "http://runtime/api/apps/templates"
    assert seen[0]["headers"]["Authorization"] == "Bearer secret"
    assert seen[0]["timeout"] == 9
    assert seen[1]["headers"]["Authorization"] == "Bearer secret"
    assert seen[1]["headers"]["Content-type"] == "application/json"
    assert json.loads(seen[1]["data"].decode("utf-8")) == {"hello": "world"}


def test_smoke_matrix_uses_template_agent_name_for_runs(monkeypatch, tmp_path) -> None:
    from tools import app_smoke_matrix

    seen: list[dict[str, object]] = []

    def fake_post_json(base_url, path, payload, timeout, token=""):
        seen.append({"kind": "post", "base_url": base_url, "path": path, "payload": payload, "timeout": timeout, "token": token})
        return {"status": "completed", "reply": "ok", "metadata": {}}

    def fake_get_json(base_url, path, timeout, token=""):
        seen.append({"kind": "get", "base_url": base_url, "path": path, "timeout": timeout, "token": token})
        return {"artifacts": []}

    monkeypatch.setattr(app_smoke_matrix, "post_json", fake_post_json)
    monkeypatch.setattr(app_smoke_matrix, "get_json", fake_get_json)

    entry = smoke_template(
        base_url="http://runtime",
        template={
            "name": "custom-agent-app",
            "agent_name": "custom-agent",
            "prompt_examples": ["hello"],
            "selected_skills": [],
            "selected_mcp_tools": [],
        },
        image_path=tmp_path / "image.jpg",
        timeout=9,
        max_tool_rounds=12,
        check_continuation=False,
        token="secret",
    )

    assert entry["agent_name"] == "custom-agent"
    assert seen[0]["path"] == "/api/agents/custom-agent/runs"


def test_smoke_matrix_fails_when_uploaded_input_is_still_required(monkeypatch, tmp_path) -> None:
    from tools import app_smoke_matrix

    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    def fake_upload_image(base_url, thread_id, path, timeout, token=""):
        return {"name": "image.jpg", "path": "/mnt/user-data/uploads/image.jpg", "mime_type": "image/jpeg", "size": 4}

    def fake_post_json(base_url, path, payload, timeout, token=""):
        return {"status": "completed", "reply": "请上传图片后继续。", "metadata": {"requires_input": True}}

    def fake_get_json(base_url, path, timeout, token=""):
        return {"artifacts": []}

    monkeypatch.setattr(app_smoke_matrix, "upload_image", fake_upload_image)
    monkeypatch.setattr(app_smoke_matrix, "post_json", fake_post_json)
    monkeypatch.setattr(app_smoke_matrix, "get_json", fake_get_json)

    entry = smoke_template(
        base_url="http://runtime",
        template={
            "name": "data-auto-annotation",
            "agent_name": "default",
            "selected_skills": ["data-auto-annotation"],
            "prompt_examples": ["标注图片"],
        },
        image_path=image_path,
        timeout=9,
        max_tool_rounds=12,
        check_continuation=False,
        token="",
    )

    assert entry["attached_smoke_image"] is True
    assert entry["requires_input"] is True
    assert entry["ok"] is True
    assert entry["checks"]["input_requirement_satisfied"] is True


def test_smoke_matrix_compares_remote_apps_to_config_apps(tmp_path) -> None:
    apps_dir = tmp_path / "config" / "apps"
    apps_dir.mkdir(parents=True)
    (apps_dir / "one.json").write_text('{"name":"one"}', encoding="utf-8")
    (apps_dir / "two.json").write_text('{"name":"two"}', encoding="utf-8")
    (apps_dir / "templates.json").write_text('{"templates":[{"name":"legacy"}]}', encoding="utf-8")

    expected = config_app_names(tmp_path)

    assert expected == ["one", "two"]
    assert missing_expected_apps([{"name": "one"}], expected) == ["two"]
    assert missing_expected_apps([{"name": "one"}, {"name": "two"}, {"name": "extra"}], expected) == []


def test_smoke_matrix_writes_json_and_markdown_outputs(tmp_path) -> None:
    output_path = tmp_path / "smoke.json"
    report_path = tmp_path / "smoke.md"
    summary = {"total": 0, "ok": 0, "failed": [{"name": "preflight", "ok": False, "error": "missing"}]}
    results = [{"name": "preflight", "ok": False, "error": "missing"}]

    write_smoke_outputs(output_path, str(report_path), summary, results, {"auth": "none"})

    json_text = output_path.read_text(encoding="utf-8")
    assert '"summary"' in json_text
    assert '"metadata"' in json_text
    report_text = report_path.read_text(encoding="utf-8")
    assert "preflight" in report_text
    assert "- auth: `none`" in report_text
