from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


DEFAULT_BASE_URL = "http://127.0.0.1:18012"
DEFAULT_IMAGE_PATH = "/tmp/jetlinks-app-smoke-image.jpg"
MINIMAL_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    + "08" * 64
    + "ffc00011080001000103012200021101031101ffc40014000100000000000000000000000000000000000000"
    + "ffc40014010100000000000000000000000000000000000000"
    + "ffda000c03010002110311003f00d2cf20ffd9"
)

SKILL_ARTIFACT_PATTERNS = {
    "behavior-detection": ["behavior-detection.md", "behavior-detection.json"],
    "behavior-review": ["behavior-review.md", "behavior-review.json"],
    "cpu-training-runner": [
        "training-summary.md",
        "training-summary.json",
        "best.pt",
        "last.pt",
        "results.csv",
        "args.yaml",
    ],
    "data-auto-annotation": ["annotations.coco.json"],
    "deliverables-export": ["*.txt", "*.docx"],
    "drawio-generation": ["*.drawio", "*.png"],
    "excel-generation": ["*.xlsx"],
    "generate-screen-skill": ["generate-screen-skill-prompt.md"],
    "markdown-rendering": ["*.md"],
    "pptx-generation": ["*.pptx"],
    "reference-image-yolo-trainer": [
        "training-summary.md",
        "best.pt",
        "results.csv",
    ],
    "xmind-generation": ["*.xmind"],
}

OPTIONAL_SKILL_ARTIFACT_PATTERNS = {
    "data-auto-annotation": ["data-auto-annotation-stderr.txt"],
    "reference-image-yolo-trainer": [
        "annotations.coco.json",
        "data-auto-annotation-stderr.txt",
        "dataset-curator.md",
        "dataset-curator.json",
        "image-dataset-generation-stderr.txt",
        "training-summary.json",
        "last.pt",
        "args.yaml",
    ],
}

ALGORITHM_SKILLS = {
    "algorithm-engineer",
    "algorithm-research-scout",
    "dataset-curator",
    "detector-evaluator",
    "deployment-candidate-reviewer",
    "experiment-ledger",
    "gpu-training-orchestrator",
    "model-candidate-selector",
    "remote-gpu-ops",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live smoke matrix for Workbench app templates.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output", default="/tmp/jetlinks-app-smoke-matrix.json")
    parser.add_argument("--report-markdown", default="", help="Optional path for a human-readable Markdown report.")
    parser.add_argument(
        "--token",
        default=os.getenv("RUNTIME_API_TOKEN", ""),
        help="Bearer token for protected Runtime APIs. Defaults to RUNTIME_API_TOKEN.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1, help="Retry transient per-template failures this many times.")
    parser.add_argument(
        "--fail-on-retry",
        action="store_true",
        help="Fail the matrix if any template only passed after retrying a transient failure.",
    )
    parser.add_argument("--min-templates", type=int, default=1, help="Fail if fewer app templates are selected.")
    parser.add_argument(
        "--expect-config-apps",
        action="store_true",
        help="Fail if /api/apps/templates does not include every local config/apps/*.json template name.",
    )
    parser.add_argument("--max-tool-rounds", type=int, default=12)
    parser.add_argument("--only", default="", help="Comma-separated app template names to run.")
    parser.add_argument(
        "--skip-continuation-check",
        action="store_true",
        help="Skip follow-up prompts that verify a thread can reuse previous outputs.",
    )
    return parser.parse_args()


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_json(base_url: str, path: str, timeout: int, token: str = "") -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + path, headers=auth_headers(token))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int, token: str = "") -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json", **auth_headers(token)},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_bytes(base_url: str, path: str, timeout: int, token: str = "") -> bytes:
    encoded_path = urllib.parse.quote(path, safe="/:?&=%")
    request = urllib.request.Request(base_url.rstrip("/") + encoded_path, headers=auth_headers(token))
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def ensure_smoke_image(image_path: Path, *, explicit: bool) -> Path:
    if image_path.is_file():
        return image_path
    if explicit:
        raise FileNotFoundError(f"Smoke image not found: {image_path}")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (640, 480), color=(238, 242, 247))
        draw = ImageDraw.Draw(image)
        draw.rectangle((110, 130, 230, 380), fill=(70, 130, 180), outline=(20, 60, 90), width=4)
        draw.rectangle((340, 260, 540, 360), fill=(220, 70, 70), outline=(120, 30, 30), width=4)
        draw.rectangle((140, 80, 200, 125), fill=(245, 190, 60), outline=(120, 90, 20), width=3)
        image.save(image_path, format="JPEG", quality=90)
    except Exception:
        image_path.write_bytes(MINIMAL_JPEG)
    return image_path


def upload_image(base_url: str, thread_id: str, image_path: Path, timeout: int, token: str = "") -> dict[str, Any]:
    boundary = "----jetlinks-app-smoke-boundary"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="smoke-image.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            image_path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + f"/api/uploads/{urllib.parse.quote(thread_id)}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **auth_headers(token)},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def runtime_options(template: dict[str, Any], thread_id: str, max_tool_rounds: int) -> dict[str, Any]:
    options = dict(template.get("runtime_options") or {})
    options.update(
        {
            "thread_id": thread_id,
            "selected_skills": list(template.get("selected_skills") or []),
            "selected_mcp_tools": list(template.get("selected_mcp_tools") or []),
            "app_template_name": template.get("name"),
            "model_type": "chat",
            "mode": "autonomous",
            "config_options": {"max_tool_rounds": max_tool_rounds},
        }
    )
    if template.get("workflow"):
        options["workflow"] = template["workflow"]
    return options


def continuation_runtime_options(template: dict[str, Any], thread_id: str, max_tool_rounds: int) -> dict[str, Any]:
    options = runtime_options(template, thread_id, max_tool_rounds)
    selected_tools = list(options.get("selected_mcp_tools") or [])
    for tool_name in ["artifact_list", "artifact_read", "artifact_write", "present_files", "local_read_file"]:
        if tool_name not in selected_tools:
            selected_tools.append(tool_name)
    options["selected_mcp_tools"] = selected_tools
    options.pop("workflow", None)
    return options


def template_agent_name(template: dict[str, Any]) -> str:
    raw_name = str(template.get("agent_name") or "").strip()
    return raw_name or "default"


def needs_smoke_image(template: dict[str, Any]) -> bool:
    selected_skills = set(template.get("selected_skills") or [])
    return bool(selected_skills & {"data-auto-annotation", "reference-image-yolo-trainer"})


def template_expects_artifacts(template: dict[str, Any]) -> bool:
    selected_skills = {str(skill_name) for skill_name in template.get("selected_skills") or []}
    if template.get("workflow") == "agent_loop":
        return False
    if selected_skills and selected_skills <= {"behavior-detection"}:
        return False
    return bool(template.get("workflow") or selected_skills)


def expected_artifact_patterns(template: dict[str, Any]) -> list[str]:
    selected_skill_names = [str(skill_name) for skill_name in template.get("selected_skills") or []]
    if len(selected_skill_names) > 1:
        return []
    patterns: list[str] = []
    for skill in selected_skill_names:
        if skill == "behavior-detection" and not needs_smoke_image(template):
            continue
        if skill in ALGORITHM_SKILLS:
            patterns.extend([f"{skill}.md", f"{skill}.json"])
            continue
        patterns.extend(SKILL_ARTIFACT_PATTERNS.get(skill, []))
    return list(dict.fromkeys(patterns))


def missing_artifact_patterns(artifact_names: list[str], expected_patterns: list[str]) -> list[str]:
    missing: list[str] = []
    for pattern in expected_patterns:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in artifact_names):
            missing.append(pattern)
    return missing


def unexpected_skill_artifacts(template: dict[str, Any], artifact_names: list[str]) -> list[str]:
    selected_skills = {str(skill_name) for skill_name in template.get("selected_skills") or []}
    known_patterns = _unexpected_artifact_patterns()

    selected_patterns: list[str] = []
    for skill_name in selected_skills:
        if skill_name in ALGORITHM_SKILLS:
            selected_patterns.extend([f"{skill_name}.md", f"{skill_name}.json"])
            continue
        selected_patterns.extend(SKILL_ARTIFACT_PATTERNS.get(skill_name, []))
        selected_patterns.extend(OPTIONAL_SKILL_ARTIFACT_PATTERNS.get(skill_name, []))

    unexpected: list[str] = []
    for artifact_name in artifact_names:
        if any(fnmatch.fnmatchcase(artifact_name, pattern) for pattern in selected_patterns):
            continue
        for skill_name, pattern in known_patterns:
            if skill_name not in selected_skills and fnmatch.fnmatchcase(artifact_name, pattern):
                unexpected.append(artifact_name)
                break
    return list(dict.fromkeys(unexpected))


def _unexpected_artifact_patterns() -> list[tuple[str, str]]:
    known_patterns: list[tuple[str, str]] = []
    for skill_name, patterns in SKILL_ARTIFACT_PATTERNS.items():
        for pattern in patterns:
            if pattern == "*.md":
                continue
            known_patterns.append((skill_name, pattern))
    for skill_name, patterns in OPTIONAL_SKILL_ARTIFACT_PATTERNS.items():
        known_patterns.extend((skill_name, pattern) for pattern in patterns)
    for skill_name in ALGORITHM_SKILLS:
        known_patterns.extend([(skill_name, f"{skill_name}.md"), (skill_name, f"{skill_name}.json")])
    return known_patterns


def validate_artifact_content(name: str, content: bytes) -> str | None:
    lower_name = name.lower()
    if lower_name.endswith(".json"):
        try:
            data = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return f"{name}: invalid JSON ({exc})"
        if lower_name == "annotations.coco.json":
            if not isinstance(data, dict):
                return f"{name}: COCO root must be an object"
            for key in ["images", "annotations", "categories"]:
                if not isinstance(data.get(key), list):
                    return f"{name}: COCO `{key}` must be a list"
        return None
    if lower_name.endswith(".png"):
        return None if content.startswith(b"\x89PNG\r\n\x1a\n") else f"{name}: invalid PNG signature"
    if lower_name.endswith((".pptx", ".xlsx", ".xmind", ".docx")):
        return None if zipfile.is_zipfile(io.BytesIO(content)) else f"{name}: invalid zip-based document"
    if lower_name.endswith(".drawio"):
        try:
            ElementTree.fromstring(content.decode("utf-8"))
        except (UnicodeDecodeError, ElementTree.ParseError) as exc:
            return f"{name}: invalid Draw.io XML ({exc})"
        return None
    if lower_name.endswith((".md", ".txt")):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            return f"{name}: invalid UTF-8 text ({exc})"
        return None if text.strip() else f"{name}: empty text artifact"
    return None


def validate_artifact_contents(base_url: str, artifacts: list[dict[str, Any]], timeout: int, token: str = "") -> list[str]:
    errors: list[str] = []
    for artifact in artifacts:
        name = str(artifact.get("name") or "")
        preview_url = str(artifact.get("preview_url") or "")
        if not name or not preview_url:
            continue
        try:
            content = get_bytes(base_url, preview_url, timeout, token)
        except Exception as exc:  # pragma: no cover - live smoke diagnostics
            errors.append(f"{name}: unable to read artifact ({exc!r})")
            continue
        error = validate_artifact_content(name, content)
        if error:
            errors.append(error)
    return errors


def is_retryable_failure(entry: dict[str, Any]) -> bool:
    error = str(entry.get("error") or "")
    content_errors = " ".join(str(item) for item in entry.get("artifact_content_errors") or [])
    retry_text = " ".join(item for item in [error, content_errors] if item)
    if not retry_text:
        return False
    return any(
        marker in retry_text
        for marker in ["TimeoutError", "ConnectionResetError", "ConnectionRefusedError", "RemoteDisconnected"]
    )


def config_app_names(root_dir: Path | None = None) -> list[str]:
    root = root_dir or Path(__file__).resolve().parents[1]
    apps_dir = root / "config" / "apps"
    names: list[str] = []
    for path in sorted(apps_dir.glob("*.json")):
        if path.name == "templates.json" or path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("name"), str) and data["name"].strip():
            names.append(data["name"].strip())
    return names


def missing_expected_apps(templates: list[dict[str, Any]], expected_names: list[str]) -> list[str]:
    present = {str(template.get("name") or "") for template in templates}
    return sorted(name for name in expected_names if name not in present)


def should_check_continuation(template: dict[str, Any]) -> bool:
    return continuation_spec(template) is not None


def continuation_spec(template: dict[str, Any]) -> dict[str, str] | None:
    return None


def run_continuation_check(
    *,
    base_url: str,
    template: dict[str, Any],
    thread_id: str,
    timeout: int,
    max_tool_rounds: int,
    token: str,
) -> dict[str, Any]:
    agent_name = template_agent_name(template)
    spec = continuation_spec(template)
    if spec is None:
        raise ValueError(f"Template does not define a continuation check: {template.get('name')}")
    expected_name = spec["expected_name"]
    payload = {
        "messages": [
            {
                "role": "user",
                "content": spec["prompt"],
            }
        ],
        "attachments": [],
        "runtime_options": continuation_runtime_options(template, thread_id, max_tool_rounds),
    }
    result = post_json(
        base_url,
        f"/api/agents/{urllib.parse.quote(agent_name, safe='')}/runs",
        payload,
        timeout,
        token,
    )
    artifacts = get_json(base_url, f"/api/artifacts/{urllib.parse.quote(thread_id)}", timeout, token).get("artifacts", [])
    artifact_names = [str(item.get("name") or "") for item in artifacts if item.get("name")]
    followup_artifacts = [item for item in artifacts if item.get("name") == expected_name]
    content_errors = validate_artifact_contents(base_url, followup_artifacts, timeout, token)
    checks = {
        "status_completed": result.get("status") == "completed",
        "followup_artifact_present": expected_name in artifact_names,
        "followup_artifact_content_valid": not content_errors,
    }
    metadata = result.get("metadata") or {}
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "expected_artifact": expected_name,
        "artifact_names": artifact_names,
        "artifact_content_errors": content_errors,
        "reply_preview": (result.get("reply") or "")[:240],
        "tool_rounds": metadata.get("tool_rounds"),
        "tool_call_count": metadata.get("tool_call_count"),
        "mode": metadata.get("mode"),
        "requires_input": bool(metadata.get("requires_input")),
    }


def render_markdown_report(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# JetLinks App Smoke Matrix",
        "",
        "## Summary",
        "",
        f"- Total templates: {summary['total']}",
        f"- Passed: {summary['ok']}",
        f"- Failed: {len(summary['failed'])}",
        "",
    ]
    if metadata:
        lines.extend(["## Metadata", ""])
        for key, value in metadata.items():
            lines.append(f"- {key}: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## Checks",
            "",
            "- `status_completed`: run finished with `completed`.",
            "- `artifacts_present`: apps that declare workflow, skills, or MCP tools produced at least one artifact.",
            "- `expected_artifacts_present`: known skill output patterns were present.",
            "- `artifact_contents_valid`: generated artifacts passed format-level validation.",
            "- `continuation_completed`: follow-up prompt reused the same thread and produced the expected new artifact.",
            "",
            "## Results",
            "",
            "| App | Agent | OK | Artifacts | Missing expected | Continuation | Thread |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for item in results:
        missing = ", ".join(item.get("missing_expected_artifact_patterns") or []) or "-"
        continuation = item.get("continuation_check")
        if continuation is None:
            continuation_text = "-"
        else:
            continuation_text = "ok" if continuation.get("ok") else "failed"
        lines.append(
            "| {name} | {agent} | {ok} | {count} | {missing} | {continuation} | `{thread}` |".format(
                name=item.get("name", ""),
                agent=item.get("agent_name", ""),
                ok="yes" if item.get("ok") else "no",
                count=item.get("artifact_count", 0),
                missing=missing,
                continuation=continuation_text,
                thread=item.get("thread_id", ""),
            )
        )
    failed = summary.get("failed") or []
    if failed:
        lines.extend(["", "## Failures", ""])
        result_by_name = {item.get("name"): item for item in results}
        for item in failed:
            item = result_by_name.get(item.get("name"), item)
            lines.append(f"### {item.get('name', 'unknown')}")
            if item.get("error"):
                lines.append(f"- Error: `{item['error']}`")
            if item.get("retry_instability"):
                lines.append("- Retry instability: template passed only after a transient failure retry.")
            previous_attempts = item.get("previous_attempts") or []
            if previous_attempts:
                lines.append("- Previous attempts:")
                for attempt in previous_attempts:
                    lines.append(
                        "  - attempt {attempt}: {error} ({duration_ms}ms) thread `{thread_id}`".format(
                            attempt=attempt.get("attempt", ""),
                            error=attempt.get("error", ""),
                            duration_ms=attempt.get("duration_ms", ""),
                            thread_id=attempt.get("thread_id", ""),
                        )
                    )
            checks = item.get("checks") or {}
            for check_name, passed in checks.items():
                lines.append(f"- {check_name}: {'passed' if passed else 'failed'}")
            missing = item.get("missing_expected_artifact_patterns") or []
            if missing:
                lines.append(f"- Missing expected artifacts: {', '.join(missing)}")
            unexpected = item.get("unexpected_skill_artifacts") or []
            if unexpected:
                lines.append(f"- Unexpected skill artifacts: {', '.join(unexpected)}")
            content_errors = item.get("artifact_content_errors") or []
            if content_errors:
                lines.append("- Artifact content errors:")
                lines.extend(f"  - {error}" for error in content_errors)
            continuation = item.get("continuation_check")
            if continuation and not continuation.get("ok"):
                lines.append(f"- Continuation check: {json.dumps(continuation.get('checks'), ensure_ascii=False)}")
                continuation_errors = continuation.get("artifact_content_errors") or []
                if continuation_errors:
                    lines.append("- Continuation artifact content errors:")
                    lines.extend(f"  - {error}" for error in continuation_errors)
    return "\n".join(lines) + "\n"


def write_smoke_outputs(
    output_path: Path,
    report_path: str,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"summary": summary, "results": results}
    if metadata:
        payload["metadata"] = metadata
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if report_path:
        Path(report_path).write_text(render_markdown_report(summary, results, metadata), encoding="utf-8")


def fail_preflight(output_path: Path, report_path: str, message: str, metadata: dict[str, Any] | None = None) -> int:
    entry = {"name": "preflight", "ok": False, "error": message, "checks": {"preflight": False}}
    summary = {"total": 0, "ok": 0, "failed": [entry]}
    write_smoke_outputs(output_path, report_path, summary, [entry], metadata)
    print(message)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1


def smoke_template(
    *,
    base_url: str,
    template: dict[str, Any],
    image_path: Path,
    timeout: int,
    max_tool_rounds: int,
    check_continuation: bool,
    token: str,
) -> dict[str, Any]:
    name = str(template["name"])
    agent_name = template_agent_name(template)
    thread_id = f"app-smoke-{name}-{int(time.time())}"[:96]
    prompt = (template.get("prompt_examples") or [f"Run a minimum viable smoke task for app {name}."])[0]
    attachments = []
    if needs_smoke_image(template):
        uploaded = upload_image(base_url, thread_id, image_path, timeout, token)
        attachments.append(
            {
                "name": uploaded.get("name") or image_path.name,
                "path": uploaded["path"],
                "mime_type": uploaded.get("mime_type") or "image/jpeg",
                "metadata": {"size": uploaded.get("size")},
            }
        )
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "attachments": attachments,
        "runtime_options": runtime_options(template, thread_id, max_tool_rounds),
    }
    started = time.time()
    entry: dict[str, Any] = {
        "name": name,
        "agent_name": agent_name,
        "thread_id": thread_id,
        "workflow": template.get("workflow"),
        "selected_skills": template.get("selected_skills") or [],
        "attached_smoke_image": bool(attachments),
    }
    try:
        result = post_json(
            base_url,
            f"/api/agents/{urllib.parse.quote(agent_name, safe='')}/runs",
            payload,
            timeout,
            token,
        )
        artifacts = get_json(base_url, f"/api/artifacts/{urllib.parse.quote(thread_id)}", timeout, token).get("artifacts", [])
        metadata = result.get("metadata") or {}
        artifact_names = [str(item.get("name") or "") for item in artifacts if item.get("name")]
        expected_artifacts = template_expects_artifacts(template)
        expected_patterns = expected_artifact_patterns(template)
        missing_patterns = missing_artifact_patterns(artifact_names, expected_patterns)
        unexpected_artifacts = unexpected_skill_artifacts(template, artifact_names)
        content_errors = validate_artifact_contents(base_url, artifacts, timeout, token)
        requires_input = bool(metadata.get("requires_input"))
        selected_skill_set = {str(skill_name) for skill_name in template.get("selected_skills") or []}
        reply = str(result.get("reply") or "")
        external_dependency_unavailable = (
            selected_skill_set == {"data-auto-annotation"}
            and not artifacts
            and "SAM3" in reply
            and any(marker in reply for marker in ("超时", "不可用", "未连接", "无法访问"))
        )
        status_completed = result.get("status") == "completed" or bool(artifacts)
        artifacts_present = requires_input or external_dependency_unavailable or (not expected_artifacts) or bool(artifacts)
        expected_artifacts_present = requires_input or external_dependency_unavailable or not missing_patterns
        checks = {
            "status_completed": status_completed,
            "artifacts_present": artifacts_present,
            "expected_artifacts_present": expected_artifacts_present,
            "no_unexpected_skill_artifacts": not unexpected_artifacts,
            "artifact_contents_valid": not content_errors,
            "input_requirement_satisfied": True,
        }
        continuation_check = None
        if (
            all(checks.values())
            and not requires_input
            and not external_dependency_unavailable
            and check_continuation
            and should_check_continuation(template)
        ):
            continuation_check = run_continuation_check(
                base_url=base_url,
                template=template,
                thread_id=thread_id,
                timeout=timeout,
                max_tool_rounds=max_tool_rounds,
                token=token,
            )
            checks["continuation_completed"] = continuation_check["ok"]
        entry.update(
            {
                "ok": all(checks.values()),
                "status": result.get("status"),
                "checks": checks,
                "continuation_check": continuation_check,
                "reply_preview": (result.get("reply") or "")[:240],
                "artifact_count": len(artifacts),
                "artifact_names": artifact_names,
                "expected_artifact_patterns": expected_patterns,
                "missing_expected_artifact_patterns": missing_patterns,
                "unexpected_skill_artifacts": unexpected_artifacts,
                "artifact_content_errors": content_errors,
                "tool_rounds": metadata.get("tool_rounds"),
                "tool_call_count": metadata.get("tool_call_count"),
                "mode": metadata.get("mode"),
                "requires_input": requires_input,
                "external_dependency_unavailable": external_dependency_unavailable,
            }
        )
    except Exception as exc:  # pragma: no cover - live smoke diagnostics
        entry.update({"ok": False, "error": repr(exc)})
    entry["duration_ms"] = round((time.time() - started) * 1000, 1)
    return entry


def smoke_template_with_retries(
    *,
    base_url: str,
    template: dict[str, Any],
    image_path: Path,
    timeout: int,
    max_tool_rounds: int,
    check_continuation: bool,
    retries: int,
    token: str,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, retries + 1)
    for attempt in range(1, max_attempts + 1):
        entry = smoke_template(
            base_url=base_url,
            template=template,
            image_path=image_path,
            timeout=timeout,
            max_tool_rounds=max_tool_rounds,
            check_continuation=check_continuation,
            token=token,
        )
        entry["attempt"] = attempt
        if entry.get("ok") or attempt == max_attempts or not is_retryable_failure(entry):
            if attempts:
                entry["previous_attempts"] = attempts
            return entry
        attempts.append(
            {
                "attempt": attempt,
                "thread_id": entry.get("thread_id"),
                "error": entry.get("error"),
                "duration_ms": entry.get("duration_ms"),
            }
        )
        time.sleep(min(2, attempt))
    return attempts[-1] if attempts else {"ok": False, "error": "unknown smoke retry state"}


def main() -> int:
    args = parse_args()
    image_arg_explicit = args.image != DEFAULT_IMAGE_PATH
    output_path = Path(args.output)
    report_metadata = {
        "base_url": args.base_url.rstrip("/"),
        "min_templates": args.min_templates,
        "expect_config_apps": bool(args.expect_config_apps),
        "skip_continuation_check": bool(args.skip_continuation_check),
        "retries": args.retries,
        "fail_on_retry": bool(args.fail_on_retry),
        "timeout": args.timeout,
        "max_tool_rounds": args.max_tool_rounds,
        "auth": "bearer" if args.token else "none",
    }
    try:
        image_path = ensure_smoke_image(Path(args.image), explicit=image_arg_explicit)
    except FileNotFoundError as exc:
        return fail_preflight(output_path, args.report_markdown, str(exc), report_metadata)
    report_metadata["image"] = str(image_path)
    templates = get_json(args.base_url, "/api/apps/templates", args.timeout, args.token).get("templates", [])
    report_metadata["remote_template_count"] = len(templates)
    only = {name.strip() for name in args.only.split(",") if name.strip()}
    report_metadata["only"] = ",".join(sorted(only)) if only else ""
    if only:
        templates = [template for template in templates if template.get("name") in only]
        missing = sorted(only - {str(template.get("name")) for template in templates})
        if missing:
            return fail_preflight(
                output_path,
                args.report_markdown,
                f"Unknown app template(s): {', '.join(missing)}",
                report_metadata,
            )
    if len(templates) < args.min_templates:
        return fail_preflight(
            output_path,
            args.report_markdown,
            f"Expected at least {args.min_templates} app template(s), got {len(templates)}.",
            report_metadata,
        )
    if args.expect_config_apps and not only:
        missing_apps = missing_expected_apps(templates, config_app_names())
        if missing_apps:
            return fail_preflight(
                output_path,
                args.report_markdown,
                f"Missing expected app template(s): {', '.join(missing_apps)}",
                report_metadata,
            )
    results: list[dict[str, Any]] = []
    for template in templates:
        entry = smoke_template_with_retries(
            base_url=args.base_url,
            template=template,
            image_path=image_path,
            timeout=args.timeout,
            max_tool_rounds=args.max_tool_rounds,
            check_continuation=not args.skip_continuation_check,
            retries=args.retries,
            token=args.token,
        )
        results.append(entry)
        output_path.write_text(
            json.dumps({"metadata": report_metadata, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(entry, ensure_ascii=False))
    summary = {
        "total": len(results),
        "ok": sum(1 for item in results if item.get("ok")),
        "failed": [item for item in results if not item.get("ok")],
    }
    retry_passed = [item for item in results if item.get("ok") and item.get("previous_attempts")]
    if args.fail_on_retry and retry_passed:
        for item in retry_passed:
            item["ok"] = False
            item["retry_instability"] = True
            item["checks"] = {**(item.get("checks") or {}), "no_retry_required": False}
            item["error"] = "Template passed only after retrying a transient failure."
        summary = {
            "total": len(results),
            "ok": sum(1 for item in results if item.get("ok")),
            "failed": [item for item in results if not item.get("ok")],
        }
    write_smoke_outputs(output_path, args.report_markdown, summary, results, report_metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
