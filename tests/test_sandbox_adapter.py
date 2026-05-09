from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.sandbox.config import SandboxConfig
from app.core.sandbox.policy import SandboxDecision, SandboxProfile
from app.core.sandbox.runner import SandboxRunContext, SandboxSkillRunner, _request_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sandbox_request_json_includes_skill_run_schema_version(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path)
    paths = store.prepare_thread("schema-version")
    profile = SandboxProfile(name="drawio")
    decision = SandboxDecision(
        skill_name="drawio-generation",
        eligible=True,
        use_sandbox=True,
        execution_mode="sandbox",
        provider="opensandbox",
        profile_name="drawio",
        profile=profile,
        request_schema_version="skill-run.v1",
        fallback_to_local=True,
        reason="test",
    )
    context = SandboxRunContext(
        skill_name="drawio-generation",
        spec={"title": "JetLinks IoT 架构"},
        paths=paths,
        decision=decision,
        config=SandboxConfig(provider="opensandbox", executor_enabled=True),
    )

    payload = json.loads(_request_json(context, profile))

    assert payload["request_schema_version"] == "skill-run.v1"
    assert payload["skill_name"] == "drawio-generation"
    assert payload["thread_id"] == "schema-version"
    assert payload["outputs_dir"] == "/mnt/user-data/outputs"


def test_unified_sandbox_adapter_generates_drawio_and_png(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    outputs_dir = tmp_path / "outputs"
    request_path.write_text(
        json.dumps(
            {
                "request_schema_version": "skill-run.v1",
                "skill_name": "drawio-generation",
                "thread_id": "adapter-drawio",
                "spec": {
                    "title": "JetLinks IoT 平台架构",
                    "diagram_type": "layered_architecture",
                    "visual_style": "polished",
                    "swimlanes": ["接入层", "规则引擎", "数据存储", "告警运维"],
                    "nodes": [
                        "MQTT/HTTP 接入",
                        "设备网关",
                        "规则引擎",
                        "AI 复判",
                        "时序存储",
                        "告警中心",
                        "运维监控",
                    ],
                    "lane_nodes": {
                        "接入层": ["MQTT/HTTP 接入", "设备网关"],
                        "规则引擎": ["规则引擎", "AI 复判"],
                        "数据存储": ["时序存储"],
                        "告警运维": ["告警中心", "运维监控"],
                    },
                    "edges": [
                        ["MQTT/HTTP 接入", "设备网关"],
                        ["设备网关", "规则引擎"],
                        ["规则引擎", "AI 复判"],
                        ["AI 复判", "告警中心"],
                        ["规则引擎", "时序存储"],
                        ["时序存储", "运维监控"],
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "sandbox" / "skills" / "run_skill.py"),
            "--request",
            str(request_path),
            "--outputs",
            str(outputs_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    drawio_path = outputs_dir / "architecture.drawio"
    png_path = outputs_dir / "architecture.png"

    assert result["ok"] is True
    assert result["skill_name"] == "drawio-generation"
    assert drawio_path.is_file()
    assert png_path.is_file()
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    root = ET.fromstring(drawio_path.read_text(encoding="utf-8"))
    values = {str(cell.get("value")) for cell in root.findall(".//mxCell")}
    assert "规则引擎" in values
    assert "AI 复判" in values
    assert "edgeStyle=orthogonalEdgeStyle" in drawio_path.read_text(encoding="utf-8")


def test_local_subprocess_runner_generates_drawio_outputs(tmp_path: Path) -> None:
    store = ArtifactStore(root_dir=tmp_path / "threads")
    paths = store.prepare_thread("local-subprocess-drawio")
    profile = SandboxProfile(
        name="drawio",
        command=f"python {PROJECT_ROOT / 'sandbox' / 'skills' / 'run_skill.py'} --request {{request_path}} --outputs {{outputs_dir}}",
        timeout_seconds=60,
    )
    decision = SandboxDecision(
        skill_name="drawio-generation",
        eligible=True,
        use_sandbox=True,
        execution_mode="sandbox",
        provider="local_subprocess",
        profile_name="drawio",
        profile=profile,
        request_schema_version="skill-run.v1",
        fallback_to_local=True,
        reason="test",
    )
    context = SandboxRunContext(
        skill_name="drawio-generation",
        spec={
            "title": "本地子进程架构图",
            "nodes": ["设备接入", "规则引擎"],
            "edges": [["设备接入", "规则引擎"]],
        },
        paths=paths,
        decision=decision,
        config=SandboxConfig(provider="local_subprocess", executor_enabled=True),
    )

    result = SandboxSkillRunner(store).run(context)

    assert result.data["execution_mode"] == "local_subprocess"
    assert {artifact.name for artifact in result.outputs} == {"architecture.drawio", "architecture.png"}
    assert (paths.outputs / "architecture.drawio").is_file()
    assert (paths.outputs / "architecture.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
