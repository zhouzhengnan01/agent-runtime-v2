from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

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


def test_skill_plugin_upload_can_replace_existing_plugin(tmp_path: Path) -> None:
    manager = SkillPluginManager(tmp_path)

    first = manager.install_zip(_plugin_zip())
    second = manager.install_zip(_plugin_zip())

    assert first.plugin_id == "summary-plugin"
    assert second.plugin_id == "summary-plugin"
    assert "uploaded-summary" in manager.load_skills()


def test_builtin_skill_plugin_uses_complete_skill_packages() -> None:
    root = Path(__file__).resolve().parents[1] / "plugins" / "skills" / "builtin-artifact-skills" / "skills"
    package_names = {package.name for package in root.iterdir() if package.is_dir()}
    assert "deliverables-export" in package_names
    for package in root.iterdir():
        if not package.is_dir():
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
        assert (package / "scripts" / "run_skill.py").is_file()


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
    assert Path(files["skill-md"]["path"]).parts[-6:] == ("plugins", "skills", "builtin-artifact-skills", "skills", "deliverables-export", "SKILL.md")

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
  "input_schema": {"type": "object"},
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
                "input_schema": {"type": "object", "properties": {"title": {"type": "string"}}},
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
    assert manifest["input_schema"]["properties"]["title"]["type"] == "string"
    assert manifest["sandbox"]["fallback_to_local"] is True


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
  "input_schema": {"type": "object"},
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
        archive.writestr("input.schema.json", """{"type": "object", "additionalProperties": true}""")
        archive.writestr("output.schema.json", """{"type": "object", "additionalProperties": true}""")
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
