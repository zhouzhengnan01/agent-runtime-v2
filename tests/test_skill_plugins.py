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
