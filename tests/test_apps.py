from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.apps import AppTemplateRegistry
from app.main import create_app
from tools.app_smoke_matrix import expected_artifact_patterns


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


def test_preconfigured_app_templates_have_no_duplicate_names_or_titles() -> None:
    registry = AppTemplateRegistry()
    templates = registry.list()
    app_files = [path for path in (registry.root_dir / "config" / "apps").glob("*.json") if path.name != "templates.json"]
    names = [template.name for template in templates]
    titles = [template.title for template in templates]

    assert len(templates) == len(app_files)
    assert len(names) == len(set(names))
    assert len(titles) == len(set(titles))


def test_preconfigured_app_templates_have_smoke_artifact_expectations() -> None:
    registry = AppTemplateRegistry()

    missing = [
        template.name
        for template in registry.list()
        if template.selected_skills and not expected_artifact_patterns(template.model_dump())
    ]

    assert missing == []
