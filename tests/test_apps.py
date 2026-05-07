from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.apps import AppTemplateRegistry
from app.main import create_app


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
