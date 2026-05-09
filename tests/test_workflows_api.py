from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from app.api import workflows as workflows_api
from app.core.agent import AgentRuntime
from app.core.artifacts import ArtifactStore
from app.core.workflow import WorkflowPluginManager, WorkflowRegistry
from app.main import create_app


def test_list_workflow_plugins() -> None:
    client = TestClient(create_app())

    response = client.get("/api/workflows")

    assert response.status_code == 200
    payload = response.json()
    names = {workflow["name"] for workflow in payload["workflows"]}
    assert {"artifact_workflow", "evidence_first_detection"} <= names
    artifact = next(workflow for workflow in payload["workflows"] if workflow["name"] == "artifact_workflow")
    assert artifact["display_name"] == "Artifact Workflow"
    assert artifact["handler"] == "artifact.py:ArtifactWorkflow"
    assert artifact["description"]
    assert artifact["trigger"] == "explicit_runtime_options"
    assert artifact["request_example"]["runtime_options"]["workflow"] == "artifact_workflow"


def test_workflow_plugin_upload_list_and_delete(tmp_path, monkeypatch) -> None:
    artifact_store = ArtifactStore(root_dir=tmp_path / "threads")
    plugin_manager = WorkflowPluginManager(root_dir=tmp_path)
    runtime = AgentRuntime(
        artifact_store=artifact_store,
        workflow_registry=WorkflowRegistry.builtin(artifact_store, root_dir=tmp_path),
    )
    monkeypatch.setattr(workflows_api, "plugin_manager", plugin_manager)
    monkeypatch.setattr(workflows_api, "runtime", runtime)
    client = TestClient(create_app())

    upload = client.post(
        "/api/workflows/plugins",
        files={"file": ("custom-workflow.zip", _workflow_plugin_zip(), "application/zip")},
    )
    assert upload.status_code == 200
    assert upload.json()["plugin"]["id"] == "custom-workflow-plugin"

    plugins = client.get("/api/workflows/plugins")
    assert plugins.status_code == 200
    plugin_ids = {plugin["id"] for plugin in plugins.json()["plugins"]}
    assert "custom-workflow-plugin" in plugin_ids

    workflows = client.get("/api/workflows")
    assert workflows.status_code == 200
    workflow_names = {workflow["name"] for workflow in workflows.json()["workflows"]}
    assert "custom_workflow" in workflow_names

    delete = client.delete("/api/workflows/plugins/custom-workflow-plugin")
    assert delete.status_code == 200
    assert delete.json()["deleted"] is True

    workflows_after_delete = client.get("/api/workflows")
    workflow_names_after_delete = {workflow["name"] for workflow in workflows_after_delete.json()["workflows"]}
    assert "custom_workflow" not in workflow_names_after_delete


def test_workflow_plugin_delete_protected_builtin_is_forbidden() -> None:
    client = TestClient(create_app())

    response = client.delete("/api/workflows/plugins/builtin-artifact-workflows")

    assert response.status_code == 403
    assert "protected" in response.json()["detail"]


def _workflow_plugin_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "custom-workflow-plugin/plugin.json",
            """
{
  "id": "custom-workflow-plugin",
  "name": "Custom Workflow Plugin",
  "version": "1.0.0",
  "workflows": ["workflows/*/workflow.json"]
}
""",
        )
        archive.writestr(
            "custom-workflow-plugin/workflows/custom_workflow/workflow.json",
            """
{
  "name": "custom_workflow",
  "display_name": "Custom Workflow",
  "description": "Uploaded custom workflow.",
  "enabled": true,
  "handler": "workflow.py:CustomWorkflow"
}
""",
        )
        archive.writestr(
            "custom-workflow-plugin/workflow.py",
            """
from app.schemas import AgentRunResult


class CustomWorkflow:
    def __init__(self, artifact_store):
        self.artifact_store = artifact_store

    def run_with_events(self, agent_config, messages, attachments, thread_id, on_event=None, workflow_name=None, runtime_options=None):
        return AgentRunResult(
            agent=agent_config.name,
            thread_id=thread_id or "custom",
            reply="custom workflow",
            metadata={"workflow": workflow_name},
        ), []
""",
        )
    return buffer.getvalue()
