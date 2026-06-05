from __future__ import annotations

import json
from pathlib import Path

from app.core.artifacts import ArtifactStore
from app.core.workflow import WorkflowPluginManager, WorkflowRegistry


def test_workflow_registry_builtin_uses_workflow_plugin_packages(tmp_path: Path) -> None:
    registry = WorkflowRegistry.builtin(ArtifactStore(root_dir=tmp_path / "threads"), root_dir=tmp_path)

    assert {"artifact_workflow", "evidence_first_detection", "yolo_training_flow"} <= registry.names()
    assert type(registry.get("artifact_workflow")).__name__ == "ArtifactWorkflow"
    assert type(registry.get("yolo_training_flow")).__name__ == "YoloTrainingWorkflow"
    assert registry.config("artifact_workflow") is not None
    assert registry.config("artifact_workflow").handler == "artifact.py:ArtifactWorkflow"  # type: ignore[union-attr]


def test_workflow_plugin_manager_imports_valid_uploaded_workflow(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "workflows" / "uploaded-workflow-plugin"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        json.dumps(
            {
                "id": "uploaded-workflow-plugin",
                "name": "Uploaded Workflow Plugin",
                "version": "1.0.0",
                "workflows": ["workflow.json"],
            }
        ),
        encoding="utf-8",
    )
    (plugin_root / "workflow.json").write_text(
        json.dumps(
            {
                "name": "uploaded_workflow",
                "display_name": "Uploaded Workflow",
                "description": "Uploaded workflow description",
                "enabled": True,
                "handler": "workflow.py:UploadedWorkflow",
            }
        ),
        encoding="utf-8",
    )
    (plugin_root / "workflow.py").write_text(
        """
from app.schemas import AgentRunResult


class UploadedWorkflow:
    def __init__(self, artifact_store):
        self.artifact_store = artifact_store

    def run_with_events(self, agent_config, messages, attachments, thread_id, on_event=None, workflow_name=None, runtime_options=None):
        return AgentRunResult(
            agent=agent_config.name,
            thread_id=thread_id or "uploaded",
            reply="uploaded workflow",
            metadata={"workflow": workflow_name},
        ), []
""",
        encoding="utf-8",
    )

    manager = WorkflowPluginManager(root_dir=tmp_path)
    loaded = manager.load_workflows(ArtifactStore(root_dir=tmp_path / "threads"))
    registry = WorkflowRegistry.builtin(ArtifactStore(root_dir=tmp_path / "threads"), root_dir=tmp_path)

    assert "uploaded_workflow" in loaded
    assert type(registry.get("uploaded_workflow")).__name__ == type(loaded["uploaded_workflow"].plugin).__name__
    assert registry.config("uploaded_workflow").handler == "workflow.py:UploadedWorkflow"  # type: ignore[union-attr]


def test_workflow_plugin_manager_filters_invalid_uploaded_workflow(tmp_path: Path) -> None:
    plugin_root = tmp_path / "plugins" / "workflows" / "invalid-workflow-plugin"
    plugin_root.mkdir(parents=True)
    (plugin_root / "plugin.json").write_text(
        json.dumps(
            {
                "id": "invalid-workflow-plugin",
                "name": "Invalid Workflow Plugin",
                "version": "1.0.0",
                "workflows": ["bad-path.json", "bad-class.json"],
            }
        ),
        encoding="utf-8",
    )
    (plugin_root / "bad-path.json").write_text(
        json.dumps(
            {
                "name": "bad_path",
                "display_name": "Bad Path",
                "enabled": True,
                "handler": "../workflow.py:BadWorkflow",
            }
        ),
        encoding="utf-8",
    )
    (plugin_root / "bad-class.json").write_text(
        json.dumps(
            {
                "name": "bad_class",
                "display_name": "Bad Class",
                "enabled": True,
                "handler": "workflow.py:BadWorkflow",
            }
        ),
        encoding="utf-8",
    )
    (plugin_root / "workflow.py").write_text(
        """
class BadWorkflow:
    pass
""",
        encoding="utf-8",
    )

    manager = WorkflowPluginManager(root_dir=tmp_path)

    plugin_ids = {plugin.plugin_id for plugin in manager.list_plugins()}
    loaded = manager.load_workflows(ArtifactStore(root_dir=tmp_path / "threads"))

    assert "invalid-workflow-plugin" in plugin_ids
    assert "bad_path" not in loaded
    assert "bad_class" not in loaded
