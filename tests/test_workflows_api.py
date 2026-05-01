from fastapi.testclient import TestClient

from app.main import create_app


def test_list_workflow_plugins() -> None:
    client = TestClient(create_app())

    response = client.get("/api/workflows")

    assert response.status_code == 200
    payload = response.json()
    names = {workflow["name"] for workflow in payload["workflows"]}
    assert {"artifact_workflow", "evidence_first_detection"} <= names
    artifact = next(workflow for workflow in payload["workflows"] if workflow["name"] == "artifact_workflow")
    assert artifact["trigger"] == "explicit_runtime_options"
    assert artifact["request_example"]["runtime_options"]["workflow"] == "artifact_workflow"
