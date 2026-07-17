from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from app.api import training
from app.api import acp_http_stream
from app.api.acp_http_stream import AcpHttpSubscriptionRequest, _stream_acp_subscription
from app.core.artifacts import ArtifactStore
from app.core.http_training_jobs import write_http_training_job_marker
from app.protocols.acp.event_broker import AcpEventBroker, acp_event_broker
from app.schemas import AgentRunResult, ChatEvent


async def _collect_subscription() -> list[str]:
    request = AcpHttpSubscriptionRequest(
        sessionId="postman-yolo-test-002",
        threadId="postman-yolo-test-002",
    )
    chunks: list[str] = []

    async def collect() -> None:
        async for chunk in _stream_acp_subscription(request):
            chunks.append(chunk)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    await acp_event_broker.publish(
        {"postman-yolo-test-002"},
        "session/update",
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "postman-yolo-test-002",
                "update": {
                    "_meta": {"jetlinksRuntimeEvent": {"type": "tool.started", "data": {}}},
                    "sessionUpdate": "tool_call",
                },
            },
        },
    )
    await acp_event_broker.publish(
        {"postman-yolo-test-002"},
        "session/update",
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "postman-yolo-test-002",
                "update": {
                    "_meta": {
                        "jetlinksRuntimeEvent": {
                            "type": "artifact.created",
                            "data": {"artifact": {"name": "best.pt"}},
                        }
                    },
                    "sessionUpdate": "agent_thought_chunk",
                },
            },
        },
    )
    await acp_event_broker.publish(
        {"postman-yolo-test-002"},
        "result",
        {"jsonrpc": "2.0", "id": 7, "result": {"stopReason": "end_turn"}},
    )
    await task
    return chunks


def test_http_subscription_receives_websocket_published_events() -> None:
    chunks = asyncio.run(_collect_subscription())
    payloads = [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ]

    assert chunks[0] == ": connected\n\n"
    assert payloads[0]["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"] == "tool.started"
    assert payloads[1]["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"] == "artifact.created"
    assert payloads[2]["result"]["stopReason"] == "end_turn"


def test_acp_event_broker_shares_websocket_events_across_workers(tmp_path: Path) -> None:
    async def collect() -> dict:
        publisher = AcpEventBroker(shared_dir=tmp_path / "broker", poll_seconds=0.01)
        subscriber = AcpEventBroker(shared_dir=tmp_path / "broker", poll_seconds=0.01)
        subscriber_id, queue = await subscriber.subscribe({"acp-websocket"})
        try:
            await publisher.publish(
                {"acp-websocket"},
                "session/update",
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": "acp-websocket",
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "hello"},
                        },
                    },
                },
                shared=True,
            )
            return (await asyncio.wait_for(queue.get(), timeout=1)).payload
        finally:
            await subscriber.unsubscribe(subscriber_id, {"acp-websocket"})

    payload = asyncio.run(collect())

    assert payload["method"] == "session/update"
    assert payload["params"]["sessionId"] == "acp-websocket"


def test_http_subscription_requires_session_or_thread() -> None:
    async def collect() -> list[str]:
        return [chunk async for chunk in _stream_acp_subscription(AcpHttpSubscriptionRequest())]

    chunks = asyncio.run(collect())

    assert "sessionId or threadId is required" in chunks[0]


def test_http_training_status_bridges_new_artifacts_to_session_update(tmp_path: Path, monkeypatch) -> None:
    thread_id = "http-training-artifact-bridge"
    root = tmp_path / ".runtime" / "threads"
    outputs = root / thread_id / "outputs"
    run_dir = outputs / "yolo_training_flow" / "runs" / "run-001"
    pipeline = run_dir / "pipeline_work"
    synthetic_images = pipeline / "synthetic_images"
    deim_weights = run_dir / "deimv2_training_run" / "runs" / "train"
    for path in (pipeline, synthetic_images, deim_weights):
        path.mkdir(parents=True, exist_ok=True)
    (root / thread_id / "workspace").mkdir(parents=True, exist_ok=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "backend": "deimv2",
                "request": {"max_synthetic_images": 1},
            }
        ),
        encoding="utf-8",
    )
    (pipeline / "real_coco.json").write_text('{"images":[{"id":1}],"annotations":[],"categories":[]}', encoding="utf-8")
    (pipeline / "synthetic_coco.json").write_text('{"images":[{"id":2}],"annotations":[],"categories":[]}', encoding="utf-8")
    (synthetic_images / "synthetic_001.jpg").write_bytes(b"jpg")
    (deim_weights / "best_stg2.pth").write_bytes(b"checkpoint")

    artifact_store = ArtifactStore(root_dir=root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "runtime", SimpleNamespace(artifact_store=artifact_store))

    async def collect() -> list[dict]:
        request = AcpHttpSubscriptionRequest(threadId=thread_id)
        chunks: list[str] = []

        async def subscribe() -> None:
            async for chunk in _stream_acp_subscription(request):
                chunks.append(chunk)

        task = asyncio.create_task(subscribe())
        await asyncio.sleep(0)
        published: set[str] = set()
        await training._publish_training_status(thread_id, published)
        await training._publish_training_status(thread_id, published)
        await acp_event_broker.publish(
            {thread_id},
            "result",
            {"jsonrpc": "2.0", "id": 7, "result": {"stopReason": "end_turn"}},
        )
        await task
        return [
            json.loads(line.removeprefix("data: "))
            for chunk in chunks
            for line in chunk.splitlines()
            if line.startswith("data: ")
        ]

    payloads = asyncio.run(collect())
    updates = [
        payload["params"]["update"]
        for payload in payloads
        if payload.get("method") == "session/update"
        and payload.get("params", {}).get("update", {}).get("_meta", {}).get("jetlinksRuntimeEvent", {}).get("type") == "artifact.created"
    ]
    artifact_name_list = [update["artifact"]["name"] for update in updates]
    artifact_names = set(artifact_name_list)

    assert {"real_coco.json", "synthetic_coco.json", "synthetic_001.jpg", "best_stg2.pth"} <= artifact_names
    assert artifact_name_list.count("best_stg2.pth") == 1


def test_http_training_result_uses_failed_run_result() -> None:
    result = AgentRunResult(
        agent="default",
        thread_id="smoking",
        status="failed",
        reply="DEIMv2 training failed because annotations were empty.",
    )
    event = ChatEvent(type="run.failed", data={"result": result.model_dump(), "error": result.reply})

    parsed = training._result_from_event(event)

    assert parsed is not None
    assert parsed.status == "failed"
    assert parsed.reply == "DEIMv2 training failed because annotations were empty."


def test_cancel_training_job_falls_back_to_active_disk_run(tmp_path: Path, monkeypatch) -> None:
    thread_id = "disk-cancel-thread"
    run_id = "run-deimv2-001"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "training",
                "created_at": "2026-07-15T08:00:00Z",
                "started_at": "2026-07-15T08:00:00Z",
                "training": {"status": "running", "started_at": "2026-07-15T08:00:00Z"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-disk-cancel",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "running",
            "created_at": "2026-07-15T08:00:00Z",
            "started_at": "2026-07-15T08:00:00Z",
            "run_id": run_id,
        },
    )
    monkeypatch.setattr(training, "_terminate_training_processes", lambda thread, run: [1234])
    monkeypatch.setattr(training.runtime, "session_manager", SimpleNamespace(cancel_active_turn=lambda thread: False))
    training._jobs_by_thread.clear()
    training._jobs_by_id.clear()

    response = asyncio.run(training.cancel_training_job(thread_id))

    state = json.loads((run_dir / "progress_state.json").read_text(encoding="utf-8"))
    assert response["cancelled"] is True
    assert response["fallback"] is True
    assert response["terminated_processes"] == [1234]
    assert response["job"]["thread_id"] == thread_id
    assert response["job"]["run_id"] == run_id
    assert response["job"]["status"] == "cancelled"
    assert state["status"] == "cancelled"


def test_cancel_training_job_marks_cancelled_without_processes(tmp_path: Path, monkeypatch) -> None:
    thread_id = "disk-cancel-no-process-thread"
    run_id = "run-deimv2-no-process"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "llm_request",
                "created_at": "2026-07-16T08:00:00Z",
                "started_at": "2026-07-16T08:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-disk-cancel-no-process",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "running",
            "created_at": "2026-07-16T08:00:00Z",
            "started_at": "2026-07-16T08:00:00Z",
            "run_id": run_id,
        },
    )
    monkeypatch.setattr(training, "_terminate_training_processes", lambda thread, run: [])
    monkeypatch.setattr(training.runtime, "session_manager", SimpleNamespace(cancel_active_turn=lambda thread: False))
    training._jobs_by_thread.clear()
    training._jobs_by_id.clear()

    response = asyncio.run(training.cancel_training_job(thread_id))

    state = json.loads((run_dir / "progress_state.json").read_text(encoding="utf-8"))
    marker = json.loads((tmp_path / ".runtime" / "http_training_jobs" / f"{thread_id}.json").read_text(encoding="utf-8"))
    assert response["cancelled"] is True
    assert response["terminated_processes"] == []
    assert response["job"]["status"] == "cancelled"
    assert marker["status"] == "cancelled"
    assert state["status"] == "cancelled"


def test_cancel_training_job_matches_deimv2_input_file_process(tmp_path: Path, monkeypatch) -> None:
    thread_id = "input-file-cancel-thread"
    run_id = "run-deimv2-input-file"
    thread_root = tmp_path / ".runtime" / "threads" / thread_id
    run_dir = thread_root / "outputs" / "yolo_training_flow" / "runs" / run_id
    run_dir.mkdir(parents=True)
    input_path = thread_root / "workspace" / "deimv2-training-input.json"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps({"work_dir": str(run_dir)}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        training,
        "_process_table",
        lambda: [
            (111, 1, f"python run_deimv2_training.py --input {input_path}"),
            (222, 111, "python tools/train.py --config train.yml"),
        ],
    )
    killed: list[int] = []

    def fake_terminate(pid: int) -> bool:
        killed.append(pid)
        return True

    monkeypatch.setattr(training, "_terminate_process", fake_terminate)

    terminated = training._terminate_training_processes(thread_id, run_id)

    assert terminated == [222, 111]
    assert killed == [222, 111]


def test_cancelled_workflow_result_maps_to_cancelled_http_job() -> None:
    result = AgentRunResult(
        agent="default",
        thread_id="face",
        status="failed",
        reply="训练任务已收到取消请求，已在启动模型训练前停止。",
        metadata={"phase": "cancelled", "cancelled": True},
    )

    assert training._job_status_from_result(result) == "cancelled"


def test_http_subscription_polls_training_status_created_after_connect(tmp_path: Path, monkeypatch) -> None:
    thread_id = "post-connect-training"
    run_id = "run-deimv2-post-connect"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acp_http_stream, "SSE_STATUS_POLL_SECONDS", 0.01)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-post-connect",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "running",
            "created_at": "2026-07-15T08:00:00Z",
            "started_at": "2026-07-15T08:00:00Z",
        },
    )

    async def collect() -> list[dict]:
        chunks: list[str] = []
        request = AcpHttpSubscriptionRequest(threadId=thread_id)

        async def subscribe() -> None:
            async for chunk in _stream_acp_subscription(request):
                chunks.append(chunk)
                if "training/status" in chunk:
                    break

        task = asyncio.create_task(subscribe())
        await asyncio.sleep(0.02)
        run_dir.mkdir(parents=True)
        (run_dir / "progress_state.json").write_text(
            json.dumps(
                {
                    "thread_id": thread_id,
                    "run_id": run_id,
                    "backend": "deimv2",
                    "status": "running",
                    "phase": "training",
                    "created_at": "2026-07-15T08:00:00Z",
                    "started_at": "2026-07-15T08:00:00Z",
                    "training": {"status": "running", "started_at": "2026-07-15T08:00:00Z"},
                }
            ),
            encoding="utf-8",
        )
        await asyncio.wait_for(task, timeout=1)
        return [
            json.loads(line.removeprefix("data: "))
            for chunk in chunks
            for line in chunk.splitlines()
            if line.startswith("data: ")
        ]

    payloads = asyncio.run(collect())

    assert payloads[-1]["run_id"] == run_id
    assert payloads[-1]["status"] == "running"
    assert payloads[-1]["phase"] == "training"


def test_http_subscription_ignores_historical_run_without_active_http_marker(tmp_path: Path, monkeypatch) -> None:
    thread_id = "historical-training"
    run_id = "run-deimv2-historical"
    run_dir = tmp_path / ".runtime" / "threads" / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "training",
                "created_at": "2026-07-15T08:00:00Z",
                "started_at": "2026-07-15T08:00:00Z",
                "training": {"status": "running", "started_at": "2026-07-15T08:00:00Z"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acp_http_stream, "SSE_STATUS_POLL_SECONDS", 0.01)
    monkeypatch.setattr(acp_http_stream, "SSE_HEARTBEAT_SECONDS", 100.0)

    async def collect() -> list[str]:
        chunks: list[str] = []
        request = AcpHttpSubscriptionRequest(threadId=thread_id)

        async def subscribe() -> None:
            async for chunk in _stream_acp_subscription(request):
                chunks.append(chunk)

        task = asyncio.create_task(subscribe())
        await asyncio.sleep(0.05)
        await acp_event_broker.publish(
            {thread_id},
            "result",
            {"jsonrpc": "2.0", "id": 7, "result": {"stopReason": "end_turn"}},
        )
        await asyncio.wait_for(task, timeout=1)
        return chunks

    chunks = asyncio.run(collect())

    assert chunks[0] == ": connected\n\n"
    assert all("training/status" not in chunk for chunk in chunks)


def test_http_subscription_polling_bridges_artifacts_to_session_update(tmp_path: Path, monkeypatch) -> None:
    thread_id = "polling-artifact-thread"
    run_id = "run-deimv2-polling-artifact"
    root = tmp_path / ".runtime" / "threads"
    run_dir = root / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    pipeline = run_dir / "pipeline_work"
    uploaded = run_dir / "uploaded_dataset"
    pipeline.mkdir(parents=True)
    uploaded.mkdir(parents=True)
    (uploaded / "image_001.jpg").write_bytes(b"jpg")
    real_per_image = uploaded / "datasets" / "generated_images"
    real_per_image.mkdir(parents=True)
    (real_per_image / "image_001_coco.json").write_text(
        json.dumps({"images": [{"id": 3}], "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    (pipeline / "real_coco.json").write_text(
        json.dumps({"images": [{"id": 1}], "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    synthetic_annotations = pipeline / "synthetic_annotations"
    synthetic_annotations.mkdir()
    (synthetic_annotations / "scene_001_coco.json").write_text(
        json.dumps({"images": [{"id": 2}], "annotations": [], "categories": []}),
        encoding="utf-8",
    )
    train_dir = run_dir / "deimv2_training_run" / "runs" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "best_stg1.onnx").write_bytes(b"onnx")
    deimv2_run = run_dir / "deimv2_training_run"
    (deimv2_run / "run_summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (deimv2_run / "training_summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "running",
                "phase": "annotation",
                "created_at": "2026-07-15T08:00:00Z",
                "started_at": "2026-07-15T08:00:00Z",
                "request": {"annotation_prompts": ["person", "face", "cigarette"]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(root_dir=root))
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-polling-artifact",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "running",
            "created_at": "2026-07-15T08:00:00Z",
            "started_at": "2026-07-15T08:00:00Z",
            "run_id": run_id,
        },
    )

    status = acp_http_stream._initial_training_status({thread_id})
    published: set[str] = set()
    payloads = acp_http_stream._artifact_session_update_payloads({thread_id}, published)

    assert status is not None
    assert payloads[0]["method"] == "session/update"
    artifact_names = [
        payload["params"]["update"]["artifact"]["name"]
        for payload in payloads
        if payload["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"] == "artifact.created"
    ]
    assert "real_coco.json" in artifact_names
    assert "image_001_coco.json" in artifact_names
    assert "scene_001_coco.json" in artifact_names
    assert "best_stg1.onnx" in artifact_names
    assert "run_summary.json" in artifact_names
    assert "training_summary.json" in artifact_names
    assert acp_http_stream._artifact_session_update_payloads({thread_id}, published) == []


def test_http_job_final_reply_is_published_as_session_update() -> None:
    thread_id = "http-final-reply-thread"

    async def collect() -> dict:
        subscriber_id, queue = await acp_event_broker.subscribe({thread_id})
        try:
            result = AgentRunResult(
                agent="default",
                thread_id=thread_id,
                status="completed",
                reply="Training completed; artifacts generated.",
            )
            await training._publish_http_job_final_reply(thread_id, "job-final-reply", result)
            return (await asyncio.wait_for(queue.get(), timeout=1)).payload
        finally:
            await acp_event_broker.unsubscribe(subscriber_id, {thread_id})

    payload = asyncio.run(collect())

    assert payload["method"] == "session/update"
    update = payload["params"]["update"]
    assert update["sessionUpdate"] == "agent_message_chunk"
    assert update["content"]["text"] == "Training completed; artifacts generated."
    assert update["_meta"]["jetlinksRuntimeEvent"]["type"] == "http.training_job.reply"


def test_http_subscription_does_not_replay_unobserved_terminal_run(tmp_path: Path, monkeypatch) -> None:
    thread_id = "terminal-replay-thread"
    run_id = "run-deimv2-terminal-replay"
    root = tmp_path / ".runtime" / "threads"
    run_dir = root / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    train_dir = run_dir / "deimv2_training_run" / "runs" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "best_stg1.onnx").write_bytes(b"onnx")
    (run_dir / "deimv2_training_run" / "run_summary.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "completed",
                "phase": "completed",
                "created_at": "2026-07-16T08:00:00Z",
                "started_at": "2026-07-16T08:00:00Z",
                "completed_at": completed_at,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(root_dir=root))
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-terminal-replay",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "completed",
            "created_at": "2026-07-16T08:00:00Z",
            "started_at": "2026-07-16T08:00:00Z",
            "completed_at": completed_at,
            "run_id": run_id,
        },
    )

    assert acp_http_stream._initial_training_status({thread_id}) is None
    assert acp_http_stream._artifact_session_update_payloads({thread_id}, set()) == []
    assert acp_http_stream._terminal_reply_session_update_payloads({thread_id}, set()) == []


def test_http_subscription_replays_observed_terminal_artifacts_and_reply(tmp_path: Path, monkeypatch) -> None:
    thread_id = "observed-terminal-replay-thread"
    run_id = "run-deimv2-observed-terminal-replay"
    root = tmp_path / ".runtime" / "threads"
    run_dir = root / thread_id / "outputs" / "yolo_training_flow" / "runs" / run_id
    train_dir = run_dir / "deimv2_training_run" / "runs" / "train"
    train_dir.mkdir(parents=True)
    (train_dir / "best_stg1.onnx").write_bytes(b"onnx")
    (run_dir / "deimv2_training_run" / "run_summary.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    (run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": run_id,
                "backend": "deimv2",
                "status": "completed",
                "phase": "completed",
                "created_at": "2026-07-16T08:00:00Z",
                "started_at": "2026-07-16T08:00:00Z",
                "completed_at": completed_at,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(acp_http_stream, "model_store", ArtifactStore(root_dir=root))
    final_reply = "# DEIMv2 training report\n\nmAP50: 0.812\n\nUse best_stg1.pth for deployment."
    marker = {
        "job_id": "job-observed-terminal-replay",
        "thread_id": thread_id,
        "agent_name": "default",
        "status": "completed",
        "created_at": "2026-07-16T08:00:00Z",
        "started_at": "2026-07-16T08:00:00Z",
        "completed_at": completed_at,
        "run_id": run_id,
        "result": {
            "agent": "default",
            "thread_id": thread_id,
            "status": "completed",
            "reply": final_reply,
            "content": [{"type": "text", "text": final_reply}],
            "artifacts": [],
            "verification": None,
            "spec": None,
            "metadata": {"run_id": run_id},
        },
    }
    write_http_training_job_marker(thread_id, marker)
    observed = {acp_http_stream._marker_job_key(thread_id, marker)}

    status = acp_http_stream._initial_training_status({thread_id}, observed)
    artifact_payloads = acp_http_stream._artifact_session_update_payloads({thread_id}, set(), observed)
    reply_payloads = acp_http_stream._terminal_reply_session_update_payloads({thread_id}, set(), observed)
    terminal = acp_http_stream._terminal_result_payload({thread_id}, set(), observed)

    assert status is not None
    assert status["status"] == "completed"
    artifact_names = [
        payload["params"]["update"]["artifact"]["name"]
        for payload in artifact_payloads
        if payload["params"]["update"]["_meta"]["jetlinksRuntimeEvent"]["type"] == "artifact.created"
    ]
    assert "best_stg1.onnx" in artifact_names
    assert "run_summary.json" in artifact_names
    assert reply_payloads[0]["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
    assert reply_payloads[0]["params"]["update"]["content"]["text"] == final_reply
    assert terminal is not None
    terminal_event, terminal_payload = terminal
    assert terminal_event == "result"
    assert terminal_payload["id"] == "job-observed-terminal-replay"
    assert terminal_payload["result"]["status"] == "completed"
    assert terminal_payload["result"]["reply"] == final_reply
    assert terminal_payload["result"]["content"][0]["text"] == final_reply
    assert terminal_payload["result"]["metadata"]["run_id"] == run_id


def test_http_subscription_ignores_stale_latest_run_before_new_run_starts(tmp_path: Path, monkeypatch) -> None:
    thread_id = "stale-latest-run-thread"
    old_run_id = "run-deimv2-old"
    root = tmp_path / ".runtime" / "threads"
    old_run_dir = root / thread_id / "outputs" / "yolo_training_flow" / "runs" / old_run_id
    old_run_dir.mkdir(parents=True)
    (old_run_dir / "progress_state.json").write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "run_id": old_run_id,
                "backend": "deimv2",
                "status": "completed",
                "phase": "completed",
                "created_at": "2026-07-15T08:00:00Z",
                "started_at": "2026-07-15T08:00:00Z",
                "completed_at": "2026-07-15T08:10:00Z",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    write_http_training_job_marker(
        thread_id,
        {
            "job_id": "job-new-without-run-yet",
            "thread_id": thread_id,
            "agent_name": "default",
            "status": "running",
            "created_at": "2026-07-16T08:00:00Z",
            "started_at": "2026-07-16T08:00:00Z",
            "run_id": None,
        },
    )

    assert acp_http_stream._initial_training_status({thread_id}, set()) is None


def test_training_status_fingerprint_ignores_volatile_fields() -> None:
    first = {
        "thread_id": "smoking",
        "run_id": "run-001",
        "status": "running",
        "phase": "training",
        "updated_at": "2026-07-15T09:00:00Z",
        "elapsed_seconds": 1,
        "resources": {"timestamp": "2026-07-15T09:00:00Z", "gpu": {"utilization_percent": 10}},
        "info": {"stage": "training", "status": "running", "metrics": {"current_epoch": 0, "progress": 0.1}},
    }
    second = {
        **first,
        "updated_at": "2026-07-15T09:00:02Z",
        "elapsed_seconds": 3,
        "resources": {"timestamp": "2026-07-15T09:00:02Z", "gpu": {"utilization_percent": 99}},
    }

    assert acp_http_stream._status_fingerprint(first) == acp_http_stream._status_fingerprint(second)
