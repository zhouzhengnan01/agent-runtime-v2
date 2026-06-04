from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any
import uuid

from app.schemas import ChatEvent


@dataclass
class EventRecorder:
    """Collect typed run events for HTTP, CLI, and UI surfaces."""

    agent: str
    thread_id: str
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    on_emit: Callable[[ChatEvent], None] | None = None
    _started_at: float = field(default_factory=time.perf_counter)
    _sequence: int = 0
    events: list[ChatEvent] = field(default_factory=list)

    def emit(self, event_type: str, data: dict[str, Any] | None = None, message: str | None = None) -> ChatEvent:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "agent": self.agent,
            "thread_id": self.thread_id,
            "sequence": self._sequence,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "elapsed_ms": round((time.perf_counter() - self._started_at) * 1000, 3),
        }
        if data:
            payload.update(data)
        if message is not None:
            payload["message"] = message

        event = ChatEvent(type=event_type, data=payload)
        self.events.append(event)
        if self.on_emit is not None:
            self.on_emit(event)
        self._sequence += 1
        return event


class RunEventStore:
    """Persist per-run event timelines for debugging and UI inspection."""

    def __init__(self, root_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.root_dir = root_dir or project_root / ".runtime" / "runs"

    def save(
        self,
        *,
        run_id: str,
        agent: str,
        thread_id: str,
        events: builtins.list[ChatEvent],
        result: dict[str, Any] | None = None,
        agent_snapshot: dict[str, Any] | None = None,
        request_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self._payload(
            run_id=run_id,
            agent=agent,
            thread_id=thread_id,
            events=events,
            result=result,
            agent_snapshot=agent_snapshot,
            request_snapshot=request_snapshot,
        )
        self.root_dir.mkdir(parents=True, exist_ok=True)
        target = self._run_path(run_id).with_suffix(".json.tmp")
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(self._run_path(run_id))
        return payload

    def get(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(self._safe_run_id(run_id))
        if not path.is_file():
            raise FileNotFoundError(f"Run not found: {run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def debug_bundle(self, run_id: str) -> dict[str, Any]:
        payload = self.get(run_id)
        return {
            "schema": "jetlinks-agent-run-debug-bundle.v1",
            "run": {
                "run_id": payload.get("run_id", run_id),
                "agent": payload.get("agent", ""),
                "thread_id": payload.get("thread_id", ""),
                "status": payload.get("status", ""),
                "workflow": payload.get("workflow", ""),
                "started_at": payload.get("started_at", ""),
                "completed_at": payload.get("completed_at", ""),
                "duration_ms": payload.get("duration_ms", 0),
                "event_count": payload.get("event_count", 0),
                "tool_call_count": payload.get("tool_call_count", 0),
            },
            "agent_snapshot": payload.get("agent_snapshot") or {},
            "request_snapshot": payload.get("request_snapshot") or {},
            "result": payload.get("result"),
            "events": payload.get("events", []),
            "diagnostics": _diagnostics(payload.get("events", [])),
        }

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root_dir.is_dir():
            return []
        capped = max(1, min(limit, 200))
        items: list[dict[str, Any]] = []
        for path in sorted(self.root_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            items.append(
                {
                    "run_id": data.get("run_id", path.stem),
                    "agent": data.get("agent", ""),
                    "thread_id": data.get("thread_id", ""),
                    "status": data.get("status", ""),
                    "workflow": data.get("workflow", ""),
                    "started_at": data.get("started_at", ""),
                    "completed_at": data.get("completed_at", ""),
                    "duration_ms": data.get("duration_ms", 0),
                    "event_count": data.get("event_count", 0),
                    "tool_call_count": data.get("tool_call_count", 0),
                }
            )
            if len(items) >= capped:
                break
        return items

    def _run_path(self, run_id: str) -> Path:
        return self.root_dir / f"{self._safe_run_id(run_id)}.json"

    @staticmethod
    def _safe_run_id(run_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in run_id.strip())
        return cleaned.strip(".-")[:128] or f"run-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _payload(
        *,
        run_id: str,
        agent: str,
        thread_id: str,
        events: builtins.list[ChatEvent],
        result: dict[str, Any] | None,
        agent_snapshot: dict[str, Any] | None,
        request_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        serialized = [event.model_dump(mode="json") for event in events]
        first = serialized[0]["data"] if serialized else {}
        last = serialized[-1]["data"] if serialized else {}
        final_type = serialized[-1]["type"] if serialized else ""
        result_payload = result or _result_from_events(events)
        metadata = result_payload.get("metadata") if isinstance(result_payload, dict) else {}
        return {
            "run_id": run_id,
            "agent": agent,
            "thread_id": thread_id,
            "status": _status_from_final_type(final_type, result_payload),
            "workflow": _workflow_from_events(events, metadata if isinstance(metadata, dict) else {}),
            "started_at": first.get("timestamp", ""),
            "completed_at": last.get("timestamp", ""),
            "duration_ms": last.get("elapsed_ms", 0),
            "event_count": len(serialized),
            "tool_call_count": _tool_call_count(events, metadata if isinstance(metadata, dict) else {}),
            "agent_snapshot": agent_snapshot or {},
            "request_snapshot": request_snapshot or {},
            "events": serialized,
            "result": result_payload,
        }


def _result_from_events(events: list[ChatEvent]) -> dict[str, Any] | None:
    for event in reversed(events):
        result = event.data.get("result")
        if isinstance(result, dict):
            return result
    return None


def _status_from_final_type(final_type: str, result: dict[str, Any] | None) -> str:
    if isinstance(result, dict) and isinstance(result.get("status"), str):
        return str(result["status"])
    if final_type == "run.completed":
        return "completed"
    if final_type == "run.failed":
        return "failed"
    return "unknown"


def _workflow_from_events(events: list[ChatEvent], metadata: dict[str, Any]) -> str:
    if isinstance(metadata.get("workflow"), str):
        return str(metadata["workflow"])
    for event in events:
        workflow = event.data.get("workflow")
        if isinstance(workflow, str):
            return workflow
    return ""


def _tool_call_count(events: list[ChatEvent], metadata: dict[str, Any]) -> int:
    value = metadata.get("tool_call_count")
    if isinstance(value, int):
        return value
    return sum(1 for event in events if event.type in {"tool.started", "skill.started"})


def _diagnostics(events: object) -> dict[str, Any]:
    serialized = events if isinstance(events, list) else []
    failures: list[dict[str, Any]] = []
    for event in serialized:
        if not isinstance(event, dict):
            continue
        if event.get("type") not in {"run.failed", "tool.failed", "skill.failed", "verifier.failed"}:
            continue
        raw_data = event.get("data")
        data = raw_data if isinstance(raw_data, dict) else {}
        failures.append(
            {
                "type": event.get("type"),
                "sequence": data.get("sequence"),
                "error_code": data.get("error_code") or _structured_error_code(data),
                "message": data.get("error") or data.get("message") or _structured_error(data),
                "tool_name": data.get("tool_name"),
                "duration_ms": data.get("duration_ms"),
            }
        )
    return {
        "failure_count": len(failures),
        "failures": failures,
        "tool_events": sum(1 for event in serialized if isinstance(event, dict) and event.get("type") == "tool.started"),
        "skill_events": sum(1 for event in serialized if isinstance(event, dict) and event.get("type") == "skill.started"),
    }


def _structured_error_code(data: dict[str, Any]) -> str:
    structured = data.get("structured_content")
    value = structured.get("error_code") if isinstance(structured, dict) else None
    if isinstance(value, str):
        return value
    return ""


def _structured_error(data: dict[str, Any]) -> str:
    structured = data.get("structured_content")
    value = structured.get("error") if isinstance(structured, dict) else None
    if isinstance(value, str):
        return value
    return ""
