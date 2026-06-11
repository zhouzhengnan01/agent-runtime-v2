from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from app.core.diagnostics import diagnostic_json, env_flag, env_int
from app.schemas import ChatEvent

logger = logging.getLogger("uvicorn.error")
RUNTIME_EVENT_TRACE_PAYLOADS = env_flag("RUNTIME_EVENT_TRACE_PAYLOADS", "0")
RUNTIME_EVENT_TRACE_MAX_CHARS = env_int("RUNTIME_EVENT_TRACE_MAX_CHARS", 100)

_LOGGED_EVENT_TYPES = {
    "agent.message",
    "artifact.created",
    "context.compacted",
    "llm.empty_response",
    "llm.request.completed",
    "llm.request.started",
    "llm.started",
    "mcp.discovery.failed",
    "memory.context.loaded",
    "review.input",
    "review.llm.raw_reply",
    "review.normalized_result",
    "review.skill_invocation.completed",
    "review.skill_invocation.failed",
    "review.skill_invocation.started",
    "review.skill_selection.completed",
    "run.completed",
    "run.failed",
    "run.started",
    "skill.completed",
    "skill.context.loaded",
    "skill.failed",
    "skill.started",
    "tool.calls.started",
    "tool.completed",
    "tool.failed",
    "tool.loop.auto_repeating",
    "tool.loop.waiting",
    "tool.started",
    "tools.available",
    "verifier.completed",
    "verifier.failed",
    "verifier.started",
    "visualization.full.completed",
    "visualization.full.initialization.completed",
    "visualization.initialization.ready",
    "visualization.region.concurrent.batch_started",
    "visualization.region.concurrent.completed",
    "visualization.region.concurrent.failed",
    "visualization.region.concurrent.submitted",
    "visualization.region.finalized",
    "visualization.region.ready",
}


@dataclass
class EventRecorder:
    """Collect typed run events for HTTP, CLI, and UI surfaces."""

    agent: str
    thread_id: str
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    on_emit: Callable[[ChatEvent], None] | None = None
    _started_at: float = field(default_factory=time.perf_counter)
    _sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    events: list[ChatEvent] = field(default_factory=list)

    def emit(self, event_type: str, data: dict[str, Any] | None = None, message: str | None = None) -> ChatEvent:
        with self._lock:
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
            self._sequence += 1
        _log_event(event)
        if self.on_emit is not None:
            self.on_emit(event)
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
        self._write_json_file(self._run_path(run_id), payload)
        return payload

    @staticmethod
    def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        tmp_file = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_file.write_text(content, encoding="utf-8")
        last_error: PermissionError | None = None
        for attempt in range(5):
            try:
                tmp_file.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            if last_error is not None:
                raise last_error
            raise
        finally:
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass

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


def _log_event(event: ChatEvent) -> None:
    if event.type not in _LOGGED_EVENT_TYPES:
        return
    data = event.data
    fields = _event_log_fields(event.type, data)
    level = (
        logging.WARNING
        if event.type.endswith(".failed") or event.type in {"run.failed", "mcp.discovery.failed"}
        else logging.INFO
    )
    logger.log(
        level,
        "\n===== 运行事件 | runtime event type=%s =====\n%s\n===== 运行事件结束 =====",
        event.type,
        _render_multiline_fields(fields),
    )
    if RUNTIME_EVENT_TRACE_PAYLOADS:
        logger.log(
            level,
            "\n----- 运行事件详情 | runtime event detail type=%s -----\n"
            "运行ID: %s\n"
            "线程ID: %s\n"
            "序号: %s\n"
            "完整数据:\n%s\n"
            "----- 运行事件详情结束 -----",
            event.type,
            data.get("run_id"),
            data.get("thread_id"),
            data.get("sequence"),
            _pretty_json(diagnostic_json(data, max_chars=RUNTIME_EVENT_TRACE_MAX_CHARS)),
        )


def _event_log_fields(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "run_id": data.get("run_id"),
        "thread_id": data.get("thread_id"),
        "agent": data.get("agent"),
        "seq": data.get("sequence"),
        "elapsed_ms": data.get("elapsed_ms"),
    }
    for key in (
        "workflow",
        "execution_mode",
        "model",
        "configured",
        "request_timeout_seconds",
        "round",
        "mode",
        "message_count",
        "tool_count",
        "tool_choice",
        "finish_reason",
        "tool_call_count",
        "duration_ms",
        "content_chars",
        "tool_name",
        "tool_call_id",
        "is_error",
        "error_code",
        "turn_phase",
        "turn_policy_reason",
        "verification_verdict",
        "verification_reason",
        "skill_name",
        "skill_md_path",
        "reference_count",
        "total_reference_chars",
        "server_name",
        "server_type",
        "endpoint",
        "review_source_id",
        "objective",
        "selected_skill",
        "attachment_count",
        "image_attachment_count",
        "raw_chars",
        "result_chars",
        "reason",
        "error",
        "stage",
        "region_id",
        "region_count",
        "submitted_count",
        "completed_count",
        "component_count",
        "failed_region_count",
        "resource_save_count",
        "advanced_component_count",
        "max_workers",
        "status",
        "duration_max_ms",
        "duration_total_ms",
    ):
        if key in data:
            fields[key] = data.get(key)
    if event_type == "tools.available":
        fields["tools"] = _compact_names(data.get("tools"))
        fields["priority_tools"] = _compact_names(data.get("priority_tools"))
        fields["hidden_tool_count"] = (
            len(data.get("hidden_tools")) if isinstance(data.get("hidden_tools"), list) else None
        )
    if event_type == "tool.calls.started":
        calls = data.get("tool_calls")
        fields["tool_calls"] = _tool_call_names(calls)
    if event_type == "skill.context.loaded":
        references = data.get("references")
        fields["references"] = _reference_names(references)
        fields["skill_md_truncated"] = data.get("skill_md_truncated")
        fields["reference_truncated_count"] = data.get("reference_truncated_count")
    if event_type in {"run.completed", "run.failed"}:
        result = data.get("result")
        metadata = result.get("metadata") if isinstance(result, dict) else None
        if isinstance(metadata, dict):
            fields["status"] = result.get("status")
            fields["reply_chars"] = len(str(result.get("reply") or ""))
            fields["tool_rounds"] = metadata.get("tool_rounds")
            fields["tool_call_count"] = metadata.get("tool_call_count")
            fields["artifacts"] = (
                len(result.get("artifacts")) if isinstance(result.get("artifacts"), list) else None
            )
    if event_type in {"tool.completed", "tool.failed"}:
        structured = data.get("structured_content")
        if isinstance(structured, dict):
            fields["structured_keys"] = _compact_names(list(structured.keys()))
            fields["error"] = structured.get("error")
    if event_type == "mcp.discovery.failed":
        fields["error"] = data.get("error")
    if event_type == "agent.message":
        fields["text_chars"] = len(str(data.get("text") or data.get("message") or ""))
    return fields


def _compact_names(value: object, *, limit: int = 12) -> str:
    if not isinstance(value, list | tuple):
        return ""
    names = [str(item) for item in value if str(item)]
    suffix = "" if len(names) <= limit else f",+{len(names) - limit}"
    return ",".join(names[:limit]) + suffix


def _tool_call_names(value: object, *, limit: int = 12) -> str:
    if not isinstance(value, list | tuple):
        return ""
    names: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        call_id = str(item.get("id") or "").strip()
        if name and call_id:
            names.append(f"{name}({call_id})")
        elif name:
            names.append(name)
    suffix = "" if len(names) <= limit else f",+{len(names) - limit}"
    return ",".join(names[:limit]) + suffix


def _reference_names(value: object, *, limit: int = 12) -> str:
    if not isinstance(value, list | tuple):
        return ""
    names: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("path"):
            names.append(str(item["path"]))
    suffix = "" if len(names) <= limit else f",+{len(names) - limit}"
    return ",".join(names[:limit]) + suffix


def _log_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    rendered = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if len(rendered) > 500:
        rendered = rendered[:497] + "..."
    if not rendered or any(char.isspace() for char in rendered):
        return json.dumps(rendered, ensure_ascii=False)
    return rendered


def _render_multiline_fields(fields: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        lines.append(f"{_field_label(key)}: {_display_value(value)}")
    return "\n".join(lines) if lines else "无摘要字段"


def _display_value(value: object) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _pretty_json(value: str) -> str:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _field_label(key: str) -> str:
    return {
        "run_id": "运行ID",
        "thread_id": "线程ID",
        "agent": "智能体",
        "seq": "序号",
        "elapsed_ms": "已耗时(ms)",
        "workflow": "工作流",
        "execution_mode": "执行模式",
        "model": "模型",
        "configured": "是否已配置",
        "request_timeout_seconds": "请求超时(秒)",
        "round": "轮次",
        "mode": "模式",
        "message_count": "消息数",
        "tool_count": "工具数",
        "tool_choice": "工具选择",
        "finish_reason": "结束原因",
        "tool_call_count": "工具调用数",
        "duration_ms": "耗时(ms)",
        "content_chars": "内容字符数",
        "tool_name": "工具名",
        "tool_call_id": "工具调用ID",
        "is_error": "是否错误",
        "error_code": "错误码",
        "turn_phase": "对话阶段",
        "turn_policy_reason": "对话策略原因",
        "verification_verdict": "校验结论",
        "verification_reason": "校验原因",
        "skill_name": "技能名",
        "skill_md_path": "技能说明路径",
        "reference_count": "参考资料数",
        "total_reference_chars": "参考资料字符数",
        "server_name": "服务名",
        "server_type": "服务类型",
        "endpoint": "端点",
        "reason": "原因",
        "tools": "可用工具",
        "priority_tools": "优先工具",
        "hidden_tool_count": "隐藏工具数",
        "tool_calls": "工具调用",
        "references": "参考资料",
        "skill_md_truncated": "技能说明是否截断",
        "reference_truncated_count": "参考资料截断数",
        "status": "状态",
        "reply_chars": "回复字符数",
        "tool_rounds": "工具轮数",
        "artifacts": "产物数",
        "structured_keys": "结构化字段",
        "error": "错误",
        "text_chars": "文本字符数",
    }.get(key, key)
