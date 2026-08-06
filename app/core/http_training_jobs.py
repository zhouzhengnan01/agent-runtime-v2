from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


ACTIVE_HTTP_TRAINING_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_HTTP_TRAINING_STATUSES = {"completed", "failed", "cancelled", "queue_rejected"}
SERVICE_INSTANCE_ENV = "JETLINKS_SERVICE_INSTANCE_ID"
SERVICE_INSTANCE_NAME = "service_instance.json"
SERVICE_INSTANCE_LOCK_NAME = "service_instance.lock"
HTTP_TRAINING_JOBS_DIR = "http_training_jobs"
HTTP_TRAINING_JOB_HISTORY_LIMIT = 200

_SERVICE_INSTANCE_ID: str | None = None
_JOB_MARKER_THREAD_LOCKS: dict[str, threading.Lock] = {}
_JOB_MARKER_THREAD_LOCKS_GUARD = threading.Lock()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def current_service_instance_id() -> str:
    global _SERVICE_INSTANCE_ID
    if _SERVICE_INSTANCE_ID:
        return _SERVICE_INSTANCE_ID
    configured = os.getenv(SERVICE_INSTANCE_ENV)
    if configured:
        _SERVICE_INSTANCE_ID = configured
        return configured
    _SERVICE_INSTANCE_ID = _load_or_create_service_instance_id()
    return _SERVICE_INSTANCE_ID


def record_http_training_job_request(thread_id: str) -> str:
    request_id = f"request-{uuid.uuid4().hex[:12]}"
    received_at = utc_now()
    path = _job_marker_path(thread_id)
    with _job_marker_lock(thread_id):
        payload = _read_json(path) or _base_job_marker(thread_id)
        stats = _request_stats(payload)
        stats["received_count"] += 1
        stats["pending_count"] += 1
        stats["last_received_at"] = received_at
        history = _job_history(payload)
        history.append(
            {
                "request_id": request_id,
                "received_at": received_at,
                "accepted": None,
                "http_status": None,
                "reason": None,
                "job_id": None,
                "run_id": None,
                "status": "received",
            }
        )
        payload["job_history"] = history[-HTTP_TRAINING_JOB_HISTORY_LIMIT:]
        payload["updated_at"] = received_at
        _write_json(path, payload)
    return request_id


def reject_http_training_job_request(
    thread_id: str,
    request_id: str,
    *,
    http_status: int,
    reason: str,
) -> None:
    path = _job_marker_path(thread_id)
    with _job_marker_lock(thread_id):
        payload = _read_json(path) or _base_job_marker(thread_id)
        _complete_request_audit(
            payload,
            request_id,
            accepted=False,
            http_status=http_status,
            reason=reason,
        )
        payload["updated_at"] = utc_now()
        _write_json(path, payload)


def mark_http_training_job_request_duplicate(
    thread_id: str,
    request_id: str,
    *,
    existing_job_id: str,
    existing_run_id: str | None = None,
) -> None:
    path = _job_marker_path(thread_id)
    with _job_marker_lock(thread_id):
        payload = _read_json(path) or _base_job_marker(thread_id)
        stats = _request_stats(payload)
        history = _job_history(payload)
        entry = next((item for item in reversed(history) if item.get("request_id") == request_id), None)
        if entry is None:
            return
        if entry.get("status") == "received":
            stats["pending_count"] = max(0, stats["pending_count"] - 1)
            stats["duplicate_count"] += 1
        entry.update(
            {
                "accepted": True,
                "duplicate": True,
                "created_job": False,
                "http_status": 200,
                "reason": "Training job already exists for this thread.",
                "job_id": existing_job_id,
                "existing_job_id": existing_job_id,
                "run_id": existing_run_id,
                "status": "returned_existing",
                "decided_at": utc_now(),
            }
        )
        payload["updated_at"] = utc_now()
        _write_json(path, payload)


def write_http_training_job_marker(
    thread_id: str,
    record: dict[str, Any],
    *,
    request_id: str | None = None,
) -> None:
    path = _job_marker_path(thread_id)
    with _job_marker_lock(thread_id):
        previous = _read_json(path)
        payload = {
            "schema": "jetlinks-http-training-job.v1",
            "service_instance_id": current_service_instance_id(),
            "thread_id": thread_id,
            "job_id": str(record.get("job_id") or f"thread-{thread_id}"),
            "agent_name": str(record.get("agent_name") or "default"),
            "status": str(record.get("status") or "queued"),
            "created_at": record.get("created_at") or utc_now(),
            "started_at": record.get("started_at"),
            "completed_at": record.get("completed_at"),
            "run_id": record.get("run_id"),
            "error": record.get("error"),
            "result": record.get("result"),
            "queue_status": record.get("queue_status"),
            "queue_position": record.get("queue_position"),
            "queued_at": record.get("queued_at"),
            "request_stats": previous.get("request_stats") if isinstance(previous.get("request_stats"), dict) else {},
            "job_history": previous.get("job_history") if isinstance(previous.get("job_history"), list) else [],
            "updated_at": utc_now(),
        }
        if request_id:
            _complete_request_audit(
                payload,
                request_id,
                accepted=True,
                http_status=200,
                job_id=payload["job_id"],
                job_status=payload["status"],
            )
        _sync_current_job_history(payload)
        _write_json(path, payload)


def update_http_training_job_marker(thread_id: str, **updates: Any) -> None:
    path = _job_marker_path(thread_id)
    with _job_marker_lock(thread_id):
        payload = _read_json(path)
        if not payload:
            payload = _base_job_marker(thread_id)
        nullable_fields = {"queue_position", "queued_at", "completed_at"}
        payload.update(
            {
                key: value
                for key, value in updates.items()
                if value is not None or key in nullable_fields
            }
        )
        # A successful write means the current runtime instance has adopted the job.
        payload["service_instance_id"] = current_service_instance_id()
        payload["thread_id"] = thread_id
        payload["updated_at"] = utc_now()
        _sync_current_job_history(payload)
        _write_json(path, payload)


def read_current_http_training_job_marker(
    thread_id: str,
    *,
    require_active: bool = True,
    allow_previous_instance: bool = False,
) -> dict[str, Any] | None:
    payload = _read_json(_job_marker_path(thread_id))
    if not payload:
        return None
    if not allow_previous_instance and payload.get("service_instance_id") != current_service_instance_id():
        return None
    status = str(payload.get("status") or "")
    if require_active and status not in ACTIVE_HTTP_TRAINING_STATUSES:
        return None
    return payload


def list_http_training_job_markers(*, require_active: bool = False) -> list[dict[str, Any]]:
    root = (_runtime_dir() / HTTP_TRAINING_JOBS_DIR).resolve()
    if not root.is_dir():
        return []
    markers: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        status = str(payload.get("status") or "")
        if require_active and status not in ACTIVE_HTTP_TRAINING_STATUSES:
            continue
        markers.append(payload)
    return markers


def is_current_http_training_job_active(thread_id: str) -> bool:
    return read_current_http_training_job_marker(thread_id, require_active=True) is not None


def _base_job_marker(thread_id: str) -> dict[str, Any]:
    return {
        "schema": "jetlinks-http-training-job.v1",
        "service_instance_id": current_service_instance_id(),
        "thread_id": thread_id,
        "request_stats": {
            "received_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "duplicate_count": 0,
            "pending_count": 0,
            "last_received_at": None,
        },
        "job_history": [],
        "updated_at": utc_now(),
    }


def _request_stats(payload: dict[str, Any]) -> dict[str, Any]:
    existing = payload.get("request_stats")
    stats = existing if isinstance(existing, dict) else {}
    normalized = {
        "received_count": _non_negative_int(stats.get("received_count")),
        "accepted_count": _non_negative_int(stats.get("accepted_count")),
        "rejected_count": _non_negative_int(stats.get("rejected_count")),
        "duplicate_count": _non_negative_int(stats.get("duplicate_count")),
        "pending_count": _non_negative_int(stats.get("pending_count")),
        "last_received_at": stats.get("last_received_at"),
    }
    payload["request_stats"] = normalized
    return normalized


def _job_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
    history = payload.get("job_history")
    if not isinstance(history, list):
        history = []
    normalized = [item for item in history if isinstance(item, dict)]
    payload["job_history"] = normalized
    return normalized


def _complete_request_audit(
    payload: dict[str, Any],
    request_id: str,
    *,
    accepted: bool,
    http_status: int,
    reason: str | None = None,
    job_id: str | None = None,
    job_status: str | None = None,
) -> None:
    stats = _request_stats(payload)
    history = _job_history(payload)
    entry = next((item for item in reversed(history) if item.get("request_id") == request_id), None)
    if entry is None:
        return
    if entry.get("accepted") is None:
        stats["pending_count"] = max(0, stats["pending_count"] - 1)
        counter = "accepted_count" if accepted else "rejected_count"
        stats[counter] += 1
    entry.update(
        {
            "accepted": accepted,
            "http_status": http_status,
            "reason": reason,
            "job_id": job_id,
            "status": job_status or ("rejected" if not accepted else "queued"),
            "decided_at": utc_now(),
        }
    )


def _sync_current_job_history(payload: dict[str, Any]) -> None:
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        return
    history = _job_history(payload)
    entry = next(
        (
            item
            for item in reversed(history)
            if str(item.get("job_id") or "") == job_id
            and not bool(item.get("duplicate"))
        ),
        None,
    )
    if entry is None:
        return
    for key in ("status", "run_id", "started_at", "completed_at", "error"):
        if key in payload:
            entry[key] = payload.get(key)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _load_or_create_service_instance_id() -> str:
    runtime = _runtime_dir()
    runtime.mkdir(parents=True, exist_ok=True)
    marker_path = runtime / SERVICE_INSTANCE_NAME
    lock_path = runtime / SERVICE_INSTANCE_LOCK_NAME
    deadline = time.monotonic() + 5.0
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _is_stale_lock(lock_path) or time.monotonic() >= deadline:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            time.sleep(0.05)
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            payload = _read_json(marker_path)
            if _service_instance_is_alive(payload):
                return str(payload["instance_id"])
            instance_id = f"svc-{uuid.uuid4().hex}"
            _write_json(
                marker_path,
                {
                    "schema": "jetlinks-service-instance.v1",
                    "instance_id": instance_id,
                    "owner_pid": os.getpid(),
                    "started_at": utc_now(),
                    "cwd": str(Path.cwd()),
                },
            )
            return instance_id
        finally:
            try:
                lock_path.unlink()
            except OSError:
                pass


def _service_instance_is_alive(payload: dict[str, Any]) -> bool:
    instance_id = payload.get("instance_id")
    if not instance_id:
        return False
    try:
        pid = int(payload.get("owner_pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0 or not _pid_exists(pid):
        return False
    cwd = payload.get("cwd")
    return not cwd or _pid_cwd_matches(pid, str(cwd))


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _pid_cwd_matches(pid: int, expected_cwd: str) -> bool:
    proc_cwd = Path(f"/proc/{pid}/cwd")
    if not proc_cwd.exists():
        return True
    try:
        return proc_cwd.resolve() == Path(expected_cwd).resolve()
    except OSError:
        return False


def _is_stale_lock(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime > 10.0
    except OSError:
        return True


@contextmanager
def _job_marker_lock(thread_id: str) -> Iterator[None]:
    with _JOB_MARKER_THREAD_LOCKS_GUARD:
        thread_lock = _JOB_MARKER_THREAD_LOCKS.setdefault(thread_id, threading.Lock())
    with thread_lock:
        marker_path = _job_marker_path(thread_id)
        lock_path = marker_path.with_suffix(".json.lock")
        deadline = time.monotonic() + 5.0
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if _is_stale_lock(lock_path):
                    try:
                        lock_path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for HTTP training job marker lock: {thread_id}")
                time.sleep(0.01)
                continue
            break
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            yield
        finally:
            try:
                lock_path.unlink()
            except OSError:
                pass


def _job_marker_path(thread_id: str) -> Path:
    if not thread_id or any(part in thread_id for part in ("..", "/", "\\")):
        raise ValueError(f"Invalid thread_id: {thread_id}")
    root = (_runtime_dir() / HTTP_TRAINING_JOBS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / f"{thread_id}.json").resolve()
    if root not in target.parents:
        raise ValueError(f"Thread marker path escapes runtime root: {thread_id}")
    return target


def _runtime_dir() -> Path:
    return Path.cwd() / ".runtime"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
