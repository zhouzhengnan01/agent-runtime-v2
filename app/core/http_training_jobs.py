from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ACTIVE_HTTP_TRAINING_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_HTTP_TRAINING_STATUSES = {"completed", "failed", "cancelled"}
SERVICE_INSTANCE_ENV = "JETLINKS_SERVICE_INSTANCE_ID"
SERVICE_INSTANCE_NAME = "service_instance.json"
SERVICE_INSTANCE_LOCK_NAME = "service_instance.lock"
HTTP_TRAINING_JOBS_DIR = "http_training_jobs"

_SERVICE_INSTANCE_ID: str | None = None


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


def write_http_training_job_marker(thread_id: str, record: dict[str, Any]) -> None:
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
        "updated_at": utc_now(),
    }
    _write_json(_job_marker_path(thread_id), payload)


def update_http_training_job_marker(thread_id: str, **updates: Any) -> None:
    path = _job_marker_path(thread_id)
    payload = _read_json(path)
    if not payload:
        payload = {
            "schema": "jetlinks-http-training-job.v1",
            "service_instance_id": current_service_instance_id(),
            "thread_id": thread_id,
            "job_id": f"thread-{thread_id}",
            "created_at": utc_now(),
        }
    payload.update({key: value for key, value in updates.items() if value is not None})
    payload["service_instance_id"] = payload.get("service_instance_id") or current_service_instance_id()
    payload["thread_id"] = thread_id
    payload["updated_at"] = utc_now()
    _write_json(path, payload)


def read_current_http_training_job_marker(thread_id: str, *, require_active: bool = True) -> dict[str, Any] | None:
    payload = _read_json(_job_marker_path(thread_id))
    if not payload:
        return None
    if payload.get("service_instance_id") != current_service_instance_id():
        return None
    status = str(payload.get("status") or "")
    if require_active and status not in ACTIVE_HTTP_TRAINING_STATUSES:
        return None
    return payload


def is_current_http_training_job_active(thread_id: str) -> bool:
    return read_current_http_training_job_marker(thread_id, require_active=True) is not None


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
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
