from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import asyncio
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import require_admin_token
from app.core.edge_machine import build_edge_machine_info
from app.schemas import (
    EdgeRuntimeDeployRequest,
    EdgeRuntimeInstallRequest,
    EdgeRuntimeStartRequest,
    EdgeRuntimeStatusItem,
    EdgeRuntimeStopRequest,
    EdgeRuntimeWeightInstallRequest,
)

router = APIRouter(
    prefix="/api/edge/runtime",
    tags=["edge-runtime"],
    dependencies=[Depends(require_admin_token)],
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_root() -> Path:
    raw = os.getenv("JETLINKS_EDGE_RUNTIME_ROOT", ".runtime/edge-runtime")
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    return _runtime_root() / "state.json"


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.is_file():
        return {"items": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"items": {}}
    return data if isinstance(data, dict) else {"items": {}}


def _save_state(state: dict[str, Any]) -> None:
    _state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reconcile_state() -> dict[str, Any]:
    state = _load_state()
    changed: list[dict[str, Any]] = []
    for key, item in state.get("items", {}).items():
        normalized = _normalize_item(item).model_dump()
        manifest = ((item.get("metadata") or {}).get("manifest") or {})
        policy = (manifest.get("restartPolicy") or {}) if isinstance(manifest.get("restartPolicy"), dict) else {}
        auto_restart = bool(policy.get("enabled", False))
        max_restarts = int(policy.get("maxRetries", 3) or 3)
        if auto_restart and normalized["status"] in {"stopped", "degraded"} and int(item.get("restart_count") or 0) < max_restarts:
            item["restart_count"] = int(item.get("restart_count") or 0) + 1
            item = _start_process(item)
            state["items"][key] = item
            normalized = _normalize_item(item).model_dump()
        changed.append(normalized)
    _save_state(state)
    return {"items": changed, "total": len(changed)}


async def run_reconcile_loop() -> None:
    interval = max(3, int(os.getenv("JETLINKS_EDGE_RUNTIME_RECONCILE_SECONDS", "15") or 15))
    while True:
        try:
            reconcile_state()
        except Exception:
            pass
        await asyncio.sleep(interval)


def _runtime_key(machine_type: str, model_id: str, runtime_name: str) -> str:
    return f"{machine_type}::{model_id}::{runtime_name}"


def _bundle_dir(machine_type: str, model_id: str, runtime_name: str) -> Path:
    safe_model = model_id.replace("/", "__")
    return _runtime_root() / machine_type / safe_model / runtime_name




def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = _sha256_file(path)
    if actual.lower() != expected.strip().lower():
        raise HTTPException(status_code=400, detail=f"sha256 mismatch: expected {expected} actual {actual}")

def _download_bundle(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "jetlinks-agent-runtime-v2-edge-runtime/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def _safe_extract_tar(archive: tarfile.TarFile, target_dir: Path) -> None:
    root = target_dir.resolve()
    members = archive.getmembers()
    for member in members:
        relative = PurePosixPath(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe archive member path: {member.name}")
        destination = (root / Path(*relative.parts)).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"archive member escapes target directory: {member.name}") from exc
        if member.issym() or member.islnk():
            raise ValueError(f"archive links are not allowed: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported archive member type: {member.name}")
    archive.extractall(root, members=members)


def _manifest_path(bundle_dir: Path) -> Path:
    direct = bundle_dir / "manifest.json"
    if direct.is_file():
        return direct
    nested = list(bundle_dir.glob("*/manifest.json"))
    if nested:
        return nested[0]
    machine_bundle = list(bundle_dir.glob("*/machine-bundle-manifest.json"))
    if machine_bundle:
        return machine_bundle[0]
    return direct


def _port_from_args(args: list[str]) -> int | None:
    for index, value in enumerate(args):
        if value in {"-p", "--port"} and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except ValueError:
                return None
        if value.startswith("--port="):
            try:
                return int(value.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _manifest_from_machine_bundle(data: dict[str, Any]) -> dict[str, Any]:
    start = data.get("start") if isinstance(data.get("start"), dict) else {}
    command = str(start.get("model") or start.get("script") or "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="machine-bundle-manifest.json missing start.model or start.script")
    parts = shlex.split(command)
    if not parts:
        raise HTTPException(status_code=400, detail="machine-bundle-manifest.json start command is empty")
    port = _port_from_args(parts[1:])
    healthcheck: dict[str, Any] = {"type": "none"}
    if port:
        healthcheck = {"type": "http", "url": f"http://127.0.0.1:{port}/health", "timeoutSeconds": 3}
    return {
        "machineType": data.get("machineTypeId"),
        "modelId": data.get("modelId"),
        "runtimeName": data.get("module") or "default",
        "modelVersion": data.get("modelVersion"),
        "runtimeVersion": data.get("runtimeVersion"),
        "weightDir": data.get("weightDir"),
        "entry": parts[0],
        "args": parts[1:],
        "env": {},
        "port": port,
        "healthcheck": healthcheck,
        "sourceManifest": data,
    }


def _read_manifest(bundle_dir: Path) -> tuple[dict[str, Any], Path]:
    path = _manifest_path(bundle_dir)
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"manifest.json missing in bundle: {bundle_dir}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid manifest.json: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="manifest.json must be an object")
    if path.name == "machine-bundle-manifest.json":
        return _manifest_from_machine_bundle(data), path
    return data, path


def _process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _health_status(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    manifest = meta.get("manifest") or {}
    health = manifest.get("healthcheck") or {}
    if not _process_running(item.get("pid")):
        return "stopped"
    health_type = str(health.get("type") or "none")
    if health_type == "none":
        return "running"
    if health_type == "file":
        target = Path(str(health.get("path") or ""))
        if not target.is_absolute():
            target = Path(item["bundle_dir"]) / target
        return "healthy" if target.exists() else "unhealthy"
    if health_type == "http":
        url = str(health.get("url") or "").strip()
        if not url and item.get("port"):
            url = f"http://127.0.0.1:{item['port']}/health"
        if not url:
            return "unknown"
        try:
            with urllib.request.urlopen(url, timeout=float(health.get("timeoutSeconds") or 3)) as response:
                return "healthy" if 200 <= getattr(response, "status", 200) < 300 else "unhealthy"
        except Exception:
            return "unhealthy"
    return "unknown"


def _status_from_item(item: dict[str, Any]) -> str:
    if not _process_running(item.get("pid")):
        return item.get("status") if item.get("status") in {"installed", "failed"} else "stopped"
    health = _health_status(item)
    return "running" if health in {"running", "healthy", "unknown"} else "degraded"


def _normalize_item(item: dict[str, Any]) -> EdgeRuntimeStatusItem:
    normalized = {
        **item,
        "status": _status_from_item(item),
        "health_status": _health_status(item),
        "updated_at": _utc_now(),
    }
    return EdgeRuntimeStatusItem(**normalized)


def _terminate_pid(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        if not _process_running(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(int(pid), signal.SIGKILL)
    except OSError:
        pass


def _start_process(item: dict[str, Any], *, port: int | None = None, extra_env: dict[str, str] | None = None, extra_args: list[str] | None = None) -> dict[str, Any]:
    bundle_dir = Path(item["bundle_dir"])
    manifest, manifest_path = _read_manifest(bundle_dir)
    work_dir = manifest_path.parent
    entry = str(manifest.get("entry") or "./start.sh")
    command = [entry, *(manifest.get("args") or []), *(extra_args or [])]
    if port is not None:
        item["port"] = port
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (manifest.get("env") or {}).items()})
    env.update({str(k): str(v) for k, v in (extra_env or {}).items()})
    if item.get("port"):
        env.setdefault("APP_PORT", str(item["port"]))
    log_path = Path(item.get("log_path") or (bundle_dir / "runtime.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(command, cwd=work_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    item.update(
        {
            "status": "running",
            "pid": process.pid,
            "command": command,
            "manifest_path": str(manifest_path),
            "log_path": str(log_path),
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "health_status": "starting",
            "metadata": {**(item.get("metadata") or {}), "manifest": manifest},
        }
    )
    return item


def _resolve_machine_type(machine_type: str | None) -> str:
    if machine_type and machine_type.strip():
        return machine_type.strip()
    machine = build_edge_machine_info()
    if machine.machineTypeId:
        return machine.machineTypeId
    if machine.deviceTypes:
        return machine.deviceTypes[0]
    raise HTTPException(status_code=400, detail="machine_type is required when local machine type cannot be detected")


def _runtime_source(body: EdgeRuntimeDeployRequest) -> tuple[str | None, str | None, str | None]:
    runtime_url = body.runtime_url or body.runtime_bundle_url or body.bundle_url
    runtime_path = body.runtime_path or body.runtime_bundle_path or body.bundle_path
    runtime_sha256 = body.runtime_sha256 or body.runtime_bundle_sha256 or body.bundle_sha256
    return runtime_url, runtime_path, runtime_sha256


def _weight_request_has_deploy_fields(body: EdgeRuntimeWeightInstallRequest) -> bool:
    return any(
        [
            body.runtime_url,
            body.runtime_path,
            body.runtime_sha256,
            body.runtime_bundle_url,
            body.runtime_bundle_path,
            body.runtime_bundle_sha256,
            body.bundle_url,
            body.bundle_path,
            body.bundle_sha256,
            body.port is not None,
            body.env,
            body.args,
            not body.install_weight,
            body.auto_start,
            body.force_runtime,
        ]
    )


def _current_runtime_item(machine_type: str, model_id: str, runtime_name: str) -> dict[str, Any] | None:
    state = _load_state()
    key = _runtime_key(machine_type, model_id, runtime_name)
    item = state.setdefault("items", {}).get(key)
    return item if isinstance(item, dict) else None


@router.post("/deploy")
def deploy_runtime(body: EdgeRuntimeDeployRequest) -> dict[str, Any]:
    machine_type = _resolve_machine_type(body.machine_type)
    runtime_url, runtime_path, runtime_sha256 = _runtime_source(body)
    steps: list[dict[str, Any]] = []
    item = _current_runtime_item(machine_type, body.model_id, body.runtime_name)

    if item and body.force_runtime:
        _terminate_pid(item.get("pid"))
        steps.append({"name": "runtime.stop", "status": "completed"})
        item = None

    if not item:
        if not runtime_url and not runtime_path:
            raise HTTPException(status_code=400, detail="runtime is not installed; runtime_url or runtime_path is required")
        install_runtime_bundle(
            EdgeRuntimeInstallRequest(
                bundle_url=runtime_url,
                bundle_path=runtime_path,
                bundle_sha256=runtime_sha256,
                machine_type=machine_type,
                model_id=body.model_id,
                runtime_name=body.runtime_name,
                force=body.force_runtime,
            )
        )
        steps.append({"name": "runtime.install", "status": "completed"})
    else:
        steps.append({"name": "runtime.install", "status": "skipped", "reason": "already_installed"})

    if body.install_weight:
        if not body.weight_url and not body.weight_path:
            raise HTTPException(status_code=400, detail="weight_url or weight_path is required when install_weight=true")
        weight_result = _install_runtime_weights(
            EdgeRuntimeWeightInstallRequest(
                machine_type=machine_type,
                model_id=body.model_id,
                runtime_name=body.runtime_name,
                version=body.version,
                weight_url=body.weight_url,
                weight_path=body.weight_path,
                weight_sha256=body.weight_sha256,
                target_subdir=body.target_subdir,
            )
        )
        steps.append({"name": "weight.install", "status": "completed", "weight": weight_result.get("weight")})
    else:
        steps.append({"name": "weight.install", "status": "skipped", "reason": "install_weight=false"})

    item = _current_runtime_item(machine_type, body.model_id, body.runtime_name)
    if not item:
        raise HTTPException(status_code=500, detail="runtime item missing after install")

    if body.auto_start:
        if _process_running(item.get("pid")):
            steps.append({"name": "runtime.start", "status": "skipped", "reason": "already_running"})
        else:
            item = _start_process(item, port=body.port, extra_env=body.env, extra_args=body.args)
            state = _load_state()
            key = _runtime_key(machine_type, body.model_id, body.runtime_name)
            state.setdefault("items", {})[key] = item
            _save_state(state)
            steps.append({"name": "runtime.start", "status": "completed"})
    else:
        steps.append({"name": "runtime.start", "status": "skipped", "reason": "auto_start=false"})

    item = _current_runtime_item(machine_type, body.model_id, body.runtime_name)
    if not item:
        raise HTTPException(status_code=500, detail="runtime item missing after deploy")
    return {
        "ok": True,
        "machine_type": machine_type,
        "model_id": body.model_id,
        "runtime_name": body.runtime_name,
        "version": body.version,
        "steps": steps,
        "item": _normalize_item(item).model_dump(),
        "machine": build_edge_machine_info().model_dump(),
    }


@router.post("/install")
def install_runtime_bundle(body: EdgeRuntimeInstallRequest) -> dict[str, Any]:
    if not body.bundle_url and not body.bundle_path:
        raise HTTPException(status_code=400, detail="bundle_url or bundle_path is required")
    bundle_dir = _bundle_dir(body.machine_type, body.model_id, body.runtime_name)
    if bundle_dir.exists():
        if body.force:
            shutil.rmtree(bundle_dir)
        elif any(bundle_dir.iterdir()):
            raise HTTPException(status_code=409, detail="runtime bundle already exists; set force=true to replace it")
    temporary = tempfile.NamedTemporaryFile(
        prefix=".edge-runtime-bundle-",
        suffix=".tar.gz",
        dir=_runtime_root(),
        delete=False,
    )
    archive_path = Path(temporary.name)
    temporary.close()
    try:
        if body.bundle_path:
            source = Path(body.bundle_path).expanduser().resolve()
            if not source.is_file():
                raise HTTPException(status_code=404, detail=f"bundle_path not found: {source}")
            shutil.copyfile(source, archive_path)
        else:
            _download_bundle(str(body.bundle_url), archive_path)
        _verify_sha256(archive_path, body.bundle_sha256)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, bundle_dir)
    except (tarfile.TarError, ValueError, OSError) as exc:
        shutil.rmtree(bundle_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"failed to extract bundle: {exc}") from exc
    finally:
        archive_path.unlink(missing_ok=True)
    manifest, manifest_path = _read_manifest(bundle_dir)
    state = _load_state()
    key = _runtime_key(body.machine_type, body.model_id, body.runtime_name)
    item = {
        "machine_type": body.machine_type,
        "model_id": body.model_id,
        "runtime_name": body.runtime_name,
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "status": "installed",
        "pid": None,
        "port": manifest.get("port"),
        "command": [],
        "log_path": str(bundle_dir / "runtime.log"),
        "installed_at": _utc_now(),
        "started_at": None,
        "updated_at": _utc_now(),
        "health_status": "unknown",
        "restart_count": 0,
        "metadata": {"manifest": manifest},
    }
    state.setdefault("items", {})[key] = item
    _save_state(state)
    return {"ok": True, "item": _normalize_item(item).model_dump()}


@router.post("/start")
def start_runtime_bundle(body: EdgeRuntimeStartRequest) -> dict[str, Any]:
    state = _load_state()
    key = _runtime_key(body.machine_type, body.model_id, body.runtime_name)
    item = state.setdefault("items", {}).get(key)
    if not item:
        raise HTTPException(status_code=404, detail="runtime bundle not installed")
    if _process_running(item.get("pid")):
        raise HTTPException(status_code=409, detail="runtime already running")
    item = _start_process(item, port=body.port, extra_env=body.env, extra_args=body.args)
    state["items"][key] = item
    _save_state(state)
    return {"ok": True, "item": _normalize_item(item).model_dump()}


@router.post("/stop")
def stop_runtime_bundle(body: EdgeRuntimeStopRequest) -> dict[str, Any]:
    state = _load_state()
    key = _runtime_key(body.machine_type, body.model_id, body.runtime_name)
    item = state.setdefault("items", {}).get(key)
    if not item:
        raise HTTPException(status_code=404, detail="runtime bundle not installed")
    _terminate_pid(item.get("pid"))
    item.update({"status": "stopped", "pid": None, "health_status": "stopped", "updated_at": _utc_now()})
    state["items"][key] = item
    _save_state(state)
    return {"ok": True, "item": _normalize_item(item).model_dump()}


@router.post("/restart")
def restart_runtime_bundle(body: EdgeRuntimeStartRequest) -> dict[str, Any]:
    state = _load_state()
    key = _runtime_key(body.machine_type, body.model_id, body.runtime_name)
    item = state.setdefault("items", {}).get(key)
    if not item:
        raise HTTPException(status_code=404, detail="runtime bundle not installed")
    _terminate_pid(item.get("pid"))
    item["pid"] = None
    item["restart_count"] = int(item.get("restart_count") or 0) + 1
    item = _start_process(item, port=body.port, extra_env=body.env, extra_args=body.args)
    state["items"][key] = item
    _save_state(state)
    return {"ok": True, "item": _normalize_item(item).model_dump()}


@router.post("/reconcile")
def reconcile_runtime_bundles() -> dict[str, Any]:
    return reconcile_state()


@router.get("/status")
def list_runtime_bundles() -> dict[str, Any]:
    state = _load_state()
    items = [_normalize_item(item).model_dump() for item in state.get("items", {}).values()]
    return {"items": items, "total": len(items)}


@router.get("/logs")
def runtime_bundle_logs(
    machine_type: str = Query(...),
    model_id: str = Query(...),
    runtime_name: str = Query("default"),
    tail: int = Query(200, ge=1, le=2000),
) -> dict[str, Any]:
    state = _load_state()
    key = _runtime_key(machine_type, model_id, runtime_name)
    item = state.setdefault("items", {}).get(key)
    if not item:
        raise HTTPException(status_code=404, detail="runtime bundle not installed")
    path = Path(str(item.get("log_path") or ""))
    if not path.is_file():
        return {"lines": [], "detail": "log file not found"}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {"lines": lines[-tail:]}


@router.post("/weights/install")
def install_runtime_weights(body: EdgeRuntimeWeightInstallRequest) -> dict[str, Any]:
    machine_type = _resolve_machine_type(body.machine_type)
    body = body.model_copy(update={"machine_type": machine_type})
    if _weight_request_has_deploy_fields(body):
        return deploy_runtime(
            EdgeRuntimeDeployRequest(
                machine_type=machine_type,
                model_id=body.model_id,
                runtime_name=body.runtime_name,
                version=body.version,
                runtime_url=body.runtime_url,
                runtime_path=body.runtime_path,
                runtime_sha256=body.runtime_sha256,
                runtime_bundle_url=body.runtime_bundle_url,
                runtime_bundle_path=body.runtime_bundle_path,
                runtime_bundle_sha256=body.runtime_bundle_sha256,
                bundle_url=body.bundle_url,
                bundle_path=body.bundle_path,
                bundle_sha256=body.bundle_sha256,
                weight_url=body.weight_url,
                weight_path=body.weight_path,
                weight_sha256=body.weight_sha256,
                target_subdir=body.target_subdir,
                port=body.port,
                env=body.env,
                args=body.args,
                install_weight=body.install_weight,
                auto_start=body.auto_start,
                force_runtime=body.force_runtime,
            )
        )
    return _install_runtime_weights(body)


def _install_runtime_weights(body: EdgeRuntimeWeightInstallRequest) -> dict[str, Any]:
    if not body.weight_url and not body.weight_path:
        raise HTTPException(status_code=400, detail="weight_url or weight_path is required")
    if not body.machine_type:
        raise HTTPException(status_code=400, detail="machine_type is required")
    state = _load_state()
    key = _runtime_key(body.machine_type, body.model_id, body.runtime_name)
    item = state.setdefault("items", {}).get(key)
    if not item:
        raise HTTPException(status_code=404, detail="runtime bundle not installed")
    bundle_dir = Path(item["bundle_dir"])
    target_dir = bundle_dir / body.target_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(body.weight_path or body.weight_url or "weights.bin").name
    target = target_dir / filename
    if body.weight_path:
        source = Path(body.weight_path).expanduser().resolve()
        if not source.is_file():
            raise HTTPException(status_code=404, detail=f"weight_path not found: {source}")
        shutil.copyfile(source, target)
    else:
        _download_bundle(str(body.weight_url), target)
    _verify_sha256(target, body.weight_sha256)
    metadata = item.setdefault("metadata", {})
    weights = metadata.setdefault("weights", [])
    weights.append({"path": str(target), "installed_at": _utc_now(), "sha256": body.weight_sha256, "version": body.version})
    item["updated_at"] = _utc_now()
    state["items"][key] = item
    _save_state(state)
    return {"ok": True, "item": _normalize_item(item).model_dump(), "weight": str(target)}
