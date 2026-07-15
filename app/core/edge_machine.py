from __future__ import annotations

import os
import platform
import socket
from pathlib import Path
from typing import Any


def _split_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default) or ""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _runtime_root() -> Path:
    return Path(os.getenv("JETLINKS_EDGE_RUNTIME_ROOT", ".runtime/edge-runtime")).expanduser().resolve()


def _state_items(runtime_root: Path) -> list[dict[str, Any]]:
    state_path = runtime_root / "state.json"
    if not state_path.is_file():
        return []
    try:
        import json

        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("items") if isinstance(data, dict) else None
    if isinstance(items, dict):
        return [item for item in items.values() if isinstance(item, dict)]
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def _process_running(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except Exception:
        return False


def build_edge_machine_info() -> dict[str, Any]:
    runtime_root = _runtime_root()
    runtime_root.mkdir(parents=True, exist_ok=True)
    items = _state_items(runtime_root)
    running = sum(1 for item in items if _process_running(item.get("pid")))
    machine_type = os.getenv("JETLINKS_EDGE_MACHINE_TYPE", "edge-qualcomm-qcs8550")
    device_types = _split_env("JETLINKS_EDGE_DEVICE_TYPES", "qualcomm")
    labels = _split_env("JETLINKS_EDGE_LABELS", ",".join(device_types))
    return {
        "nodeId": os.getenv("JETLINKS_EDGE_NODE_ID", socket.gethostname()),
        "nodeName": os.getenv("JETLINKS_EDGE_NODE_NAME", socket.gethostname()),
        "nodeRole": "edge",
        "deviceTypes": device_types,
        "deviceModel": os.getenv("JETLINKS_EDGE_DEVICE_MODEL", "QCS8550"),
        "machineTypeId": machine_type,
        "machineTypeIds": _split_env("JETLINKS_EDGE_MACHINE_TYPES", machine_type),
        "labels": labels,
        "managerVersion": os.getenv("JETLINKS_AGENT_VERSION", "0.1.0"),
        "runtimeGitCommit": os.getenv("JETLINKS_AGENT_GIT_COMMIT"),
        "serviceRoot": str(Path.cwd()),
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "capabilities": {
            "runtimeDeploy": True,
            "weightInstall": True,
            "runtimeStartStop": True,
            "machineDiscovery": True,
        },
        "agentRuntime": {
            "runtimeRoot": str(runtime_root),
            "installedRuntimes": len(items),
            "runningRuntimes": running,
            "runtimes": items,
        },
        "cloudQuery": {
            "role": "edge",
            "device": device_types[0] if device_types else "qualcomm",
            "deviceModel": os.getenv("JETLINKS_EDGE_DEVICE_MODEL", "QCS8550"),
            "machineTypeId": machine_type,
        },
    }
