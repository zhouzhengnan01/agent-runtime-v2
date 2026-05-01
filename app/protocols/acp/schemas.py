from __future__ import annotations

from dataclasses import dataclass
from typing import Any


JsonRpcId = str | int | None


@dataclass
class AcpWebSocketSession:
    session_id: str
    thread_id: str
    agent_name: str
    cwd: str
    backend_type: str = "local"
    backend_session_id: str | None = None
    backend: Any | None = None

