from __future__ import annotations

from dataclasses import dataclass, field
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
    model_id: str | None = None
    model_name: str | None = None
    app_template_name: str | None = None
    runtime_options: dict[str, Any] = field(default_factory=dict)
