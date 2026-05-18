from __future__ import annotations

from dataclasses import dataclass, field
from asyncio.subprocess import Process
from typing import Any


JsonRpcId = str | int | None


@dataclass
class AcpTerminal:
    process: Process
    output_limit: int
    output: bytearray = field(default_factory=bytearray)
    truncated: bool = False
    reader_task: Any | None = None


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
    mode_id: str = "edit"
    config_options: dict[str, Any] = field(default_factory=dict)
    terminals: dict[str, AcpTerminal] = field(default_factory=dict)
