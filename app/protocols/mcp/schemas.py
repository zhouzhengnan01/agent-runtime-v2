from __future__ import annotations

from dataclasses import dataclass


MCP_PROTOCOL_VERSION = "2025-06-18"


@dataclass(frozen=True)
class McpRequestHeaders:
    method: str | None = None
    name: str | None = None

