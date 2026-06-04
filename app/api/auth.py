from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from fastapi import Header, HTTPException, Query, WebSocket, WebSocketException, status


RuntimeAuthMode = Literal["development", "production"]


@dataclass(frozen=True)
class RuntimeAuthSettings:
    """Authentication settings for runtime and admin surfaces."""

    mode: RuntimeAuthMode
    token: str

    @property
    def token_required(self) -> bool:
        return self.mode == "production"


def runtime_auth_settings() -> RuntimeAuthSettings:
    """Read runtime authentication settings from environment variables.

    Development mode keeps local workbench and tests frictionless. Production
    mode requires ``RUNTIME_API_TOKEN`` on every non-health runtime operation.
    """

    raw_mode = os.getenv("RUNTIME_AUTH_MODE", os.getenv("JETLINKS_RUNTIME_MODE", "development"))
    normalized: RuntimeAuthMode = "production" if raw_mode.strip().lower() in {"prod", "production"} else "development"
    return RuntimeAuthSettings(mode=normalized, token=os.getenv("RUNTIME_API_TOKEN", "").strip())


def require_runtime_token(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """Require bearer token only when runtime auth is in production mode."""

    settings = runtime_auth_settings()
    if not settings.token_required:
        return
    if _token_valid(settings.token, authorization=authorization, query_token=token):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid runtime token.")


def require_admin_token(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """Require admin token whenever ``RUNTIME_API_TOKEN`` is configured.

    Admin write endpoints predate ``RUNTIME_AUTH_MODE`` and intentionally keep
    their lightweight safety behavior: setting ``RUNTIME_API_TOKEN`` protects
    mutable management surfaces even during local/development mode.
    """

    expected_token = os.getenv("RUNTIME_API_TOKEN", "").strip()
    if not expected_token:
        return
    if _token_valid(expected_token, authorization=authorization, query_token=token):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid runtime admin token.")


async def require_websocket_runtime_token(websocket: WebSocket, token: str | None = None) -> None:
    """Validate runtime token for ACP WebSocket connections in production mode."""

    settings = runtime_auth_settings()
    if not settings.token_required:
        return
    authorization = websocket.headers.get("authorization")
    if _token_valid(settings.token, authorization=authorization, query_token=token):
        return
    raise WebSocketException(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Missing or invalid runtime token.",
    )


def _token_valid(expected_token: str, *, authorization: str | None, query_token: str | None) -> bool:
    if not expected_token:
        return False
    return authorization == f"Bearer {expected_token}" or query_token == expected_token
