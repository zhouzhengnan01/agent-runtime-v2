from __future__ import annotations

import os

from fastapi import Header, HTTPException


def require_admin_token(authorization: str | None = Header(default=None)) -> None:
    token = os.getenv("RUNTIME_API_TOKEN", "").strip()
    if not token:
        return
    if authorization == f"Bearer {token}":
        return
    raise HTTPException(status_code=401, detail="Missing or invalid runtime admin token.")
