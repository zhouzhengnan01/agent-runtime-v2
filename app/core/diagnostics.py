from __future__ import annotations

import json
import os
from typing import Any


def env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int = 0) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def diagnostic_json(value: Any, *, max_chars: int = 0) -> str:
    serialized = json.dumps(
        redact_diagnostic_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    if max_chars > 0 and len(serialized) > max_chars:
        return f"{serialized[:max_chars]}...<truncated chars={len(serialized)}>"
    return serialized


def redact_diagnostic_value(value: Any, *, key: str = "") -> Any:
    if diagnostic_key_is_sensitive(key):
        return "********"
    if isinstance(value, dict):
        return {str(item_key): redact_diagnostic_value(item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_diagnostic_value(item) for item in value]
    return value


def diagnostic_key_is_sensitive(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if normalized in {
        "api_key",
        "api_key_enc",
        "apikey",
        "authorization",
        "cookie",
        "set_cookie",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "id_token",
        "bearer_token",
        "runtime_token",
        "auth_token",
    }:
        return True
    if normalized.endswith(("_api_key", "_secret", "_password", "_token")):
        return True
    return normalized.startswith(("authorization_", "x_api_key"))
