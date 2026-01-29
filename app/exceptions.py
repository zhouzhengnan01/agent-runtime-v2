from __future__ import annotations


class DatabaseUnavailableError(RuntimeError):
    """Raised when DB is temporarily unavailable (pool exhausted/network issues)."""

