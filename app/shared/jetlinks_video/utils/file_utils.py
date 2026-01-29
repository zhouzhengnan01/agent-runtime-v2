from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def read_env_kv(path: Union[str, Path], key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read a key from a dotenv-style file (no os.environ injection).

    Supports:
    - `KEY=value`
    - `export KEY=value`
    - surrounding quotes: `KEY="value"` / `KEY='value'`
    """
    k_target = (key or "").strip()
    if not k_target:
        return default

    p = Path(path)
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return default
    except Exception:
        return default

    for line in lines:
        s = (line or "").strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        k, v = s.split("=", 1)
        if (k or "").strip() != k_target:
            continue
        val = (v or "").strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        return val if val else default

    return default
