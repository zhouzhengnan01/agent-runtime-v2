"""
Session Markdown logger.

Stores per-session conversation and tool history into a markdown file.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict


class SessionMarkdownLogger:
    """Append-only markdown logger per session."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_")
        return self.base_dir / f"{safe_id}.md"

    def _ensure_header(self, path: Path, session_id: str, agent_id: Optional[str]) -> None:
        if path.exists():
            return
        header_lines = [
            "---",
            f"session_id: {session_id}",
            f"agent_id: {agent_id or ''}",
            f"started_at: {datetime.now().isoformat()}",
            "---",
            "",
        ]
        path.write_text("\n".join(header_lines), encoding="utf-8")

    def append(
        self,
        session_id: str,
        title: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        path = self._session_path(session_id)
        self._ensure_header(path, session_id, agent_id)

        timestamp = datetime.now().isoformat()
        lines = [f"\n## {title} ({timestamp})", ""]

        if meta:
            lines.append("```json")
            lines.append(json.dumps(meta, ensure_ascii=True, indent=2))
            lines.append("```")
            lines.append("")

        lines.append(content or "")

        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")

