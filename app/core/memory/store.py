from __future__ import annotations

import builtins
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

MemoryScope = Literal["agent", "global"]

_SAFE_KEY_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class MemoryItem:
    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """Small local JSON memory store scoped by agent or globally."""

    def __init__(self, root_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.root_dir = root_dir or project_root / ".runtime" / "memory"

    def remember(
        self,
        agent_name: str,
        text: str,
        *,
        tags: builtins.list[str] | None = None,
        scope: MemoryScope = "agent",
    ) -> MemoryItem:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("memory text is required")
        clean_tags = self._clean_tags(tags or [])
        now = self._now()
        item = MemoryItem(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            text=clean_text,
            tags=clean_tags,
            created_at=now,
            updated_at=now,
        )
        items = self._load_items(agent_name, scope)
        items.append(item)
        self._save_items(agent_name, scope, items)
        return item

    def list(
        self,
        agent_name: str,
        *,
        query: str | None = None,
        tags: builtins.list[str] | None = None,
        limit: int = 20,
        scope: MemoryScope = "agent",
    ) -> builtins.list[MemoryItem]:
        items: builtins.list[MemoryItem] = builtins.list(reversed(self._load_items(agent_name, scope)))
        clean_query = (query or "").strip().lower()
        clean_tags: set[str] = set(self._clean_tags(tags or []))
        if clean_query:
            items = [
                item
                for item in items
                if clean_query in item.text.lower()
                or any(clean_query in tag.lower() for tag in item.tags)
            ]
        if clean_tags:
            items = [item for item in items if clean_tags.intersection(item.tags)]
        return items[: self._bounded_limit(limit)]

    def forget(self, agent_name: str, memory_id: str, *, scope: MemoryScope = "agent") -> bool:
        clean_id = memory_id.strip()
        if not clean_id:
            raise ValueError("memory id is required")
        items = self._load_items(agent_name, scope)
        remaining = [item for item in items if item.id != clean_id]
        removed = len(remaining) != len(items)
        if removed:
            self._save_items(agent_name, scope, remaining)
        return removed

    def clear(self, agent_name: str, *, scope: MemoryScope = "agent") -> int:
        items = self._load_items(agent_name, scope)
        self._save_items(agent_name, scope, [])
        return len(items)

    def _load_items(self, agent_name: str, scope: MemoryScope) -> builtins.list[MemoryItem]:
        path = self._path(agent_name, scope)
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_items = data.get("memories") if isinstance(data, dict) else []
        if not isinstance(raw_items, list):
            return []
        items: builtins.list[MemoryItem] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = self._item_from_payload(raw_item)
            if item is not None:
                items.append(item)
        return items

    def _save_items(self, agent_name: str, scope: MemoryScope, items: builtins.list[MemoryItem]) -> None:
        path = self._path(agent_name, scope)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": scope,
            "agent": self._store_key(agent_name, scope),
            "memories": [item.to_payload() for item in items],
        }
        target = path.with_suffix(".json.tmp")
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(path)

    def _path(self, agent_name: str, scope: MemoryScope) -> Path:
        return self.root_dir / f"{self._store_key(agent_name, scope)}.json"

    @classmethod
    def _store_key(cls, agent_name: str, scope: MemoryScope) -> str:
        if scope == "global":
            return "global"
        cleaned = _SAFE_KEY_RE.sub("-", agent_name.strip()).strip(".-")
        return cleaned[:96] or "default"

    @staticmethod
    def _item_from_payload(payload: dict[str, Any]) -> MemoryItem | None:
        raw_id = payload.get("id")
        raw_text = payload.get("text")
        if not isinstance(raw_id, str) or not raw_id.strip() or not isinstance(raw_text, str) or not raw_text.strip():
            return None
        raw_tags = payload.get("tags")
        tags = MemoryStore._clean_tags(raw_tags if isinstance(raw_tags, list) else [])
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        return MemoryItem(
            id=raw_id,
            text=raw_text,
            tags=tags,
            created_at=created_at if isinstance(created_at, str) else "",
            updated_at=updated_at if isinstance(updated_at, str) else "",
        )

    @staticmethod
    def _clean_tags(tags: builtins.list[Any]) -> builtins.list[str]:
        cleaned: builtins.list[str] = []
        seen: set[str] = set()
        for tag in tags:
            value = str(tag).strip()
            if not value or value in seen:
                continue
            cleaned.append(value[:64])
            seen.add(value)
        return cleaned[:20]

    @staticmethod
    def _bounded_limit(value: int) -> int:
        return max(1, min(100, int(value)))

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
