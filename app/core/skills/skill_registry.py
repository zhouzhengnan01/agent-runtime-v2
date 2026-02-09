from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillInfo:
    name: str
    description: str
    location: str
    tags: List[str] = field(default_factory=list)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _split_paths(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _normalize_path(raw: str) -> Path:
    expanded = os.path.expanduser(raw)
    path = Path(expanded)
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in {"'", '"'}):
        return value[1:-1]
    return value


def _parse_frontmatter_lines(lines: Iterable[str]) -> Dict[str, object]:
    data: Dict[str, object] = {}
    current_key: Optional[str] = None
    list_values: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        if current_key:
            if re.match(r"^\s*-\s+", line):
                item = re.sub(r"^\s*-\s+", "", line).strip()
                if item:
                    list_values.append(_strip_quotes(item))
                continue
            data[current_key] = list_values
            current_key = None
            list_values = []

        match = re.match(r"^\s*([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
        if not match:
            continue

        key = match.group(1)
        value = match.group(2).strip()

        if value == "":
            current_key = key
            list_values = []
            continue

        if value.startswith("[") and value.endswith("]"):
            items = [
                _strip_quotes(item.strip())
                for item in value[1:-1].split(",")
                if item.strip()
            ]
            data[key] = items
            continue

        if "," in value and key.lower() in {"tags", "tag"}:
            items = [
                _strip_quotes(item.strip())
                for item in value.split(",")
                if item.strip()
            ]
            data[key] = items
            continue

        data[key] = _strip_quotes(value)

    if current_key:
        data[current_key] = list_values

    return data


def _read_skill_file(path: Path) -> Optional[Tuple[Dict[str, object], str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Skill read failed: %s (%s)", path, exc)
        return None

    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    front_lines: List[str] = []
    body_lines: List[str] = []
    in_frontmatter = True

    for line in lines[1:]:
        if in_frontmatter and line.strip() in {"---", "..."}:
            in_frontmatter = False
            continue
        if in_frontmatter:
            front_lines.append(line)
        else:
            body_lines.append(line)

    meta = _parse_frontmatter_lines(front_lines)
    content = "\n".join(body_lines).strip()
    return meta, content


class SkillRegistry:
    def __init__(self) -> None:
        self._cache: Dict[str, SkillInfo] = {}
        self._cache_at = 0.0

    def _enabled(self) -> bool:
        return bool(getattr(settings, "SKILL_ENABLED", True))

    def _cache_ttl(self) -> int:
        try:
            return int(getattr(settings, "SKILL_CACHE_TTL", 60))
        except Exception:
            return 60

    def _max_items(self) -> int:
        try:
            return int(getattr(settings, "SKILL_MAX_ITEMS", 50))
        except Exception:
            return 50

    def _paths(self) -> List[Path]:
        raw = getattr(settings, "SKILL_PATHS", None)
        paths = _split_paths(raw)
        if not paths:
            paths = ["skills"]
        resolved: List[Path] = []
        for path in paths:
            try:
                resolved.append(_normalize_path(path))
            except Exception:
                continue
        unique: List[Path] = []
        for path in resolved:
            if path not in unique:
                unique.append(path)
        return unique

    def _scan_paths(self) -> Dict[str, SkillInfo]:
        skills: Dict[str, SkillInfo] = {}

        for root in self._paths():
            if root.is_file() and root.name.upper() == "SKILL.md":
                self._load_skill_file(root, skills)
                continue
            if not root.exists():
                continue
            for path in root.rglob("SKILL.md"):
                self._load_skill_file(path, skills)

        return skills

    def _load_skill_file(self, path: Path, skills: Dict[str, SkillInfo]) -> None:
        parsed = _read_skill_file(path)
        if not parsed:
            return
        meta, _ = parsed
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        if not name or not description:
            return
        tags_raw = meta.get("tags") or meta.get("tag") or []
        tags: List[str] = []
        if isinstance(tags_raw, list):
            tags = [str(item).strip() for item in tags_raw if str(item).strip()]
        elif isinstance(tags_raw, str):
            tags = [part.strip() for part in tags_raw.split(",") if part.strip()]

        if name in skills:
            logger.warning("Duplicate skill name detected: %s (%s)", name, path)
            return

        skills[name] = SkillInfo(
            name=name,
            description=description,
            location=str(path),
            tags=tags,
        )

    def _load(self) -> Dict[str, SkillInfo]:
        if not self._enabled():
            return {}

        ttl = self._cache_ttl()
        now = time.monotonic()
        if ttl > 0 and self._cache and (now - self._cache_at) < ttl:
            return self._cache

        skills = self._scan_paths()
        self._cache = skills
        self._cache_at = now
        return skills

    def list_skills(self) -> List[SkillInfo]:
        skills = list(self._load().values())
        skills.sort(key=lambda item: item.name.lower())
        limit = self._max_items()
        if limit > 0:
            skills = skills[:limit]
        return skills

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        if not name:
            return None
        return self._load().get(name)

    def read_skill_content(self, skill: SkillInfo) -> str:
        if not skill:
            return ""
        path = Path(skill.location)
        parsed = _read_skill_file(path)
        if not parsed:
            return ""
        _, content = parsed
        return content


_SKILL_REGISTRY: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _SKILL_REGISTRY
    if _SKILL_REGISTRY is None:
        _SKILL_REGISTRY = SkillRegistry()
    return _SKILL_REGISTRY
