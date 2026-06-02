from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.skills.registry import SkillDefinition


_REFERENCE_PATH_PATTERN = re.compile(
    r"(?P<path>references/[A-Za-z0-9][A-Za-z0-9_.\-/]*\.(?:md|json|txt|yaml|yml))"
)


@dataclass(frozen=True)
class SkillContextFile:
    path: str
    content: str
    truncated: bool


@dataclass(frozen=True)
class SkillMarkdownContext:
    skill_md_path: str
    skill_md: str
    skill_md_truncated: bool
    references: tuple[SkillContextFile, ...]


def load_skill_markdown_context(
    skill: SkillDefinition,
    *,
    skill_md_max_chars: int = 30_000,
    reference_max_chars: int = 40_000,
    total_reference_max_chars: int = 120_000,
) -> SkillMarkdownContext | None:
    """Load package-level SKILL.md context and declared local reference files."""

    skill_md_path = _find_skill_md_path(skill)
    if skill_md_path is None:
        return None
    package_root = skill_md_path.parent.resolve()
    try:
        skill_md_full = skill_md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    skill_md, skill_md_truncated = _truncate(skill_md_full, skill_md_max_chars)
    reference_paths = _declared_reference_paths(skill_md_full)
    references: list[SkillContextFile] = []
    remaining = max(total_reference_max_chars, 0)
    for reference_path in reference_paths:
        if remaining <= 0:
            break
        path = (package_root / reference_path).resolve()
        if not _is_inside(path, package_root) or not path.is_file():
            continue
        try:
            content_full = path.read_text(encoding="utf-8")
        except OSError:
            continue
        limit = min(reference_max_chars, remaining)
        content, truncated = _truncate(content_full, limit)
        remaining -= len(content)
        references.append(SkillContextFile(path=reference_path, content=content, truncated=truncated))
    return SkillMarkdownContext(
        skill_md_path=str(skill_md_path),
        skill_md=skill_md,
        skill_md_truncated=skill_md_truncated,
        references=tuple(references),
    )


def _find_skill_md_path(skill: SkillDefinition) -> Path | None:
    candidates: list[Path] = []
    if skill.manifest_path is not None:
        if skill.manifest_path.name == "SKILL.md":
            candidates.append(skill.manifest_path)
        candidates.append(skill.manifest_path.parent / "SKILL.md")
    if skill.plugin_root is not None:
        plugin_root = skill.plugin_root
        candidates.append(plugin_root / "skills" / skill.name / "SKILL.md")
        candidates.append(plugin_root / "SKILL.md")
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            return resolved
    return None


def _declared_reference_paths(skill_md: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in _REFERENCE_PATH_PATTERN.finditer(skill_md):
        path = match.group("path").strip()
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
