from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

MarkdownMemoryScope = Literal["global", "user", "project", "session"]

_SAFE_KEY_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class MarkdownMemoryContext:
    agent_name: str
    user_id: str | None = None
    project_id: str | None = None
    thread_id: str | None = None


@dataclass(frozen=True)
class MarkdownMemoryFile:
    scope: MarkdownMemoryScope
    path: str
    size: int
    updated_at: str

    def to_payload(self) -> dict[str, str | int]:
        return {
            "scope": self.scope,
            "path": self.path,
            "size": self.size,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MarkdownMemorySearchMatch:
    scope: MarkdownMemoryScope
    path: str
    line: int
    text: str

    def to_payload(self) -> dict[str, str | int]:
        return {
            "scope": self.scope,
            "path": self.path,
            "line": self.line,
            "text": self.text,
        }


@dataclass(frozen=True)
class MarkdownMemoryCompression:
    scope: MarkdownMemoryScope
    source_path: str
    target_path: str
    original_chars: int
    compressed_chars: int
    truncated: bool
    keywords: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, str | int | bool | list[str]]:
        return {
            "scope": self.scope,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "original_chars": self.original_chars,
            "compressed_chars": self.compressed_chars,
            "truncated": self.truncated,
            "keywords": self.keywords,
        }


class MarkdownMemoryStore:
    """Human-readable Markdown memory store with user/project/session isolation."""

    def __init__(self, root_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.root_dir = root_dir or project_root / ".runtime" / "memory-md"

    def list_files(
        self,
        context: MarkdownMemoryContext,
        scope: MarkdownMemoryScope,
        *,
        path: str = ".",
        max_results: int = 100,
    ) -> list[MarkdownMemoryFile]:
        root = self._scope_root(context, scope)
        target = self._resolve_dir(root, path)
        if not target.exists():
            return []
        if not target.is_dir():
            raise NotADirectoryError(f"memory path is not a directory: {path}")
        files: list[MarkdownMemoryFile] = []
        for file_path in sorted(target.rglob("*.md")):
            if len(files) >= self._bounded_int(max_results, default=100, minimum=1, maximum=500):
                break
            if not file_path.is_file():
                continue
            stat = file_path.stat()
            files.append(
                MarkdownMemoryFile(
                    scope=scope,
                    path=self._display_path(root, file_path),
                    size=stat.st_size,
                    updated_at=self._format_timestamp(stat.st_mtime),
                )
            )
        return files

    def read(
        self,
        context: MarkdownMemoryContext,
        scope: MarkdownMemoryScope,
        path: str,
        *,
        max_chars: int = 12000,
    ) -> tuple[str, bool]:
        root = self._scope_root(context, scope)
        target = self._resolve_file(root, path)
        if not target.is_file():
            raise FileNotFoundError(f"memory file not found: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        bounded = self._bounded_int(max_chars, default=12000, minimum=100, maximum=500000)
        return text[:bounded], len(text) > bounded

    def write(
        self,
        context: MarkdownMemoryContext,
        scope: MarkdownMemoryScope,
        path: str,
        content: str,
    ) -> MarkdownMemoryFile:
        target = self._write_text(context, scope, path, content)
        root = self._scope_root(context, scope)
        stat = target.stat()
        return MarkdownMemoryFile(
            scope=scope,
            path=self._display_path(root, target),
            size=stat.st_size,
            updated_at=self._format_timestamp(stat.st_mtime),
        )

    def append(
        self,
        context: MarkdownMemoryContext,
        scope: MarkdownMemoryScope,
        path: str,
        content: str,
        *,
        heading: str | None = None,
    ) -> MarkdownMemoryFile:
        root = self._scope_root(context, scope)
        target = self._resolve_file(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else ""
        chunks: list[str] = []
        if existing:
            chunks.append(existing.rstrip())
        if heading and heading.strip():
            chunks.append(f"## {heading.strip()}")
        chunks.append(content.strip())
        return self.write(context, scope, path, "\n\n".join(chunk for chunk in chunks if chunk) + "\n")

    def search(
        self,
        context: MarkdownMemoryContext,
        scopes: list[MarkdownMemoryScope],
        query: str,
        *,
        max_results: int = 50,
    ) -> list[MarkdownMemorySearchMatch]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query is required")
        needle = clean_query.lower()
        bounded = self._bounded_int(max_results, default=50, minimum=1, maximum=200)
        matches: list[MarkdownMemorySearchMatch] = []
        for scope in scopes:
            root = self._scope_root(context, scope)
            if not root.exists():
                continue
            for file_path in sorted(root.rglob("*.md")):
                if len(matches) >= bounded:
                    return matches
                if not file_path.is_file() or self._skip_file(file_path):
                    continue
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                for line_number, line in enumerate(lines, start=1):
                    if needle in line.lower():
                        matches.append(
                            MarkdownMemorySearchMatch(
                                scope=scope,
                                path=self._display_path(root, file_path),
                                line=line_number,
                                text=line[:500],
                            )
                        )
                        if len(matches) >= bounded:
                            return matches
        return matches

    def compress(
        self,
        context: MarkdownMemoryContext,
        scope: MarkdownMemoryScope,
        source_path: str,
        *,
        target_path: str | None = None,
        max_chars: int = 4000,
        keywords: list[str] | None = None,
    ) -> MarkdownMemoryCompression:
        original, _ = self.read(context, scope, source_path, max_chars=500000)
        bounded = self._bounded_int(max_chars, default=4000, minimum=500, maximum=100000)
        clean_keywords = [keyword.strip() for keyword in keywords or [] if keyword.strip()]
        compressed = self._compress_text(original, max_chars=bounded, keywords=clean_keywords)
        target = target_path or self._compressed_path(source_path)
        self.write(context, scope, target, compressed)
        return MarkdownMemoryCompression(
            scope=scope,
            source_path=self._normalize_md_path(source_path),
            target_path=self._normalize_md_path(target),
            original_chars=len(original),
            compressed_chars=len(compressed),
            truncated=len(compressed) < len(original),
            keywords=clean_keywords,
        )

    def scope_root(self, context: MarkdownMemoryContext, scope: MarkdownMemoryScope) -> Path:
        return self._scope_root(context, scope)

    def _scope_root(self, context: MarkdownMemoryContext, scope: MarkdownMemoryScope) -> Path:
        if scope == "global":
            return self.root_dir / "global"
        user_key = self._safe_key(context.user_id or f"agent-{context.agent_name}")
        if scope == "user":
            return self.root_dir / "users" / user_key
        if scope == "project":
            project_key = self._safe_key(context.project_id or "default")
            return self.root_dir / "users" / user_key / "projects" / project_key
        thread_key = self._safe_key(context.thread_id or "default")
        return self.root_dir / "users" / user_key / "sessions" / thread_key

    def _write_text(
        self,
        context: MarkdownMemoryContext,
        scope: MarkdownMemoryScope,
        path: str,
        content: str,
    ) -> Path:
        root = self._scope_root(context, scope)
        target = self._resolve_file(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    @classmethod
    def _compress_text(cls, text: str, *, max_chars: int, keywords: list[str]) -> str:
        title = cls._first_heading(text) or "Compressed Memory"
        lines = text.splitlines()
        sections = cls._sections(lines)
        selected_lines: list[str] = []
        keyword_needles = [keyword.lower() for keyword in keywords]
        for heading, body in sections:
            if keyword_needles and not cls._section_matches(heading, body, keyword_needles):
                continue
            selected_lines.append(f"## {heading}" if heading else "## Notes")
            selected_lines.extend(cls._important_lines(body, keyword_needles, max_lines=12))
            if len("\n".join(selected_lines)) >= max_chars:
                break
        if not selected_lines:
            selected_lines = cls._important_lines(lines, keyword_needles, max_lines=80)
        body_text = "\n".join(line for line in selected_lines if line.strip())
        header = "\n".join(
            [
                "---",
                "compressed: true",
                f"original_chars: {len(text)}",
                f"generated_at: {cls._now()}",
                "---",
                "",
                f"# {title} 摘要",
                "",
            ]
        )
        budget = max_chars - len(header)
        if budget <= 0:
            return header[:max_chars]
        truncated_body = body_text[:budget].rstrip()
        if len(body_text) > budget and budget > 80:
            marker = "\n\n> 已按 max_chars 截断。"
            truncated_body = f"{body_text[: max(0, budget - len(marker) - 1)].rstrip()}{marker}"
        rendered = f"{header}{truncated_body}\n"
        return rendered[:max_chars]

    @staticmethod
    def _sections(lines: list[str]) -> list[tuple[str, list[str]]]:
        sections: list[tuple[str, list[str]]] = []
        current_heading = ""
        current_body: list[str] = []
        for line in lines:
            if line.startswith("#"):
                if current_heading or current_body:
                    sections.append((current_heading, current_body))
                current_heading = line.lstrip("#").strip()
                current_body = []
            else:
                current_body.append(line)
        if current_heading or current_body:
            sections.append((current_heading, current_body))
        return sections

    @staticmethod
    def _important_lines(lines: list[str], keyword_needles: list[str], *, max_lines: int) -> list[str]:
        scored: list[tuple[int, int, str]] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            score = 0
            if stripped.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", ">")):
                score += 2
            if any(needle in stripped.lower() for needle in keyword_needles):
                score += 5
            if any(marker in stripped for marker in ("决定", "结论", "TODO", "todo", "问题", "风险", "偏好")):
                score += 3
            if score == 0 and len(scored) < max_lines // 3:
                score = 1
            if score > 0:
                scored.append((score, index, stripped))
        selected = sorted(sorted(scored, key=lambda item: (-item[0], item[1]))[:max_lines], key=lambda item: item[1])
        return [line for _, _, line in selected]

    @staticmethod
    def _section_matches(heading: str, body: list[str], keyword_needles: list[str]) -> bool:
        text = "\n".join([heading, *body]).lower()
        return any(needle in text for needle in keyword_needles)

    @staticmethod
    def _first_heading(text: str) -> str | None:
        for line in text.splitlines():
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                if heading:
                    return heading
        return None

    @staticmethod
    def _compressed_path(source_path: str) -> str:
        normalized = MarkdownMemoryStore._normalize_md_path(source_path)
        source = Path(normalized)
        return f"{source.with_suffix('').as_posix()}.summary.md"

    @staticmethod
    def _resolve_file(root: Path, raw_path: str) -> Path:
        normalized = MarkdownMemoryStore._normalize_md_path(raw_path)
        return MarkdownMemoryStore._resolve_under(root, normalized)

    @staticmethod
    def _resolve_dir(root: Path, raw_path: str) -> Path:
        normalized = raw_path.strip() or "."
        if normalized != ".":
            normalized = normalized.replace("\\", "/").lstrip("/")
        return MarkdownMemoryStore._resolve_under(root, normalized)

    @staticmethod
    def _resolve_under(root: Path, normalized_path: str) -> Path:
        candidate = (root / normalized_path).resolve()
        resolved_root = root.resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Markdown memory path traversal blocked") from exc
        return candidate

    @staticmethod
    def _normalize_md_path(raw_path: str) -> str:
        normalized = raw_path.replace("\\", "/").lstrip("/").strip()
        if not normalized or normalized in {".", ".."}:
            raise ValueError("path is required")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("invalid markdown memory path")
        if normalized.startswith("~"):
            raise ValueError("home-relative paths are not allowed")
        if not normalized.endswith(".md"):
            raise ValueError("only .md memory files are allowed")
        return normalized

    @staticmethod
    def _display_path(root: Path, path: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    @staticmethod
    def _safe_key(value: str) -> str:
        cleaned = _SAFE_KEY_RE.sub("-", value.strip()).strip(".-")
        return cleaned[:96] or "default"

    @staticmethod
    def _skip_file(path: Path) -> bool:
        try:
            return path.stat().st_size > 1_000_000
        except OSError:
            return True

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
