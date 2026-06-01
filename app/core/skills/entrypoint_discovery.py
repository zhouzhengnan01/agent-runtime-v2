from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.core.skills.plugin_manifest import skill_md_frontmatter, string_metadata


EntrypointKind = Literal["function", "script"]

_PYTHON_PATH_RE = re.compile(r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.py)")
_WORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")
_SKILL_RUN_KEYS = {"skill_name", "spec", "thread_id", "workspace_dir", "uploads_dir", "outputs_dir"}
_EXCLUDED_NAMES = {"setup.py", "conftest.py", "spec_builder.py", "utils.py", "__init__.py"}
_EXCLUDED_PARTS = {"__pycache__", ".venv", "venv", "tests", "test"}
_PREFERRED_SCRIPT_PATHS = ("scripts/run_skill.py", "run_skill.py", "scripts/run.py", "run.py", "scripts/main.py", "main.py")


@dataclass(frozen=True)
class DiscoveredEntrypoint:
    path: Path
    relative_path: str
    kind: EntrypointKind
    reason: str
    score: int


@dataclass(frozen=True)
class PythonFileAnalysis:
    path: Path
    relative_path: str
    has_run_skill: bool
    has_run: bool
    has_skill_runner: bool
    has_main: bool
    has_main_guard: bool
    supports_stdin_json: bool
    tokens: frozenset[str]

    @property
    def function_entry(self) -> bool:
        return self.has_run_skill or self.has_run or self.has_skill_runner

    @property
    def script_entry(self) -> bool:
        return self.has_main or self.has_main_guard or self.supports_stdin_json


def discover_skill_entrypoint(manifest_path: Path, manifest: dict[str, Any] | None = None) -> DiscoveredEntrypoint | None:
    package_root = manifest_path.parent.resolve()
    skill_md_text = _skill_md_text(manifest_path)
    explicit = _explicit_entrypoint(package_root, manifest or {}, skill_md_text)
    analyses = _python_file_analyses(package_root)
    if not analyses:
        return None

    if explicit is not None:
        explicit_analysis = analyses.get(explicit.resolve())
        if explicit_analysis is not None:
            return _entrypoint_from_analysis(explicit_analysis, "explicit_entrypoint", score=10_000)

    mentioned = _mentioned_python_paths(package_root, skill_md_text)
    mentioned_candidates = [
        analysis
        for path in mentioned
        if (analysis := analyses.get(path.resolve())) is not None
    ]
    if len(mentioned_candidates) == 1:
        return _entrypoint_from_analysis(
            mentioned_candidates[0],
            "skill_md_python_reference",
            score=2_500,
            force_kind="script",
        )
    selected = _select_best(
        mentioned_candidates,
        skill_md_text,
        reason="skill_md_reference",
        base_score=2_000,
        require_clear_winner=False,
    )
    if selected is not None:
        return selected

    preferred_candidates = [
        analysis
        for relative in _PREFERRED_SCRIPT_PATHS
        if (analysis := analyses.get((package_root / relative).resolve())) is not None
    ]
    selected = _select_best(
        preferred_candidates,
        skill_md_text,
        reason="preferred_python_entrypoint",
        base_score=1_000,
        require_clear_winner=False,
    )
    if selected is not None:
        return selected

    entry_candidates = [analysis for analysis in analyses.values() if analysis.function_entry or analysis.supports_stdin_json]
    selected = _select_best(
        entry_candidates,
        skill_md_text,
        reason="python_entrypoint_signature",
        base_score=500,
        require_clear_winner=len(entry_candidates) > 1,
    )
    if selected is not None:
        return selected

    single_candidate = _single_fallback_candidate(list(analyses.values()))
    if single_candidate is not None:
        return _entrypoint_from_analysis(single_candidate, "single_python_file_fallback", score=100)
    return None


def manifest_with_discovered_execution(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest.get("execution"), dict):
        return manifest
    entrypoint = discover_skill_entrypoint(manifest_path, manifest)
    if entrypoint is None or entrypoint.kind != "script":
        return manifest
    updated = dict(manifest)
    updated["execution"] = {
        "type": "python_script",
        "script": entrypoint.relative_path,
        "input_mode": "stdin_json",
        "collect_outputs": True,
        "runtime": _frontmatter_runtime(_skill_md_text(manifest_path)),
        "discovered": True,
        "discovery_reason": entrypoint.reason,
    }
    return updated


def _skill_md_text(manifest_path: Path) -> str:
    if manifest_path.name == "SKILL.md" and manifest_path.is_file():
        return manifest_path.read_text(encoding="utf-8")
    candidate = manifest_path.parent / "SKILL.md"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return ""


def _explicit_entrypoint(package_root: Path, manifest: dict[str, Any], skill_md_text: str) -> Path | None:
    for value in (manifest.get("entrypoint"), manifest.get("entry_point")):
        path = _safe_relative_python_path(package_root, string_metadata(value))
        if path is not None:
            return path
    metadata = skill_md_frontmatter(skill_md_text)
    for key in ("entrypoint", "entry_point"):
        raw = metadata.get(key)
        if isinstance(raw, list):
            values = [item for item in raw if isinstance(item, str)]
        else:
            values = [string_metadata(raw)]
        for value in values:
            path = _safe_relative_python_path(package_root, value)
            if path is not None:
                return path
    return None


def _frontmatter_runtime(skill_md_text: str) -> str:
    runtime = string_metadata(skill_md_frontmatter(skill_md_text).get("runtime"))
    return runtime or "python"


def _mentioned_python_paths(package_root: Path, skill_md_text: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for match in _PYTHON_PATH_RE.finditer(skill_md_text):
        path = _safe_relative_python_path(package_root, match.group("path"))
        if path is None or path in seen:
            continue
        paths.append(path)
        seen.add(path)
    return paths


def _python_file_analyses(package_root: Path) -> dict[Path, PythonFileAnalysis]:
    analyses: dict[Path, PythonFileAnalysis] = {}
    for path in sorted(package_root.rglob("*.py")):
        if not _is_candidate_python_file(package_root, path):
            continue
        analysis = _analyze_python_file(package_root, path)
        if analysis is not None:
            analyses[path.resolve()] = analysis
    return analyses


def _is_candidate_python_file(package_root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(package_root)
    except ValueError:
        return False
    if path.name.startswith("test_") or path.name in _EXCLUDED_NAMES:
        return False
    return not any(part in _EXCLUDED_PARTS for part in relative.parts)


def _analyze_python_file(package_root: Path, path: Path) -> PythonFileAnalysis | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    top_level_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    relative_path = path.relative_to(package_root).as_posix()
    lowered = text.lower()
    return PythonFileAnalysis(
        path=path.resolve(),
        relative_path=relative_path,
        has_run_skill="run_skill" in top_level_names,
        has_run="run" in top_level_names,
        has_skill_runner="SkillRunner" in top_level_names,
        has_main="main" in top_level_names,
        has_main_guard='__name__ == "__main__"' in text or "__name__ == '__main__'" in text,
        supports_stdin_json="sys.stdin" in text
        or "json.load(sys.stdin)" in text
        or len(_SKILL_RUN_KEYS.intersection(_WORD_RE.findall(lowered))) >= 3,
        tokens=frozenset(_tokens(f"{relative_path}\n{text[:8000]}")),
    )


def _select_best(
    analyses: list[PythonFileAnalysis],
    skill_md_text: str,
    *,
    reason: str,
    base_score: int,
    require_clear_winner: bool,
) -> DiscoveredEntrypoint | None:
    scored = [
        _entrypoint_from_analysis(analysis, reason, score=base_score + _analysis_score(analysis, skill_md_text))
        for analysis in analyses
        if analysis.function_entry or analysis.script_entry
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item.score, item.relative_path))
    if require_clear_winner and len(scored) > 1 and scored[0].score - scored[1].score < 25:
        return None
    return scored[0]


def _entrypoint_from_analysis(
    analysis: PythonFileAnalysis,
    reason: str,
    *,
    score: int,
    force_kind: EntrypointKind | None = None,
) -> DiscoveredEntrypoint:
    kind: EntrypointKind = force_kind or ("function" if analysis.function_entry else "script")
    return DiscoveredEntrypoint(
        path=analysis.path,
        relative_path=analysis.relative_path,
        kind=kind,
        reason=reason,
        score=score + _kind_score(analysis),
    )


def _analysis_score(analysis: PythonFileAnalysis, skill_md_text: str) -> int:
    score = _kind_score(analysis)
    if analysis.relative_path.startswith("scripts/"):
        score += 20
    if analysis.relative_path in _PREFERRED_SCRIPT_PATHS:
        score += 80
    md_tokens = set(_tokens(skill_md_text))
    score += min(100, len(md_tokens.intersection(analysis.tokens)) * 4)
    return score


def _kind_score(analysis: PythonFileAnalysis) -> int:
    if analysis.has_run_skill:
        return 250
    if analysis.has_run:
        return 220
    if analysis.has_skill_runner:
        return 210
    if analysis.supports_stdin_json:
        return 170
    if analysis.has_main:
        return 80
    if analysis.has_main_guard:
        return 60
    return 0


def _single_fallback_candidate(analyses: list[PythonFileAnalysis]) -> PythonFileAnalysis | None:
    fallback = [
        analysis
        for analysis in analyses
        if analysis.relative_path.startswith("scripts/")
        and not Path(analysis.relative_path).name.startswith("_")
    ]
    return fallback[0] if len(fallback) == 1 else None


def _safe_relative_python_path(package_root: Path, value: str) -> Path | None:
    if not value:
        return None
    candidate_text = value.strip().strip("`\"'")
    if candidate_text.startswith("python "):
        parts = candidate_text.split()
        candidate_text = next((part for part in parts[1:] if part.endswith(".py")), "")
    if not candidate_text.endswith(".py") or Path(candidate_text).is_absolute():
        return None
    path = (package_root / candidate_text).resolve()
    try:
        path.relative_to(package_root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]
