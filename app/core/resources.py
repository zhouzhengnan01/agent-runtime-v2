from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResourceRoots:
    """Resolve runtime resources from upload dirs, legacy mounts, and image defaults."""

    project_root: Path
    upload_root: Path
    builtin_root: Path
    config_upload_dir: Path
    plugins_upload_dir: Path
    static_upload_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path | None = None) -> ResourceRoots:
        resolved_project_root = project_root or Path(__file__).resolve().parents[2]
        resolved_project_root = resolved_project_root.resolve()
        upload_root = Path(
            os.getenv("JETLINKS_AGENT_UPLOAD_ROOT")
            or os.getenv("JETLINKS_AGENT_USER_ROOT")
            or resolved_project_root
        ).resolve()
        builtin_root = Path(
            os.getenv("JETLINKS_AGENT_BUILTIN_ROOT")
            or os.getenv("JETLINKS_AGENT_DEFAULTS_DIR")
            or "/opt/jetlinks-agent-runtime-agent-v2-defaults"
        ).resolve()
        config_upload_dir = Path(
            os.getenv("JETLINKS_AGENT_CONFIG_UPLOAD_DIR") or upload_root / "config" / "upload"
        ).resolve()
        plugins_upload_dir = Path(
            os.getenv("JETLINKS_AGENT_PLUGINS_UPLOAD_DIR") or upload_root / "plugins" / "upload"
        ).resolve()
        static_upload_dir = Path(
            os.getenv("JETLINKS_AGENT_STATIC_UPLOAD_DIR") or upload_root / "static" / "upload"
        ).resolve()
        return cls(
            project_root=resolved_project_root,
            upload_root=upload_root,
            builtin_root=builtin_root,
            config_upload_dir=config_upload_dir,
            plugins_upload_dir=plugins_upload_dir,
            static_upload_dir=static_upload_dir,
        )

    def config_dirs(self, relative: str) -> list[Path]:
        return self._unique_paths(
            [
                self.config_upload_dir / relative,
                self.project_root / "config" / relative,
                self.builtin_root / "config" / relative,
            ]
        )

    def plugin_dirs(self, relative: str) -> list[Path]:
        return self._unique_paths(
            [
                self.plugins_upload_dir / relative,
                self.project_root / "plugins" / relative,
                self.builtin_root / "plugins" / relative,
            ]
        )

    def static_dirs(self) -> list[Path]:
        return self._unique_paths([self.static_upload_dir, *self.system_static_dirs()])

    def system_static_dirs(self) -> list[Path]:
        return self._unique_paths([self.builtin_root / "static", self.project_root / "static"])

    def writable_config_dir(self, relative: str) -> Path:
        return self.config_upload_dir / relative

    def writable_plugin_dir(self, relative: str) -> Path:
        return self.plugins_upload_dir / relative

    @staticmethod
    def _unique_paths(candidates: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            unique.append(candidate)
            seen.add(resolved)
        return unique


def first_existing_file(paths: list[Path], filename: str) -> Path | None:
    for directory in paths:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def merged_json_paths(paths: list[Path], *, skip_names: set[str] | None = None) -> list[Path]:
    paths_by_name: dict[str, Path] = {}
    skipped = skip_names or set()
    # Lower-priority dirs are visited first; user overlay dirs visited later win.
    for directory in reversed(paths):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name.startswith(".") or path.name in skipped:
                continue
            paths_by_name[path.stem] = path
    return [paths_by_name[name] for name in sorted(paths_by_name)]
