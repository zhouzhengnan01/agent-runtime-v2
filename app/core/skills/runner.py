from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any

from app.core.artifacts.store import ArtifactStore, ThreadPaths
from app.core.skills.plugins import SkillPluginManager
from app.core.skills.runner_types import SkillRunResult

__all__ = ["SkillRunResult", "SkillRunner"]


class SkillRunner:
    """Execute skills through installed skill plugins.

    The runtime deliberately does not dispatch on concrete skill names here.
    Skill-specific code lives in uploaded plugin directories under
    ``plugins/skills/<plugin-id>/``.
    """

    def __init__(self, artifact_store: ArtifactStore, root_dir: Path | None = None) -> None:
        self.artifact_store = artifact_store
        self.plugin_manager = SkillPluginManager(root_dir)

    def run(
        self,
        skill_name: str,
        spec: dict[str, Any],
        paths: ThreadPaths,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> SkillRunResult:
        return self.plugin_manager.run_skill(skill_name, spec, paths, self.artifact_store, on_event=on_event)
