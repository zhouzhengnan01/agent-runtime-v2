from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.skills.plugins import SkillPluginManager


PLATFORM_SKILL_ALIASES: dict[str, list[str]] = {
    "1778483741456a5glxkmk": [
        "algorithm-engineer",
        "dataset-curator",
        "data-auto-annotation",
        "image-dataset-generation",
        "algorithm-research-scout",
        "model-candidate-selector",
        "remote-gpu-ops",
        "gpu-training-orchestrator",
        "cpu-training-runner",
        "detector-evaluator",
        "deployment-candidate-reviewer",
        "experiment-ledger",
    ],
}


def expand_skill_aliases(values: list[str], root_dir: Path | None = None) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    aliases = skill_aliases(root_dir)
    for value in values:
        names = aliases.get(value, [value])
        for name in names:
            clean = str(name).strip()
            if clean and clean not in seen:
                expanded.append(clean)
                seen.add(clean)
    return expanded


def skill_aliases(root_dir: Path | None = None) -> dict[str, list[str]]:
    aliases = {key: list(value) for key, value in PLATFORM_SKILL_ALIASES.items()}
    aliases.update(_plugin_skill_aliases(str((root_dir or _project_root()).resolve())))
    return aliases


def invalidate_skill_alias_cache() -> None:
    _plugin_skill_aliases.cache_clear()


@lru_cache(maxsize=16)
def _plugin_skill_aliases(root_dir: str) -> dict[str, list[str]]:
    manager = _manager(Path(root_dir))
    aliases: dict[str, list[str]] = {}
    try:
        plugins = manager.list_plugins()
    except (OSError, ValueError, TypeError):
        return aliases
    for plugin in plugins:
        names = sorted(plugin.manifest_paths)
        if not names:
            continue
        aliases[plugin.plugin_id] = names
        plugin_name = plugin.name.strip()
        if plugin_name and plugin_name != plugin.plugin_id:
            aliases[plugin_name] = names
    return aliases


def _manager(root_dir: Path) -> SkillPluginManager:
    from app.core.skills.plugins import SkillPluginManager

    return SkillPluginManager(root_dir)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]
