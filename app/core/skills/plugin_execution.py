from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.skills.runner_types import SkillRunResult
from app.schemas import ArtifactRef


def load_runner_module(runner_path: Path) -> ModuleType:
    module_name = f"jetlinks_skill_plugin_{abs(hash(runner_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load skill plugin runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    added_paths = runner_import_paths(runner_path)
    try:
        for path in added_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        spec.loader.exec_module(module)
    finally:
        for path in added_paths:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
    return module


def runner_import_paths(runner_path: Path) -> list[str]:
    paths = [str(runner_path.parent)]
    current = runner_path.parent
    while current != current.parent:
        if (current / "plugin.json").is_file():
            paths.append(str(current))
            break
        current = current.parent
    return paths


def call_runner(
    module: ModuleType,
    skill_name: str,
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
    on_event: Callable[[str, dict[str, Any]], Any] | None = None,
) -> object:
    if hasattr(module, "run_skill"):
        return invoke_callable(getattr(module, "run_skill"), skill_name, spec, paths, artifact_store, on_event=on_event)
    if hasattr(module, "run"):
        return invoke_callable(getattr(module, "run"), skill_name, spec, paths, artifact_store, on_event=on_event)
    runner_class = getattr(module, "SkillRunner", None)
    if runner_class is not None:
        runner = runner_class(artifact_store)
        return runner.run(skill_name, spec, paths)
    raise ValueError("Skill plugin runner must expose run_skill(), run(), or SkillRunner.")


def invoke_hook(func: Callable[..., object], kwargs: dict[str, object]) -> object:
    signature = inspect.signature(func)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return func(**accepted)


def invoke_callable(
    func: object,
    skill_name: str,
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
    on_event: Callable[[str, dict[str, Any]], Any] | None = None,
) -> object:
    if not callable(func):
        raise ValueError("Skill plugin runner hook is not callable.")
    signature = inspect.signature(func)
    kwargs = {
        "skill_name": skill_name,
        "spec": spec,
        "paths": paths,
        "artifact_store": artifact_store,
        "on_event": on_event,
    }
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    if accepted:
        return func(**accepted)
    return func(skill_name, spec, paths, artifact_store)


def normalize_result(skill_name: str, result: object) -> SkillRunResult:
    if isinstance(result, SkillRunResult):
        return result
    if isinstance(result, dict):
        outputs = artifact_refs(result.get("outputs", result.get("artifacts", [])))
        data = result.get("data")
        return SkillRunResult(
            skill_name=str(result.get("skill_name") or skill_name),
            outputs=outputs,
            data=dict(data) if isinstance(data, dict) else {},
        )
    result_skill_name = getattr(result, "skill_name", skill_name)
    result_outputs = getattr(result, "outputs", [])
    result_data = getattr(result, "data", {})
    return SkillRunResult(
        skill_name=str(result_skill_name),
        outputs=artifact_refs(result_outputs),
        data=dict(result_data) if isinstance(result_data, dict) else {},
    )


def artifact_refs(value: object) -> list[ArtifactRef]:
    if not isinstance(value, list):
        return []
    refs: list[ArtifactRef] = []
    for item in value:
        if isinstance(item, ArtifactRef):
            refs.append(item)
        elif isinstance(item, dict):
            refs.append(ArtifactRef.model_validate(item))
    return refs
