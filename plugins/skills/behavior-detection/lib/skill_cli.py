from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths


def main(skill_name: str, package_root: Path) -> int:
    parser = argparse.ArgumentParser(description=f"Run {skill_name} skill package")
    parser.add_argument("--request")
    parser.add_argument("--outputs")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--workspace")
    args = parser.parse_args()

    request_path = Path(args.request or args.input or "")
    if not request_path.is_file():
        parser.error("--request or --input must point to a JSON file")

    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        payload = {}
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else payload.get("input")
    if not isinstance(spec, dict):
        spec = payload

    outputs_dir = Path(args.outputs or payload.get("outputs_dir") or args.workspace or ".").resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    thread_id = str(payload.get("thread_id") or skill_name)
    store = ArtifactStore(root_dir=outputs_dir / ".runtime")
    paths = store.prepare_thread(thread_id)

    module = _load_package_runner(package_root)
    result = _call_runner(module, skill_name, spec, paths, store)
    copied = _copy_outputs(store, paths, outputs_dir, result)
    response = {
        "success": True,
        "skill_name": skill_name,
        "outputs": [{"type": "file", "path": path.name} for path in copied],
        "result": _result_data(result),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _load_package_runner(package_root: Path) -> ModuleType:
    runner_path = package_root / "runner.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"Skill package runner not found: {runner_path}")
    plugin_root = _plugin_root(package_root)
    added_paths = [str(package_root), str(plugin_root)]
    spec = importlib.util.spec_from_file_location(f"skill_package_{abs(hash(runner_path))}", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
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


def _plugin_root(package_root: Path) -> Path:
    current = package_root.resolve()
    while current != current.parent:
        if (current / "plugin.json").is_file():
            return current
        current = current.parent
    return package_root.resolve()


def _call_runner(
    module: ModuleType,
    skill_name: str,
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> object:
    runner = getattr(module, "run_skill", None) or getattr(module, "run", None)
    if not callable(runner):
        raise RuntimeError("Skill package runner must expose run_skill() or run().")
    signature = inspect.signature(runner)
    kwargs = {
        "skill_name": skill_name,
        "spec": spec,
        "paths": paths,
        "artifact_store": artifact_store,
    }
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return runner(**accepted) if accepted else runner(skill_name, spec, paths, artifact_store)


def _copy_outputs(store: ArtifactStore, paths: ThreadPaths, outputs_dir: Path, result: object) -> list[Path]:
    copied: list[Path] = []
    for output in _result_outputs(result):
        virtual_path = getattr(output, "path", None)
        name = getattr(output, "name", None)
        if isinstance(output, dict):
            virtual_path = output.get("path")
            name = output.get("name")
        if not isinstance(virtual_path, str) or not isinstance(name, str):
            continue
        source = store.resolve_virtual_path(paths.thread_id, virtual_path)
        target = (outputs_dir / Path(name).name).resolve()
        target.relative_to(outputs_dir)
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def _result_outputs(result: object) -> list[object]:
    if isinstance(result, dict):
        outputs = result.get("outputs", result.get("artifacts", []))
        return outputs if isinstance(outputs, list) else []
    outputs = getattr(result, "outputs", [])
    return outputs if isinstance(outputs, list) else []


def _result_data(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        data = result.get("data")
        return dict(data) if isinstance(data, dict) else {}
    data = getattr(result, "data", {})
    return dict(data) if isinstance(data, dict) else {}
