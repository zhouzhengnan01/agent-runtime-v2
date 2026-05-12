from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.skills.plugin_execution import load_runner_module
from app.core.skills.plugin_manifest import read_json, safe_zip_members, single_root_prefix, validated_plugin_id
from app.core.workflow.config import WorkflowConfig


@dataclass(frozen=True)
class WorkflowPluginPackage:
    plugin_id: str
    name: str
    version: str
    description: str
    root: Path
    manifest_paths: dict[str, Path] = field(default_factory=dict)
    protected: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "root_path": str(self.root),
            "workflows": sorted(self.manifest_paths),
            "protected": self.protected,
        }


@dataclass(frozen=True)
class LoadedWorkflowPlugin:
    config: WorkflowConfig
    manifest_path: Path
    plugin: object
    package: WorkflowPluginPackage
    entrypoint_path: Path
    entrypoint_class: str


class WorkflowPluginManager:
    """Discover, validate, and import workflow plugins from plugins/workflows."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[4]
        self.root_dir = root_dir or self.project_root
        self.plugin_dir = self.root_dir / "plugins" / "workflows"
        self.config_dir = self.root_dir / "config" / "workflows"

    def list_plugins(self) -> list[WorkflowPluginPackage]:
        plugins: list[WorkflowPluginPackage] = []
        for plugin_root in self._plugin_roots():
            try:
                plugins.append(self._load_package(plugin_root))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sorted(plugins, key=lambda item: item.plugin_id)

    def load_workflows(self, artifact_store: ArtifactStore) -> dict[str, LoadedWorkflowPlugin]:
        plugin_workflows = self._load_plugin_workflows(artifact_store)
        loaded: dict[str, LoadedWorkflowPlugin] = {}
        for entity_path in self._config_entity_paths():
            try:
                config = WorkflowConfig.model_validate(read_json(entity_path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            implementation = plugin_workflows.get(config.name)
            if implementation is None:
                continue
            loaded[config.name] = LoadedWorkflowPlugin(
                config=config,
                manifest_path=entity_path,
                plugin=implementation.plugin,
                package=implementation.package,
                entrypoint_path=implementation.entrypoint_path,
                entrypoint_class=implementation.entrypoint_class,
            )
        for workflow_name, implementation in plugin_workflows.items():
            if workflow_name in loaded or not self._is_overlay_plugin_root(implementation.package.root):
                continue
            loaded[workflow_name] = implementation
        return loaded

    def install_zip(self, content: bytes) -> WorkflowPluginPackage:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = safe_zip_members(archive)
            prefix = single_root_prefix(members)
            plugin_json_name = f"{prefix}plugin.json" if prefix else "plugin.json"
            if plugin_json_name not in members:
                raise ValueError("Workflow plugin zip must contain plugin.json.")
            metadata = json.loads(archive.read(plugin_json_name).decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("plugin.json must be a JSON object.")
            plugin_id = validated_plugin_id(str(metadata.get("id") or "").strip())
            target_root = self.plugin_dir / plugin_id
            if self._is_protected_plugin_root(target_root):
                raise ValueError(f"Workflow plugin is protected and cannot be replaced: {plugin_id}")
            temp_root = self.plugin_dir / f".{plugin_id}.tmp"
            if temp_root.exists():
                shutil.rmtree(temp_root)
            temp_root.mkdir(parents=True, exist_ok=True)
            for member in members:
                relative_name = member[len(prefix) :] if prefix and member.startswith(prefix) else member
                if not relative_name or relative_name.endswith("/"):
                    continue
                target = (temp_root / relative_name).resolve()
                try:
                    target.relative_to(temp_root.resolve())
                except ValueError as exc:
                    raise ValueError(f"Invalid plugin archive path: {member}") from exc
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        plugin = self._load_package(temp_root)
        if plugin.plugin_id != plugin_id:
            raise ValueError("plugin.json id changed during installation.")
        if plugin.protected:
            raise ValueError("Uploaded workflow plugins cannot be protected.")
        if target_root.exists():
            shutil.rmtree(target_root)
        temp_root.replace(target_root)
        plugin = self._load_package(target_root)
        self._materialize_workflow_entities(plugin)
        return plugin

    def delete_plugin(self, plugin_id: str) -> WorkflowPluginPackage:
        safe_id = validated_plugin_id(plugin_id.strip())
        target_root = self.plugin_dir / safe_id
        if not target_root.is_dir() or not (target_root / "plugin.json").is_file():
            raise FileNotFoundError(f"Workflow plugin not found: {safe_id}")
        plugin = self._load_package(target_root)
        if plugin.protected:
            raise PermissionError(f"Workflow plugin is protected and cannot be deleted: {safe_id}")
        shutil.rmtree(target_root)
        return plugin

    def _plugin_roots(self) -> list[Path]:
        roots: list[Path] = []
        project_plugins = self.project_root / "plugins" / "workflows"
        seen_names: set[str] = set()
        for parent in self._unique_dirs(self.plugin_dir, project_plugins):
            if not parent.is_dir():
                continue
            for child in sorted(parent.iterdir()):
                if child.is_dir() and (child / "plugin.json").is_file() and child.name not in seen_names:
                    roots.append(child)
                    seen_names.add(child.name)
        return roots

    def _config_entity_paths(self) -> list[Path]:
        paths_by_name: dict[str, Path] = {}
        project_config_dir = self.project_root / "config" / "workflows"
        for config_dir in self._unique_dirs(project_config_dir, self.config_dir):
            if not config_dir.is_dir():
                continue
            for entity_path in sorted(config_dir.glob("*.json")):
                paths_by_name[entity_path.stem] = entity_path
        return [paths_by_name[name] for name in sorted(paths_by_name)]

    def _is_overlay_plugin_root(self, plugin_root: Path) -> bool:
        if self.root_dir == self.project_root:
            return False
        try:
            plugin_root.resolve().relative_to(self.plugin_dir.resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _unique_dirs(*paths: Path) -> tuple[Path, ...]:
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            unique.append(path)
            seen.add(resolved)
        return tuple(unique)

    def _load_package(self, plugin_root: Path) -> WorkflowPluginPackage:
        metadata = read_json(plugin_root / "plugin.json")
        plugin_id = validated_plugin_id(str(metadata.get("id") or plugin_root.name).strip())
        manifest_paths = self._manifest_paths(plugin_root, metadata)
        if not manifest_paths:
            raise ValueError(f"Workflow plugin contains no valid workflow manifests: {plugin_id}")
        return WorkflowPluginPackage(
            plugin_id=plugin_id,
            name=str(metadata.get("name") or plugin_id),
            version=str(metadata.get("version") or ""),
            description=str(metadata.get("description") or ""),
            root=plugin_root,
            manifest_paths=manifest_paths,
            protected=bool(metadata.get("protected", False)),
        )

    def _load_plugin_workflows(self, artifact_store: ArtifactStore) -> dict[str, LoadedWorkflowPlugin]:
        loaded: dict[str, LoadedWorkflowPlugin] = {}
        for package in self.list_plugins():
            for workflow_name, manifest_path in package.manifest_paths.items():
                try:
                    config = WorkflowConfig.model_validate(read_json(manifest_path))
                    plugin, entrypoint_path, entrypoint_class = self._instantiate_plugin(
                        package,
                        config,
                        artifact_store,
                    )
                    self._validate_plugin(plugin, config)
                except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
                    continue
                loaded[workflow_name] = LoadedWorkflowPlugin(
                    config=config,
                    manifest_path=manifest_path,
                    plugin=plugin,
                    package=package,
                    entrypoint_path=entrypoint_path,
                    entrypoint_class=entrypoint_class,
                )
        return loaded

    def _materialize_workflow_entities(self, plugin: WorkflowPluginPackage) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        for workflow_name, manifest_path in plugin.manifest_paths.items():
            target = self.config_dir / f"{workflow_name}.json"
            if target.is_file():
                continue
            try:
                config = WorkflowConfig.model_validate(read_json(manifest_path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            target.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _manifest_paths(plugin_root: Path, metadata: dict[str, Any]) -> dict[str, Path]:
        patterns = metadata.get("workflows") or ["workflow.json", "workflows/*.json", "workflows/*/workflow.json"]
        raw_patterns = patterns if isinstance(patterns, list) else [patterns]
        manifests: dict[str, Path] = {}
        for raw_pattern in raw_patterns:
            if not isinstance(raw_pattern, str):
                continue
            for path in sorted(plugin_root.glob(raw_pattern)):
                if not path.is_file() or path.suffix.lower() != ".json":
                    continue
                try:
                    path.resolve().relative_to(plugin_root.resolve())
                    config = WorkflowConfig.model_validate(read_json(path))
                    WorkflowPluginManager._entrypoint(plugin_root, config.handler)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                manifests[config.name] = path
        return manifests

    @staticmethod
    def _instantiate_plugin(
        package: WorkflowPluginPackage,
        config: WorkflowConfig,
        artifact_store: ArtifactStore,
    ) -> tuple[object, Path, str]:
        entrypoint_path, class_name = WorkflowPluginManager._entrypoint(package.root, config.handler)
        module = load_runner_module(entrypoint_path)
        plugin_class = getattr(module, class_name, None)
        if plugin_class is None or not callable(plugin_class):
            raise ValueError(f"Workflow plugin class not found: {config.handler}")
        try:
            plugin = plugin_class(artifact_store)
        except TypeError:
            plugin = plugin_class()
        return plugin, entrypoint_path, class_name

    @staticmethod
    def _entrypoint(plugin_root: Path, handler: str) -> tuple[Path, str]:
        if ":" not in handler:
            raise ValueError("Uploaded workflow handler must use module.py:ClassName.")
        module_path, class_name = handler.split(":", 1)
        if not module_path.strip() or not class_name.strip():
            raise ValueError("Uploaded workflow handler must include module path and class name.")
        if not module_path.endswith(".py"):
            raise ValueError("Uploaded workflow handler module must be a .py file.")
        path = (plugin_root / module_path.strip()).resolve()
        try:
            path.relative_to(plugin_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Workflow plugin handler must stay inside plugin root: {handler}") from exc
        if not path.is_file():
            raise ValueError(f"Workflow plugin handler file not found: {module_path}")
        return path, class_name.strip()

    @staticmethod
    def _validate_plugin(plugin: object, config: WorkflowConfig) -> None:
        runner = getattr(plugin, "run_with_events", None)
        if not callable(runner):
            raise ValueError(f"Workflow plugin must expose run_with_events(): {config.name}")

    @staticmethod
    def _is_protected_plugin_root(plugin_root: Path) -> bool:
        plugin_json = plugin_root / "plugin.json"
        if not plugin_json.is_file():
            return False
        try:
            return bool(read_json(plugin_json).get("protected", False))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
