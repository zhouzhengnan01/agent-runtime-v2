from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.resources import ResourceRoots, merged_json_paths
from app.core.skills.plugin_execution import (
    call_runner as _call_runner,
    invoke_hook as _invoke_hook,
    load_runner_module as _load_runner_module,
    normalize_result as _normalize_result,
)
from app.core.skills.generic_runner import can_run_generic as _can_run_generic
from app.core.skills.generic_runner import run_generic_skill as _run_generic_skill
from app.core.skills.aliases import invalidate_skill_alias_cache
from app.core.skills.entrypoint_discovery import discover_skill_entrypoint as _discover_skill_entrypoint
from app.core.skills.entrypoint_discovery import manifest_with_discovered_execution as _manifest_with_discovered_execution
from app.core.skills.plugin_manifest import (
    json_object_from_text as _json_object_from_text,
    plugin_metadata_from_skill_md as _plugin_metadata_from_skill_md,
    read_json as _read_json,
    read_manifest_source as _read_manifest_source,
    safe_zip_members as _safe_zip_members,
    single_root_prefix as _single_root_prefix,
    skill_package_root as _skill_package_root,
    normalize_skill_manifest as _normalize_skill_manifest,
    validated_plugin_id as _validated_plugin_id,
)
from app.core.skills.registry import SkillDefinition, definition_from_manifest
from app.core.skills.runner_types import SkillRunResult


_PACKAGE_FILE_SPECS: dict[str, tuple[str, str, str]] = {
    "manifest": ("manifest.json", "Manifest JSON", "json"),
    "skill-md": ("SKILL.md", "SKILL.md", "markdown"),
    "requirements": ("requirements.txt", "requirements.txt", "text"),
    "sandbox": ("sandbox.yml", "sandbox.yml", "yaml"),
    "runner": ("runner.py", "runner.py", "python"),
    "spec-builder": ("spec_builder.py", "spec_builder.py", "python"),
    "script-runner": ("scripts/run_skill.py", "scripts/run_skill.py", "python"),
}


@dataclass(frozen=True)
class SkillPlugin:
    plugin_id: str
    name: str
    version: str
    description: str
    root: Path
    runner_path: Path | None = None
    spec_builder_path: Path | None = None
    manifest_paths: dict[str, Path] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "root_path": str(self.root),
            "runner_path": str(self.runner_path) if self.runner_path is not None else None,
            "spec_builder_path": str(self.spec_builder_path) if self.spec_builder_path is not None else None,
            "skills": sorted(self.manifest_paths),
        }


@dataclass(frozen=True)
class SkillPackageFile:
    file_id: str
    label: str
    path: Path
    exists: bool
    editable: bool
    language: str

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.file_id,
            "label": self.label,
            "path": str(self.path),
            "exists": self.exists,
            "editable": self.editable,
            "language": self.language,
        }


@dataclass(frozen=True)
class LoadedSkill:
    definition: SkillDefinition
    manifest_path: Path
    plugin: SkillPlugin | None = None
    runner_path: Path | None = None
    spec_builder_path: Path | None = None

    @property
    def executable(self) -> bool:
        return (
            self.runner_path is not None
            or _can_run_generic(self.definition.to_event_payload())
            or _can_run_prompt_only(self.definition)
        )


class SkillPluginManager:
    """Discover, install, and execute uploaded skill plugins."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[3]
        self.root_dir = root_dir or self.project_root
        self.resources = ResourceRoots.from_project_root(self.root_dir)
        self.plugin_dirs = self.resources.plugin_dirs("skills")
        self.config_dirs = self.resources.config_dirs("skills")
        self.plugin_dir = self.root_dir / "plugins" / "skills"
        self.config_dir = self.root_dir / "config" / "skills"

    def list_plugins(self) -> list[SkillPlugin]:
        plugins: list[SkillPlugin] = []
        for plugin_root in self._plugin_roots():
            try:
                plugins.append(self._load_plugin(plugin_root))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return sorted(plugins, key=lambda plugin: plugin.plugin_id)

    def load_skills(self, *, include_plugin_only: bool | None = None) -> dict[str, LoadedSkill]:
        plugin_skills = self._load_plugin_skills()

        loaded: dict[str, LoadedSkill] = {}
        for entity_path in self._config_entity_paths():
            try:
                definition = definition_from_manifest(_read_json(entity_path), entity_path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            implementation = plugin_skills.get(definition.name)
            if implementation is None:
                loaded[definition.name] = LoadedSkill(definition=definition, manifest_path=entity_path, plugin=None)
                continue
            if definition.execution is None and implementation.definition.execution is not None:
                definition = replace(definition, execution=implementation.definition.execution)
            loaded[definition.name] = LoadedSkill(
                definition=_with_plugin_metadata(
                    definition,
                    implementation.plugin,
                    entity_path,
                    runner_path=implementation.runner_path,
                    spec_builder_path=implementation.spec_builder_path,
                ),
                manifest_path=entity_path,
                plugin=implementation.plugin,
                runner_path=implementation.runner_path,
                spec_builder_path=implementation.spec_builder_path,
            )

        if include_plugin_only is None:
            include_plugin_only = self.root_dir != self.project_root
        if include_plugin_only:
            for skill_name, implementation in plugin_skills.items():
                loaded.setdefault(skill_name, implementation)

        return loaded

    def get_loaded_skill(self, skill_name: str) -> LoadedSkill:
        try:
            return self.load_skills()[skill_name]
        except KeyError as exc:
            plugin_skill = self._load_plugin_skills().get(skill_name)
            if plugin_skill is not None:
                return plugin_skill
            raise KeyError(f"Unknown skill: {skill_name}") from exc

    def manifest_path_for(self, skill_name: str) -> Path:
        loaded = self.load_skills().get(skill_name)
        if loaded is not None and loaded.manifest_path.suffix.lower() == ".json":
            if self.root_dir == self.project_root:
                return loaded.manifest_path
            try:
                loaded.manifest_path.resolve().relative_to(self.config_dir.resolve())
                return loaded.manifest_path
            except ValueError:
                pass
        return self.config_dir / f"{skill_name}.json"

    def read_manifest(self, skill_name: str) -> dict[str, Any]:
        loaded = self.get_loaded_skill(skill_name)
        manifest = _read_manifest_with_discovery(loaded.manifest_path)
        if isinstance(manifest.get("execution"), dict) or loaded.plugin is None:
            return manifest
        plugin_manifest_path = loaded.plugin.manifest_paths.get(skill_name)
        if plugin_manifest_path is None or plugin_manifest_path.resolve() == loaded.manifest_path.resolve():
            return manifest
        plugin_manifest = _read_manifest_with_discovery(plugin_manifest_path)
        execution = plugin_manifest.get("execution")
        if isinstance(execution, dict):
            manifest = dict(manifest)
            manifest["execution"] = execution
        return manifest

    def list_package_files(self, skill_name: str) -> list[SkillPackageFile]:
        loaded = self.get_loaded_skill(skill_name)
        files: list[SkillPackageFile] = []
        for file_id, (_relative, label, language) in _PACKAGE_FILE_SPECS.items():
            path = self._package_file_path(loaded, file_id)
            if path is None:
                continue
            files.append(
                SkillPackageFile(
                    file_id=file_id,
                    label=label,
                    path=path,
                    exists=path.is_file(),
                    editable=True,
                    language=language,
                )
            )
        execution_script = self._execution_script_file(loaded)
        if execution_script is not None and execution_script.path not in {item.path for item in files}:
            files.append(execution_script)
        return files

    def read_package_file(self, skill_name: str, file_id: str) -> tuple[SkillPackageFile, str]:
        loaded = self.get_loaded_skill(skill_name)
        package_file = self._package_file(loaded, file_id)
        if package_file.file_id == "manifest" and not package_file.path.is_file():
            content = json.dumps(self.read_manifest(skill_name), ensure_ascii=False, indent=2) + "\n"
            return package_file, content
        content = package_file.path.read_text(encoding="utf-8") if package_file.path.is_file() else ""
        return package_file, content

    def write_package_file(self, skill_name: str, file_id: str, content: str) -> tuple[SkillPackageFile, str]:
        loaded = self.get_loaded_skill(skill_name)
        package_file = self._package_file(loaded, file_id)
        if package_file.file_id == "manifest":
            manifest = _json_object_from_text(content, "manifest.json")
            self.save_manifest(skill_name, manifest)
            refreshed = self._package_file(self.get_loaded_skill(skill_name), file_id)
            saved = refreshed.path.read_text(encoding="utf-8") if refreshed.path.is_file() else (
                json.dumps(self.read_manifest(skill_name), ensure_ascii=False, indent=2) + "\n"
            )
            return refreshed, saved
        if package_file.language == "json":
            parsed = _json_object_from_text(content, package_file.label)
            content = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
        package_file.path.parent.mkdir(parents=True, exist_ok=True)
        package_file.path.write_text(content, encoding="utf-8")
        invalidate_skill_alias_cache()
        return self._package_file(self.get_loaded_skill(skill_name), file_id), content

    def save_manifest(self, skill_name: str, manifest: dict[str, Any]) -> LoadedSkill:
        existing = self.load_skills().get(skill_name) or self._load_plugin_skills().get(skill_name)
        path = self.manifest_path_for(skill_name)
        data = dict(manifest)
        data["name"] = skill_name
        data = _normalize_skill_manifest(data)
        definition = definition_from_manifest(data, path)
        if existing is not None:
            self._validate_execution(data, existing)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path.with_suffix(".json.tmp")
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        target.replace(path)
        if (
            existing is not None
            and existing.plugin is not None
        ):
            package_root = self._sandbox_write_root(existing)
            if package_root is not None:
                self._write_sandbox_package_file(package_root, data.get("sandbox"))
        invalidate_skill_alias_cache()
        loaded = self.load_skills().get(skill_name)
        if loaded is not None:
            return loaded
        return LoadedSkill(definition=definition, manifest_path=path, plugin=None)

    def install_zip(self, content: bytes) -> SkillPlugin:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = _safe_zip_members(archive)
            prefix = _single_root_prefix(members)
            plugin_json_name = f"{prefix}plugin.json" if prefix else "plugin.json"
            skill_md_name = f"{prefix}SKILL.md" if prefix else "SKILL.md"
            generated_plugin_json: dict[str, Any] | None = None
            if plugin_json_name not in members:
                if skill_md_name not in members:
                    raise ValueError("Skill plugin zip must contain plugin.json or a single SKILL.md package at the archive root.")
                generated_plugin_json = _plugin_metadata_from_skill_md(
                    archive.read(skill_md_name).decode("utf-8"),
                    Path(skill_md_name).parent,
                )
                metadata = generated_plugin_json
            else:
                metadata = json.loads(archive.read(plugin_json_name).decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("plugin.json must be a JSON object.")
            plugin_id = _validated_plugin_id(str(metadata.get("id") or "").strip())
            target_root = self.plugin_dir / plugin_id
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
            if generated_plugin_json is not None:
                (temp_root / "plugin.json").write_text(
                    json.dumps(generated_plugin_json, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        plugin = self._load_plugin(temp_root)
        if plugin.plugin_id != plugin_id:
            raise ValueError("plugin.json id changed during installation.")
        self._normalize_uploaded_package_files(plugin)
        if target_root.exists():
            shutil.rmtree(target_root)
        temp_root.replace(target_root)
        plugin = self._load_plugin(target_root)
        self._materialize_plugin_entities(plugin)
        invalidate_skill_alias_cache()
        return plugin

    def delete_plugin(self, plugin_id: str) -> SkillPlugin:
        safe_id = _validated_plugin_id(plugin_id.strip())
        target_root = self.plugin_dir / safe_id
        if not target_root.is_dir() or not (target_root / "plugin.json").is_file():
            raise FileNotFoundError(f"Skill plugin not found: {safe_id}")
        plugin = self._load_plugin(target_root)
        shutil.rmtree(target_root)
        invalidate_skill_alias_cache()
        return plugin

    def _materialize_plugin_entities(self, plugin: SkillPlugin) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        for skill_name, manifest_path in plugin.manifest_paths.items():
            target = self.config_dir / f"{skill_name}.json"
            if target.is_file():
                continue
            try:
                manifest = _read_manifest_with_discovery(manifest_path)
                manifest["name"] = skill_name
                manifest = _normalize_skill_manifest(manifest)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load_plugin_skills(self) -> dict[str, LoadedSkill]:
        loaded: dict[str, LoadedSkill] = {}
        for plugin in self.list_plugins():
            for skill_name, manifest_path in plugin.manifest_paths.items():
                try:
                    definition = definition_from_manifest(_read_manifest_with_discovery(manifest_path), manifest_path)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                runner_path = self._skill_runner_path(plugin, manifest_path)
                spec_builder_path = self._skill_spec_builder_path(plugin, manifest_path)
                candidate = LoadedSkill(
                    definition=_with_plugin_metadata(
                        definition,
                        plugin,
                        manifest_path,
                        runner_path=runner_path,
                        spec_builder_path=spec_builder_path,
                    ),
                    manifest_path=manifest_path,
                    plugin=plugin,
                    runner_path=runner_path,
                    spec_builder_path=spec_builder_path,
                )
                existing = loaded.get(skill_name)
                if existing is not None and existing.plugin is not None and existing.plugin.plugin_id == skill_name:
                    continue
                loaded[skill_name] = candidate
        return loaded

    def _config_entity_paths(self) -> list[Path]:
        return merged_json_paths(self.config_dirs)

    def run_skill(
        self,
        skill_name: str,
        spec: dict[str, Any],
        paths: ThreadPaths,
        artifact_store: ArtifactStore,
        on_event: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> SkillRunResult:
        loaded = self.get_loaded_skill(skill_name)
        if loaded.plugin is None:
            raise KeyError(f"Skill is not backed by an installed plugin: {skill_name}")
        manifest = self.read_manifest(skill_name)
        if loaded.runner_path is None and _can_run_generic(manifest):
            return _run_generic_skill(
                skill_name,
                manifest,
                spec,
                paths,
                artifact_store,
                package_root=self._execution_package_root(loaded),
            )
        if loaded.runner_path is None:
            return self._run_prompt_only_skill(skill_name, manifest, spec, paths, artifact_store, loaded)
        if loaded.runner_path.resolve() == loaded.manifest_path.resolve() and _can_run_generic(manifest):
            return _run_generic_skill(
                skill_name,
                manifest,
                spec,
                paths,
                artifact_store,
                package_root=self._execution_package_root(loaded),
            )
        module = _load_runner_module(loaded.runner_path)
        result = _call_runner(module, skill_name, spec, paths, artifact_store, on_event=on_event)
        return _normalize_result(skill_name, result)

    def select_skill(self, routing_text: str, attachments: list[Any], allowed_skills: list[str]) -> str:
        candidate = self.select_skill_candidate(routing_text, attachments, allowed_skills)
        if candidate[0]:
            return candidate[0]
        loaded = self.load_skills()
        executable = [name for name in allowed_skills if name in loaded and loaded[name].executable]
        return executable[0] if executable else (allowed_skills[0] if allowed_skills else "")

    def select_skill_candidate(self, routing_text: str, attachments: list[Any], allowed_skills: list[str]) -> tuple[str, int]:
        loaded = self.load_skills()
        best_name = ""
        best_score = 0
        for skill_name in allowed_skills:
            skill = loaded.get(skill_name)
            if skill is None or skill.plugin is None:
                continue
            score = _score_skill(skill, skill_name, routing_text, attachments, allowed_skills)
            if score > best_score:
                best_name = skill_name
                best_score = score
        if best_name:
            return best_name, best_score
        return "", 0

    def build_spec(
        self,
        skill_name: str,
        user_text: str,
        routing_text: str,
        attachments: list[Any],
        base_spec: dict[str, Any],
    ) -> dict[str, Any]:
        loaded = self.load_skills().get(skill_name)
        if loaded is None or loaded.plugin is None:
            return base_spec
        return _build_plugin_spec(loaded, skill_name, user_text, routing_text, attachments, base_spec)

    def format_reply(self, skill_name: str, verification: object, run_result: SkillRunResult) -> str | None:
        loaded = self.load_skills().get(skill_name)
        if loaded is None or loaded.runner_path is None:
            return None
        if loaded.runner_path.resolve() == loaded.manifest_path.resolve() and _can_run_generic(self.read_manifest(skill_name)):
            return None
        module = _load_runner_module(loaded.runner_path)
        formatter = getattr(module, "format_reply", None)
        if not callable(formatter):
            return None
        result = _invoke_hook(
            cast(Callable[..., object], formatter),
            {
                "skill_name": skill_name,
                "verification": verification,
                "run_result": run_result,
            },
        )
        return result if isinstance(result, str) else None

    def _run_prompt_only_skill(
        self,
        skill_name: str,
        manifest: dict[str, Any],
        spec: dict[str, Any],
        paths: ThreadPaths,
        artifact_store: ArtifactStore,
        loaded: LoadedSkill,
    ) -> SkillRunResult:
        content = _prompt_only_markdown(skill_name, manifest, spec, loaded)
        filename = f"{_safe_artifact_stem(skill_name)}-prompt.md"
        artifact = artifact_store.write_text_artifact(paths, filename, content)
        return SkillRunResult(
            skill_name=skill_name,
            outputs=[artifact],
            data={
                "execution_type": "prompt_only",
                "message": "Skill has no local runner or execution block; generated a prompt-only skill task package.",
                "skill_name": skill_name,
                "objective": spec.get("objective") or spec.get("message") or spec.get("prompt") or "",
                "manifest_path": str(loaded.manifest_path),
                "plugin_id": loaded.plugin.plugin_id if loaded.plugin is not None else None,
            },
        )

    def _plugin_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen_names: set[str] = set()
        for parent in self.plugin_dirs:
            if not parent.is_dir():
                continue
            for child in sorted(parent.iterdir()):
                if child.is_dir() and (child / "plugin.json").is_file() and child.name not in seen_names:
                    roots.append(child)
                    seen_names.add(child.name)
        return roots

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

    def _load_plugin(self, plugin_root: Path) -> SkillPlugin:
        metadata = _read_json(plugin_root / "plugin.json")
        plugin_id = _validated_plugin_id(str(metadata.get("id") or plugin_root.name).strip())
        runner_path = self._optional_child_path(plugin_root, metadata.get("runner") or "runner.py")
        if isinstance(metadata.get("runner"), str) and runner_path is None:
            raise ValueError(f"Plugin runner not found: {metadata['runner']}")
        spec_builder_path = self._optional_child_path(plugin_root, metadata.get("spec_builder"))
        manifest_paths = self._manifest_paths(plugin_root, metadata)
        if not manifest_paths:
            raise ValueError(f"Plugin contains no skill manifests: {plugin_id}")
        return SkillPlugin(
            plugin_id=plugin_id,
            name=str(metadata.get("name") or plugin_id),
            version=str(metadata.get("version") or ""),
            description=str(metadata.get("description") or ""),
            root=plugin_root,
            runner_path=runner_path,
            spec_builder_path=spec_builder_path,
            manifest_paths=manifest_paths,
        )

    @staticmethod
    def _optional_child_path(plugin_root: Path, value: object) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        path = (plugin_root / value.strip()).resolve()
        try:
            path.relative_to(plugin_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Plugin path must stay inside plugin root: {value}") from exc
        return path if path.is_file() else None

    @staticmethod
    def _manifest_paths(plugin_root: Path, metadata: dict[str, Any]) -> dict[str, Path]:
        patterns = metadata.get("skills") or ["manifest.json", "SKILL.md", "skills/*.json", "skills/*/manifest.json", "skills/*/SKILL.md"]
        raw_patterns = patterns if isinstance(patterns, list) else [patterns]
        manifests: dict[str, Path] = {}
        package_roots_with_json: set[Path] = set()
        candidate_paths: list[Path] = []
        for raw_pattern in raw_patterns:
            if not isinstance(raw_pattern, str):
                continue
            for path in sorted(plugin_root.glob(raw_pattern)):
                if not path.is_file():
                    continue
                candidate_paths.append(path)
                if path.name == "manifest.json":
                    package_roots_with_json.add(path.parent.resolve())

        for path in sorted(candidate_paths, key=lambda item: (item.name != "manifest.json", str(item))):
            if path.name == "SKILL.md" and path.parent.resolve() in package_roots_with_json:
                continue
            if path.suffix.lower() != ".json" and path.name != "SKILL.md":
                continue
            try:
                manifest = _read_manifest_with_discovery(path)
                definition = definition_from_manifest(manifest, path)
                _validate_manifest_execution(manifest, path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            manifests[definition.name] = path
        return manifests

    @staticmethod
    def _skill_runner_path(plugin: SkillPlugin, manifest_path: Path) -> Path | None:
        local = manifest_path.parent / "runner.py"
        if local.is_file():
            return local.resolve()
        if plugin.runner_path is not None:
            return plugin.runner_path
        entrypoint = _discover_skill_entrypoint(manifest_path, _read_manifest_source(manifest_path))
        if entrypoint is not None and entrypoint.kind == "function":
            return entrypoint.path
        return None

    @staticmethod
    def _skill_spec_builder_path(plugin: SkillPlugin, manifest_path: Path) -> Path | None:
        local = manifest_path.parent / "spec_builder.py"
        if local.is_file():
            return local.resolve()
        return plugin.spec_builder_path

    def _package_file(self, loaded: LoadedSkill, file_id: str) -> SkillPackageFile:
        if file_id == "execution-script":
            package_file = self._execution_script_file(loaded)
            if package_file is None:
                raise ValueError(f"Skill package file is not available for {loaded.definition.name}: {file_id}")
            return package_file
        if file_id not in _PACKAGE_FILE_SPECS:
            raise ValueError(f"Unknown skill package file: {file_id}")
        path = self._package_file_path(loaded, file_id)
        if path is None:
            raise ValueError(f"Skill package file is not available for {loaded.definition.name}: {file_id}")
        _relative, label, language = _PACKAGE_FILE_SPECS[file_id]
        return SkillPackageFile(
            file_id=file_id,
            label=label,
            path=path,
            exists=path.is_file(),
            editable=True,
            language=language,
        )

    def _package_file_path(self, loaded: LoadedSkill, file_id: str) -> Path | None:
        relative, _label, _language = _PACKAGE_FILE_SPECS[file_id]
        if file_id == "manifest":
            return self.manifest_path_for(loaded.definition.name).resolve()
        if loaded.plugin is None:
            return None
        package_root = self._loaded_package_root(loaded)
        if package_root is None:
            return None
        path = (package_root / relative).resolve()
        try:
            path.relative_to(package_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Skill package file path escaped package root: {file_id}") from exc
        return path

    def _validate_execution(self, manifest: dict[str, Any], loaded: LoadedSkill) -> None:
        execution = manifest.get("execution")
        if not isinstance(execution, dict) or execution.get("type") != "python_script":
            return
        script = execution.get("script")
        if not isinstance(script, str) or not script.strip():
            raise ValueError("Python script execution requires execution.script.")
        package_root = self._loaded_package_root(loaded)
        if package_root is None:
            raise ValueError("Python script execution requires a skill package root.")
        script_path = (package_root / script.strip()).resolve()
        try:
            script_path.relative_to(package_root.resolve())
        except ValueError as exc:
            raise ValueError("Python script path must stay inside the skill package root.") from exc
        if not script_path.is_file():
            raise ValueError(f"Python script not found: {script.strip()}")

    def _execution_script_file(self, loaded: LoadedSkill) -> SkillPackageFile | None:
        if loaded.plugin is None:
            return None
        try:
            manifest = self.read_manifest(loaded.definition.name)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            return None
        execution = manifest.get("execution")
        if not isinstance(execution, dict) or execution.get("type") != "python_script":
            return None
        script = execution.get("script")
        if not isinstance(script, str) or not script.strip():
            return None
        package_root = self._loaded_package_root(loaded)
        if package_root is None:
            return None
        path = (package_root / script.strip()).resolve()
        try:
            path.relative_to(package_root.resolve())
        except ValueError:
            return None
        return SkillPackageFile(
            file_id="execution-script",
            label=script.strip(),
            path=path,
            exists=path.is_file(),
            editable=True,
            language="python" if path.suffix == ".py" else "text",
        )

    @staticmethod
    def _execution_package_root(loaded: LoadedSkill) -> Path | None:
        package_root = SkillPluginManager._loaded_package_root(loaded)
        if package_root is not None:
            return package_root
        if loaded.plugin is not None:
            return loaded.plugin.root.resolve()
        return None

    @staticmethod
    def _loaded_package_root(loaded: LoadedSkill) -> Path | None:
        if loaded.plugin is not None:
            standard_package = loaded.plugin.root / "skills" / loaded.definition.name
            if standard_package.is_dir():
                return standard_package.resolve()
            if loaded.manifest_path.name == "SKILL.md" and loaded.manifest_path.parent.resolve() == loaded.plugin.root.resolve():
                return loaded.plugin.root.resolve()
            package_root = _skill_package_root(loaded)
            if package_root is not None:
                return package_root
            return loaded.plugin.root.resolve()
        package_root = _skill_package_root(loaded)
        if package_root is not None:
            return package_root
        return None

    @staticmethod
    def _sandbox_write_root(loaded: LoadedSkill) -> Path | None:
        if loaded.plugin is not None:
            plugin_manifest_path = loaded.plugin.manifest_paths.get(loaded.definition.name)
            if (
                plugin_manifest_path is not None
                and plugin_manifest_path.name == "SKILL.md"
                and plugin_manifest_path.parent.resolve() == loaded.plugin.root.resolve()
            ):
                return loaded.plugin.root.resolve()
        return SkillPluginManager._loaded_package_root(loaded)

    @staticmethod
    def _normalize_uploaded_package_files(plugin: SkillPlugin) -> None:
        for manifest_path in plugin.manifest_paths.values():
            if manifest_path.suffix.lower() != ".json":
                continue
            manifest = _read_manifest_with_discovery(manifest_path)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_sandbox_package_file(package_root: Path, sandbox: object) -> None:
        sandbox_data = sandbox if isinstance(sandbox, dict) else {}
        enabled = bool(sandbox_data.get("enabled", False))
        profile = sandbox_data.get("profile")
        request_schema_version = sandbox_data.get("request_schema_version")
        adapter_command = sandbox_data.get("adapter_command")
        fallback_to_local = sandbox_data.get("fallback_to_local")

        lines = ["sandbox:", f"  enabled: {'true' if enabled else 'false'}"]
        if isinstance(profile, str):
            lines.append(f"  profile: {json.dumps(profile, ensure_ascii=False)}")
        else:
            lines.append('  profile: ""')
        if isinstance(request_schema_version, str) and request_schema_version.strip():
            lines.append(f"  request_schema_version: {json.dumps(request_schema_version.strip(), ensure_ascii=False)}")
        if isinstance(adapter_command, str) and adapter_command.strip():
            lines.append(f"  adapter_command: {json.dumps(adapter_command.strip(), ensure_ascii=False)}")
        if isinstance(fallback_to_local, bool):
            lines.append(f"  fallback_to_local: {'true' if fallback_to_local else 'false'}")

        sandbox_path = package_root / "sandbox.yml"
        sandbox_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _with_plugin_metadata(
    definition: SkillDefinition,
    plugin: SkillPlugin | None,
    manifest_path: Path,
    runner_path: Path | None,
    spec_builder_path: Path | None,
) -> SkillDefinition:
    if runner_path is None and _can_run_generic(definition.to_event_payload()):
        runner_path = manifest_path
    if plugin is None:
        return definition.with_source(
            source_type="manifest",
            plugin_id=None,
            plugin_name=None,
            plugin_version=None,
            plugin_root=None,
            manifest_path=manifest_path,
            runner_path=runner_path,
            spec_builder_path=spec_builder_path,
        )
    return definition.with_source(
        source_type="plugin",
        plugin_id=plugin.plugin_id,
        plugin_name=plugin.name,
        plugin_version=plugin.version,
        plugin_root=plugin.root,
        manifest_path=manifest_path,
        runner_path=runner_path,
        spec_builder_path=spec_builder_path,
    )


def _can_run_prompt_only(definition: SkillDefinition) -> bool:
    if definition.source_type != "plugin":
        return False
    routing = definition.routing
    if not isinstance(routing, dict):
        return False
    summary = routing.get("summary")
    if isinstance(summary, str) and summary.strip():
        return True
    for key in ("keywords", "examples"):
        value = routing.get(key)
        if isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value):
            return True
    return False


def _prompt_only_markdown(skill_name: str, manifest: dict[str, Any], spec: dict[str, Any], loaded: LoadedSkill) -> str:
    description = str(manifest.get("description") or skill_name).strip()
    objective = str(spec.get("objective") or spec.get("message") or spec.get("prompt") or "").strip()
    skill_md = _prompt_only_skill_md(loaded)
    references = _prompt_only_reference_paths(loaded)
    lines = [
        f"# {skill_name}",
        "",
        "This is a prompt-only skill package. The skill is installed and selected, but it does not define a local runner or manifest.execution block.",
        "Use the instructions below as the active skill contract for the current request.",
        "",
        "## Objective",
        "",
        objective or "(not provided)",
        "",
        "## Manifest Summary",
        "",
        f"- Description: {description}",
        f"- Output kind: {manifest.get('output_kind') or 'json'}",
    ]
    routing = manifest.get("routing")
    if isinstance(routing, dict):
        keywords = [str(item) for item in routing.get("keywords", []) if isinstance(item, str)]
        if keywords:
            lines.append(f"- Keywords: {', '.join(keywords[:30])}")
    lines.extend(["", "## Input Spec", "", "```json", json.dumps(spec, ensure_ascii=False, indent=2), "```"])
    if skill_md:
        lines.extend(["", "## SKILL.md", "", skill_md[:20_000]])
    if references:
        lines.extend(["", "## Reference Files", ""])
        lines.extend(f"- `{path}`" for path in references[:80])
    lines.extend(
        [
            "",
            "## Expected Result",
            "",
            "Generate the requested deliverable according to the manifest, SKILL.md, and references. If files are required, create them as artifacts in the thread outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def _prompt_only_skill_md(loaded: LoadedSkill) -> str:
    candidates = []
    if loaded.manifest_path.name == "SKILL.md":
        candidates.append(loaded.manifest_path)
    candidates.append(loaded.manifest_path.parent / "SKILL.md")
    if loaded.plugin is not None:
        candidates.append(loaded.plugin.root / "SKILL.md")
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return ""
    return ""


def _prompt_only_reference_paths(loaded: LoadedSkill) -> list[str]:
    if loaded.plugin is None:
        return []
    roots = [loaded.manifest_path.parent / "references", loaded.plugin.root / "references"]
    paths: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                paths.append(path.relative_to(loaded.plugin.root).as_posix())
            except ValueError:
                paths.append(path.as_posix())
    return paths


def _safe_artifact_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "-" for char in value.strip())
    return cleaned.strip(".-") or "skill"

def _validate_manifest_execution(manifest: dict[str, Any], manifest_path: Path) -> None:
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("type") != "python_script":
        return
    script = execution.get("script")
    if not isinstance(script, str) or not script.strip():
        raise ValueError(f"Python script execution requires execution.script: {manifest_path}")
    package_root = manifest_path.parent.resolve()
    script_path = (package_root / script.strip()).resolve()
    try:
        script_path.relative_to(package_root)
    except ValueError as exc:
        raise ValueError(f"Python script path must stay inside the skill package root: {script}") from exc
    if not script_path.is_file():
        raise ValueError(f"Python script not found: {script}")


def _score_skill(
    loaded: LoadedSkill,
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if loaded.plugin is None:
        return 0
    if loaded.spec_builder_path is None:
        return _manifest_keyword_score(loaded.plugin, skill_name, routing_text)
    module = _load_runner_module(loaded.spec_builder_path)
    scorer = getattr(module, "score_skill", None)
    if not callable(scorer):
        return _manifest_keyword_score(loaded.plugin, skill_name, routing_text)
    result = _invoke_hook(
        cast(Callable[..., object], scorer),
        {
            "skill_name": skill_name,
            "routing_text": routing_text,
            "attachments": attachments,
            "allowed_skills": allowed_skills,
        },
    )
    return int(result) if isinstance(result, int | float) else 0


def _build_plugin_spec(
    loaded: LoadedSkill,
    skill_name: str,
    user_text: str,
    routing_text: str,
    attachments: list[Any],
    base_spec: dict[str, Any],
) -> dict[str, Any]:
    if loaded.spec_builder_path is None:
        return base_spec
    module = _load_runner_module(loaded.spec_builder_path)
    builder = getattr(module, "build_spec", None)
    if not callable(builder):
        return base_spec
    result = _invoke_hook(
        cast(Callable[..., object], builder),
        {
            "skill_name": skill_name,
            "user_text": user_text,
            "routing_text": routing_text,
            "attachments": attachments,
            "base_spec": base_spec,
        },
    )
    return dict(result) if isinstance(result, dict) else base_spec


def _manifest_keyword_score(plugin: SkillPlugin, skill_name: str, routing_text: str) -> int:
    manifest_path = plugin.manifest_paths.get(skill_name)
    if manifest_path is None:
        return 0
    try:
        manifest = _read_manifest_with_discovery(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0
    routing = manifest.get("routing")
    keywords = []
    if isinstance(routing, dict) and isinstance(routing.get("keywords"), list):
        keywords = [str(item).lower() for item in routing["keywords"] if isinstance(item, str)]
    text = routing_text.lower()
    return 50 if any(keyword in text for keyword in keywords) else 0


def _read_manifest_with_discovery(path: Path) -> dict[str, Any]:
    manifest = _read_manifest_source(path)
    return _manifest_with_discovered_execution(path, manifest)
