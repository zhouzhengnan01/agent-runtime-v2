from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.model_tags import normalize_model_tags

if TYPE_CHECKING:
    from app.core.skills.plugins import SkillPluginManager


@dataclass(frozen=True)
class SkillSandboxSpec:
    enabled: bool = False
    profile: str | None = None
    request_schema_version: str = "skill-run.v1"
    adapter_command: str | None = None
    fallback_to_local: bool | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "profile": self.profile,
            "request_schema_version": self.request_schema_version,
            "adapter_command": self.adapter_command,
            "fallback_to_local": self.fallback_to_local,
        }


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    output_kind: str
    generation: bool = True
    model_tags: tuple[str, ...] = ()
    quality_template: tuple[str, ...] = ()
    skill_type: str = "atomic"
    child_skills: tuple[str, ...] = ()
    stages: tuple[dict[str, Any], ...] = ()
    done_when: tuple[str, ...] = ()
    routing: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    sandbox: SkillSandboxSpec = field(default_factory=SkillSandboxSpec)
    source_type: str = "manifest"
    plugin_id: str | None = None
    plugin_name: str | None = None
    plugin_version: str | None = None
    plugin_root: Path | None = None
    manifest_path: Path | None = None
    runner_path: Path | None = None
    spec_builder_path: Path | None = None

    @property
    def executable(self) -> bool:
        return self.runner_path is not None or _has_generic_execution(self.execution)

    @property
    def composite(self) -> bool:
        return self.skill_type == "composite" and bool(self.child_skills)

    def to_event_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "output_kind": self.output_kind,
            "generation": self.generation,
            "model_tags": list(self.model_tags),
            "quality_template": list(self.quality_template),
            "skill_type": self.skill_type,
            "child_skills": list(self.child_skills),
            "stages": list(self.stages),
            "done_when": list(self.done_when),
            "routing": self.routing or {},
            "execution": self.execution or {},
            "input_schema": self.input_schema or {},
            "output_schema": self.output_schema or {},
            "sandbox": self.sandbox.to_payload(),
            "executable": self.executable,
            "source": {
                "type": self.source_type,
                "plugin_id": self.plugin_id,
                "plugin_name": self.plugin_name,
                "plugin_version": self.plugin_version,
                "plugin_root": str(self.plugin_root) if self.plugin_root is not None else None,
                "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
                "runner_path": str(self.runner_path) if self.runner_path is not None else None,
                "spec_builder_path": str(self.spec_builder_path) if self.spec_builder_path is not None else None,
            },
        }

    def with_source(
        self,
        *,
        source_type: str,
        plugin_id: str | None,
        plugin_name: str | None,
        plugin_version: str | None,
        plugin_root: Path | None,
        manifest_path: Path | None,
        runner_path: Path | None,
        spec_builder_path: Path | None = None,
    ) -> SkillDefinition:
        return SkillDefinition(
            name=self.name,
            description=self.description,
            output_kind=self.output_kind,
            generation=self.generation,
            model_tags=self.model_tags,
            quality_template=self.quality_template,
            skill_type=self.skill_type,
            child_skills=self.child_skills,
            stages=self.stages,
            done_when=self.done_when,
            routing=self.routing,
            execution=self.execution,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            sandbox=self.sandbox,
            source_type=source_type,
            plugin_id=plugin_id,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            plugin_root=plugin_root,
            manifest_path=manifest_path,
            runner_path=runner_path,
            spec_builder_path=spec_builder_path,
        )


class SkillRegistry:
    """Load skill manifests from installed skill plugins and optional config overrides."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or Path(__file__).resolve().parents[3]
        self.config_dir = self.root_dir / "config" / "skills"
        self._skills = self._load_skills()

    def list(self, allowed: list[str] | None = None, *, executable_only: bool = False) -> list[SkillDefinition]:
        names = allowed if allowed is not None else sorted(self._skills)
        skills = [self._skills[name] for name in names if name in self._skills]
        if executable_only:
            return [skill for skill in skills if skill.executable]
        return skills

    def get(self, name: str) -> SkillDefinition:
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name}")
        return self._skills[name]

    def read_manifest(self, name: str) -> dict[str, Any]:
        skill = self.get(name)
        path = self.manifest_source(name)
        if path.is_file():
            try:
                return self._manager().read_manifest(name)
            except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
                pass
        return skill.to_event_payload()

    def save_manifest(self, name: str, data: dict[str, Any]) -> SkillDefinition:
        self._validate_skill_name(name)
        loaded = self._manager().save_manifest(name, data)
        self._skills = self._load_skills()
        return self._skills.get(name, loaded.definition)

    def manifest_source(self, name: str) -> Path:
        self._validate_skill_name(name)
        return self._manager().manifest_path_for(name)

    def reload(self) -> None:
        self._skills = self._load_skills()

    def _load_skills(self) -> dict[str, SkillDefinition]:
        return {name: loaded.definition for name, loaded in self._manager().load_skills().items()}

    def _manifest_path(self, name: str) -> Path:
        self._validate_skill_name(name)
        return self.config_dir / f"{name}.json"

    def _manager(self) -> SkillPluginManager:
        from app.core.skills.plugins import SkillPluginManager

        return SkillPluginManager(self.root_dir)

    @staticmethod
    def _validate_skill_name(name: str) -> None:
        if not name or "/" in name or "\\" in name or name in {".", ".."} or ".." in name:
            raise ValueError(f"Invalid skill name: {name}")


def _definition_from_json(path: Path) -> SkillDefinition:
    data = json.loads(path.read_text(encoding="utf-8"))
    return definition_from_manifest(data, path)


def definition_from_manifest(data: object, path: Path) -> SkillDefinition:
    if not isinstance(data, dict):
        raise ValueError(f"Skill manifest must be a JSON object: {path}")

    name = _required_string(data, "name", path)
    description = _string_value(data.get("description")) or name
    output_kind = _string_value(data.get("output_kind")) or "json"
    sandbox_data = data.get("sandbox", {})
    sandbox = _sandbox_from_json(sandbox_data if isinstance(sandbox_data, dict) else {})
    return SkillDefinition(
        name=name,
        description=description,
        output_kind=output_kind,
        generation=bool(data.get("generation", True)),
        model_tags=tuple(normalize_model_tags(data.get("model_tags"))),
        quality_template=_string_tuple(data.get("quality_template")),
        skill_type=_skill_type(data),
        child_skills=_child_skills(data),
        stages=_dict_tuple(data.get("stages")),
        done_when=_string_tuple(data.get("done_when")),
        routing=_dict_value(data.get("routing")),
        execution=_dict_value(data.get("execution")),
        input_schema=_dict_value(data.get("input_schema")),
        output_schema=_dict_value(data.get("output_schema")),
        sandbox=sandbox,
        manifest_path=path,
    )


def _sandbox_from_json(data: dict[str, object]) -> SkillSandboxSpec:
    return SkillSandboxSpec(
        enabled=bool(data.get("enabled", False)),
        profile=_optional_string(data.get("profile")),
        request_schema_version=_string_value(data.get("request_schema_version")) or "skill-run.v1",
        adapter_command=_optional_string(data.get("adapter_command")),
        fallback_to_local=_optional_bool(data.get("fallback_to_local")),
    )


def _required_string(data: dict[str, object], key: str, path: Path) -> str:
    value = _string_value(data.get(key))
    if not value:
        raise ValueError(f"Skill manifest field {key!r} is required: {path}")
    return value


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _dict_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _dict_value(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return dict(value)


def _skill_type(data: dict[str, object]) -> str:
    raw_value = data.get("skill_type", data.get("kind"))
    if isinstance(raw_value, str) and raw_value.strip() == "composite":
        return "composite"
    return "atomic"


def _child_skills(data: dict[str, object]) -> tuple[str, ...]:
    raw_children = data.get("child_skills", data.get("children"))
    if not isinstance(raw_children, list):
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for item in raw_children:
        name = ""
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict) and isinstance(item.get("skill"), str):
            name = item["skill"].strip()
        if not name or name in seen:
            continue
        names.append(name)
        seen.add(name)
    return tuple(names)


def _has_generic_execution(execution: dict[str, Any] | None) -> bool:
    if not isinstance(execution, dict):
        return False
    return _string_value(execution.get("type")) in {"python_script", "template", "http", "command"}
