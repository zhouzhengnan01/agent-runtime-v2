from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Protocol


PLUGIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return data


def read_manifest_source(path: Path) -> dict[str, Any]:
    if path.name == "SKILL.md":
        return manifest_from_skill_md(path.read_text(encoding="utf-8"), path.parent)
    data = read_json(path)
    if path.name == "manifest.json":
        return with_package_asset_contracts(data, path.parent)
    return data


def with_package_asset_contracts(data: dict[str, Any], package_root: Path) -> dict[str, Any]:
    merged = dict(data)
    input_schema = read_optional_json(package_root / "input.schema.json")
    output_schema = read_optional_json(package_root / "output.schema.json")
    if input_schema is not None:
        merged["input_schema"] = input_schema
    if output_schema is not None:
        merged["output_schema"] = output_schema
    return merged


def manifest_from_skill_md(text: str, package_root: Path) -> dict[str, Any]:
    metadata = skill_md_frontmatter(text)
    name = string_metadata(metadata.get("name")) or package_root.name
    description = string_metadata(metadata.get("description")) or name
    tags = string_list_metadata(metadata.get("tags"))
    return {
        "name": name,
        "description": description,
        "output_kind": infer_output_kind(name, tags),
        "generation": infer_generation(name, tags),
        "quality_template": tags,
        "input_schema": read_optional_json(package_root / "input.schema.json") or {"type": "object", "additionalProperties": True},
        "output_schema": read_optional_json(package_root / "output.schema.json") or {"type": "object", "additionalProperties": True},
        "sandbox": sandbox_from_package(package_root),
    }


def plugin_metadata_from_skill_md(text: str, package_root: Path) -> dict[str, Any]:
    metadata = skill_md_frontmatter(text)
    skill_name = validated_plugin_id(string_metadata(metadata.get("name")) or package_root.name)
    return {
        "id": skill_name,
        "name": string_metadata(metadata.get("name")) or skill_name,
        "version": "0.1.0",
        "description": string_metadata(metadata.get("description")) or skill_name,
        "skills": ["SKILL.md"],
    }


def skill_md_frontmatter(text: str) -> dict[str, object]:
    lines = text.lstrip().splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter.append(line.rstrip())

    data: dict[str, object] = {}
    current_key = ""
    for raw in frontmatter:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped.startswith("- ") and current_key:
            existing = data.setdefault(current_key, [])
            if isinstance(existing, list):
                existing.append(stripped[2:].strip().strip("\"'"))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        current_key = key.strip()
        normalized = value.strip()
        if not normalized:
            data[current_key] = []
        else:
            data[current_key] = normalized.strip("\"'")
    return data


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def json_object_from_text(content: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a valid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


class LoadedSkillLike(Protocol):
    @property
    def runner_path(self) -> Path | None: ...

    @property
    def spec_builder_path(self) -> Path | None: ...

    @property
    def manifest_path(self) -> Path: ...


def skill_package_root(loaded: LoadedSkillLike) -> Path | None:
    for path in (loaded.runner_path, loaded.spec_builder_path, loaded.manifest_path):
        if path is None:
            continue
        if path.name in {"runner.py", "spec_builder.py", "manifest.json", "SKILL.md"}:
            return path.parent.resolve()
    return None


def sandbox_from_package(package_root: Path) -> dict[str, object]:
    sandbox_path = package_root / "sandbox.yml"
    if not sandbox_path.is_file():
        return {"enabled": False, "profile": None, "request_schema_version": "skill-run.v1", "fallback_to_local": True}
    profile: str | None = None
    enabled: bool | None = None
    in_sandbox = False
    for raw in sandbox_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")) and stripped.endswith(":"):
            in_sandbox = stripped[:-1] == "sandbox"
            continue
        if not in_sandbox or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        normalized = value.strip().strip("\"'")
        if key.strip() == "profile":
            profile = normalized or None
        elif key.strip() == "enabled":
            enabled = normalized.lower() in {"1", "true", "yes", "on"}
    return {
        "enabled": bool(profile) if enabled is None else enabled,
        "profile": profile,
        "request_schema_version": "skill-run.v1",
        "fallback_to_local": True,
    }


def infer_output_kind(name: str, tags: list[str]) -> str:
    haystack = " ".join([name, *tags]).lower()
    if "ppt" in haystack or "powerpoint" in haystack:
        return "presentation"
    if "excel" in haystack or "xlsx" in haystack:
        return "spreadsheet"
    if "drawio" in haystack or "diagram" in haystack:
        return "drawio"
    if "xmind" in haystack or "mindmap" in haystack:
        return "mindmap"
    if "docx" in haystack or "deliverable" in haystack:
        return "document"
    if "markdown" in haystack:
        return "markdown"
    return "json"


def infer_generation(name: str, tags: list[str]) -> bool:
    haystack = " ".join([name, *tags]).lower()
    return "detection" not in haystack and "detect" not in haystack


def string_metadata(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def string_list_metadata(value: object) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def validated_plugin_id(value: str) -> str:
    if not PLUGIN_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid plugin id: {value}")
    return value


def safe_zip_members(archive: zipfile.ZipFile) -> list[str]:
    members: list[str] = []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"Invalid plugin archive path: {info.filename}")
        if name and not name.endswith("/"):
            members.append(name)
    return members


def single_root_prefix(members: list[str]) -> str:
    roots = {member.split("/", 1)[0] for member in members if "/" in member}
    top_level_files = [member for member in members if "/" not in member]
    if len(roots) == 1 and not top_level_files:
        return f"{next(iter(roots))}/"
    return ""
