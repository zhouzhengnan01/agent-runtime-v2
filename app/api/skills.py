from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.auth import require_admin_token
from app.core.sandbox.config import load_sandbox_config
from app.core.sandbox.env_cache import SkillEnvironmentCache
from app.core.sandbox.policy import load_sandbox_policy
from app.core.skills import SkillDefinition, SkillRegistry
from app.core.skills.aliases import invalidate_skill_alias_cache
from app.core.skills.local_subprocess import LocalSubprocessEnvironmentCache
from app.core.skills.plugins import SkillPluginManager


router = APIRouter(prefix="/api/skills", tags=["skills"])
registry = SkillRegistry()
plugin_manager = SkillPluginManager()
environment_cache = SkillEnvironmentCache()
local_environment_cache = LocalSubprocessEnvironmentCache()


@router.get("")
async def list_skills() -> dict[str, list[dict[str, object]]]:
    return {"skills": [_skill_payload(skill) for skill in registry.list()]}


@router.get("/plugins")
async def list_skill_plugins() -> dict[str, list[dict[str, object]]]:
    return {"plugins": [plugin.to_payload() for plugin in plugin_manager.list_plugins()]}


@router.post("/plugins", dependencies=[Depends(require_admin_token)])
async def upload_skill_plugin(file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    try:
        plugin = plugin_manager.install_zip(content)
        registry.reload()
        environments = [_environment_summary(skill_name) for skill_name in plugin.manifest_paths]
        return {"plugin": plugin.to_payload(), "environments": environments}
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/plugins/local-path", dependencies=[Depends(require_admin_token)])
async def install_skill_plugin_from_local_path(payload: dict[str, Any]) -> dict[str, object]:
    raw_path = payload.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(status_code=400, detail="Skill plugin zip path is required.")
    path = Path(raw_path.strip()).expanduser()
    try:
        if not path.is_file():
            raise ValueError(f"Skill plugin zip not found: {path}")
        if path.suffix.lower() != ".zip":
            raise ValueError("Skill plugin local path must point to a .zip file.")
        plugin = plugin_manager.install_zip(path.read_bytes())
        registry.reload()
        environments = [_environment_summary(skill_name) for skill_name in plugin.manifest_paths]
        return {"plugin": plugin.to_payload(), "environments": environments}
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/plugins/cache/invalidate", dependencies=[Depends(require_admin_token)])
async def invalidate_skill_plugin_cache() -> dict[str, object]:
    invalidate_skill_alias_cache()
    registry.reload()
    return {"invalidated": True}


@router.delete("/plugins/{plugin_id}", dependencies=[Depends(require_admin_token)])
async def delete_skill_plugin(plugin_id: str) -> dict[str, object]:
    try:
        plugin = plugin_manager.delete_plugin(plugin_id)
        registry.reload()
        return {"plugin": plugin.to_payload(), "deleted": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{skill_name}/files")
async def list_skill_files(skill_name: str) -> dict[str, list[dict[str, object]]]:
    try:
        return {"files": [package_file.to_payload() for package_file in plugin_manager.list_package_files(skill_name)]}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{skill_name}/environment")
async def get_skill_environment(skill_name: str) -> dict[str, object]:
    try:
        loaded = plugin_manager.get_loaded_skill(skill_name)
        sandbox_config = load_sandbox_config()
        policy = load_sandbox_policy(sandbox_config, skill_registry=registry)
        decision = policy.resolve(skill_name, sandbox_config)
        package_root = _skill_package_root(loaded)
        requirements_text = environment_cache.requirements_text(package_root)
        requirements_hash = None
        if requirements_text:
            from app.core.sandbox.env_cache import requirements_hash_for_text

            profile_image = decision.profile.image if decision.profile else ""
            base_image = environment_cache._base_image(profile_image, requirements_text, sandbox_config)
            requirements_hash = requirements_hash_for_text(f"# base-image: {base_image}\n{requirements_text}")
        return {
            "skill_name": skill_name,
            "enabled": sandbox_config.skill_env_cache_enabled,
            "sandbox_enabled": decision.use_sandbox,
            "profile_name": decision.profile_name,
            "base_image": decision.profile.image if decision.profile else None,
            "requirements_hash": requirements_hash,
            "has_requirements": bool(requirements_text),
            "status": _environment_status(requirements_hash),
            "local_subprocess": _local_environment_payload(skill_name, loaded),
        }
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{skill_name}/environment/warm", dependencies=[Depends(require_admin_token)])
async def warm_skill_environment(skill_name: str) -> dict[str, object]:
    try:
        loaded = plugin_manager.get_loaded_skill(skill_name)
        sandbox_config = load_sandbox_config()
        policy = load_sandbox_policy(sandbox_config, skill_registry=registry)
        decision = policy.resolve(skill_name, sandbox_config)
        if decision.profile is None:
            raise ValueError("Skill has no sandbox profile.")
        environment = environment_cache.prepare(
            skill_name=skill_name,
            package_root=_skill_package_root(loaded),
            profile=decision.profile,
            config=sandbox_config,
        )
        return {
            "skill_name": skill_name,
            "requirements_hash": environment.requirements_hash,
            "image": environment.image,
            "status": environment.status,
            "message": environment.message,
            "local_subprocess": _warm_local_environment_if_configured(skill_name),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{skill_name}/files/{file_id}")
async def get_skill_file(skill_name: str, file_id: str) -> dict[str, object]:
    try:
        package_file, content = plugin_manager.read_package_file(skill_name, file_id)
        return {"file": package_file.to_payload(), "content": content}
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{skill_name}/files/{file_id}", dependencies=[Depends(require_admin_token)])
async def update_skill_file(skill_name: str, file_id: str, payload: dict[str, Any]) -> dict[str, object]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="Skill package file content must be a string.")
    try:
        package_file, saved_content = plugin_manager.write_package_file(skill_name, file_id, content)
        registry.reload()
        skill = registry.get(skill_name)
        return {
            "file": package_file.to_payload(),
            "content": saved_content,
            "skill": _skill_payload(skill, include_manifest=True),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/{skill_name}/manifest-config", dependencies=[Depends(require_admin_token)])
async def configure_skill_manifest(skill_name: str, payload: dict[str, Any]) -> dict[str, object]:
    raw_config = payload.get("config") if isinstance(payload.get("config"), dict) else payload
    if not isinstance(raw_config, dict):
        raise HTTPException(status_code=400, detail="Skill manifest config must be a JSON object.")
    try:
        existing = registry.read_manifest(skill_name)
    except (KeyError, ValueError, OSError, TypeError, json.JSONDecodeError):
        existing = {}
    try:
        manifest = _manifest_from_config(skill_name, raw_config, existing)
        skill = registry.save_manifest(skill_name, manifest)
        payload = _skill_payload(skill, include_manifest=True)
        payload["environment"] = _warm_skill_environment_if_configured(skill_name)
        return payload
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{skill_name}")
async def get_skill(skill_name: str) -> dict[str, object]:
    try:
        skill = registry.get(skill_name)
        return _skill_payload(skill, include_manifest=True)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/{skill_name}", dependencies=[Depends(require_admin_token)])
async def update_skill(skill_name: str, payload: dict[str, Any]) -> dict[str, object]:
    manifest = payload.get("manifest") if isinstance(payload.get("manifest"), dict) else payload
    if not isinstance(manifest, dict):
        raise HTTPException(status_code=400, detail="Skill manifest must be a JSON object.")
    try:
        skill = registry.save_manifest(skill_name, manifest)
        payload = _skill_payload(skill, include_manifest=True)
        payload["environment"] = _warm_skill_environment_if_configured(skill_name)
        return payload
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _manifest_from_config(skill_name: str, config: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(existing)
    manifest["name"] = skill_name
    manifest["description"] = _string_config(config.get("description"), manifest.get("description") or skill_name)
    manifest["output_kind"] = _string_config(config.get("output_kind"), manifest.get("output_kind") or "json")
    manifest["generation"] = _bool_config(config.get("generation"), bool(manifest.get("generation", True)))
    manifest["quality_template"] = _string_list_config(config.get("quality_template"), manifest.get("quality_template"))
    routing = config.get("routing", manifest.get("routing"))
    if isinstance(routing, dict):
        manifest["routing"] = dict(routing)
    execution = config.get("execution", manifest.get("execution"))
    if isinstance(execution, dict):
        manifest["execution"] = _execution_config(execution)
    manifest["input_schema"] = _object_config(config.get("input_schema"), "input_schema")
    manifest["output_schema"] = _object_config(config.get("output_schema"), "output_schema")
    manifest["sandbox"] = _sandbox_config(config.get("sandbox"), manifest.get("sandbox"))
    return manifest


def _string_config(value: object, default: object) -> str:
    if isinstance(value, str):
        return value.strip() or str(default or "")
    return str(default or "")


def _bool_config(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _string_list_config(value: object, default: object) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.replace("，", ",").replace("\n", ",").split(",") if item.strip()]
    if isinstance(default, list):
        return [item for item in default if isinstance(item, str)]
    return []


def _object_config(value: object, label: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"Skill manifest config field {label!r} must be a JSON object.")


def _sandbox_config(value: object, default: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else default if isinstance(default, dict) else {}
    fallback = source.get("fallback_to_local")
    return {
        "enabled": _bool_config(source.get("enabled"), False),
        "profile": _nullable_string(source.get("profile")),
        "request_schema_version": _string_config(source.get("request_schema_version"), "skill-run.v1") or "skill-run.v1",
        "adapter_command": _nullable_string(source.get("adapter_command")),
        "fallback_to_local": fallback if isinstance(fallback, bool) else None,
    }


def _execution_config(value: dict[str, Any]) -> dict[str, Any]:
    execution = dict(value)
    execution_type = _string_config(execution.get("type"), "").strip()
    if not execution_type:
        return {}
    if execution_type == "python_script" and not _nullable_string(execution.get("script")):
        raise ValueError("Python script execution requires execution.script.")
    return execution


def _nullable_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _skill_package_root(loaded: Any) -> Path:
    plugin = getattr(loaded, "plugin", None)
    plugin_root = getattr(plugin, "root", None)
    if isinstance(plugin_root, Path):
        return plugin_root
    manifest_path = getattr(loaded, "manifest_path", None)
    if isinstance(manifest_path, Path):
        return manifest_path.parent
    raise ValueError("Loaded skill has no package root.")


def _warm_skill_environment_if_configured(skill_name: str) -> dict[str, object]:
    local_payload = _warm_local_environment_if_configured(skill_name)
    try:
        loaded = plugin_manager.get_loaded_skill(skill_name)
        sandbox_config = load_sandbox_config()
        policy = load_sandbox_policy(sandbox_config, skill_registry=registry)
        decision = policy.resolve(skill_name, sandbox_config)
        if not sandbox_config.skill_env_cache_enabled or not decision.use_sandbox or decision.profile is None:
            return {
                "skill_name": skill_name,
                "status": "skipped",
                "reason": decision.reason,
                "cache_enabled": sandbox_config.skill_env_cache_enabled,
                "local_subprocess": local_payload,
            }
        environment = environment_cache.prepare(
            skill_name=skill_name,
            package_root=_skill_package_root(loaded),
            profile=decision.profile,
            config=sandbox_config,
        )
        return {
            "skill_name": skill_name,
            "requirements_hash": environment.requirements_hash,
            "image": environment.image,
            "status": environment.status,
            "message": environment.message,
            "local_subprocess": local_payload,
        }
    except Exception as exc:
        return {"skill_name": skill_name, "status": "failed", "message": str(exc), "local_subprocess": local_payload}


def _environment_summary(skill_name: str) -> dict[str, object]:
    try:
        loaded = plugin_manager.get_loaded_skill(skill_name)
        return {
            "skill_name": skill_name,
            "status": "pending",
            "local_subprocess": _local_environment_payload(skill_name, loaded),
        }
    except Exception as exc:
        return {"skill_name": skill_name, "status": "unknown", "message": str(exc)}


def _warm_local_environment_if_configured(skill_name: str) -> dict[str, object]:
    try:
        loaded = plugin_manager.get_loaded_skill(skill_name)
        manifest = plugin_manager.read_manifest(skill_name)
        execution = manifest.get("execution")
        if not isinstance(execution, dict) or (execution.get("runtime") or execution.get("mode")) != "local_subprocess":
            return {"skill_name": skill_name, "status": "skipped", "reason": "execution.runtime is not local_subprocess"}
        environment = local_environment_cache.prepare(
            skill_name=skill_name,
            package_root=_skill_package_root(loaded),
            base_python=str(execution.get("python")) if isinstance(execution.get("python"), str) else None,
            install_timeout_seconds=int(execution.get("install_timeout_seconds") or 1800),
        )
        return {
            "skill_name": skill_name,
            "requirements_hash": environment.requirements_hash,
            "python": str(environment.python),
            "status": environment.status,
            "message": environment.message,
        }
    except Exception as exc:
        return {"skill_name": skill_name, "status": "failed", "message": str(exc)}


def _local_environment_payload(skill_name: str, loaded: Any) -> dict[str, object]:
    manifest = plugin_manager.read_manifest(skill_name)
    execution = manifest.get("execution")
    runtime = execution.get("runtime") or execution.get("mode") if isinstance(execution, dict) else None
    package_root = _skill_package_root(loaded)
    requirements_text = local_environment_cache.requirements_text(package_root)
    payload: dict[str, object] = {
        "enabled": runtime == "local_subprocess",
        "runtime": runtime or "",
        "has_requirements": bool(requirements_text),
        "status": local_environment_cache.status(
            skill_name=skill_name,
            package_root=package_root,
            base_python=str(execution.get("python")) if isinstance(execution, dict) and isinstance(execution.get("python"), str) else None,
        ),
    }
    return payload


def _skill_payload(skill: SkillDefinition, *, include_manifest: bool = False) -> dict[str, object]:
    payload = skill.to_event_payload()
    name = str(payload.get("name") or "")
    try:
        source = skill.manifest_path if skill.manifest_path is not None else registry.manifest_source(name)
        payload["editable"] = True
        payload["source_path"] = str(source)
        payload["source_exists"] = source.is_file()
        if include_manifest:
            manifest = registry.read_manifest(name)
            payload["manifest"] = manifest
            try:
                payload["package_files"] = [package_file.to_payload() for package_file in plugin_manager.list_package_files(name)]
            except (KeyError, ValueError, OSError):
                payload["package_files"] = []
            payload["source_text"] = (
                source.read_text(encoding="utf-8")
                if source.is_file()
                else json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
            )
    except (KeyError, ValueError, OSError):
        payload["editable"] = False
        payload["source_path"] = ""
        if include_manifest:
            payload["manifest"] = dict(payload)
            payload["package_files"] = []
            payload["source_exists"] = False
            payload["source_text"] = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return payload


def _environment_status(requirements_hash: str | None) -> dict[str, object]:
    if not requirements_hash:
        return {"status": "base", "image": None}
    index = environment_cache._read_index()
    entry = index.get(requirements_hash)
    return entry if isinstance(entry, dict) else {"status": "pending", "image": None}
