from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.auth import require_admin_token
from app.core.skills import SkillDefinition, SkillRegistry
from app.core.skills.plugins import SkillPluginManager


router = APIRouter(prefix="/api/skills", tags=["skills"])
registry = SkillRegistry()
plugin_manager = SkillPluginManager()


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
        return {"plugin": plugin.to_payload()}
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{skill_name}/files")
async def list_skill_files(skill_name: str) -> dict[str, list[dict[str, object]]]:
    try:
        return {"files": [package_file.to_payload() for package_file in plugin_manager.list_package_files(skill_name)]}
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        return _skill_payload(skill, include_manifest=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _skill_payload(skill: SkillDefinition, *, include_manifest: bool = False) -> dict[str, object]:
    payload = skill.to_event_payload()
    name = str(payload.get("name") or "")
    try:
        source = registry.manifest_source(name)
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
