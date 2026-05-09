from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.auth import require_admin_token
from app.core.runtime import default_container
from app.core.workflow import WorkflowConfig, WorkflowPluginManager, WorkflowRegistry


router = APIRouter(prefix="/api/workflows", tags=["workflows"])
runtime = default_container.runtime
plugin_manager = WorkflowPluginManager()


@router.get("")
async def list_workflows() -> dict[str, list[dict[str, object]]]:
    return {"workflows": [_workflow_payload(item.config) for item in runtime.workflow_registry.list_registered()]}


@router.get("/plugins")
async def list_workflow_plugins() -> dict[str, list[dict[str, object]]]:
    return {"plugins": [plugin.to_payload() for plugin in plugin_manager.list_plugins()]}


@router.post("/plugins", dependencies=[Depends(require_admin_token)])
async def upload_workflow_plugin(file: UploadFile = File(...)) -> dict[str, object]:
    content = await file.read()
    try:
        plugin = plugin_manager.install_zip(content)
        _reload_runtime_registry()
        return {"plugin": plugin.to_payload()}
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/plugins/{plugin_id}", dependencies=[Depends(require_admin_token)])
async def delete_workflow_plugin(plugin_id: str) -> dict[str, object]:
    try:
        plugin = plugin_manager.delete_plugin(plugin_id)
        _reload_runtime_registry()
        return {"plugin": plugin.to_payload(), "deleted": True}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _workflow_payload(config: WorkflowConfig) -> dict[str, object]:
    return {
        "name": config.name,
        "display_name": config.display_name,
        "description": config.description,
        "enabled": config.enabled,
        "handler": config.handler,
        "trigger": config.trigger,
        "metadata": config.metadata,
        "request_example": {
            "runtime_options": {
                "workflow": config.name,
            },
        },
    }


def _reload_runtime_registry() -> None:
    runtime.workflow_registry = WorkflowRegistry.builtin(runtime.artifact_store, root_dir=plugin_manager.root_dir)
