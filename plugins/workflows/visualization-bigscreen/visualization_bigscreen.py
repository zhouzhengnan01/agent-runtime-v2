from __future__ import annotations

import copy
import httpx
import json
import os
import re
import time
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm import OpenAICompatibleClient
from app.core.skills import SkillRegistry
from app.core.tools.invocation import ToolInvocationService
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


WORKFLOW_NAME = "visualization_bigscreen_workflow"
BIGSCREEN_OUTPUT_DIR = "generated-bigscreen"
BACKGROUND_PATH = f"{BIGSCREEN_OUTPUT_DIR}/background.svg"
RESOURCE_DIR = f"{BIGSCREEN_OUTPUT_DIR}/resources"
ADVANCED_COMPONENT_DIR = f"{BIGSCREEN_OUTPUT_DIR}/advanced-components"
DEFAULT_RESOURCE_GROUP = "[\"vis_oneself_dimension_line-chart\"]"
SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
DEFAULT_VISUALIZATION_SKILL = "ai-vis-page"
PSEUDO_TEMPLATE_FILES = (
    SKILLS_ROOT / DEFAULT_VISUALIZATION_SKILL / "references" / "components" / "pseudo.json",
    SKILLS_ROOT / "1780654477379uykyzod7" / "references" / "components" / "pseudo.json",
)


BLUEPRINT_REGIONS: list[dict[str, Any]] = [
    {
        "id": "header",
        "name": "顶部标题区",
        "role": "header",
        "x": 32,
        "y": 24,
        "width": 1856,
        "height": 84,
        "contentBox": {"x": 56, "y": 40, "width": 1808, "height": 52},
        "slots": [
            {"id": "header_left", "name": "左侧辅助信息", "x": 56, "y": 40, "width": 360, "height": 52, "preferredComponents": ["text"]},
            {"id": "header_title", "name": "主标题", "x": 560, "y": 40, "width": 800, "height": 52, "preferredComponents": ["text"]},
            {"id": "header_time", "name": "右侧时间", "x": 1540, "y": 40, "width": 324, "height": 52, "preferredComponents": ["dateTime"]},
        ],
    },
    {
        "id": "global_metrics",
        "name": "全局指标区",
        "role": "metrics",
        "x": 32,
        "y": 124,
        "width": 1856,
        "height": 112,
        "contentBox": {"x": 56, "y": 148, "width": 1808, "height": 64},
        "slots": [
            {"id": "metric_1", "name": "指标卡一", "x": 56, "y": 148, "width": 344, "height": 64, "preferredComponents": ["text"]},
            {"id": "metric_2", "name": "指标卡二", "x": 422, "y": 148, "width": 344, "height": 64, "preferredComponents": ["text"]},
            {"id": "metric_3", "name": "指标卡三", "x": 788, "y": 148, "width": 344, "height": 64, "preferredComponents": ["text"]},
            {"id": "metric_4", "name": "指标卡四", "x": 1154, "y": 148, "width": 344, "height": 64, "preferredComponents": ["text"]},
            {"id": "metric_5", "name": "指标卡五", "x": 1520, "y": 148, "width": 344, "height": 64, "preferredComponents": ["text"]},
        ],
    },
    {
        "id": "left_top_panel",
        "name": "左上分析区",
        "role": "analysis",
        "x": 32,
        "y": 260,
        "width": 420,
        "height": 236,
        "contentBox": {"x": 56, "y": 316, "width": 372, "height": 156},
        "slots": [{"id": "left_top_body", "name": "左上主体内容", "x": 56, "y": 316, "width": 372, "height": 156, "preferredComponents": ["custom-chart", "table", "text", "video"]}],
    },
    {
        "id": "left_middle_panel",
        "name": "左中趋势区",
        "role": "trend",
        "x": 32,
        "y": 520,
        "width": 420,
        "height": 236,
        "contentBox": {"x": 56, "y": 576, "width": 372, "height": 156},
        "slots": [{"id": "left_middle_body", "name": "左中主体内容", "x": 56, "y": 576, "width": 372, "height": 156, "preferredComponents": ["custom-chart"]}],
    },
    {
        "id": "left_bottom_panel",
        "name": "左下排行区",
        "role": "ranking",
        "x": 32,
        "y": 780,
        "width": 420,
        "height": 268,
        "contentBox": {"x": 56, "y": 836, "width": 372, "height": 188},
        "slots": [{"id": "left_bottom_body", "name": "左下主体内容", "x": 56, "y": 836, "width": 372, "height": 188, "preferredComponents": ["table", "custom-chart"]}],
    },
    {
        "id": "center_main_panel",
        "name": "中心主视觉区",
        "role": "main-visual",
        "x": 476,
        "y": 260,
        "width": 968,
        "height": 548,
        "contentBox": {"x": 500, "y": 316, "width": 920, "height": 468},
        "slots": [{"id": "center_main_visual", "name": "中心主视觉内容", "x": 500, "y": 316, "width": 920, "height": 468, "preferredComponents": ["pseudo", "custom-chart", "video"]}],
    },
    {
        "id": "center_bottom_panel",
        "name": "中心底部趋势区",
        "role": "bottom-trend",
        "x": 476,
        "y": 832,
        "width": 968,
        "height": 216,
        "contentBox": {"x": 500, "y": 888, "width": 920, "height": 136},
        "slots": [{"id": "center_bottom_body", "name": "中心底部主体内容", "x": 500, "y": 888, "width": 920, "height": 136, "preferredComponents": ["custom-chart", "table", "video"]}],
    },
    {
        "id": "right_top_panel",
        "name": "右上状态区",
        "role": "status",
        "x": 1468,
        "y": 260,
        "width": 420,
        "height": 236,
        "contentBox": {"x": 1492, "y": 316, "width": 372, "height": 156},
        "slots": [{"id": "right_top_body", "name": "右上主体内容", "x": 1492, "y": 316, "width": 372, "height": 156, "preferredComponents": ["custom-chart", "table", "text", "video"]}],
    },
    {
        "id": "right_middle_panel",
        "name": "右中告警区",
        "role": "alerts",
        "x": 1468,
        "y": 520,
        "width": 420,
        "height": 236,
        "contentBox": {"x": 1492, "y": 576, "width": 372, "height": 156},
        "slots": [{"id": "right_middle_body", "name": "右中主体内容", "x": 1492, "y": 576, "width": 372, "height": 156, "preferredComponents": ["table"]}],
    },
    {
        "id": "right_bottom_panel",
        "name": "右下分布区",
        "role": "distribution",
        "x": 1468,
        "y": 780,
        "width": 420,
        "height": 268,
        "contentBox": {"x": 1492, "y": 836, "width": 372, "height": 188},
        "slots": [{"id": "right_bottom_body", "name": "右下主体内容", "x": 1492, "y": 836, "width": 372, "height": 188, "preferredComponents": ["custom-chart", "table"]}],
    },
]


class VisualizationBigscreenWorkflow:
    """Workflow for the Java-orchestrated visualization page generator."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.tool_service = ToolInvocationService(artifact_store=artifact_store)

    def run_with_events(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        attachments: list[Attachment],
        thread_id: str | None,
        on_event: Callable[[ChatEvent], None] | None = None,
        workflow_name: str | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        started_at = time.perf_counter()
        runtime_options = runtime_options or RuntimeOptions()
        workflow = workflow_name or WORKFLOW_NAME
        paths = self.artifact_store.prepare_thread(thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id, on_emit=on_event)
        recorder.emit(
            "run.started",
            {
                "workflow": workflow,
                "execution_mode": workflow,
                "app_template": runtime_options.app_template_name or "",
            },
        )

        prompt_text = _last_user_text(messages)
        stage = _detect_stage(prompt_text)
        recorder.emit("visualization.stage.detected", {"stage": stage})

        try:
            if stage == "initialization":
                reply, metadata = self._run_initialization(agent_config, messages, paths, runtime_options, recorder)
            elif stage == "region":
                reply, metadata = self._run_region(agent_config, messages, paths, runtime_options, recorder)
            else:
                raise ValueError("未识别可视化生成阶段，请在提示词中包含 当前阶段：initialization 或 当前阶段：region。")
        except Exception as exc:
            reply = f"请求处理失败：可视化工作流执行失败：{exc}"
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="failed",
                reply=reply,
                metadata={
                    "workflow": workflow,
                    "stage": stage,
                    "error": str(exc),
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.failed", {"result": result.model_dump()})
            return result, recorder.events

        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=paths.thread_id,
            status="completed",
            reply=reply,
            metadata={
                "workflow": workflow,
                "stage": stage,
                **metadata,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        )
        recorder.emit("agent.message", {"text": reply})
        recorder.emit("run.completed", {"result": result.model_dump()})
        return result, recorder.events

    def _run_initialization(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        paths: ThreadPaths,
        runtime_options: RuntimeOptions,
        recorder: EventRecorder,
    ) -> tuple[str, dict[str, Any]]:
        recorder.emit("llm.started", _llm_event_payload(agent_config, runtime_options))
        llm_reply = self._complete_json(
            agent_config,
            runtime_options,
            _initialization_system_prompt(),
            messages,
            recorder,
            stage="initialization",
        )
        payload = _loads_json_object(llm_reply)
        page_json = _dict(payload.get("pageJson"))
        blueprint = _dict(payload.get("blueprint")) or _default_blueprint()
        background_svg = _string(payload.get("backgroundSvg") or payload.get("backgroundSVG") or payload.get("svg"))
        if not background_svg:
            background_svg = _fallback_background_svg()
        background_svg = _ensure_svg(background_svg)
        page_json = _normalize_page_json(page_json)
        blueprint = _normalize_blueprint(blueprint)

        artifact = self.artifact_store.write_text_artifact(paths, BACKGROUND_PATH, background_svg)
        recorder.emit(
            "artifact.created",
            {"path": f"outputs/{BACKGROUND_PATH}", "name": artifact.name, "mime_type": artifact.mime_type},
        )
        upload_result = self._upload_background(paths, runtime_options, recorder)
        file_id = _find_id(upload_result.structured_content) or _find_id(upload_result.content)
        if not file_id:
            raise RuntimeError("背景图上传成功但平台返回缺少 id/fileId。")
        page_json.setdefault("canvas", {}).setdefault("backgroundImage", {})["fileId"] = file_id
        reply = _json_reply({"pageJson": page_json, "blueprint": blueprint})
        return reply, {"background_file_id": file_id, "background_artifact": f"outputs/{BACKGROUND_PATH}"}

    def _run_region(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        paths: ThreadPaths,
        runtime_options: RuntimeOptions,
        recorder: EventRecorder,
    ) -> tuple[str, dict[str, Any]]:
        prompt_text = _last_user_text(messages)
        requested_region_id = _requested_region_id(prompt_text)
        recorder.emit("llm.started", _llm_event_payload(agent_config, runtime_options))
        llm_reply = self._complete_json(
            agent_config,
            runtime_options,
            _region_system_prompt(requested_region_id),
            messages,
            recorder,
            stage="region",
        )
        payload = _loads_json_object(llm_reply)
        region_id = _string(payload.get("regionId")) or requested_region_id
        if requested_region_id and region_id != requested_region_id:
            region_id = requested_region_id
        components = _list(payload.get("components"))
        advanced_components = _advanced_components_from_payload(payload)
        _append_advanced_components(components, advanced_components)
        _normalize_pseudo_components(components, region_id)
        _validate_component_types(components)
        _validate_pseudo_components(components)
        _validate_region_title_component(region_id, _string(payload.get("regionTitle")), components)
        _validate_map_intent_components(region_id, prompt_text, _string(payload.get("regionTitle")), components)
        resources = _resources_from_payload(payload)
        resources.extend(_resources_from_components(components))
        resources.extend(
            self._process_advanced_components(
                advanced_components,
                components,
                paths,
                runtime_options,
                recorder,
            )
        )
        resources = _dedupe_resources(resources)

        saved_count = 0
        for resource in resources:
            normalized = _normalize_resource(resource)
            resource_id = _string(normalized.get("resourceId"))
            if not resource_id:
                raise ValueError("custom-chart resource 缺少 resourceId。")
            artifact = self.artifact_store.write_text_artifact(
                paths,
                f"{RESOURCE_DIR}/{_safe_file_stem(resource_id)}.resource.json",
                _json_reply(normalized),
            )
            recorder.emit(
                "artifact.created",
                {"path": f"outputs/{RESOURCE_DIR}/{artifact.name}", "name": artifact.name, "mime_type": artifact.mime_type},
            )
            self._save_resource(normalized, runtime_options, recorder)
            saved_count += 1
        _validate_component_resources(components, resources)
        reply = _json_reply({"regionId": region_id, "components": components})
        return reply, {"resource_save_count": saved_count, "advanced_component_count": len(advanced_components)}

    def _complete_json(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        system_prompt: str,
        messages: list[Message],
        recorder: EventRecorder,
        stage: str,
    ) -> str:
        llm_base_options, direct_llm = _direct_llm_runtime_options(runtime_options)
        if direct_llm.get("enabled"):
            recorder.emit("llm.direct_override.enabled", direct_llm)
        llm_options = llm_base_options.model_copy(
            update={
                "selected_mcp_tools": [],
                "response_format": "json",
                "config_options": {**llm_base_options.config_options, "enableWorkspaceTools": False},
            },
            deep=True,
        )
        client = OpenAICompatibleClient(agent_config, runtime_options=llm_options)
        if llm_options.request_timeout_seconds is not None:
            client.request_timeout_seconds = llm_options.request_timeout_seconds
        prompt_text = _last_user_text(messages)
        full_system_prompt = self._prompt_with_stage_context(
            system_prompt,
            llm_options,
            recorder,
            stage,
            prompt_text=prompt_text,
        )
        try:
            reply = client.complete_sync(full_system_prompt, messages)
        except httpx.HTTPStatusError as exc:
            if not _is_retryable_llm_status_error(exc):
                raise
            recorder.emit(
                "llm.request.retry",
                {
                    "reason": "upstream_http_error",
                    "status_code": exc.response.status_code if exc.response is not None else None,
                    "strategy": "stage_aware_context_retry",
                    "stage": stage,
                },
            )
            reply = client.complete_sync(full_system_prompt, messages)
        if not reply.strip():
            raise RuntimeError("模型返回为空。")
        return reply

    def _prompt_with_stage_context(
        self,
        system_prompt: str,
        runtime_options: RuntimeOptions,
        recorder: EventRecorder,
        stage: str,
        *,
        prompt_text: str = "",
    ) -> str:
        skill_name = next((name.strip() for name in runtime_options.selected_skills if name.strip()), "")
        requested_skill_name = skill_name or DEFAULT_VISUALIZATION_SKILL
        skill = None
        try:
            skill = SkillRegistry(self.tool_service.root_dir).get(requested_skill_name)
        except KeyError:
            recorder.emit(
                "skill.context.loaded",
                {
                    "skill_name": requested_skill_name,
                    "found": False,
                    "reason": "skill_not_found",
                    "stage_aware": True,
                    "stage": stage,
                },
            )
            if requested_skill_name != DEFAULT_VISUALIZATION_SKILL:
                try:
                    skill = SkillRegistry(self.tool_service.root_dir).get(DEFAULT_VISUALIZATION_SKILL)
                except KeyError:
                    skill = None
        package_root = _skill_package_root(skill) if skill is not None else None
        if package_root is None:
            package_root = _default_visualization_package_root()
        if package_root is None:
            return system_prompt
        context, reference_paths = _stage_aware_visualization_context(
            package_root,
            stage,
            prompt_text=prompt_text,
            requested_region_id=_requested_region_id(prompt_text),
        )
        recorder.emit(
            "skill.context.loaded",
            {
                "skill_name": skill.name if skill is not None else package_root.name,
                "requested_skill_name": requested_skill_name,
                "found": True,
                "compact": True,
                "stage_aware": True,
                "stage": stage,
                "primary_context": True,
                "chars": len(context),
                "reference_count": len(reference_paths),
                "references": [{"path": path} for path in reference_paths],
            },
        )
        if not context:
            return system_prompt
        return f"{system_prompt}\n\n{context}"

    def _upload_background(self, paths: ThreadPaths, runtime_options: RuntimeOptions, recorder: EventRecorder):
        tool_name = "visual_bigscreen_upload_file"
        tool_call_id = f"workflow-{tool_name}"
        arguments = self._tool_runtime_arguments(paths, runtime_options)
        arguments.update(
            {
                "file_path": f"/mnt/user-data/outputs/{BACKGROUND_PATH}",
                "file_name": "background.svg",
                "content_type": "image/svg+xml;charset=UTF-8",
            }
        )
        recorder.emit("tool.started", {"tool_name": tool_name, "tool_call_id": tool_call_id})
        result = self.tool_service.call_tool(tool_name, arguments)
        recorder.emit(
            "tool.completed" if not result.is_error else "tool.failed",
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "is_error": result.is_error,
                "arguments": {"file_path": arguments["file_path"], "file_name": arguments["file_name"]},
                "structured_content": result.structured_content,
            },
        )
        if result.is_error:
            raise RuntimeError(_tool_error_text(result))
        return result

    def _process_advanced_components(
        self,
        advanced_components: list[dict[str, Any]],
        components: list[Any],
        paths: ThreadPaths,
        runtime_options: RuntimeOptions,
        recorder: EventRecorder,
    ) -> list[dict[str, Any]]:
        resources: list[dict[str, Any]] = []
        for package in advanced_components:
            files = _advanced_component_files(package)
            if not files:
                continue
            resource = _dict(package.get("resource") or package.get("resourceEntity"))
            resource_id = _advanced_resource_id(package, resource, _dict(package.get("component") or package.get("componentJson")))
            if not resource_id:
                raise ValueError("advanced component package is missing resourceId.")
            component = _advanced_component_for_package(package, components, resource_id)
            _validate_advanced_component_files(resource_id, files)

            safe_resource_id = _safe_file_stem(resource_id)
            for file_item in files:
                artifact_path = f"{ADVANCED_COMPONENT_DIR}/{safe_resource_id}/{file_item['path']}"
                artifact = self.artifact_store.write_text_artifact(paths, artifact_path, file_item["content"])
                recorder.emit(
                    "artifact.created",
                    {"path": f"outputs/{artifact_path}", "name": artifact.name, "mime_type": artifact.mime_type},
                )

            zip_name = f"{safe_resource_id}.zip"
            zip_path = f"{ADVANCED_COMPONENT_DIR}/{zip_name}"
            zip_artifact = self.artifact_store.write_bytes_artifact(paths, zip_path, _zip_component_files(files))
            recorder.emit(
                "artifact.created",
                {"path": f"outputs/{zip_path}", "name": zip_artifact.name, "mime_type": zip_artifact.mime_type},
            )

            upload_result = self._upload_advanced_component_zip(paths, zip_path, zip_name, runtime_options, recorder)
            file_id = _find_id(upload_result.structured_content) or _find_id(upload_result.content)
            if not file_id:
                raise RuntimeError(f"advanced component {resource_id} uploaded but returned no id/fileId.")
            _patch_remote_component(component, resource_id, file_id)
            resources.append(_normalize_remote_resource(resource, resource_id, _string(component.get("name")), file_id))
            recorder.emit(
                "visualization.advanced_component.prepared",
                {"resourceId": resource_id, "fileId": file_id, "zip": f"outputs/{zip_path}"},
            )
        return resources

    def _upload_advanced_component_zip(
        self,
        paths: ThreadPaths,
        zip_path: str,
        zip_name: str,
        runtime_options: RuntimeOptions,
        recorder: EventRecorder,
    ):
        tool_name = "visual_bigscreen_upload_file"
        tool_call_id = f"workflow-{tool_name}-{zip_name}"
        arguments = self._tool_runtime_arguments(paths, runtime_options)
        arguments.update(
            {
                "file_path": f"/mnt/user-data/outputs/{zip_path}",
                "file_name": zip_name,
                "content_type": "application/zip",
            }
        )
        recorder.emit("tool.started", {"tool_name": tool_name, "tool_call_id": tool_call_id})
        result = self.tool_service.call_tool(tool_name, arguments)
        recorder.emit(
            "tool.completed" if not result.is_error else "tool.failed",
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "is_error": result.is_error,
                "arguments": {"file_path": arguments["file_path"], "file_name": arguments["file_name"]},
                "structured_content": result.structured_content,
            },
        )
        if result.is_error:
            raise RuntimeError(_tool_error_text(result))
        return result

    def _save_resource(self, resource: dict[str, Any], runtime_options: RuntimeOptions, recorder: EventRecorder) -> None:
        tool_name = "visualization_bigscreen_save_resource"
        resource_id = _string(resource.get("resourceId"))
        tool_call_id = f"workflow-{tool_name}-{resource_id or 'resource'}"
        arguments = self._tool_runtime_arguments(None, runtime_options)
        arguments["data"] = [resource]
        recorder.emit("tool.started", {"tool_name": tool_name, "tool_call_id": tool_call_id})
        result = self.tool_service.call_tool(tool_name, arguments)
        recorder.emit(
            "tool.completed" if not result.is_error else "tool.failed",
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "is_error": result.is_error,
                "arguments": {"resourceId": resource_id},
                "structured_content": result.structured_content,
            },
        )
        if result.is_error:
            raise RuntimeError(_tool_error_text(result))

    @staticmethod
    def _tool_runtime_arguments(paths: ThreadPaths | None, runtime_options: RuntimeOptions) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        if paths is not None:
            arguments["_thread_id"] = paths.thread_id
        session_cwd = runtime_options.config_options.get("session_cwd")
        if isinstance(session_cwd, str) and session_cwd.strip():
            arguments["_session_cwd"] = session_cwd
        mcp_servers = runtime_options.config_options.get("mcpServers") or runtime_options.config_options.get("mcp_servers")
        if isinstance(mcp_servers, list):
            arguments["_runtime_mcp_servers"] = [dict(item) for item in mcp_servers if isinstance(item, dict)]
        runtime_mcp_tools = runtime_options.config_options.get("runtime_mcp_tools")
        if isinstance(runtime_mcp_tools, list):
            arguments["_runtime_mcp_tools"] = [dict(item) for item in runtime_mcp_tools if isinstance(item, dict)]
        upload_url = runtime_options.config_options.get("visualBigscreenUploadUrl") or runtime_options.config_options.get("fileUploadUrl")
        if isinstance(upload_url, str) and upload_url.strip():
            arguments["_visual_bigscreen_upload_url"] = upload_url.strip()
            arguments["_visual_bigscreen_upload_prefer_rest"] = True
        upload_authorization = _first_config_string(
            runtime_options.config_options,
            "visualBigscreenUploadAuthorization",
            "visual_bigscreen_upload_authorization",
            "fileUploadAuthorization",
            "file_upload_authorization",
        )
        if upload_authorization:
            arguments["_visual_bigscreen_upload_authorization"] = upload_authorization
        upload_token = _first_config_string(
            runtime_options.config_options,
            "visualBigscreenUploadToken",
            "visual_bigscreen_upload_token",
            "fileUploadToken",
            "file_upload_token",
        )
        if upload_token:
            arguments["_visual_bigscreen_upload_token"] = upload_token
        upload_token_header = _first_config_string(
            runtime_options.config_options,
            "visualBigscreenUploadTokenHeader",
            "visual_bigscreen_upload_token_header",
            "fileUploadTokenHeader",
            "file_upload_token_header",
        )
        if upload_token_header:
            arguments["_visual_bigscreen_upload_token_header"] = upload_token_header
        upload_headers = (
            runtime_options.config_options.get("visualBigscreenUploadHeaders")
            or runtime_options.config_options.get("visual_bigscreen_upload_headers")
            or runtime_options.config_options.get("fileUploadHeaders")
            or runtime_options.config_options.get("file_upload_headers")
        )
        if isinstance(upload_headers, dict | list):
            arguments["_visual_bigscreen_upload_headers"] = upload_headers
        upload_tenant_domain = _first_config_string(
            runtime_options.config_options,
            "visualBigscreenUploadTenantDomain",
            "visual_bigscreen_upload_tenant_domain",
            "fileUploadTenantDomain",
            "file_upload_tenant_domain",
        )
        if upload_tenant_domain:
            arguments["_visual_bigscreen_upload_tenant_domain"] = upload_tenant_domain
        upload_field = _first_config_string(
            runtime_options.config_options,
            "visualBigscreenUploadField",
            "visual_bigscreen_upload_field",
            "fileUploadField",
            "file_upload_field",
        )
        if upload_field:
            arguments["_visual_bigscreen_upload_field"] = upload_field
        resource_save_url = _first_config_string(
            runtime_options.config_options,
            "visualBigscreenResourceSaveUrl",
            "visual_bigscreen_resource_save_url",
            "visualBigscreenSaveResourceUrl",
            "visual_bigscreen_save_resource_url",
            "resourceSaveUrl",
            "resource_save_url",
        )
        if resource_save_url:
            arguments["_visual_bigscreen_resource_save_url"] = resource_save_url
        return arguments


def _initialization_system_prompt() -> str:
    return (
        "你是 JetLinks 可视化大屏 JSON 生成器。只返回一个 JSON 对象，不要 Markdown。\n"
        "当前任务是 initialization。大模型只负责生成内容，不要调用任何工具。\n"
        "必须返回字段：pageJson、blueprint、backgroundSvg。\n"
        "pageJson.canvas 使用平台结构，components 必须为空数组。\n"
        "backgroundSvg 必须是完整 SVG 字符串，viewBox 为 0 0 1920 1080，包含固定区域背景和 data-region-id 元数据。\n"
        "backgroundSvg 不要包含真实业务值、表格行、告警文本或交互按钮。\n"
        "backgroundSvg 面板样式必须采用深蓝渐变面板、蓝青色发光边框、四角短线强调和底部微光；装饰必须克制，不要堆叠多层杂线。\n"
        "backgroundSvg 只负责面板外框和标题安全区装饰，不要在 SVG 里写面板标题文字；标题文字由后续 region 阶段的 text 组件生成。\n"
        "标题安全区必须保持可读：每个非 header/metrics 面板左侧从 x+18 到 x+min(width*0.62, 300)、从 y+8 到 contentBox.y-6 是标题 text 组件保留区，禁止任何 path/polyline/line/rect 装饰穿过；标题下划线只能放在 contentBox.y-10 到 contentBox.y-4 之间。\n"
        "所有 contentBox 内部必须完全干净，默认不要画 contentBox 边界；禁止在 contentBox 内画网格、纹理线、分割线、路径线、径向光斑、图表占位、地图引导线或任何会压住后续组件的装饰。\n"
        "center_main_panel.contentBox 尤其不能出现任何背景线条或中心装饰，避免遮挡后续平台组件。\n"
        "边框必须有可见但轻量的动效：每个面板外框至少包含一个 class=border-flow 的 stroke-dasharray 流光层，并使用 SVG <animate> 动画 stroke-dashoffset 或 opacity；不要只依赖静态边框。\n"
        "blueprint 使用 fixed standard-1920x1080-v1，区域坐标必须和约定一致。\n"
        f"固定 blueprint JSON：{json.dumps(_default_blueprint(), ensure_ascii=False)}"
    )


def _region_system_prompt(requested_region_id: str) -> str:
    return (
        "HIGH PRIORITY CURRENT WORKFLOW RULES:\n"
        "Return only one JSON object. The model must not call tools; Python performs uploads and resource saves.\n"
        "Component selection order: use built-in platform components first when they can render the content; use resourceComponentEcharts for all non-map chart-like visualizations; use pseudo for any map/geography/distribution-map intent; use remote advanced Vue components for everything else that needs richer layout, custom interaction, cards, timelines, topology, complex status panels, or polished composite visuals.\n"
        "Built-in platform component types are text, dateTime, table, video, and pseudo. The tabs component is forbidden.\n"
        "For remote advanced components, return them in components and also return advancedComponents[]. Each advancedComponents item must contain resourceId, component, resource, and files. files must include root component.vue and config.mjs. component.configuration.componentType must be remote. resource.configuration.componentType must be remote. Leave fileId empty; Python will zip, upload, and write fileId back.\n"
        "Advanced component resourceId must be a valid JavaScript identifier such as ai_smartParkStatus_v1 because config.mjs must export <resourceId>ConfigProps and <resourceId>Config and ConfigProps.type must equal resourceId.\n"
        "Advanced component Vue files must follow the platform remote component rules: component.vue uses Options API, not <script setup>; config.mjs uses complete .vue import extensions; only whitelisted imports are allowed.\n"
        "The region JSON shape may include optional resources and advancedComponents, but the final user-visible response remains {regionId, components} after Python side effects.\n\n"
        "你是 JetLinks 可视化大屏区域组件 JSON 生成器。只返回一个 JSON 对象，不要 Markdown。\n"
        "当前任务是 region。大模型只负责生成内容，不要调用任何工具。\n"
        "必须返回字段：regionId、components，可选返回 regionTitle、resources。\n"
        f"regionId 必须是 {requested_region_id or 'requestedRegionId'}。\n"
        "regionTitle 是该区域面板标题文本；每个非 header 区域必须把标题作为 components[0] 的平台 text 组件返回。\n"
        "标题组件样式必须由大模型按区域主题自行设计，但仍必须克隆平台 text 模板并使用 dataSourceProps.defaultValue[0].text 存标题。\n"
        "标题组件应放在面板标题带：region.y 到 contentBox.y 之间，不要占用主体内容槽位。\n"
        "components 必须是平台真实 ComponentInfo JSON。不要返回完整 pageJson。\n"
        "允许组件类型仅包括 text、dateTime、table、video、pseudo、resourceComponentEcharts；禁止返回 tabs。\n"
        "只要用户提示词、区域标题或业务语义涉及地图、地理、区域分布、点位、经纬度、园区/城市/省市/全国空间分布，必须使用 type=pseudo 的系列分布地图组件，不要用 ECharts 地图。\n"
        "pseudo 必须按 references/components/pseudo.json 模板返回，不要自由发明结构；点位只能写 dataSourceProps.defaultValue，点位字段只能是 name、longitude、dimension、value，其中 dimension 是纬度字段。\n"
        "pseudo 只能改 id/name、componentProps.style.x/y/width/height/rotate.angle、dataSourceProps.defaultValue 和 componentProps.basicMap 下的颜色/点位显示配置。\n"
        "pseudo 禁止出现 latitude、lat、lng、series、geo、option、echartsOption、markers、points、componentType、x、y、width、height、locked、hidden、layer、animations、events、dataSources。\n"
        "除地图以外，所有折线图、柱状图、饼图、仪表盘、趋势、排行可视化、统计图表都必须使用 resourceComponentEcharts/custom-chart，并在 resources 中返回 ECharts 资源实体。\n"
        "视频/监控/摄像头/直播/画面/播放相关内容使用 type=video 的平台视频组件。\n"
        "每个组件必须使用顶层 type、visible、isLocked，布局必须写入 componentProps.style.x/y/width/height/rotate.angle。\n"
        "禁止在组件顶层返回 componentType、x、y、width、height、rotate、locked、hidden、layer、animations、events、dataSources。\n"
        "文本组件必须使用 text 模板，文本值写入 dataSourceProps.defaultValue[0].text；时间组件必须使用 dateTime 模板。\n"
        "如果生成 resourceComponentEcharts/custom-chart，必须同时在 resources 数组中返回资源实体："
        "resourceId、name、version、thumbnailUrl、provider、type、group、configuration.componentType、configuration.javaScript。\n"
        "componentProps.resource.id 必须等于对应 resource.resourceId，version 固定 0。\n"
        "不要生成 preview.html、React、Vue 或解释说明。"
    )


def _is_retryable_llm_status_error(exc: httpx.HTTPStatusError) -> bool:
    response = exc.response
    return response is not None and response.status_code in {500, 502, 503, 504}


def _skill_package_root(skill: Any) -> Path | None:
    manifest_path = getattr(skill, "manifest_path", None)
    if isinstance(manifest_path, Path) and (manifest_path.parent / "SKILL.md").is_file():
        return manifest_path.parent
    plugin_root = getattr(skill, "plugin_root", None)
    if isinstance(plugin_root, Path):
        if (plugin_root / "SKILL.md").is_file():
            return plugin_root
        nested = plugin_root / "skills" / getattr(skill, "name", "") / "SKILL.md"
        if nested.is_file():
            return nested.parent
    return None


def _default_visualization_package_root() -> Path | None:
    root = SKILLS_ROOT / DEFAULT_VISUALIZATION_SKILL
    return root if (root / "SKILL.md").is_file() else None


def _stage_aware_visualization_context(
    package_root: Path,
    stage: str,
    *,
    prompt_text: str = "",
    requested_region_id: str = "",
) -> tuple[str, list[str]]:
    normalized_stage = stage if stage in {"initialization", "region"} else "unknown"
    if normalized_stage == "initialization":
        paths = [
            "references/blueprint-standard.md",
        ]
        rules = [
            "Stage-aware visualization skill context is active as the primary prompt path.",
            'Current stage: "initialization".',
            "Return only JSON with pageJson, blueprint, and backgroundSvg.",
            "pageJson.canvas must use the platform canvas structure and components must be empty.",
            "backgroundSvg must be a complete 1920x1080 SVG background with region metadata only.",
            "Do not include business values, table rows, alarms, or interactive controls in backgroundSvg.",
            "Use clean cyber panel chrome: dark blue gradient panel fill, cyan/blue glowing border, corner accents, and subtle bottom glow. Keep decoration sparse; avoid stacked ornamental lines.",
            "The SVG background must not write literal panel title text. Reserve title safe space only; the region stage returns title text as platform text components.",
            "Keep title text safe areas clean. For non-header/non-metrics panels, the left title zone from x+18 to x+min(width*0.62, 300), y+8 to contentBox.y-6 must not be crossed by path/polyline/line/rect decoration. Put the optional title underline only between contentBox.y-10 and contentBox.y-4.",
            "Keep every contentBox completely clean for later components. Prefer no contentBox boundary. Do not draw grids, texture lines, divider lines, route lines, radial glows, chart placeholders, map guide marks, or center visual decoration inside any contentBox.",
            "center_main_panel.contentBox must have no background lines or center decoration.",
            "Every panel border must include a lightweight animated flow highlight: use a border-flow stroke layer with SVG <animate> on stroke-dashoffset or opacity. Do not rely on a static border only.",
        ]
    else:
        paths = _region_reference_paths(prompt_text, requested_region_id)
        rules = [
            "Stage-aware visualization skill context is active as the primary prompt path.",
            f'Current stage: "{normalized_stage}".',
            "Return only JSON with regionId and components; resources and advancedComponents are optional.",
            "Built-in components: text, dateTime, table, video, pseudo. tabs is forbidden.",
            "Map/geography/spatial distribution uses pseudo.",
            "pseudo must clone references/components/pseudo.json; point data only belongs in dataSourceProps.defaultValue.",
            "pseudo point fields must be exactly name, longitude, dimension, value; dimension is latitude.",
            "Non-map charts use resourceComponentEcharts/custom-chart and must include a matching ECharts resource in resources[].",
            "Complex non-chart visuals use remote advanced Vue component in advancedComponents[].",
        ]
    lines = [*rules, "", "## Stage Reference Files"]
    loaded_paths: list[str] = []
    for relative in paths:
        path = package_root / relative
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        loaded_paths.append(relative)
        limit = 4000 if relative.endswith(".md") else 6000
        lines.extend(["", f"### {relative}", "", _truncate_text(content, limit)])
    return "\n".join(lines).strip(), loaded_paths


def _region_reference_paths(prompt_text: str, region_id: str) -> list[str]:
    text = _region_context_text(prompt_text, region_id)
    preferred = _region_preferred_components(region_id)
    map_intent = _is_map_intent(text)
    video_intent = _is_video_intent(text)
    paths = ["references/components/text.json"]

    if _is_time_intent(text) or "dateTime" in preferred:
        paths.append("references/components/dateTime.json")
    if _is_table_intent(text) or "table" in preferred:
        paths.append("references/components/table.json")
    if video_intent:
        paths.append("references/components/video.json")
    if map_intent:
        paths.append("references/components/pseudo.json")
    if _is_chart_intent(text) or ("custom-chart" in preferred and not map_intent and not video_intent):
        paths.extend(
            [
                "references/components/custom-chart.json",
                "references/resources/echarts-resource.json",
                "references/echarts-script-standard.md",
            ]
        )
    if _is_advanced_component_intent(text):
        paths.extend(
            [
                "references/components/custom-component.json",
                "references/resources/custom-resource.json",
                "references/advanced-component-standard.md",
            ]
        )
    return _dedupe_path_list(paths)


def _region_context_text(prompt_text: str, region_id: str) -> str:
    region = _region_definition(region_id)
    scoped_prompt = _region_scoped_prompt_text(prompt_text, region_id)
    pieces = [scoped_prompt or prompt_text]
    if region is not None:
        pieces.append(_string(region.get("name")))
        pieces.append(_string(region.get("role")))
        for slot in _list(region.get("slots")):
            if isinstance(slot, dict):
                pieces.append(_string(slot.get("name")))
    return "\n".join(item for item in pieces if item)


def _region_preferred_components(region_id: str) -> set[str]:
    region = _region_definition(region_id)
    preferred: set[str] = set()
    if region is None:
        return preferred
    for slot in _list(region.get("slots")):
        if not isinstance(slot, dict):
            continue
        raw = slot.get("preferredComponents")
        if isinstance(raw, list):
            preferred.update(_string(item) for item in raw if _string(item))
    return preferred


def _is_time_intent(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("时间", "日期", "当前时间", "date", "time", "datetime"))


def _is_table_intent(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("表格", "列表", "排行", "排名", "告警", "事件", "设备清单", "top", "rank", "table", "list", "alarm"))


def _is_video_intent(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ("视频", "监控", "摄像头", "直播", "播放", "video", "camera", "live"))


def _is_chart_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in lowered
        for keyword in (
            "图表",
            "趋势",
            "折线",
            "柱状",
            "饼图",
            "仪表盘",
            "面积图",
            "占比",
            "对比",
            "统计",
            "分析",
            "chart",
            "line",
            "bar",
            "pie",
            "gauge",
            "trend",
        )
    )


def _is_advanced_component_intent(text: str) -> bool:
    lowered = text.lower()
    return any(
        keyword in lowered
        for keyword in (
            "高级组件",
            "远程组件",
            "自定义组件",
            "拓扑",
            "时间轴",
            "流程",
            "交互式",
            "复杂卡片",
            "advanced",
            "remote",
            "topology",
            "timeline",
        )
    )


def _dedupe_path_list(paths: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path and path not in seen:
            result.append(path)
            seen.add(path)
    return result


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...<truncated chars={len(value) - max_chars}>"


def _default_page_json() -> dict[str, Any]:
    return {
        "canvas": {
            "width": 1920,
            "height": 1080,
            "scale": 0.64,
            "name": "画布",
            "sizeKey": "pc",
            "adaptationType": "AUTO",
            "backgroundColor": "#424242",
            "backgroundImage": {"fileId": ""},
            "gridLayout": {
                "backgroundColor": "",
                "marginHorizontal": 8,
                "marginVertical": 8,
                "borderColor": "",
                "borderWidth": 1,
                "borderStyle": "solid",
                "fontColor": "rgba(0,0,0,1)",
            },
            "enablePreviewZoom": False,
            "filter": {
                "hue": 0,
                "saturation": 0,
                "brightness": 0,
                "contrast": 0,
                "opacity": 100,
                "grayscale": 0,
            },
        },
        "components": [],
    }


def _default_blueprint() -> dict[str, Any]:
    return {
        "id": "standard-1920x1080-v1",
        "canvas": {"width": 1920, "height": 1080},
        "regions": copy.deepcopy(BLUEPRINT_REGIONS),
    }


def _fallback_background_svg() -> str:
    panels = []
    for region in BLUEPRINT_REGIONS:
        x = int(region["x"])
        y = int(region["y"])
        width = int(region["width"])
        height = int(region["height"])
        content_box = region["contentBox"]
        cx = int(content_box["x"])
        cy = int(content_box["y"])
        cw = int(content_box["width"])
        ch = int(content_box["height"])
        radius = 20 if height >= 180 else 16
        title_x = x + 24
        title_line_y = max(y + 18, cy - 8)
        title_line_width = max(90, min(190, width // 3))
        deco_x = max(x + width - 136, title_x + title_line_width + 48)
        deco_y = y + 34
        dot_y = min(y + 58, cy - 10 if cy - y > 44 else y + height - 20)
        bottom_width = max(86, min(180, width // 4))
        bottom_x = x + (width - bottom_width) / 2
        bottom_y = y + height - 3
        title_chrome = ""
        if region["role"] not in {"header", "metrics"}:
            title_chrome = (
                f'<rect x="{title_x}" y="{title_line_y}" width="{title_line_width}" height="3" rx="1.5" '
                'fill="url(#titleLine)" class="title-breath" filter="url(#softGlow)">'
                '<animate attributeName="opacity" values="0.55;0.95;0.55" dur="2.8s" repeatCount="indefinite"/>'
                '</rect>'
                f'<path d="M{deco_x} {deco_y}H{min(deco_x + 52, x + width - 80)}'
                f'L{min(deco_x + 64, x + width - 64)} {deco_y + 8}H{x + width - 36}" '
                'stroke="#59DFFF" stroke-opacity="0.34" stroke-width="1.2" stroke-linecap="round"/>'
                f'<rect x="{x + width - 66}" y="{dot_y}" width="8" height="2" rx="1" fill="#59DFFF" class="dot-float"/>'
                f'<rect x="{x + width - 52}" y="{dot_y}" width="8" height="2" rx="1" fill="#59DFFF" opacity="0.65" class="dot-float"/>'
                f'<rect x="{x + width - 38}" y="{dot_y}" width="8" height="2" rx="1" fill="#59DFFF" opacity="0.35" class="dot-float"/>'
            )
        panels.append(
            f'<g id="region-{region["id"]}" data-region-id="{region["id"]}" data-role="{region["role"]}" '
            f'data-x="{x}" data-y="{y}" data-width="{width}" data-height="{height}" '
            f'data-content-x="{cx}" data-content-y="{cy}" data-content-width="{cw}" data-content-height="{ch}">'
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="url(#panelFill)"/>'
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            'fill="none" stroke="#2D7FBE" stroke-opacity="0.52" stroke-width="1.2"/>'
            f'<rect class="border-base" x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            'fill="none" stroke="url(#strokeGlow)" stroke-width="1.2">'
            '<animate attributeName="opacity" values="0.26;0.52;0.26" dur="3.2s" repeatCount="indefinite"/>'
            '</rect>'
            f'<rect class="border-flow" x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            'fill="none" stroke="#65E4FF" stroke-width="2" filter="url(#softGlow)" stroke-linecap="round" '
            'stroke-dasharray="90 520" stroke-dashoffset="0">'
            '<animate attributeName="stroke-dashoffset" from="0" to="-610" dur="5s" repeatCount="indefinite"/>'
            '<animate attributeName="opacity" values="0.3;0.85;0.3" dur="2.8s" repeatCount="indefinite"/>'
            '</rect>'
            f"{title_chrome}"
            f'<path d="M{x} {y + 40}V{y + radius}C{x} {y + 7} {x + 7} {y} {x + radius} {y}H{x + 58}" '
            'stroke="#64E6FF" stroke-width="1.5" stroke-opacity="0.78" class="corner-blink" fill="none"/>'
            f'<path d="M{x + width - 58} {y}H{x + width - radius}C{x + width - 7} {y} {x + width} {y + 7} {x + width} {y + radius}V{y + 40}" '
            'stroke="#64E6FF" stroke-width="1.5" stroke-opacity="0.55" class="corner-blink" fill="none"/>'
            f'<path d="M{x + width} {y + height - 56}V{y + height - radius}C{x + width} {y + height - 7} {x + width - 7} {y + height} {x + width - radius} {y + height}H{x + width - 58}" '
            'stroke="#64E6FF" stroke-width="1.5" stroke-opacity="0.4" class="corner-blink" fill="none"/>'
            f'<path d="M{x + 58} {y + height}H{x + radius}C{x + 7} {y + height} {x} {y + height - 7} {x} {y + height - radius}V{y + height - 56}" '
            'stroke="#64E6FF" stroke-width="1.5" stroke-opacity="0.35" class="corner-blink" fill="none"/>'
            f'<rect x="{bottom_x:.1f}" y="{bottom_y}" width="{bottom_width}" height="2" rx="1" '
            'fill="#52DBFF" opacity="0.55" filter="url(#softGlow)"/>'
            "</g>"
        )
    return (
        '<svg viewBox="0 0 1920 1080" width="1920" height="1080" xmlns="http://www.w3.org/2000/svg">'
        "<defs>"
        '<linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#06111F"/><stop offset="100%" stop-color="#081A30"/></linearGradient>'
        '<linearGradient id="panelFill" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#0A2A46" stop-opacity="0.88"/>'
        '<stop offset="0.55" stop-color="#071F35" stop-opacity="0.88"/>'
        '<stop offset="1" stop-color="#061A2D" stop-opacity="0.88"/></linearGradient>'
        '<linearGradient id="strokeGlow" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#45D8FF" stop-opacity="0.25"/>'
        '<stop offset="0.2" stop-color="#45D8FF" stop-opacity="0.95"/>'
        '<stop offset="0.5" stop-color="#4A8BFF" stop-opacity="0.35"/>'
        '<stop offset="0.8" stop-color="#45D8FF" stop-opacity="0.9"/>'
        '<stop offset="1" stop-color="#45D8FF" stop-opacity="0.22"/></linearGradient>'
        '<linearGradient id="titleLine" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#5DE2FF"/><stop offset="0.55" stop-color="#4AA8FF"/>'
        '<stop offset="1" stop-color="#4AA8FF" stop-opacity="0"/></linearGradient>'
        '<filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/>'
        '<feMergeNode in="SourceGraphic"/></feMerge></filter>'
        "</defs>"
        "<style>"
        ".border-base{stroke-dasharray:8 8;animation:borderPulse 3.2s linear infinite}"
        ".border-flow{stroke-dasharray:90 520;stroke-dashoffset:0;animation:borderRun 5s linear infinite}"
        ".title-breath{transform-box:fill-box;transform-origin:center;animation:titleBreath 2.6s ease-in-out infinite}"
        ".corner-blink{animation:cornerBlink 2.4s ease-in-out infinite}"
        ".dot-float{animation:dotFloat 2.8s ease-in-out infinite}"
        "@keyframes borderRun{from{stroke-dashoffset:0}to{stroke-dashoffset:-610}}"
        "@keyframes borderPulse{0%,100%{opacity:.28}50%{opacity:.55}}"
        "@keyframes titleBreath{0%,100%{opacity:.72;transform:scaleX(1)}50%{opacity:1;transform:scaleX(1.04)}}"
        "@keyframes cornerBlink{0%,100%{opacity:.35}50%{opacity:.95}}"
        "@keyframes dotFloat{0%,100%{opacity:.35;transform:translateY(0)}50%{opacity:.9;transform:translateY(-1.5px)}}"
        "</style>"
        '<rect width="1920" height="1080" fill="url(#bgGrad)"/>'
        + "".join(panels)
        + "</svg>"
    )


def _normalize_page_json(page_json: dict[str, Any]) -> dict[str, Any]:
    base = _default_page_json()
    canvas = _dict(page_json.get("canvas"))
    if canvas:
        merged_canvas = {**base["canvas"], **canvas}
        merged_canvas["gridLayout"] = {**base["canvas"]["gridLayout"], **_dict(canvas.get("gridLayout"))}
        merged_canvas["filter"] = {**base["canvas"]["filter"], **_dict(canvas.get("filter"))}
        merged_canvas["backgroundImage"] = {**base["canvas"]["backgroundImage"], **_dict(canvas.get("backgroundImage"))}
        base["canvas"] = merged_canvas
    base["canvas"]["width"] = 1920
    base["canvas"]["height"] = 1080
    base["canvas"]["scale"] = 0.64
    base["canvas"]["sizeKey"] = "pc"
    base["canvas"]["adaptationType"] = "AUTO"
    base["components"] = []
    return base


def _normalize_blueprint(blueprint: dict[str, Any]) -> dict[str, Any]:
    if not blueprint:
        return _default_blueprint()
    result = _default_blueprint()
    result.update({key: value for key, value in blueprint.items() if key not in {"canvas", "regions"}})
    result["id"] = "standard-1920x1080-v1"
    result["canvas"] = {"width": 1920, "height": 1080}
    regions = blueprint.get("regions")
    result["regions"] = regions if isinstance(regions, list) and regions else copy.deepcopy(BLUEPRINT_REGIONS)
    return result


def _advanced_components_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("advancedComponents", "advanced_components", "componentPackages", "component_packages", "remoteComponents"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, dict)]
    return []


def _append_advanced_components(components: list[Any], advanced_components: list[dict[str, Any]]) -> None:
    for package in advanced_components:
        component = _dict(package.get("component") or package.get("componentJson"))
        if not component:
            continue
        component.setdefault("configuration", {})
        if isinstance(component["configuration"], dict):
            component["configuration"].setdefault("componentType", "remote")
        if not _contains_component(components, component):
            components.append(component)


def _contains_component(components: list[Any], component: dict[str, Any]) -> bool:
    component_id = _string(component.get("id"))
    component_type = _string(component.get("type"))
    for existing in components:
        if not isinstance(existing, dict):
            continue
        if component_id and _string(existing.get("id")) == component_id:
            return True
        if component_type and _string(existing.get("type")) == component_type:
            return True
    return False


def _advanced_component_files(package: dict[str, Any]) -> list[dict[str, str]]:
    raw_files = (
        package.get("files")
        or package.get("componentFiles")
        or package.get("component_files")
        or package.get("sourceFiles")
        or package.get("source_files")
    )
    files: dict[str, str] = {}
    if isinstance(raw_files, dict):
        for path, content in raw_files.items():
            if isinstance(content, str):
                files[_safe_component_file_path(str(path))] = content
    elif isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            path = _string(item.get("path") or item.get("name") or item.get("fileName"))
            content = item.get("content")
            if path and isinstance(content, str):
                files[_safe_component_file_path(path)] = content

    shortcut_files = {
        "component.vue": package.get("componentVue") or package.get("component_vue"),
        "config.mjs": package.get("configMjs") or package.get("config_mjs"),
        "Config.vue": package.get("configVue") or package.get("config_vue"),
    }
    for path, content in shortcut_files.items():
        if isinstance(content, str) and content.strip():
            files[_safe_component_file_path(path)] = content

    return [{"path": path, "content": content} for path, content in files.items()]


def _safe_component_file_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip().lstrip("/")
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe advanced component file path: {path}")
    if any("\x00" in part for part in parts):
        raise ValueError(f"unsafe advanced component file path: {path}")
    return "/".join(parts)


def _validate_advanced_component_files(resource_id: str, files: list[dict[str, str]]) -> None:
    if not re.match(r"^[A-Za-z_$][A-Za-z0-9_$]*$", resource_id):
        raise ValueError(f"advanced component resourceId must be a valid JavaScript identifier: {resource_id}")
    by_name = {item["path"]: item["content"] for item in files}
    if "component.vue" not in by_name:
        raise ValueError(f"advanced component {resource_id} is missing component.vue.")
    if "config.mjs" not in by_name:
        raise ValueError(f"advanced component {resource_id} is missing config.mjs.")
    component_vue = by_name["component.vue"].lower()
    if "<script setup" in component_vue:
        raise ValueError(f"advanced component {resource_id} must use Options API and cannot use <script setup>.")
    config_mjs = by_name["config.mjs"]
    if f"export const {resource_id}ConfigProps" not in config_mjs or f"export const {resource_id}Config" not in config_mjs:
        raise ValueError(
            f"advanced component {resource_id} config.mjs must export "
            f"{resource_id}ConfigProps and {resource_id}Config."
        )


def _advanced_component_for_package(
    package: dict[str, Any],
    components: list[Any],
    resource_id: str,
) -> dict[str, Any]:
    raw_component = _dict(package.get("component") or package.get("componentJson"))
    component_id = _string(raw_component.get("id"))
    for component in components:
        if not isinstance(component, dict):
            continue
        if component_id and _string(component.get("id")) == component_id:
            return component
        if _string(component.get("type")) == resource_id:
            return component
    if raw_component:
        components.append(raw_component)
        return raw_component
    raise ValueError(f"advanced component {resource_id} is missing matching page component JSON.")


def _advanced_resource_id(package: dict[str, Any], resource: dict[str, Any], component: dict[str, Any]) -> str:
    for value in (
        package.get("resourceId"),
        package.get("resource_id"),
        resource.get("resourceId"),
        resource.get("id"),
        component.get("type"),
    ):
        text = _string(value)
        if text and text != "visualization-component-0mv7k-01":
            return text
    return ""


def _zip_component_files(files: list[dict[str, str]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            archive.writestr(item["path"], item["content"].encode("utf-8"))
    return buffer.getvalue()


def _patch_remote_component(component: dict[str, Any], resource_id: str, file_id: str) -> None:
    component["type"] = resource_id
    component_id = _string(component.get("id"))
    if not component_id or component_id.startswith("visualization-component-0mv7k-01"):
        component["id"] = f"{resource_id}_{int(time.time() * 1000)}"
    component.setdefault("name", resource_id)
    component.setdefault("visible", True)
    component.setdefault("isLocked", False)
    configuration = _dict(component.get("configuration"))
    configuration["componentType"] = "remote"
    configuration["fileId"] = file_id
    configuration["zipUrl"] = file_id
    component["configuration"] = configuration


def _normalize_remote_resource(resource: dict[str, Any], resource_id: str, name: str, file_id: str) -> dict[str, Any]:
    normalized = dict(resource)
    normalized["resourceId"] = resource_id
    normalized["name"] = _string(normalized.get("name")) or name or resource_id
    normalized["version"] = int(normalized.get("version") or 0)
    normalized.setdefault("thumbnailUrl", "")
    normalized.setdefault("provider", "local")
    normalized.setdefault("type", "component")
    normalized["group"] = _normalized_resource_group(normalized.get("group"))
    configuration = _dict(normalized.get("configuration"))
    configuration["componentType"] = "remote"
    configuration["fileId"] = file_id
    configuration.pop("javaScript", None)
    normalized["configuration"] = configuration
    return normalized


def _normalized_resource_group(value: Any) -> str:
    group = _string(value)
    if not group or "__" in group:
        return DEFAULT_RESOURCE_GROUP
    return group


def _is_remote_component(component: dict[str, Any]) -> bool:
    configuration = _dict(component.get("configuration"))
    return _string(configuration.get("componentType")).lower() == "remote"


def _is_advanced_component_type(component_type: str) -> bool:
    return component_type.startswith(("ai_", "remote_", "visualization-component-"))


def _normalize_pseudo_components(components: list[Any], region_id: str) -> None:
    for index, component in enumerate(list(components)):
        if not isinstance(component, dict) or _string(component.get("type")) != "pseudo":
            continue
        components[index] = _normalize_pseudo_component(component, region_id)


def _normalize_pseudo_component(component: dict[str, Any], region_id: str) -> dict[str, Any]:
    normalized = copy.deepcopy(_pseudo_template())
    normalized["id"] = _string(component.get("id")) or f"pseudo_{_safe_file_stem(region_id or 'map')}_{int(time.time() * 1000)}"
    normalized["name"] = _string(component.get("name")) or _string(normalized.get("name")) or "系列分布地图"
    normalized["visible"] = _bool(component.get("visible"), bool(normalized.get("visible", True)))
    normalized["isLocked"] = _bool(component.get("isLocked", component.get("locked")), bool(normalized.get("isLocked", False)))

    source_style = _component_style(component)
    target_style = _dict(_dict(normalized.get("componentProps")).get("style"))
    fallback_box = _pseudo_fallback_box(region_id)
    for key in ("x", "y", "width", "height"):
        value = _number(source_style.get(key))
        if value is None:
            value = _number(component.get(key))
        if value is None:
            value = _number(fallback_box.get(key))
        if value is not None:
            target_style[key] = int(value) if float(value).is_integer() else value
    source_rotate = _dict(source_style.get("rotate"))
    angle = _number(source_rotate.get("angle"))
    if angle is None:
        angle = _number(component.get("rotate"))
    if angle is not None:
        rotate = _dict(target_style.get("rotate"))
        rotate["angle"] = int(angle) if float(angle).is_integer() else angle
        target_style["rotate"] = rotate

    component_props = _dict(normalized.get("componentProps"))
    component_props["style"] = target_style
    _patch_basic_map(component_props, component)
    normalized["componentProps"] = component_props

    data_source = _dict(normalized.get("dataSourceProps"))
    points = _pseudo_points_from_component(component)
    if points:
        data_source["defaultValue"] = points
    normalized["dataSourceProps"] = data_source
    return normalized


def _pseudo_template() -> dict[str, Any]:
    for template_file in PSEUDO_TEMPLATE_FILES:
        try:
            template = json.loads(template_file.read_text(encoding="utf-8"))
            if isinstance(template, dict) and template.get("type") == "pseudo":
                return template
        except (OSError, json.JSONDecodeError):
            continue
    return {
        "type": "pseudo",
        "name": "系列分布地图",
        "visible": True,
        "isLocked": False,
        "dataSourceProps": {
            "mode": "static",
            "type": "array",
            "defaultValue": [
                {"name": "湖北", "longitude": 113.289984, "dimension": 31.42, "value": 2000},
                {"name": "湖南", "longitude": 112.03042, "dimension": 27, "value": 200000},
                {"name": "四川", "longitude": 102.112035, "dimension": 30.630737, "value": 5000},
                {"name": "重庆", "longitude": 108.112035, "dimension": 30.630737, "value": 60000},
                {"name": "山东", "longitude": 118.19, "dimension": 36.22, "value": 20050},
            ],
            "sourceId": "",
            "action": {"enable": False, "actions": []},
            "mapping": {"geo": [], "geoMap": {"name": "", "longitude": "", "dimension": "", "value": ""}},
        },
        "animationProps": [],
        "componentProps": {
            "style": {"x": 664, "y": 288, "width": 800, "height": 600, "rotate": {"angle": 0}},
            "background": {
                "backgroundColor": "#0000",
                "imgField": "",
                "opacity": 1,
                "borderColor": "#000",
                "borderType": "solid",
                "borderWidth": 1,
                "borderRadius": 0,
                "shadowColor": "#0000",
                "shadowBlur": 0,
                "shadowX": 0,
                "shadowY": 0,
                "shadowDiff": 0,
            },
            "basicMap": {
                "level": "chinaMap",
                "parent": "",
                "bgColor": "rgba(4, 70, 198, 0.6)",
                "boundaryLinColor": "rgba(0, 144, 255, 0.6)",
                "interactionEffect": "mouseHover",
                "mouseHover": "rgba(4, 70, 198, 0.6)",
                "mouseClick": "rgba(232, 254, 0, 1)",
                "coordinateColor": "rgba(100, 149, 237, 1)",
                "coordinateSize": 5,
                "NfontSize": 12,
                "NfontColor": "#ffc72b",
                "Nblod": False,
                "Nitalic": False,
                "seat": "top",
                "FfontSize": 12,
                "FfontColor": "#ffffff",
                "Fblod": False,
                "Fitalic": False,
                "suspendedBabel": True,
                "flash": True,
                "flashTime": 2,
            },
        },
        "id": "pseudo_ZYY52HmzWkMx8hyN",
    }


def _pseudo_fallback_box(region_id: str) -> dict[str, Any]:
    region = _region_definition(region_id)
    if region is None:
        return {"x": 664, "y": 288, "width": 800, "height": 600}
    content_box = _dict(region.get("contentBox"))
    return {
        "x": content_box.get("x", region.get("x", 664)),
        "y": content_box.get("y", region.get("y", 288)),
        "width": content_box.get("width", region.get("width", 800)),
        "height": content_box.get("height", region.get("height", 600)),
    }


def _component_style(component: dict[str, Any]) -> dict[str, Any]:
    return _dict(_dict(component.get("componentProps")).get("style"))


def _patch_basic_map(component_props: dict[str, Any], component: dict[str, Any]) -> None:
    basic_map = _dict(component_props.get("basicMap"))
    source = _dict(_dict(component.get("componentProps")).get("basicMap"))
    if not source:
        source = _dict(component.get("basicMap") or component.get("mapStyle") or component.get("pointStyle"))
    allowed = {
        "bgColor",
        "boundaryLinColor",
        "interactionEffect",
        "mouseHover",
        "mouseClick",
        "coordinateColor",
        "coordinateSize",
        "NfontSize",
        "NfontColor",
        "Nblod",
        "Nitalic",
        "seat",
        "FfontSize",
        "FfontColor",
        "Fblod",
        "Fitalic",
        "suspendedBabel",
        "flash",
        "flashTime",
    }
    for key in allowed:
        if key in source:
            basic_map[key] = source[key]
    component_props["basicMap"] = basic_map


def _pseudo_points_from_component(component: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        _dict(component.get("dataSourceProps")).get("defaultValue"),
        component.get("defaultValue"),
        component.get("points"),
        component.get("markers"),
        component.get("data"),
        component.get("value"),
    ]
    for candidate in candidates:
        points = _normalize_pseudo_points(candidate)
        if points:
            return points
    return []


def _normalize_pseudo_points(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("defaultValue", "points", "markers", "data", "list"):
            points = _normalize_pseudo_points(value.get(key))
            if points:
                return points
        return []
    if not isinstance(value, list):
        return []
    points: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        name = _string(item.get("name") or item.get("label") or item.get("title")) or f"点位{index + 1}"
        longitude = _number(item.get("longitude"))
        if longitude is None:
            longitude = _number(item.get("lng"))
        if longitude is None:
            longitude = _number(item.get("lon"))
        dimension = _number(item.get("dimension"))
        if dimension is None:
            dimension = _number(item.get("latitude"))
        if dimension is None:
            dimension = _number(item.get("lat"))
        raw_value = _number(item.get("value"))
        if raw_value is None:
            raw_value = _number(item.get("count"))
        if raw_value is None:
            raw_value = _number(item.get("num"))
        if longitude is None or dimension is None:
            continue
        points.append(
            {
                "name": name,
                "longitude": _int_if_whole(longitude),
                "dimension": _int_if_whole(dimension),
                "value": _int_if_whole(raw_value if raw_value is not None else 0),
            }
        )
    return points


def _validate_pseudo_components(components: list[Any]) -> None:
    allowed_top_level = {"type", "name", "visible", "isLocked", "dataSourceProps", "animationProps", "componentProps", "id"}
    for component in components:
        if not isinstance(component, dict) or component.get("type") != "pseudo":
            continue
        extra_keys = set(component) - allowed_top_level
        if extra_keys:
            raise ValueError(f"pseudo 组件包含平台模板外字段: {sorted(extra_keys)}")
        data_source = _dict(component.get("dataSourceProps"))
        default_value = data_source.get("defaultValue")
        if not isinstance(default_value, list):
            raise ValueError("pseudo 组件 dataSourceProps.defaultValue 必须是点位数组。")
        for point in default_value:
            if not isinstance(point, dict):
                raise ValueError("pseudo 点位必须是对象。")
            if set(point) != {"name", "longitude", "dimension", "value"}:
                raise ValueError("pseudo 点位字段必须且只能是 name/longitude/dimension/value。")
        component_props = _dict(component.get("componentProps"))
        if not isinstance(component_props.get("basicMap"), dict):
            raise ValueError("pseudo 组件必须保留 componentProps.basicMap。")


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _int_if_whole(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _validate_component_types(components: list[Any]) -> None:
    allowed = {"text", "dateTime", "table", "video", "pseudo", "resourceComponentEcharts"}
    forbidden = {"tabs", "line", "bar", "pie", "gauge", "chart", "map", "echarts"}
    for component in components:
        if not isinstance(component, dict):
            continue
        component_type = _string(component.get("type"))
        if _is_remote_component(component) or _is_advanced_component_type(component_type):
            continue
        if component_type in forbidden:
            raise ValueError(f"禁止生成组件类型 {component_type}，请按规则改用平台模板或 custom ECharts 资源组件。")
        if component_type and component_type not in allowed:
            raise ValueError(f"不支持的可视化组件类型 {component_type}，允许类型为 text/dateTime/table/video/pseudo/resourceComponentEcharts。")


def _validate_map_intent_components(region_id: str, prompt_text: str, region_title: str, components: list[Any]) -> None:
    if not _region_requires_map_validation(region_id, prompt_text, region_title, components):
        return
    has_pseudo = False
    echarts_components: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        component_type = _string(component.get("type"))
        if component_type == "pseudo":
            has_pseudo = True
        if component_type == "resourceComponentEcharts":
            echarts_components.append(_string(component.get("id")) or _string(component.get("name")) or "resourceComponentEcharts")
    if echarts_components:
        raise ValueError("地图相关提示词必须使用 pseudo 系列分布地图组件，不能使用 resourceComponentEcharts/ECharts 地图。")
    if not has_pseudo:
        raise ValueError("地图相关提示词必须返回至少一个 type=pseudo 的系列分布地图组件。")


def _region_requires_map_validation(region_id: str, prompt_text: str, region_title: str, components: list[Any]) -> bool:
    if not _region_can_host_map(region_id):
        return _is_map_intent(_component_map_hint_text(components))
    scoped_text = "\n".join(
        item
        for item in (
            _region_scoped_prompt_text(prompt_text, region_id),
            region_title,
            _component_map_hint_text(components),
        )
        if item
    )
    if _is_map_intent(scoped_text):
        return True
    if region_id == "center_main_panel" and _is_global_central_map_intent(prompt_text):
        return True
    return False


def _region_can_host_map(region_id: str) -> bool:
    region = _region_definition(region_id)
    if region is None:
        return True
    if region_id == "center_main_panel":
        return True
    if _is_map_intent(_string(region.get("name")) or ""):
        return True
    for slot in _list(region.get("slots")):
        if not isinstance(slot, dict):
            continue
        preferred = slot.get("preferredComponents")
        if isinstance(preferred, list) and "pseudo" in preferred:
            return True
        if _is_map_intent(_string(slot.get("name"))):
            return True
    return False


def _region_scoped_prompt_text(prompt_text: str, region_id: str) -> str:
    if not region_id:
        return ""
    lines = [line.strip() for line in prompt_text.splitlines()]
    scoped: list[str] = []
    capture_next = False
    for line in lines:
        if not line:
            continue
        lower = line.lower()
        if "requestedregionid" in lower or "requested region" in lower:
            scoped.append(line)
            if region_id in line:
                capture_next = True
            continue
        if line.startswith(("本区域", "当前区域", "区域目标", "生成目标")):
            scoped.append(line)
            continue
        if region_id in line:
            scoped.append(line)
            capture_next = True
            continue
        if capture_next:
            scoped.append(line)
            if line.endswith(("。", ".", "；", ";")):
                capture_next = False
    return "\n".join(scoped)


def _component_map_hint_text(components: list[Any]) -> str:
    hints: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        hints.append(_string(component.get("type")))
        hints.append(_string(component.get("name")))
        hints.append(_string(component.get("id")))
    return "\n".join(item for item in hints if item)


def _is_global_central_map_intent(prompt_text: str) -> bool:
    for line in prompt_text.splitlines():
        lowered = line.lower()
        if ("central visual" in lowered or "中心主视觉" in line or "中心区域" in line or "center_main_panel" in lowered) and _is_map_intent(line):
            return True
    return False


def _is_map_intent(text: str) -> bool:
    lowered = text.lower()
    keywords = (
        "地图",
        "地理",
        "区域分布",
        "空间分布",
        "点位",
        "坐标",
        "经纬度",
        "经度",
        "纬度",
        "园区分布",
        "城市分布",
        "省市分布",
        "省份分布",
        "全国地图",
        "全国分布",
        "地图",
        "gis",
        "geo",
        "geographic",
        "map",
        "经纬度",
        "经度",
        "纬度",
        "点位",
        "坐标",
        "省市分布",
        "省份分布",
        "城市分布",
        "全国地图",
        "全国分布",
        "区域分布",
        "地理",
        "空间分布",
    )
    return any(keyword in lowered for keyword in keywords)


def _validate_region_title_component(
    region_id: str,
    requested_title: str,
    components: list[Any],
) -> None:
    region = _region_definition(region_id)
    if region is None or region_id == "header":
        return
    title = requested_title or _string(region.get("name")) or region_id
    if title and _has_region_title_component(region, title, components):
        return
    raise ValueError(
        f"区域 {region_id} 缺少大模型生成的标题 text 组件；"
        "components[0] 必须是平台 text 模板组件，标题文本写入 dataSourceProps.defaultValue[0].text。"
    )


def _region_definition(region_id: str) -> dict[str, Any] | None:
    for region in BLUEPRINT_REGIONS:
        if _string(region.get("id")) == region_id:
            return region
    return None


def _has_region_title_component(region: dict[str, Any], title: str, components: list[Any]) -> bool:
    region_id = _string(region.get("id"))
    expected_id = f"text_{_safe_file_stem(region_id)}_title"
    content_box = _dict(region.get("contentBox"))
    title_band_bottom = float(content_box.get("y") or region.get("y") or 0)
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        if _string(component.get("id")) == expected_id:
            return True
        if component.get("type") != "text":
            continue
        name = _string(component.get("name"))
        text = _component_static_text(component)
        style = _dict(_dict(component.get("componentProps")).get("style"))
        y = _number(style.get("y"))
        looks_like_title = "标题" in name or text == title or (index == 0 and bool(text))
        in_title_band = y is not None and title_band_bottom and y < title_band_bottom
        if looks_like_title and in_title_band:
            return True
    return False


def _component_static_text(component: dict[str, Any]) -> str:
    data_source = _dict(component.get("dataSourceProps"))
    default_value = data_source.get("defaultValue")
    if isinstance(default_value, list) and default_value and isinstance(default_value[0], dict):
        return _string(default_value[0].get("text"))
    return ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalize_resource(resource: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(resource)
    normalized["version"] = int(normalized.get("version") or 0)
    normalized.setdefault("thumbnailUrl", "")
    normalized.setdefault("provider", "local")
    normalized.setdefault("type", "component")
    normalized["group"] = _normalized_resource_group(normalized.get("group"))
    configuration = _dict(normalized.get("configuration"))
    component_type = _string(configuration.get("componentType")) or "echarts"
    configuration["componentType"] = component_type
    if component_type == "remote":
        configuration.setdefault("fileId", "")
        configuration.pop("javaScript", None)
    else:
        configuration.setdefault("javaScript", "")
    normalized["configuration"] = configuration
    return normalized


def _validate_component_resources(components: list[Any], resources: list[dict[str, Any]]) -> None:
    resource_ids = {_string(resource.get("resourceId")) for resource in resources}
    resources_by_id = {_string(resource.get("resourceId")): resource for resource in resources}
    for component in components:
        if not isinstance(component, dict):
            continue
        resource = _dict(_dict(component.get("componentProps")).get("resource"))
        component_resource_id = _string(resource.get("id"))
        if _is_remote_component(component) or _is_advanced_component_type(_string(component.get("type"))):
            remote_resource_id = _string(component.get("type"))
            if remote_resource_id not in resource_ids:
                raise ValueError(f"advanced component references unsaved resourceId: {remote_resource_id}")
            file_id = _string(_dict(component.get("configuration")).get("fileId"))
            resource_file_id = _string(_dict(resources_by_id.get(remote_resource_id, {}).get("configuration")).get("fileId"))
            if not file_id or not resource_file_id:
                raise ValueError(f"advanced component {remote_resource_id} is missing uploaded fileId.")
        if component.get("type") == "resourceComponentEcharts" and component_resource_id and component_resource_id not in resource_ids:
            raise ValueError(f"组件引用了未保存的 resourceId: {component_resource_id}")


def _resources_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    resources = payload.get("resources")
    if isinstance(resources, list):
        return [dict(item) for item in resources if isinstance(item, dict)]
    return []


def _resources_from_components(components: list[Any]) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        raw = component.get("resource") or component.get("resourceEntity")
        if isinstance(raw, dict):
            resources.append(dict(raw))
    return resources


def _dedupe_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for resource in resources:
        resource_id = _string(resource.get("resourceId"))
        if resource_id:
            deduped[resource_id] = resource
    return list(deduped.values())


def _detect_stage(text: str) -> str:
    lowered = text.lower()
    if "initialization" in lowered or "当前阶段：initialization" in text or "当前阶段: initialization" in text:
        return "initialization"
    if "requestedregionid" in lowered or "当前阶段：region" in text or "当前阶段: region" in text:
        return "region"
    return ""


def _requested_region_id(text: str) -> str:
    match = re.search(r"requestedRegionId\s*=\s*([A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1)
    match = re.search(r'"requestedRegionId"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    return ""


def _loads_json_object(text: str) -> dict[str, Any]:
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("模型必须返回 JSON 对象。")
    return parsed


def _parse_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _ensure_svg(value: str) -> str:
    text = value.strip()
    if not text.startswith("<svg"):
        raise ValueError("backgroundSvg 必须是完整 SVG 字符串。")
    return text


def _find_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("id", "fileId", "file_id", "resourceId", "resource_id"):
            value = payload.get(key)
            if isinstance(value, str | int | float) and str(value).strip():
                return str(value).strip()
        for value in payload.values():
            found = _find_id(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_id(item)
            if found:
                return found
    return ""


def _tool_error_text(result: Any) -> str:
    structured = getattr(result, "structured_content", {})
    if isinstance(structured, dict) and isinstance(structured.get("error"), str):
        return structured["error"]
    content = getattr(result, "content", [])
    texts = [str(item.get("text")) for item in content if isinstance(item, dict) and item.get("text")]
    return "; ".join(texts) or "unknown tool error"


def _direct_llm_runtime_options(runtime_options: RuntimeOptions) -> tuple[RuntimeOptions, dict[str, Any]]:
    config = runtime_options.config_options
    chat_url = _first_config_string(
        config,
        "visualBigscreenDirectLlmChatUrl",
        "visual_bigscreen_direct_llm_chat_url",
        "directLlmChatUrl",
        "direct_llm_chat_url",
        "openaiChatCompletionsUrl",
        "openai_chat_completions_url",
        "codexChatCompletionsUrl",
        "codex_chat_completions_url",
    ) or _first_env_string(
        "VISUAL_BIGSCREEN_DIRECT_LLM_CHAT_URL",
        "CC_SWITCH_OPENAI_CHAT_URL",
        "CODEX_OPENAI_CHAT_URL",
    )
    base_url = _first_config_string(
        config,
        "visualBigscreenDirectLlmBaseUrl",
        "visual_bigscreen_direct_llm_base_url",
        "directLlmBaseUrl",
        "direct_llm_base_url",
        "openaiBaseUrl",
        "openai_base_url",
        "codexBaseUrl",
        "codex_base_url",
        "ccSwitch.openai.baseUrl",
        "ccSwitch.openai.base_url",
        "ccSwitch.openai.url",
        "cc_switch.openai.base_url",
    ) or _first_env_string(
        "VISUAL_BIGSCREEN_DIRECT_LLM_BASE_URL",
        "CC_SWITCH_OPENAI_BASE_URL",
        "CODEX_OPENAI_BASE_URL",
    )
    if chat_url and not base_url:
        base_url = chat_url
    api_key = _first_config_string(
        config,
        "visualBigscreenDirectLlmApiKey",
        "visual_bigscreen_direct_llm_api_key",
        "directLlmApiKey",
        "direct_llm_api_key",
        "openaiApiKey",
        "openai_api_key",
        "codexApiKey",
        "codex_api_key",
        "ccSwitch.openai.apiKey",
        "ccSwitch.openai.api_key",
        "cc_switch.openai.api_key",
    ) or _first_env_string(
        "VISUAL_BIGSCREEN_DIRECT_LLM_API_KEY",
        "CC_SWITCH_OPENAI_API_KEY",
        "CODEX_OPENAI_API_KEY",
    )
    model_name = _first_config_string(
        config,
        "visualBigscreenDirectLlmModel",
        "visual_bigscreen_direct_llm_model",
        "directLlmModel",
        "direct_llm_model",
        "openaiModel",
        "openai_model",
        "codexModel",
        "codex_model",
        "ccSwitch.openai.model",
        "ccSwitch.openai.modelName",
        "ccSwitch.openai.model_name",
        "cc_switch.openai.model",
    ) or _first_env_string(
        "VISUAL_BIGSCREEN_DIRECT_LLM_MODEL",
        "CC_SWITCH_OPENAI_MODEL",
        "CODEX_OPENAI_MODEL",
    )
    timeout_seconds = _first_config_number(
        config,
        "visualBigscreenDirectLlmTimeoutSeconds",
        "visual_bigscreen_direct_llm_timeout_seconds",
        "directLlmTimeoutSeconds",
        "direct_llm_timeout_seconds",
        "openaiTimeoutSeconds",
        "openai_timeout_seconds",
    )
    if timeout_seconds is None:
        timeout_seconds = _first_env_number(
            "VISUAL_BIGSCREEN_DIRECT_LLM_TIMEOUT_SECONDS",
            "CC_SWITCH_OPENAI_TIMEOUT_SECONDS",
            "CODEX_OPENAI_TIMEOUT_SECONDS",
        )

    configured = bool(chat_url or base_url or model_name)
    if not configured:
        return runtime_options, {"enabled": False}
    missing: list[str] = []
    if not base_url:
        missing.append("base_url")
    if not model_name:
        missing.append("model")
    if missing:
        raise RuntimeError("直连模型配置不完整，缺少 " + ", ".join(missing))

    normalized_base_url = _normalize_openai_base_url(base_url)
    update: dict[str, Any] = {
        "base_url": normalized_base_url,
        "api_key": api_key,
        "model_name": model_name,
    }
    if timeout_seconds is not None:
        update["request_timeout_seconds"] = max(1.0, min(float(timeout_seconds), 600.0))
    options = runtime_options.model_copy(update=update, deep=True)
    return options, {
        "enabled": True,
        "model": model_name,
        "base_url": normalized_base_url,
        "chat_url_configured": bool(chat_url),
        "api_key_configured": bool(api_key),
        "timeout_seconds": options.request_timeout_seconds,
    }


def _normalize_openai_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    suffix = "/chat/completions"
    if url.lower().endswith(suffix):
        url = url[: -len(suffix)].rstrip("/")
    return url


def _first_config_string(config: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _config_value(config, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_env_string(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "").strip()
        if value:
            return value
    return ""


def _first_config_number(config: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        parsed = _number_or_none(_config_value(config, key))
        if parsed is not None:
            return parsed
    return None


def _first_env_number(*keys: str) -> float | None:
    for key in keys:
        parsed = _number_or_none(os.getenv(key))
        if parsed is not None:
            return parsed
    return None


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _config_value(config: dict[str, Any], key: str) -> Any:
    if key in config:
        return config.get(key)
    if "." not in key:
        return None
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _llm_event_payload(agent_config: AgentConfig, runtime_options: RuntimeOptions) -> dict[str, Any]:
    return {
        "model": runtime_options.model_name or agent_config.model.model or agent_config.model.default_model,
        "base_url_configured": bool(runtime_options.base_url or agent_config.model.base_url),
        "tool_count": 0,
    }


def _last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return messages[-1].content if messages else ""


def _json_reply(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _string(value: Any) -> str:
    return str(value).strip() if isinstance(value, str | int | float) else ""


def _safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "resource"
