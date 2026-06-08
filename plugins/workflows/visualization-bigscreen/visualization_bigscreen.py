from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm import OpenAICompatibleClient
from app.core.skills import SkillRegistry
from app.core.skills.context_files import SkillMarkdownContext, load_skill_markdown_context
from app.core.tools.invocation import ToolInvocationService
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


WORKFLOW_NAME = "visualization_bigscreen_workflow"
BIGSCREEN_OUTPUT_DIR = "generated-bigscreen"
BACKGROUND_PATH = f"{BIGSCREEN_OUTPUT_DIR}/background.svg"
RESOURCE_DIR = f"{BIGSCREEN_OUTPUT_DIR}/resources"
DEFAULT_RESOURCE_GROUP = "[\"vis_oneself_dimension_line-chart\"]"


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
        llm_reply = self._complete_json(agent_config, runtime_options, _initialization_system_prompt(), messages, recorder)
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
        llm_reply = self._complete_json(agent_config, runtime_options, _region_system_prompt(requested_region_id), messages, recorder)
        payload = _loads_json_object(llm_reply)
        region_id = _string(payload.get("regionId")) or requested_region_id
        if requested_region_id and region_id != requested_region_id:
            region_id = requested_region_id
        components = _list(payload.get("components"))
        _validate_component_types(components)
        _validate_region_title_component(region_id, _string(payload.get("regionTitle")), components)
        _validate_map_intent_components(prompt_text, _string(payload.get("regionTitle")), components)
        resources = _resources_from_payload(payload)
        resources.extend(_resources_from_components(components))
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
        return reply, {"resource_save_count": saved_count}

    def _complete_json(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        system_prompt: str,
        messages: list[Message],
        recorder: EventRecorder,
    ) -> str:
        llm_options = runtime_options.model_copy(
            update={
                "selected_mcp_tools": [],
                "response_format": "json",
                "config_options": {**runtime_options.config_options, "enableWorkspaceTools": False},
            },
            deep=True,
        )
        system_prompt = self._prompt_with_selected_skill_context(system_prompt, llm_options, recorder)
        client = OpenAICompatibleClient(agent_config, runtime_options=llm_options)
        reply = client.complete_sync(system_prompt, messages)
        if not reply.strip():
            raise RuntimeError("模型返回为空。")
        return reply

    def _prompt_with_selected_skill_context(
        self,
        system_prompt: str,
        runtime_options: RuntimeOptions,
        recorder: EventRecorder,
    ) -> str:
        skill_name = next((name.strip() for name in runtime_options.selected_skills if name.strip()), "")
        if not skill_name:
            return system_prompt
        try:
            skill = SkillRegistry(self.tool_service.root_dir).get(skill_name)
        except KeyError:
            recorder.emit(
                "skill.context.loaded",
                {
                    "skill_name": skill_name,
                    "found": False,
                    "reason": "skill_not_found",
                },
            )
            return system_prompt
        context = load_skill_markdown_context(skill)
        if context is None:
            recorder.emit(
                "skill.context.loaded",
                {
                    "skill_name": skill_name,
                    "found": True,
                    "skill_md_found": False,
                    "reason": "skill_md_not_found",
                    "manifest_path": str(skill.manifest_path) if skill.manifest_path is not None else "",
                    "plugin_root": str(skill.plugin_root) if skill.plugin_root is not None else "",
                },
            )
            return system_prompt
        references = [
            {
                "path": reference.path,
                "chars": len(reference.content),
                "truncated": reference.truncated,
            }
            for reference in context.references
        ]
        recorder.emit(
            "skill.context.loaded",
            {
                "skill_name": skill_name,
                "found": True,
                "skill_md_found": True,
                "skill_md_path": context.skill_md_path,
                "skill_md_chars": len(context.skill_md),
                "skill_md_truncated": context.skill_md_truncated,
                "reference_count": len(references),
                "total_reference_chars": sum(int(item["chars"]) for item in references),
                "reference_truncated_count": sum(1 for item in references if item["truncated"]),
                "references": references,
            },
        )
        rendered = _render_workflow_skill_context(context)
        if not rendered:
            return system_prompt
        return f"{system_prompt}\n\n{rendered}"

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
        return arguments


def _initialization_system_prompt() -> str:
    return (
        "你是 JetLinks 可视化大屏 JSON 生成器。只返回一个 JSON 对象，不要 Markdown。\n"
        "当前任务是 initialization。大模型只负责生成内容，不要调用任何工具。\n"
        "必须返回字段：pageJson、blueprint、backgroundSvg。\n"
        "pageJson.canvas 使用平台结构，components 必须为空数组。\n"
        "backgroundSvg 必须是完整 SVG 字符串，viewBox 为 0 0 1920 1080，包含固定区域背景和 data-region-id 元数据。\n"
        "backgroundSvg 不要包含真实业务值、表格行、告警文本或交互按钮。\n"
        "blueprint 使用 fixed standard-1920x1080-v1，区域坐标必须和约定一致。\n"
        f"固定 blueprint JSON：{json.dumps(_default_blueprint(), ensure_ascii=False)}"
    )


def _region_system_prompt(requested_region_id: str) -> str:
    return (
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


def _render_workflow_skill_context(context: SkillMarkdownContext | None) -> str:
    if context is None:
        return ""
    lines = [
        "Visualization workflow skill instructions are active.",
        "Use the embedded SKILL.md and declared reference files as schema rules for JSON generation.",
        "The workflow, not the model, performs UploadFile and visualizationService:resource/Add command calls.",
        "Do not invent component keys such as componentType, x, y, width, height, locked, hidden, animations, events, or dataSources when the platform templates do not contain them.",
        "Return only the JSON object required by the current workflow stage.",
        "",
        "## SKILL.md",
        "",
        context.skill_md,
    ]
    if context.skill_md_truncated:
        lines.append("\n[SKILL.md truncated by runtime context limit]")
    if context.references:
        lines.extend(["", "## Declared Reference Files"])
        for reference in context.references:
            lines.extend(
                [
                    "",
                    f"### {reference.path}",
                    "",
                    reference.content,
                ]
            )
            if reference.truncated:
                lines.append(f"\n[{reference.path} truncated by runtime context limit]")
    return "\n".join(lines).strip()


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
        panels.append(
            "<g id=\"region-{id}\" data-region-id=\"{id}\" data-role=\"{role}\" data-x=\"{x}\" data-y=\"{y}\" "
            "data-width=\"{width}\" data-height=\"{height}\" data-content-x=\"{cx}\" data-content-y=\"{cy}\" "
            "data-content-width=\"{cw}\" data-content-height=\"{ch}\">"
            "<rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" rx=\"18\" "
            "fill=\"rgba(8,28,48,0.68)\" stroke=\"rgba(54,220,255,0.58)\"/>"
            "<path d=\"M {line_x} {line_y} H {line_x2}\" stroke=\"#1FEAFF\" stroke-width=\"3\" opacity=\"0.72\"/>"
            "</g>".format(
                id=region["id"],
                role=region["role"],
                x=region["x"],
                y=region["y"],
                width=region["width"],
                height=region["height"],
                cx=region["contentBox"]["x"],
                cy=region["contentBox"]["y"],
                cw=region["contentBox"]["width"],
                ch=region["contentBox"]["height"],
                line_x=region["x"] + 18,
                line_y=region["y"] + 28,
                line_x2=region["x"] + min(260, region["width"] - 24),
            )
        )
    return (
        '<svg viewBox="0 0 1920 1080" width="1920" height="1080" xmlns="http://www.w3.org/2000/svg">'
        "<defs>"
        '<linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%" stop-color="#06111F"/><stop offset="100%" stop-color="#081A30"/></linearGradient>'
        '<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">'
        '<path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(55,180,255,0.12)" stroke-width="1"/></pattern>'
        "</defs>"
        '<rect width="1920" height="1080" fill="url(#bgGrad)"/>'
        '<rect width="1920" height="1080" fill="url(#grid)" opacity="0.72"/>'
        '<ellipse cx="960" cy="520" rx="560" ry="300" fill="rgba(31,234,255,0.12)"/>'
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


def _validate_component_types(components: list[Any]) -> None:
    allowed = {"text", "dateTime", "table", "video", "pseudo", "resourceComponentEcharts"}
    forbidden = {"tabs", "line", "bar", "pie", "gauge", "chart", "map", "echarts"}
    for component in components:
        if not isinstance(component, dict):
            continue
        component_type = _string(component.get("type"))
        if component_type in forbidden:
            raise ValueError(f"禁止生成组件类型 {component_type}，请按规则改用平台模板或 custom ECharts 资源组件。")
        if component_type and component_type not in allowed:
            raise ValueError(f"不支持的可视化组件类型 {component_type}，允许类型为 text/dateTime/table/video/pseudo/resourceComponentEcharts。")


def _validate_map_intent_components(prompt_text: str, region_title: str, components: list[Any]) -> None:
    if not _is_map_intent(f"{prompt_text}\n{region_title}"):
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


def _is_map_intent(text: str) -> bool:
    lowered = text.lower()
    keywords = (
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
    normalized.setdefault("group", DEFAULT_RESOURCE_GROUP)
    configuration = _dict(normalized.get("configuration"))
    configuration.setdefault("componentType", "echarts")
    configuration.setdefault("javaScript", "")
    normalized["configuration"] = configuration
    return normalized


def _validate_component_resources(components: list[Any], resources: list[dict[str, Any]]) -> None:
    resource_ids = {_string(resource.get("resourceId")) for resource in resources}
    for component in components:
        if not isinstance(component, dict):
            continue
        resource = _dict(_dict(component.get("componentProps")).get("resource"))
        component_resource_id = _string(resource.get("id"))
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
