from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.artifacts.store import ArtifactStore, ThreadPaths
from app.core.skills.runner_types import SkillRunResult


DEFAULT_API_BASE_URL = "http://127.0.0.1:18080/api/v1"
DEFAULT_TIMEOUT_SECONDS = 45


@dataclass(frozen=True)
class EndpointSpec:
    method: str
    path: str
    body_fields: tuple[str, ...] = ()
    query_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    dangerous: bool = False
    download_field: str | None = None
    download_filename: str | None = None


SKILL_TITLES: dict[str, str] = {
    "tianjin-chatbi-analyst": "天津园区智能问数结果",
    "tianjin-document-generator": "天津园区商务文档结果",
    "tianjin-content-operator": "天津园区内容运营结果",
    "tianjin-sentiment-monitor": "天津园区舆情分析结果",
    "tianjin-market-research": "天津园区市场调研结果",
    "tianjin-meeting-minutes": "天津园区会议纪要结果",
    "tianjin-calendar-reminder": "天津园区日历提醒结果",
    "tianjin-rpa-operator": "天津园区 RPA 结果",
    "tianjin-system-monitor": "天津园区系统监控结果",
    "tianjin-efficiency-training": "天津园区效率培训结果",
}


ENDPOINTS: dict[str, dict[str, EndpointSpec]] = {
    "tianjin-chatbi-analyst": {
        "query": EndpointSpec(
            "POST",
            "/chatbi/query",
            body_fields=("query", "question", "time_range", "timeRange", "datasource", "chart_type", "context", "session_id"),
            required_fields=("query",),
        ),
        "list_datasources": EndpointSpec("GET", "/chatbi/datasources"),
        "get_datasource_schema": EndpointSpec("GET", "/chatbi/datasources/{datasource_id}/schema", required_fields=("datasource_id",)),
        "create_datasource": EndpointSpec(
            "POST",
            "/chatbi/datasources",
            body_fields=("name", "ds_type", "config", "description"),
            required_fields=("name", "ds_type", "config"),
        ),
        "test_datasource": EndpointSpec("POST", "/chatbi/datasources/{datasource_id}/test", required_fields=("datasource_id",)),
        "get_chart": EndpointSpec("GET", "/chatbi/charts/{chart_id}", required_fields=("chart_id",)),
        "export_chart": EndpointSpec(
            "POST",
            "/chatbi/export",
            query_fields=("chart_id", "format"),
            required_fields=("chart_id",),
        ),
    },
    "tianjin-document-generator": {
        "generate": EndpointSpec(
            "POST",
            "/document/generate",
            body_fields=("document_type", "template_type", "project_name", "client_name", "key_info", "requirements"),
            required_fields=("document_type", "template_type", "project_name", "client_name", "key_info"),
            download_field="download_url",
            download_filename="generated-document.txt",
        ),
        "templates": EndpointSpec("GET", "/document/templates"),
        "estimate_quote": EndpointSpec(
            "POST",
            "/document/quote-records/estimate",
            body_fields=("clientName", "projectName", "projectDescription", "serviceItems"),
            required_fields=("clientName", "projectName"),
        ),
        "list_quote_records": EndpointSpec("GET", "/document/quote-records", query_fields=("limit",)),
        "save_quote_record": EndpointSpec(
            "POST",
            "/document/quote-records",
            body_fields=(
                "id",
                "clientName",
                "projectName",
                "projectDescription",
                "serviceItems",
                "creator",
                "templateId",
                "templateName",
                "no",
                "content",
                "documentId",
                "provider",
                "status",
                "signMan",
                "signDate",
                "file",
            ),
            required_fields=("clientName", "projectName"),
        ),
        "list_bid_records": EndpointSpec("GET", "/document/bid-records", query_fields=("limit",)),
        "save_bid_record": EndpointSpec(
            "POST",
            "/document/bid-records",
            body_fields=("id", "title", "description", "creator", "status", "files"),
            required_fields=("title",),
        ),
        "export_document": EndpointSpec(
            "POST",
            "/document/export/{document_id}",
            query_fields=("format",),
            required_fields=("document_id",),
            download_field="download_url",
            download_filename="exported-document.txt",
        ),
    },
    "tianjin-content-operator": {
        "generate": EndpointSpec(
            "POST",
            "/content/generate",
            body_fields=("platform", "topic", "content_type", "audience", "keywords", "style", "length"),
            required_fields=("platform", "topic", "content_type"),
        ),
        "generate_xhs_poster": EndpointSpec(
            "POST",
            "/content/xiaohongshu/poster",
            body_fields=("title", "subtitle", "keywords", "scene", "style"),
            required_fields=("keywords",),
        ),
        "publish_xhs": EndpointSpec(
            "POST",
            "/content/xiaohongshu/publish",
            body_fields=("id", "title", "content", "tags", "category", "scheduled_time", "auto_generate_images", "auto_tags", "style"),
            required_fields=("title", "content"),
            dangerous=True,
        ),
        "list_xhs_records": EndpointSpec("GET", "/content/xiaohongshu/records", query_fields=("limit",)),
        "save_xhs_record": EndpointSpec(
            "POST",
            "/content/xiaohongshu/records",
            body_fields=(
                "id",
                "title",
                "cover",
                "tags",
                "content",
                "column",
                "creator",
                "status",
                "publishTime",
                "previewUrl",
                "postId",
                "provider",
                "publishPlatforms",
                "publishNote",
            ),
            required_fields=("title",),
        ),
        "xhs_insights": EndpointSpec("GET", "/content/xiaohongshu/insights"),
        "list_wechat_accounts": EndpointSpec("GET", "/content/wechat/accounts", query_fields=("limit",)),
        "list_wechat_records": EndpointSpec("GET", "/content/wechat/records", query_fields=("limit", "status", "keyword")),
        "save_wechat_record": EndpointSpec(
            "POST",
            "/content/wechat/records",
            body_fields=(
                "id",
                "title",
                "cover",
                "image",
                "content",
                "column",
                "creator",
                "status",
                "publishTime",
                "publishAccount",
                "suggestion",
                "auto",
                "wechat",
                "push",
                "approval",
            ),
            required_fields=("title",),
        ),
        "publish_wechat": EndpointSpec(
            "POST",
            "/content/wechat/publish",
            body_fields=("ids", "publish_time", "publish_account"),
            required_fields=("ids",),
            dangerous=True,
        ),
        "history": EndpointSpec("GET", "/content/history", query_fields=("content_type", "limit")),
    },
    "tianjin-sentiment-monitor": {
        "analyze": EndpointSpec("POST", "/sentiment/analyze", body_fields=("text", "source"), required_fields=("text",)),
        "monitor": EndpointSpec("GET", "/sentiment/monitor", query_fields=("days", "source")),
        "generate_report": EndpointSpec(
            "POST",
            "/sentiment/reports/generate",
            body_fields=("report_type", "days", "title", "source", "created_by"),
        ),
        "list_reports": EndpointSpec("GET", "/sentiment/reports", query_fields=("report_type", "limit")),
        "list_records": EndpointSpec("GET", "/sentiment/records", query_fields=("limit", "source", "sentiment_label")),
        "list_rules": EndpointSpec("GET", "/sentiment/rules", query_fields=("limit",)),
        "save_rule": EndpointSpec(
            "POST",
            "/sentiment/rules",
            body_fields=("id", "name", "channel", "rules", "description", "keywords", "sentiment_threshold", "action", "is_active"),
            required_fields=("name",),
        ),
        "list_alerts": EndpointSpec("GET", "/sentiment/alerts", query_fields=("limit",)),
        "list_knowledge": EndpointSpec("GET", "/sentiment/knowledge", query_fields=("keyword", "limit")),
        "search": EndpointSpec(
            "POST",
            "/sentiment/search",
            body_fields=("query", "search_type", "page", "page_size"),
            required_fields=("query",),
        ),
    },
    "tianjin-market-research": {
        "search": EndpointSpec(
            "POST",
            "/market/search",
            body_fields=("query", "search_type", "page", "page_size"),
            required_fields=("query",),
        ),
        "latest": EndpointSpec("GET", "/market/latest", query_fields=("search_type", "limit")),
        "parse_demand": EndpointSpec(
            "POST",
            "/market/parse-demand",
            body_fields=("demand", "search_type"),
            required_fields=("demand",),
        ),
        "search_multi": EndpointSpec(
            "POST",
            "/market/search-multi",
            body_fields=("keywords", "search_type", "page_size"),
            required_fields=("keywords",),
        ),
        "ai_insights": EndpointSpec(
            "POST",
            "/market/ai-insights",
            body_fields=("query", "results"),
            required_fields=("query",),
        ),
        "list_surveys": EndpointSpec("GET", "/market/surveys"),
        "survey_templates": EndpointSpec("GET", "/market/survey-templates"),
        "save_survey": EndpointSpec(
            "POST",
            "/market/surveys",
            body_fields=("id", "title", "type", "targetAudience", "sampleSize", "description", "dateRange", "questions"),
            required_fields=("title", "type"),
        ),
        "analyze_survey": EndpointSpec("POST", "/market/surveys/{survey_id}/analyze", required_fields=("survey_id",)),
        "prediction": EndpointSpec(
            "POST",
            "/market/prediction",
            body_fields=("topic", "results", "competitors", "focus"),
            required_fields=("topic",),
        ),
    },
    "tianjin-meeting-minutes": {
        "analyze": EndpointSpec(
            "POST",
            "/meeting/analyze",
            body_fields=("name", "description", "transcriptLines"),
            required_fields=("name", "transcriptLines"),
        ),
        "list_records": EndpointSpec("GET", "/meeting/records", query_fields=("limit", "status", "keyword")),
        "get_record": EndpointSpec("GET", "/meeting/records/{record_id}", required_fields=("record_id",)),
        "save_record": EndpointSpec(
            "POST",
            "/meeting/records",
            body_fields=(
                "id",
                "name",
                "description",
                "mode",
                "status",
                "files",
                "transcriptLines",
                "summaryPoints",
                "tasks",
                "keywords",
                "topics",
                "speakerStats",
                "minutesHtml",
            ),
            required_fields=("name",),
        ),
    },
    "tianjin-calendar-reminder": {
        "parse_event": EndpointSpec(
            "POST",
            "/calendar/events/parse",
            body_fields=("natural_text", "base_time", "user_timezone"),
            required_fields=("natural_text",),
        ),
        "create_event": EndpointSpec(
            "POST",
            "/calendar/events",
            body_fields=(
                "title",
                "description",
                "start_time",
                "end_time",
                "all_day",
                "location",
                "priority",
                "category",
                "repeat_type",
                "repeat_config",
                "reminder_config",
            ),
            required_fields=("title", "start_time"),
        ),
        "list_events": EndpointSpec("GET", "/calendar/events", query_fields=("start_date", "end_date", "category", "priority", "status", "keyword")),
        "create_reminder": EndpointSpec(
            "POST",
            "/calendar/reminders",
            body_fields=("event_id", "title", "content", "remind_time", "priority", "repeat_type", "repeat_config", "reminder_type", "metadata"),
            required_fields=("title", "remind_time"),
        ),
        "list_reminders": EndpointSpec("GET", "/calendar/reminders", query_fields=("status", "priority", "limit")),
        "efficiency_suggestions": EndpointSpec("GET", "/calendar/efficiency/suggestions"),
    },
    "tianjin-rpa-operator": {
        "list_processes": EndpointSpec("GET", "/rpa/processes"),
        "templates": EndpointSpec("GET", "/rpa/templates"),
        "create_process": EndpointSpec(
            "POST",
            "/rpa/processes",
            body_fields=("name", "description", "process_type", "config", "schedule", "enabled"),
            required_fields=("name", "description", "process_type", "config"),
        ),
        "start_process": EndpointSpec("POST", "/rpa/processes/{process_id}/start", required_fields=("process_id",), dangerous=True),
        "stop_process": EndpointSpec("POST", "/rpa/processes/{process_id}/stop", required_fields=("process_id",), dangerous=True),
        "test_process": EndpointSpec("POST", "/rpa/processes/{process_id}/test", required_fields=("process_id",)),
        "logs": EndpointSpec("GET", "/rpa/processes/{process_id}/logs", query_fields=("limit",), required_fields=("process_id",)),
        "statistics": EndpointSpec("GET", "/rpa/statistics"),
    },
    "tianjin-system-monitor": {
        "health": EndpointSpec("GET", "/system/health"),
        "info": EndpointSpec("GET", "/system/info"),
        "stats": EndpointSpec("GET", "/system/stats"),
        "overview": EndpointSpec("GET", "/system/overview", query_fields=("time_range",)),
        "system_logs": EndpointSpec("GET", "/system/logs/system", query_fields=("level", "name", "service_name", "limit")),
        "access_logs": EndpointSpec("GET", "/system/logs/access", query_fields=("ip", "path", "method", "limit")),
    },
    "tianjin-efficiency-training": {
        "training_overview": EndpointSpec("GET", "/efficiency/training/overview"),
        "create_training_plan": EndpointSpec(
            "POST",
            "/efficiency/training/plans",
            body_fields=("title", "description", "courseId", "targetPath"),
            required_fields=("title",),
        ),
        "start_course": EndpointSpec("POST", "/efficiency/training/courses/{course_id}/start", required_fields=("course_id",)),
        "list_test_records": EndpointSpec("GET", "/efficiency/training/test-records"),
        "save_test_record": EndpointSpec(
            "POST",
            "/efficiency/training/test-records",
            body_fields=("testName", "score", "totalQuestions", "correctAnswers", "duration", "passed"),
            required_fields=("testName", "score", "totalQuestions", "correctAnswers"),
        ),
    },
}


DEFAULT_ACTION: dict[str, str] = {
    "tianjin-chatbi-analyst": "query",
    "tianjin-document-generator": "generate",
    "tianjin-content-operator": "generate",
    "tianjin-sentiment-monitor": "analyze",
    "tianjin-market-research": "search",
    "tianjin-meeting-minutes": "analyze",
    "tianjin-calendar-reminder": "parse_event",
    "tianjin-rpa-operator": "list_processes",
    "tianjin-system-monitor": "overview",
    "tianjin-efficiency-training": "training_overview",
}


def run(skill_name: str, spec: dict[str, Any], paths: ThreadPaths, artifact_store: ArtifactStore) -> SkillRunResult:
    action = _action_for(skill_name, spec)
    endpoint = _endpoint(skill_name, action)
    normalized_spec = _normalize_spec(skill_name, action, spec)
    mode = _execution_mode(normalized_spec)
    if endpoint.dangerous and not _confirmed(normalized_spec):
        return _input_required(skill_name, action, ["confirm"], dangerous=True)
    if mode != "backend":
        _apply_local_defaults(skill_name, action, normalized_spec)
        missing = _missing_fields(normalized_spec, _local_required_fields(skill_name, action))
        if missing:
            return _input_required(skill_name, action, missing)
        return _run_local_skill(skill_name, action, normalized_spec, paths, artifact_store)

    missing = _missing_fields(normalized_spec, endpoint.required_fields)
    if missing:
        return _input_required(skill_name, action, missing)

    request = _build_request(normalized_spec, endpoint)
    response = _call_tianjin_api(request)
    download_artifact = _download_linked_artifact(response, endpoint, request, paths, artifact_store)
    markdown = _render_markdown(skill_name, action, request, response, download_artifact)
    base_name = _safe_name(f"{skill_name}-{action}")
    md_artifact = artifact_store.write_text_artifact(paths, f"{base_name}.md", markdown)
    json_artifact = artifact_store.write_text_artifact(
        paths,
        f"{base_name}.json",
        json.dumps(
            {
                "skill_name": skill_name,
                "action": action,
                "request": request_public_dict(request),
                "response": response,
                "download_artifact": download_artifact.name if download_artifact else None,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )
    outputs = [md_artifact, json_artifact]
    if download_artifact is not None:
        outputs.append(download_artifact)
    return SkillRunResult(
        skill_name=skill_name,
        outputs=outputs,
        data={
            "skill_name": skill_name,
            "action": action,
            "status": response["status"],
            "ok": response["ok"],
            "success": _response_success(response),
            "message": _response_message(response),
            "api_base_url": request["api_base_url"],
            "path": request["path"],
            "method": request["method"],
            "artifact_names": [artifact.name for artifact in outputs],
            "result_preview": _preview_data(response.get("json")),
            "diagnostic": response.get("diagnostic"),
        },
    )


def _execution_mode(spec: dict[str, Any]) -> str:
    raw = str(spec.get("mode") or spec.get("execution_mode") or os.getenv("TIANJIN_SKILL_MODE") or "local")
    return "backend" if raw.strip().lower() in {"backend", "http", "external", "tianjin-backend"} else "local"


LOCAL_REQUIRED_FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("tianjin-chatbi-analyst", "query"): ("query",),
    ("tianjin-sentiment-monitor", "analyze"): ("text",),
    ("tianjin-market-research", "search"): ("query",),
    ("tianjin-market-research", "parse_demand"): ("demand",),
    ("tianjin-calendar-reminder", "parse_event"): ("natural_text",),
    ("tianjin-meeting-minutes", "analyze"): ("transcriptLines",),
}


def _local_required_fields(skill_name: str, action: str) -> tuple[str, ...]:
    return LOCAL_REQUIRED_FIELDS.get((skill_name, action), ())


def _apply_local_defaults(skill_name: str, action: str, spec: dict[str, Any]) -> None:
    if skill_name == "tianjin-document-generator" and action in {"generate", "export_document"}:
        spec.setdefault("document_type", "proposal")
        spec.setdefault("template_type", "standard")
        if _missing_value(spec.get("project_name")):
            spec["project_name"] = _spec_text(spec, ("projectName", "title", "project", "name"), "天津园区项目")
        if _missing_value(spec.get("client_name")):
            spec["client_name"] = _spec_text(spec, ("clientName", "customer_name", "customerName"), "客户")
        if _missing_value(spec.get("key_info")):
            spec["key_info"] = _spec_text(spec, ("keyInfo", "description", "content", "requirements"), "园区运营智能化建设")
    elif skill_name == "tianjin-content-operator" and action == "generate":
        spec.setdefault("platform", "wechat")
        if _missing_value(spec.get("topic")):
            spec["topic"] = _spec_text(spec, ("title", "query", "text", "content"), "天津园区智能运营")
        spec.setdefault("content_type", "article")
    elif skill_name == "tianjin-efficiency-training" and action == "create_training_plan":
        if _missing_value(spec.get("title")):
            spec["title"] = "天津园区运营效率提升计划"
    elif skill_name == "tianjin-rpa-operator" and action == "create_process":
        spec.setdefault("name", "园区运营辅助流程")
        spec.setdefault("description", "基于当前会话文件生成 RPA 流程方案")
        spec.setdefault("process_type", "report")
        spec.setdefault("config", {})


def _run_local_skill(
    skill_name: str,
    action: str,
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> SkillRunResult:
    prompt = _local_prompt(skill_name, action, spec, paths)
    generated = _complete_with_runtime_model(spec, prompt) or _local_fallback_content(skill_name, action, spec, paths)
    data = _local_structured_data(skill_name, action, spec, generated, paths)
    title = SKILL_TITLES.get(skill_name, skill_name)
    base_name = _safe_name(f"{skill_name}-{action}")
    markdown = _render_local_markdown(skill_name, action, title, generated, data, spec, paths)
    md_artifact = artifact_store.write_text_artifact(paths, f"{base_name}.md", markdown)
    json_artifact = artifact_store.write_text_artifact(
        paths,
        f"{base_name}.json",
        json.dumps(
            {
                "skill_name": skill_name,
                "action": action,
                "mode": "local",
                "model": _llm_model_name(spec),
                "llm_configured": _llm_configured(spec),
                "input": _public_spec(spec),
                "data": data,
                "content": generated,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
    )
    outputs = [md_artifact, json_artifact]
    if skill_name == "tianjin-document-generator" and action in {"generate", "export_document"}:
        filename = _safe_name(f"{_spec_text(spec, ('project_name', 'projectName', 'title'), '天津园区项目')}-{_document_type_label(spec)}.md")
        document_artifact = artifact_store.write_text_artifact(paths, filename, generated.strip() + "\n")
        outputs.append(document_artifact)
    return SkillRunResult(
        skill_name=skill_name,
        outputs=outputs,
        data={
            "skill_name": skill_name,
            "action": action,
            "mode": "local",
            "ok": True,
            "success": True,
            "message": "已使用运行时大模型生成结果。" if _llm_configured(spec) else "已使用本地模板生成结果。",
            "llm_configured": _llm_configured(spec),
            "model": _llm_model_name(spec),
            "artifact_names": [artifact.name for artifact in outputs],
            "result_preview": _preview_data(data),
        },
    )


def format_reply(skill_name: str, verification: object, run_result: Any) -> str:
    data = getattr(run_result, "data", {}) or {}
    artifacts = getattr(run_result, "outputs", []) or []
    names = [getattr(item, "name", "") for item in artifacts if getattr(item, "name", "")]
    status = data.get("status")
    success = data.get("success")
    message = str(data.get("message") or "").strip()
    if success is False:
        prefix = f"{SKILL_TITLES.get(skill_name, skill_name)}调用失败"
    else:
        prefix = f"{SKILL_TITLES.get(skill_name, skill_name)}已完成"
    lines = [prefix]
    mode = data.get("mode")
    if mode:
        lines.append(f"执行模式：{mode}")
    if status:
        lines.append(f"HTTP 状态：{status}")
    if message:
        lines.append(f"接口消息：{message}")
    if names:
        lines.append(f"产物：{', '.join(names)}")
    return "\n".join(lines)


def _local_prompt(skill_name: str, action: str, spec: dict[str, Any], paths: ThreadPaths) -> str:
    context = _thread_context(paths)
    task = _task_summary(skill_name, action, spec)
    return (
        "你是天津园区运营智能体的专业技能执行器。请基于用户输入和当前会话文件，"
        "直接产出可交付的中文 Markdown 成果，不要声明你需要访问天津后端。\n\n"
        f"技能：{skill_name}\n"
        f"动作：{action}\n"
        f"任务：{task}\n\n"
        f"结构化输入：\n```json\n{json.dumps(_public_spec(spec), ensure_ascii=False, indent=2, default=str)[:12000]}\n```\n\n"
        f"当前会话文件上下文：\n{context}\n\n"
        "输出要求：\n"
        "1. 给出清晰标题、结论、关键发现、执行步骤和后续建议。\n"
        "2. 如果是文档/纪要/报告类任务，直接写成可交付正文。\n"
        "3. 如果信息不足，基于已知输入给出合理假设，并列出需要用户补充的字段。\n"
        "4. 不要输出空泛说明，不要返回只有 JSON 的结果。\n"
    )


def _complete_with_runtime_model(spec: dict[str, Any], prompt: str) -> str:
    base_url = str(spec.get("_llm_base_url") or "").rstrip("/")
    model = _llm_model_name(spec)
    if not base_url or not model:
        return ""
    headers = {"Content-Type": "application/json"}
    api_key = str(spec.get("_llm_api_key") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是面向园区运营的企业级助手，擅长把输入整理成可执行、可交付的中文成果。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": _float_value(spec.get("_llm_temperature"), 0.4),
        "max_tokens": _int_value(spec.get("_llm_max_tokens"), 2048),
    }
    top_p = spec.get("_llm_top_p")
    if top_p is not None:
        payload["top_p"] = _float_value(top_p, 1.0)
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(spec.get("_llm_request_timeout_seconds") or 120)) as response:
            parsed = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return ""
    choices = parsed.get("choices") if isinstance(parsed, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else ""
    return str(content or "").strip()


def _local_fallback_content(skill_name: str, action: str, spec: dict[str, Any], paths: ThreadPaths) -> str:
    title = SKILL_TITLES.get(skill_name, skill_name)
    task = _task_summary(skill_name, action, spec)
    context = _thread_context(paths)
    if skill_name == "tianjin-meeting-minutes":
        return _meeting_minutes_content(spec)
    if skill_name == "tianjin-calendar-reminder":
        return _calendar_content(spec)
    if skill_name == "tianjin-document-generator":
        return _document_content(spec)
    if skill_name == "tianjin-sentiment-monitor":
        return _sentiment_content(spec)
    if skill_name == "tianjin-market-research":
        return _market_content(spec)
    if skill_name == "tianjin-content-operator":
        return _content_ops_content(spec)
    if skill_name == "tianjin-rpa-operator":
        return _rpa_content(spec)
    if skill_name == "tianjin-system-monitor":
        return _system_content(spec, paths)
    if skill_name == "tianjin-efficiency-training":
        return _efficiency_content(spec)
    if skill_name == "tianjin-chatbi-analyst":
        return _chatbi_content(spec)
    return (
        f"# {title}\n\n"
        f"## 任务\n\n{task}\n\n"
        "## 当前结论\n\n已基于当前输入完成园区运营分析，并生成本地成果。\n\n"
        f"## 会话文件上下文\n\n{context}\n"
    )


def _local_structured_data(skill_name: str, action: str, spec: dict[str, Any], content: str, paths: ThreadPaths) -> dict[str, Any]:
    return {
        "title": SKILL_TITLES.get(skill_name, skill_name),
        "action": action,
        "mode": "local",
        "summary": _first_heading_or_line(content),
        "thread_id": paths.thread_id,
        "workspace": str(paths.workspace),
        "uploads": str(paths.uploads),
        "outputs": str(paths.outputs),
        "input_fields": sorted(key for key in _public_spec(spec)),
    }


def _render_local_markdown(
    skill_name: str,
    action: str,
    title: str,
    content: str,
    data: dict[str, Any],
    spec: dict[str, Any],
    paths: ThreadPaths,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- 技能：`{skill_name}`",
        f"- 动作：`{action}`",
        "- 执行模式：`local`",
        f"- 大模型：`{_llm_model_name(spec) or '未配置，使用本地模板'}`",
        f"- 会话：`{paths.thread_id}`",
        f"- 生成时间：`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        "## 成果",
        "",
        content.strip(),
        "",
        "## 结构化摘要",
        "",
        "```json",
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    return "\n".join(lines).strip() + "\n"


def _action_for(skill_name: str, spec: dict[str, Any]) -> str:
    raw = str(spec.get("action") or DEFAULT_ACTION.get(skill_name) or "").strip()
    if not raw:
        raise KeyError(f"Skill action is required: {skill_name}")
    return raw


def _meeting_minutes_content(spec: dict[str, Any]) -> str:
    name = _spec_text(spec, ("name", "title"), "天津园区会议")
    lines = _normalize_transcript_lines(spec.get("transcriptLines") or spec.get("transcript_text") or "")
    summary_items = [line["content"] for line in lines[:6] if line.get("content")]
    task_items = [
        line["content"]
        for line in lines
        if any(word in line.get("content", "") for word in ["负责", "跟进", "完成", "确认", "整改", "交付", "请"])
    ]
    transcript = "\n".join(
        f"- {line.get('timestamp') or '未标注时间'} {line.get('speaker') or '现场发言'}：{line.get('content')}"
        for line in lines
    ) or "- 暂无转写内容。"
    return (
        f"# {name}纪要\n\n"
        "## 会议摘要\n\n"
        + "\n".join(f"- {item}" for item in (summary_items or ["当前输入未提供明确转写内容，需补充会议文本。"]))
        + "\n\n## 待办事项\n\n"
        + "\n".join(f"- {item}" for item in (task_items or ["暂无明确行动项。"]))
        + "\n\n## 原始转写\n\n"
        + transcript
        + "\n"
    )


def _calendar_content(spec: dict[str, Any]) -> str:
    text = _spec_text(spec, ("natural_text", "text", "title"), "天津园区日程")
    normalized = _normalize_chinese_time_text(text)
    return (
        "# 日程与提醒建议\n\n"
        f"## 原始描述\n\n{normalized}\n\n"
        "## 建议日程\n\n"
        f"- 标题：{_extract_calendar_title(normalized)}\n"
        f"- 时间：{_extract_calendar_time_hint(normalized)}\n"
        f"- 地点：{_extract_location_hint(normalized)}\n"
        "- 优先级：medium\n\n"
        "## 后续动作\n\n"
        "- 如需写入日历，请确认具体开始时间、结束时间和提醒方式。\n"
    )


def _document_content(spec: dict[str, Any]) -> str:
    project = _spec_text(spec, ("project_name", "projectName", "project", "title"), "天津园区项目")
    client = _spec_text(spec, ("client_name", "clientName", "customer_name"), "客户")
    key_info = _spec_text(spec, ("key_info", "keyInfo", "description", "content"), "建设园区智能运营能力")
    doc_type = _document_type_label(spec)
    return (
        f"# {project} - {doc_type}\n\n"
        "## 一、项目概述\n\n"
        f"{client}拟建设「{project}」，核心目标是{key_info}。\n\n"
        "## 二、建设范围\n\n"
        "- 园区运营数据汇总与智能问答\n"
        "- 舆情监控、会议纪要、日程提醒和商务文档生成\n"
        "- RPA 流程辅助、系统巡检和效率培训\n\n"
        "## 三、交付内容\n\n"
        "- 需求梳理与实施方案\n"
        "- 应用模板、技能工具和产物归档机制\n"
        "- 试运行验证报告与交付清单\n\n"
        "## 四、实施计划\n\n"
        "1. 第 1 周：需求确认与数据/文档样例收集\n"
        "2. 第 2 周：技能编排、应用配置和联调\n"
        "3. 第 3 周：试运行、问题修复和交付验收\n\n"
        "## 五、风险与建议\n\n"
        "- 明确数据权限和上传文件范围。\n"
        "- 对发布、启动流程等副作用动作保留人工确认。\n"
    )


def _sentiment_content(spec: dict[str, Any]) -> str:
    text = _spec_text(spec, ("text", "content", "query"), "暂无舆情文本")
    negative_terms = ["延迟", "投诉", "故障", "风险", "不满", "告警", "异常"]
    positive_terms = ["稳定", "满意", "提升", "顺利", "完成", "优化"]
    negative = [word for word in negative_terms if word in text]
    positive = [word for word in positive_terms if word in text]
    tendency = "负向风险" if len(negative) > len(positive) else "正向/中性"
    return (
        "# 舆情分析报告\n\n"
        f"## 情绪倾向\n\n{tendency}\n\n"
        f"## 关键词\n\n- 正向：{', '.join(positive) or '无明显正向词'}\n- 风险：{', '.join(negative) or '无明显风险词'}\n\n"
        "## 分析文本\n\n"
        f"> {text}\n\n"
        "## 建议\n\n"
        "- 对涉及告警、延迟、投诉的内容建立跟踪项。\n"
        "- 将处理结论同步到会议纪要或运营周报。\n"
    )


def _market_content(spec: dict[str, Any]) -> str:
    topic = _spec_text(spec, ("query", "topic", "demand", "keyword"), "天津智慧园区运营平台")
    return (
        "# 市场调研与机会分析\n\n"
        f"## 调研主题\n\n{topic}\n\n"
        "## 需求拆解\n\n"
        "- 园区综合运营数字化\n"
        "- 数据问答、文档生成、流程自动化和舆情监控\n"
        "- 面向运营人员的低门槛智能助手\n\n"
        "## 机会点\n\n"
        "- 将分散工具整合为统一智能体入口。\n"
        "- 通过会话产物和文件目录沉淀可复用运营知识。\n"
        "- 对日常会议、报价、报告、巡检形成闭环。\n\n"
        "## 风险\n\n"
        "- 真实数据权限和接口质量会影响自动化深度。\n"
        "- 对外发布和流程启动需设置确认机制。\n"
    )


def _content_ops_content(spec: dict[str, Any]) -> str:
    topic = _spec_text(spec, ("topic", "title", "query"), "天津园区智能运营")
    platform = _spec_text(spec, ("platform", "channel"), "wechat")
    return (
        f"# {topic}内容草案\n\n"
        f"发布平台：{platform}\n\n"
        "## 标题建议\n\n"
        f"- {topic}：让园区运营从分散工具走向智能协同\n\n"
        "## 正文草案\n\n"
        "天津园区智能运营能力围绕数据问答、会议纪要、商务文档、舆情监控、日程提醒和流程辅助展开，"
        "帮助运营团队把日常信息沉淀为可检索、可复用、可交付的成果。\n\n"
        "## 标签\n\n"
        "- 智慧园区\n- 智能运营\n- 企业智能体\n"
    )


def _rpa_content(spec: dict[str, Any]) -> str:
    name = _spec_text(spec, ("name", "process_name", "title"), "园区运营辅助流程")
    return (
        "# RPA 流程方案\n\n"
        f"## 流程名称\n\n{name}\n\n"
        "## 适用场景\n\n"
        "- 定期汇总运营数据\n"
        "- 生成日报、周报或巡检记录\n"
        "- 将产物归档到当前会话 outputs\n\n"
        "## 执行步骤\n\n"
        "1. 读取输入文件和历史产物。\n"
        "2. 提取结构化字段。\n"
        "3. 生成报告或同步清单。\n"
        "4. 等待人工确认后执行有副作用动作。\n"
    )


def _system_content(spec: dict[str, Any], paths: ThreadPaths) -> str:
    del spec
    files = _thread_file_summary(paths)
    return (
        "# 园区智能体运行概览\n\n"
        "## 会话目录\n\n"
        f"- workspace：`{paths.workspace}`\n"
        f"- uploads：`{paths.uploads}`\n"
        f"- outputs：`{paths.outputs}`\n"
        f"- memory：`{paths.memory}`\n\n"
        "## 当前文件摘要\n\n"
        f"{files}\n\n"
        "## 建议\n\n"
        "- 先读取 uploads 和历史 outputs，再生成新产物。\n"
        "- 新成果写入 outputs 并更新 manifest，避免只在聊天中返回大段文本。\n"
    )


def _efficiency_content(spec: dict[str, Any]) -> str:
    title = _spec_text(spec, ("title", "topic"), "园区运营效率提升计划")
    return (
        f"# {title}\n\n"
        "## 培训目标\n\n"
        "- 掌握会话文件、产物和技能工具的使用方式。\n"
        "- 能独立完成会议纪要、商务文档、舆情报告和日程计划。\n\n"
        "## 建议路径\n\n"
        "1. 基础：上传文件、读取文件、生成产物。\n"
        "2. 进阶：用技能生成结构化报告。\n"
        "3. 实战：围绕一个园区运营任务完成端到端交付。\n\n"
        "## 考核方式\n\n"
        "- 产物是否写入 outputs。\n"
        "- 报告是否包含结论、行动项和后续计划。\n"
    )


def _chatbi_content(spec: dict[str, Any]) -> str:
    query = _spec_text(spec, ("query", "question", "text"), "园区运营数据分析")
    return (
        "# 智能问数分析\n\n"
        f"## 用户问题\n\n{query}\n\n"
        "## 可执行分析思路\n\n"
        "- 明确指标口径：时间范围、对象范围、统计维度。\n"
        "- 从上传文件、历史产物或用户提供的数据表中寻找数据源。\n"
        "- 输出表格、趋势判断和运营建议。\n\n"
        "## 当前结论\n\n"
        "当前未直接绑定外部数据库。若会话中存在 CSV、Excel、JSON 或历史分析产物，"
        "模型应优先读取这些文件并据此回答；没有数据时应列出所需字段和数据样例。\n"
    )


def _endpoint(skill_name: str, action: str) -> EndpointSpec:
    endpoints = ENDPOINTS.get(skill_name)
    if endpoints is None:
        raise KeyError(f"Unsupported Tianjin skill: {skill_name}")
    endpoint = endpoints.get(action)
    if endpoint is None:
        supported = ", ".join(sorted(endpoints))
        raise KeyError(f"Unsupported action {action!r} for {skill_name}. Supported actions: {supported}")
    return endpoint


def _missing_fields(spec: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for field in fields:
        value = spec.get(field)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(field)
    return missing


FIELD_ALIASES: dict[tuple[str, str], dict[str, tuple[str, ...]]] = {
    ("tianjin-chatbi-analyst", "query"): {
        "query": ("question", "prompt", "text"),
    },
    ("tianjin-chatbi-analyst", "get_datasource_schema"): {
        "datasource_id": ("datasourceId", "id"),
    },
    ("tianjin-chatbi-analyst", "test_datasource"): {
        "datasource_id": ("datasourceId", "id"),
    },
    ("tianjin-document-generator", "generate"): {
        "document_type": ("documentType", "doc_type", "type"),
        "template_type": ("templateType", "template"),
        "project_name": ("projectName", "project", "title"),
        "client_name": ("clientName", "customer_name", "customerName"),
        "key_info": ("keyInfo", "summary", "description", "content"),
    },
    ("tianjin-document-generator", "estimate_quote"): {
        "clientName": ("client_name", "customer_name", "customerName"),
        "projectName": ("project_name", "project"),
        "projectDescription": ("project_description", "description"),
        "serviceItems": ("service_items", "items"),
    },
    ("tianjin-document-generator", "save_quote_record"): {
        "clientName": ("client_name", "customer_name", "customerName"),
        "projectName": ("project_name", "project"),
        "projectDescription": ("project_description", "description"),
        "serviceItems": ("service_items", "items"),
    },
    ("tianjin-document-generator", "export_document"): {
        "document_id": ("documentId", "id"),
    },
    ("tianjin-content-operator", "generate"): {
        "content_type": ("contentType", "type"),
        "topic": ("title", "subject", "query"),
        "platform": ("channel",),
    },
    ("tianjin-content-operator", "publish_xhs"): {
        "scheduled_time": ("scheduledTime", "publish_time", "publishTime"),
    },
    ("tianjin-content-operator", "publish_wechat"): {
        "publish_time": ("publishTime", "scheduled_time", "scheduledTime"),
        "publish_account": ("publishAccount", "account"),
    },
    ("tianjin-sentiment-monitor", "analyze"): {
        "text": ("content", "query", "message"),
    },
    ("tianjin-market-research", "search"): {
        "query": ("keyword", "topic", "text"),
    },
    ("tianjin-market-research", "parse_demand"): {
        "demand": ("query", "topic", "text", "content"),
    },
    ("tianjin-market-research", "ai_insights"): {
        "query": ("topic", "keyword", "text"),
    },
    ("tianjin-market-research", "prediction"): {
        "topic": ("query", "keyword", "text"),
    },
    ("tianjin-meeting-minutes", "analyze"): {
        "name": ("title", "meeting_name", "meetingName"),
        "transcriptLines": ("transcript_lines", "transcript", "lines", "transcripts"),
    },
    ("tianjin-meeting-minutes", "save_record"): {
        "name": ("title", "meeting_name", "meetingName"),
        "transcriptLines": ("transcript_lines", "transcript", "lines", "transcripts"),
    },
    ("tianjin-meeting-minutes", "get_record"): {
        "record_id": ("recordId", "id"),
    },
    ("tianjin-calendar-reminder", "parse_event"): {
        "natural_text": ("naturalText", "text", "prompt", "content", "description"),
        "base_time": ("baseTime", "now"),
        "user_timezone": ("userTimezone", "timezone", "time_zone"),
    },
    ("tianjin-calendar-reminder", "create_event"): {
        "start_time": ("startTime", "start", "time"),
        "end_time": ("endTime", "end"),
        "all_day": ("allDay",),
        "repeat_type": ("repeatType",),
        "repeat_config": ("repeatConfig",),
        "reminder_config": ("reminderConfig",),
    },
    ("tianjin-calendar-reminder", "create_reminder"): {
        "event_id": ("eventId",),
        "remind_time": ("remindTime", "time"),
        "reminder_type": ("reminderType",),
        "repeat_type": ("repeatType",),
        "repeat_config": ("repeatConfig",),
    },
    ("tianjin-rpa-operator", "create_process"): {
        "process_type": ("processType", "type"),
    },
    ("tianjin-rpa-operator", "start_process"): {
        "process_id": ("processId", "id"),
    },
    ("tianjin-rpa-operator", "stop_process"): {
        "process_id": ("processId", "id"),
    },
    ("tianjin-rpa-operator", "test_process"): {
        "process_id": ("processId", "id"),
    },
    ("tianjin-rpa-operator", "logs"): {
        "process_id": ("processId", "id"),
    },
    ("tianjin-efficiency-training", "create_training_plan"): {
        "courseId": ("course_id",),
        "targetPath": ("target_path",),
    },
    ("tianjin-efficiency-training", "start_course"): {
        "course_id": ("courseId", "id"),
    },
}


def _normalize_spec(skill_name: str, action: str, spec: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    for canonical, aliases in FIELD_ALIASES.get((skill_name, action), {}).items():
        _copy_first_alias(normalized, canonical, aliases)
    if skill_name == "tianjin-meeting-minutes":
        _normalize_meeting_spec(normalized)
    elif skill_name == "tianjin-calendar-reminder":
        _normalize_calendar_spec(normalized)
    return normalized


def _copy_first_alias(spec: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    if not _missing_value(spec.get(canonical)):
        return
    for alias in aliases:
        value = spec.get(alias)
        if not _missing_value(value):
            spec[canonical] = value
            return


def _missing_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _normalize_meeting_spec(spec: dict[str, Any]) -> None:
    raw_lines = spec.get("transcriptLines")
    if _missing_value(raw_lines):
        raw_text = _first_present(spec, ("transcript_text", "transcriptText", "text", "content"))
        if not _missing_value(raw_text):
            raw_lines = str(raw_text)
    if not _missing_value(raw_lines):
        spec["transcriptLines"] = _normalize_transcript_lines(raw_lines)


def _normalize_transcript_lines(raw_lines: Any) -> list[dict[str, str]]:
    if isinstance(raw_lines, str):
        return _transcript_lines_from_text(raw_lines)
    if not isinstance(raw_lines, list):
        return []
    lines: list[dict[str, str]] = []
    for item in raw_lines:
        if isinstance(item, str):
            lines.extend(_transcript_lines_from_text(item))
            continue
        if not isinstance(item, dict):
            continue
        content = _first_present(item, ("content", "text", "message", "sentence", "utterance"))
        if _missing_value(content):
            continue
        lines.append(
            {
                "speaker": str(_first_present(item, ("speaker", "speakerName", "name", "role", "user")) or "现场发言"),
                "timestamp": str(_first_present(item, ("timestamp", "time", "start_time", "startTime", "at")) or ""),
                "content": str(content),
            }
        )
    return lines


def _transcript_lines_from_text(text: str) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:\[(?P<time>[^\]]+)\]\s*)?(?P<speaker>[^:：]{1,24})[:：]\s*(?P<content>.+)$", line)
        if match:
            lines.append(
                {
                    "speaker": match.group("speaker").strip() or "现场发言",
                    "timestamp": (match.group("time") or "").strip(),
                    "content": match.group("content").strip(),
                }
            )
            continue
        lines.append({"speaker": "现场发言", "timestamp": "", "content": line})
    return lines


def _normalize_calendar_spec(spec: dict[str, Any]) -> None:
    raw_text = spec.get("natural_text")
    if isinstance(raw_text, str):
        spec["natural_text"] = _normalize_chinese_time_text(raw_text)
    if _missing_value(spec.get("user_timezone")):
        spec["user_timezone"] = "Asia/Shanghai"


CHINESE_HOURS: dict[str, int] = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}


def _normalize_chinese_time_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        value = CHINESE_HOURS.get(match.group("hour"))
        if value is None:
            return match.group(0)
        return f"{match.group('prefix')}{value}{match.group('suffix')}"

    return re.sub(
        r"(?P<prefix>[上下]午|晚上|早上|中午)?(?P<hour>十一|十二|十|[零一二两三四五六七八九])(?P<suffix>[点时])",
        replace,
        text,
    )


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if not _missing_value(value):
            return value
    return None


def _confirmed(spec: dict[str, Any]) -> bool:
    value = spec.get("confirm")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "confirm", "confirmed", "确认"}
    return False


def _input_required(skill_name: str, action: str, missing: list[str], *, dangerous: bool = False) -> SkillRunResult:
    reason = "该动作会触发发布/启动/停止等副作用，需要 confirm=true。" if dangerous else f"缺少必填字段：{', '.join(missing)}"
    return SkillRunResult(
        skill_name=skill_name,
        outputs=[],
        data={
            "requires_input": True,
            "required_inputs": [
                {
                    "stage": action,
                    "type": "confirmation" if dangerous else "missing_fields",
                    "fields": missing,
                    "reason": reason,
                }
            ],
            "action": action,
        },
    )


def _build_request(spec: dict[str, Any], endpoint: EndpointSpec) -> dict[str, Any]:
    api_base_url = _api_base_url(spec)
    path = _render_path(endpoint.path, spec)
    query = {field: spec[field] for field in endpoint.query_fields if field in spec and spec[field] is not None}
    url = _join_url(api_base_url, path)
    if query:
        url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"
    body = {field: spec[field] for field in endpoint.body_fields if field in spec and spec[field] is not None}
    headers = {"Accept": "application/json"}
    token = str(spec.get("api_token") or os.getenv("TIANJIN_API_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    if endpoint.method not in {"GET", "HEAD"}:
        headers["Content-Type"] = "application/json"
    return {
        "method": endpoint.method,
        "url": url,
        "path": path,
        "query": query,
        "body": body,
        "headers": headers,
        "api_base_url": api_base_url,
        "timeout_seconds": _timeout_seconds(spec),
    }


def _api_base_url(spec: dict[str, Any]) -> str:
    raw = str(spec.get("api_base_url") or os.getenv("TIANJIN_API_BASE_URL") or DEFAULT_API_BASE_URL).strip()
    return raw.rstrip("/")


def _timeout_seconds(spec: dict[str, Any]) -> int:
    raw = spec.get("timeout_seconds") or os.getenv("TIANJIN_API_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS
    try:
        return min(max(int(raw), 1), 300)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def _render_path(path: str, spec: dict[str, Any]) -> str:
    rendered = path
    for match in re.findall(r"\{([A-Za-z0-9_]+)\}", path):
        value = str(spec.get(match) or "").strip()
        if not value:
            raise ValueError(f"Path parameter is required: {match}")
        rendered = rendered.replace("{" + match + "}", urllib.parse.quote(value, safe=""))
    return rendered


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _call_tianjin_api(request: dict[str, Any]) -> dict[str, Any]:
    data = None
    if request["method"] not in {"GET", "HEAD"}:
        data = json.dumps(request["body"], ensure_ascii=False, default=str).encode("utf-8")
    urllib_request = urllib.request.Request(
        str(request["url"]),
        data=data,
        headers=dict(request["headers"]),
        method=str(request["method"]),
    )
    try:
        with urllib.request.urlopen(urllib_request, timeout=float(request["timeout_seconds"])) as response:
            body = response.read()
            return _response_payload(response.status, response.headers.get("Content-Type", ""), body)
    except urllib.error.HTTPError as exc:
        return _response_payload(exc.code, exc.headers.get("Content-Type", ""), exc.read())
    except urllib.error.URLError as exc:
        return {
            "ok": False,
            "status": 0,
            "content_type": "",
            "json": None,
            "text": "",
            "diagnostic": f"天津后端不可达：{exc.reason}",
        }
    except TimeoutError:
        return {
            "ok": False,
            "status": 0,
            "content_type": "",
            "json": None,
            "text": "",
            "diagnostic": "天津后端请求超时。",
        }


def _response_payload(status: int, content_type: str, body: bytes) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    parsed: Any = None
    if "json" in content_type.lower() or text.strip().startswith(("{", "[")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    return {
        "ok": 200 <= status < 300,
        "status": status,
        "content_type": content_type,
        "json": parsed,
        "text": "" if parsed is not None else text,
        "diagnostic": None,
    }


def _download_linked_artifact(
    response: dict[str, Any],
    endpoint: EndpointSpec,
    request: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> Any | None:
    if not endpoint.download_field or not response.get("ok"):
        return None
    data = _response_data(response)
    if not isinstance(data, dict):
        return None
    raw_url = data.get(endpoint.download_field)
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    download_url = _absolute_download_url(request["api_base_url"], raw_url)
    download_request = {
        **request,
        "method": "GET",
        "url": download_url,
        "body": {},
        "query": {},
    }
    downloaded = _call_binary(download_request)
    if not downloaded["ok"]:
        return None
    filename = _filename_from_disposition(downloaded.get("content_disposition", "")) or endpoint.download_filename or "download.bin"
    return artifact_store.write_bytes_artifact(paths, _safe_name(filename, keep_extension=True), downloaded["body"])


def _absolute_download_url(api_base_url: str, raw_url: str) -> str:
    if raw_url.startswith(("http://", "https://")):
        return raw_url
    parsed = urllib.parse.urlparse(api_base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return urllib.parse.urljoin(origin, raw_url)


def _call_binary(request: dict[str, Any]) -> dict[str, Any]:
    urllib_request = urllib.request.Request(str(request["url"]), headers=dict(request["headers"]), method="GET")
    try:
        with urllib.request.urlopen(urllib_request, timeout=float(request["timeout_seconds"])) as response:
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "content_type": response.headers.get("Content-Type", ""),
                "content_disposition": response.headers.get("Content-Disposition", ""),
                "body": response.read(),
            }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return {"ok": False, "status": 0, "body": b""}


def _filename_from_disposition(value: str) -> str:
    if not value:
        return ""
    utf8_match = re.search(r"filename\*=UTF-8''([^;]+)", value)
    if utf8_match:
        return urllib.parse.unquote(utf8_match.group(1))
    ascii_match = re.search(r'filename="?([^";]+)"?', value)
    return ascii_match.group(1) if ascii_match else ""


def _render_markdown(
    skill_name: str,
    action: str,
    request: dict[str, Any],
    response: dict[str, Any],
    download_artifact: Any | None,
) -> str:
    title = SKILL_TITLES.get(skill_name, skill_name)
    lines = [
        f"# {title}",
        "",
        f"- 技能：`{skill_name}`",
        f"- 动作：`{action}`",
        f"- 方法：`{request['method']}`",
        f"- 接口：`{request['path']}`",
        f"- HTTP 状态：`{response['status']}`",
        f"- 成功：`{_response_success(response)}`",
        f"- 生成时间：`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
    ]
    message = _response_message(response)
    if message:
        lines.append(f"- 接口消息：{message}")
    if response.get("diagnostic"):
        lines.extend(["", "## 诊断", "", str(response["diagnostic"])])
    if download_artifact is not None:
        lines.extend(["", "## 下载产物", "", f"- `{download_artifact.name}`"])
    data = _response_data(response)
    if data is not None:
        lines.extend(["", "## 业务数据", "", _markdown_value(data)])
    elif response.get("text"):
        lines.extend(["", "## 响应文本", "", "```text", str(response["text"])[:12000], "```"])
    lines.extend(["", "## 请求摘要", "", "```json", json.dumps(request_public_dict(request), ensure_ascii=False, indent=2, default=str), "```"])
    return "\n".join(lines).strip() + "\n"


def _markdown_value(value: Any) -> str:
    if isinstance(value, dict):
        table = _dict_summary_table(value)
        body = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        return f"{table}\n\n```json\n{body[:20000]}\n```"
    if isinstance(value, list):
        preview = value[:20]
        body = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
        suffix = f"\n\n列表共 {len(value)} 项，仅展示前 {len(preview)} 项。" if len(value) > len(preview) else ""
        return f"```json\n{body[:20000]}\n```{suffix}"
    return f"```text\n{str(value)[:20000]}\n```"


def _dict_summary_table(value: dict[str, Any]) -> str:
    rows = ["| 字段 | 摘要 |", "| --- | --- |"]
    for key, item in list(value.items())[:30]:
        rows.append(f"| `{key}` | {_short_cell(item)} |")
    return "\n".join(rows)


def _short_cell(value: Any) -> str:
    if isinstance(value, dict):
        return f"object({len(value)})"
    if isinstance(value, list):
        return f"list({len(value)})"
    text = str(value).replace("\n", " ").strip()
    return text[:160] + ("..." if len(text) > 160 else "")


def _response_data(response: dict[str, Any]) -> Any:
    parsed = response.get("json")
    if isinstance(parsed, dict) and "data" in parsed:
        return parsed.get("data")
    return parsed


def _response_message(response: dict[str, Any]) -> str:
    parsed = response.get("json")
    if isinstance(parsed, dict):
        message = parsed.get("message") or parsed.get("detail") or parsed.get("error")
        if message is not None:
            return str(message)
    return str(response.get("diagnostic") or "")


def _response_success(response: dict[str, Any]) -> bool:
    parsed = response.get("json")
    if isinstance(parsed, dict) and isinstance(parsed.get("success"), bool):
        return bool(parsed["success"])
    return bool(response.get("ok"))


def _preview_data(value: Any) -> Any:
    if isinstance(value, dict):
        data = value.get("data", value)
        if isinstance(data, list):
            return data[:3]
        if isinstance(data, dict):
            return {key: data[key] for key in list(data)[:8]}
        return data
    if isinstance(value, list):
        return value[:3]
    return value


def _task_summary(skill_name: str, action: str, spec: dict[str, Any]) -> str:
    candidates = (
        "query",
        "question",
        "topic",
        "title",
        "name",
        "natural_text",
        "text",
        "content",
        "project_name",
        "projectName",
        "demand",
    )
    text = _spec_text(spec, candidates, "")
    return text or f"执行 {skill_name}/{action}"


def _thread_context(paths: ThreadPaths) -> str:
    files = _thread_file_summary(paths)
    memory_preview = _read_optional_text(paths.memory / "conversation.md", max_chars=3000)
    parts = [files]
    if memory_preview:
        parts.append("### conversation.md 摘要\n\n" + memory_preview)
    return "\n\n".join(parts)


def _thread_file_summary(paths: ThreadPaths) -> str:
    rows: list[str] = []
    for label, root in [
        ("uploads", paths.uploads),
        ("workspace", paths.workspace),
        ("outputs", paths.outputs),
        ("previews", paths.previews),
        ("versions", paths.versions),
    ]:
        files = [path for path in sorted(root.rglob("*")) if path.is_file()][:20]
        if not files:
            rows.append(f"- {label}: 暂无文件")
            continue
        for path in files:
            rows.append(f"- {label}/{path.relative_to(root).as_posix()} ({path.stat().st_size} bytes)")
    return "\n".join(rows)


def _read_optional_text(path: Any, *, max_chars: int) -> str:
    try:
        candidate = path
        if not candidate.is_file():
            return ""
        return candidate.read_text(encoding="utf-8", errors="ignore")[:max_chars].strip()
    except OSError:
        return ""


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in spec.items() if not key.startswith("_")}
    for key in ("api_key", "api_token"):
        if key in public:
            public[key] = "********"
    return public


def _llm_model_name(spec: dict[str, Any]) -> str:
    return str(spec.get("_llm_model") or spec.get("model") or "").strip()


def _llm_configured(spec: dict[str, Any]) -> bool:
    return bool(str(spec.get("_llm_base_url") or "").strip() and _llm_model_name(spec))


def _spec_text(spec: dict[str, Any], keys: tuple[str, ...], fallback: str) -> str:
    for key in keys:
        value = spec.get(key)
        if not _missing_value(value):
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False, default=str)
            return str(value).strip()
    return fallback


def _document_type_label(spec: dict[str, Any]) -> str:
    raw = _spec_text(spec, ("document_type", "documentType", "type"), "商务文档")
    labels = {
        "price_file": "报价文件",
        "quote": "报价文件",
        "bid": "投标文件",
        "contract": "合同草案",
        "proposal": "方案建议书",
    }
    return labels.get(raw, raw)


def _extract_calendar_title(text: str) -> str:
    cleaned = re.sub(r"(今天|明天|后天|上午|下午|晚上|早上|中午|\d{1,2}[点时:：]\d{0,2})", "", text)
    cleaned = re.sub(r"在[^，。,.!?！？]{1,24}", "", cleaned)
    return cleaned.strip(" ，。,.!?！？") or text[:30]


def _extract_calendar_time_hint(text: str) -> str:
    day = "明天" if "明天" in text else "后天" if "后天" in text else "今天" if "今天" in text else "待确认日期"
    time_match = re.search(r"(上午|下午|晚上|早上|中午)?\d{1,2}[点时:：]\d{0,2}", text)
    return f"{day} {time_match.group(0) if time_match else '待确认时间'}"


def _extract_location_hint(text: str) -> str:
    match = re.search(r"在([^，。,.!?！？]{2,30})", text)
    return match.group(1).strip() if match else "待确认地点"


def _first_heading_or_line(content: str) -> str:
    for line in content.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned[:120]
    return ""


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def request_public_dict(request: dict[str, Any]) -> dict[str, Any]:
    headers = dict(request.get("headers") or {})
    if "Authorization" in headers:
        headers["Authorization"] = "********"
    return {
        "method": request.get("method"),
        "url": request.get("url"),
        "path": request.get("path"),
        "query": request.get("query"),
        "body": request.get("body"),
        "headers": headers,
    }


def _safe_name(value: str, *, keep_extension: bool = False) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "-", value).strip(".-")
    if not normalized:
        normalized = "tianjin-result"
    if keep_extension:
        return normalized[:180]
    return normalized[:120]
