from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from typing import Any

from app.core.artifacts.store import ArtifactStore, ThreadPaths
from app.core.skills.runner_types import SkillRunResult


DEFAULT_API_BASE_URL = "/api/dashboard"
DEFAULT_EVENT_STREAM_URL = "/api/dashboard/stream"


def run(skill_name: str, spec: dict[str, Any], paths: ThreadPaths, artifact_store: ArtifactStore) -> SkillRunResult:
    action = _string(spec.get("action")) or "generate_dashboard"
    dashboard = _dashboard_spec(action, spec)
    markdown = _markdown_brief(dashboard)
    preview = _html_preview(dashboard)

    base_name = _safe_name(f"{skill_name}-{action}")
    md_artifact = artifact_store.write_text_artifact(paths, f"{base_name}.md", markdown)
    json_artifact = artifact_store.write_text_artifact(
        paths,
        f"{base_name}.json",
        json.dumps(dashboard, ensure_ascii=False, indent=2, default=str),
    )
    html_artifact = artifact_store.write_text_artifact(paths, f"{base_name}.html", preview)
    outputs = [html_artifact, md_artifact, json_artifact]
    return SkillRunResult(
        skill_name=skill_name,
        outputs=outputs,
        data={
            "skill_name": skill_name,
            "action": action,
            "mode": "local",
            "ok": True,
            "success": True,
            "message": "已生成天津园区大屏驾驶舱方案、数据契约和静态预览。",
            "dashboard": dashboard,
            "event_stream_contract": dashboard["event_stream_contract"],
            "artifact_names": [artifact.name for artifact in outputs],
            "result_preview": {
                "title": dashboard["title"],
                "panel_count": len(dashboard["layout"]["panels"]),
                "kpi_count": len(dashboard["kpis"]),
                "refresh_seconds": dashboard["refresh_seconds"],
            },
        },
    )


def format_reply(skill_name: str, verification: object, run_result: Any) -> str:
    del verification
    data = getattr(run_result, "data", {}) or {}
    artifacts = getattr(run_result, "outputs", []) or []
    names = [getattr(item, "name", "") for item in artifacts if getattr(item, "name", "")]
    title = str((data.get("dashboard") or {}).get("title") or "天津园区大屏驾驶舱")
    lines = [f"{title}已生成"]
    if names:
        lines.append(f"产物：{', '.join(names)}")
    lines.append("建议先查看 HTML 预览，再按 JSON 数据契约接入真实大屏 API 和事件流。")
    return "\n".join(lines)


def _dashboard_spec(action: str, spec: dict[str, Any]) -> dict[str, Any]:
    title = _string(spec.get("screen_title") or spec.get("title")) or _default_title(action)
    scenario = _string(spec.get("scenario")) or _default_scenario(action)
    refresh_seconds = _bounded_int(spec.get("refresh_seconds"), default=5, minimum=1, maximum=3600)
    api_base_url = (_string(spec.get("api_base_url")) or DEFAULT_API_BASE_URL).rstrip("/")
    event_stream_url = _string(spec.get("event_stream_url")) or DEFAULT_EVENT_STREAM_URL
    camera_count = _bounded_int(spec.get("camera_count"), default=128, minimum=0, maximum=10000)
    include_behavior_review = _bool(spec.get("include_behavior_review"), default=True)
    include_public_opinion = _bool(spec.get("include_public_opinion"), default=True)
    include_rpa_status = _bool(spec.get("include_rpa_status"), default=True)
    data_sources = _data_sources(spec, api_base_url, event_stream_url, include_behavior_review, include_public_opinion, include_rpa_status)
    kpis = _kpis(camera_count, include_behavior_review, include_public_opinion, include_rpa_status)
    panels = _panels(include_behavior_review, include_public_opinion, include_rpa_status)
    event_stream_contract = _event_stream_contract(event_stream_url)
    return {
        "schema_version": "tianjin-park-big-screen.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "title": title,
        "scenario": scenario,
        "refresh_seconds": refresh_seconds,
        "theme": {
            "background": "#07131f",
            "panel": "#0d2233",
            "accent": "#22d3ee",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "success": "#10b981",
        },
        "viewport": {"width": 1920, "height": 1080, "scale_mode": "contain"},
        "api_contract": {
            "base_url": api_base_url,
            "overview": f"{api_base_url}/overview",
            "cameras": f"{api_base_url}/cameras",
            "alerts": f"{api_base_url}/alerts",
            "review_stats": f"{api_base_url}/review-stats",
            "system_health": f"{api_base_url}/system-health",
        },
        "event_stream_contract": event_stream_contract,
        "data_sources": data_sources,
        "kpis": kpis,
        "layout": {"columns": 24, "rows": 12, "panels": panels},
        "implementation_notes": [
            "长连接、摄像头持续事件和 checkpoint 建议放在后台任务或 MCP server 层，不放在单次 agent run 内。",
            "大屏前端只消费结构化聚合 API 和事件流，复判结果由 behavior-review 批处理写回业务系统。",
            "所有实时事件都应携带 event_id、camera_id、occurred_at、severity、review_status 和 trace_id，便于去重和追踪。",
        ],
    }


def _data_sources(
    spec: dict[str, Any],
    api_base_url: str,
    event_stream_url: str,
    include_behavior_review: bool,
    include_public_opinion: bool,
    include_rpa_status: bool,
) -> list[dict[str, Any]]:
    raw = spec.get("data_sources")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw if isinstance(item, dict)]
    sources: list[dict[str, Any]] = [
        {"id": "overview", "name": "园区运行总览", "type": "http", "url": f"{api_base_url}/overview", "refresh_seconds": 5},
        {"id": "event-stream", "name": "大屏实时事件流", "type": "sse", "url": event_stream_url, "refresh_seconds": 0},
        {"id": "cameras", "name": "摄像头在线状态", "type": "http", "url": f"{api_base_url}/cameras", "refresh_seconds": 10},
        {"id": "system-health", "name": "系统健康与接口延迟", "type": "http", "url": f"{api_base_url}/system-health", "refresh_seconds": 15},
    ]
    if include_behavior_review:
        sources.append(
            {
                "id": "behavior-review",
                "name": "行为识别复判队列",
                "type": "mcp-or-http",
                "url": f"{api_base_url}/review-stats",
                "recommended_skill": "behavior-review",
                "refresh_seconds": 3,
            }
        )
    if include_public_opinion:
        sources.append(
            {
                "id": "public-opinion",
                "name": "舆情监控摘要",
                "type": "http",
                "url": f"{api_base_url}/public-opinion",
                "recommended_skill": "tianjin-sentiment-monitor",
                "refresh_seconds": 60,
            }
        )
    if include_rpa_status:
        sources.append(
            {
                "id": "rpa-status",
                "name": "RPA 与巡检任务状态",
                "type": "http",
                "url": f"{api_base_url}/rpa-status",
                "recommended_skill": "tianjin-rpa-operator",
                "refresh_seconds": 30,
            }
        )
    return sources


def _kpis(camera_count: int, include_behavior_review: bool, include_public_opinion: bool, include_rpa_status: bool) -> list[dict[str, Any]]:
    kpis = [
        {"id": "camera_online_rate", "label": "摄像头在线率", "value": "98.6%", "trend": "+0.8%", "severity": "normal"},
        {"id": "camera_total", "label": "接入摄像头", "value": camera_count, "trend": "实时", "severity": "normal"},
        {"id": "today_events", "label": "今日事件", "value": 238, "trend": "+12%", "severity": "warning"},
        {"id": "open_alerts", "label": "未闭环告警", "value": 17, "trend": "-5", "severity": "warning"},
    ]
    if include_behavior_review:
        kpis.extend(
            [
                {"id": "pending_review", "label": "待复判", "value": 9, "trend": "3 秒刷新", "severity": "warning"},
                {"id": "false_positive_rate", "label": "误报抑制率", "value": "42%", "trend": "+6%", "severity": "normal"},
            ]
        )
    if include_public_opinion:
        kpis.append({"id": "sentiment_risk", "label": "舆情风险", "value": "低", "trend": "稳定", "severity": "normal"})
    if include_rpa_status:
        kpis.append({"id": "rpa_success_rate", "label": "RPA 成功率", "value": "96.2%", "trend": "+1.1%", "severity": "normal"})
    return kpis


def _panels(include_behavior_review: bool, include_public_opinion: bool, include_rpa_status: bool) -> list[dict[str, Any]]:
    panels = [
        {"id": "kpi-strip", "title": "核心指标", "type": "kpi-grid", "x": 0, "y": 0, "w": 24, "h": 2, "data_source": "overview"},
        {"id": "map", "title": "园区态势分布", "type": "map-or-floorplan", "x": 6, "y": 2, "w": 12, "h": 7, "data_source": "cameras"},
        {"id": "camera-health", "title": "摄像头在线状态", "type": "bar-list", "x": 0, "y": 2, "w": 6, "h": 4, "data_source": "cameras"},
        {"id": "event-timeline", "title": "事件趋势", "type": "line-chart", "x": 18, "y": 2, "w": 6, "h": 4, "data_source": "overview"},
        {"id": "live-events", "title": "实时事件流", "type": "rolling-list", "x": 0, "y": 6, "w": 6, "h": 6, "data_source": "event-stream"},
        {"id": "system-health", "title": "系统健康", "type": "status-grid", "x": 18, "y": 6, "w": 6, "h": 3, "data_source": "system-health"},
    ]
    if include_behavior_review:
        panels.append({"id": "review-queue", "title": "AI 行为复判", "type": "review-board", "x": 6, "y": 9, "w": 12, "h": 3, "data_source": "behavior-review"})
    if include_public_opinion:
        panels.append({"id": "public-opinion", "title": "舆情风险", "type": "sentiment-card", "x": 18, "y": 9, "w": 3, "h": 3, "data_source": "public-opinion"})
    if include_rpa_status:
        panels.append({"id": "rpa-status", "title": "RPA 巡检", "type": "task-status", "x": 21, "y": 9, "w": 3, "h": 3, "data_source": "rpa-status"})
    return panels


def _event_stream_contract(event_stream_url: str) -> dict[str, Any]:
    return {
        "url": event_stream_url,
        "transport": "sse-or-websocket",
        "event_types": [
            "camera.status.changed",
            "alert.created",
            "review.started",
            "review.completed",
            "review.failed",
            "system.health.changed",
        ],
        "payload_schema": {
            "event_id": "string",
            "type": "string",
            "occurred_at": "ISO-8601 string",
            "camera_id": "string?",
            "camera_name": "string?",
            "severity": "low|medium|high|critical",
            "review_status": "pending|confirmed|false_positive|manual_required",
            "trace_id": "string",
            "data": "object",
        },
    }


def _markdown_brief(dashboard: dict[str, Any]) -> str:
    kpi_lines = "\n".join(f"- {item['label']}: {item['value']} ({item['trend']})" for item in dashboard["kpis"])
    source_lines = "\n".join(f"- {item['id']}: {item['name']} -> {item['url']}" for item in dashboard["data_sources"])
    panel_lines = "\n".join(
        f"- {item['title']}: {item['type']} ({item['x']},{item['y']},{item['w']},{item['h']})"
        for item in dashboard["layout"]["panels"]
    )
    notes = "\n".join(f"- {item}" for item in dashboard["implementation_notes"])
    return (
        f"# {dashboard['title']}\n\n"
        f"场景：{dashboard['scenario']}\n\n"
        f"刷新周期：{dashboard['refresh_seconds']} 秒\n\n"
        "## 核心指标\n\n"
        f"{kpi_lines}\n\n"
        "## 数据源契约\n\n"
        f"{source_lines}\n\n"
        "## 页面布局\n\n"
        f"{panel_lines}\n\n"
        "## 实时事件流\n\n"
        f"- URL: `{dashboard['event_stream_contract']['url']}`\n"
        f"- 事件类型: {', '.join(dashboard['event_stream_contract']['event_types'])}\n\n"
        "## 实施建议\n\n"
        f"{notes}\n"
    )


def _html_preview(dashboard: dict[str, Any]) -> str:
    kpis = "".join(
        f"<div class='kpi {html.escape(item['severity'])}'><span>{html.escape(item['label'])}</span><strong>{html.escape(str(item['value']))}</strong><em>{html.escape(str(item['trend']))}</em></div>"
        for item in dashboard["kpis"]
    )
    events = "".join(
        f"<li><b>{label}</b><span>{detail}</span></li>"
        for label, detail in [
            ("北门摄像头", "禁区逗留已进入 AI 复判"),
            ("3 号楼", "设备离线告警已派发巡检"),
            ("停车场", "高峰车流量超过阈值"),
            ("舆情监控", "风险稳定，暂无高危词"),
        ]
    )
    panels = "".join(
        f"<section class='panel panel-{html.escape(item['id'])}'><h2>{html.escape(item['title'])}</h2><p>{html.escape(item['type'])}</p></section>"
        for item in dashboard["layout"]["panels"]
        if item["id"] not in {"kpi-strip", "live-events"}
    )
    title = html.escape(dashboard["title"])
    scenario = html.escape(dashboard["scenario"])
    generated_at = html.escape(str(dashboard["generated_at"]))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; background: #07131f; color: #d7f7ff; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    .screen {{ width: 100vw; min-height: 100vh; padding: 24px; display: grid; grid-template-rows: auto auto 1fr; gap: 16px; }}
    header {{ display: flex; align-items: end; justify-content: space-between; border-bottom: 1px solid rgba(34, 211, 238, .35); padding-bottom: 12px; }}
    h1 {{ margin: 0; font-size: 32px; letter-spacing: 0; }}
    .meta {{ color: #91b9c7; font-size: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(8, minmax(120px, 1fr)); gap: 10px; }}
    .kpi {{ min-height: 92px; background: #0d2233; border: 1px solid rgba(34, 211, 238, .18); padding: 12px; border-radius: 6px; }}
    .kpi span, .kpi em {{ display: block; color: #93b8c6; font-style: normal; }}
    .kpi strong {{ display: block; margin: 8px 0; font-size: 30px; color: #ffffff; }}
    .kpi.warning strong {{ color: #fbbf24; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 2fr 1.1fr; grid-template-rows: 1fr 1fr; gap: 14px; min-height: 0; }}
    .panel {{ background: #0d2233; border: 1px solid rgba(34, 211, 238, .18); border-radius: 6px; padding: 16px; min-height: 180px; overflow: hidden; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 18px; color: #8eeaff; }}
    .panel p {{ color: #98bdc9; }}
    .panel-map {{ grid-row: span 2; min-height: 470px; display: flex; flex-direction: column; }}
    .panel-map::after {{ content: ""; flex: 1; border: 1px dashed rgba(34, 211, 238, .35); background: linear-gradient(135deg, rgba(34,211,238,.08), rgba(16,185,129,.06)); border-radius: 6px; }}
    .live {{ grid-column: 1; grid-row: 2; }}
    ul {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }}
    li {{ border-left: 3px solid #22d3ee; padding: 8px 10px; background: rgba(255,255,255,.04); }}
    li b, li span {{ display: block; }}
    li span {{ color: #9fc4cf; margin-top: 4px; }}
    @media (max-width: 1200px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} .grid {{ grid-template-columns: 1fr; }} .panel-map {{ grid-row: auto; }} }}
  </style>
</head>
<body>
  <main class="screen">
    <header><div><h1>{title}</h1><div class="meta">{scenario}</div></div><div class="meta">generated {generated_at}</div></header>
    <div class="kpis">{kpis}</div>
    <div class="grid">
      {panels}
      <section class="panel live"><h2>实时事件流</h2><ul>{events}</ul></section>
    </div>
  </main>
</body>
</html>
"""


def _default_title(action: str) -> str:
    if action == "camera_review":
        return "天津园区摄像头复判态势大屏"
    if action == "operations_overview":
        return "天津园区运营驾驶舱"
    return "天津园区智能运营大屏"


def _default_scenario(action: str) -> str:
    if action == "camera_review":
        return "摄像头持续告警、AI 行为复判、处置闭环和事件追踪"
    if action == "operations_overview":
        return "园区运营、系统巡检、舆情监控、RPA 任务和业务问数"
    return "园区综合运营态势、实时事件监控和智能复判"


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "开启", "是"}
    return default


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return normalized or "tianjin-park-big-screen"
