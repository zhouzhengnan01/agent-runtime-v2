from __future__ import annotations

import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = json.load(sys.stdin)
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    outputs_dir = Path(str(payload["outputs_dir"]))
    objective = _objective(spec)
    title = _title(objective)
    output_root = outputs_dir / "generated-bigscreen"
    resources_dir = output_root / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    chart_specs = _chart_specs(objective)
    resources = [_resource(chart) for chart in chart_specs]
    page = _page(title, objective, chart_specs)

    (output_root / "page.json").write_text(
        json.dumps(page, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for resource in resources:
        path = resources_dir / f"{resource['resourceId']}.resource.json"
        path.write_text(json.dumps(resource, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _summary(title, page, resources)
    (output_root / "summary.md").write_text(summary, encoding="utf-8")
    print(
        json.dumps(
            {
                "title": title,
                "outputs": [
                    {"path": "generated-bigscreen/page.json"},
                    *[
                        {"path": f"generated-bigscreen/resources/{resource['resourceId']}.resource.json"}
                        for resource in resources
                    ],
                    {"path": "generated-bigscreen/summary.md"},
                ],
                "data": {
                    "execution_type": "deterministic_bigscreen_generator",
                    "component_count": len(page["components"]),
                    "resource_count": len(resources),
                    "primary_artifact": "generated-bigscreen/page.json",
                    "platform_commands": _platform_commands(resources),
                },
            },
            ensure_ascii=False,
        )
    )


def _objective(spec: dict[str, Any]) -> str:
    for key in ("objective", "message", "prompt", "user_text", "description"):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "生成一个智慧园区运营可视化大屏"


def _title(objective: str) -> str:
    if "能耗" in objective:
        return "智慧能耗运营大屏"
    if "停车" in objective or "车辆" in objective:
        return "园区停车运营大屏"
    if "安防" in objective or "告警" in objective:
        return "智慧园区安防大屏"
    return "智慧园区运营大屏"


def _chart_specs(objective: str) -> list[dict[str, Any]]:
    prefix = f"component_{int(time.time() * 1000)}"
    if "能耗" in objective:
        return [
            {"id": f"{prefix}_trend", "name": "能耗趋势", "kind": "line", "slot": (72, 276, 520, 270)},
            {"id": f"{prefix}_ratio", "name": "能耗构成", "kind": "pie", "slot": (1328, 276, 520, 270)},
            {"id": f"{prefix}_saving", "name": "节能达成率", "kind": "gauge", "slot": (1328, 620, 520, 260)},
        ]
    return [
        {"id": f"{prefix}_alarm", "name": "告警趋势", "kind": "line", "slot": (72, 276, 520, 270)},
        {"id": f"{prefix}_device", "name": "设备在线率", "kind": "gauge", "slot": (1328, 276, 520, 270)},
        {"id": f"{prefix}_traffic", "name": "通行统计", "kind": "bar", "slot": (1328, 620, 520, 260)},
    ]


def _page(title: str, objective: str, chart_specs: list[dict[str, Any]]) -> dict[str, Any]:
    components: list[dict[str, Any]] = [
        _text("screen_title", title, 520, 36, 880, 64, 34, "#eaf7ff", "bold"),
        _text("screen_subtitle", _subtitle(objective), 650, 96, 620, 32, 16, "#84d8ff"),
        _text("metric_device", "设备在线率\n98.6%", 72, 160, 260, 88, 28, "#ffffff", "bold"),
        _text("metric_alarm", "今日告警\n23", 370, 160, 220, 88, 28, "#ffffff", "bold"),
        _text("metric_energy", "能耗同比\n-8.2%", 1328, 160, 240, 88, 28, "#ffffff", "bold"),
        _text("metric_people", "通行人数\n12,486", 1608, 160, 240, 88, 28, "#ffffff", "bold"),
        _text("center_map", "园区运营态势\n设备、人员、车辆、事件联动监控", 650, 286, 650, 440, 30, "#ffffff", "bold"),
        _text("event_list", "重点事件\n1. 东门车辆拥堵已处理\n2. A栋设备离线恢复中\n3. 能耗峰值触发策略\n4. 周界告警复核完成", 72, 620, 520, 260, 20, "#dff7ff"),
    ]
    components.extend(_chart_component(chart) for chart in chart_specs)
    return {
        "canvas": {
            "width": 1920,
            "height": 1080,
            "scale": 0.64,
            "sizeKey": "pc",
            "adaptationType": "AUTO",
            "backgroundColor": "#061526",
            "backgroundImage": {"fileId": ""},
        },
        "components": components,
    }


def _subtitle(objective: str) -> str:
    text = re.sub(r"\s+", " ", objective).strip()
    if len(text) > 36:
        text = text[:36] + "..."
    return text or "实时监测园区运营、告警、能耗与通行态势"


def _text(
    component_id: str,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    font_size: int,
    color: str,
    weight: str = "normal",
) -> dict[str, Any]:
    template = _json_reference("components/text.json")
    template["id"] = f"text_{component_id}"
    template["name"] = text.splitlines()[0][:24] or "文本"
    template["dataSourceProps"]["defaultValue"] = [{"text": text}]
    style = template["componentProps"]["style"]
    style.update({"x": x, "y": y, "width": width, "height": height})
    font = template["componentProps"]["font"]
    font.update(
        {
            "fontSize": font_size,
            "fontWeight": weight,
            "color": color,
            "horizontalAlign": "center",
            "verticalAlign": "center",
            "isTextShadow": True,
            "textShadow": "0 0 16px rgba(77, 208, 255, 0.72)",
        }
    )
    return template


def _chart_component(chart: dict[str, Any]) -> dict[str, Any]:
    template = _json_reference("components/custom-chart.json")
    x, y, width, height = chart["slot"]
    template["id"] = f"resourceComponentEcharts_{chart['id']}"
    template["name"] = chart["name"]
    template["componentProps"]["style"].update({"x": x, "y": y, "width": width, "height": height})
    template["componentProps"]["resource"]["id"] = chart["id"]
    template["componentProps"]["resource"]["version"] = 0
    return template


def _resource(chart: dict[str, Any]) -> dict[str, Any]:
    template = _json_reference("resources/echarts-resource.json")
    template["resourceId"] = chart["id"]
    template["name"] = chart["name"]
    template["group"] = "[\"vis_oneself_dimension__park\"]"
    template["configuration"]["javaScript"] = _echarts_script(chart["kind"], chart["name"])
    return template


def _echarts_script(kind: str, title: str) -> str:
    if kind == "gauge":
        return """function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [{ name: "达成率", value: 98.6 }]
  if (_getflag === "data") return data
  if (_setdata) data = _setdata
  const value = Number((data[0] || {}).value || 0)
  return {
    title: { text: "%s", left: "center", top: 8, textStyle: { color: "#dff7ff", fontSize: 16 } },
    series: [{ type: "gauge", min: 0, max: 100, progress: { show: true, width: 12 }, axisLine: { lineStyle: { width: 12 } }, detail: { color: "#fff", formatter: "{value}%%" }, data: [{ value }] }]
  }
}""" % title
    if kind == "pie":
        return """function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [{ name: "照明", value: 32 }, { name: "空调", value: 44 }, { name: "设备", value: 24 }]
  if (_getflag === "data") return data
  if (_setdata) data = _setdata
  return {
    title: { text: "%s", left: "center", top: 8, textStyle: { color: "#dff7ff", fontSize: 16 } },
    tooltip: { trigger: "item" },
    legend: { bottom: 0, textStyle: { color: "#a9dfff" } },
    series: [{ type: "pie", radius: ["42%%", "68%%"], center: ["50%%", "52%%"], data }]
  }
}""" % title
    if kind == "bar":
        series_type = "bar"
        values = "[1200, 1580, 1760, 1390, 1920, 2180]"
    else:
        series_type = "line"
        values = "[12, 18, 15, 22, 19, 23]"
    return """function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [{ name: "周一", value: %s[0] }, { name: "周二", value: %s[1] }, { name: "周三", value: %s[2] }, { name: "周四", value: %s[3] }, { name: "周五", value: %s[4] }, { name: "周六", value: %s[5] }]
  if (_getflag === "data") return data
  if (_setdata) data = _setdata
  return {
    title: { text: "%s", left: "center", top: 8, textStyle: { color: "#dff7ff", fontSize: 16 } },
    grid: { left: 44, right: 24, top: 58, bottom: 34 },
    xAxis: { type: "category", data: data.map(item => item.name), axisLabel: { color: "#9edcff" } },
    yAxis: { type: "value", axisLabel: { color: "#9edcff" }, splitLine: { lineStyle: { color: "rgba(120,220,255,0.16)" } } },
    series: [{ type: "%s", smooth: true, data: data.map(item => item.value), itemStyle: { color: "#31d7ff" }, areaStyle: %s }]
  }
}""" % (values, values, values, values, values, values, title, series_type, "{}" if series_type == "line" else "undefined")


def _summary(title: str, page: dict[str, Any], resources: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            "已生成平台大屏 JSON 包。",
            "",
            "- `generated-bigscreen/page.json`",
            *[f"- `generated-bigscreen/resources/{item['resourceId']}.resource.json`" for item in resources],
            "",
            f"组件数量：{len(page['components'])}",
            f"ECharts 资源数量：{len(resources)}",
            "",
            "平台命令建议：先上传 SVG 背景并回填 `canvas.backgroundImage.fileId`，再保存资源，最后创建大屏项目。",
        ]
    )


def _platform_commands(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"serviceId": "fileService", "commandId": "UploadFile", "required": False},
        *[
            {
                "serviceId": "visualizationService:resource",
                "commandId": "Add",
                "resourceId": item["resourceId"],
                "required": False,
            }
            for item in resources
        ],
        {"serviceId": "visualizationService:project", "commandId": "CreateBigScreenProject", "required": False},
    ]


def _json_reference(relative: str) -> dict[str, Any]:
    return deepcopy(json.loads((ROOT / "references" / relative).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
