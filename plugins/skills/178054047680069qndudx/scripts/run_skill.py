#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATA = [
    {"name": "一月", "value": 120},
    {"name": "二月", "value": 180},
    {"name": "三月", "value": 150},
    {"name": "四月", "value": 220},
]


def main() -> int:
    payload = read_payload()
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else payload
    outputs_dir = Path(str(payload.get("outputs_dir") or os.environ.get("OUTPUTS_DIR") or ".")).resolve()

    objective = string_value(spec.get("objective") or spec.get("message") or spec.get("prompt"))
    display_name = string_value(
        spec.get("display_name")
        or spec.get("displayName")
        or spec.get("component_name")
        or spec.get("componentName")
        or spec.get("name")
    )
    if not display_name:
        display_name = title_from_objective(objective) or "AI 可视化组件"

    resource_id = normalize_resource_id(
        spec.get("resource_id")
        or spec.get("resourceId")
        or spec.get("component_id")
        or spec.get("componentId")
        or display_name
    )
    chart_type = normalize_chart_type(spec.get("chart_type") or spec.get("chartType") or spec.get("type") or objective)
    width = bounded_int(spec.get("width"), 400, 120, 2000)
    height = bounded_int(spec.get("height"), 270, 120, 1600)
    default_data = normalize_default_data(spec.get("default_data") or spec.get("defaultData") or spec.get("data"))

    component_dir = outputs_dir / resource_id
    component_dir.mkdir(parents=True, exist_ok=True)
    (component_dir / "component.vue").write_text(
        component_vue(resource_id, chart_type),
        encoding="utf-8",
    )
    (component_dir / "config.mjs").write_text(
        config_mjs(resource_id, display_name, chart_type, width, height, default_data),
        encoding="utf-8",
    )
    (component_dir / "README.md").write_text(
        readme(resource_id, display_name, chart_type, objective),
        encoding="utf-8",
    )

    result = {
        "resource_id": resource_id,
        "display_name": display_name,
        "chart_type": chart_type,
        "component_dir": str(component_dir),
        "artifacts": [
            {"path": f"{resource_id}/component.vue"},
            {"path": f"{resource_id}/config.mjs"},
            {"path": f"{resource_id}/README.md"},
        ],
        "data": {
            "resource_id": resource_id,
            "display_name": display_name,
            "chart_type": chart_type,
            "component_dir": str(component_dir),
        },
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def read_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raw = os.environ.get("SPEC_JSON", "{}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON input: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("Skill input must be a JSON object.")
    return value


def string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def title_from_objective(objective: str) -> str:
    text = re.sub(r"\s+", " ", objective).strip()
    if not text:
        return ""
    return text[:24]


def normalize_resource_id(value: Any) -> str:
    raw = string_value(value) or "ai_component_v1"
    ascii_text = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    if not ascii_text or ascii_text.lower() in {"ai", "v1"}:
        ascii_text = "component"
    if not ascii_text.lower().startswith("ai_"):
        ascii_text = f"ai_{ascii_text}"
    if not re.search(r"_v\d+$", ascii_text, re.IGNORECASE):
        ascii_text = f"{ascii_text}_v1"
    if not re.match(r"^[A-Za-z_]", ascii_text):
        ascii_text = f"ai_{ascii_text}"
    return ascii_text


def normalize_chart_type(value: Any) -> str:
    text = string_value(value).lower()
    if any(key in text for key in ("pie", "饼")):
        return "pie"
    if any(key in text for key in ("line", "折线", "趋势")):
        return "line"
    return "bar"


def bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def normalize_default_data(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return DEFAULT_DATA
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            name = item.get("name") or item.get("label") or item.get("category") or f"项{index + 1}"
            amount = item.get("value") if item.get("value") is not None else item.get("amount")
        else:
            name = f"项{index + 1}"
            amount = item
        try:
            numeric = float(amount)
        except (TypeError, ValueError):
            numeric = 0
        rows.append({"name": str(name), "value": numeric})
    return rows or DEFAULT_DATA


def component_name(resource_id: str) -> str:
    parts = [part for part in resource_id.split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def chart_import(chart_type: str) -> tuple[str, str]:
    if chart_type == "line":
        return "LineChart", "line"
    if chart_type == "pie":
        return "PieChart", "pie"
    return "BarChart", "bar"


def component_vue(resource_id: str, chart_type: str) -> str:
    chart_class, series_type = chart_import(chart_type)
    axis_block = ""
    series_block = ""
    if chart_type == "pie":
        series_block = """        series: [
          {
            type: 'pie',
            radius: ['42%', '68%'],
            center: ['50%', '52%'],
            avoidLabelOverlap: true,
            label: { color: '#fff' },
            data: rows
          }
        ]"""
    else:
        axis_block = """        grid: { top: 36, right: 20, bottom: 32, left: 42 },
        xAxis: {
          type: 'category',
          data: rows.map(item => item.name),
          axisLabel: { color: '#b8c7e0' },
          axisLine: { lineStyle: { color: 'rgba(255,255,255,0.28)' } }
        },
        yAxis: {
          type: 'value',
          axisLabel: { color: '#b8c7e0' },
          splitLine: { lineStyle: { color: 'rgba(255,255,255,0.12)' } }
        },"""
        series_extra = ""
        if chart_type == "line":
            series_extra = "            smooth: true,\n            areaStyle: {},\n"
        elif chart_type == "bar":
            series_extra = "            barWidth: 18,\n"
        series_block = f"""        series: [
          {{
            type: '{series_type}',
{series_extra}            showSymbol: false,
            itemStyle: {{
              color: '#36d6ff'
            }},
            data: rows.map(item => item.value)
          }}
        ]"""
    return f"""<template>
  <div class="component-box" :style="backgroundStyle">
    <VChart class="chart" :option="option" autoresize />
  </div>
</template>

<script>
import {{ ref, computed, watch }} from 'vue'
import {{ use }} from 'echarts/core'
import {{ CanvasRenderer }} from 'echarts/renderers'
import {{ TooltipComponent, LegendComponent, GridComponent }} from 'echarts/components'
import {{ {chart_class} }} from 'echarts/charts'
import VChart from 'vue-echarts'
import {{ setComponentBackground }} from '@visualization-resources/packages/component/utils'

use([CanvasRenderer, {chart_class}, TooltipComponent, LegendComponent, GridComponent])

export default {{
  name: '{component_name(resource_id)}',
  components: {{ VChart }},
  props: {{
    info: {{ type: Object, default: () => ({{}}) }},
    isEdit: Boolean
  }},
  setup(props) {{
    const option = ref({{}})
    const backgroundStyle = computed(() => setComponentBackground(props.info?.componentProps?.background))

    const normalizeRows = value => {{
      const rows = Array.isArray(value) ? value : []
      return rows.map((item, index) => {{
        const name = item?.name || item?.label || item?.category || `项${{index + 1}}`
        const rawValue = item?.value ?? item?.amount ?? 0
        const numericValue = Number(rawValue)
        return {{ name, value: Number.isFinite(numericValue) ? numericValue : 0 }}
      }})
    }}

    const rebuildOption = () => {{
      const source = props.info?.dataSourceProps?.defaultValue || []
      const rows = normalizeRows(source)
      option.value = {{
        backgroundColor: 'transparent',
        color: ['#36d6ff', '#52ffaa', '#ffd166', '#ff7a90'],
        tooltip: {{ trigger: '{'item' if chart_type == 'pie' else 'axis'}' }},
        legend: {{
          show: true,
          top: 4,
          right: 8,
          textStyle: {{ color: '#d8e6ff' }}
        }},
{axis_block}
{series_block}
      }}
    }}

    watch(() => props.info, rebuildOption, {{ deep: true, immediate: true }})

    return {{ backgroundStyle, option }}
  }}
}}
</script>

<style scoped lang="less">
.component-box {{
  width: 100%;
  height: 100%;
  overflow: hidden;
}}

.chart {{
  width: 100%;
  height: 100%;
}}
</style>
"""


def config_mjs(
    resource_id: str,
    display_name: str,
    chart_type: str,
    width: int,
    height: int,
    default_data: list[dict[str, Any]],
) -> str:
    default_data_json = json.dumps(default_data, ensure_ascii=False, indent=2)
    chart_config_entry = ""
    if chart_type == "bar":
        chart_config_entry = ",\n    basicBar: {}"
    elif chart_type == "line":
        chart_config_entry = ",\n    lineBar: {}"
    return f"""export const {resource_id}ConfigProps = {{
  name: '{js_string(display_name)}',
  type: '{resource_id}',
  componentProps: {{
    style: {{ x: 0, y: 0, width: {width}, height: {height} }},
    background: {{
      type: 'color',
      color: 'rgba(5, 18, 38, 0.82)'
    }},
    legend: {{
      show: true
    }},
    theme: {{}}{chart_config_entry}
  }},
  dataSourceProps: {{
    mode: 'static',
    type: 'array',
    defaultValue: {indent(default_data_json, 4)}
  }},
  animationProps: []
}}

export const {resource_id}Config = []
"""


def readme(resource_id: str, display_name: str, chart_type: str, objective: str) -> str:
    return f"""# {display_name}

- resourceId: `{resource_id}`
- chartType: `{chart_type}`
- objective: {objective or '未提供'}

生成文件满足 JetLinks AI 组件基础格式:

- `component.vue` 使用 Options API,没有 `<script setup>`。
- `config.mjs` 导出 `{resource_id}ConfigProps` 和 `{resource_id}Config`。
- `ConfigProps.type` 等于 `{resource_id}`。
- `componentProps.style` 包含 `x/y/width/height`。
"""


def indent(text: str, spaces: int) -> str:
    padding = " " * spaces
    return "\n".join((padding + line) if index else line for index, line in enumerate(text.splitlines()))


def js_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


if __name__ == "__main__":
    raise SystemExit(main())
