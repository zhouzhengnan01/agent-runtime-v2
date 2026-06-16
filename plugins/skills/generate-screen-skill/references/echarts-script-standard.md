# ECharts Script Standard

This reference defines how the skill must generate ECharts scripts for the platform `resourceComponentEcharts` custom chart component.

The ECharts script belongs to the ECharts resource entity JSON.

The script must be written to:

```text
resource entity JSON -> configuration.javaScript
```

The page component must only reference the generated resource by:

```text
page component JSON -> componentProps.resource.id
page component JSON -> componentProps.resource.version
```

Do not write ECharts script code directly into `page.json.components`.

---

## 1. Resource Entity Relationship

Every generated `custom-chart` component must have two related JSON objects:

```text
1. ECharts resource entity JSON
2. page component JSON
```

The resource entity JSON must contain the script:

```text
configuration.javaScript
```

The page component JSON must reference the resource entity:

```text
componentProps.resource.id
componentProps.resource.version
```

The same generated resource id must be used in both places:

```text
resource entity JSON -> resourceId
page component JSON  -> componentProps.resource.id
```

Resource version must always be:

```text
0
```

---

## 2. Required Platform Function Signature

Every generated ECharts script must expose exactly this function:

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
}
```

Do not use other entry function names.

Do not use:

```text
createOption
render
main
init
mounted
setup
```

The platform provides these parameters:

```text
_getflag       Used by the platform to request the default data binding schema.
_setdata       Incoming mapped data passed by the platform.
echarts        ECharts runtime object provided by the platform.
chartInstance  Existing chart instance provided by the platform.
```

The script must return an ECharts `option` object.

The script must not manually initialize ECharts.

---

## 3. Required Data Binding Format

The script must define a default `data` array.

Each item in `data` represents one data source field binding object.

Required shape:

```js
let data = [
  {
    label: '字段名称',
    value: []
  }
]
```

Field meaning:

```text
label = display field name
value = bound field value
```

Example for line chart:

```js
let data = [
  {
    label: '时间',
    value: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
  },
  {
    label: '用电量',
    value: [120, 180, 260, 420, 390, 310]
  }
]
```

Example for pie chart:

```js
let data = [
  {
    label: '类型',
    value: ['照明', '空调', '动力', '办公']
  },
  {
    label: '占比',
    value: [28, 36, 22, 14]
  }
]
```

Example for gauge:

```js
let data = [
  {
    label: '指标',
    value: '设备在线率'
  },
  {
    label: '数值',
    value: 96.8
  }
]
```

---

## 4. Required Data Return Rule

The script must support platform data schema retrieval.

Required code:

```js
if (_getflag === 'data') return data
```

This must appear before the chart option is created.

---

## 5. Required Data Injection Rule

The script must support incoming mapped data.

Required code:

```js
if (_setdata) {
  data = _setdata
}
```

This allows the platform to replace default mock data with mapped platform data.

---

## 6. Required Return Rule

The script must return a valid ECharts option.

Required code:

```js
return option
```

Do not call:

```text
chartInstance.setOption(option)
echarts.init(...)
```

The platform is responsible for rendering the returned option.

---

## 7. Forbidden Code

Generated scripts must not include:

```text
document.getElementById
document.querySelector
window.fetch
XMLHttpRequest
axios
external CDN imports
import statements
require statements
unsafe eval
new Function
setInterval
setTimeout for polling
manual echarts.init
manual chart disposal
platform command calls
hard-coded DOM container id
```

The script must only build and return the option.

---

## 8. Common Visual Rules

All generated chart options should follow these visual rules:

```text
backgroundColor: transparent
dark big-screen friendly colors
readable axis labels
subtle split lines
compact grid for small panels
tooltip enabled
legend enabled only when useful
avoid excessive labels in small panels
avoid huge font sizes inside side panels
avoid white chart background
```

Recommended colors:

```text
primary cyan      #1FEAFF
secondary blue    #2B6CFF
success green     #4DFFB8
warning orange    #FFB84D
danger red        #FF5C7A
text primary      rgba(240,250,255,0.92)
text secondary    rgba(210,238,255,0.72)
axis line         rgba(180,220,255,0.35)
split line        rgba(120,200,255,0.12)
```

---

## 9. Chart Type Mapping

When generating `configuration.javaScript`, map business intent to chart type.

| Business Intent | Chart Type | Resource Group Suffix |
|---|---|---|
| 趋势 | `line` | `line-chart` |
| 分时趋势 | `area` | `line-chart` |
| 分类对比 | `bar` | `bar-chart` |
| 排名对比 | `bar` | `bar-chart` |
| 占比 | `pie` | `pie-chart` |
| 分布 | `pie` | `pie-chart` |
| 在线率 | `gauge` | `gauge` |
| 利用率 | `gauge` | `gauge` |
| 地图/园区主视觉 | `custom-map` | `map` |

The resource entity field `group` must be a JSON string array.

Example:

```json
{
  "group": "[\"vis_oneself_dimension_line-chart\"]"
}
```

---

## 10. Line Chart Script Template

Use this template for line trend charts.

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [
    {
      label: '时间',
      value: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
    },
    {
      label: '数据',
      value: [120, 180, 260, 420, 390, 310]
    }
  ]

  if (_getflag === 'data') return data

  if (_setdata) {
    data = _setdata
  }

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis'
    },
    grid: {
      left: 36,
      right: 18,
      top: 28,
      bottom: 28
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.isArray(data[0]?.value) ? data[0].value : [],
      axisLine: {
        lineStyle: {
          color: 'rgba(180,220,255,0.35)'
        }
      },
      axisTick: {
        show: false
      },
      axisLabel: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    yAxis: {
      type: 'value',
      splitLine: {
        lineStyle: {
          color: 'rgba(120,200,255,0.12)'
        }
      },
      axisLabel: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    series: [
      {
        name: data[1]?.label || '数据',
        data: Array.isArray(data[1]?.value) ? data[1].value : [],
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: {
          width: 3,
          color: '#1FEAFF'
        },
        itemStyle: {
          color: '#1FEAFF'
        },
        areaStyle: {
          opacity: 0.18,
          color: '#1FEAFF'
        }
      }
    ]
  }

  return option
}
```

---

## 11. Bar Chart Script Template

Use this template for category comparison and ranking charts.

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [
    {
      label: '分类',
      value: ['一区', '二区', '三区', '四区', '五区']
    },
    {
      label: '数据',
      value: [320, 452, 301, 534, 390]
    }
  ]

  if (_getflag === 'data') return data

  if (_setdata) {
    data = _setdata
  }

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow'
      }
    },
    grid: {
      left: 36,
      right: 18,
      top: 26,
      bottom: 30
    },
    xAxis: {
      type: 'category',
      data: Array.isArray(data[0]?.value) ? data[0].value : [],
      axisLine: {
        lineStyle: {
          color: 'rgba(180,220,255,0.35)'
        }
      },
      axisTick: {
        show: false
      },
      axisLabel: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    yAxis: {
      type: 'value',
      splitLine: {
        lineStyle: {
          color: 'rgba(120,200,255,0.12)'
        }
      },
      axisLabel: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    series: [
      {
        name: data[1]?.label || '数据',
        data: Array.isArray(data[1]?.value) ? data[1].value : [],
        type: 'bar',
        barWidth: 16,
        itemStyle: {
          color: '#1FEAFF',
          borderRadius: [6, 6, 0, 0]
        }
      }
    ]
  }

  return option
}
```

---

## 12. Area Chart Script Template

Use this template for load, energy, traffic, or event flow trends.

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [
    {
      label: '时间',
      value: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
    },
    {
      label: '负载',
      value: [42, 58, 76, 88, 73, 64]
    }
  ]

  if (_getflag === 'data') return data

  if (_setdata) {
    data = _setdata
  }

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis'
    },
    grid: {
      left: 36,
      right: 18,
      top: 28,
      bottom: 28
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: Array.isArray(data[0]?.value) ? data[0].value : [],
      axisLine: {
        lineStyle: {
          color: 'rgba(180,220,255,0.35)'
        }
      },
      axisTick: {
        show: false
      },
      axisLabel: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    yAxis: {
      type: 'value',
      splitLine: {
        lineStyle: {
          color: 'rgba(120,200,255,0.12)'
        }
      },
      axisLabel: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    series: [
      {
        name: data[1]?.label || '数据',
        data: Array.isArray(data[1]?.value) ? data[1].value : [],
        type: 'line',
        smooth: true,
        symbol: 'none',
        lineStyle: {
          width: 2,
          color: '#2B6CFF'
        },
        areaStyle: {
          opacity: 0.28,
          color: '#2B6CFF'
        }
      }
    ]
  }

  return option
}
```

---

## 13. Pie Chart Script Template

Use this template for proportion and distribution charts.

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [
    {
      label: '类型',
      value: ['照明', '空调', '动力', '办公']
    },
    {
      label: '数值',
      value: [28, 36, 22, 14]
    }
  ]

  if (_getflag === 'data') return data

  if (_setdata) {
    data = _setdata
  }

  const names = Array.isArray(data[0]?.value) ? data[0].value : []
  const values = Array.isArray(data[1]?.value) ? data[1].value : []

  const pieData = names.map((name, index) => ({
    name,
    value: values[index] || 0
  }))

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item'
    },
    legend: {
      bottom: 0,
      textStyle: {
        color: 'rgba(220,245,255,0.72)',
        fontSize: 11
      }
    },
    series: [
      {
        name: data[1]?.label || '数值',
        type: 'pie',
        radius: ['45%', '68%'],
        center: ['50%', '44%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 6,
          borderColor: 'rgba(6,17,31,0.8)',
          borderWidth: 2
        },
        label: {
          color: 'rgba(240,250,255,0.88)',
          fontSize: 11
        },
        labelLine: {
          lineStyle: {
            color: 'rgba(180,220,255,0.35)'
          }
        },
        data: pieData
      }
    ]
  }

  return option
}
```

---

## 14. Gauge Script Template

Use this template for rate, utilization, score, and online rate.

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [
    {
      label: '指标',
      value: '设备在线率'
    },
    {
      label: '数值',
      value: 96.8
    }
  ]

  if (_getflag === 'data') return data

  if (_setdata) {
    data = _setdata
  }

  const title = data[0]?.value || data[0]?.label || '指标'
  const value = Number(data[1]?.value || 0)

  const option = {
    backgroundColor: 'transparent',
    series: [
      {
        type: 'gauge',
        min: 0,
        max: 100,
        radius: '86%',
        center: ['50%', '55%'],
        progress: {
          show: true,
          width: 10,
          itemStyle: {
            color: '#4DFFB8'
          }
        },
        axisLine: {
          lineStyle: {
            width: 10,
            color: [[1, 'rgba(120,200,255,0.16)']]
          }
        },
        axisTick: {
          show: false
        },
        splitLine: {
          show: false
        },
        axisLabel: {
          show: false
        },
        pointer: {
          show: false
        },
        detail: {
          valueAnimation: true,
          formatter: '{value}%',
          color: 'rgba(240,250,255,0.95)',
          fontSize: 24,
          offsetCenter: [0, '8%']
        },
        title: {
          show: true,
          offsetCenter: [0, '42%'],
          color: 'rgba(210,238,255,0.72)',
          fontSize: 12
        },
        data: [
          {
            value,
            name: title
          }
        ]
      }
    ]
  }

  return option
}
```

---

## 15. Custom Map Visual Script Template

Use this template for center main visual when there is no real map engine yet.

This template uses ECharts `scatter` and `lines` to simulate a park, city, or device distribution visual.

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
  let data = [
    {
      label: '点位',
      value: [
        { name: 'A区', value: [20, 60, 128] },
        { name: 'B区', value: [42, 38, 96] },
        { name: 'C区', value: [66, 72, 156] },
        { name: 'D区', value: [78, 44, 88] }
      ]
    },
    {
      label: '流向',
      value: [
        { coords: [[20, 60], [42, 38]] },
        { coords: [[42, 38], [66, 72]] },
        { coords: [[66, 72], [78, 44]] }
      ]
    }
  ]

  if (_getflag === 'data') return data

  if (_setdata) {
    data = _setdata
  }

  const points = Array.isArray(data[0]?.value) ? data[0].value : []
  const flows = Array.isArray(data[1]?.value) ? data[1].value : []

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item'
    },
    xAxis: {
      min: 0,
      max: 100,
      show: false
    },
    yAxis: {
      min: 0,
      max: 100,
      show: false
    },
    grid: {
      left: 20,
      right: 20,
      top: 20,
      bottom: 20
    },
    series: [
      {
        name: '流向',
        type: 'lines',
        coordinateSystem: 'cartesian2d',
        zlevel: 1,
        effect: {
          show: true,
          period: 4,
          trailLength: 0.28,
          symbolSize: 5
        },
        lineStyle: {
          color: '#1FEAFF',
          width: 1,
          opacity: 0.45,
          curveness: 0.25
        },
        data: flows
      },
      {
        name: '点位',
        type: 'scatter',
        coordinateSystem: 'cartesian2d',
        zlevel: 2,
        symbolSize: function (val) {
          return Math.max(10, Math.min(28, Number(val[2] || 80) / 6))
        },
        itemStyle: {
          color: '#1FEAFF',
          shadowBlur: 18,
          shadowColor: 'rgba(31,234,255,0.65)'
        },
        label: {
          show: true,
          formatter: '{b}',
          color: 'rgba(240,250,255,0.86)',
          fontSize: 12,
          position: 'right'
        },
        data: points
      }
    ]
  }

  return option
}
```

---

## 16. Script Selection Rules

When generating scripts, choose templates by chart role:

```text
line chart      -> Line Chart Script Template
bar chart       -> Bar Chart Script Template
area chart      -> Area Chart Script Template
pie chart       -> Pie Chart Script Template
gauge           -> Gauge Script Template
main visual map -> Custom Map Visual Script Template
```

For unknown chart intent:

```text
trend-like words      -> line
comparison-like words -> bar
distribution words    -> pie
rate or percentage    -> gauge
main visual or map    -> custom-map
```

---

## 17. Resource Entity JavaScript String Rules

When writing the script into `configuration.javaScript`, it must be stored as a valid JSON string.

Rules:

- Escape newlines as `\n` if writing inline JSON.
- Escape double quotes inside the script if needed.
- Keep the function body readable.
- Do not wrap the script in markdown code fences.
- Do not include extra explanation before or after the script.
- The value of `configuration.javaScript` must be the JavaScript source string only.

Correct:

```json
{
  "configuration": {
    "componentType": "echarts",
    "javaScript": "function defaultFunc(_getflag, _setdata, echarts, chartInstance) {\n  let data = []\n  if (_getflag === 'data') return data\n  if (_setdata) {\n    data = _setdata\n  }\n  const option = {}\n  return option\n}"
  }
}
```

Incorrect:

```json
{
  "configuration": {
    "componentType": "echarts",
    "javaScript": "```js\nfunction defaultFunc() {}\n```"
  }
}
```

---

## 18. Validation Requirements

A generated ECharts resource script is valid only when all conditions are true:

```text
resource entity has resourceId
resource entity version is 0
resource entity provider is local
resource entity type is component
resource entity configuration.componentType is echarts
resource entity configuration.javaScript exists
configuration.javaScript contains function defaultFunc(_getflag, _setdata, echarts, chartInstance)
configuration.javaScript contains if (_getflag === 'data') return data
configuration.javaScript accepts _setdata
configuration.javaScript returns option
configuration.javaScript does not call echarts.init
configuration.javaScript does not call chartInstance.setOption
configuration.javaScript does not query DOM
configuration.javaScript does not import external libraries
page component does not contain script, option, echartsOption, code, or javaScript fields
page component componentProps.resource.id equals resource entity resourceId
page component componentProps.resource.version is 0
```

---

## 19. Most Important Rule

The ECharts script belongs to the resource entity JSON:

```text
generated-bigscreen/resources/{resourceId}.resource.json
  -> configuration.javaScript
```

The page component only references that resource:

```text
generated-bigscreen/page.json
  -> components[n].componentProps.resource.id
```

Never embed the ECharts script directly into the page component.