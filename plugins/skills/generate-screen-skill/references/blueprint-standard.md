# Standard Bigscreen Blueprint

This reference defines the standard blueprint for generating one visualization big-screen page.

The generated result is not a frontend code project.

The base generated result must include:

```text
generated-bigscreen/
  page.json
```

When the generated page contains `custom-chart` components, the generated result must also include matching ECharts resource entity JSON files:

```text
generated-bigscreen/
  resources/
    {resourceId}.resource.json
```

Do not generate:

```text
preview.html
screen-blueprint.json
README.md
extra source code files
frontend source code
React code
Vue code
standalone demo app
```

The layout blueprint in this document is the source of truth.

Both background SVG content and `page.json` component placement must be generated from the same fixed layout regions.

The background SVG content must be uploaded through `fileService / UploadFile`, and the returned file id must be written to `page.canvas.backgroundImage.fileId`.

For `custom-chart`, the page component must reference a generated resource entity by `componentProps.resource.id`.

---

## 1. Fixed Blueprint

The skill currently supports exactly one blueprint:

```text
standard-1920x1080-v1
```

Hard rules:

- Canvas size must always be `1920 x 1080`.
- SVG must always use `viewBox="0 0 1920 1080"`.
- Component positions must use the same absolute `1920 x 1080` coordinate system.
- Real platform component position and size must be written to `componentProps.style`.
- Components must be placed inside region `contentBox`.
- Components should prefer using slots when a suitable slot exists.
- Do not place components directly on panel borders.
- Do not place components inside the panel title decoration area.
- Do not generate a decorative SVG first and then guess component positions.
- Do not change region coordinates unless the user explicitly asks to redesign the skill blueprint itself.
- Do not generate `preview.html`.
- Do not generate `screen-blueprint.json`.
- Do not generate frontend source code.
- Do not invent unavailable platform APIs or commands.

---

## 2. Output Files

For each generated big screen, create this directory:

```text
generated-bigscreen/
  page.json
```

When the page contains one or more `custom-chart` components, also create:

```text
generated-bigscreen/
  resources/
    {resourceId}.resource.json
```

File responsibilities:

```text
uploaded background SVG content  = decorative and structural SVG background
page.json                        = platform-compatible canvas JSON and component JSON
resources/*.resource.json         = ECharts resource entity JSON for custom-chart components
```

The SVG background content must be generated from the fixed regions in this document, uploaded through the file command, and referenced by `canvas.backgroundImage.fileId`.

The `page.json` must place platform components inside the corresponding region `contentBox`.

Every `custom-chart` page component must have a matching resource entity JSON file.

---

## 3. Platform Page JSON Structure

The generated `page.json` must use this exact root structure:

```json
{
  "canvas": {
    "width": 1920,
    "height": 1080,
    "scale": 0.64,
    "name": "画布",
    "sizeKey": "pc",
    "adaptationType": "AUTO",
    "backgroundColor": "#424242",
    "backgroundImage": {
      "fileId": ""
    },
    "gridLayout": {
      "backgroundColor": "",
      "marginHorizontal": 8,
      "marginVertical": 8,
      "borderColor": "",
      "borderWidth": 1,
      "borderStyle": "solid",
      "fontColor": "rgba(0,0,0,1)"
    },
    "enablePreviewZoom": false,
    "filter": {
      "hue": 0,
      "saturation": 0,
      "brightness": 0,
      "contrast": 0,
      "opacity": 100,
      "grayscale": 0
    }
  },
  "components": []
}
```

Rules:

- Do not wrap the result with a `page` object.
- Do not rename `canvas`.
- Do not rename `components`.
- Do not change the canvas field names.
- `canvas.width` must be `1920`.
- `canvas.height` must be `1080`.
- `canvas.scale` must be `0.64`.
- `canvas.sizeKey` must be `pc`.
- `canvas.adaptationType` must be `AUTO`.
- `canvas.backgroundImage.fileId` must be the `id` returned by `fileService / UploadFile`.
- Do not leave `canvas.backgroundImage.fileId` empty after successful background upload.
- Do not add non-platform fields into `canvas`.
- Do not write layout debug fields into `page.json`.
- Do not write `regionId`, `slotId`, `position`, or `size` into components unless the real platform component template already contains those fields.

---

## 4. Background SVG Upload Usage

The generated background SVG content is the visual background asset for the big screen.

Upload the SVG content through:

```text
fileService / UploadFile
```

Then write the returned file id to:

```json
{
  "backgroundImage": {
    "fileId": "uploaded-svg-file-id"
  }
}
```

Do not keep `background.svg` as a required output file.

The skill must generate SVG content, Base64 encode the SVG UTF-8 bytes, call the upload command, and write the returned `id` into `canvas.backgroundImage.fileId`.

The background SVG is only the decorative and structural layer.

It must not contain real business values, chart data, ranking rows, alarm text, interactive buttons, or table content.

---

## 5. Real Platform Component Files

The skill must use real platform component JSON templates.

Place component JSON files here:

```text
.codex/skills/generate-screen-skill/references/components/
```

Required component template files:

```text
references/components/text.json
references/components/table.json
references/components/dateTime.json
references/components/tabs.json
references/components/custom-chart.json
```

Place resource entity template files here:

```text
.codex/skills/generate-screen-skill/references/resources/
```

Required resource template file:

```text
references/resources/echarts-resource.json
```

Create a component registry file here:

```text
.codex/skills/generate-screen-skill/references/component-registry.json
```

Create a command rule file here:

```text
.codex/skills/generate-screen-skill/references/command-standard.md
```

The registry must define these five component keys:

```text
text
table
dateTime
tabs
custom-chart
```

Rules:

- The component JSON files must come from the real platform component output.
- Do not invent new component JSON structures when real component templates exist.
- When generating a component, copy the closest matching real component JSON template.
- Then patch only safe fields that already exist in the platform template.
- Do not remove unknown platform fields from component JSON.
- Do not rename platform-specific fields.
- Do not simplify real platform components into fake logical components.
- Keep the original component JSON structure as much as possible.
- For position and size, patch `componentProps.style.x`, `componentProps.style.y`, `componentProps.style.width`, and `componentProps.style.height`.

---

## 6. Allowed Platform Components

Only these five component categories are allowed in the current skill stage:

```text
文本框
表格组件
时间框
选项卡
图表类自定义组件
```

Mapped keys:

```text
text
table
dateTime
tabs
custom-chart
```

Do not generate old test component types such as:

```text
number-card
bar-chart
line-chart
area-chart
pie-chart
rank-list
progress
map-panel
status-list
alert-list
gauge
```

Instead, map business intent to the five real platform component categories.

Mapping rules:

| Business Need | Platform Component Key | Platform Component |
|---|---|---|
| 大屏标题 | `text` | 文本框 |
| 副标题 | `text` | 文本框 |
| 指标数字 | `text` | 文本框 |
| 当前时间 | `dateTime` | 时间框 |
| 排行榜 | `table` | 表格组件 |
| 告警列表 | `table` | 表格组件 |
| 事件列表 | `table` | 表格组件 |
| 设备状态列表 | `table` | 表格组件 |
| 折线图 | `custom-chart` | 图表类自定义组件 |
| 柱状图 | `custom-chart` | 图表类自定义组件 |
| 饼图 | `custom-chart` | 图表类自定义组件 |
| 面积图 | `custom-chart` | 图表类自定义组件 |
| 仪表盘 | `custom-chart` | 图表类自定义组件 |
| 地图/园区主视觉 | `custom-chart` | 图表类自定义组件 |
| 多视图切换 | `tabs` | 选项卡 |

---

## 7. Custom ECharts Resource Component Rules

The `custom-chart` component is a special platform component.

It is not only a page component.

It must generate two related JSON objects:

```text
1. ECharts resource entity JSON
2. page component JSON
```

The resource entity contains the ECharts rendering script.

The page component is the actual component instance placed on the canvas.

The page component must reference the generated resource entity by:

```text
componentProps.resource.id
componentProps.resource.version
```

The resource entity JSON must use the platform resource structure:

```text
resourceId
name
version
thumbnailUrl
provider
type
group
configuration.componentType
configuration.javaScript
```

The page component must use the platform component type:

```text
resourceComponentEcharts
```

The resource id must be generated before any save command.

Do not wait for the resource save command response to decide the resource id.

The same generated resource id must be written to both places:

```text
resource entity JSON -> resourceId
page component JSON  -> componentProps.resource.id
```

The resource version is always:

```text
0
```

The page component resource version is always:

```text
0
```

Generation flow:

```text
generate random resource id
  -> generate ECharts resource entity JSON using this resource id
  -> write resource entity JSON to generated-bigscreen/resources/{resourceId}.resource.json
  -> call visualizationService:resource / Add with the resource JSON in data[0]
  -> clone custom-chart component template
  -> patch componentProps.resource.id with the same generated resource id
  -> patch componentProps.resource.version to 0
  -> patch componentProps.style.x/y/width/height
  -> add component to page.json.components
```

Resource id generation rule:

```text
resource id = component_{timestamp}_{random4}
```

Random id requirements:

```text
prefix: component_
timestamp: current millisecond timestamp or stable timestamp-like value
random length: 4
allowed characters: 0-9
example: component_1779881457212_4829
```

Example local output:

```text
generated-bigscreen/
  page.json
  resources/
    component_1779881457212_4829.resource.json
    component_1779881457333_1094.resource.json
```

Example page component resource reference:

```json
{
  "componentProps": {
    "resource": {
      "id": "component_1779881457212_4829",
      "version": 0
    }
  }
}
```

Rules:

- Do not treat `custom-chart` as a normal static chart component.
- Do not write the ECharts script directly into `page.json.components`.
- Do not invent `option`, `script`, `code`, `echartsOption`, or `javaScript` fields inside the page component unless the real platform component schema already contains them.
- The ECharts script belongs to the resource entity JSON at `configuration.javaScript`.
- The page component only references the resource by `componentProps.resource.id`.
- Every generated custom chart component must have a matching generated resource entity.
- The component resource id must match the resource entity `resourceId`.
- The component resource version must always be `0`.
- The resource entity version must always be `0`.
- The resource save command must be called after generating the resource entity JSON.
- The page JSON must already contain the generated resource id.
- Do not replace the generated resource id with another value unless a later platform workflow explicitly requires migration.
- Do not use the command result to replace the generated resource id.

---

## 8. Resource Entity JSON Rules

The resource entity JSON template is stored at:

```text
references/resources/echarts-resource.json
```

During generation, every `custom-chart` must produce one resource entity JSON file:

```text
generated-bigscreen/resources/{resourceId}.resource.json
```

Required resource entity structure:

```json
{
  "resourceId": "__RESOURCE_ID__",
  "name": "__RESOURCE_NAME__",
  "version": 0,
  "thumbnailUrl": "",
  "provider": "local",
  "type": "component",
  "group": "[\"vis_oneself_dimension___CHART_GROUP__\"]",
  "configuration": {
    "componentType": "echarts",
    "javaScript": "__ECHARTS_JAVASCRIPT__"
  }
}
```

Patch rules:

```text
resourceId                    = generated random resource id
name                          = readable resource name
version                       = 0
thumbnailUrl                  = keep empty string unless a thumbnail id exists
provider                      = local
type                          = component
group                         = JSON string array, such as ["vis_oneself_dimension_line-chart"]
configuration.componentType   = echarts
configuration.javaScript      = platform-standard ECharts script
```

Chart group mapping:

| Chart Type | Group Suffix | Full Group Value |
|---|---|---|
| line | `line-chart` | `["vis_oneself_dimension_line-chart"]` |
| bar | `bar-chart` | `["vis_oneself_dimension_bar-chart"]` |
| area | `line-chart` | `["vis_oneself_dimension_line-chart"]` |
| pie | `pie-chart` | `["vis_oneself_dimension_pie-chart"]` |
| gauge | `gauge` | `["vis_oneself_dimension_gauge"]` |
| custom-map | `map` | `["vis_oneself_dimension_map"]` |
| map | `map` | `["vis_oneself_dimension_map"]` |

The resource entity JSON must not be embedded into `page.json`.

The page component must only reference the resource id.

---

## 9. Resource Save Command Rules

Every generated `custom-chart` resource entity must be saved through the platform resource command.

Command rules are defined in:

```text
references/command-standard.md
```

Current command:

```text
serviceId = visualizationService:resource
commandId = Add
```

Command parameters:

```json
{
  "data": [
    "__RESOURCE_ENTITY_JSON__"
  ]
}
```

`data[0]` must be the full generated ECharts resource entity JSON.

For every generated resource entity file:

```text
generated-bigscreen/resources/{resourceId}.resource.json
```

The skill must call:

```text
visualizationService:resource / Add
```

Rules:

- The command must be called only for generated `custom-chart` resource entities.
- Do not call this command for `text`, `table`, `dateTime`, `tabs`, background SVG content, or `page.json`.
- The generated `resourceId` must be created before the command call.
- Do not wait for the save command result to decide the resource id.
- Do not replace the generated resource id with a command result value.
- The same generated resource id must be written to `resource entity JSON -> resourceId`.
- The same generated resource id must be written to `page component JSON -> componentProps.resource.id`.
- The version must always be `0`.
- If the command call fails, keep the generated files and report the command failure in the final summary.

Before creating the project, upload the generated background SVG content:

```text
fileService / UploadFile
parameters = {"fileName":"background.svg","contentType":"image/svg+xml;charset=UTF-8","content":"__BASE64_UTF8_SVG__"}
```

Write the returned upload result `id` to `page.canvas.backgroundImage.fileId`.

After `page.json` has been generated, background SVG upload succeeds, `page.canvas.backgroundImage.fileId` is set, and every custom-chart resource command has been attempted, call:

```text
visualizationService:project / CreateBigScreenProject
parameters = {"page": pageJsonRoot, "projectType": "bigScreen"}
```

The `page` parameter must be exactly the generated `generated-bigscreen/page.json` root object.

Do not pass `pageJSON`.

Do not wrap the generated page root inside another nested `page` field.

`projectType` defaults to `bigScreen`; only pass another supported project type when the user explicitly asks for it.

No preview command is available yet.

Do not invent unavailable APIs or commands.

---

## 10. ECharts Script Rules

The ECharts script belongs to the resource entity JSON.

The script must be written to:

```text
configuration.javaScript
```

The page component must only reference the resource by:

```text
componentProps.resource.id
componentProps.resource.version
```

Required function signature:

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
}
```

Do not use other entry function names.

The platform provides:

```text
_getflag
_setdata
echarts
chartInstance
```

The script must build and return an ECharts `option`.

The script must not manually initialize a chart instance.

The default data binding format must use this structure:

```js
let data = [
  {
    label: '时间',
    value: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  },
  {
    label: '数据',
    value: [820, 932, 901, 934, 1290, 1330, 1320]
  }
]
```

Rules:

- `label` represents the display field name.
- `value` represents the bound field value.
- Each item in `data` represents one data source field binding object.
- The script must return the default data configuration when `_getflag === 'data'`.
- The script must accept incoming mapped data through `_setdata`.
- The script must return a valid ECharts option.

Required data return rule:

```js
if (_getflag === 'data') return data
```

Required data injection rule:

```js
if (_setdata) {
  data = _setdata
}
```

Required return rule:

```js
return option
```

Script must include:

```text
transparent background
tooltip
grid when needed
legend when needed
xAxis and yAxis when needed
series
theme colors suitable for dark big-screen SVG background
safe fallback data
```

Script must not include:

```text
document.getElementById
manual DOM query
external CDN imports
network requests
unsafe eval
platform command calls
hard-coded DOM container id
manual echarts.init
chartInstance.setOption
```

Page component must not contain:

```text
script
echartsOption
option
code
javaScript
```

When more detailed chart templates are needed, follow:

```text
references/echarts-script-standard.md
```

---

## 11. Component Generation Strategy

When creating a normal component:

1. Determine the business role.
2. Select one of the five platform component categories.
3. Read the corresponding real component JSON template from `references/components/`.
4. Deep clone the template.
5. Patch `componentProps.style.x`.
6. Patch `componentProps.style.y`.
7. Patch `componentProps.style.width`.
8. Patch `componentProps.style.height`.
9. Patch display content or mock data.
10. Keep all unknown platform fields unchanged.
11. Add the result to `components`.

For `custom-chart`, the generation strategy is extended:

1. Determine chart type and business role.
2. Generate a random `resourceId`.
3. Generate a matching resource entity JSON file.
4. Write the ECharts script into `configuration.javaScript`.
5. Call `visualizationService:resource / Add` with the generated resource entity JSON in `data[0]`.
6. Clone `references/components/custom-chart.json`.
7. Patch `componentProps.resource.id` with the generated `resourceId`.
8. Patch `componentProps.resource.version` to `0`.
9. Patch `componentProps.style`.
10. Add the component to `page.json.components`.

Component placement must follow:

```text
region slot bounds > region contentBox bounds > region bounds
```

Rules:

- Prefer one main component per panel.
- Use multiple text components inside `global_metrics` for KPI values.
- Use custom chart components for chart panels.
- Use table components for ranking, alarms, device lists, and event lists.
- Use time component only in the header time area.
- Use tabs only when a panel needs data category switching.
- Do not place components outside the target region `contentBox`.
- Do not overlap the SVG panel title area.
- Use static mock data during testing.
- Do not generate non-platform `position` and `size` fields.

---

## 12. Fixed Layout Regions

The following fixed regions define the screen layout.

The skill must use these coordinates exactly.

```json
{
  "blueprintVersion": "1.0.0",
  "blueprintType": "standard-1920x1080-v1",
  "screen": {
    "width": 1920,
    "height": 1080,
    "scene": "auto",
    "theme": "tech-blue",
    "coordinateSystem": "absolute",
    "scaleMode": "fit"
  },
  "designTokens": {
    "outerMargin": 32,
    "gutter": 24,
    "panelRadius": 18,
    "panelTitleHeight": 48,
    "panelPadding": {
      "top": 56,
      "right": 24,
      "bottom": 24,
      "left": 24
    },
    "background": {
      "baseColor": "#06111F",
      "gridColor": "rgba(55, 180, 255, 0.12)",
      "primaryGlow": "#1FEAFF",
      "secondaryGlow": "#2B6CFF",
      "panelFill": "rgba(8, 28, 48, 0.72)",
      "panelStroke": "rgba(54, 220, 255, 0.62)",
      "panelTitleColor": "#DDF7FF",
      "textPrimary": "#FFFFFF",
      "textSecondary": "rgba(210, 238, 255, 0.72)",
      "warningColor": "#FFB84D",
      "dangerColor": "#FF5C7A",
      "successColor": "#4DFFB8"
    }
  },
  "layout": {
    "regions": [
      {
        "id": "header",
        "name": "顶部标题区",
        "role": "header",
        "x": 32,
        "y": 24,
        "width": 1856,
        "height": 84,
        "padding": {
          "top": 16,
          "right": 24,
          "bottom": 16,
          "left": 24
        },
        "contentBox": {
          "x": 56,
          "y": 40,
          "width": 1808,
          "height": 52
        },
        "slots": [
          {
            "id": "header_left",
            "name": "左侧辅助信息",
            "x": 56,
            "y": 40,
            "width": 360,
            "height": 52,
            "preferredComponents": ["text"]
          },
          {
            "id": "header_title",
            "name": "主标题",
            "x": 560,
            "y": 40,
            "width": 800,
            "height": 52,
            "preferredComponents": ["text"]
          },
          {
            "id": "header_time",
            "name": "右侧时间",
            "x": 1540,
            "y": 40,
            "width": 324,
            "height": 52,
            "preferredComponents": ["dateTime"]
          }
        ],
        "componentHint": "Place the main title, subtitle, current time, or system status text here."
      },
      {
        "id": "global_metrics",
        "name": "全局指标区",
        "role": "metrics",
        "x": 32,
        "y": 124,
        "width": 1856,
        "height": 112,
        "padding": {
          "top": 24,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 56,
          "y": 148,
          "width": 1808,
          "height": 64
        },
        "slots": [
          {
            "id": "metric_1",
            "name": "指标卡一",
            "x": 56,
            "y": 148,
            "width": 344,
            "height": 64,
            "preferredComponents": ["text"]
          },
          {
            "id": "metric_2",
            "name": "指标卡二",
            "x": 422,
            "y": 148,
            "width": 344,
            "height": 64,
            "preferredComponents": ["text"]
          },
          {
            "id": "metric_3",
            "name": "指标卡三",
            "x": 788,
            "y": 148,
            "width": 344,
            "height": 64,
            "preferredComponents": ["text"]
          },
          {
            "id": "metric_4",
            "name": "指标卡四",
            "x": 1154,
            "y": 148,
            "width": 344,
            "height": 64,
            "preferredComponents": ["text"]
          },
          {
            "id": "metric_5",
            "name": "指标卡五",
            "x": 1520,
            "y": 148,
            "width": 344,
            "height": 64,
            "preferredComponents": ["text"]
          }
        ],
        "componentHint": "Place 4 to 5 KPI metric text boxes here."
      },
      {
        "id": "left_top_panel",
        "name": "左上分析区",
        "role": "analysis",
        "x": 32,
        "y": 260,
        "width": 420,
        "height": 236,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 56,
          "y": 316,
          "width": 372,
          "height": 156
        },
        "slots": [
          {
            "id": "left_top_body",
            "name": "左上主体内容",
            "x": 56,
            "y": 316,
            "width": 372,
            "height": 156,
            "preferredComponents": ["custom-chart", "table", "text"]
          }
        ],
        "componentHint": "Place a compact analysis chart, table, or summary component here."
      },
      {
        "id": "left_middle_panel",
        "name": "左中趋势区",
        "role": "trend",
        "x": 32,
        "y": 520,
        "width": 420,
        "height": 236,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 56,
          "y": 576,
          "width": 372,
          "height": 156
        },
        "slots": [
          {
            "id": "left_middle_body",
            "name": "左中主体内容",
            "x": 56,
            "y": 576,
            "width": 372,
            "height": 156,
            "preferredComponents": ["custom-chart"]
          }
        ],
        "componentHint": "Place a time-series trend chart here."
      },
      {
        "id": "left_bottom_panel",
        "name": "左下排行区",
        "role": "ranking",
        "x": 32,
        "y": 780,
        "width": 420,
        "height": 268,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 56,
          "y": 836,
          "width": 372,
          "height": 188
        },
        "slots": [
          {
            "id": "left_bottom_body",
            "name": "左下主体内容",
            "x": 56,
            "y": 836,
            "width": 372,
            "height": 188,
            "preferredComponents": ["table", "custom-chart"]
          }
        ],
        "componentHint": "Place ranking, top list, or compact table content here."
      },
      {
        "id": "center_main_panel",
        "name": "中心主视觉区",
        "role": "main-visual",
        "x": 476,
        "y": 260,
        "width": 968,
        "height": 548,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 500,
          "y": 316,
          "width": 920,
          "height": 468
        },
        "slots": [
          {
            "id": "center_main_visual",
            "name": "中心主视觉内容",
            "x": 500,
            "y": 316,
            "width": 920,
            "height": 468,
            "preferredComponents": ["custom-chart"]
          }
        ],
        "componentHint": "Place the main map, park visualization, city visualization, topology, or main chart here."
      },
      {
        "id": "center_bottom_panel",
        "name": "中心底部趋势区",
        "role": "bottom-trend",
        "x": 476,
        "y": 832,
        "width": 968,
        "height": 216,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 500,
          "y": 888,
          "width": 920,
          "height": 136
        },
        "slots": [
          {
            "id": "center_bottom_body",
            "name": "中心底部主体内容",
            "x": 500,
            "y": 888,
            "width": 920,
            "height": 136,
            "preferredComponents": ["custom-chart", "table", "tabs"]
          }
        ],
        "componentHint": "Place the main bottom trend, timeline, event flow, or comparison chart here."
      },
      {
        "id": "right_top_panel",
        "name": "右上状态区",
        "role": "status",
        "x": 1468,
        "y": 260,
        "width": 420,
        "height": 236,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 1492,
          "y": 316,
          "width": 372,
          "height": 156
        },
        "slots": [
          {
            "id": "right_top_body",
            "name": "右上主体内容",
            "x": 1492,
            "y": 316,
            "width": 372,
            "height": 156,
            "preferredComponents": ["custom-chart", "table", "text"]
          }
        ],
        "componentHint": "Place device status, online rate, utilization, or status summary here."
      },
      {
        "id": "right_middle_panel",
        "name": "右中告警区",
        "role": "alerts",
        "x": 1468,
        "y": 520,
        "width": 420,
        "height": 236,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 1492,
          "y": 576,
          "width": 372,
          "height": 156
        },
        "slots": [
          {
            "id": "right_middle_body",
            "name": "右中主体内容",
            "x": 1492,
            "y": 576,
            "width": 372,
            "height": 156,
            "preferredComponents": ["table"]
          }
        ],
        "componentHint": "Place alarms, abnormal events, risk list, or warning messages here."
      },
      {
        "id": "right_bottom_panel",
        "name": "右下分布区",
        "role": "distribution",
        "x": 1468,
        "y": 780,
        "width": 420,
        "height": 268,
        "padding": {
          "top": 56,
          "right": 24,
          "bottom": 24,
          "left": 24
        },
        "contentBox": {
          "x": 1492,
          "y": 836,
          "width": 372,
          "height": 188
        },
        "slots": [
          {
            "id": "right_bottom_body",
            "name": "右下主体内容",
            "x": 1492,
            "y": 836,
            "width": 372,
            "height": 188,
            "preferredComponents": ["custom-chart", "table"]
          }
        ],
        "componentHint": "Place distribution, proportion, category comparison, or composition chart here."
      }
    ]
  }
}
```

---

## 13. SVG Background Rules

Background SVG content must be generated from the fixed layout regions.

Do not keep `generated-bigscreen/background.svg` as a required output file.

Required SVG root:

```xml
<svg viewBox="0 0 1920 1080" width="1920" height="1080" xmlns="http://www.w3.org/2000/svg">
</svg>
```

Each region must have a group:

```xml
<g
  id="region-left_top_panel"
  data-region-id="left_top_panel"
  data-role="analysis"
  data-x="32"
  data-y="260"
  data-width="420"
  data-height="236"
  data-content-x="56"
  data-content-y="316"
  data-content-width="372"
  data-content-height="156">
</g>
```

SVG should include:

```text
background base color
subtle grid
soft radial glow
panel rectangles
panel borders
panel corner decorations
section dividers
center visual base decoration
```

SVG must not include:

```text
real metric values
chart bars generated from real data
chart lines generated from real data
ranking rows
alarm text
interactive buttons
business table content
```

SVG is only the decorative and structural background layer.

After generating the SVG content:

```text
1. Encode the SVG UTF-8 bytes as Base64.
2. Call fileService / UploadFile.
3. Read the returned FileInfo.id.
4. Write FileInfo.id to page.canvas.backgroundImage.fileId.
```

The upload command parameters must be:

```json
{
  "fileName": "background.svg",
  "contentType": "image/svg+xml;charset=UTF-8",
  "content": "__BASE64_UTF8_SVG__"
}
```

The `content` parameter must be Base64 of the SVG UTF-8 bytes, not the raw SVG string.

---

## 14. Platform Component Placement Rules

Every generated component must be based on one of the five real platform component templates.

Allowed component template keys:

```text
text
table
dateTime
tabs
custom-chart
```

Real platform component position and size must be written to:

```text
componentProps.style.x
componentProps.style.y
componentProps.style.width
componentProps.style.height
```

Component placement formula:

```text
componentProps.style.x >= contentBox.x
componentProps.style.y >= contentBox.y
componentProps.style.x + componentProps.style.width <= contentBox.x + contentBox.width
componentProps.style.y + componentProps.style.height <= contentBox.y + contentBox.height
```

Slot placement formula:

```text
componentProps.style.x >= slot.x
componentProps.style.y >= slot.y
componentProps.style.x + componentProps.style.width <= slot.x + slot.width
componentProps.style.y + componentProps.style.height <= slot.y + slot.height
```

Rules:

- Use `header_title` slot for the main title text component.
- Use `header_time` slot for the time component.
- Use `global_metrics` slots for KPI text components.
- Use `center_main_panel` for the main visual custom chart.
- Use side panels for charts, rankings, status, and alerts.
- Use table component for ranking and alert list.
- Use custom chart component for all charts and visual maps.
- Use tabs only when the user asks for switchable categories or multiple data views.
- Do not place dense table data into very small slots.
- Do not use components that are not defined in `component-registry.json`.
- Do not generate `position` and `size` fields unless they already exist in the platform component template.
- Keep `componentProps.style.rotate` unchanged unless the user explicitly asks for rotation.

---

## 15. Scene Mapping Rules

When the user describes a scene, map it into the fixed blueprint.

### 智慧园区能耗

Recommended components:

```text
header_title        文本框
header_time         时间框
metric_1            文本框 今日能耗
metric_2            文本框 本月能耗
metric_3            文本框 碳排放
metric_4            文本框 节能率
metric_5            文本框 异常设备
left_top_panel      图表类自定义组件 能耗结构
left_middle_panel   图表类自定义组件 能耗趋势
left_bottom_panel   表格组件 楼宇能耗排行
center_main_panel   图表类自定义组件 园区能耗分布
center_bottom_panel 图表类自定义组件 分时负载趋势
right_top_panel     图表类自定义组件 设备在线率
right_middle_panel  表格组件 实时告警
right_bottom_panel  图表类自定义组件 区域能耗对比
```

### 城市运行

Recommended components:

```text
KPI text boxes: 今日事件、在线设备、处理效率、拥堵指数、告警数量
Center visual: 图表类自定义组件 城市地图
Left panels: 事件分类、运行趋势、区域排行
Right panels: 设备状态、风险告警、事件分布
Bottom panel: 事件趋势
```

### 工业生产

Recommended components:

```text
KPI text boxes: 今日产量、设备稼动率、良品率、能耗、告警数
Center visual: 图表类自定义组件 产线拓扑或工厂可视化
Left panels: 产量分析、生产趋势、产线排行
Right panels: 设备状态、异常告警、质量分布
Bottom panel: 产线趋势
```

### 水务环保

Recommended components:

```text
KPI text boxes: 总流量、水质指数、泵站在线率、告警数、处理效率
Center visual: 图表类自定义组件 水务地图或管网图
Left panels: 水质分布、流量趋势、站点排行
Right panels: 设备状态、风险告警、污染物占比
Bottom panel: 趋势分析
```

---

## 16. Data Source Rules

During current testing, use static mock data inside the real platform component JSON template.

Do not call platform data APIs or data commands in this stage.

Allowed data mode for page components:

```json
{
  "mode": "static"
}
```

Rules:

- Do not delete the platform component's original data source structure if it exists.
- Patch mock values into the existing data field.
- Keep unknown data source fields unchanged.
- If the template has no data source field, add only the minimum field required by the platform component format.
- For `custom-chart`, chart script and chart mock data primarily belong to the resource entity JSON.
- The `custom-chart` page component can keep `dataSourceProps.defaultValue` as `{}` unless the platform template requires local mock data there.

---

## 17. Validation Requirements

A generated page is valid only when all conditions are true:

```text
generated-bigscreen/page.json exists
preview.html does not exist
screen-blueprint.json does not exist
page.json root has canvas
page.json root has components
page.json must not be wrapped in page
canvas.width === 1920
canvas.height === 1080
canvas.scale === 0.64
canvas.sizeKey === "pc"
canvas.adaptationType === "AUTO"
generated background SVG content has been uploaded through fileService / UploadFile
background upload command uses Base64 SVG content
background upload command result id exists
page.canvas.backgroundImage.fileId equals background upload result id
every component is cloned from a real platform component template
every component id is unique
every component writes position and size through componentProps.style
every component is placed inside its region contentBox
every slotted component is placed inside its slot
background SVG content has viewBox 0 0 1920 1080
background SVG content has one data-region-id group for every region
only the five allowed platform component categories are used
every custom-chart component has a matching resource entity JSON file
every custom-chart resource entity uses resourceId
every custom-chart component resource id matches the resource entity resourceId
every custom-chart component resource version is 0
every custom-chart resource entity version is 0
every custom-chart resource entity writes script to configuration.javaScript
custom-chart page components do not contain script, option, echartsOption, code, or javaScript fields
every generated custom-chart resource entity has been submitted through visualizationService:resource / Add
every resource save command uses {"data":[resourceEntityJson]}
every resource save command data[0] equals the generated resource entity JSON
command result is not used to replace resourceId
generated page.json has been submitted through visualizationService:project / CreateBigScreenProject
project create command uses {"page": pageJsonRoot, "projectType": "bigScreen"} unless the user requested another supported projectType
```

If validation fails, fix all related files consistently:

```text
page.json
generated-bigscreen/resources/*.resource.json
```

Do not only fix one file when coordinates or resource bindings are inconsistent.

---

## 18. Final Response Rules

After generation, summarize only:

```text
generated files
scene
blueprint type
component source
component count
custom-chart resource count
background upload result
resource command save result
project command create result
validation result
```

Do not paste the full JSON in the final response unless explicitly requested.

---

## 19. Most Important Rule

The fixed blueprint is the layout source.

```text
fixed blueprint regions -> background SVG content
background SVG content -> fileService / UploadFile -> page.canvas.backgroundImage.fileId
fixed blueprint regions -> page.json components
```

For custom charts:

```text
generated resource id -> resource entity JSON resourceId
generated resource id -> page component componentProps.resource.id
generated resource entity JSON -> visualizationService:resource / Add
generated page.json -> visualizationService:project / CreateBigScreenProject
```

Never do this:

```text
background SVG content -> guess component positions
```

Never do this:

```text
page component -> embed ECharts script directly
```

Never do this:

```text
command result -> replace generated resourceId
```

The goal is to make the SVG background look cool while keeping every real platform component placed in a predictable and valid coordinate region.

The goal for custom charts is to generate a resource entity first, save it through the resource command, then let the page component render through `componentProps.resource.id`.
