---
name: generate-screen-skill
description: Generate platform-compatible visualization big-screen JSON, upload SVG background through platform file commands, generate ECharts resource entities, save custom-chart resources through platform commands, and create visualization projects through platform commands from natural language. Use this skill for 大屏、可视化大屏、数据大屏、智慧园区、能耗大屏、SVG背景上传、page.json、resourceComponentEcharts、自定义图表资源、ECharts resource generation、可视化资源新增、可视化项目生成. Do not use this skill for normal frontend code generation.
---

# Generate Screen Skill

This skill generates platform-compatible visualization big-screen files from natural language.

The generated result is not a frontend source code project.

The output is a platform big-screen JSON package.

This skill must also describe platform command calls for uploading the generated SVG background, saving generated custom ECharts resource entities, and creating the generated visualization project.

---

## 1. Always Read These References

Before generating any big-screen files, read and follow these references:

```text
references/blueprint-standard.md
references/component-registry.json
references/echarts-script-standard.md
references/command-standard.md
references/resources/echarts-resource.json
references/components/text.json
references/components/table.json
references/components/dateTime.json
references/components/tabs.json
references/components/custom-chart.json
```

The references define:

```text
fixed 1920x1080 layout blueprint
platform canvas JSON structure
allowed platform component templates
real component patch rules
ECharts resource entity structure
ECharts script standard
custom-chart resource binding rules
platform command calling rules
```

Never ignore these references.

Never invent a new page schema when the reference files define the platform schema.

Never invent a new HTTP API or command when `references/command-standard.md` defines the available command list.

---

## 2. Skill Goal

Generate a visualization big screen from user intent.

The generated package must use:

```text
fixed blueprint regions -> background SVG content
background SVG content -> fileService / UploadFile -> canvas.backgroundImage.fileId
fixed blueprint regions -> page.json components
```

For custom ECharts charts:

```text
generated resourceId -> generated resource entity JSON
generated resourceId -> page component componentProps.resource.id
generated resource entity JSON -> visualizationService:resource / Add
generated page.json -> visualizationService:project / CreateBigScreenProject
```

The goal is to make the SVG background visually cool while keeping every real platform component placed inside predictable coordinate regions.

The goal for custom charts is:

```text
generate ECharts resource entity JSON
save it through platform resource command
bind page component to the same resourceId
create a visualization project through platform project command
```

---

## 3. Required Output

For each generated big screen, create:

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

Do not generate:

```text
preview.html
screen-blueprint.json
README.md
frontend source code
React code
Vue code
standalone demo app
```

The generated result must be usable as a platform big-screen JSON package.

Do not keep `background.svg` as a required output file. Generate the SVG content, upload it through the file command, and write the returned file id into `page.json`.

---

## 4. Platform Command Rules

The skill must use platform command descriptions instead of direct HTTP API calls.

Do not call HTTP APIs directly.

Do not use fixed tokens.

Do not generate `curl` commands.

Supported platform commands are defined in:

```text
references/command-standard.md
```

For custom-chart resource persistence, use:

```text
serviceId = visualizationService:resource
commandId = Add
parameters = {"data":[resourceEntityJson]}
```

For visualization project creation, use:

```text
serviceId = visualizationService:project
commandId = CreateBigScreenProject
parameters = {"page": pageJsonRoot, "projectType": "bigScreen"}
```

For SVG background upload, use:

```text
serviceId = fileService
commandId = UploadFile
parameters = {"fileName":"background.svg","contentType":"image/svg+xml;charset=UTF-8","content":"__BASE64_UTF8_SVG__"}
```

The file upload command result `id` must be written to `page.canvas.backgroundImage.fileId`.

The generated `resourceId` must already exist before the resource command call.

The command result is used only to determine whether the operation succeeded.

Do not use the command result to replace the generated `resourceId`.

Do not wait for the command result to decide the `resourceId`.

---

## 5. Command Failure Rule

If the resource save command fails:

```text
keep generated-bigscreen/resources/{resourceId}.resource.json
keep generated-bigscreen/page.json
keep page component componentProps.resource.id as the generated resourceId
do not claim resource command success
report the resource command failure in the final summary
```

If the project create command fails:

```text
keep generated-bigscreen/page.json
keep generated-bigscreen/resources/*.resource.json
do not claim project command success
report the project command failure in the final summary
```

If the background upload command fails:

```text
keep generated-bigscreen/page.json if it has already been generated
do not claim background upload success
do not call CreateBigScreenProject unless the user explicitly accepts a page without background fileId
report the background upload command failure in the final summary
```

Do not delete generated files because of command failure.

Do not regenerate a different `resourceId` only because the command failed.

Do not block generation of `page.json` when the resource command fails.

---

## 6. Page JSON Root Structure

The generated `page.json` must use the platform root structure:

```json
{
  "canvas": {},
  "components": []
}
```

Do not wrap it with:

```json
{
  "page": {}
}
```

Do not add debug layout fields into `page.json`.

Do not write `regionId`, `slotId`, `position`, or `size` into platform components unless the real platform component template already contains those fields.

Use the canvas structure defined in:

```text
references/blueprint-standard.md
```

The canvas must stay:

```text
width: 1920
height: 1080
scale: 0.64
sizeKey: pc
adaptationType: AUTO
```

---

## 7. Fixed Blueprint Rule

The skill currently supports exactly one blueprint:

```text
standard-1920x1080-v1
```

Hard rules:

```text
canvas size = 1920 x 1080
SVG viewBox = 0 0 1920 1080
component coordinates = absolute 1920 x 1080 coordinate system
component placement = inside region contentBox or slot
```

Do not redesign the layout.

Do not invent new global regions.

Do not generate a decorative SVG first and then guess component positions.

The fixed blueprint regions in `references/blueprint-standard.md` are the layout source of truth.

---

## 8. Background SVG Rules

Generate SVG background content from the fixed blueprint regions.

Do not write `generated-bigscreen/background.svg` as a required output file.

The SVG must use:

```xml
<svg viewBox="0 0 1920 1080" width="1920" height="1080" xmlns="http://www.w3.org/2000/svg">
```

The SVG must be generated from the fixed layout regions.

Each region must have a corresponding SVG group with region metadata:

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

SVG may include:

```text
dark technology background
grid lines
soft glow
panel borders
panel corner decorations
section dividers
center visual base decoration
```

SVG must not include:

```text
real metric values
chart data
ranking rows
alarm text
business table content
interactive buttons
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

Treat a missing upload result `id` as upload failure.

---

## 9. Platform Component Rule

Only use real platform component templates from:

```text
references/components/
```

Allowed component keys:

```text
text
table
dateTime
tabs
custom-chart
```

Allowed platform component categories:

```text
文本框
表格组件
时间框
选项卡
图表类自定义组件
```

Do not generate old logical test components such as:

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

Instead, map business needs to real platform component templates:

```text
大屏标题 -> text
指标数字 -> text
当前时间 -> dateTime
排行/告警/列表 -> table
折线图/柱状图/饼图/面积图/仪表盘/地图主视觉 -> custom-chart
多视图切换 -> tabs
```

---

## 10. Component Clone and Patch Rule

When generating a component:

1. Select the correct component key from `references/component-registry.json`.
2. Read the matching template from `references/components/`.
3. Deep clone the template.
4. Keep all unknown platform fields unchanged.
5. Patch only allowed fields.
6. Write the result into `page.json.components`.

Do not simplify real platform components.

Do not rename platform fields.

Do not remove fields that exist in the original template.

---

## 11. Component Position Rule

Real platform component position and size must be written to:

```text
componentProps.style.x
componentProps.style.y
componentProps.style.width
componentProps.style.height
```

Do not write new fields:

```text
position
size
x
y
width
height
```

unless those fields already exist in the real platform component template.

Placement formula:

```text
componentProps.style.x >= contentBox.x
componentProps.style.y >= contentBox.y
componentProps.style.x + componentProps.style.width <= contentBox.x + contentBox.width
componentProps.style.y + componentProps.style.height <= contentBox.y + contentBox.height
```

When using a slot:

```text
componentProps.style.x >= slot.x
componentProps.style.y >= slot.y
componentProps.style.x + componentProps.style.width <= slot.x + slot.width
componentProps.style.y + componentProps.style.height <= slot.y + slot.height
```

Prefer slot placement when a suitable slot exists.

---

## 12. Custom Chart Resource Rule

The `custom-chart` component is special.

It must generate two related JSON objects:

```text
1. ECharts resource entity JSON
2. page component JSON
```

The resource entity contains the script:

```text
configuration.javaScript
```

The page component only references the resource:

```text
componentProps.resource.id
componentProps.resource.version
```

Do not put ECharts script code into `page.json.components`.

Do not add these fields to page components:

```text
script
option
echartsOption
code
javaScript
```

Every generated custom-chart resource entity must be saved through the platform resource command:

```text
visualizationService:resource / Add
```

---

## 13. Custom Chart Generation and Save Flow

For every `custom-chart`:

1. Determine the chart type and business role.
2. Generate a random resource id.
3. Clone `references/resources/echarts-resource.json`.
4. Patch the resource entity JSON.
5. Write it to `generated-bigscreen/resources/{resourceId}.resource.json`.
6. Call the resource save command with the resource JSON in `data[0]`.
7. Clone `references/components/custom-chart.json`.
8. Patch the page component resource id.
9. Patch the page component position and size.
10. Add the page component to `page.json.components`.

Resource id rule:

```text
resourceId = component_{timestamp}_{random4}
```

Example:

```text
component_1779881457212_4829
```

Resource version rule:

```text
version = 0
```

The same resource id must be used in both places:

```text
resource entity JSON -> resourceId
page component JSON  -> componentProps.resource.id
```

The version must be `0` in both places:

```text
resource entity JSON -> version
page component JSON  -> componentProps.resource.version
```

Do not wait for a save command result to determine the resource id.

Do not use the save command result to replace the generated resource id.

---

## 14. Resource Save Command Call

For every generated file:

```text
generated-bigscreen/resources/{resourceId}.resource.json
```

Call the platform command:

```text
serviceId = visualizationService:resource
commandId = Add
```

Command parameters:

```json
{
  "data": [
    {
      "resourceId": "__RESOURCE_ID__",
      "name": "__RESOURCE_NAME__",
      "version": 0,
      "thumbnailUrl": "",
      "provider": "local",
      "type": "component",
      "group": "[\"vis_oneself_dimension_line-chart\"]",
      "configuration": {
        "componentType": "echarts",
        "javaScript": "__ECHARTS_JAVASCRIPT__"
      }
    }
  ]
}
```

Rules:

```text
Command serviceId = visualizationService:resource
Command commandId = Add
Command body data[0] = full resource entity JSON
```

The resource save command must be called only for generated custom-chart resource entities.

Do not call this command for:

```text
text
table
dateTime
tabs
background SVG content
page.json
```

Before creating the project, upload the generated background SVG content:

```text
serviceId = fileService
commandId = UploadFile
parameters = {"fileName":"background.svg","contentType":"image/svg+xml;charset=UTF-8","content":"__BASE64_UTF8_SVG__"}
```

Write the returned `id` to:

```text
page.canvas.backgroundImage.fileId
```

After `page.json` has been generated, the background upload has succeeded, and every custom-chart resource save command has been attempted, call the project create command:

```text
serviceId = visualizationService:project
commandId = CreateBigScreenProject
parameters = {"page": pageJsonRoot, "projectType": "bigScreen"}
```

The `page` parameter must be exactly the generated `generated-bigscreen/page.json` root object.

Do not pass `pageJSON`.

Do not wrap the generated page root inside another nested `page` field.

`projectType` defaults to `bigScreen`; only pass another supported project type when the user explicitly asks for it.

No preview command is available yet.

Do not invent unavailable APIs or commands.

---

## 15. ECharts Resource Entity Rule

Every generated ECharts resource entity must follow:

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
resourceId                  = generated component_{timestamp}_{random4}
name                        = readable chart resource name
version                     = 0
thumbnailUrl                = keep empty unless user/platform provides one
provider                    = local
type                        = component
group                       = JSON string array
configuration.componentType = echarts
configuration.javaScript    = generated platform-standard ECharts script
```

Chart group mapping:

```text
line       -> ["vis_oneself_dimension_line-chart"]
bar        -> ["vis_oneself_dimension_bar-chart"]
area       -> ["vis_oneself_dimension_line-chart"]
pie        -> ["vis_oneself_dimension_pie-chart"]
gauge      -> ["vis_oneself_dimension_gauge"]
custom-map -> ["vis_oneself_dimension_map"]
map        -> ["vis_oneself_dimension_map"]
```

---

## 16. ECharts Script Rule

Generate ECharts scripts according to:

```text
references/echarts-script-standard.md
```

Every script must expose exactly:

```js
function defaultFunc(_getflag, _setdata, echarts, chartInstance) {
}
```

Every script must support:

```js
if (_getflag === 'data') return data
```

Every script must support:

```js
if (_setdata) {
  data = _setdata
}
```

Every script must return:

```js
return option
```

Do not use:

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
manual echarts.init
chartInstance.setOption
manual chart disposal
hard-coded DOM container id
```

The platform handles chart rendering.

The script only builds and returns ECharts option.

---

## 17. Business Scene Mapping

When the user describes a scene, map it into the fixed blueprint.

For 智慧园区能耗:

```text
header_title        text         智慧园区能耗监测大屏
header_time         dateTime     当前时间
metric_1            text         今日能耗
metric_2            text         本月能耗
metric_3            text         碳排放
metric_4            text         节能率
metric_5            text         异常设备
left_top_panel      custom-chart 能耗结构
left_middle_panel   custom-chart 能耗趋势
left_bottom_panel   table        楼宇能耗排行
center_main_panel   custom-chart 园区能耗分布
center_bottom_panel custom-chart 分时负载趋势
right_top_panel     custom-chart 设备在线率
right_middle_panel  table        实时告警
right_bottom_panel  custom-chart 区域能耗对比
```

For 城市运行:

```text
metrics: 今日事件、在线设备、处理效率、拥堵指数、告警数量
center main visual: 城市地图
left panels: 事件分类、运行趋势、区域排行
right panels: 设备状态、风险告警、事件分布
bottom panel: 事件趋势
```

For 工业生产:

```text
metrics: 今日产量、设备稼动率、良品率、能耗、告警数
center main visual: 产线拓扑或工厂可视化
left panels: 产量分析、生产趋势、产线排行
right panels: 设备状态、异常告警、质量分布
bottom panel: 产线趋势
```

For 水务环保:

```text
metrics: 总流量、水质指数、泵站在线率、告警数、处理效率
center main visual: 水务地图或管网图
left panels: 水质分布、流量趋势、站点排行
right panels: 设备状态、风险告警、污染物占比
bottom panel: 趋势分析
```

For unknown scenes, still use the same fixed blueprint and infer suitable component roles.

---

## 18. Data Rules

During current testing, use static mock data.

Do not call platform data APIs or data commands.

For normal platform components:

```text
patch mock values into existing dataSourceProps only when needed
keep unknown dataSourceProps fields unchanged
```

For `custom-chart`:

```text
chart data mainly belongs inside configuration.javaScript default data
page component dataSourceProps.defaultValue can remain {}
```

Do not delete the platform component's original data source structure.

---

## 19. Validation Checklist

Before finishing, validate the generated files manually or through scripts when available.

The generated output is valid only when all conditions are true:

```text
generated-bigscreen/page.json exists
preview.html does not exist
screen-blueprint.json does not exist
page.json root has canvas
page.json root has components
page.json is not wrapped in page
canvas.width is 1920
canvas.height is 1080
canvas.scale is 0.64
canvas.sizeKey is pc
canvas.adaptationType is AUTO
generated background SVG content has been uploaded through fileService / UploadFile
background upload command uses Base64 SVG content
background upload command result id exists
page.canvas.backgroundImage.fileId equals background upload result id
all components are cloned from references/components templates
all component ids are unique
all components write position and size through componentProps.style
all components stay inside assigned contentBox or slot
background SVG content has viewBox 0 0 1920 1080
background SVG content has one data-region-id group for every fixed region
only allowed component keys are used
every custom-chart component has a matching resource JSON file
every custom-chart resource JSON has resourceId
every custom-chart component resource id equals resourceId
every custom-chart resource entity version is 0
every custom-chart page component resource version is 0
every custom-chart resource entity writes script to configuration.javaScript
custom-chart page components do not contain script, option, echartsOption, code, or javaScript
every generated custom-chart resource entity has been submitted through visualizationService:resource / Add
every resource save command uses {"data":[resourceEntityJson]}
every resource save command data[0] equals the generated resource entity JSON
command result is not used to replace resourceId
generated page.json has been submitted through visualizationService:project / CreateBigScreenProject
project create command uses {"page": pageJsonRoot, "projectType": "bigScreen"} unless the user requested another supported projectType
```

If validation fails, fix related files consistently:

```text
page.json
generated-bigscreen/resources/*.resource.json
```

Do not only fix one file when coordinates or resource bindings are inconsistent.

---

## 20. Workflow

When invoked, follow this workflow:

```text
1. Read user description.
2. Identify scene and business intent.
3. Read blueprint-standard.md.
4. Read component-registry.json.
5. Read echarts-script-standard.md.
6. Read command-standard.md.
7. Read all needed component templates.
8. Generate background SVG content from fixed blueprint regions.
9. Upload background SVG content through fileService / UploadFile.
10. Generate page.json using platform canvas root and set canvas.backgroundImage.fileId from the upload result id.
11. Clone and patch real platform components.
12. For every custom-chart, generate matching resource entity JSON.
13. For every custom-chart resource entity, call visualizationService:resource / Add.
14. Call visualizationService:project / CreateBigScreenProject with the generated page parameter.
15. Validate files and command attempts against the checklist.
16. Fix inconsistencies.
17. Summarize generated result and command results.
```

When custom-chart exists, the resource command call must happen before final summary.

The background upload command call must happen before the project create command.

The project create command call must happen before final summary.

If a command call fails, the final summary must clearly state the failed command result.

---

## 21. Final Response Rules

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

Do not paste the full JSON unless the user explicitly asks.

Do not explain internal reasoning.

Do not ask follow-up questions.

---

## 22. Most Important Rules

The fixed blueprint is the layout source:

```text
fixed blueprint regions -> background SVG content
background SVG content -> fileService / UploadFile -> page.canvas.backgroundImage.fileId
fixed blueprint regions -> page.json components
```

Custom charts must use resource binding:

```text
generated resourceId -> resource entity JSON resourceId
generated resourceId -> page component componentProps.resource.id
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

Never generate frontend code for this skill.

Never invent unavailable platform APIs or commands.

The output is a platform big-screen JSON package.
