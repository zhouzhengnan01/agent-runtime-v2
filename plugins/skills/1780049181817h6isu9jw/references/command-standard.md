# Command Standard

This reference defines platform command calling rules for this skill.

The skill must describe platform command invocations when generated background assets, resources, or projects require persistence. The platform intelligent agent is responsible for executing the command call.

Do not call HTTP APIs directly.

Do not use fixed tokens.

Do not generate `curl` commands.

---

## 1. Current Supported Commands

```text
Service ID: fileService
Command ID: UploadFile
Purpose: upload the generated SVG background asset
Used by: canvas.backgroundImage.fileId
```

```text
Service ID: visualizationService:resource
Command ID: Add
Purpose: create visualization resource entities for custom-chart / resourceComponentEcharts
Used by: custom-chart resource entity persistence
```

```text
Service ID: visualizationService:project
Command ID: CreateBigScreenProject
Purpose: create a visualization project from the generated page object
Used by: final generated page persistence
```

No other platform command is available in the current skill stage.

Do not call data source commands, preview commands, or HTTP APIs unless this reference is updated.

---

## 2. Background SVG Upload Command

Generate the background SVG content from the fixed blueprint regions, then call:

```text
serviceId = fileService
commandId = UploadFile
```

Command parameters:

```json
{
  "fileName": "background.svg",
  "contentType": "image/svg+xml;charset=UTF-8",
  "content": "__BASE64_UTF8_SVG__"
}
```

The `content` parameter must be Base64 of the generated SVG UTF-8 bytes.

Do not pass raw SVG text as `content`.

The command result is `FileInfo`.

Use the command result `id` as:

```text
page.canvas.backgroundImage.fileId
```

Treat a missing command result `id` as upload failure.

Do not call `visualizationService:project / CreateBigScreenProject` until the background upload has succeeded and `page.canvas.backgroundImage.fileId` has been set.

---

## 3. Custom Chart Resource Save Command

For every generated `custom-chart` resource entity file:

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

The object in `data[0]` must be exactly the generated resource entity JSON file content.

The generated `resourceId` must already exist before the command call.

Do not use the command result to replace the generated `resourceId`.

Do not wait for the command result to decide the `resourceId`.

---

## 4. Custom Chart Save Flow

For every `custom-chart` component:

```text
1. Generate resourceId before command call.
2. Generate generated-bigscreen/resources/{resourceId}.resource.json.
3. Write the same resourceId to the resource JSON field resourceId.
4. Call visualizationService:resource / Add with {"data":[resourceEntityJson]}.
5. Clone references/components/custom-chart.json.
6. Write the same resourceId to componentProps.resource.id.
7. Set componentProps.resource.version to 0.
8. Add the component to page.json.components.
```

Resource id must not depend on command result.

Version is always:

```text
0
```

---

## 5. Project Create Command

After `page.json` is generated, background SVG upload succeeds, `page.canvas.backgroundImage.fileId` is set, and all custom-chart resource save commands have been attempted, call the platform command:

```text
serviceId = visualizationService:project
commandId = CreateBigScreenProject
```

Command parameters:

```json
{
  "page": {
    "canvas": {},
    "components": []
  },
  "projectType": "bigScreen"
}
```

The `page` parameter must be exactly the generated `generated-bigscreen/page.json` root object.

Do not pass `pageJSON`.

Do not wrap the generated page root inside another nested `page` field.

`projectType` should be `"bigScreen"` unless the user explicitly asks for another supported visualization project type.

Supported `projectType` values are:

```text
bigScreen
configuration
threeDimension
dashboard
report
component
other
```

The command result is used only to determine whether project creation succeeded and to report the created project identity if available.

---

## 6. Command Failure Rules

If `fileService / UploadFile` fails:

```text
1. Keep generated-bigscreen/page.json if it has already been generated.
2. Do not claim background upload success.
3. Do not call CreateBigScreenProject unless the user explicitly accepts a page without background fileId.
4. Report the failed background upload command in the final summary.
```

If `visualizationService:resource / Add` fails:

```text
1. Keep the generated resource JSON file.
2. Keep page.json referencing the generated resourceId.
3. Do not claim resource command success.
4. Report the failed resource command in the final summary.
```

If `visualizationService:project / CreateBigScreenProject` fails:

```text
1. Keep generated-bigscreen/page.json.
2. Keep generated-bigscreen/resources/*.resource.json.
3. Do not claim project command success.
4. Report the failed project command in the final summary.
```
