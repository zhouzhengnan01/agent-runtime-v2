# Command Standard

This reference defines platform command calling rules for this skill.

The skill must describe platform command invocations when generated background assets or resources require persistence. The platform intelligent agent is responsible for executing the command call.

Do not call HTTP APIs directly.

Do not use fixed tokens.

Do not generate `curl` commands.

---

## 1. Current Supported Commands

```text
Service ID: fileService
Command ID: UploadFile
Purpose: upload the generated SVG background asset
Used by: pageJson.canvas.backgroundImage.fileId
```

```text
Service ID: visualizationService:resource
Command ID: Add
Purpose: create visualization resource entities for custom-chart / resourceComponentEcharts
Used by: custom-chart resource entity persistence
```

No other platform command is available in the current skill stage.

Do not call data source commands, preview commands, or HTTP APIs unless this reference is updated.

Do not call `visualizationService:project / CreateBigScreenProject`; the frontend is responsible for inserting returned region components into its current pageJson and creating or updating the big-screen project.

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
pageJson.canvas.backgroundImage.fileId
```

Treat a missing command result `id` as upload failure.

Do not return the initialization response as ready until the background upload has succeeded and `pageJson.canvas.backgroundImage.fileId` has been set, unless the user explicitly accepts a pageJson without background fileId.

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
8. Add the component to the current region response components array.
```

Resource id must not depend on command result.

Version is always:

```text
0
```

---

## 5. Progressive Return Rules

After initialization `pageJson` is generated, background SVG upload succeeds, and `pageJson.canvas.backgroundImage.fileId` is set, return:

```json
{
  "pageJson": {},
  "blueprint": {}
}
```

After one region's components are generated and all custom-chart resource save commands for that region have been attempted, return:

```json
{
  "regionId": "left_top_panel",
  "components": []
}
```

Do not pass `pageJSON`.

Do not wrap the generated pageJson root inside another nested `page` field.

Do not call `visualizationService:project / CreateBigScreenProject`.

---

## 6. Command Failure Rules

If `fileService / UploadFile` fails:

```text
1. Do not claim background upload success.
2. Do not return the initialization response as ready unless the user explicitly accepts a pageJson without background fileId.
3. Report the failed background upload command.
```

If `visualizationService:resource / Add` fails:

```text
1. Keep the generated resource JSON file.
2. Keep returned component JSON referencing the generated resourceId unless the user asks to omit unsaved custom-chart components.
3. Do not claim resource command success.
4. Report the failed resource command.
```
