---
name: ai-vis-page
description: "Progressive JetLinks visualization big-screen generation workflow. Use this skill for 大屏、可视化大屏、数据大屏、渐进式画布生成、SVG 背景、区域组件生成、平台组件 JSON、ECharts 资源和高级远程组件编排。Do not use it for normal frontend source-code generation."
---

# AI Vis Page Skill

This skill describes the progressive business workflow for generating a JetLinks visualization big-screen page.

The output is a platform big-screen JSON package, not a Vue, React, HTML, or standalone frontend project.

Detailed platform JSON schemas, component templates, resource entity templates, and advanced component packaging rules live in the skill reference files. The Python workflow reads only the reference subset needed by the current stage and injects that subset into the model prompt. Do not inject this whole skill package into every model request.

## Goal

Generate a big-screen page in multiple small model turns:

1. Initialization turn creates the canvas shell and decorative SVG background.
2. Later region turns create one blueprint region at a time.
3. Python performs side effects such as background upload, ECharts resource save, advanced component zip upload, and fileId/resourceId backfill.
4. The final page is assembled by the caller from the initialization result plus every region response.

The workflow exists to keep model prompts small and deterministic. The model should only produce the JSON artifact for the current stage. The Python workflow decides which platform rules and templates to include for that stage.

## Stage Contract

### Initialization

Use this stage for the first canvas/background request.

The model receives:

- User business intent, such as a park, city, industrial, energy, or water/environment scene.
- Fixed 1920 x 1080 blueprint rules selected by Python.
- SVG background constraints selected by Python.

The model returns only:

```json
{
  "pageJson": {
    "canvas": {},
    "components": []
  },
  "blueprint": {},
  "backgroundSvg": "<svg ...></svg>"
}
```

Initialization responsibilities:

- Create a decorative structural SVG background from the fixed blueprint.
- Keep the SVG as panel chrome only.
- Do not draw real metrics, chart data, table rows, alarm text, buttons, or panel title text in the SVG.
- Keep every content area clean for later platform components.
- Return an empty `pageJson.components` array.

After the model returns, Python uploads the SVG and writes the returned file id into `pageJson.canvas.backgroundImage.fileId`.

### Region

Use this stage for one requested blueprint region.

The model receives:

- User business intent.
- The requested region id and relevant blueprint coordinates.
- Only the component/resource rules selected by Python for that region intent.
- Existing orchestration state supplied by the caller when available.

The model returns only:

```json
{
  "regionId": "left_top_panel",
  "components": []
}
```

The response may include workflow-only fields such as `regionTitle`, `resources`, or `advancedComponents` when Python needs them for side effects. The final user-visible response remains `regionId` plus `components`.

Region responsibilities:

- Return real platform component JSON, not simplified logical widgets.
- For non-header panel regions, return the panel title as the first platform text component.
- Place every component inside the requested region title band, content box, or slot.
- Use built-in platform components when they can express the content.
- Use map components for map, geography, point, coordinate, or spatial distribution intent.
- Use ECharts resource components for non-map chart intent.
- Use advanced remote components only for complex visuals that cannot be represented cleanly by built-in components or normal charts.

After the model returns, Python normalizes/validates components, saves generated resources, uploads advanced component packages, and backfills file ids.

## Component Selection Intent

Use these business-level mappings. Python injects exact template rules for the selected component types.

- Title, subtitle, label, KPI, metric text: platform text.
- Current date or system time: platform date/time.
- Ranking, alarm, event, device, and status lists: platform table.
- Video, monitoring, camera, live stream, playback: platform video.
- Map, geography, region distribution, points, coordinates, longitude/latitude, GIS: platform map component.
- Line, bar, area, pie, gauge, trend, comparison, non-map distribution: ECharts resource component.
- Topology, timeline, process, rich composite panel, custom interaction: advanced remote component.

## Non-Goals

- Do not generate frontend source code.
- Do not create a standalone preview page.
- Do not call platform commands from the model.
- Do not create the final visualization project from the model.
- Do not put all component JSON rules into the first prompt.
- Do not use this skill as a generic chat assistant.

## Success Criteria

- Each model turn has a small, stage-specific prompt.
- Initialization focuses on SVG and canvas only.
- Region generation focuses on one region and only the relevant component rules.
- Python owns uploads, resource persistence, normalization, validation, and response cleanup.
- The caller can progressively assemble the big-screen page without hidden model-side state.
