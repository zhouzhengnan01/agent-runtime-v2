# Advanced Remote Component Standard

This reference defines the remote advanced Vue component rules used by the visualization bigscreen workflow.

Use an advanced remote component only when built-in platform components cannot express the requested visual or interaction, and when the content is not a chart that should use `resourceComponentEcharts` and is not a map that should use `pseudo`.

## Required Package Files

Every advanced component package must be zipped with these files at the zip root:

```text
component.vue
config.mjs
```

Optional:

```text
Config.vue
README.md
```

Do not put `component.vue` or `config.mjs` in a nested folder.

## Resource Id Rule

`resourceId` must be generated before Python uploads or saves anything.

Rules:

```text
resourceId must be a valid JavaScript identifier.
Use ids like ai_smartParkStatus_v1 or ai_energyTopology_v1.
Do not use hyphens, spaces, Chinese characters, or dots in resourceId.
component.type must equal resourceId.
resource.resourceId must equal resourceId.
config.mjs ConfigProps.type must equal resourceId.
```

## component.vue Rule

`component.vue` must use Vue Options API.

Forbidden:

```vue
<script setup>
```

Required pattern:

```vue
<script>
export default {
  name: 'AiSmartParkStatus',
  props: {
    info: { type: Object, default: () => ({}) },
    isEdit: Boolean
  },
  setup(props) {
    return {}
  }
}
</script>
```

Any variable, ref, computed value, or function used by the template must be explicitly returned from `setup`.

If importing Vue APIs, use:

```js
import { ref, computed, watch } from 'vue'
```

Allowed bare imports:

```text
vue
vue-echarts
echarts/core
echarts/charts
echarts/components
echarts/renderers
lodash-es
@visualization-resources/components/common/Theme/data
@visualization-resources/packages/component/utils
@jetlinks-web-core/utils/module-registry
```

Do not import external CDN scripts, axios, request clients, or non-whitelisted packages.

Use CSS only in `<style scoped lang="less">`; do not use Less variables, mixins, or nested Less-only syntax.

## config.mjs Rule

`config.mjs` must export exactly the resource-specific config symbols:

```js
export const ai_smartParkStatus_v1ConfigProps = {
  name: 'Smart park status',
  type: 'ai_smartParkStatus_v1',
  componentProps: {
    style: { x: 0, y: 0, width: 400, height: 270 },
    background: {},
    theme: {}
  },
  dataSourceProps: { mode: 'static', type: 'array', defaultValue: [] },
  animationProps: []
}

export const ai_smartParkStatus_v1Config = []
```

For a different `resourceId`, replace every `ai_smartParkStatus_v1` prefix with that exact `resourceId`.

`ConfigProps.componentProps.style` must include:

```text
x
y
width
height
```

If `config.mjs` imports `Config.vue`, the import must include the full extension:

```js
import Panel from './Config.vue'
```

Do not write:

```js
import Panel from './Config'
```

## Optional Config.vue Rule

Only create `Config.vue` when introducing a brand-new `componentProps` key that the host does not already know how to configure.

`Config.vue` must also use Options API and must emit:

```js
emit('change', value, key)
```

The `key` must equal the target key under `componentProps`.

Use this root:

```vue
<div class="card-container">
</div>
```

Use host designer controls from:

```js
import { moduleRegistry } from '@jetlinks-web-core/utils/module-registry'
const { ConfigItem, ColorPicker, InputNumber } = moduleRegistry.getResource('visualization-designer-ui', 'components')
```

## Page Component JSON

Clone `references/components/custom-component.json` and patch only safe fields.

Required fields:

```json
{
  "type": "__RESOURCE_ID__",
  "visible": true,
  "isLocked": false,
  "configuration": {
    "componentType": "remote",
    "fileId": "",
    "zipUrl": ""
  },
  "componentProps": {
    "style": {
      "x": 0,
      "y": 0,
      "width": 400,
      "height": 270,
      "rotate": {
        "angle": 0
      }
    }
  }
}
```

Python will write the uploaded zip `fileId` to:

```text
component.configuration.fileId
component.configuration.zipUrl
```

## Resource Entity JSON

Clone `references/resources/custom-resource.json`.

Required fields:

```json
{
  "resourceId": "__RESOURCE_ID__",
  "name": "__RESOURCE_NAME__",
  "version": 0,
  "thumbnailUrl": "",
  "provider": "local",
  "type": "component",
  "group": "[\"vis_oneself_dimension_line-chart\"]",
  "configuration": {
    "componentType": "remote",
    "fileId": ""
  }
}
```

Python will write the uploaded zip `fileId` to:

```text
resource.configuration.fileId
```

The resource entity is saved through:

```text
serviceId = visualizationService:resource
commandId = Add
parameters = {"data":[resourceEntityJson]}
```
