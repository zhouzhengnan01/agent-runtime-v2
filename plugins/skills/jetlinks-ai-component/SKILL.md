---
name: jetlinks-ai-component
description: Generate JetLinks visualization designer remote component files for agent calls, especially component.vue and config.mjs that satisfy the runtime component format.
entrypoint: scripts/run_skill.py
runtime: python
license: Proprietary (jetlinks internal)
compatibility: >-
  For the jetlinks cloud.jetlinks.ui visualization designer. Components are loaded at
  runtime by vue3-sfc-loader (browser SFC compiler), stored via fileupload, and
  registered as visualization/resource records. Requires the ai-component-loader in
  jetlinks-web-core.
metadata:
  author: jetlinks
  version: "1.0"
---

# jetlinks 大屏 AI 组件生成规范

为 jetlinks 大屏可视化设计器(`visualization` 模块)生成**运行时加载**的图表组件。Agent 调用时默认先生成组件源码目录,不强制打 zip;前端仍用 `vue3-sfc-loader` 在浏览器编译,无需后端构建。

## Agent 调用

可通过 runtime 的 `manifest.execution` 调用 `scripts/run_skill.py`。脚本从 stdin 读取 `stdin_json`,使用 `spec` 里的字段生成组件文件:

- `objective`: 组件目标描述,必填或强烈建议提供。
- `resource_id` / `resourceId`: 组件资源标识,默认会从名称推导为 `ai_<name>_v1`。
- `display_name` / `displayName`: 设计器展示名称。
- `chart_type` / `chartType`: `bar`、`line`、`pie`,默认 `bar`。
- `default_data` / `defaultData`: 默认静态数据数组。

输出目录:

```
<outputs_dir>/<resource_id>/
├── component.vue   # 必需:主组件(Options API)
├── config.mjs      # 必需:导出 ConfigProps + Config 数组
└── README.md       # 生成摘要,便于检查
```

## 组件产物格式

如后续需要上传到设计器,可把生成目录打包成 zip:

```
ai_<name>_v<n>.zip
├── component.vue   # 必需:主组件(Options API)
├── config.mjs      # 必需:导出 ConfigProps + Config 数组
└── Config.vue      # 可选:仅当组件有"全新配置 key"时需要
```

`<name>` 是组件标识(小驼峰或下划线),与后端 `resource.resourceId` 一致,例如 `ai_basicBar_v1`。

## 硬约束(违反就无法渲染)

### 1. ⚠ 必须用 Options API,严禁 `<script setup>`
`vue3-sfc-loader` 0.9.5 浏览器版编译 `<script setup>` 时**不会把顶层 import 暴露给模板**,导致 `<VChart>` 取到 undefined → 渲染空 vnode `<!---->`(架子在但图不出)。必须用 `<script>` + `export default { components, setup() { return {...} } }`。

### 2. 模板用到的组件必须显式 `components: { ... }` 注册
`import VChart from 'vue-echarts'` 后必须 `components: { VChart }`,且模板用 **PascalCase** `<VChart>`(不要 `<v-chart>`)。

### 3. setup 必须显式 `return` 模板用到的所有 ref / computed
没有 auto-expose,模板里用到的 `backgroundStyle`、`option` 等都要在 `return` 里。

### 4. 配置文件用 `.mjs` 扩展名,zip 内部 import 必须带完整扩展名
- 配置文件 `.js` 会被当 CommonJS 处理而不识别 `export const`,必须 `.mjs`
- 内部 import 必须写完整扩展名:`import X from './Config.vue'`(✗ `./Config`),加载器不做扩展名 fallback 找文件

### 5. 导出命名约定(host 自动匹配,错了不渲染)
`config.mjs` 必须导出 **`<resourceId>ConfigProps`** 和 **`<resourceId>Config`**,`ConfigProps.type` 也必须等于 `resourceId`。例:`ai_basicBar_v1ConfigProps` / `ai_basicBar_v1Config` / `type: 'ai_basicBar_v1'`。

### 6. `ConfigProps.componentProps.style` 必填 `{ x, y, width, height }`
设计器拖入时读取,缺失会抛错。

### 7. 只能 import 依赖白名单
```
vue · vue-echarts · echarts/core · echarts/charts · echarts/components · echarts/renderers
lodash-es
@visualization-resources/components/common/Theme/data
@visualization-resources/packages/component/utils
@jetlinks-web-core/utils/module-registry   (仅 Config.vue 取 ConfigItem 等时用)
```
白名单外的 bare import 会找不到模块。新增白名单需改 host `shared-modules.ts`。

### 8. `<style scoped lang="less">` 里只写纯 CSS
sfc-loader 没有 less 编译器(host 把 less 当 CSS 处理),不能用 less 变量 / mixin。

## 右侧配置面板的机制(决定 Config.vue 是否需要)

设计器右侧的"配置分组" = `componentProps` 的每个 key(除 `style`)。每个 key 用哪个面板组件,来自一个**全局字典**,汇总了所有组件注册的面板。

- **复用 host 已有 key** → 面板自动出现,**无需 Config.vue**:
  `background` `font` `xAxis` `yAxis` `legend` `theme` `basicBar` `lineBar` 等(host 系统组件已注册)。
- **用全新 key**(host 字典里没有,如自定义 `aiExtra`)→ 该分组会**空白**,**必须**在 `config.mjs` 的 `Config` 数组里注册自带的 `Config.vue`。

## 最小骨架

`component.vue`:
```vue
<template>
  <div class="component-box" :style="backgroundStyle">
    <VChart class="chart" :option="option" autoresize />
  </div>
</template>
<script>
import { ref, computed, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { TooltipComponent, LegendComponent } from 'echarts/components'
import { BarChart } from 'echarts/charts'
import VChart from 'vue-echarts'
import { setComponentBackground } from '@visualization-resources/packages/component/utils'
use([CanvasRenderer, BarChart, TooltipComponent, LegendComponent])
export default {
  name: 'AiBasicBar',
  components: { VChart },
  props: { info: { type: Object, default: () => ({}) }, isEdit: Boolean },
  setup(props) {
    const backgroundStyle = computed(() => setComponentBackground(props.info?.componentProps?.background))
    const option = ref({ /* echarts option */ })
    watch(() => props.info, () => { /* 重算 option */ }, { deep: true, immediate: true })
    return { backgroundStyle, option }   // 必须显式 return
  }
}
</script>
<style scoped lang="less">.chart { width: 100%; height: 100%; }</style>
```

`config.mjs`:
```js
export const ai_basicBar_v1ConfigProps = {
  name: 'AI 基础柱状图',
  type: 'ai_basicBar_v1',                       // 必须 = resourceId
  componentProps: {
    style: { x: 0, y: 0, width: 400, height: 270 },   // 必填
    background: {/*...*/}, xAxis: {/*...*/}, yAxis: {/*...*/},
    legend: {/*...*/}, theme: {/*...*/}, basicBar: {/*...*/}   // 复用 host key
  },
  dataSourceProps: { mode: 'static', type: 'array', defaultValue: [/*...*/] },
  animationProps: []
}
export const ai_basicBar_v1Config = []          // 全用 host key → 空数组即可
```

`Config.vue`(仅全新 key 时,Options API):
```vue
<template>
  <div class="card-container">
    <ConfigItem label="高亮显示"><a-switch v-model:checked="cfg.highlight" @change="onChange" /></ConfigItem>
  </div>
</template>
<script>
import { ref, watch } from 'vue'
import { cloneDeep } from 'lodash-es'
import { moduleRegistry } from '@jetlinks-web-core/utils/module-registry'
const { ConfigItem, ColorPicker, InputNumber } = moduleRegistry.getResource('visualization-designer-ui', 'components')
export default {
  name: 'AiExtraConfig', components: { ConfigItem, ColorPicker, InputNumber },
  props: { activeComponent: { type: Object, default: () => ({}) } },
  emits: ['change'],
  setup(props, { emit }) {
    const cfg = ref({})
    const onChange = () => emit('change', cfg.value, 'aiExtra')   // 第二参 = componentProps 里的 key
    watch(() => props.activeComponent?.componentProps?.aiExtra, v => { cfg.value = cloneDeep(v || {}) }, { deep: true, immediate: true })
    return { cfg, onChange }
  }
}
</script>
```

**Config.vue 开发规范**(与 host 系统组件保持一致):
- 根元素:`<div class="card-container">`(flex column gap 12px)
- 每个配置项用 `<ConfigItem label="...">` 包裹表单控件
- 可用的面板组件(从 `moduleRegistry.getResource('visualization-designer-ui', 'components')` 解构):
  - `ConfigItem` — 标签 + 内容布局容器
  - `InputNumber` — 数字输入(支持 min/max/step/precision)
  - `ColorPicker` — 颜色选择器(`:isInput="false"` 无输入框模式)
  - `ImageUpload` — 图片上传
  - antd 全局组件直接用:`a-switch` `a-select` `a-slider` `a-input` `a-space`
- **emit 格式**:`emit('change', value, key)` — `value` 是该 key 下完整配置对象,`key` 必须等于 `componentProps` 中的属性名(设计器据此精准更新对应字段)
- **watch 模式**:监听 `props.activeComponent?.componentProps?.[key]` → `cloneDeep` 到本地 ref,确保引用隔离
对应 `config.mjs`:`import Panel from './Config.vue'` → `componentProps.aiExtra = { highlight: false }` → `export const ai_basicBar_v1Config = [{ name: 'aiExtra', component: Panel }]`。

## 完整可运行范例

`assets/ai_basicBar_v1/` — 经端到端验证的三件套(主组件 + config + 自定义 Config.vue),直接参考或复制改写。

## 打包与上线

打包:`node scripts/build-zip.mjs <componentDir>`(递归收集目录所有文件成 zip)。
上传 fileupload 拿 fileId、创建 `visualization/resource` 记录、host 接入细节 → 见 `references/integration.md`。

## 出问题先查

渲染空白 `<!---->`、配置分组空白、跨域 403、less 报错等 → 见 `references/troubleshooting.md`。
