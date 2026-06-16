# 集成与上线

AI 组件从 zip 到设计器可拖拽的完整链路。

## 1. 后端 resource 结构

AI 组件的 resource 跟普通系统组件结构一致,**只在 `configuration` 里加两个字段**,不需要扩库:

```json
{
  "name": "AI 基础柱状图",
  "type": "component",
  "resourceId": "ai_basicBar_v1",
  "group": "[\"<分类id>\"]",
  "configuration": {
    "componentType": "remote",
    "fileId": "<zip 的 fileId>"
  },
  "thumbnailUrl": "<fileId 或独立图片 fileId>",
  "options": {},
  "version": 0
}
```

关键点:
- `configuration.componentType = "remote"` — 标识为运行时加载组件
- `configuration.fileId` — zip 文件 id
- `group` 必须是 **JSON 字符串**(`"[\"id\"]"`),不能是数组
- `resourceId` 必须与 `config.mjs` 里 `ConfigProps.type` 一致

## 2. 打包 + 上传 + 注册(curl)

```bash
# 打包
node scripts/build-zip.mjs ai_basicBar_v1
# → 输出与 build-zip 同级目录的 ai_basicBar_v1.zip

# 上传 fileupload,拿 fileId
curl -X POST 'http://localhost:9200/api/file/upload' \
  -H 'x-access-token: <TOKEN>' -H 'x-tenant-domain: <TENANT>' \
  -F "file=@.../ai_basicBar_v1.zip"
# result.id 即 fileId

# 创建分类(只需一次)
curl -X POST 'http://localhost:9200/api/visualization/classification' \
  -H 'x-access-token: <TOKEN>' -H 'x-tenant-domain: <TENANT>' \
  -H 'Content-Type: application/json' \
  -d '{"name":"AI 组件","type":"component"}'
# result.id 即 分类id

# 创建 / 更新 resource(PATCH 幂等,group 是 JSON 字符串)
curl -X PATCH 'http://localhost:9200/api/visualization/resource' \
  -H 'x-access-token: <TOKEN>' -H 'x-tenant-domain: <TENANT>' \
  -H 'Content-Type: application/json' \
  -d '[{"name":"AI 基础柱状图","type":"component","resourceId":"ai_basicBar_v1",
       "group":"[\"<分类id>\"]",
       "configuration":{"componentType":"remote","fileId":"<fileId>"},
       "thumbnailUrl":"<fileId>","options":{},"version":0}]'

# 验证设计器查询能拿到
curl -X POST 'http://localhost:9200/api/visualization/resource/tree/_query' \
  -H 'x-access-token: <TOKEN>' -H 'x-tenant-domain: <TENANT>' \
  -H 'Content-Type: application/json' -d '{}'
```

## 3. 加载器(host 侧)

`modules/visualization-resources/utils/ai-component-loader/`:
- `index.ts` — 主入口 `loadAiComponentZip({ fileId, componentName, displayName })` + `ensureAiComponentReady`
  - fileId 走 axios `request.getStream('/file/'+fileId)`(自动加 baseURL + tenant header + token,绕开 CORS)
  - 解压(fflate)→ sfc-loader 编译 component.vue + config.mjs(含其 import 的 Config.vue)
  - 注入 `ResourceBasicComponentsInstance` + 同步 designer-ui `defaultStore` 三个 ref:
    - `componentsInstance[name] = { name, component }`(**必须 `{name,component}` 包装**,画布取 `.component` 渲染)
    - `componentPropsConfig['<name>ConfigProps']`
    - `componentPropsInstance['<name>Config']`
- `shared-modules.ts` — import 白名单(`buildSharedModuleCache`)
- `manifest.ts` — 类型定义

## 4. 按需加载机制(首屏不阻塞)

设计器组件面板**只放元数据,不下载 zip**;hover 预热;drop 兜底:

- `modules/visualization-manager-ui/components/Designer/index.vue`
  - `normalizeAiComponents()`(同步):把后端 `componentType==='remote'` 的 child 的 `type` 改成 `resourceId`,让画布按 key 命中
- `modules/visualization-designer-ui/layout/LeftSider/components/Collapse.vue`
  - 卡片 `@mouseenter="handleAiPrefetch"` → 调 `loadAiComponentZip(...).catch(()=>{})` 静默预热
  - `handleDragStart`:把 `configuration: { componentType, fileId, zipUrl }` 写进拖拽实例;就绪用真实 `ConfigProps`,未就绪用占位
- `modules/visualization-designer-ui/hooks/useDropEvent.ts`
  - `dropCreate` 检查 `copied.configuration?.componentType==='remote' && !componentsInstance[type]`:未就绪 → `await ensureAiComponentReady(configuration.fileId)` → `Object.assign(copied, realProps)`

## 5. 后端生产化工作量

| 项 | 说明 | 量 |
|---|---|---|
| 文件存储 | 复用现有 `/file/upload` | 已有 |
| resource 表 | 不改表结构,只多 `configuration` 两字段 | 0 |
| AI 服务对接 | LLM 按 SKILL.md 约束生成 → 打 zip → 上传 → 写 resource | 0.5–1 人周 |

## 6. 保存恢复时的 AI 组件加载(已实现)

保存的组件实例只有 `type`,**没有 `fileId`**(fileId 在 resource.configuration 里)。重新加载画布时 zip 不会自动加载,需要两处配合:

1. **`ComponentAssemble.vue`** — `componentInstance` 用 `computed(() => defaultStore.componentsInstance[props.info.type])` 取值(不是 `const`),保证 zip 异步注入后**响应式重渲染**。
2. **`Designer/index.vue` `preloadAiComponents(list)`** — 从 `resourceComponentList` 建 `resourceId → { fileId, zipUrl }` 映射,递归扫描画布组件(含 `children`),对 `componentType==='remote'` 且 `componentsInstance` 里还没有的,调 `ensureAiComponentReady` 按需加载。
3. **watch** `[() => designerStore.componentList, () => resourceComponentList.value]`(`immediate`)— 画布数据或资源面板任一就绪都触发,`ensureAiComponentReady` 自带 `loadedZipCache` 去重。

关键不变量:**fileId 不存在组件实例里**,必须运行时从 resource 反查;组件实例 `type` == `resourceId` == 注入 key,三者一致才能反查命中。
