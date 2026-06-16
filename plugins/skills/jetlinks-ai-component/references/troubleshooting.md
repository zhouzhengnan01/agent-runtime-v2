# 踩坑表

按现象查。# 编号对应 SKILL.md 硬约束。

| # | 现象 | 根因 | 解法 |
|---|------|------|------|
| 1 | 画布上组件是空注释 `<!---->`,架子在但图不出 | `<script setup>` 在 sfc-loader 0.9.5 编译后,顶层 import 的 `VChart` 没暴露给模板,`createVNode(undefined)` → null vnode | **改 Options API** + `components: { VChart }` + setup 显式 return(SKILL 约束 1/2/3) |
| 2 | 配置面板某个分组**空白**(其他分组正常) | 该 `componentProps` key 是 host 没有的全新 key,全局面板字典里没有对应组件 | 在 `config.mjs` 的 `Config` 数组注册自带 `Config.vue`(SKILL「面板机制」) |
| 3 | `'export' may appear only with sourceType: "module"` | 配置文件用了 `.js`,被当 CommonJS | 改 `.mjs` 扩展名(约束 4) |
| 4 | `ref is not defined` / `computed is not defined` | 没显式 import,且无 auto-import | 从 `vue` 显式 import 所有用到的 API |
| 5 | 组件不渲染,无明显报错 | 导出名或 `type` 不符约定,host 自动匹配失败 | `<resourceId>ConfigProps` / `<resourceId>Config` / `ConfigProps.type === resourceId`(约束 5) |
| 6 | 拖入时 `Cannot destructure property 'width'` | `componentProps.style` 缺字段 | 补全 `style: { x, y, width, height }`(约束 6) |
| 7 | `[ai-component] 警告:import 了非白名单依赖 "xxx"` + 模块找不到 | import 了白名单外的 bare 依赖 | 改用白名单内依赖,或在 host `shared-modules.ts` 扩白名单(约束 7) |
| 8 | `zip 内未找到文件: less` 之类样式报错 | sfc-loader 无 less 编译器 | `<style lang="less">` 里只写纯 CSS(约束 8) |
| 9 | fetch zip 报 403(OPTIONS preflight) | 自己 fetch 加 header 触发跨域预检 | 用 axios `request.getStream('/file/'+fileId)`,自动带 baseURL+tenant,同源 proxy 绕开 |
| 10 | 注入了组件但画布还是空 | `componentsInstance[type]` 直接给了 component 本身 | 必须包装成 `{ name, component }`,画布 `ComponentAssemble` 取 `.component` 渲染 |
| 11 | saaS 环境下 componentName 取不到组件 | saaS 路径把 `child.type` 改写成 `'component'` | 用 `resourceId` 作 componentName,不要用 `type` |
| 12 | 刷新/重新加载后,已保存的 AI 组件变 `<!---->` | ① 保存的实例只有 `type` 没 `fileId`,恢复时不会重新加载 zip ② `ComponentAssemble` 取值非响应式,注入后不重渲染 | ✅已修:`ComponentAssemble` 改 `computed` 取值 + `Designer` 的 `preloadAiComponents` 从 `resourceComponentList` 反查 fileId 预加载(详见 integration.md §6) |

## sfc-loader 关键约定(踩过的)

- `handleModule` 不处理某类型时必须 `return undefined`(不是 `null`)—— sfc-loader 源码用 `=== undefined` 判定是否走内置处理
- `getFile` 要做扩展名 fallback(`.js`↔`.mjs`↔`.ts`),因为 import 路径可能省略扩展名
- config.mjs 里 `import './Config.vue'` 会触发 sfc-loader 对 Config.vue 的递归编译,getFile 从 zip 取 —— **这是 Config.vue 能用的原理**
- Config.vue 里 `moduleRegistry.getResource('visualization-designer-ui', 'components')` 在模块顶层执行,此时设计器已加载,能拿到 ConfigItem/InputNumber/ColorPicker 等;`a-switch` 等 antd 全局组件在 sfc-loader 编译的组件里可正常 resolve(继承主 app context)
