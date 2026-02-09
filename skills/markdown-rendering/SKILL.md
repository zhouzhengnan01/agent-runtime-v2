---
name: markdown-rendering
description: Frontend markdown rendering features and custom blocks
tags:
  - markdown
  - echarts
  - image
  - video
  - prompt
  - mindmap
  - timeline
---

适用场景
- 用户询问前端 Markdown 展示能力/支持的组件
- 需要输出带图表、图片、视频、思维导图、时间轴的 Markdown

能力清单
- 基础 Markdown：标题、列表、表格、引用、粗体/斜体/删除线、链接
- 实时预览、语法高亮、导出 HTML、响应式、多语言

自定义块规范
- ECharts 图表：
  - 使用 `:::echarts` 包裹
  - 内部为 ```json 的完整 JSON
- 图片：
  - 使用 `:::image` 包裹
  - 内部为 ```json，内容为图片 URL 字符串
- 视频：
  - 使用 `:::video` 包裹
  - 内部为 ```json，内容为视频 URL 字符串
- 提示词：
  - 使用 `:::prompt` 包裹
  - 内部为 ```json，内容为数组
- 思维导图：
  - 使用 `:::mindMap` 包裹
  - 内部为 ```json 或结构化 Markdown 文本
- 时间轴：
  - 使用 `:::timeline` 包裹
  - 内部为 ```json，内容为对象数组
  - 推荐字段：`title`、`description`、`image`

输出要求
- JSON 必须使用双引号
- URL 直接写成字符串
- 块标签与代码块必须完整闭合

示例片段
```md
:::echarts
```json
{
  "title": { "text": "示例" },
  "tooltip": {},
  "xAxis": { "data": ["A", "B", "C"] },
  "yAxis": {},
  "series": [{ "type": "bar", "data": [5, 20, 36] }]
}
```
:::

:::image
```json
https://example.com/image.jpg
```
:::

:::video
```json
https://example.com/video.mp4
```
:::

:::prompt
```json
["提示词1", "提示词2"]
```
:::

:::timeline
```json
[
  {
    "title": "阶段1",
    "description": "说明文字",
    "image": "https://example.com/cover.png"
  }
]
```
:::
```
