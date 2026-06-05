---
name: markdown-rendering
description: 当用户询问前端 Markdown 渲染能力，或需要输出可被前端自定义块渲染的 Markdown 内容时使用
tags:
  - markdown
  - echarts
  - image
  - video
  - prompt
  - mindmap
  - timeline
---

# 何时使用

- 用户问“前端 Markdown 支持什么”“图表/视频/思维导图怎么写”时使用。
- 用户需要一段可直接喂给前端渲染的 Markdown 内容时使用。

# 能力范围

- 基础 Markdown：标题、列表、表格、引用、粗体、斜体、删除线、链接
- 扩展能力：ECharts、图片、视频、提示词、思维导图、时间轴

# 自定义块规范

- `:::echarts`
  - 内部放 `json` 代码块，内容为完整 ECharts option
- `:::image`
  - 内部放 `json` 代码块，内容为图片 URL 字符串
- `:::video`
  - 内部放 `json` 代码块，内容为视频 URL 字符串
- `:::prompt`
  - 内部放 `json` 代码块，内容为字符串数组
- `:::mindMap`
  - 内部可放 `json` 或结构化 Markdown 文本
- `:::timeline`
  - 内部放 `json` 代码块，内容为对象数组
  - 推荐字段：`title`、`description`、`image`

# 输出要求

- JSON 必须使用双引号。
- URL 直接写成字符串，不要包成对象。
- 块标签与代码块必须完整闭合。
- 若用户要“可直接渲染的 Markdown”，尽量直接返回 Markdown 本体，不要夹杂多余解释。

# 示例片段

````md
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
"https://example.com/image.jpg"
```
:::

:::video
```json
"https://example.com/video.mp4"
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
````

# 使用建议

- 纯能力说明场景：优先列出支持的块类型和示例。
- 实际内容生成场景：根据用户目标直接产出最终 Markdown。
- 如果用户要图表/时间轴，先确认数据结构，再生成对应 JSON。
