---
name: pptx-generation
description: 当需要生成或修改 PowerPoint(PPTX) 演示文稿、汇报材料、方案页或产品介绍时使用
tools:
  - pptx_generate
tags:
  - pptx
  - powerpoint
  - python-pptx
  - slides
  - presentation
---

# 何时使用

- 用户要“生成 PPT”“输出汇报材料”“做方案演示文稿”“导出 PowerPoint”时使用。

# 入口脚本

- `scripts/generate_ppt.py`
- 内部工具：`pptx_generate`

# 当前脚本特点

- 当前脚本支持两种模式：
  - 无 `--input`：生成内置示例模板（偏“智巡云 SaaS 产品方向规划”）
  - 有 `--input`：按结构化 JSON 生成通用 PPT
- 如果只是验证生成能力，直接运行即可。
- 如果运行时存在内部工具 `pptx_generate`，优先直接调用工具，不必手改脚本。

# 使用方式

- 从仓库根目录运行：
  - `python skills/pptx-generation/scripts/generate_ppt.py`
- 指定输出路径：
  - `python skills/pptx-generation/scripts/generate_ppt.py --output /tmp/solution-deck.pptx`
- 使用通用 JSON 输入：
  - `python skills/pptx-generation/scripts/generate_ppt.py --input @deck.json --output /tmp/solution-deck.pptx`

# 通用输入格式

- `pptx_generate` 和 `--input` 都使用同一种 JSON 结构：

```json
{
  "title": "工业智能巡检方案",
  "subtitle": "面向电力与制造场景",
  "slides": [
    {
      "title": "方案概览",
      "paragraphs": ["本方案聚焦巡检、预警与复判闭环。"],
      "bullets": ["统一接入多类设备", "缺陷自动识别", "结果可追溯"]
    },
    {
      "title": "核心模块",
      "sections": [
        {
          "heading": "巡检执行",
          "bullets": ["任务编排", "现场采集", "实时回传"]
        },
        {
          "heading": "智能分析",
          "bullets": ["模型推理", "缺陷分级", "人工复核"]
        }
      ]
    }
  ],
  "closing_title": "总结",
  "closing_message": "方案以低成本快速落地为目标",
  "closing_bullets": ["先试点后推广", "平台化沉淀能力"]
}
```

# 依赖

- Python 3
- `python-pptx`

# 推荐工作流

1. 先确认是“直接套模板”还是“按结构化数据生成”。
2. 有内部工具时，优先调用 `pptx_generate`。
3. 没有工具时，用 `--input @deck.json` 调脚本。
4. 只有在需要保留旧模板版式时，才修改 `create_presentation()` 里的示例内容。

# 输出与验证

- 默认输出目录：`storage/skill_outputs/pptx-generation/`
- 生成后检查：
  - `.pptx` 文件存在
  - 文件大小大于 0
  - 文件名能反映主题

# 注意事项

- 通用模式下，优先通过 `slides / sections / bullets` 组织内容，而不是硬编码页面。
- 若只是为了导出一次结果，优先传 JSON 和输出文件名，不要无关重构版式代码。
