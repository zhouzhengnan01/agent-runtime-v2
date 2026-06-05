---
name: xmind-generation
description: 当需要生成或修改 XMind 思维导图、方案结构图、功能拆解或路线图时使用
tools:
  - xmind_generate
tags:
  - xmind
  - mindmap
  - planning
  - python
---

# 何时使用

- 用户要“生成思维导图”“导出 XMind”“做方案拆解/功能结构/路线图”时使用。

# 入口脚本

- `scripts/generate_xmind.py`
- 内部工具：`xmind_generate`

# 当前脚本特点

- 当前脚本支持两种模式：
  - 无 `--input`：生成内置示例模板（偏“智巡云 SaaS 产品方向规划”）
  - 有 `--input`：按结构化 JSON 生成通用 XMind
- 适合基于现有树结构快速改主题、模块和层级。
- 如果运行时存在内部工具 `xmind_generate`，优先直接调用工具，不必手改脚本。

# 使用方式

- 直接生成：
  - `python skills/xmind-generation/scripts/generate_xmind.py`
- 指定输出路径：
  - `python skills/xmind-generation/scripts/generate_xmind.py --output /tmp/roadmap.xmind`
- 基于模板生成：
  - `python skills/xmind-generation/scripts/generate_xmind.py --template /path/to/base.xmind --output /tmp/custom.xmind`
- 使用通用 JSON 输入：
  - `python skills/xmind-generation/scripts/generate_xmind.py --input @mindmap.json --output /tmp/roadmap.xmind`

# 通用输入格式

- `xmind_generate` 和 `--input` 都使用同一种 JSON 结构：

```json
{
  "title": "工业智能巡检方案",
  "sheet_title": "工业智能巡检方案",
  "topics": [
    {
      "title": "方案概览",
      "children": [
        { "title": "设备接入" },
        { "title": "缺陷识别" },
        { "title": "结果复判" }
      ]
    },
    {
      "title": "实施路径",
      "children": [
        {
          "title": "第一阶段",
          "children": [
            { "title": "试点" },
            { "title": "验证" }
          ]
        },
        {
          "title": "第二阶段",
          "children": [
            { "title": "扩面" },
            { "title": "平台化" }
          ]
        }
      ]
    }
  ]
}
```

# 依赖

- Python 3
- `xmind`
- 当前仓库根 `requirements.txt` 未显式列出 `xmind`，运行前需确认环境已安装

# 推荐工作流

1. 先确认是“直接套模板”还是“按结构化树生成”。
2. 有内部工具时，优先调用 `xmind_generate`。
3. 没有工具时，用 `--input @mindmap.json` 调脚本。
4. 只有在需要保留旧模板结构时，才修改 `create_xmind()` 里的示例主题树。

# 输出与验证

- 默认输出目录：`storage/skill_outputs/xmind-generation/`
- 生成后检查：
  - `.xmind` 文件存在
  - 文件大小大于 0
  - 文件名能反映主题

# 注意事项

- 不提供 `--template` 时会创建/覆盖目标输出文件。
- 若环境缺少 `xmind` 依赖，脚本会直接退出，需要先补装依赖。
- 通用模式下，优先通过 `topics[].children[]` 的树结构组织内容，而不是硬编码主题节点。
