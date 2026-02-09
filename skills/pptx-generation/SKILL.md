---
name: pptx-generation
description: 使用 python-pptx 生成 PowerPoint 演示文稿
tags:
  - pptx
  - powerpoint
  - python-pptx
  - slides
  - presentation
---

适用场景
- 生成“智巡云 SaaS 产品方向规划”PPT。
- 修改内容后批量重建演示文稿。

入口脚本（已内置到仓库）
- `scripts/generate_ppt.py`

使用方式
- 从仓库根目录运行（推荐）：
  - `python skills/pptx-generation/scripts/generate_ppt.py`
- 或进入技能目录运行：
  - `cd skills/pptx-generation && python scripts/generate_ppt.py`
- 可选参数：
  - `--output <path>`：指定输出文件路径；不传则默认输出到 `storage/skill_outputs/pptx-generation/`
- 修改 `create_presentation()` 内的文本、模块、样式。

依赖
- Python 3
- `python-pptx`

输出
- 默认输出到 `storage/skill_outputs/pptx-generation/`（可通过 `--output` 覆盖）。
