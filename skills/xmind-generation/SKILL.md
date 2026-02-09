---
name: xmind-generation
description: 使用 xmind 库生成 XMind 思维导图
tags:
  - xmind
  - mindmap
  - planning
  - python
---

适用场景
- 生成“智巡云 SaaS 产品方向规划”思维导图。
- 更新内容后重新导出 XMind 文件。

入口脚本（已内置到仓库）
- `scripts/generate_xmind.py`

使用方式
- 从仓库根目录运行（推荐）：
  - `python skills/xmind-generation/scripts/generate_xmind.py`
- 或进入技能目录运行：
  - `cd skills/xmind-generation && python scripts/generate_xmind.py`
- 可选参数：
  - `--output <path>`：指定输出文件路径；不传则默认输出到 `storage/skill_outputs/xmind-generation/`
- 修改 `create_xmind()` 内的主题与层级。
- 如需基于已有 XMind 文件生成，可使用 `--template <path>` 指定模板（可选）。

依赖
- Python 3
- `xmind`

输出
- 默认输出到 `storage/skill_outputs/xmind-generation/`（可通过 `--output` 覆盖）。

备注
- 不提供 `--template` 时会创建新 XMind；提供 `--template` 时会基于模板生成。
