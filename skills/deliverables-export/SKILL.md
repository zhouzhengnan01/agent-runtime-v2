---
name: deliverables-export
description: 生成「命令调用.txt」与「步骤说明.docx」等交付物（适合发到企业微信/飞书等）
tags:
  - docx
  - txt
  - wework
  - deliverables
  - python
---

适用场景
- 把一段“命令调用说明（纯文本）”和“一段步骤说明（Word）”输出成两个文件，方便作为附件发送。

入口脚本（已内置到仓库）
- `scripts/export_deliverables.py`

使用方式
- 基于内容文件生成（推荐）：
  - `python skills/deliverables-export/scripts/export_deliverables.py --commands-in commands.txt --steps-in steps.md --out-dir <dir>`
- 指定两个输出路径（完全自定义文件名/目录）：
  - `python skills/deliverables-export/scripts/export_deliverables.py --commands-in commands.txt --steps-in steps.md --commands-out <path.txt> --steps-out <path.docx>`

可选参数
- `--out-dir <dir>`：输出目录（会生成 `命令调用.txt` 与 `步骤说明.docx`）
- `--commands-name <name>`：命令文件名（默认 `命令调用.txt`）
- `--steps-name <name>`：步骤文件名（默认 `步骤说明.docx`）

说明
- `steps` 支持简单 Markdown（标题/列表/代码块），会转成基础的 Word 排版。
