---
name: deliverables-export
description: 当需要把命令说明、操作步骤、实施方案或交付说明导出为可发送附件时使用；生成 TXT 与 DOCX 交付物
tools:
  - deliverables_export
tags:
  - docx
  - txt
  - wework
  - deliverables
  - python
---

# 何时使用

- 用户要“导出交付物”“生成附件”“给企业微信/飞书发操作说明”时使用。
- 适合把两类内容落盘：
  - 命令调用说明：纯文本，输出为 `.txt`
  - 操作步骤/实施说明：支持简单 Markdown，输出为 `.docx`

# 入口脚本

- `scripts/export_deliverables.py`
- 内部工具：`deliverables_export`

# 输入规则

- `--commands-in` 与 `--steps-in` 都支持 3 种输入形式：
  - 文件路径，如 `commands.txt`
  - `@文件路径`，如 `@steps.md`
  - 直接内联文本，适合短内容
- `steps` 支持简单 Markdown：标题、编号列表、无序列表、代码块。

# 推荐工作流

1. 先整理两段内容：命令说明 + 步骤说明。
2. 有内部工具时，优先调用 `deliverables_export`。
3. 没有工具时，使用 `--out-dir` 让脚本一次生成两份文件。
4. 生成后返回两个输出路径，并说明各文件用途。

# 使用方式

- 推荐：
  - `python skills/deliverables-export/scripts/export_deliverables.py --commands-in commands.txt --steps-in steps.md --out-dir <dir>`
- 完全自定义输出文件名：
  - `python skills/deliverables-export/scripts/export_deliverables.py --commands-in commands.txt --steps-in steps.md --commands-out <path.txt> --steps-out <path.docx>`
- 直接传短文本：
  - `python skills/deliverables-export/scripts/export_deliverables.py --commands-in "docker compose up -d" --steps-in "# 部署步骤\n1. 拉代码\n2. 启服务" --out-dir <dir>`

# 关键参数

- `--out-dir <dir>`：输出目录，会生成 `命令调用.txt` 与 `步骤说明.docx`
- `--commands-name <name>`：命令文件名，默认 `命令调用.txt`
- `--steps-name <name>`：步骤文件名，默认 `步骤说明.docx`
- `--commands-out <path>` / `--steps-out <path>`：分别指定两个输出文件

# 工具参数

- `commands`：命令调用说明内容，输出为 TXT
- `steps`：步骤说明内容，支持简单 Markdown，输出为 DOCX
- `out_dir`：输出目录，不传则输出到 `storage/skill_outputs/deliverables-export/`
- `commands_name`：命令文件名，默认 `命令调用.txt`
- `steps_name`：步骤文件名，默认 `步骤说明.docx`

# 验证与返回

- 检查 `.txt` 与 `.docx` 是否都已生成且大小大于 0。
- 回复时给出最终文件路径，并简述内容来源。
- 若用户要求发附件，优先生成中文文件名，方便直接发送。
