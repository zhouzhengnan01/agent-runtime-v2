---
name: drawio-generation
description: 当需要生成或修改可在 diagrams.net 或 draw.io 中继续编辑的系统架构图、产品架构图、流程图或思维导图时使用
tags:
  - drawio
  - diagrams.net
  - architecture
  - flowchart
  - mindmap
  - python
---

# 何时使用

- 用户需要“画架构图”“导出 draw.io”“生成 diagrams.net 文件”“做流程图/思维导图”时使用。
- 目标输出是可编辑的 `.drawio` 文件，而不是静态图片。

# 入口脚本

- `scripts/drawio_generator_tool.py`

# 选择图类型

- `product_architecture`：产品/模块架构
- `system_architecture`：分层系统架构
- `flowchart`：流程图
- `mindmap`：思维导图

# 数据结构

- `product_architecture`
  - `modules=[{name, description}]`
  - `connections=[{source, target, label}]`
- `system_architecture`
  - `layers=[{name, components:[{name}]}]`
- `flowchart`
  - `steps=[{name, type}]`
  - `type` 取值：`start | process | decision | end`
- `mindmap`
  - 可复用 `product_architecture` 风格的数据结构

# 推荐工作流

1. 先把用户需求整理成结构化 JSON。
2. 选择 `diagram_type` 和 `style`。
3. 用脚本生成 `.drawio` 文件。
4. 返回文件路径；若结果里有 `web_url`，一并返回。

# 使用方式

- 直接运行示例：
  - `python skills/drawio-generation/scripts/drawio_generator_tool.py`
- 使用 JSON 文件输入：
  - `python skills/drawio-generation/scripts/drawio_generator_tool.py --diagram-type system_architecture --title "设备接入架构" --data @data.json --style modern --output /tmp/device-arch.drawio`
- 使用内联 JSON：
  - `python skills/drawio-generation/scripts/drawio_generator_tool.py --diagram-type flowchart --title "发布流程" --data '{"steps":[{"name":"开始","type":"start"},{"name":"审批","type":"decision"},{"name":"发布","type":"process"},{"name":"结束","type":"end"}]}'`

# 代码调用

- 脚本内置 `DrawIOGeneratorTool`，可直接调用 `run()` 或 `run_async()`。
- 在本项目内它也是已注册的内部工具：`DrawIOGenerator`。

# 输出与验证

- 默认输出目录：`storage/skill_outputs/drawio-generation/`
- 生成后检查：
  - `.drawio` 文件存在
  - 返回结果中的 `success` 为真
  - 如有 `web_url`，可直接提供给用户在 diagrams.net 中打开

# 注意事项

- 如果用户给的是自然语言需求，先整理成结构化 `data`，再调用脚本。
- 优先保留中文标题，输出文件名会自动做安全处理。
