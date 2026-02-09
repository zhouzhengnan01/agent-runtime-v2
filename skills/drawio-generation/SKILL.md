---
name: drawio-generation
description: 生成 Draw.io 架构图、流程图与思维导图
tags:
  - drawio
  - diagrams.net
  - architecture
  - flowchart
  - mindmap
  - python
---

适用场景
- 生成产品/系统架构图、流程图、思维导图。
- 输出可在 diagrams.net 编辑的 .drawio 文件与在线链接。

入口脚本（已内置到仓库）
- `scripts/drawio_generator_tool.py`

使用方式
- 脚本包含 `DrawIOGeneratorTool`，可直接调用 `run()` 或 `run_async()`。
- 在本项目内可作为内部工具（`@register_tool("DrawIOGenerator")` 已内置）。
- 也可直接运行脚本生成示例文件：
  - `python skills/drawio-generation/scripts/drawio_generator_tool.py`

示例
```python
from drawio_generator_tool import DrawIOGeneratorTool

tool = DrawIOGeneratorTool()
result = tool.run(
    diagram_type="system_architecture",
    title="My System",
    data={
        "layers": [
            {"name": "Access", "components": [{"name": "Gateway"}]},
            {"name": "Service", "components": [{"name": "API"}, {"name": "Worker"}]}
        ]
    },
    output_path="/tmp/my-system.drawio",
    style="modern",
)
print(result)
```

数据结构
- `product_architecture`: `modules=[{name, description}], connections=[{source, target, label}]`
- `system_architecture`: `layers=[{name, components:[{name}]}]`
- `flowchart`: `steps=[{name, type}]`，`type` 为 `start|process|decision|end`
- `mindmap`: 与 `product_architecture` 相同的数据结构

输出
- `.drawio` 文件保存到 `output_path`，默认输出到 `storage/skill_outputs/drawio-generation/`。
- 结果中包含 `web_url` 可直接打开 diagrams.net。
