---
name: excel-generation
description: 当需要生成或修改 Excel 报价单、工作量表、人天估算、里程碑或技术规格工作簿时使用
tools:
  - excel_generate
tags:
  - excel
  - xlsx
  - pandas
  - openpyxl
  - quotation
  - work-statement
---

# 何时使用

- 用户要“生成报价单”“导出 Excel”“做工作量/人天估算表”“做项目分解表”时使用。
- 输出目标是 `.xlsx` 工作簿，而不是 Markdown 表格。

# 入口脚本

- 通用脚本：`scripts/generate_excel.py`
- 内部工具：`excel_generate`
- 兼容模板脚本：
- `scripts/create_professional_quotation.py`
- `scripts/create_professional_quotation_v2.py`
- `scripts/create_work_statement_excel.py`
- `scripts/create_work_statement_excel_fixed.py`

# 选型建议

- `generate_excel.py` / `excel_generate`
  - 通用优先；适合按结构化 JSON 直接生成多工作表 Excel
- `create_professional_quotation.py`
  - 偏精细样式，适合正式报价单
- `create_professional_quotation_v2.py`
  - 偏多工作表与数据整理，适合快速生成
- `create_work_statement_excel.py`
  - 工作说明与人天报价基础版
- `create_work_statement_excel_fixed.py`
  - 结构更清晰，优先作为工作量表默认方案

# 推荐工作流

1. 先判断是“按结构化数据生成”还是“沿用历史模板”。
2. 通用场景优先调用 `excel_generate`，或运行 `generate_excel.py --input @workbook.json`。
3. 只有在需要延续既有报价单/工作量表版式时，才使用 legacy 模板脚本。
4. 修改 legacy 模板时，优先替换数据区块，不要大幅改动排版逻辑。

# 使用方式

- 生成示例工作簿：
  - `python skills/excel-generation/scripts/generate_excel.py`
- 使用通用 JSON 输入：
  - `python skills/excel-generation/scripts/generate_excel.py --input @workbook.json --output /tmp/workbook.xlsx`
- 从仓库根目录运行：
  - `python skills/excel-generation/scripts/create_professional_quotation.py`
  - `python skills/excel-generation/scripts/create_professional_quotation_v2.py`
  - `python skills/excel-generation/scripts/create_work_statement_excel.py`
  - `python skills/excel-generation/scripts/create_work_statement_excel_fixed.py`
- 指定输出路径：
  - `python skills/excel-generation/scripts/create_work_statement_excel_fixed.py --output /tmp/work-statement.xlsx`

# 依赖

- Python 3
- `openpyxl`
- `pandas`（仅 legacy 模板脚本需要）

# 通用输入格式

- `excel_generate` 和 `generate_excel.py --input` 使用同一种 JSON 结构：

```json
{
  "title": "工业智能巡检报价与计划",
  "sheets": [
    {
      "name": "报价明细",
      "title": "报价明细",
      "columns": [
        { "key": "module", "title": "模块", "width": 18 },
        { "key": "desc", "title": "说明", "width": 32 },
        { "key": "amount", "title": "金额(元)", "width": 14, "number_format": "#,##0", "align": "right" }
      ],
      "rows": [
        { "module": "平台搭建", "desc": "基础环境与权限配置", "amount": 30000 },
        { "module": "模型服务", "desc": "识别与复判能力接入", "amount": 50000 }
      ],
      "summary_rows": [
        { "module": "合计", "desc": "", "amount": 80000 }
      ],
      "freeze_panes": "A3"
    }
  ]
}
```

# 输出与验证

- 默认输出目录：`storage/skill_outputs/excel-generation/`
- 生成后检查：
  - `.xlsx` 文件存在
  - 文件大小大于 0
  - 文件名与业务语义一致，例如“报价单”“工作量表”“人天估算”

# 注意事项

- 通用模式下，优先通过 `sheets / columns / rows / summary_rows` 组织内容，不要把业务数据硬编码进脚本。
- 若用户给了明确业务数据，优先走 `excel_generate` 或 `generate_excel.py --input`。
- 只有在要复用历史样式时，才修改 legacy 模板脚本。
