---
name: excel-generation
description: 基于本地脚本生成 Excel 报价单与工作量表
tags:
  - excel
  - xlsx
  - pandas
  - openpyxl
  - quotation
  - work-statement
---

适用场景
- 生成客户报价单或技术规格书类 Excel。
- 生成项目工作量表、人天报价、人员配置等表格。

入口脚本（已内置到仓库）
- `scripts/create_professional_quotation.py`
- `scripts/create_professional_quotation_v2.py`
- `scripts/create_work_statement_excel.py`
- `scripts/create_work_statement_excel_fixed.py`

使用方式
- 从仓库根目录运行（推荐）：
  - `python skills/excel-generation/scripts/create_professional_quotation.py`
  - `python skills/excel-generation/scripts/create_work_statement_excel_fixed.py`
- 或进入技能目录运行：
  - `cd skills/excel-generation && python scripts/create_professional_quotation.py`
- 可选参数：
  - `--output <path>`：指定输出文件路径；不传则默认输出到 `storage/skill_outputs/excel-generation/`
- 按需修改脚本中的数据列表（项目信息、工作分解、里程碑等）。

依赖
- Python 3
- `pandas`
- `openpyxl`

输出
- XLSX 文件，默认输出到 `storage/skill_outputs/excel-generation/`（可通过 `--output` 覆盖）。

备注
- `create_professional_quotation.py` 采用 openpyxl 做精细样式。
- `create_professional_quotation_v2.py` 以 pandas 多工作表为主。
- `create_work_statement_excel_fixed.py` 为更清晰的结构版本。
