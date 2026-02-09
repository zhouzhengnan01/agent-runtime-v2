#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_path() -> Path:
    out_dir = _project_root() / "storage" / "skill_outputs" / "excel-generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "技术规格书及人天报价单.xlsx"


def create_haina_work_statement():
    """创建海纳设备接入系统工作量表"""

    # 工作分解结构
    work_items = [
        # 需求分析阶段
        ["1.0", "需求分析阶段", "", "", "", "", 15, 15000],
        ["1.1", "需求调研与分析", "业务需求调研、用户访谈", "需求分析师", "10人天", 1, 10000],
        ["1.2", "技术需求分析", "技术架构分析、集成需求分析", "架构师", "3人天", 1, 3000],
        ["1.3", "需求规格书编写", "编写详细需求文档", "需求分析师", "2人天", 1, 2000],

        # 系统设计阶段
        ["2.0", "系统设计阶段", "", "", "", "", 25, 25000],
        ["2.1", "系统架构设计", "总体架构设计、技术选型", "架构师", "5人天", 1, 5000],
        ["2.2", "详细设计", "数据库设计、接口设计", "系统设计师", "15人天", 1, 15000],
        ["2.3", "UI/UX设计", "界面原型设计、用户体验设计", "UI设计师", "5人天", 1, 5000],

        # 开发实施阶段
        ["3.0", "开发实施阶段", "", "", "", "", 80, 80000],
        ["3.1", "后端开发", "API开发、业务逻辑实现", "后端工程师", "40人天", 1.5, 60000],
        ["3.2", "前端开发", "用户界面开发", "前端工程师", "25人天", 1, 25000],
        ["3.3", "数据库开发", "数据库建模、优化", "数据库工程师", "10人天", 1, 10000],
        ["3.4", "集成开发", "与统一身份认证、应用云平台集成", "集成工程师", "5人天", 1, 5000],

        # 测试阶段
        ["4.0", "测试阶段", "", "", "", "", 30, 30000],
        ["4.1", "功能测试", "功能验证、业务流程测试", "测试工程师", "15人天", 1, 15000],
        ["4.2", "性能测试", "性能测试、压力测试", "测试工程师", "8人天", 1, 8000],
        ["4.3", "安全测试", "安全漏洞扫描、渗透测试", "安全工程师", "7人天", 1, 7000],

        # 部署上线阶段
        ["5.0", "部署上线阶段", "", "", "", "", 20, 20000],
        ["5.1", "环境准备", "开发、测试、生产环境搭建", "运维工程师", "10人天", 1, 10000],
        ["5.2", "系统部署", "系统安装、配置、部署", "运维工程师", "5人天", 1, 5000],
        ["5.3", "上线支持", "上线支持、问题排查", "运维工程师", "5人天", 1, 5000],

        # 培训移交阶段
        ["6.0", "培训移交阶段", "", "", "", "", 15, 15000],
        ["6.1", "用户培训", "系统使用培训", "培训师", "5人天", 1, 5000],
        ["6.2", "管理员培训", "系统管理培训", "培训师", "3人天", 1, 3000],
        ["6.3", "文档移交", "技术文档、用户手册编写", "技术文档工程师", "7人天", 1, 7000]
    ]

    # 创建DataFrame
    columns = [
        "序号", "工作包", "工作内容描述", "角色", "人天", "单价(元/人天)", "小计(元)"
    ]
    df = pd.DataFrame(work_items, columns=columns)

    # 计算小计
    df['小计(元)'] = df['人天'] * df['单价(元/人天)']

    # 添加合计行
    summary_row = ["合计", "", "", "", df['人天'].sum(), "", df['小计(元)'].sum()]
    df_summary = pd.DataFrame([summary_row], columns=columns)

    # 添加项目信息
    project_info = pd.DataFrame([
        ["项目名称", "海纳设备接入系统定制开发项目"],
        ["合同金额", "49万元"],
        ["工期", "4-5个月"],
        ["项目经理", "待定"],
        ["技术负责人", "待定"]
    ], columns=["信息项", "内容"])

    return df, df_summary, project_info

def create_fire_control_work_statement():
    """创建火调技术服务工作量表"""

    # 工作分解结构
    work_items = [
        # 需求调研阶段
        ["1.0", "需求调研阶段", "", "", "", "", 20, 20000],
        ["1.1", "现场调研", "现有系统调研、业务流程分析", "需求分析师", "10人天", 1, 10000],
        ["1.2", "需求分析", "业务需求分析、技术需求分析", "业务分析师", "8人天", 1, 8000],
        ["1.3", "方案设计", "技术方案设计、实施方案设计", "架构师", "2人天", 1, 2000],

        # 系统开发阶段
        ["2.0", "系统开发阶段", "", "", "", "", 60, 60000],
        ["2.1", "模块开发", "核心功能模块开发", "软件工程师", "40人天", 1, 40000],
        ["2.2", "接口开发", "系统接口开发", "接口开发工程师", "15人天", 1, 15000],
        ["2.3", "数据迁移", "现有数据迁移、转换", "数据工程师", "5人天", 1, 5000],

        # 系统测试阶段
        ["3.0", "系统测试阶段", "", "", "", "", 25, 25000],
        ["3.1", "功能测试", "系统功能测试、业务流程测试", "测试工程师", "15人天", 1, 15000],
        ["3.2", "集成测试", "系统集成测试、接口测试", "测试工程师", "10人天", 1, 10000],

        # 部署实施阶段
        ["4.0", "部署实施阶段", "", "", "", "", 30, 30000],
        ["4.1", "系统部署", "生产环境部署、配置", "实施工程师", "15人天", 1, 15000],
        ["4.2", "数据初始化", "基础数据初始化、配置", "数据工程师", "10人天", 1, 10000],
        ["4.3", "上线支持", "上线支持、问题处理", "运维工程师", "5人天", 1, 5000],

        # 培训移交阶段
        ["5.0", "培训移交阶段", "", "", "", "", 15, 15000],
        ["5.1", "技术培训", "技术人员培训", "培训师", "8人天", 1, 8000],
        ["5.2", "用户培训", "最终用户培训", "培训师", "5人天", 1, 5000],
        ["5.3", "文档编写", "技术文档、用户手册", "技术文档工程师", "2人天", 1, 2000]
    ]

    # 创建DataFrame
    columns = [
        "序号", "工作包", "工作内容描述", "角色", "人天", "单价(元/人天)", "小计(元)"
    ]
    df = pd.DataFrame(work_items, columns=columns)

    # 计算小计
    df['小计(元)'] = df['人天'] * df['单价(元/人天)']

    # 添加合计行
    summary_row = ["合计", "", "", "", df['人天'].sum(), "", df['小计(元)'].sum()]
    df_summary = pd.DataFrame([summary_row], columns=columns)

    # 添加项目信息
    project_info = pd.DataFrame([
        ["项目名称", "火调技术服务项目"],
        ["合同金额", "24万元"],
        ["工期", "3-4个月"],
        ["项目经理", "待定"],
        ["技术负责人", "待定"]
    ], columns=["信息项", "内容"])

    return df, df_summary, project_info

def save_to_excel_with_multiple_sheets(output_path: str | os.PathLike | None = None):
    """保存多个工作表到Excel文件"""

    # 创建Excel写入器
    resolved_path = (
        Path(output_path).expanduser().resolve() if output_path else _default_output_path()
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(resolved_path), engine='openpyxl') as writer:

        # 海纳设备接入系统 - 工作量表
        df_haina, summary_haina, info_haina = create_haina_work_statement()

        # 写入项目信息
        info_haina.to_excel(writer, sheet_name='海纳-项目信息', index=False)

        # 写入工作量表
        df_haina.to_excel(writer, sheet_name='海纳-工作量表', index=False)

        # 写入合计
        summary_haina.to_excel(writer, sheet_name='海纳-工作量表', startrow=len(df_haina)+2, index=False)

        # 火调技术服务 - 工作量表
        df_fire, summary_fire, info_fire = create_fire_control_work_statement()

        # 写入项目信息
        info_fire.to_excel(writer, sheet_name='火调-项目信息', index=False)

        # 写入工作量表
        df_fire.to_excel(writer, sheet_name='火调-工作量表', index=False)

        # 写入合计
        summary_fire.to_excel(writer, sheet_name='火调-工作量表', startrow=len(df_fire)+2, index=False)

        # 人员单价参考表
        rate_info = pd.DataFrame([
            ["角色", "标准单价(元/人天)", "高级单价(元/人天)"],
            ["项目经理", "2000", "3000"],
            ["架构师", "2000", "3000"],
            ["系统设计师", "1500", "2000"],
            ["后端工程师", "1500", "2000"],
            ["前端工程师", "1500", "2000"],
            ["数据库工程师", "1500", "2000"],
            ["测试工程师", "1200", "1500"],
            ["安全工程师", "2000", "2500"],
            ["运维工程师", "1500", "2000"],
            ["集成工程师", "1500", "2000"],
            ["UI设计师", "1200", "1500"],
            ["培训师", "1000", "1500"],
            ["需求分析师", "1500", "2000"],
            ["业务分析师", "1200", "1500"],
            ["数据工程师", "1500", "2000"],
            ["技术文档工程师", "1000", "1500"]
        ])

        rate_info.to_excel(writer, sheet_name='人员单价参考', index=False)

        # 项目阶段说明
        phase_description = pd.DataFrame([
            ["阶段", "主要工作内容", "交付物", "预计工期"],
            ["需求分析", "需求调研、分析、文档编写", "需求规格说明书", "2-3周"],
            ["系统设计", "架构设计、详细设计、UI设计", "设计文档、原型图", "3-4周"],
            ["开发实施", "功能开发、集成开发", "可运行的系统", "8-12周"],
            ["测试", "功能测试、性能测试、安全测试", "测试报告", "3-4周"],
            ["部署上线", "环境准备、系统部署、上线支持", "生产系统", "2-3周"],
            ["培训移交", "用户培训、管理员培训、文档移交", "培训材料、技术文档", "2-3周"]
        ])

        phase_description.to_excel(writer, sheet_name="项目阶段说明", index=False)

        # 质量保证说明
        quality_info = pd.DataFrame([
            ["质量保证项", "具体措施"],
            ["代码质量", "代码审查、单元测试覆盖率≥80%"],
            ["文档质量", "文档标准化审查、版本控制"],
            ["测试质量", "测试用例覆盖率≥90%、自动化测试"],
            ["交付质量", "阶段性验收、最终验收"],
            ["维护质量", "1年免费维护、7×24小时技术支持"]
        ])

        quality_info.to_excel(writer, sheet_name="质量保证", index=False)

    return str(resolved_path)

def main():
    """主函数"""
    try:
        parser = argparse.ArgumentParser(description="生成技术规格书及人天报价单（Excel）")
        parser.add_argument(
            "--output",
            help="输出文件路径（.xlsx）；不传则输出到 storage/skill_outputs/excel-generation/",
        )
        args = parser.parse_args()

        # 生成Excel文件
        output_path = save_to_excel_with_multiple_sheets(args.output)

        # 检查文件大小
        file_size = os.path.getsize(output_path)

        print(f"✅ 技术规格书及人天报价单已生成: {output_path}")
        print(f"📄 文件大小: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print("")
        print("📋 包含以下工作表:")
        print("  • 海纳-项目信息")
        print("  • 海纳-工作量表")
        print("  • 火调-项目信息")
        print("  • 火调-工作量表")
        print("  • 人员单价参考")
        print("  • 项目阶段说明")
        print("  • 质量保证")
        print("")
        print("💡 您可以使用Excel打开文件进行进一步编辑和调整")

        return output_path

    except Exception as e:
        print(f"❌ 生成Excel文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
