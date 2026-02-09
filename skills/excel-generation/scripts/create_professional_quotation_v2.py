#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import os
import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_path() -> Path:
    out_dir = _project_root() / "storage" / "skill_outputs" / "excel-generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "海纳设备接入系统-专业报价单.xlsx"


def create_professional_style_excel(output_path: str | os.PathLike | None = None):
    """创建专业样式的Excel报价单"""

    resolved_path = (
        Path(output_path).expanduser().resolve() if output_path else _default_output_path()
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(resolved_path), engine='openpyxl') as writer:

        # 1. 项目概览
        project_overview_data = [
            ['项目名称', '海纳设备接入系统定制开发项目'],
            ['合同金额', '490,000元'],
            ['项目周期', '4-5个月'],
            ['启动时间', '合同签订后1周内'],
            ['项目经理', '待定'],
            ['技术总监', '待定'],
            ['质量保证', '1年免费维护'],
            [],
            ['项目范围'],
            ['• 需求分析与系统设计'],
            ['• 系统开发与集成'],
            ['• 测试与质量保证'],
            ['• 部署上线与培训'],
            ['• 技术支持与维护']
        ]

        df_overview = pd.DataFrame(project_overview_data, columns=['项目信息', '详情'])
        df_overview.to_excel(writer, sheet_name='项目概览', index=False)

        # 2. 详细工作分解与报价
        work_breakdown_data = [
            ['阶段', '工作内容', '技术要求', '人天', '单价(元/天)', '小计(元)', '交付物'],

            # 需求分析阶段
            ['需求分析阶段', '', '', '', '', '', ''],
            ['1.1', '需求调研', '业务需求调研、用户访谈', 10, 1000, 10000, '需求调研报告'],
            ['1.2', '技术需求分析', '技术架构分析、集成方案', 5, 1000, 5000, '技术需求文档'],
            ['1.3', '需求规格书', '详细需求规格编写', 5, 1000, 5000, '需求规格说明书'],
            ['', '阶段小计', '', 20, '', 20000, ''],

            # 系统设计阶段
            ['系统设计阶段', '', '', '', '', '', ''],
            ['2.1', '架构设计', '系统架构设计、技术选型', 8, 1500, 12000, '架构设计文档'],
            ['2.2', '详细设计', '数据库设计、接口设计', 15, 1200, 18000, '详细设计文档'],
            ['2.3', 'UI/UX设计', '用户界面设计、原型', 7, 1000, 7000, 'UI原型图'],
            ['', '阶段小计', '', 30, '', 37000, ''],

            # 开发实施阶段
            ['开发实施阶段', '', '', '', '', '', ''],
            ['3.1', '后端开发', 'API开发、业务逻辑', 40, 1500, 60000, '后端服务'],
            ['3.2', '前端开发', '用户界面开发', 25, 1200, 30000, '前端应用'],
            ['3.3', '数据库开发', '数据库建模、优化', 10, 1200, 12000, '数据库脚本'],
            ['3.4', '集成开发', '第三方系统集成', 8, 1200, 9600, '集成接口'],
            ['', '阶段小计', '', 83, '', 111600, ''],

            # 测试阶段
            ['测试阶段', '', '', '', '', '', ''],
            ['4.1', '功能测试', '功能测试、用例编写', 15, 1000, 15000, '测试用例'],
            ['4.2', '性能测试', '性能测试、压力测试', 8, 1200, 9600, '性能测试报告'],
            ['4.3', '安全测试', '安全扫描、渗透测试', 7, 1500, 10500, '安全测试报告'],
            ['', '阶段小计', '', 30, '', 35100, ''],

            # 部署上线
            ['部署上线阶段', '', '', '', '', '', ''],
            ['5.1', '环境准备', '开发、测试、生产环境', 10, 1200, 12000, '环境配置文档'],
            ['5.2', '系统部署', '系统安装、配置、上线', 8, 1200, 9600, '部署文档'],
            ['5.3', '上线支持', '上线技术支持', 5, 1000, 5000, '上线报告'],
            ['', '阶段小计', '', 23, '', 26600, ''],

            # 培训移交
            ['培训移交阶段', '', '', '', '', '', ''],
            ['6.1', '用户培训', '系统使用培训', 5, 1000, 5000, '培训材料'],
            ['6.2', '管理员培训', '系统管理培训', 3, 1000, 3000, '管理员手册'],
            ['6.3', '文档移交', '技术文档、用户手册', 7, 1000, 7000, '完整技术文档'],
            ['', '阶段小计', '', 15, '', 15000, ''],

            # 项目总计
            ['项目总计', '', '', 201, '', 245300, '']
        ]

        df_work = pd.DataFrame(work_breakdown_data[1:], columns=work_breakdown_data[0])
        df_work.to_excel(writer, sheet_name='工作明细报价', index=False)

        # 3. 人员配置计划
        staff_plan_data = [
            ['阶段', '项目经理', '架构师', '后端工程师', '前端工程师', '测试工程师', '其他人员', '合计人天'],
            ['需求分析', 1, 1, 0, 0, 0, 1, 20],
            ['系统设计', 1, 1, 1, 1, 0, 2, 30],
            ['开发实施', 1, 1, 3, 2, 1, 4, 83],
            ['测试阶段', 1, 0, 1, 1, 2, 2, 30],
            ['部署上线', 1, 0, 1, 0, 1, 2, 23],
            ['培训移交', 1, 0, 0, 0, 0, 3, 15],
            ['总计', 6, 3, 6, 4, 4, 14, 201]
        ]

        df_staff = pd.DataFrame(staff_plan_data[1:], columns=staff_plan_data[0])
        df_staff.to_excel(writer, sheet_name='人员配置', index=False)

        # 4. 项目里程碑
        milestone_data = [
            ['里程碑', '完成时间', '主要交付物', '验收标准', '负责人'],
            ['M1: 需求确认', '第2周末', '需求规格说明书', '客户签字确认', '项目经理'],
            ['M2: 设计完成', '第6周末', '设计文档、原型图', '内部评审通过', '架构师'],
            ['M3: 开发完成', '第14周末', '可运行的系统', '功能测试通过', '技术总监'],
            ['M4: 测试完成', '第18周末', '测试报告', '验收测试通过', '测试负责人'],
            ['M5: 部署完成', '第20周末', '生产系统', '系统稳定运行', '运维负责人'],
            ['M6: 项目验收', '第22周末', '完整项目文档', '客户最终验收', '项目经理']
        ]

        df_milestone = pd.DataFrame(milestone_data[1:], columns=milestone_data[0])
        df_milestone.to_excel(writer, sheet_name='项目里程碑', index=False)

        # 5. 质量保证措施
        quality_data = [
            ['质量维度', '保证措施', '完成标准', '责任人'],
            ['需求质量', '需求评审、需求确认', '需求规格书评审通过', '需求分析师'],
            ['设计质量', '设计评审、同行评审', '设计文档评审通过', '架构师'],
            ['代码质量', '代码审查、单元测试', '测试覆盖率≥80%', '技术负责人'],
            ['测试质量', '完整测试流程', '所有测试用例通过', '测试负责人'],
            ['文档质量', '文档标准化', '文档审查通过', '技术文档工程师'],
            ['交付质量', '分阶段验收', '客户签字确认', '项目经理']
        ]

        df_quality = pd.DataFrame(quality_data[1:], columns=quality_data[0])
        df_quality.to_excel(writer, sheet_name='质量保证', index=False)

        # 6. 服务与支持
        service_data = [
            ['服务类型', '服务内容', '服务期限', '响应时间'],
            ['免费维护期', '系统bug修复、小版本升级', '验收后1年', '7×24小时响应'],
            ['技术支持', '技术咨询、故障排查', '长期', '工作日2小时内响应'],
            ['培训服务', '使用培训、管理培训', '项目期间', '按计划安排'],
            ['文档支持', '技术文档更新', '长期', '及时更新']
        ]

        df_service = pd.DataFrame(service_data[1:], columns=service_data[0])
        df_service.to_excel(writer, sheet_name='服务支持', index=False)

        # 7. 公司联系信息
        contact_data = [
            ['公司名称', '重庆浩鲸科技有限公司'],
            ['联系地址', '重庆市渝北区'],
            ['联系电话', '400-888-8888'],
            ['技术支持', 'tech@example.com'],
            ['商务合作', 'business@example.com'],
            [],
            ['专业团队'],
            ['✓ 10+年软件开发经验'],
            ['✓ 资深技术团队'],
            ['✓ 完善的项目管理体系'],
            ['✓ 优质的售后服务']
        ]

        df_contact = pd.DataFrame(contact_data, columns=['信息', '内容'])
        df_contact.to_excel(writer, sheet_name='联系我们', index=False)

    return str(resolved_path)

def main():
    """主函数"""
    try:
        print("🚀 开始生成专业Excel报价单...")

        # 生成Excel文件
        parser = argparse.ArgumentParser(description="生成专业Excel报价单（pandas/openpyxl）")
        parser.add_argument(
            "--output",
            help="输出文件路径（.xlsx）；不传则输出到 storage/skill_outputs/excel-generation/",
        )
        args = parser.parse_args()

        output_path = create_professional_style_excel(args.output)

        # 检查文件大小
        file_size = os.path.getsize(output_path)

        print(f"✅ 专业Excel报价单已生成: {output_path}")
        print(f"📄 文件大小: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print("")
        print("📋 包含以下工作表:")
        print("  • 项目概览 - 项目基本信息和范围")
        print("  • 工作明细报价 - 详细的工作分解和报价")
        print("  • 人员配置 - 各阶段人员投入计划")
        print("  • 项目里程碑 - 关键节点和交付计划")
        print("  • 质量保证 - 质量措施和验收标准")
        print("  • 服务支持 - 售后服务和技术支持")
        print("  • 联系我们 - 公司信息和联系方式")
        print("")
        print("🎨 专业特色:")
        print("  ✅ 清晰的项目概览和范围定义")
        print("  ✅ 详细的工作分解结构")
        print("  ✅ 合理的人天报价计算")
        print("  ✅ 完整的人员配置计划")
        print("  ✅ 明确的项目里程碑")
        print("  ✅ 全面的质量保证措施")
        print("  ✅ 完善的服务支持体系")
        print("")
        print("📧 报价单已准备就绪，可以直接发送给客户！")

        return output_path

    except Exception as e:
        print(f"❌ 生成Excel文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
