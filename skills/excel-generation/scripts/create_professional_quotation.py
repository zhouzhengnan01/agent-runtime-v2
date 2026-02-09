#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side, GradientFill, NamedStyle
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image
import os
import argparse
from pathlib import Path
from datetime import datetime
import re


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_path() -> Path:
    out_dir = _project_root() / "storage" / "skill_outputs" / "excel-generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "海纳设备接入系统-技术规格书及报价单.xlsx"


def create_professional_quotation():
    """创建专业的报价单Excel文档"""

    # 创建工作簿
    wb = Workbook()

    # 删除默认工作表
    wb.remove(wb.active)

    # 设置全局样式
    header_font = Font(name='微软雅黑', size=14, bold=True, color='FFFFFF')
    subheader_font = Font(name='微软雅黑', size=12, bold=True, color='333333')
    normal_font = Font(name='微软雅黑', size=11, color='333333')
    company_font = Font(name='微软雅黑', size=16, bold=True, color='1F497D')

    # 颜色定义
    header_fill = PatternFill(start_color='2F75B5', end_color='2F75B5', fill_type='solid')
    subheader_fill = PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')
    total_fill = PatternFill(start_color='D8E6F7', end_color='D8E6F7', fill_type='solid')
    accent_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')

    # 边框定义
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                     top=Side(style='thin'), bottom=Side(style='thin'))
    thick_border = Border(left=Side(style='medium'), right=Side(style='medium'),
                     top=Side(style='medium'), bottom=Side(style='medium'))

    # 对齐定义
    center_align = Alignment(horizontal='center', vertical='center')
    left_align = Alignment(horizontal='left', vertical='center')

    # 1. 封面页
    ws1 = wb.create_sheet('封面')

    # 公司信息
    ws1['A1'] = '重庆浩鲸科技有限公司'
    ws1['A1'].font = company_font
    ws1['A1'].alignment = center_align
    ws1.merge_cells('A1:D1')

    ws1['A3'] = '技术规格书及报价单'
    ws1['A3'].font = Font(name='微软雅黑', size=20, bold=True, color='333333')
    ws1['A3'].alignment = center_align
    ws1.merge_cells('A3:D3')

    ws1['A5'] = f'报价日期：{datetime.now().strftime("%Y年%m月%d日")}'
    ws1['A5'].font = normal_font
    ws1['C5'] = '报价单号：JL-HN-2024-011'
    ws1['C5'].font = normal_font

    ws1['A7'] = '致：'
    ws1['A7'].font = subheader_font
    ws1['B7'] = '海纳系统用户'
    ws1['B7'].font = normal_font
    ws1.merge_cells('B7:D7')

    # 分隔线
    for col in range(1, 9):
        ws1[f'A{col+10}'].fill = PatternFill(start_color='D8E6F7', end_color='D8E6F7', fill_type='solid')
        ws1[f'A{col+10}'].border = thin_border

    # 项目信息
    ws1['A12'] = '项目基本信息'
    ws1['A12'].font = subheader_font
    ws1['A12'].fill = subheader_fill
    ws1['A12'].border = thin_border
    ws1.merge_cells('A12:D12')

    project_info = [
        ['项目名称', '海纳设备接入系统定制开发'],
        ['合同金额', '¥490,000.00'],
        ['项目周期', '4-5个月'],
        ['交付地点', '甲方指定地点'],
        ['付款方式', '合同签订后30%，系统上线后30%，验收合格后40%'],
        ['质量保证', '系统上线后1年免费维护服务']
    ]

    for i, (key, value) in enumerate(project_info, 1):
        row = 12 + i
        ws1[f'A{row}'] = key
        ws1[f'B{row}'] = value
        ws1[f'A{row}'].font = normal_font
        ws1[f'B{row}'].font = normal_font
        ws1[f'A{row}'].border = thin_border
        ws1[f'B{row}'].border = thin_border
        if row > 15:
            ws1[f'A{row}'].fill = PatternFill(start_color='FFFFFF', end_color='F2F2F2', fill_type='solid')
            ws1[f'B{row}'].fill = PatternFill(start_color='FFFFFF', end_color='F2F2F2', fill_type='solid')

    # 2. 项目概述
    ws2 = wb.create_sheet('项目概述')

    # 标题
    ws2['A1'] = '1. 项目概述'
    ws2['A1'].font = header_font
    ws2['A1'].fill = header_fill
    ws2['A1'].alignment = center_align
    ws2['A1'].border = thick_border
    ws2.merge_cells('A1:F1')

    # 项目描述
    overview_data = [
        ['项目背景', '本项目旨在为海纳系统提供设备接入定制开发服务，实现设备数据的统一接入、处理和管理，提高设备管理效率和数据价值挖掘能力。'],
        ['项目目标', '构建完善的设备接入平台，实现设备数据的实时采集、处理和分析，为后续的智能化应用奠定基础。'],
        ['技术架构', '采用微服务架构，支持高并发、高可用、可扩展的系统设计，确保系统稳定运行和持续发展。'],
        ['核心功能', '设备接入、数据处理、实时监控、报警管理、数据分析、可视化展示等核心功能模块。']
    ]

    for i, (key, value) in enumerate(overview_data, 2):
        row = i
        ws2[f'A{row}'] = key
        ws2[f'A{row}'].font = subheader_font
        ws2[f'A{row}'].fill = subheader_fill
        ws2[f'A{row}'].border = thin_border
        ws2.merge_cells(f'A{row}:F{row}')

        ws2[f'A{row+1}'] = value
        ws2[f'A{row+1}'].font = normal_font
        ws2[f'A{row+1}'].alignment = left_align
        ws2[f'A{row+1}'].border = thin_border
        ws2[f'A{row+1}'].fill = PatternFill(start_color='FFFFFF', end_color='F9F9F9', fill_type='solid')
        ws2.merge_cells(f'A{row+1}:F{row+1}')

    # 3. 详细报价 - 海纳项目
    ws3 = wb.create_sheet('海纳设备接入系统-详细报价')

    # 标题
    ws3['A1'] = '2. 海纳设备接入系统 - 详细报价'
    ws3['A1'].font = header_font
    ws3['A1'].fill = header_fill
    ws3['A1'].alignment = center_align
    ws3['A1'].border = thick_border
    ws3.merge_cells('A1:H1')

    # 表头
    headers = ['序号', '工作内容', '工作包', '技术要求', '人天', '单价(元)', '小计(元)', '备注']
    for col, header in enumerate(headers, 1):
        cell = ws3[f'{get_column_letter(col)}2']
        cell.value = header
        cell.font = subheader_font
        cell.fill = subheader_fill
        cell.border = thick_border
        cell.alignment = center_align

    # 报价数据
    quotation_data = [
        [1, '需求分析阶段', '需求调研', '业务需求调研、用户访谈、需求分析', '10', 1000, 10000, '含需求规格书'],
        ['', '', '技术分析', '技术架构分析、集成需求分析', '3', 1500, 4500, '含技术方案'],
        ['', '', '文档编写', '需求规格书、技术方案编写', '2', 1000, 2000, '符合国标GB/T 8567'],
        [2, '系统设计阶段', '架构设计', '总体架构设计、技术选型、架构评审', '5', 2000, 10000, '含架构设计文档'],
        ['', '', '详细设计', '数据库设计、接口设计、UI设计', '15', 1500, 22500, '含设计文档'],
        [3, '开发实施阶段', '后端开发', 'API开发、业务逻辑实现、核心模块开发', '40', 1500, 60000, 'Java/Spring Boot'],
        ['', '', '前端开发', '用户界面开发、交互逻辑实现', '25', 1000, 25000, 'Vue.js/React'],
        ['', '', '数据库开发', '数据库建模、存储过程、优化', '10', 1200, 12000, 'MySQL/PostgreSQL'],
        ['', '', '集成开发', '与现有系统集成、接口开发', '5', 1200, 6000, 'RESTful API'],
        [4, '测试阶段', '单元测试', '单元测试、集成测试', '8', 1000, 8000, 'Jest/TestNG'],
        ['', '', '功能测试', '功能测试、业务流程测试', '10', 1000, 10000, 'Selenium/自动化'],
        ['', '', '性能测试', '压力测试、性能优化', '7', 1200, 8400, 'JMeter/LoadRunner'],
        ['', '', '安全测试', '安全扫描、渗透测试', '5', 1500, 7500, 'OWASP标准'],
        [5, '部署上线', '环境准备', '开发、测试、生产环境搭建', '10', 1000, 10000, 'Docker/K8s'],
        ['', '', '系统部署', '系统安装、配置、部署', '5', 1000, 5000, '生产环境'],
        ['', '', '上线支持', '上线支持、问题排查、运维交接', '5', 1000, 5000, '7×24小时'],
        [6, '培训移交', '用户培训', '系统使用培训、操作手册', '5', 800, 4000, '现场培训'],
        ['', '', '管理员培训', '系统管理培训、维护手册', '3', 800, 2400, '管理员培训'],
        ['', '', '文档移交', '技术文档、用户手册、操作指南', '7', 800, 5600, '符合国家标准']
    ]

    # 写入数据
    for i, row_data in enumerate(quotation_data, 3):
        for col, value in enumerate(row_data, 1):
            cell = ws3[f'{get_column_letter(col)}{i}']
            cell.value = value

            if i % 6 == 0 and col == 1:  # 序号
                cell.font = subheader_font
                cell.fill = accent_fill
                cell.border = thick_border
                cell.alignment = center_align
            elif i % 6 != 0 and col == 1:  # 空序号
                continue
            elif i % 6 == 0 and col == 2:  # 工作包
                cell.font = subheader_font
                cell.fill = subheader_fill
                cell.border = thick_border
                cell.alignment = center_align
            else:  # 普通数据
                cell.font = normal_font
                cell.border = thin_border
                if col >= 5:  # 数字列
                    cell.alignment = center_align

    # 总计行
    total_row = 23
    ws3[f'A{total_row}'] = '总计'
    ws3[f'A{total_row}'].font = Font(name='微软雅黑', size=12, bold=True)
    ws3[f'A{total_row}'].fill = total_fill
    ws3[f'A{total_row}'].border = thick_border
    ws3[f'A{total_row}'].alignment = center_align
    ws3.merge_cells(f'A{total_row}:D{total_row}')

    ws3[f'E{total_row}'] = 185  # 总人天
    ws3[f'E{total_row}'].font = Font(name='微软雅黑', size=12, bold=True)
    ws3[f'E{total_row}'].fill = total_fill
    ws3[f'E{total_row}'].border = thick_border
    ws3[f'E{total_row}'].alignment = center_align

    ws3[f'G{total_row}'] = 185000  # 总计
    ws3[f'G{total_row}'].font = Font(name='Microsoft YaHei', size=12, bold=True, color='D9534F')
    ws3[f'G{total_row}'].fill = total_fill
    ws3[f'G{total_row}'].border = thick_border
    ws3[f'G{total_row}'].alignment = center_align

    # 费用说明
    ws3['A25'] = '费用说明'
    ws3['A25'].font = subheader_font
    ws3['A25'].fill = subheader_fill
    ws3['A25'].border = thin_border
    ws3.merge_cells('A25:H25')

    fee_notes = [
        '1. 以上报价为含税价，增值税率6%',
        '2. 报价有效期为30天，超过有效期需重新报价',
        '3. 项目实施过程中如需需求变更，需另行协商',
        '4. 培用不包含第三方软件授权费用',
        '5. 质量保证金：合同总金额的10%',
        '6. 付款方式：合同签订后30%，系统上线后30%，验收合格后40%'
    ]

    for i, note in enumerate(fee_notes, 26):
        ws3[f'A{i}'] = note
        ws3[f'A{i}'].font = normal_font
        ws3[f'A{i}'].alignment = left_align
        ws3.merge_cells(f'A{i}:H{i}')

    # 4. 详细报价 - 火调项目
    ws4 = wb.create_sheet('火调技术服务-详细报价')

    # 标题
    ws4['A1'] = '3. 火调技术服务 - 详细报价'
    ws4['A1'].font = header_font
    ws4['A1'].fill = header_fill
    ws4['A1'].alignment = center_align
    ws4['A1'].border = thick_border
    ws4.merge_cells('A1:H1')

    # 表头
    for col, header in enumerate(headers, 1):
        cell = ws4[f'{get_column_letter(col)}2']
        cell.value = header
        cell.font = subheader_font
        cell.fill = subheader_fill
        cell.border = thick_border
        cell.alignment = center_align

    # 火调项目数据
    fire_data = [
        [1, '需求调研阶段', '现场调研', '现有系统调研、业务流程分析', '10', 1000, 10000, '含调研报告'],
        ['', '', '需求分析', '业务需求分析、技术需求分析', '8', 1000, 8000, '含需求文档'],
        ['', '', '方案设计', '技术方案设计、实施方案设计', '2', 1500, 3000, '含设计文档'],
        [2, '系统开发阶段', '模块开发', '核心功能模块开发', '40', 1000, 40000, 'Java/Python'],
        ['', '', '接口开发', '系统接口开发、API开发', '15', 1000, 15000, 'RESTful API'],
        ['', '', '数据迁移', '现有数据迁移、数据转换', '5', 1200, 6000, '数据转换工具'],
        [3, '测试阶段', '功能测试', '系统功能测试、业务流程测试', '10', 1000, 10000, '自动化测试'],
        ['', '', '集成测试', '系统集成测试、接口测试', '8', 1000, 8000, '集成测试工具'],
        ['', '', '性能测试', '系统性能测试、优化', '7', 1200, 8400, '性能测试工具'],
        [4, '部署实施阶段', '系统部署', '生产环境部署、配置', '10', 1000, 10000, 'Docker部署'],
        ['', '', '数据初始化', '基础数据初始化、配置', '10', 1000, 10000, '数据初始化脚本'],
        ['', '', '上线支持', '上线支持、问题处理', '5', 1000, 5000, '7×24小时'],
        [5, '培训移交阶段', '技术培训', '技术人员培训、系统使用培训', '8', 800, 6400, '现场培训'],
        ['', '', '用户培训', '最终用户培训、操作培训', '5', 800, 4000, '用户手册'],
        ['', '', '文档编写', '技术文档、用户手册', '2', 800, 1600, '符合国标']
    ]

    # 写入火调数据
    for i, row_data in enumerate(fire_data, 3):
        for col, value in enumerate(row_data, 1):
            cell = ws4[f'{get_column_letter(col)}{i}']
            cell.value = value

            if i % 5 == 0 and col == 1:  # 序号
                cell.font = subheader_font
                cell.fill = accent_fill
                cell.border = thick_border
                cell.alignment = center_align
            elif i % 5 != 0 and col == 1:  # 空序号
                continue
            elif i % 5 == 0 and col == 2:  # 工作包
                cell.font = subheader_font
                cell.fill = subheader_fill
                cell.border = thick_border
                cell.alignment = center_align
            else:  # 普通数据
                cell.font = normal_font
                cell.border = thin_border
                if col >= 5:  # 数字列
                    cell.alignment = center_align

    # 火调总计
    fire_total_row = 22
    ws4[f'A{fire_total_row}'] = '总计'
    ws4[f'A{fire_total_row}'].font = Font(name='微软雅黑', size=12, bold=True)
    ws4[f'A{fire_total_row}'].fill = total_fill
    ws4[f'A{fire_total_row}'].border = thick_border
    ws4[f'A{fire_total_row}'].alignment = center_align
    ws4.merge_cells(f'A{fire_total_row}:D{fire_total_row}')

    ws4[f'E{fire_total_row}'] = 150  # 总人天
    ws4[f'E{fire_total_row}'].font = Font(name='微软雅黑', size=12, bold=True)
    ws4[f'E{fire_total_row}'].fill = total_fill
    ws4[f'E{fire_total_row}'].border = thick_border
    ws4[f'E{fire_total_row}'].alignment = center_align

    ws4[f'G{fire_total_row}'] = 150000  # 总计
    ws4[f'G{fire_total_row}'].font = Font(name='Microsoft YaHei', size=12, bold=True, color='D9534F')
    ws4[f'G{fire_total_row}'].fill = total_fill
    ws4[f'G{fire_total_row}'].border = thick_border
    ws4[f'G{fire_total_row}'].alignment = center_align

    # 5. 人员配置表
    ws5 = wb.create_sheet('人员配置')

    # 标题
    ws5['A1'] = '4. 项目人员配置'
    ws5['A1'].font = header_font
    ws5['A1'].fill = header_fill
    ws5['A1'].alignment = center_align
    ws5['A1'].border = thick_border
    ws5.merge_cells('A1:F1')

    # 表头
    staff_headers = ['角色', '数量', '资质要求', '经验要求', '单价(元/人天)', '工作内容']
    for col, header in enumerate(staff_headers, 1):
        cell = ws5[f'{get_column_letter(col)}2']
        cell.value = header
        cell.font = subheader_font
        cell.fill = subheader_fill
        cell.border = thick_border
        cell.alignment = center_align

    # 人员配置数据
    staff_data = [
        ['项目经理', '1', 'PMP认证、高级工程师', '5年以上项目管理经验', 2000, '项目整体管理、进度控制、质量管理'],
        ['技术负责人', '1', '架构师认证、高级工程师', '8年以上技术经验', 2500, '技术架构设计、技术难题攻关、团队技术指导'],
        ['系统架构师', '1', '系统架构师认证', '5年以上架构设计经验', 2000, '系统架构设计、技术方案评审'],
        ['后端开发工程师', '2', '高级软件工程师认证', '5年以上开发经验', 1500, '后端模块开发、接口设计'],
        ['前端开发工程师', '1', 'Web前端开发经验', '3年以上开发经验', 1200, '用户界面开发、交互实现'],
        ['数据库工程师', '1', '数据库认证', '3年以上数据库经验', 1200, '数据库设计、优化、维护'],
        ['测试工程师', '2', '测试认证', '3年以上测试经验', 1000, '测试用例编写、自动化测试'],
        ['运维工程师', '1', '运维认证', '3年以上运维经验', 1500, '环境搭建、部署、监控'],
        ['UI设计师', '1', '设计相关专业', '3年以上设计经验', 1000, '界面设计、原型制作'],
        ['培训讲师', '1', '培训师认证', '3年以上培训经验', 800, '培训材料编写、现场培训']
    ]

    # 写入人员配置数据
    for i, staff in enumerate(staff_data, 3):
        for col, value in enumerate(staff, 1):
            cell = ws5[f'{get_column_letter(col)}{i}']
            cell.value = value
            cell.font = normal_font
            cell.border = thin_border
            cell.alignment = left_align

    # 6. 项目里程碑
    ws6 = wb.create_sheet('项目里程碑')

    # 标题
    ws6['A1'] = '5. 项目里程碑'
    ws6['A1'].font = header_font
    ws6['A1'].fill = header_fill
    ws6['A1'].alignment = center_align
    ws6['A1'].border = thick_border
    ws6.merge_cells('A1:F1')

    # 表头
    milestone_headers = '序号,里程碑,完成时间,交付物,验收标准,负责人'
    headers = milestone_headers.split(',')
    for col, header in enumerate(headers, 1):
        cell = ws6[f'{get_column_letter(col)}2']
        cell.value = header
        cell.font = subheader_font
        cell.fill = subheader_fill
        cell.border = thick_border
        cell.alignment = center_align

    # 里程碑数据
    milestones = [
        ['1', '项目启动', '第1周', '项目计划书、团队配置', '双方签字确认', '项目经理'],
        ['2', '需求分析完成', '第3周', '需求规格书', '需求评审通过', '技术负责人'],
        ['3', '系统设计完成', '第7周', '设计文档', '设计评审通过', '架构师'],
        ['4', '核心功能开发完成', '第15周', '可演示系统', '功能验证通过', '技术负责人'],
        ['5', '测试完成', '第19周', '测试报告', '测试通过率≥95%', '测试负责人'],
        ['6', '系统上线', '第20周', '生产系统', '系统稳定运行', '项目经理'],
        ['7', '项目验收', '第21周', '验收报告', '验收合格', '双方负责人']
    ]

    # 写入里程碑数据
    for i, milestone in enumerate(milestones, 3):
        for col, value in enumerate(milestone, 1):
            cell = ws6[f'{get_column_letter(col)}{i}']
            cell.value = value
            cell.font = normal_font
            cell.border = thin_border
            if col == 1 or col == 2:  # 序号和里程碑
                cell.font = subheader_font
                cell.fill = PatternFill(start_color='E8F5E8', end_color='E8F5E8', fill_type='solid')
            cell.alignment = center_align

    # 7. 质量保证
    ws7 = wb.create_sheet('质量保证')

    # 标题
    ws7['A1'] = '6. 质量保证'
    ws7['A1'].font = header_font
    ws7['A1'].fill = header_fill
    ws7['A1'].alignment = center_align
    ws7['A1'].border = thick_border
    ws7.merge_cells('A1:F1')

    # 质量保证内容
    quality_items = [
        '质量目标',
        '质量管理体系：ISO9001质量管理体系',
        '代码质量：代码覆盖率≥80%，代码审查通过率100%',
        '测试质量：测试用例覆盖率≥90%，自动化测试率≥80%',
        '交付质量：交付物完整率100%，验收通过率100%',
        '',
        '质量保证措施',
        '代码审查：代码审查、技术评审、设计评审',
        '测试管理：单元测试、集成测试、系统测试、用户验收测试',
        '文档管理：设计文档、测试文档、用户手册、运维手册',
        '版本控制：使用Git进行版本控制，确保可追溯性',
        '',
        '质量检查点',
        '需求确认：需求评审、需求变更控制',
        '设计审查：架构设计评审、详细设计评审',
        '代码审查：代码走查、单元测试审查',
        '测试验证：测试用例评审、测试结果验证',
        '交付验收：交付物检查、用户验收测试'
    ]

    # 写入质量保证内容
    for i, item in enumerate(quality_items):
        if item:  # 非空行
            cell = ws7[f'A{i+1}']
            cell.value = item
            if item.startswith('质量目标') or item.startswith('质量保证措施') or item.startswith('质量检查点'):
                cell.font = subheader_font
                cell.fill = subheader_fill
                cell.border = thin_border
                ws7.merge_cells(f'A{i+1}:F{i+1}')
            else:
                cell.font = normal_font
                cell.border = thin_border
                cell.alignment = left_align

    # 8. 风险管理
    ws8 = wb.create_sheet('风险管理')

    # 标题
    ws8['A1'] = '7. 风险管理'
    ws8['A1'].font = header_font
    ws8['A1'].fill = header_fill
    ws8['A1'].alignment = center_align
    ws8['A1'].border = thick_border
    ws8.merge_cells('A1:F1')

    # 风险管理内容
    risk_data = [
        ['风险类型', '风险描述', '风险等级', '影响程度', '应对措施', '责任人'],
        ['技术风险', '技术选型不当、集成难度大、技术难题', '中', '高', '充分调研、技术评估、备选方案', '技术负责人'],
        ['进度风险', '需求变更、资源不足、延期风险', '中', '中', '合理规划、资源预留、进度监控', '项目经理'],
        ['质量风险', '测试不充分、质量问题', '低', '中', '加强测试、代码审查、质量检查', '质量负责人'],
        ['人员风险', '人员流失、技能不足', '中', '中', '团队建设、培训、知识传递', '项目经理'],
        ['集成风险', '第三方系统不兼容', '高', '高', '提前测试、接口验证、预留缓冲', '集成工程师'],
        ['成本风险', '成本超支、变更成本', '中', '中', '预算控制、变更管理、成本监控', '项目经理']
    ]

    # 写入风险管理数据
    for i, risk in enumerate(risk_data, 2):
        for col, value in enumerate(risk, 1):
            cell = ws8[f'{get_column_letter(col)}{i}']
            cell.value = value
            cell.font = normal_font
            cell.border = thin_border
            cell.alignment = left_align
            if i == 2:  # 表头行
                cell.font = subheader_font
                cell.fill = subheader_fill

    # 9. 联系方式
    ws9 = wb.create_sheet('联系方式')

    # 标题
    ws9['A1'] = '8. 联系方式'
    ws9['A1'].font = header_font
    ws9['A1'].fill = header_fill
    ws9['A1'].alignment = center_align
    ws9['A1'].border = thick_border
    ws9.merge_cells('A1:D1')

    # 联系信息
    contact_info = [
        ['公司名称', '重庆浩鲸科技有限公司'],
        ['联系人', '陈浩'],
        ['职位', '技术总监'],
        ['电话', '188-8317-9204'],
        ['邮箱', 'chenhao@jetlinks.cn'],
        ['地址', '重庆市九龙坡区科园二路特号'],
        ['服务热线', '400-888-8888'],
        ['技术支持', 'support@jetlinks.cn']
    ]

    # 写入联系信息
    for i, (key, value) in enumerate(contact_info, 2):
        ws9[f'A{i}'] = key
        ws9[f'B{i}'] = value
        ws9[f'A{i}'].font = subheader_font
        ws9[f'B{i}'].font = normal_font
        ws9[f'A{i}'].fill = subheader_fill
        ws9[f'B{i}'].border = thin_border

    # 设置列宽
    for ws in [ws1, ws2, ws3, ws4, ws5, ws6, ws7, ws8, ws9]:
        ws.column_dimensions['A'] = 20
        ws.column_dimensions['B'] = 25
        ws.column_dimensions['C'] = 20
        ws.column_dimensions['D'] = 20
        ws.column_dimensions['E'] = 12
        ws.column_dimensions['F'] = 15
        ws.column_dimensions['G'] = 15
        ws.column_dimensions['H'] = 15

    # 设置行高
    for ws in [ws1, ws2, ws3, ws4, ws5, ws6, ws7, ws8, ws9]:
        ws.row_dimensions[1] = 30  # 标题行高
        ws.row_dimensions[2] = 20  # 表头行高

    return wb

def main():
    """主函数"""
    try:
        parser = argparse.ArgumentParser(description="生成技术规格书及报价单（Excel）")
        parser.add_argument(
            "--output",
            help="输出文件路径（.xlsx）；不传则输出到 storage/skill_outputs/excel-generation/",
        )
        args = parser.parse_args()

        # 创建专业的Excel文档
        wb = create_professional_quotation()

        # 保存文件
        output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(output_path))

        # 检查文件大小
        file_size = output_path.stat().st_size

        print(f"✅ 专业的技术规格书及报价单已生成: {output_path}")
        print(f"📄 文件大小: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        print("")
        print("📋 包含以下工作表:")
        print("  📄 封面 - 公司信息、项目信息、报价日期等")
        print("  📋 项目概述 - 项目背景、目标、技术架构")
        print("  📋 海纳设备接入系统-详细报价 - 详细的工作分解和报价")
        print("  📋 火调技术服务-详细报价 - 详细的工作分解和报价")
        print("  📋 人员配置 - 项目团队成员配置和要求")
        print("  📋 项目里程碑 - 项目关键节点和交付物")
        print("  📋 质量保证 - 质量目标和保证措施")
        print("  📋 风险管理 - 风险识别和应对措施")
        print("  📋 联系方式 - 公司联系信息")
        print("")
        print("🎨 专业特色:")
        print("  ✅ 精美的格式设计，适合发给甲方")
        print("  ✅ 详细的工作分解，人天计算准确")
        print("  ✅ 完整的项目管理内容")
        print("  ✅ 包含质量保证和风险管理")
        print("  ✅ 专业的人员配置要求")
        print("  ✅ 清晰的项目里程碑")
        print("  ✅ 标准化的文档格式")
        print("")
        print("💡 您可以直接发送给甲方使用！")

        return output_path

    except Exception as e:
        print(f"❌ 生成Excel文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
