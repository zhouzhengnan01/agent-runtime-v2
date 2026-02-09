#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

try:
    import xmind
except ImportError as exc:
    raise SystemExit("Missing dependency: xmind. Install it first (e.g. `pip install xmind`).") from exc


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_path() -> Path:
    out_dir = _project_root() / "storage" / "skill_outputs" / "xmind-generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "智巡云SaaS方向规划.xmind"


def create_xmind(*, output_path: Path, template_path: Optional[Path] = None) -> Path:
    # 创建工作簿（可选基于模板）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if template_path:
            if not template_path.exists():
                raise FileNotFoundError(f"template not found: {template_path}")
            workbook = xmind.load(str(template_path))
        else:
            workbook = xmind.load(str(output_path))
    except Exception:
        # 兜底：尝试创建新工作簿
        workbook = xmind.load(None)

    # 获取第一个画布
    sheet = workbook.getPrimarySheet()
    sheet.setTitle("智巡云 SaaS 产品方向规划")

    # 获取根主题
    root_topic = sheet.getRootTopic()
    root_topic.setTitle("智巡云 SaaS 产品方向规划")

    # 1. 产品定位
    topic1 = root_topic.addSubTopic()
    topic1.setTitle("1. 产品定位")

    topic1_1 = topic1.addSubTopic()
    topic1_1.setTitle("核心定位")
    topic1_1.addSubTopic().setTitle("AI赋能的企业级SaaS服务平台")
    topic1_1.addSubTopic().setTitle("为微小企业提供低成本、高效率的数字化解决方案")
    topic1_1.addSubTopic().setTitle("聚焦垂直行业的智能化升级")

    topic1_2 = topic1.addSubTopic()
    topic1_2.setTitle("目标客户")
    topic1_2.addSubTopic().setTitle("微小型企业（50人以下）")
    topic1_2.addSubTopic().setTitle("传统行业数字化转型企业")
    topic1_2.addSubTopic().setTitle("需要智能巡检服务的企业")

    topic1_3 = topic1.addSubTopic()
    topic1_3.setTitle("核心价值主张")
    topic1_3.addSubTopic().setTitle("让微小企业也能享受AI带来的效率革命")

    # 2. API开放平台
    topic2 = root_topic.addSubTopic()
    topic2.setTitle("2. 核心能力：API开放平台")

    topic2_1 = topic2.addSubTopic()
    topic2_1.setTitle("API Hub能力")
    topic2_1.addSubTopic().setTitle("参考网站：https://www.explinks.com/apihub")
    api_capabilities = topic2_1.addSubTopic()
    api_capabilities.setTitle("对外开放AI能力接口")
    api_capabilities.addSubTopic().setTitle("智能识别")
    api_capabilities.addSubTopic().setTitle("数据分析")
    api_capabilities.addSubTopic().setTitle("内容生成")
    api_capabilities.addSubTopic().setTitle("按调用次数计费")

    topic2_2 = topic2.addSubTopic()
    topic2_2.setTitle("系统集成能力")
    integration = topic2_2.addSubTopic()
    integration.setTitle("对接已有系统")
    integration.addSubTopic().setTitle("明道云")
    integration.addSubTopic().setTitle("企业微信")
    integration.addSubTopic().setTitle("其他第三方平台")
    protocol = topic2_2.addSubTopic()
    protocol.setTitle("标准协议支持")
    protocol.addSubTopic().setTitle("Webhook")
    protocol.addSubTopic().setTitle("OAuth")

    topic2_3 = topic2.addSubTopic()
    topic2_3.setTitle("开发者生态")
    topic2_3.addSubTopic().setTitle("API文档")
    topic2_3.addSubTopic().setTitle("SDK工具包")
    topic2_3.addSubTopic().setTitle("示例代码")
    topic2_3.addSubTopic().setTitle("开发者社区")
    topic2_3.addSubTopic().setTitle("技术支持")

    # 3. 微小企业AI工具箱
    topic3 = root_topic.addSubTopic()
    topic3.setTitle("3. 业务方向一：微小企业AI工具箱")

    # 3.1 AI营销获客
    topic3_1 = topic3.addSubTopic()
    topic3_1.setTitle("3.1 AI营销获客系统")
    core_func_31 = topic3_1.addSubTopic()
    core_func_31.setTitle("核心功能")
    core_func_31.addSubTopic().setTitle("智能推荐合适企业客户")
    core_func_31.addSubTopic().setTitle("构建企业信息库")
    core_func_31.addSubTopic().setTitle("精准判断客户意向")
    core_func_31.addSubTopic().setTitle("调用谷歌搜索接口")
    core_func_31.addSubTopic().setTitle("自动化线索挖掘")
    value_31 = topic3_1.addSubTopic()
    value_31.setTitle("价值主张")
    value_31.addSubTopic().setTitle("降低获客成本 60%")
    value_31.addSubTopic().setTitle("提升线索质量 3倍")

    # 3.2 智能邮件营销
    topic3_2 = topic3.addSubTopic()
    topic3_2.setTitle("3.2 智能邮件营销")
    core_func_32 = topic3_2.addSubTopic()
    core_func_32.setTitle("核心功能")
    core_func_32.addSubTopic().setTitle("基于历史邮件智能生成")
    core_func_32.addSubTopic().setTitle("个性化邮件模板库")
    core_func_32.addSubTopic().setTitle("A/B测试")
    core_func_32.addSubTopic().setTitle("效果追踪")
    scenario_32 = topic3_2.addSubTopic()
    scenario_32.setTitle("应用场景")
    scenario_32.addSubTopic().setTitle("客户跟进")
    scenario_32.addSubTopic().setTitle("产品推广")
    scenario_32.addSubTopic().setTitle("活动邀请")

    # 3.3 AI内容运营
    topic3_3 = topic3.addSubTopic()
    topic3_3.setTitle("3.3 AI内容运营推广")
    channel_33 = topic3_3.addSubTopic()
    channel_33.setTitle("全渠道覆盖")

    xiaohongshu = channel_33.addSubTopic()
    xiaohongshu.setTitle("小红书自动化运营")
    xiaohongshu.addSubTopic().setTitle("内容生成")
    xiaohongshu.addSubTopic().setTitle("自动发布")
    xiaohongshu.addSubTopic().setTitle("数据分析")

    wechat = channel_33.addSubTopic()
    wechat.setTitle("微信公众号")
    wechat.addSubTopic().setTitle("文章生成")
    wechat.addSubTopic().setTitle("排版优化")
    wechat.addSubTopic().setTitle("定时发布")

    video = channel_33.addSubTopic()
    video.setTitle("AI视频生成")
    video.addSubTopic().setTitle("脚本生成")
    video.addSubTopic().setTitle("视频剪辑")
    video.addSubTopic().setTitle("配音配乐")

    value_33 = topic3_3.addSubTopic()
    value_33.setTitle("价值主张")
    value_33.addSubTopic().setTitle("节约运营成本 70%")
    value_33.addSubTopic().setTitle("快速获客")
    value_33.addSubTopic().setTitle("提升品牌曝光")

    # 3.4 AI招投标
    topic3_4 = topic3.addSubTopic()
    topic3_4.setTitle("3.4 AI招投标助手")
    core_func_34 = topic3_4.addSubTopic()
    core_func_34.setTitle("核心功能")
    core_func_34.addSubTopic().setTitle("招标文件智能解析")
    core_func_34.addSubTopic().setTitle("投标文档自动生成")
    core_func_34.addSubTopic().setTitle("历史案例库")
    core_func_34.addSubTopic().setTitle("中标率分析")
    value_34 = topic3_4.addSubTopic()
    value_34.setTitle("价值主张")
    value_34.addSubTopic().setTitle("节约标书制作时间 80%")
    value_34.addSubTopic().setTitle("提升中标率")
    value_34.addSubTopic().setTitle("降低人力成本")

    # 3.5 AI财务运营
    topic3_5 = topic3.addSubTopic()
    topic3_5.setTitle("3.5 AI财务运营")
    core_func_35 = topic3_5.addSubTopic()
    core_func_35.setTitle("核心功能")
    core_func_35.addSubTopic().setTitle("票据智能识别与录入")
    core_func_35.addSubTopic().setTitle("财务报表自动生成")
    core_func_35.addSubTopic().setTitle("税务合规性检查")
    core_func_35.addSubTopic().setTitle("费用分析与预测")
    value_35 = topic3_5.addSubTopic()
    value_35.setTitle("价值主张")
    value_35.addSubTopic().setTitle("降低人工成本")
    value_35.addSubTopic().setTitle("减少错误率")
    value_35.addSubTopic().setTitle("提升效率")

    # 3.6 AI舆情分析
    topic3_6 = topic3.addSubTopic()
    topic3_6.setTitle("3.6 AI市场舆情分析")
    core_func_36 = topic3_6.addSubTopic()
    core_func_36.setTitle("核心功能")
    core_func_36.addSubTopic().setTitle("基于搜索引擎的实时监控")
    core_func_36.addSubTopic().setTitle("品牌情感分析")
    core_func_36.addSubTopic().setTitle("竞品动态追踪")
    core_func_36.addSubTopic().setTitle("舆情预警")
    scenario_36 = topic3_6.addSubTopic()
    scenario_36.setTitle("应用场景")
    scenario_36.addSubTopic().setTitle("品牌危机预警")
    scenario_36.addSubTopic().setTitle("市场趋势洞察")
    scenario_36.addSubTopic().setTitle("竞品分析")

    # 4. 巡检与复判服务
    topic4 = root_topic.addSubTopic()
    topic4.setTitle("4. 业务方向二：巡检与复判服务")

    # 4.1 无人机巡检
    topic4_1 = topic4.addSubTopic()
    topic4_1.setTitle("4.1 无人机巡检SaaS平台")

    tech_41 = topic4_1.addSubTopic()
    tech_41.setTitle("技术对接")
    tech_41.addSubTopic().setTitle("大疆无人机SDK集成")
    tech_41.addSubTopic().setTitle("开发者平台：https://developer.dji.com/cn/")
    tech_41.addSubTopic().setTitle("参考文章：https://mp.weixin.qq.com/s/ZktTiOV4U6dh9XaaP7082Q")

    core_func_41 = topic4_1.addSubTopic()
    core_func_41.setTitle("核心功能")
    core_func_41.addSubTopic().setTitle("飞行任务规划与调度")
    core_func_41.addSubTopic().setTitle("实时视频流监控")
    core_func_41.addSubTopic().setTitle("巡检数据自动采集")
    core_func_41.addSubTopic().setTitle("AI智能分析")
    core_func_41.addSubTopic().setTitle("缺陷识别")

    scenario_41 = topic4_1.addSubTopic()
    scenario_41.setTitle("应用场景")
    scenario_41.addSubTopic().setTitle("电力巡检")
    scenario_41.addSubTopic().setTitle("石油天然气管道")
    scenario_41.addSubTopic().setTitle("铁路/公路基础设施")
    scenario_41.addSubTopic().setTitle("建筑工地安全监测")
    scenario_41.addSubTopic().setTitle("农林植保")

    # 4.2 AR眼镜巡检
    topic4_2 = topic4.addSubTopic()
    topic4_2.setTitle("4.2 AR眼镜巡检方案")

    device_42 = topic4_2.addSubTopic()
    device_42.setTitle("设备对接")
    device_42.addSubTopic().setTitle("雷鸟AR眼镜SDK集成")
    device_42.addSubTopic().setTitle("开放平台：https://open.rayneo.cn/")
    device_42.addSubTopic().setTitle("支持用户自主选择眼镜型号")

    core_func_42 = topic4_2.addSubTopic()
    core_func_42.setTitle("核心功能")
    core_func_42.addSubTopic().setTitle("第一视角巡检记录")
    core_func_42.addSubTopic().setTitle("AR辅助标注与指引")
    core_func_42.addSubTopic().setTitle("语音交互")
    core_func_42.addSubTopic().setTitle("远程协助")
    core_func_42.addSubTopic().setTitle("数据实时上传")

    advantage_42 = topic4_2.addSubTopic()
    advantage_42.setTitle("技术优势")
    advantage_42.addSubTopic().setTitle("多品牌兼容策略")
    advantage_42.addSubTopic().setTitle("解放双手操作")
    advantage_42.addSubTopic().setTitle("实时指导")

    # 4.3 智能复判
    topic4_3 = topic4.addSubTopic()
    topic4_3.setTitle("4.3 智能复判SaaS服务")

    core_func_43 = topic4_3.addSubTopic()
    core_func_43.setTitle("核心功能")
    core_func_43.addSubTopic().setTitle("AI自动识别缺陷")
    core_func_43.addSubTopic().setTitle("人工+AI混合复判机制")
    core_func_43.addSubTopic().setTitle("接口对接系统")
    core_func_43.addSubTopic().setTitle("准确率保障")

    scenario_43 = topic4_3.addSubTopic()
    scenario_43.setTitle("应用场景")
    scenario_43.addSubTopic().setTitle("电力巡检复判")
    scenario_43.addSubTopic().setTitle("设备检测复判")
    scenario_43.addSubTopic().setTitle("质量把控")

    value_43 = topic4_3.addSubTopic()
    value_43.setTitle("价值主张")
    value_43.addSubTopic().setTitle("提升准确率")
    value_43.addSubTopic().setTitle("降低漏检率")
    value_43.addSubTopic().setTitle("历史数据沉淀")
    value_43.addSubTopic().setTitle("形成行业知识库")
    value_43.addSubTopic().setTitle("支持预测性维护")

    # 5. 商业模式
    topic5 = root_topic.addSubTopic()
    topic5.setTitle("5. 商业模式与市场策略")

    topic5_1 = topic5.addSubTopic()
    topic5_1.setTitle("产品版本定价")

    free = topic5_1.addSubTopic()
    free.setTitle("免费版")
    free.addSubTopic().setTitle("目标客户：个人/初创团队")
    free.addSubTopic().setTitle("价格：免费")
    free.addSubTopic().setTitle("功能：基础功能，有使用限制")

    basic = topic5_1.addSubTopic()
    basic.setTitle("基础版")
    basic.addSubTopic().setTitle("目标客户：微小企业")
    basic.addSubTopic().setTitle("价格：¥299/月")
    basic.addSubTopic().setTitle("功能：核心功能包")

    pro = topic5_1.addSubTopic()
    pro.setTitle("专业版")
    pro.addSubTopic().setTitle("目标客户：中小企业")
    pro.addSubTopic().setTitle("价格：¥999/月")
    pro.addSubTopic().setTitle("功能：全功能 + API调用")

    enterprise = topic5_1.addSubTopic()
    enterprise.setTitle("企业版")
    enterprise.addSubTopic().setTitle("目标客户：大型企业")
    enterprise.addSubTopic().setTitle("价格：定制报价")
    enterprise.addSubTopic().setTitle("功能：私有化部署 + 定制开发")

    topic5_2 = topic5.addSubTopic()
    topic5_2.setTitle("盈利模式")
    topic5_2.addSubTopic().setTitle("订阅费（按月/按年）")
    topic5_2.addSubTopic().setTitle("API按量计费")
    increase_service = topic5_2.addSubTopic()
    increase_service.setTitle("增值服务")
    increase_service.addSubTopic().setTitle("人工复判服务")
    increase_service.addSubTopic().setTitle("定制开发")
    increase_service.addSubTopic().setTitle("咨询服务")
    topic5_2.addSubTopic().setTitle("硬件+软件捆绑销售")
    topic5_2.addSubTopic().setTitle("渠道代理分成")

    topic5_3 = topic5.addSubTopic()
    topic5_3.setTitle("市场策略")
    topic5_3.addSubTopic().setTitle("垂直行业切入")
    topic5_3.addSubTopic().setTitle("开发者生态建设")
    topic5_3.addSubTopic().setTitle("渠道合作")
    topic5_3.addSubTopic().setTitle("内容营销")
    topic5_3.addSubTopic().setTitle("案例沉淀")

    # 6. 技术架构
    topic6 = root_topic.addSubTopic()
    topic6.setTitle("6. 技术架构")

    topic6_1 = topic6.addSubTopic()
    topic6_1.setTitle("前端架构")
    topic6_1.addSubTopic().setTitle("响应式Web应用")
    topic6_1.addSubTopic().setTitle("移动端App")
    topic6_1.addSubTopic().setTitle("跨平台适配")

    topic6_2 = topic6.addSubTopic()
    topic6_2.setTitle("后端架构")
    topic6_2.addSubTopic().setTitle("微服务架构")
    topic6_2.addSubTopic().setTitle("云原生部署")
    topic6_2.addSubTopic().setTitle("弹性伸缩")
    topic6_2.addSubTopic().setTitle("高可用设计")

    topic6_3 = topic6.addSubTopic()
    topic6_3.setTitle("AI能力")
    topic6_3.addSubTopic().setTitle("自研AI模型")
    topic6_3.addSubTopic().setTitle("第三方模型集成")
    topic6_3.addSubTopic().setTitle("持续训练优化")
    topic6_3.addSubTopic().setTitle("行业模型定制")

    topic6_4 = topic6.addSubTopic()
    topic6_4.setTitle("数据安全")
    topic6_4.addSubTopic().setTitle("等保三级认证")
    topic6_4.addSubTopic().setTitle("数据加密传输")
    topic6_4.addSubTopic().setTitle("权限管理")
    topic6_4.addSubTopic().setTitle("隐私保护")

    topic6_5 = topic6.addSubTopic()
    topic6_5.setTitle("设备接入")
    topic6_5.addSubTopic().setTitle("大疆无人机SDK")
    topic6_5.addSubTopic().setTitle("雷鸟AR眼镜SDK")
    topic6_5.addSubTopic().setTitle("统一设备管理平台")
    topic6_5.addSubTopic().setTitle("多品牌兼容")

    # 7. 核心竞争力
    topic7 = root_topic.addSubTopic()
    topic7.setTitle("7. 核心竞争力")

    topic7_1 = topic7.addSubTopic()
    topic7_1.setTitle("技术护城河")
    topic7_1.addSubTopic().setTitle("多设备兼容能力")
    topic7_1.addSubTopic().setTitle("垂直行业数据积累")
    topic7_1.addSubTopic().setTitle("AI模型持续优化")

    topic7_2 = topic7.addSubTopic()
    topic7_2.setTitle("产品护城河")
    topic7_2.addSubTopic().setTitle("开放API生态")
    topic7_2.addSubTopic().setTitle("易用性优势")
    topic7_2.addSubTopic().setTitle("快速部署能力")

    topic7_3 = topic7.addSubTopic()
    topic7_3.setTitle("成本护城河")
    topic7_3.addSubTopic().setTitle("低成本高效率")
    topic7_3.addSubTopic().setTitle("微小企业负担得起")
    topic7_3.addSubTopic().setTitle("规模化降低成本")

    topic7_4 = topic7.addSubTopic()
    topic7_4.setTitle("数据护城河")
    topic7_4.addSubTopic().setTitle("行业知识库沉淀")
    topic7_4.addSubTopic().setTitle("用户行为数据")
    topic7_4.addSubTopic().setTitle("持续优化反馈")

    # 8. 实施路线图
    topic8 = root_topic.addSubTopic()
    topic8.setTitle("8. 实施路线图")

    topic8_1 = topic8.addSubTopic()
    topic8_1.setTitle("第一阶段：MVP（3个月）")
    dev_goal_81 = topic8_1.addSubTopic()
    dev_goal_81.setTitle("开发目标")
    dev_goal_81.addSubTopic().setTitle("API开放平台基础能力")
    dev_goal_81.addSubTopic().setTitle("AI营销获客模块")
    dev_goal_81.addSubTopic().setTitle("邮件营销模块")
    dev_goal_81.addSubTopic().setTitle("大疆无人机SDK对接")
    verify_goal_81 = topic8_1.addSubTopic()
    verify_goal_81.setTitle("验证目标")
    verify_goal_81.addSubTopic().setTitle("产品市场契合度")
    verify_goal_81.addSubTopic().setTitle("核心功能可用性")

    topic8_2 = topic8.addSubTopic()
    topic8_2.setTitle("第二阶段：功能扩展（6个月）")
    dev_goal_82 = topic8_2.addSubTopic()
    dev_goal_82.setTitle("开发目标")
    dev_goal_82.addSubTopic().setTitle("内容运营模块")
    dev_goal_82.addSubTopic().setTitle("招投标模块")
    dev_goal_82.addSubTopic().setTitle("财务运营模块")
    dev_goal_82.addSubTopic().setTitle("雷鸟AR眼镜对接")
    dev_goal_82.addSubTopic().setTitle("智能复判服务体系")
    market_goal_82 = topic8_2.addSubTopic()
    market_goal_82.setTitle("市场目标")
    market_goal_82.addSubTopic().setTitle("获取种子用户")
    market_goal_82.addSubTopic().setTitle("收集反馈迭代")

    topic8_3 = topic8.addSubTopic()
    topic8_3.setTitle("第三阶段：生态建设（12个月）")
    dev_goal_83 = topic8_3.addSubTopic()
    dev_goal_83.setTitle("开发目标")
    dev_goal_83.addSubTopic().setTitle("完善开发者生态")
    dev_goal_83.addSubTopic().setTitle("API市场建设")
    dev_goal_83.addSubTopic().setTitle("行业解决方案")
    market_goal_83 = topic8_3.addSubTopic()
    market_goal_83.setTitle("市场目标")
    market_goal_83.addSubTopic().setTitle("建立渠道代理体系")
    market_goal_83.addSubTopic().setTitle("扩大市场份额")
    market_goal_83.addSubTopic().setTitle("品牌建设")

    # 9. 风险与挑战
    topic9 = root_topic.addSubTopic()
    topic9.setTitle("9. 风险与挑战")

    topic9_1 = topic9.addSubTopic()
    topic9_1.setTitle("技术风险")
    topic9_1.addSubTopic().setTitle("AI模型准确率")
    topic9_1.addSubTopic().setTitle("设备兼容性")
    topic9_1.addSubTopic().setTitle("系统稳定性")

    topic9_2 = topic9.addSubTopic()
    topic9_2.setTitle("市场风险")
    topic9_2.addSubTopic().setTitle("竞品竞争")
    topic9_2.addSubTopic().setTitle("客户教育成本")
    topic9_2.addSubTopic().setTitle("市场接受度")

    topic9_3 = topic9.addSubTopic()
    topic9_3.setTitle("运营风险")
    topic9_3.addSubTopic().setTitle("团队能力")
    topic9_3.addSubTopic().setTitle("资金储备")
    topic9_3.addSubTopic().setTitle("客户服务")

    # 10. 下一步行动
    topic10 = root_topic.addSubTopic()
    topic10.setTitle("10. 下一步行动")

    topic10_1 = topic10.addSubTopic()
    topic10_1.setTitle("产品规划")
    topic10_1.addSubTopic().setTitle("确定MVP功能范围")
    topic10_1.addSubTopic().setTitle("制定详细开发计划")
    topic10_1.addSubTopic().setTitle("设计产品原型")

    topic10_2 = topic10.addSubTopic()
    topic10_2.setTitle("团队建设")
    topic10_2.addSubTopic().setTitle("组建技术团队")
    topic10_2.addSubTopic().setTitle("招募产品/运营人员")
    topic10_2.addSubTopic().setTitle("建立组织架构")

    topic10_3 = topic10.addSubTopic()
    topic10_3.setTitle("市场验证")
    topic10_3.addSubTopic().setTitle("启动市场调研")
    topic10_3.addSubTopic().setTitle("寻找种子用户")
    topic10_3.addSubTopic().setTitle("收集需求反馈")

    topic10_4 = topic10.addSubTopic()
    topic10_4.setTitle("资源准备")
    topic10_4.addSubTopic().setTitle("融资计划")
    topic10_4.addSubTopic().setTitle("技术选型")
    topic10_4.addSubTopic().setTitle("供应商对接")

    # 保存
    xmind.save(workbook, str(output_path))
    print(f"XMind思维导图生成成功：{output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 XMind 思维导图")
    parser.add_argument(
        "--output",
        help="输出文件路径（.xmind）；不传则输出到 storage/skill_outputs/xmind-generation/",
    )
    parser.add_argument(
        "--template",
        help="可选：基于已有 .xmind 模板生成",
    )
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path()
    template_path = Path(args.template).expanduser().resolve() if args.template else None
    create_xmind(output_path=output_path, template_path=template_path)
