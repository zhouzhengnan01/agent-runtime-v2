from __future__ import annotations

from runner import SkillRunner


def run(skill_name, spec, paths, artifact_store):
    return SkillRunner(artifact_store).run(skill_name, spec, paths)


def format_reply(skill_name, verification, run_result):
    del skill_name, verification
    data = run_result.data
    if data.get("review_decision") == "need_more_evidence":
        return "复判结论：证据不足。已生成复判报告，需补充图片、视频或结构化视觉证据后再确认事件。"
    return f"复判结论：{data.get('review_decision')}，风险等级：{data.get('risk_level')}。已生成复判报告。"
