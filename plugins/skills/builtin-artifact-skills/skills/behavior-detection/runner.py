from __future__ import annotations

from runner import SkillRunner


def run(skill_name, spec, paths, artifact_store):
    return SkillRunner(artifact_store).run(skill_name, spec, paths)


def format_reply(skill_name, verification, run_result):
    del skill_name, verification
    data = run_result.data
    decision = data.get("decision")
    if decision == "needs_visual_confirmation":
        return "文本规则判断：疑似命中行为规则；待上传图片、视频或结构化视觉证据后才能给出视觉识别结果。"
    return f"识别结果：{data.get('decision')} / {data.get('category')} / {data.get('confidence')}"

