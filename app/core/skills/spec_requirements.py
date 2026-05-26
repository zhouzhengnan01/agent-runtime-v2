from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def missing_required_spec_fields(skill_name: str, spec: Mapping[str, Any]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    if skill_name == "pptx-generation":
        if len(_slides(spec.get("slides"))) < 4:
            missing.append(
                {
                    "skill": skill_name,
                    "field": "slides",
                    "label": "至少 4 页结构化 slides（每页需包含 title）",
                }
            )
    elif skill_name == "xmind-generation":
        if not _string(spec.get("root_topic")) and not _string(spec.get("title")):
            missing.append(
                {
                    "skill": skill_name,
                    "field": "root_topic",
                    "label": "脑图主题 root_topic",
                }
            )
        if len(_topics(spec.get("topics"))) < 3:
            missing.append(
                {
                    "skill": skill_name,
                    "field": "topics",
                    "label": "至少 3 个 topics，且每个 topic 需要子节点",
                }
            )
    elif skill_name == "website-design":
        if not _string(spec.get("brand")):
            missing.append({"skill": skill_name, "field": "brand", "label": "品牌名或站点名 brand"})
        if not _string(spec.get("headline")):
            missing.append({"skill": skill_name, "field": "headline", "label": "首页主标题 headline"})
        if not _string(spec.get("subcopy")):
            missing.append({"skill": skill_name, "field": "subcopy", "label": "首页说明文案 subcopy"})
        if not _string(spec.get("primary_cta")):
            missing.append({"skill": skill_name, "field": "primary_cta", "label": "主按钮文案 primary_cta"})
        if len(_strings(spec.get("sections"))) < 3:
            missing.append(
                {
                    "skill": skill_name,
                    "field": "sections",
                    "label": "至少 3 个页面 sections",
                }
            )
    elif skill_name == "deliverables-export":
        if not _string(spec.get("commands")):
            missing.append({"skill": skill_name, "field": "commands", "label": "命令记录 commands"})
        if not _string(spec.get("steps")):
            missing.append({"skill": skill_name, "field": "steps", "label": "实施步骤 steps"})
    return dedupe_missing_fields(missing)


def dedupe_missing_fields(missing: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in missing:
        key = (item["skill"], item["field"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_missing_spec_reply(
    missing: list[dict[str, str]],
    header: str = "当前还不能生成有效结果，需要先补齐这些结构化字段：",
) -> str:
    lines = [header]
    lines.extend(f"- {item['skill']}：{item['label']}（{item['field']}）" for item in missing)
    lines.extend(
        [
            "",
            "请在下一条消息里补这些字段，或给出更具体的页面结构、PPT 大纲、脑图层级、命令记录和实施步骤。",
        ]
    )
    return "\n".join(lines)


def _string(value: object) -> str:
    return str(value or "").strip()


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _slides(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    slides: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        title = _string(item.get("title"))
        if not title:
            continue
        slides.append({"title": title, "kind": _string(item.get("kind")) or "content"})
    return slides


def _topics(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    topics: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        title = _string(item.get("title"))
        children = _strings(item.get("children"))
        if not title or not children:
            continue
        topics.append({"title": title, "children": children})
    return topics
