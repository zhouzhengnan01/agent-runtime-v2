from __future__ import annotations

import re
from typing import Any

KEYWORDS = (
    "deliverables-export",
    "交付物",
    "导出附件",
    "生成附件",
    "发企业微信",
    "发飞书",
    "操作说明",
    "实施步骤",
    "命令说明",
    "docx",
    "word",
    "txt",
)


def score_skill(
    skill_name: str,
    routing_text: str,
    attachments: list[Any],
    allowed_skills: list[str],
) -> int:
    if skill_name not in allowed_skills:
        return 0
    text = routing_text.lower()
    return 100 if any(keyword.lower() in text for keyword in KEYWORDS) else 0


def build_spec(
    skill_name: str,
    user_text: str,
    routing_text: str,
    attachments: list[Any],
    base_spec: dict[str, Any],
) -> dict[str, Any]:
    spec = dict(base_spec)
    commands = _extract_commands(routing_text)
    steps = _extract_steps(routing_text)
    spec.update(
        {
            "skill_name": skill_name,
            "title": spec.get("title") or "交付物导出",
            "commands": commands or f"需求说明：\n{routing_text.strip()}",
            "steps": steps or f"# 交付说明\n\n{routing_text.strip()}",
            "commands_name": "commands.txt",
            "steps_name": "steps.docx",
            "quality_requirements": ["commands_txt", "steps_docx", "non_empty_files"],
            "verification_rules": ["commands_txt", "steps_docx", "non_empty_files"],
        }
    )
    return spec


def _extract_commands(text: str) -> str:
    fenced = re.findall(r"```(?:bash|shell|sh|zsh)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced if block.strip())
    command_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(python|uv|pip|npm|pnpm|yarn|docker|kubectl|git|curl|ssh|scp|systemctl)\b", stripped):
            command_lines.append(stripped)
    return "\n".join(command_lines)


def _extract_steps(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if any(line.lstrip().startswith(("#", "1.", "-", "*")) for line in lines):
        return "\n".join(lines)
    return ""
