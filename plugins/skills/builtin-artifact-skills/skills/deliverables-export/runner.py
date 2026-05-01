from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Mapping
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from app.core.artifacts.store import ArtifactStore, ThreadPaths
from app.core.skills.runner_types import SkillRunResult


def run(
    skill_name: str,
    spec: dict[str, Any],
    paths: ThreadPaths,
    artifact_store: ArtifactStore,
) -> SkillRunResult:
    commands = str(spec.get("commands") or spec.get("summary") or "").strip()
    steps = str(spec.get("steps") or spec.get("summary") or commands).strip()
    if not commands:
        commands = "No command notes were provided."
    if not steps:
        steps = "# Implementation Steps\n\nNo implementation steps were provided."

    commands_name = _safe_output_name(str(spec.get("commands_name") or "commands.txt"), ".txt")
    steps_name = _safe_output_name(str(spec.get("steps_name") or "steps.docx"), ".docx")
    txt_artifact = artifact_store.write_text_artifact(paths, commands_name, commands)
    docx_artifact = artifact_store.write_bytes_artifact(paths, steps_name, _docx_from_markdown(steps))
    return SkillRunResult(
        skill_name=skill_name,
        outputs=[txt_artifact, docx_artifact],
        data={
            "commands_chars": len(commands),
            "steps_chars": len(steps),
            "commands_name": commands_name,
            "steps_name": steps_name,
        },
    )


def _safe_output_name(value: str, suffix: str) -> str:
    name = re.sub(r"[\\/]+", "-", value.strip()) or f"deliverable{suffix}"
    if not name.lower().endswith(suffix):
        name = f"{name}{suffix}"
    return name


def _docx_from_markdown(markdown: str) -> bytes:
    paragraphs = [_paragraph_xml(kind, text) for kind, text in _markdown_blocks(markdown)]
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
        "</w:body></w:document>"
    )
    return _zip_bytes(
        {
            "[Content_Types].xml": (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ),
            "_rels/.rels": (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                'Target="word/document.xml"/></Relationships>'
            ),
            "word/document.xml": document,
        }
    )


def _markdown_blocks(markdown: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    in_code = False
    code_lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                blocks.append(("code", "\n".join(code_lines)))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line.strip():
            blocks.append(("blank", ""))
        elif line.startswith("#"):
            blocks.append(("heading", line.lstrip("#").strip()))
        elif line.lstrip().startswith(("- ", "* ")):
            blocks.append(("bullet", line.lstrip()[2:].strip()))
        else:
            blocks.append(("text", line))
    if in_code and code_lines:
        blocks.append(("code", "\n".join(code_lines)))
    return blocks or [("text", markdown)]


def _paragraph_xml(kind: str, text: str) -> str:
    style = ""
    if kind == "heading":
        style = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
    elif kind == "bullet":
        text = f"- {text}"
    elif kind == "code":
        style = '<w:pPr><w:pStyle w:val="NoSpacing"/></w:pPr>'
    escaped = xml_escape(text)
    return f"<w:p>{style}<w:r><w:t xml:space=\"preserve\">{escaped}</w:t></w:r></w:p>"


def _zip_bytes(files: Mapping[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return buffer.getvalue()
