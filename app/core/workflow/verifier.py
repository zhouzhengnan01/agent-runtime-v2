from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.skills.runner import SkillRunResult
from app.schemas import ArtifactRef, VerificationCheck, VerificationResult


class OutputVerifier:
    """Content-level verifier for generated outputs."""

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def verify(self, spec: dict[str, Any], run_result: SkillRunResult, retry_count: int = 0) -> VerificationResult:
        skill_name = run_result.skill_name
        if skill_name == "behavior-detection":
            return self._verify_behavior(run_result, retry_count)

        checks: list[VerificationCheck] = [
            VerificationCheck(name="has_output", passed=bool(run_result.outputs), detail="至少生成一个输出文件"),
        ]
        for artifact in run_result.outputs:
            checks.extend(self._verify_artifact(skill_name, artifact))
        if skill_name == "drawio-generation":
            names = [artifact.name for artifact in run_result.outputs]
            checks.append(
                VerificationCheck(
                    name="has_drawio_source",
                    passed=any(name.endswith(".drawio") for name in names),
                    detail=", ".join(names),
                )
            )
            checks.append(
                VerificationCheck(
                    name="has_png_preview",
                    passed=any(name.endswith(".png") for name in names),
                    detail=", ".join(names),
                )
            )

        failed = [check.name for check in checks if not check.passed]
        return VerificationResult(passed=not failed, retry_count=retry_count, checks=checks, failed_checks=failed)

    def _verify_artifact(self, skill_name: str, artifact: ArtifactRef) -> list[VerificationCheck]:
        try:
            actual_path = self.artifact_store.resolve_virtual_path(artifact.thread_id, artifact.path)
        except ValueError as exc:
            return [VerificationCheck(name="artifact_path_safe", passed=False, detail=str(exc))]

        checks = [
            VerificationCheck(name="artifact_exists", passed=actual_path.is_file(), detail=artifact.name),
            VerificationCheck(name="artifact_non_empty", passed=actual_path.is_file() and actual_path.stat().st_size > 0, detail=artifact.name),
        ]
        if not actual_path.is_file():
            return checks

        if skill_name == "drawio-generation":
            if actual_path.suffix.lower() == ".png":
                checks.extend(self._verify_png_preview(actual_path))
            else:
                checks.extend(self._verify_drawio(actual_path))
        elif skill_name == "markdown-rendering":
            checks.extend(self._verify_markdown(actual_path))
        elif skill_name == "excel-generation":
            checks.extend(self._verify_xlsx(actual_path))
        elif skill_name == "pptx-generation":
            checks.extend(self._verify_pptx(actual_path))
        elif skill_name == "xmind-generation":
            checks.extend(self._verify_xmind(actual_path))
        else:
            checks.append(VerificationCheck(name="artifact_named", passed=bool(Path(artifact.name).name), detail=artifact.name))
        return checks

    @staticmethod
    def _verify_drawio(path: Path) -> list[VerificationCheck]:
        text = path.read_text(encoding="utf-8")
        vertex_count = text.count('vertex="1"')
        edge_count = text.count('edge="1"')
        fill_count = text.count("fillColor=")
        return [
            VerificationCheck(name="drawio_extension", passed=path.suffix.lower() == ".drawio", detail=path.name),
            VerificationCheck(name="drawio_xml", passed="<mxfile" in text and "<mxGraphModel" in text, detail="mxfile + graph model"),
            VerificationCheck(name="has_nodes", passed=vertex_count >= 8, detail=f"vertices={vertex_count}"),
            VerificationCheck(name="has_edges", passed=edge_count >= 6, detail=f"edges={edge_count}"),
            VerificationCheck(name="has_swimlanes", passed="swimlane" in text, detail="泳道布局"),
            VerificationCheck(name="has_semantic_colors", passed=fill_count >= 5, detail=f"fillColor={fill_count}"),
        ]

    @staticmethod
    def _verify_png_preview(path: Path) -> list[VerificationCheck]:
        data = path.read_bytes()
        width = int.from_bytes(data[16:20], "big") if len(data) >= 24 else 0
        height = int.from_bytes(data[20:24], "big") if len(data) >= 24 else 0
        return [
            VerificationCheck(name="png_extension", passed=path.suffix.lower() == ".png", detail=path.name),
            VerificationCheck(name="png_signature", passed=data.startswith(b"\x89PNG\r\n\x1a\n"), detail="PNG signature"),
            VerificationCheck(name="png_dimensions", passed=width >= 640 and height >= 360, detail=f"{width}x{height}"),
        ]

    @staticmethod
    def _verify_markdown(path: Path) -> list[VerificationCheck]:
        text = path.read_text(encoding="utf-8")
        return [
            VerificationCheck(name="markdown_extension", passed=path.suffix.lower() in {".md", ".markdown"}, detail=path.name),
            VerificationCheck(name="has_mermaid", passed="```mermaid" in text, detail="Mermaid block"),
            VerificationCheck(name="has_table", passed="| 字段 | 类型 | 说明 |" in text, detail="字段表格"),
            VerificationCheck(name="has_echarts_block", passed="```json echarts" in text, detail="ECharts 配置块"),
            VerificationCheck(name="has_attachment_block", passed="artifact://" in text, detail="附件块"),
        ]

    @staticmethod
    def _verify_xlsx(path: Path) -> list[VerificationCheck]:
        if not zipfile.is_zipfile(path):
            return [VerificationCheck(name="xlsx_zip", passed=False, detail="not a zip package")]
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            sheet1 = archive.read("xl/worksheets/sheet1.xml").decode("utf-8") if "xl/worksheets/sheet1.xml" in names else ""
            sheet2 = archive.read("xl/worksheets/sheet2.xml").decode("utf-8") if "xl/worksheets/sheet2.xml" in names else ""
        return [
            VerificationCheck(name="xlsx_extension", passed=path.suffix.lower() == ".xlsx", detail=path.name),
            VerificationCheck(name="xlsx_zip", passed=True, detail="zip package"),
            VerificationCheck(name="has_workbook", passed="xl/workbook.xml" in names, detail="workbook.xml"),
            VerificationCheck(name="has_two_sheets", passed={"xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"}.issubset(names), detail="设备模板 + 状态汇总"),
            VerificationCheck(name="has_freeze_panes", passed="<pane" in sheet1, detail="freeze panes"),
            VerificationCheck(name="has_data_validation", passed="<dataValidation" in sheet1, detail="在线状态枚举"),
            VerificationCheck(name="has_formulas", passed="<f>" in sheet2, detail="summary formulas"),
        ]

    @staticmethod
    def _verify_pptx(path: Path) -> list[VerificationCheck]:
        if not zipfile.is_zipfile(path):
            return [VerificationCheck(name="pptx_zip", passed=False, detail="not a zip package")]
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
        slide_count = len([name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")])
        slide_rel_count = len(
            [name for name in names if name.startswith("ppt/slides/_rels/slide") and name.endswith(".xml.rels")]
        )
        return [
            VerificationCheck(name="pptx_extension", passed=path.suffix.lower() == ".pptx", detail=path.name),
            VerificationCheck(name="pptx_zip", passed=True, detail="zip package"),
            VerificationCheck(name="has_content_types", passed="[Content_Types].xml" in names, detail="[Content_Types].xml"),
            VerificationCheck(name="has_presentation_xml", passed="ppt/presentation.xml" in names, detail="presentation.xml"),
            VerificationCheck(name="has_6_to_8_slides", passed=6 <= slide_count <= 8, detail=f"slides={slide_count}"),
            VerificationCheck(name="has_rels", passed="ppt/_rels/presentation.xml.rels" in names, detail="relationships"),
            VerificationCheck(name="has_slide_relationships", passed=slide_rel_count == slide_count, detail=f"slide_rels={slide_rel_count}"),
            VerificationCheck(name="has_slide_master", passed=any(name.startswith("ppt/slideMasters/") for name in names), detail="slide master"),
            VerificationCheck(name="has_slide_layout", passed=any(name.startswith("ppt/slideLayouts/") for name in names), detail="slide layout"),
            VerificationCheck(name="has_theme", passed=any(name.startswith("ppt/theme/") for name in names), detail="theme"),
            VerificationCheck(name="has_doc_props", passed={"docProps/core.xml", "docProps/app.xml"} <= names, detail="docProps"),
        ]

    @staticmethod
    def _verify_xmind(path: Path) -> list[VerificationCheck]:
        if not zipfile.is_zipfile(path):
            return [VerificationCheck(name="xmind_zip", passed=False, detail="not a zip package")]
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            content_raw = archive.read("content.json").decode("utf-8") if "content.json" in names else "[]"
        try:
            content = json.loads(content_raw)
        except json.JSONDecodeError:
            content = []
        topics = content[0].get("rootTopic", {}).get("children", {}).get("attached", []) if content else []
        child_count = sum(len(topic.get("children", {}).get("attached", [])) for topic in topics if isinstance(topic, dict))
        notes_count = sum(1 for topic in topics if isinstance(topic, dict) and topic.get("notes"))
        return [
            VerificationCheck(name="xmind_extension", passed=path.suffix.lower() == ".xmind", detail=path.name),
            VerificationCheck(name="xmind_zip", passed=True, detail="zip package"),
            VerificationCheck(name="has_content_json", passed="content.json" in names, detail="content.json"),
            VerificationCheck(name="has_multi_level_topics", passed=len(topics) >= 3 and child_count >= 6, detail=f"topics={len(topics)}, children={child_count}"),
            VerificationCheck(name="has_notes", passed=notes_count >= 1, detail=f"notes={notes_count}"),
        ]

    @staticmethod
    def _verify_behavior(run_result: SkillRunResult, retry_count: int) -> VerificationResult:
        data = run_result.data
        evidence_mode = data.get("evidence_mode")
        checks = [
            VerificationCheck(
                name="evidence_mode_declared",
                passed=evidence_mode in {"text_only", "visual_or_structured"},
                detail=str(evidence_mode),
            ),
            VerificationCheck(
                name="no_visual_score_without_evidence",
                passed=not (evidence_mode == "text_only" and float(data.get("confidence", 0.0)) > 0.0),
                detail=f"confidence={data.get('confidence')}",
            ),
            VerificationCheck(
                name="text_rule_confidence_separated",
                passed=evidence_mode != "text_only" or "text_rule_confidence" in data,
                detail=f"text_rule_confidence={data.get('text_rule_confidence')}",
            ),
        ]
        failed = [check.name for check in checks if not check.passed]
        return VerificationResult(passed=not failed, retry_count=retry_count, checks=checks, failed_checks=failed)
