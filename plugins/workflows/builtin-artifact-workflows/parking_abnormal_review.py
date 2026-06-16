from __future__ import annotations

import hashlib
import importlib
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.config import AgentConfig
from app.core.diagnostics import diagnostic_json
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.core.skills import SkillDefinition, SkillRegistry, SkillRunner
from app.core.skills.runner_types import SkillRunResult
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


WORKFLOW_NAME = "parking_abnormal_review"
OUTPUT_NAME = "parking_abnormal_review_result.json"
VIDEO_FRAME_OUTPUT_DIR = "parking_abnormal_review_video_frames"
VIDEO_FRAME_MAX_ATTACHMENTS = 6
VIDEO_FRAME_MAX_CANDIDATES = 24
VIDEO_FRAME_MAX_WIDTH = 1280
VIDEO_FRAME_MIN_DIFFERENCE = 8.0
SKILL_CONTEXT_MAX_CHARS = 6000
SKILL_LLM_SELECTION_SCORE_GAP = 3.0
REVIEW_LLM_REPLY_LOG_MAX_CHARS = 4000


logger = logging.getLogger("uvicorn.error")


@dataclass
class VideoFrameCandidate:
    frame_index: int
    timestamp_ms: float
    sharpness: float
    brightness: float
    score: float
    thumbnail: Any
    frame: Any


@dataclass(frozen=True)
class ExtractedVideoFrame:
    path: Path
    source_video: str
    frame_index: int
    timestamp_ms: float
    sharpness: float
    brightness: float
    score: float


@dataclass(frozen=True)
class ReviewSkillCandidate:
    skill: SkillDefinition
    context: str
    score: float
    score_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewSkillSelection:
    skill_name: str
    method: str
    confidence: float
    reason: str
    candidates: tuple[ReviewSkillCandidate, ...]
    selected_context: str


class ParkingAbnormalReviewWorkflow:
    """Deterministic parking abnormal review workflow that returns only JSON."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        skill_registry: SkillRegistry | None = None,
        skill_runner: SkillRunner | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_runner = skill_runner or SkillRunner(artifact_store)

    def run_with_events(
        self,
        agent_config: AgentConfig,
        messages: list[Message],
        attachments: list[Attachment],
        thread_id: str | None,
        on_event: Callable[[ChatEvent], None] | None = None,
        workflow_name: str | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        started_at = time.perf_counter()
        runtime_options = runtime_options or RuntimeOptions()
        workflow = workflow_name or WORKFLOW_NAME
        paths = self.artifact_store.prepare_thread(thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id, on_emit=on_event)
        recorder.emit("run.started", {"workflow": workflow, "execution_mode": workflow})

        prompt_text = _last_user_text(messages)
        review_source_id = _review_source_id(prompt_text)
        objective = _objective(prompt_text)
        image_attachments = _image_attachments(attachments)
        video_attachments = _video_attachments(attachments)
        video_frame_attachments, video_frame_reports = _video_frame_attachments(video_attachments, paths)
        review_attachments = [*image_attachments, *video_frame_attachments]
        image_sources = _attachment_source_payloads(review_attachments)
        skill_selection = self._select_review_skill(
            agent_config,
            runtime_options,
            prompt_text,
            review_source_id,
            objective,
        )
        skill_result, skill_error = self._invoke_review_skill(
            skill_selection,
            prompt_text=prompt_text,
            review_source_id=review_source_id,
            objective=objective,
            attachments=review_attachments,
            paths=paths,
            recorder=recorder,
        )
        recorder.emit(
            "review.input",
            {
                "review_source_id": review_source_id,
                "objective": objective,
                "selected_skill": skill_selection.skill_name if skill_selection is not None else "",
                "skill_selection": _skill_selection_event_payload(skill_selection),
                "skill_error": skill_error,
                "attachment_count": len(attachments),
                "image_attachment_count": len(image_attachments),
                "video_attachment_count": len(video_attachments),
                "video_frame_attachment_count": len(video_frame_attachments),
                "image_sources": image_sources,
                "video_frame_reports": video_frame_reports,
            },
        )
        logger.info(
            "\n===== 复判图片来源 | parking review image sources =====\n"
            "reviewSourceId: %s\n"
            "识别目标: %s\n"
            "图片数量: %s\n"
            "图片链接/路径:\n%s\n"
            "===== 复判图片来源结束 =====",
            review_source_id,
            objective,
            len(image_sources),
            _pretty_payload_json(image_sources),
        )

        if not review_attachments:
            reply = _json_reply(
                [
                    {
                        "reviewSourceId": review_source_id,
                        "reviewEventId": _review_event_id(review_source_id, objective),
                        "hit": 0,
                        "result": _missing_media_result(video_attachments, video_frame_reports),
                    }
                ]
            )
        else:
            raw_reply = self._complete_review(
                agent_config,
                runtime_options,
                prompt_text,
                review_attachments,
                paths,
                review_source_id,
                objective,
                skill_selection,
                skill_result,
                skill_error,
            )
            logger.info(
                "parking review llm raw reply review_source_id=%s objective=%s attachment_count=%s "
                "skill_name=%s raw_chars=%s raw_reply=%s",
                review_source_id,
                objective,
                len(review_attachments),
                skill_selection.skill_name if skill_selection is not None else "",
                len(raw_reply),
                diagnostic_json({"reply": raw_reply}, max_chars=REVIEW_LLM_REPLY_LOG_MAX_CHARS),
            )
            recorder.emit(
                "review.llm.raw_reply",
                {
                    "review_source_id": review_source_id,
                    "objective": objective,
                    "attachment_count": len(review_attachments),
                    "skill_name": skill_selection.skill_name if skill_selection is not None else "",
                    "raw_chars": len(raw_reply),
                    "raw_reply": raw_reply,
                },
            )
            reply = _normalize_json_reply(raw_reply, review_source_id=review_source_id, objective=objective)
            logger.info(
                "\n===== 复判归一化结果 | parking review normalized result =====\n"
                "reviewSourceId: %s\n"
                "识别目标: %s\n"
                "结果字符数: %s\n"
                "复判结果:\n%s\n"
                "===== 复判归一化结果结束 =====",
                review_source_id,
                objective,
                len(reply),
                _pretty_review_json(reply),
            )
            recorder.emit(
                "review.normalized_result",
                {
                    "review_source_id": review_source_id,
                    "objective": objective,
                    "result_chars": len(reply),
                    "result": reply,
                },
            )

        logger.info(
            "\n===== 复判最终结果 | parking review final result =====\n"
            "reviewSourceId: %s\n"
            "识别目标: %s\n"
            "选中技能: %s\n"
            "图片数量: %s\n"
            "图片链接/路径:\n%s\n"
            "结果字符数: %s\n"
            "复判结果:\n%s\n"
            "===== 复判最终结果结束 =====",
            review_source_id,
            objective,
            skill_selection.skill_name if skill_selection is not None else "",
            len(image_sources),
            _pretty_payload_json(image_sources),
            len(reply),
            _pretty_review_json(reply),
        )

        artifact = self.artifact_store.write_text_artifact(paths, OUTPUT_NAME, reply)
        recorder.emit(
            "artifact.created",
            {"path": f"outputs/{artifact.name}", "name": artifact.name, "mime_type": artifact.mime_type},
        )
        recorder.emit("agent.message", {"text": reply})
        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=paths.thread_id,
            status="completed",
            reply=reply,
            metadata={
                "workflow": workflow,
                "review_source_id": review_source_id,
                "objective": objective,
                "artifact_path": f"outputs/{artifact.name}",
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        )
        recorder.emit("run.completed", {"result": result.model_dump()})
        return result, recorder.events

    def _select_review_skill(
        self,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        review_source_id: str,
        objective: str,
    ) -> ReviewSkillSelection | None:
        candidates = _review_skill_candidates(
            self.skill_registry,
            runtime_options.selected_skills,
            objective=objective,
            prompt_text=prompt_text,
        )
        if not candidates:
            return None
        if len(candidates) == 1:
            candidate = candidates[0]
            return ReviewSkillSelection(
                skill_name=candidate.skill.name,
                method="single_candidate",
                confidence=1.0,
                reason="只有一个候选 skill，直接用于复判。",
                candidates=tuple(candidates),
                selected_context=candidate.context,
            )

        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        score_gap = ranked[0].score - ranked[1].score
        if ranked[0].score > 0 and score_gap >= SKILL_LLM_SELECTION_SCORE_GAP:
            return ReviewSkillSelection(
                skill_name=ranked[0].skill.name,
                method="keyword_score",
                confidence=min(1.0, max(0.35, ranked[0].score / 12.0)),
                reason="根据 CV 标签、识别目标和 skill 描述关键词匹配选择。",
                candidates=tuple(ranked),
                selected_context=ranked[0].context,
            )

        llm_selection = _llm_select_review_skill(
            agent_config,
            runtime_options,
            prompt_text=prompt_text,
            review_source_id=review_source_id,
            objective=objective,
            candidates=ranked,
        )
        if llm_selection is not None:
            return llm_selection

        return ReviewSkillSelection(
            skill_name=ranked[0].skill.name,
            method="score_fallback",
            confidence=min(1.0, max(0.2, ranked[0].score / 12.0)),
            reason="大模型选择不可用，使用最高关键词匹配分数的 skill。",
            candidates=tuple(ranked),
            selected_context=ranked[0].context,
        )

    def _invoke_review_skill(
        self,
        selection: ReviewSkillSelection | None,
        *,
        prompt_text: str,
        review_source_id: str,
        objective: str,
        attachments: list[Attachment],
        paths: ThreadPaths,
        recorder: EventRecorder,
    ) -> tuple[SkillRunResult | None, str]:
        if selection is None:
            recorder.emit("review.skill_selection.completed", {"selected_skill": "", "reason": "no_candidate_skill"})
            return None, "no_candidate_skill"
        recorder.emit("review.skill_selection.completed", _skill_selection_event_payload(selection))
        spec = _review_skill_spec(
            selection,
            prompt_text=prompt_text,
            review_source_id=review_source_id,
            objective=objective,
            attachments=attachments,
        )
        recorder.emit("review.skill_invocation.started", {"skill_name": selection.skill_name, "spec": _public_spec(spec)})
        try:
            result = self.skill_runner.run(selection.skill_name, spec, paths, on_event=recorder.emit)
        except Exception as exc:
            error = str(exc)
            recorder.emit("review.skill_invocation.failed", {"skill_name": selection.skill_name, "error": error[:1000]})
            return None, error
        recorder.emit(
            "review.skill_invocation.completed",
            {
                "skill_name": selection.skill_name,
                "output_count": len(result.outputs),
                "data": result.data,
                "artifacts": [artifact.model_dump(mode="json") for artifact in result.outputs],
            },
        )
        return result, ""

    @staticmethod
    def _complete_review(
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: ThreadPaths,
        review_source_id: str,
        objective: str,
        skill_selection: ReviewSkillSelection | None,
        skill_result: SkillRunResult | None,
        skill_error: str,
    ) -> str:
        client = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
        evidence_context = _attachment_evidence_context(image_attachments)
        skill_context = _review_skill_prompt_context(skill_selection, skill_result, skill_error)
        system_prompt = (
            "你是机器视觉异常事件复判工作流。"
            "必须只基于用户文本和随附图片中可直接看见的内容判断识别目标是否命中。"
            "不要根据应用名称、场所类型、识别目标名称或经验常识推断画面外信息。"
            "只返回 JSON 数组，不要 Markdown，不要解释文字。"
        )
        user_prompt = (
            f"{prompt_text}\n\n"
            "输出要求：\n"
            "- 只输出 JSON 数组。\n"
            "- reviewSourceId 必须原样回填为："
            f"{review_source_id}\n"
            "- hit 只能是 0 或 1。\n"
            "- 没有命中时 hit=0，命中时 hit=1。\n"
            "- result 用中文说明复判依据。\n"
            "- result 必须描述图片中清晰可见的证据；图片中不可确认的内容不要编造。\n"
            "- 如果图片来自视频抽帧，应综合所有抽帧画面判断，不要只依据单帧偶然现象下结论。\n"
            "- 如存在已调用的复判 skill，必须优先遵循该 skill 的规则、判定边界和返回数据。\n"
            "- 不要使用“空旷路面”“停车位”“商场通道”“墙边堆放”等场景描述，除非这些内容在图片中清晰可见。\n"
            f"- 本次目标：{objective}\n"
            f"{skill_context}"
            f"{evidence_context}"
        )
        payload = {
            "role": "user",
            "content": user_prompt,
            "_attachments": _attachment_payloads(image_attachments, paths),
        }
        return client.complete_sync(system_prompt, [payload])


def _last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _review_source_id(text: str) -> str:
    patterns = [
        r"reviewSourceId为\[([^\]]+)\]",
        r"复判事件来源reviewSourceId为\[([^\]]+)\]",
        r"复判事件来源id为\[([^\]]+)\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return "unknown_review_source"


def _objective(text: str) -> str:
    match = re.search(r"识别目标为\[([^\]]+)\]", text)
    if match:
        return _canonical_objective(match.group(1).strip())
    payload = _parse_json_object(text)
    target = _target_from_payload(payload)
    if target:
        return _canonical_objective(target)
    inferred = _objective_from_text(text)
    if inferred:
        return inferred
    return "ClutterDetection"


def _canonical_objective(value: str) -> str:
    normalized = _normalize_match_text(value)
    aliases = {
        "fall": "FallDetection",
        "fallen": "FallDetection",
        "falldetection": "FallDetection",
        "跌倒": "FallDetection",
        "摔倒": "FallDetection",
        "倒地": "FallDetection",
        "人员跌倒": "FallDetection",
        "人员倒地": "FallDetection",
        "人员跌倒/倒地检测": "FallDetection",
        "人员跌倒倒地检测": "FallDetection",
        "smoking": "SmokingDetection",
        "smoke": "SmokingDetection",
        "smokingdetection": "SmokingDetection",
        "抽烟": "SmokingDetection",
        "吸烟": "SmokingDetection",
        "抽烟检测": "SmokingDetection",
        "吸烟检测": "SmokingDetection",
    }
    return aliases.get(normalized, value)


def _objective_from_text(text: str) -> str:
    normalized = _normalize_match_text(text)
    fall_terms = ("人员跌倒/倒地检测", "人员跌倒", "人员倒地", "跌倒", "摔倒", "倒地", "falldetection", "fall")
    if any(_normalize_match_text(term) in normalized for term in fall_terms):
        return "FallDetection"
    smoking_terms = ("smokingdetection", "抽烟检测", "吸烟检测", "抽烟", "吸烟", "smoking", "smoke")
    if any(_normalize_match_text(term) in normalized for term in smoking_terms):
        return "SmokingDetection"
    return ""


def _target_from_payload(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    for path in (
        ("taskTarget", "value"),
        ("taskTarget", "text"),
        ("schemaResults", "taskTarget", "value"),
        ("schemaResults", "taskTarget", "text"),
        ("value",),
        ("text",),
    ):
        value = _nested_string(payload, path)
        if value:
            return value
    return ""


def _nested_string(payload: dict[str, Any], path: tuple[str, ...]) -> str:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return current.strip() if isinstance(current, str) and current.strip() else ""


def _image_attachments(attachments: list[Attachment]) -> list[Attachment]:
    return [
        attachment
        for attachment in attachments
        if (attachment.mime_type or "").split(";", 1)[0].strip().lower().startswith("image/")
    ]


def _attachment_source_payloads(attachments: list[Attachment]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments, start=1):
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        source_url = _first_non_empty(
            metadata.get("original_uri"),
            metadata.get("url"),
            metadata.get("uri"),
            attachment.path,
        )
        payload: dict[str, Any] = {
            "index": index,
            "name": attachment.name,
            "mime_type": attachment.mime_type,
            "path": attachment.path,
            "url": source_url,
        }
        for key in (
            "dataId",
            "data_id",
            "sourceId",
            "source_id",
            "timestamp",
            "sha1",
            "size",
            "source",
            "source_video",
            "frame_index",
            "timestamp_ms",
        ):
            value = metadata.get(key)
            if value not in (None, ""):
                payload[key] = value
        payloads.append(payload)
    return payloads


def _video_attachments(attachments: list[Attachment]) -> list[Attachment]:
    return [
        attachment
        for attachment in attachments
        if (attachment.mime_type or "").split(";", 1)[0].strip().lower().startswith("video/")
    ]


def _review_skill_candidates(
    skill_registry: SkillRegistry,
    selected_skills: list[str],
    *,
    objective: str,
    prompt_text: str,
) -> list[ReviewSkillCandidate]:
    candidate_names = _expanded_candidate_skill_names(skill_registry, selected_skills)
    candidates: list[ReviewSkillCandidate] = []
    for name in candidate_names:
        try:
            skill = skill_registry.get(name)
        except KeyError:
            continue
        context = _skill_context(skill)
        score, reasons = _score_review_skill(skill, context, objective=objective, prompt_text=prompt_text)
        candidates.append(ReviewSkillCandidate(skill=skill, context=context, score=score, score_reasons=tuple(reasons)))
    return candidates


def _expanded_candidate_skill_names(skill_registry: SkillRegistry, selected_skills: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_name in selected_skills:
        name = str(raw_name or "").strip()
        if not name:
            continue
        try:
            skill = skill_registry.get(name)
        except KeyError:
            _append_unique(names, seen, name)
            continue
        if skill.composite:
            for child in skill.child_skills:
                _append_unique(names, seen, child)
            continue
        _append_unique(names, seen, name)
    return names


def _append_unique(target: list[str], seen: set[str], value: str) -> None:
    normalized = value.strip()
    if normalized and normalized not in seen:
        target.append(normalized)
        seen.add(normalized)


def _skill_context(skill: SkillDefinition) -> str:
    parts = [
        f"skill_name: {skill.name}",
        f"description: {skill.description}",
    ]
    routing = skill.routing if isinstance(skill.routing, dict) else {}
    if routing:
        parts.append("routing: " + json.dumps(routing, ensure_ascii=False, default=str))
    skill_md = _read_skill_markdown(skill)
    if skill_md:
        parts.append("SKILL.md:\n" + skill_md)
    context = "\n".join(parts).strip()
    if len(context) > SKILL_CONTEXT_MAX_CHARS:
        return context[:SKILL_CONTEXT_MAX_CHARS] + "\n...[truncated]"
    return context


def _read_skill_markdown(skill: SkillDefinition) -> str:
    candidates: list[Path] = []
    if skill.manifest_path is not None:
        candidates.append(skill.manifest_path.parent / "SKILL.md")
    if skill.plugin_root is not None:
        candidates.append(skill.plugin_root / "SKILL.md")
        candidates.extend(skill.plugin_root.glob("skills/*/SKILL.md"))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
    return ""


def _score_review_skill(
    skill: SkillDefinition,
    context: str,
    *,
    objective: str,
    prompt_text: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    haystack = _normalize_match_text(" ".join([skill.name, skill.description, context]))
    objective_terms = _objective_terms(objective, prompt_text)
    for term in objective_terms:
        normalized = _normalize_match_text(term)
        if not normalized:
            continue
        if normalized == _normalize_match_text(skill.name):
            score += 8.0
            reasons.append(f"skill_name={term}")
        elif normalized in haystack:
            weight = 3.0 if len(normalized) >= 4 else 1.0
            score += weight
            reasons.append(f"keyword={term}")
    return score, reasons


def _objective_terms(objective: str, prompt_text: str) -> list[str]:
    raw_terms = [objective]
    raw_terms.extend(re.findall(r"\[([^\]]+)\]", prompt_text))
    payload = _parse_json_object(prompt_text)
    if payload is not None:
        raw_terms.extend(_target_terms_from_payload(payload))
    aliases = {
        "clutterdetection": ["clutter", "杂物", "杂物检测", "堆积", "占道"],
        "smokingdetection": ["smoking", "smoke", "抽烟", "吸烟"],
        "fightdetection": ["fight", "fighting", "打架", "斗殴", "冲突"],
        "falldetection": ["fall", "fallen", "跌倒", "摔倒", "倒地", "人员跌倒", "人员跌倒/倒地检测"],
    }
    normalized_objective = _normalize_match_text(objective)
    raw_terms.extend(aliases.get(normalized_objective, []))
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        cleaned = str(term or "").strip()
        normalized = _normalize_match_text(cleaned)
        if cleaned and normalized not in seen:
            terms.append(cleaned)
            seen.add(normalized)
    return terms


def _target_terms_from_payload(payload: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for path in (
        ("taskTarget", "value"),
        ("taskTarget", "text"),
        ("schemaResults", "taskTarget", "value"),
        ("schemaResults", "taskTarget", "text"),
        ("value",),
        ("text",),
        ("modelId",),
        ("modelName",),
        ("taskName",),
        ("taskTarget",),
    ):
        if path == ("taskTarget",):
            target = payload.get("taskTarget")
            if isinstance(target, dict):
                terms.extend(str(value).strip() for value in target.values() if isinstance(value, str) and value.strip())
            continue
        value = _nested_string(payload, path)
        if value:
            terms.append(value)
    return terms


def _normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _llm_select_review_skill(
    agent_config: AgentConfig,
    runtime_options: RuntimeOptions,
    *,
    prompt_text: str,
    review_source_id: str,
    objective: str,
    candidates: list[ReviewSkillCandidate],
) -> ReviewSkillSelection | None:
    client = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    if not client.configured:
        return None
    allowed = {candidate.skill.name: candidate for candidate in candidates}
    system_prompt = (
        "你是复判工作流的 skill 路由器。"
        "只能从候选 skill_name 中选择一个最适合本次 CV 复判目标的 skill。"
        "只返回 JSON 对象，不要 Markdown。"
    )
    candidate_payload = [
        {
            "skill_name": candidate.skill.name,
            "description": candidate.skill.description,
            "score": candidate.score,
            "score_reasons": list(candidate.score_reasons),
            "context_preview": candidate.context[:1200],
        }
        for candidate in candidates
    ]
    user_prompt = (
        f"reviewSourceId: {review_source_id}\n"
        f"objective: {objective}\n"
        f"用户请求:\n{prompt_text[:4000]}\n\n"
        "候选 skills:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "返回格式：{\"skill_name\":\"候选之一\",\"confidence\":0.0到1.0,\"reason\":\"选择原因\"}"
    )
    try:
        raw_reply = client.complete_sync(system_prompt, [{"role": "user", "content": user_prompt}])
    except Exception:
        return None
    parsed = _parse_json_object(raw_reply)
    if parsed is None:
        return None
    skill_name = str(parsed.get("skill_name") or "").strip()
    candidate = allowed.get(skill_name)
    if candidate is None:
        return None
    return ReviewSkillSelection(
        skill_name=skill_name,
        method="llm",
        confidence=_bounded_float(parsed.get("confidence"), default=0.5, minimum=0.0, maximum=1.0),
        reason=str(parsed.get("reason") or "大模型在候选 skill 中选择。")[:1000],
        candidates=tuple(candidates),
        selected_context=candidate.context,
    )


def _review_skill_spec(
    selection: ReviewSkillSelection,
    *,
    prompt_text: str,
    review_source_id: str,
    objective: str,
    attachments: list[Attachment],
) -> dict[str, Any]:
    attachment_payloads = [attachment.model_dump(mode="python") for attachment in attachments]
    return {
        "skill_name": selection.skill_name,
        "objective": objective,
        "review_source_id": review_source_id,
        "event_id": review_source_id,
        "prompt": prompt_text,
        "text_rule_candidates": [objective],
        "has_visual_evidence": bool(attachments),
        "attachments": attachment_payloads,
        "review_context": {
            "selection_method": selection.method,
            "selection_confidence": selection.confidence,
            "selection_reason": selection.reason,
        },
    }


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return _redact_base64_fields(spec)


def _redact_base64_fields(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"data_base64", "dataBase64", "base64", "content", "fileContent", "file_content"} and isinstance(item, str):
                redacted[key] = f"<base64 omitted chars={len(item)}>" if item else item
            else:
                redacted[key] = _redact_base64_fields(item)
        return redacted
    if isinstance(value, list):
        return [_redact_base64_fields(item) for item in value]
    return value


def _skill_selection_event_payload(selection: ReviewSkillSelection | None) -> dict[str, Any]:
    if selection is None:
        return {"selected_skill": "", "candidate_skills": [], "method": "none", "confidence": 0.0, "reason": ""}
    return {
        "selected_skill": selection.skill_name,
        "method": selection.method,
        "confidence": selection.confidence,
        "reason": selection.reason,
        "candidate_skills": [
            {
                "skill_name": candidate.skill.name,
                "description": candidate.skill.description,
                "score": candidate.score,
                "score_reasons": list(candidate.score_reasons),
            }
            for candidate in selection.candidates
        ],
    }


def _review_skill_prompt_context(
    selection: ReviewSkillSelection | None,
    skill_result: SkillRunResult | None,
    skill_error: str,
) -> str:
    if selection is None:
        return "\n复判 skill：未选择到可用 skill，请仅依据图片/视频证据和本次目标保守判断。\n"
    parts = [
        "\n已选择并调用的复判 skill：",
        f"- skill_name: {selection.skill_name}",
        f"- selection_method: {selection.method}",
        f"- selection_reason: {selection.reason}",
        "\n该 skill 的规则/说明：",
        selection.selected_context,
    ]
    if skill_result is not None:
        parts.extend(
            [
                "\n该 skill 的调用结果：",
                json.dumps(
                    {
                        "skill_name": skill_result.skill_name,
                        "data": skill_result.data,
                        "artifacts": [artifact.model_dump(mode="json") for artifact in skill_result.outputs],
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ]
        )
    elif skill_error:
        parts.extend(["\n该 skill 调用失败：", skill_error[:1000]])
    return "\n".join(parts) + "\n"


def _video_frame_attachments(
    attachments: list[Attachment],
    paths: ThreadPaths,
) -> tuple[list[Attachment], list[dict[str, Any]]]:
    frame_attachments: list[Attachment] = []
    reports: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments, start=1):
        source_path = _attachment_local_path(str(attachment.path or ""), paths)
        report: dict[str, Any] = {
            "name": attachment.name,
            "mime_type": attachment.mime_type,
            "path": attachment.path,
        }
        if source_path is None or not source_path.is_file():
            report.update({"status": "skipped", "error_code": "video_file_not_found"})
            reports.append(report)
            continue
        try:
            frames = _extract_video_frames(
                source_path,
                paths.workspace / VIDEO_FRAME_OUTPUT_DIR,
                prefix=f"video_{index}_{_safe_name(source_path.stem)}_{_short_hash(source_path)}",
            )
        except Exception as exc:
            report.update({"status": "failed", "error_code": _video_extract_error_code(exc), "error": str(exc)[:500]})
            reports.append(report)
            continue
        for frame in frames:
            frame_attachments.append(
                Attachment(
                    name=frame.path.name,
                    path=str(frame.path),
                    mime_type="image/jpeg",
                    metadata={
                        "source": "video_frame",
                        "source_video": frame.source_video,
                        "frame_index": frame.frame_index,
                        "timestamp_ms": round(frame.timestamp_ms, 3),
                        "sharpness": round(frame.sharpness, 3),
                        "brightness": round(frame.brightness, 3),
                        "score": round(frame.score, 3),
                    },
                )
            )
        report.update(
            {
                "status": "completed",
                "frame_count": len(frames),
                "frames": [
                    {
                        "name": frame.path.name,
                        "frame_index": frame.frame_index,
                        "timestamp_ms": round(frame.timestamp_ms, 3),
                        "sharpness": round(frame.sharpness, 3),
                        "brightness": round(frame.brightness, 3),
                    }
                    for frame in frames
                ],
            }
        )
        reports.append(report)
    return frame_attachments, reports


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    prefix: str,
    max_frames: int = VIDEO_FRAME_MAX_ATTACHMENTS,
    max_candidates: int = VIDEO_FRAME_MAX_CANDIDATES,
) -> list[ExtractedVideoFrame]:
    cv2 = _import_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob(f"{prefix}_*.jpg"):
        stale.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        candidates = _read_video_frame_candidates(
            cv2,
            capture,
            fps=fps,
            frame_count=frame_count,
            max_candidates=max_candidates,
        )
    finally:
        capture.release()

    if not candidates:
        raise ValueError(f"No readable frames extracted from video: {video_path}")

    selected = _select_video_frame_candidates(cv2, candidates, max_frames=max_frames)
    extracted: list[ExtractedVideoFrame] = []
    for output_index, candidate in enumerate(selected, start=1):
        resized = _resize_frame(cv2, candidate.frame, max_width=VIDEO_FRAME_MAX_WIDTH)
        frame_path = output_dir / f"{prefix}_{output_index:02d}_{int(candidate.timestamp_ms):07d}ms.jpg"
        if not cv2.imwrite(str(frame_path), resized):
            raise ValueError(f"OpenCV failed to write extracted frame: {frame_path}")
        extracted.append(
            ExtractedVideoFrame(
                path=frame_path,
                source_video=str(video_path),
                frame_index=candidate.frame_index,
                timestamp_ms=candidate.timestamp_ms,
                sharpness=candidate.sharpness,
                brightness=candidate.brightness,
                score=candidate.score,
            )
        )
    return extracted


def _import_cv2() -> Any:
    try:
        return importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError("OpenCV is unavailable; install opencv-python to enable video frame extraction.") from exc


def _read_video_frame_candidates(
    cv2: Any,
    capture: Any,
    *,
    fps: float,
    frame_count: int,
    max_candidates: int,
) -> list[VideoFrameCandidate]:
    if frame_count > 0:
        return [
            candidate
            for index in _candidate_frame_indices(frame_count, max_candidates)
            if (candidate := _read_video_frame_at(cv2, capture, index, fps=fps)) is not None
        ]
    return _read_sequential_video_frame_candidates(cv2, capture, fps=fps, max_candidates=max_candidates)


def _read_video_frame_at(cv2: Any, capture: Any, frame_index: int, *, fps: float) -> VideoFrameCandidate | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    timestamp_ms = (frame_index / fps * 1000.0) if fps > 0 else float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    return _video_frame_candidate(cv2, frame, frame_index=frame_index, timestamp_ms=timestamp_ms)


def _read_sequential_video_frame_candidates(
    cv2: Any,
    capture: Any,
    *,
    fps: float,
    max_candidates: int,
) -> list[VideoFrameCandidate]:
    candidates: list[VideoFrameCandidate] = []
    frame_index = 0
    stride = 15
    max_reads = max_candidates * stride * 2
    while len(candidates) < max_candidates and frame_index < max_reads:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if frame_index % stride == 0:
            timestamp_ms = (
                (frame_index / fps * 1000.0)
                if fps > 0
                else float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            )
            candidates.append(_video_frame_candidate(cv2, frame, frame_index=frame_index, timestamp_ms=timestamp_ms))
        frame_index += 1
    return candidates


def _video_frame_candidate(cv2: Any, frame: Any, *, frame_index: int, timestamp_ms: float) -> VideoFrameCandidate:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0) * 100.0
    score = sharpness + brightness_score
    thumbnail = cv2.resize(gray, (96, 54))
    return VideoFrameCandidate(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        sharpness=sharpness,
        brightness=brightness,
        score=score,
        thumbnail=thumbnail,
        frame=frame,
    )


def _candidate_frame_indices(frame_count: int, max_candidates: int) -> list[int]:
    if frame_count <= 0 or max_candidates <= 0:
        return []
    count = min(frame_count, max_candidates)
    if count == 1:
        return [0]
    return sorted({round(index * (frame_count - 1) / (count - 1)) for index in range(count)})


def _select_video_frame_candidates(
    cv2: Any,
    candidates: list[VideoFrameCandidate],
    *,
    max_frames: int,
) -> list[VideoFrameCandidate]:
    selected: list[VideoFrameCandidate] = []
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    for candidate in ranked:
        if len(selected) >= max_frames:
            break
        if all(
            _thumbnail_difference(cv2, candidate.thumbnail, chosen.thumbnail) >= VIDEO_FRAME_MIN_DIFFERENCE
            for chosen in selected
        ):
            selected.append(candidate)
    if len(selected) < max_frames:
        selected_ids = {id(candidate) for candidate in selected}
        for candidate in ranked:
            if len(selected) >= max_frames:
                break
            if id(candidate) not in selected_ids:
                selected.append(candidate)
                selected_ids.add(id(candidate))
    return sorted(selected, key=lambda item: item.frame_index)


def _thumbnail_difference(cv2: Any, left: Any, right: Any) -> float:
    return float(cv2.absdiff(left, right).mean())


def _resize_frame(cv2: Any, frame: Any, *, max_width: int) -> Any:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    target_height = max(1, round(height * (max_width / width)))
    return cv2.resize(frame, (max_width, target_height))


def _video_extract_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "opencv is unavailable" in message:
        return "opencv_unavailable"
    if "cannot open video" in message:
        return "video_open_failed"
    if "no readable frames" in message:
        return "video_no_readable_frames"
    return "video_frame_extract_failed"


def _attachment_payloads(attachments: list[Attachment], paths: ThreadPaths) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for attachment in attachments:
        payload = attachment.model_dump(mode="python")
        local_path = _attachment_local_path(str(attachment.path or ""), paths)
        if local_path is not None:
            payload["_local_path"] = str(local_path)
        payloads.append(payload)
    return payloads


def _attachment_local_path(raw_path: str, paths: ThreadPaths) -> Path | None:
    normalized = raw_path.replace("\\", "/").strip()
    if not normalized:
        return None
    virtual_roots = {
        "/mnt/user-data/uploads": paths.uploads,
        "/mnt/user-data/workspace": paths.workspace,
        "/mnt/user-data/outputs": paths.outputs,
    }
    for prefix, root in virtual_roots.items():
        if normalized == prefix or normalized.startswith(prefix + "/"):
            suffix = normalized[len(prefix) :].lstrip("/")
            candidate = (root / suffix).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                return None
            return candidate
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        return candidate.resolve()
    except OSError:
        return None


def _attachment_evidence_context(attachments: list[Attachment]) -> str:
    video_frames: list[str] = []
    for index, attachment in enumerate(attachments, start=1):
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        if metadata.get("source") != "video_frame":
            continue
        timestamp_ms = metadata.get("timestamp_ms")
        timestamp_text = _format_timestamp_ms(timestamp_ms) if isinstance(timestamp_ms, int | float) else "未知时间"
        video_frames.append(f"{index}. {attachment.name}，来源视频抽帧，时间点 {timestamp_text}")
    if not video_frames:
        return ""
    return "\n视频抽帧说明：\n" + "\n".join(video_frames) + "\n"


def _format_timestamp_ms(value: int | float) -> str:
    seconds = max(0.0, float(value) / 1000.0)
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    if minutes:
        return f"{minutes}分{remaining:.1f}秒"
    return f"{remaining:.1f}秒"


def _missing_media_result(video_attachments: list[Attachment], video_frame_reports: list[dict[str, Any]]) -> str:
    if not video_attachments:
        return "无法完成复判：当前请求未提供可访问的图片/视频文件，无法依据画面内容判断是否命中。"
    errors = [
        str(report.get("error_code") or report.get("status") or "unknown")
        for report in video_frame_reports
        if report.get("status") != "completed" or not report.get("frame_count")
    ]
    if errors:
        return f"无法完成复判：已收到视频文件，但视频抽帧失败（{';'.join(errors)}），无法依据画面内容判断是否命中。"
    return "无法完成复判：已收到视频文件，但未提取到可用于复判的视频画面。"


def _normalize_json_reply(raw_reply: str, *, review_source_id: str, objective: str) -> str:
    parsed = _parse_json_array(raw_reply)
    if parsed is None:
        parsed = [
            {
                "reviewSourceId": review_source_id,
                "reviewEventId": _review_event_id(review_source_id, objective),
                "hit": 0,
                "result": f"无法完成复判：模型未返回合法 JSON 数组。原始回复：{raw_reply.strip()[:500]}",
            }
        ]
    normalized = []
    for index, item in enumerate(parsed, start=1):
        event = item if isinstance(item, dict) else {}
        normalized.append(
            {
                "reviewSourceId": str(event.get("reviewSourceId") or review_source_id),
                "reviewEventId": str(event.get("reviewEventId") or _review_event_id(review_source_id, objective, index)),
                "hit": 1 if event.get("hit") == 1 else 0,
                "result": str(event.get("result") or "复判完成。"),
            }
        )
    return _json_reply(normalized)


def _parse_json_array(value: str) -> list[Any] | None:
    text = value.strip()
    if not text:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    for candidate in (text, _extract_first_json_array(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    return None


def _pretty_review_json(value: str) -> str:
    parsed = _parse_json_array(value)
    if parsed is None:
        text = value.strip()
        if len(text) <= REVIEW_LLM_REPLY_LOG_MAX_CHARS:
            return text
        return f"{text[:REVIEW_LLM_REPLY_LOG_MAX_CHARS]}...<truncated chars={len(text) - REVIEW_LLM_REPLY_LOG_MAX_CHARS}>"
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _pretty_payload_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= REVIEW_LLM_REPLY_LOG_MAX_CHARS:
        return text
    return f"{text[:REVIEW_LLM_REPLY_LOG_MAX_CHARS]}...<truncated chars={len(text) - REVIEW_LLM_REPLY_LOG_MAX_CHARS}>"


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, str):
            return str(value)
    return ""


def _parse_json_object(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if not text:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _extract_first_json_array(text: str) -> str | None:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


def _review_event_id(review_source_id: str, objective: str, index: int = 1) -> str:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", review_source_id).strip("_") or "review"
    safe_objective = re.sub(r"[^A-Za-z0-9_.-]+", "_", objective).strip("_") or "event"
    return f"{safe_source}_{safe_objective}_{index:02d}"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "video"


def _short_hash(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]


def _json_reply(items: list[dict[str, Any]]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
