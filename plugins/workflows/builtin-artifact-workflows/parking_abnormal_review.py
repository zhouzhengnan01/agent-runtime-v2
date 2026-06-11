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
from urllib.parse import urlparse

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.config import AgentConfig
from app.core.diagnostics import diagnostic_json, env_flag, env_int
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.core.skills import SkillDefinition, SkillRegistry, SkillRunner
from app.core.skills.runner_types import SkillRunResult
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


WORKFLOW_NAME = "parking_abnormal_review"
OUTPUT_NAME = "parking_abnormal_review_result.json"
VIDEO_FRAME_OUTPUT_DIR = "parking_abnormal_review_video_frames"
VIDEO_FRAME_MAX_ATTACHMENTS = max(1, env_int("PARKING_REVIEW_VIDEO_FRAME_MAX_ATTACHMENTS", 4))
VIDEO_FRAME_MAX_CANDIDATES = max(
    VIDEO_FRAME_MAX_ATTACHMENTS,
    env_int("PARKING_REVIEW_VIDEO_FRAME_MAX_CANDIDATES", 8),
)
VIDEO_FRAME_MAX_WIDTH = max(320, env_int("PARKING_REVIEW_VIDEO_FRAME_MAX_WIDTH", 960))
VIDEO_FRAME_SCORE_MAX_WIDTH = max(64, env_int("PARKING_REVIEW_VIDEO_FRAME_SCORE_MAX_WIDTH", 240))
VIDEO_FRAME_MIN_DIFFERENCE = 8.0
VIDEO_FRAME_CANDIDATE_MULTIPLIER = max(1, env_int("PARKING_REVIEW_VIDEO_FRAME_CANDIDATE_MULTIPLIER", 2))
VIDEO_FRAME_MIN_CANDIDATES = max(1, env_int("PARKING_REVIEW_VIDEO_FRAME_MIN_CANDIDATES", 4))
VIDEO_FRAME_SEQUENTIAL_SCAN_MAX_FRAMES = max(
    1,
    env_int("PARKING_REVIEW_VIDEO_FRAME_SEQUENTIAL_SCAN_MAX_FRAMES", 150),
)
VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST = max(
    0,
    env_int("PARKING_REVIEW_VIDEO_SKIP_WHEN_IMAGE_COUNT_AT_LEAST", 4),
)
SKILL_CONTEXT_MAX_CHARS = 6000
SKILL_LLM_SELECTION_SCORE_GAP = 3.0
SKILL_AUTO_CANDIDATE_MIN_SCORE = 2.0
REVIEW_SKILL_EXCLUDED_NAMES = frozenset({"behavior-detection"})
SMOKING_REVIEW_SKILL_NAME = "smoking-review"
SMOKING_OBJECTIVE = "smokingdetection"
SMOKING_EXPLICIT_TERMS = ("SmokingDetection", "抽烟检测", "吸烟检测", "持烟检测", "抽烟", "吸烟", "持烟")
SMOKING_EXCLUDED_OBJECTIVES = frozenset(
    {
        "clutterdetection",
        "garbageoverflowdetection",
        "firedetection",
        "parkingabnormaldetection",
        "parkingviolationdetection",
        "parkingcongestiondetection",
    }
)
REVIEW_LLM_TIMEOUT_SECONDS = max(1, env_int("PARKING_REVIEW_LLM_TIMEOUT_SECONDS", 90))
REVIEW_ENABLE_LLM_SKILL_ROUTER = env_flag("PARKING_REVIEW_ENABLE_LLM_SKILL_ROUTER", "0")
VIDEO_FRAME_TIMEOUT_SECONDS = max(1, env_int("PARKING_REVIEW_VIDEO_FRAME_TIMEOUT_SECONDS", 30))
REVIEW_LLM_REPLY_LOG_MAX_CHARS = max(200, env_int("PARKING_REVIEW_REPLY_LOG_MAX_CHARS", 1200))
PARKING_REVIEW_TRACE_DETAILS = env_flag("PARKING_REVIEW_TRACE_DETAILS", "0")
PARKING_REVIEW_LOG_INPUT_SUMMARY = env_flag("PARKING_REVIEW_LOG_INPUT_SUMMARY", "0")
PARKING_REVIEW_LOG_LINK_LIMIT = max(0, env_int("PARKING_REVIEW_LOG_LINK_LIMIT", 5))
OBJECTIVE_LABEL_ALIASES = frozenset(
    _normalize
    for _normalize in (
        "场景",
        "任务名称",
        "模型名称",
        "识别目标",
        "检测目标",
        "算法",
        "算法模型",
        "模型",
        "置信度",
    )
)
OBJECTIVE_METADATA_KEYS = (
    "target",
    "targetName",
    "target_name",
    "taskTarget",
    "task_target",
    "taskName",
    "task_name",
    "sourceTaskName",
    "source_task_name",
    "edgeTaskName",
    "edge_task_name",
    "sourceTaskId",
    "source_task_id",
    "scene",
    "sceneName",
    "scene_name",
    "category",
    "categoryName",
    "category_name",
    "algorithm",
    "algorithmName",
    "algorithm_name",
    "algorithmModel",
    "algorithm_model",
    "model",
    "modelName",
    "model_name",
    "modelId",
    "model_id",
    "label",
    "labels",
    "class",
    "className",
    "class_name",
    "confidence",
    "confidenceLabel",
    "confidence_label",
    "alarmType",
    "alarm_type",
    "alarmName",
    "alarm_name",
    "eventType",
    "event_type",
    "eventName",
    "event_name",
    "description",
    "title",
)
OBJECTIVE_METADATA_SKIP_KEYS = {
    "accessKey",
    "apiKey",
    "authorization",
    "base64",
    "blob",
    "content",
    "data",
    "data_base64",
    "dataBase64",
    "download_url",
    "fileContent",
    "file_content",
    "original_uri",
    "path",
    "sha1",
    "token",
    "uri",
    "url",
}
SUMMARY_FIELD_KEYS = frozenset(
    {
        "summary",
        "result",
        "resulttext",
        "result_text",
        "alarmsummary",
        "alarm_summary",
        "alarmresult",
        "alarm_result",
        "reviewsummary",
        "review_summary",
        "reviewresult",
        "review_result",
        "告警摘要",
        "告警结果",
        "复判摘要",
        "复判结果",
    }
)
SUMMARY_SECTION_LABELS = frozenset(
    _normalize
    for _normalize in (
        "告警摘要",
        "告警结果",
        "复判摘要",
        "复判结果",
        "alarm summary",
        "alarm result",
        "review summary",
        "review result",
        "summary",
        "result",
    )
)
PROMPT_SECTION_BOUNDARY_LABELS = frozenset(
    {
        *(_normalize for _normalize in OBJECTIVE_LABEL_ALIASES),
        "reviewsourceid",
        "reviewsource",
        "tasktarget",
        "schemaresults",
        "modelid",
        "modelname",
        "taskname",
        "sourcetaskname",
        "edgetaskname",
    }
)


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

    def dedup_key(
        self,
        messages: list[Message],
        attachments: list[Attachment],
        runtime_options: RuntimeOptions | None = None,
    ) -> str | None:
        del attachments, runtime_options
        review_source_id = _review_source_id(_last_user_text(messages))
        if not review_source_id or review_source_id == "unknown_review_source":
            return None
        return review_source_id

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

        image_attachments = _image_attachments(attachments)
        video_attachments = _video_attachments(attachments)
        prompt_text = _last_user_text(messages)
        sanitized_prompt_text = _sanitize_review_prompt_text(prompt_text)
        prompt_sanitized = sanitized_prompt_text != prompt_text
        review_source_id = _review_source_id(prompt_text)
        objective, objective_source = _objective_with_source(prompt_text, attachments)
        video_frame_attachments, video_frame_reports = _video_frame_attachments_for_review(
            image_attachments,
            video_attachments,
            paths,
        )
        review_attachments = [*image_attachments, *video_frame_attachments]
        image_sources = _attachment_source_payloads(review_attachments)
        visual_regions = _visual_region_payloads(review_attachments, prompt_text)
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
            visual_regions=visual_regions,
            paths=paths,
            recorder=recorder,
        )
        recorder.emit(
            "review.input",
            {
                "review_source_id": review_source_id,
                "objective": objective,
                "objective_source": objective_source,
                "app_template_name": runtime_options.app_template_name,
                "configured_selected_skills": runtime_options.selected_skills,
                "prompt_sanitized": prompt_sanitized,
                "selected_skill": skill_selection.skill_name if skill_selection is not None else "",
                "skill_selection": _skill_selection_event_payload(skill_selection),
                "skill_error": skill_error,
                "attachment_count": len(attachments),
                "image_attachment_count": len(image_attachments),
                "video_attachment_count": len(video_attachments),
                "video_frame_attachment_count": len(video_frame_attachments),
                "image_sources": image_sources,
                "visual_regions": visual_regions,
                "video_frame_reports": video_frame_reports,
            },
        )
        logger.info(
            "parking review routing review_source_id=%s app_template=%s objective=%s objective_source=%s "
            "prompt_sanitized=%s configured_selected_skills=%s selected_skill=%s method=%s confidence=%s candidates=%s",
            review_source_id,
            runtime_options.app_template_name or "",
            objective,
            objective_source,
            prompt_sanitized,
            ",".join(runtime_options.selected_skills or []),
            skill_selection.skill_name if skill_selection is not None else "",
            skill_selection.method if skill_selection is not None else "",
            skill_selection.confidence if skill_selection is not None else 0.0,
            _compact_payload_json(
                [
                    {
                        "skill_name": candidate.skill.name,
                        "score": candidate.score,
                        "score_reasons": list(candidate.score_reasons),
                    }
                    for candidate in (skill_selection.candidates if skill_selection is not None else ())
                ]
            ),
        )
        link_summary = _attachment_link_summary(image_sources)
        error_summary = _review_error_summary(skill_error, video_frame_reports)
        if PARKING_REVIEW_LOG_INPUT_SUMMARY:
            logger.info(
                "parking review input summary review_source_id=%s objective=%s objective_source=%s skill_name=%s "
                "attachments=%s image_inputs=%s video_inputs=%s video_frames=%s link_count=%s "
                "remote_link_count=%s local_path_count=%s region_count=%s links=%s errors=%s",
                review_source_id,
                objective,
                objective_source,
                skill_selection.skill_name if skill_selection is not None else "",
                len(attachments),
                len(image_attachments),
                len(video_attachments),
                len(video_frame_attachments),
                link_summary["link_count"],
                link_summary["remote_link_count"],
                link_summary["local_path_count"],
                len(visual_regions),
                _compact_payload_json(link_summary["links"]),
                _compact_payload_json(error_summary),
            )
        if PARKING_REVIEW_TRACE_DETAILS:
            logger.info(
                "parking review input details review_source_id=%s image_sources=%s visual_regions=%s video_frame_reports=%s",
                review_source_id,
                _pretty_payload_json(image_sources),
                _pretty_payload_json(visual_regions),
                _pretty_payload_json(video_frame_reports),
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
            try:
                raw_reply = self._complete_review(
                    agent_config=agent_config,
                    runtime_options=runtime_options,
                    prompt_text=prompt_text,
                    image_attachments=review_attachments,
                    paths=paths,
                    review_source_id=review_source_id,
                    objective=objective,
                    visual_regions=visual_regions,
                    skill_selection=skill_selection,
                    skill_result=skill_result,
                    skill_error=skill_error,
                )
            except Exception as exc:
                error = str(exc)
                failure_reason = "模型请求失败或超时，已按证据不足处理。"
                recorder.emit(
                    "review.llm.failed",
                    {
                        "review_source_id": review_source_id,
                        "objective": objective,
                        "skill_name": skill_selection.skill_name if skill_selection is not None else "",
                        "error": error[:1000],
                    },
                )
                logger.warning(
                    "parking review failed review_source_id=%s objective=%s skill_name=%s reason=%s error=%s",
                    review_source_id,
                    objective,
                    skill_selection.skill_name if skill_selection is not None else "",
                    failure_reason,
                    error[:1000],
                )
                raw_reply = _json_reply(
                    [
                        {
                            "reviewSourceId": review_source_id,
                            "reviewEventId": _review_event_id(review_source_id, objective),
                            "hit": 0,
                            "result": f"无法完成复判：{failure_reason}",
                        }
                    ]
                )
            if PARKING_REVIEW_TRACE_DETAILS:
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
            "parking review final result review_source_id=%s objective=%s skill_name=%s attachments=%s "
            "link_count=%s region_count=%s result_chars=%s result=%s",
            review_source_id,
            objective,
            skill_selection.skill_name if skill_selection is not None else "",
            len(review_attachments),
            link_summary["link_count"],
            len(visual_regions),
            len(reply),
            _compact_review_json(reply),
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

        if ranked[0].score <= 0:
            return None

        if REVIEW_ENABLE_LLM_SKILL_ROUTER:
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
            reason="使用最高关键词匹配分数的 skill。",
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
        visual_regions: list[dict[str, Any]],
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
            visual_regions=visual_regions,
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
        *,
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: ThreadPaths,
        review_source_id: str,
        objective: str,
        visual_regions: list[dict[str, Any]],
        skill_selection: ReviewSkillSelection | None,
        skill_result: SkillRunResult | None,
        skill_error: str,
    ) -> str:
        client = OpenAICompatibleClient(agent_config, runtime_options=_review_llm_runtime_options(runtime_options))
        evidence_context = _attachment_evidence_context(image_attachments)
        visual_region_context = _visual_region_prompt_context(visual_regions)
        skill_context = _review_skill_prompt_context(skill_selection, skill_result, skill_error)
        raw_prompt_text = str(prompt_text or "")
        sanitized_prompt_text = _sanitize_review_prompt_text(raw_prompt_text)
        objective_rule_context = _objective_review_prompt_context(
            objective,
            prompt_text=raw_prompt_text,
            selection=skill_selection,
        )
        system_prompt = (
            "你是机器视觉异常事件复判工作流。"
            "必须只基于用户文本和随附图片中可直接看见的内容判断识别目标是否命中。"
            "不要根据应用名称、场所类型、识别目标名称或经验常识推断画面外信息。"
            "只返回 JSON 数组，不要 Markdown，不要解释文字。"
        )
        user_prompt = (
            f"{sanitized_prompt_text}\n\n"
            "输出要求：\n"
            "- 只输出 JSON 数组。\n"
            "- reviewSourceId 必须原样回填为："
            f"{review_source_id}\n"
            "- hit 只能是 0 或 1。\n"
            "- 没有命中时 hit=0，命中时 hit=1。\n"
            "- result 用中文说明复判依据。\n"
            "- result 必须描述图片中清晰可见的证据；图片中不可确认的内容不要编造。\n"
            "- 如果图片来自视频抽帧，应综合所有抽帧画面判断，不要只依据单帧偶然现象下结论。\n"
            "- 如果提供了告警框、检测框或 ROI 区域，应优先查看这些区域，但不能仅凭框或 ROI 存在就判定命中；仍必须依据画面中可见证据判断。\n"
            "- 如存在已调用的复判 skill，必须优先遵循该 skill 的规则、判定边界和返回数据。\n"
            "- 不要使用“空旷路面”“停车位”“商场通道”“墙边堆放”等场景描述，除非这些内容在图片中清晰可见。\n"
            f"- 本次目标：{objective}\n"
            f"{objective_rule_context}"
            f"{skill_context}"
            f"{evidence_context}"
            f"{visual_region_context}"
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


def _review_llm_runtime_options(runtime_options: RuntimeOptions) -> RuntimeOptions:
    current = runtime_options.request_timeout_seconds
    capped = float(REVIEW_LLM_TIMEOUT_SECONDS)
    if current is not None:
        capped = min(float(current), capped)
    return runtime_options.model_copy(update={"request_timeout_seconds": capped}, deep=True)


def _review_source_id(text: str) -> str:
    payload = _parse_json_object(text)
    if payload:
        for path in (
            ("reviewSourceId",),
            ("review_source_id",),
            ("sourceId",),
            ("source_id",),
            ("headers", "reviewSourceId"),
            ("headers", "review_source_id"),
        ):
            value = _nested_string(payload, path)
            if value:
                return value
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
    objective, _source = _objective_from_text_with_source(text)
    return objective or "UnknownDetection"


def _objective_with_source(text: str, attachments: list[Attachment]) -> tuple[str, str]:
    objective, source = _objective_from_text_with_source(text)
    if objective:
        return objective, source
    metadata_objective, metadata_source = _objective_from_attachments(attachments)
    if metadata_objective:
        return metadata_objective, metadata_source
    return "UnknownDetection", "unknown"


def _objective_from_text_with_source(text: str) -> tuple[str, str]:
    match = re.search(r"识别目标为\[([^\]]+)\]", text)
    if match:
        return _canonical_objective(match.group(1).strip()), "prompt.bracket_target"
    detection_type_match = re.search(r"检测类型为\[([^\]]+)\]", text)
    if detection_type_match:
        return _canonical_objective(detection_type_match.group(1).strip()), "prompt.detection_type"
    payload = _parse_json_object(text)
    target = _target_from_payload(payload)
    if target:
        return _canonical_objective(target), "prompt.json_target"
    labeled_target = _objective_from_labeled_text(text)
    if labeled_target:
        return labeled_target, "prompt.labeled_text"
    inferred = _objective_from_text(text)
    if inferred:
        return inferred, "prompt.keyword"
    return "", ""


def _objective_from_attachments(attachments: list[Attachment]) -> tuple[str, str]:
    for attachment_index, attachment in enumerate(attachments, start=1):
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        objective, source = _objective_from_metadata(metadata)
        if objective:
            return objective, f"attachment[{attachment_index}].metadata.{source}"
    return "", ""


def _objective_from_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
    for key in OBJECTIVE_METADATA_KEYS:
        if key not in metadata:
            continue
        objective = _objective_from_metadata_value(metadata[key])
        if objective:
            return objective, key
    for key, value in _iter_metadata_strings(metadata):
        objective = _canonical_objective(value)
        if objective != value:
            return objective, key
        inferred = _objective_from_text(value)
        if inferred:
            return inferred, key
    return "", ""


def _objective_from_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        canonical = _canonical_objective(text)
        if canonical != text:
            return canonical
        return _objective_from_text(text)
    if isinstance(value, dict):
        for item in value.values():
            objective = _objective_from_metadata_value(item)
            if objective:
                return objective
    if isinstance(value, list):
        for item in value:
            objective = _objective_from_metadata_value(item)
            if objective:
                return objective
    return ""


def _iter_metadata_strings(value: Any, *, prefix: str = "", depth: int = 0) -> list[tuple[str, str]]:
    if depth > 4:
        return []
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or key in OBJECTIVE_METADATA_SKIP_KEYS:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            found.extend(_iter_metadata_strings(item, prefix=child_prefix, depth=depth + 1))
    elif isinstance(value, list):
        for index, item in enumerate(value[:20]):
            child_prefix = f"{prefix}[{index}]"
            found.extend(_iter_metadata_strings(item, prefix=child_prefix, depth=depth + 1))
    elif isinstance(value, str):
        text = value.strip()
        if text and len(text) <= 500:
            found.append((prefix, text))
    return found


def _canonical_objective(value: str) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return raw_value
    normalized = _normalize_match_text(raw_value)
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
        "fight": "FightDetection",
        "fighting": "FightDetection",
        "fightdetection": "FightDetection",
        "argument": "FightDetection",
        "argumentdetection": "FightDetection",
        "争吵": "FightDetection",
        "争吵检测": "FightDetection",
        "打架": "FightDetection",
        "打架检测": "FightDetection",
        "肢体冲突": "FightDetection",
        "顾客行为监管": "CustomerBehaviorDetection",
        "顾客行为检测": "CustomerBehaviorDetection",
        "客户行为监管": "CustomerBehaviorDetection",
        "客户行为检测": "CustomerBehaviorDetection",
        "顾客行为安全监管": "CustomerBehaviorDetection",
        "客户行为安全监管": "CustomerBehaviorDetection",
        "customerbehaviordetection": "CustomerBehaviorDetection",
        "杂物": "ClutterDetection",
        "杂物检测": "ClutterDetection",
        "杂物堆积": "ClutterDetection",
        "后厨通道卫生": "ClutterDetection",
        "后厨通道卫生检测": "ClutterDetection",
        "后厨通道卫生监管": "ClutterDetection",
        "后厨通道卫生安全监管": "ClutterDetection",
        "后厨通道卫生安全检测": "ClutterDetection",
        "通道卫生": "ClutterDetection",
        "clutter": "ClutterDetection",
        "clutterdetection": "ClutterDetection",
        "垃圾满溢": "GarbageOverflowDetection",
        "垃圾满溢检测": "GarbageOverflowDetection",
        "垃圾桶满溢": "GarbageOverflowDetection",
        "garbageoverflow": "GarbageOverflowDetection",
        "garbageoverflowdetection": "GarbageOverflowDetection",
        "火焰": "FireDetection",
        "明火": "FireDetection",
        "火焰/明火检测": "FireDetection",
        "火焰检测": "FireDetection",
        "明火检测": "FireDetection",
        "fire": "FireDetection",
        "firedetection": "FireDetection",
        "消防通道占用": "FireLaneComplianceDetection",
        "消防通道占用检测": "FireLaneComplianceDetection",
        "消防通道堵塞": "FireLaneComplianceDetection",
        "消防通道堵塞检测": "FireLaneComplianceDetection",
        "消防通道监管": "FireLaneComplianceDetection",
        "消防通道检测": "FireLaneComplianceDetection",
        "消防通道合规检测": "FireLaneComplianceDetection",
        "消防通道安全监管": "FireLaneComplianceDetection",
        "firelane": "FireLaneComplianceDetection",
        "firelanecompliancedetection": "FireLaneComplianceDetection",
        "车场异常事件监控": "ParkingAbnormalDetection",
        "车场异常事件监管": "ParkingAbnormalDetection",
        "车场异常监控": "ParkingAbnormalDetection",
        "车场异常监管": "ParkingAbnormalDetection",
        "停车场异常事件监控": "ParkingAbnormalDetection",
        "停车场异常事件监管": "ParkingAbnormalDetection",
        "停车场异常监控": "ParkingAbnormalDetection",
        "停车场异常监管": "ParkingAbnormalDetection",
        "车场异常事件检测": "ParkingAbnormalDetection",
        "停车场异常事件检测": "ParkingAbnormalDetection",
        "parkingabnormaldetection": "ParkingAbnormalDetection",
        "违规停车": "ParkingViolationDetection",
        "车辆违停": "ParkingViolationDetection",
        "违停": "ParkingViolationDetection",
        "车辆违停检测": "ParkingViolationDetection",
        "违规停车检测": "ParkingViolationDetection",
        "非法停车": "ParkingViolationDetection",
        "非法停车检测": "ParkingViolationDetection",
        "parkingviolation": "ParkingViolationDetection",
        "parkingviolationdetection": "ParkingViolationDetection",
        "illegalparking": "ParkingViolationDetection",
        "illegalparkingdetection": "ParkingViolationDetection",
        "停车场通道拥堵检测": "ParkingCongestionDetection",
        "车辆拥堵检测": "ParkingCongestionDetection",
        "车辆拥堵": "ParkingCongestionDetection",
        "通道拥堵": "ParkingCongestionDetection",
        "车道拥堵": "ParkingCongestionDetection",
        "车场拥堵": "ParkingCongestionDetection",
        "停车场拥堵": "ParkingCongestionDetection",
        "parkingcongestion": "ParkingCongestionDetection",
        "parkingcongestiondetection": "ParkingCongestionDetection",
        "trafficcongestion": "ParkingCongestionDetection",
        "trafficcongestiondetection": "ParkingCongestionDetection",
        "vehiclecongestion": "ParkingCongestionDetection",
        "vehiclecongestiondetection": "ParkingCongestionDetection",
    }
    canonical = aliases.get(normalized)
    if canonical:
        return canonical
    split_candidates = re.split(r"[/|,，;；]", raw_value)
    for candidate in split_candidates:
        normalized_candidate = _normalize_match_text(candidate)
        canonical = aliases.get(normalized_candidate)
        if canonical:
            return canonical
    return raw_value


def _objective_from_text(text: str) -> str:
    normalized = _normalize_match_text(text)
    fall_terms = ("人员跌倒/倒地检测", "人员跌倒", "人员倒地", "跌倒", "摔倒", "倒地", "falldetection", "fall")
    if any(_normalize_match_text(term) in normalized for term in fall_terms):
        return "FallDetection"
    smoking_terms = ("smokingdetection", "抽烟检测", "吸烟检测", "抽烟", "吸烟", "smoking", "smoke")
    if any(_normalize_match_text(term) in normalized for term in smoking_terms):
        return "SmokingDetection"
    fight_terms = ("fightdetection", "argumentdetection", "争吵检测", "打架检测", "争吵", "打架", "肢体冲突", "fighting", "fight", "argument")
    if any(_normalize_match_text(term) in normalized for term in fight_terms):
        return "FightDetection"
    behavior_terms = (
        "customerbehaviordetection",
        "顾客行为监管",
        "顾客行为检测",
        "客户行为监管",
        "客户行为检测",
        "顾客行为安全监管",
        "客户行为安全监管",
    )
    if any(_normalize_match_text(term) in normalized for term in behavior_terms):
        return "CustomerBehaviorDetection"
    garbage_terms = (
        "garbageoverflowdetection",
        "garbageoverflow",
        "trashoverflowdetection",
        "binoverflowdetection",
        "垃圾满溢检测",
        "垃圾满溢",
        "垃圾桶满溢",
        "垃圾外溢",
        "垃圾散落",
        "垃圾超出桶口",
    )
    if any(_normalize_match_text(term) in normalized for term in garbage_terms):
        return "GarbageOverflowDetection"
    clutter_terms = (
        "clutterdetection",
        "clutter",
        "杂物检测",
        "杂物堆积",
        "杂物",
        "后厨通道卫生安全监管",
        "后厨通道卫生检测",
        "后厨通道卫生",
        "通道卫生",
        "堆积",
        "占道",
    )
    if any(_normalize_match_text(term) in normalized for term in clutter_terms):
        return "ClutterDetection"
    fire_terms = ("firedetection", "fire", "火焰/明火检测", "火焰检测", "明火检测", "火焰", "明火")
    if any(_normalize_match_text(term) in normalized for term in fire_terms):
        return "FireDetection"
    fire_lane_terms = (
        "firelanecompliancedetection",
        "firelane",
        "消防通道合规检测",
        "消防通道占用检测",
        "消防通道占用",
        "消防通道堵塞检测",
        "消防通道堵塞",
        "消防通道监管",
        "消防通道检测",
        "消防通道安全监管",
    )
    if any(_normalize_match_text(term) in normalized for term in fire_lane_terms):
        return "FireLaneComplianceDetection"
    parking_violation_terms = (
        "parkingviolationdetection",
        "parkingviolation",
        "illegalparkingdetection",
        "illegalparking",
        "违规停车检测",
        "车辆违停检测",
        "违规停车",
        "车辆违停",
        "违停",
        "非法停车",
    )
    if any(_normalize_match_text(term) in normalized for term in parking_violation_terms):
        return "ParkingViolationDetection"
    parking_congestion_terms = (
        "parkingcongestiondetection",
        "parkingcongestion",
        "trafficcongestiondetection",
        "trafficcongestion",
        "vehiclecongestiondetection",
        "vehiclecongestion",
        "停车场通道拥堵检测",
        "车辆拥堵检测",
        "停车场拥堵",
        "车辆拥堵",
        "通道拥堵",
        "车道拥堵",
        "车场拥堵",
        "拥堵",
    )
    if any(_normalize_match_text(term) in normalized for term in parking_congestion_terms):
        return "ParkingCongestionDetection"
    parking_abnormal_terms = (
        "parkingabnormaldetection",
        "车场异常事件监控",
        "车场异常事件监管",
        "车场异常监控",
        "车场异常监管",
        "停车场异常事件监控",
        "停车场异常事件监管",
        "停车场异常监控",
        "停车场异常监管",
        "车场异常事件检测",
        "停车场异常事件检测",
    )
    if any(_normalize_match_text(term) in normalized for term in parking_abnormal_terms):
        return "ParkingAbnormalDetection"
    return ""


def _objective_from_labeled_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    normalized_labels = {_normalize_match_text(item) for item in OBJECTIVE_LABEL_ALIASES}
    objectives: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        normalized_line = _normalize_match_text(line).strip(":：")
        if normalized_line not in normalized_labels:
            continue
        for candidate in lines[index + 1 : index + 4]:
            if not candidate:
                continue
            objective = _objective_from_candidate_text(candidate)
            if objective:
                objectives.append(objective)
            break
    labeled_patterns = (
        r"(?:场景|识别目标|检测目标|算法|算法模型|模型)\s*[:：]\s*([^\n，,;；]+)",
        r"(?:置信度|标签|类别)\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]+)",
    )
    for pattern in labeled_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = match.group(1).strip()
        objective = _objective_from_candidate_text(candidate)
        if objective:
            objectives.append(objective)
    return _best_objective(objectives)


def _target_from_payload(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    candidates: list[str] = []
    for path in (
        ("taskTarget", "text"),
        ("taskTarget", "value"),
        ("schemaResults", "taskTarget", "text"),
        ("schemaResults", "taskTarget", "value"),
        ("headers", "target"),
        ("headers", "taskName"),
        ("headers", "sourceTaskName"),
        ("headers", "edgeTaskName"),
        ("headers", "modelName"),
        ("value",),
        ("text",),
        ("taskName",),
        ("sourceTaskName",),
        ("edgeTaskName",),
        ("modelName",),
    ):
        value = _nested_string(payload, path)
        if value:
            candidates.append(value)
    task_target = payload.get("taskTarget")
    if isinstance(task_target, dict):
        candidates.extend(str(value).strip() for value in task_target.values() if isinstance(value, str) and value.strip())
    objectives = [objective for candidate in candidates if (objective := _objective_from_candidate_text(candidate))]
    return _best_objective(objectives) or (candidates[0] if candidates else "")


def _objective_from_candidate_text(candidate: str) -> str:
    canonical = _canonical_objective(candidate)
    if canonical != candidate:
        return canonical
    return _objective_from_text(candidate)


def _best_objective(objectives: list[str]) -> str:
    if not objectives:
        return ""
    scored: list[tuple[int, int, str]] = [
        (_objective_specificity(objective), -index, objective)
        for index, objective in enumerate(objectives)
        if objective
    ]
    if not scored:
        return ""
    return max(scored)[2]


def _objective_specificity(objective: str) -> int:
    normalized = _normalize_match_text(objective)
    generic = {
        "clutterdetection",
        "customerbehaviordetection",
        "parkingabnormaldetection",
    }
    return 1 if normalized in generic else 2


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


def _attachment_link_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    remote_count = 0
    local_count = 0
    links: list[dict[str, Any]] = []
    for source in sources:
        ref = _first_non_empty(source.get("url"), source.get("path"))
        if not ref:
            continue
        is_remote = _is_http_url(ref)
        if is_remote:
            remote_count += 1
        else:
            local_count += 1
        if len(links) < PARKING_REVIEW_LOG_LINK_LIMIT:
            links.append(
                {
                    "index": source.get("index"),
                    "name": source.get("name"),
                    "mime_type": source.get("mime_type"),
                    "source": source.get("source") or "attachment",
                    "ref": _redacted_log_ref(ref),
                    "dataId": _first_non_empty(source.get("dataId"), source.get("data_id")),
                    "sourceId": _first_non_empty(source.get("sourceId"), source.get("source_id")),
                    "frame_index": source.get("frame_index"),
                    "timestamp_ms": source.get("timestamp_ms"),
                }
            )
    return {
        "link_count": remote_count + local_count,
        "remote_link_count": remote_count,
        "local_path_count": local_count,
        "links": links,
        "omitted": max(0, remote_count + local_count - len(links)),
    }


def _review_error_summary(skill_error: str, video_frame_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if skill_error:
        errors.append({"type": "skill_error", "error": skill_error[:500]})
    for report in video_frame_reports:
        if report.get("error_code") == "image_attachments_preferred":
            continue
        if report.get("status") == "completed" and report.get("frame_count"):
            continue
        errors.append(
            {
                "type": "video_frame",
                "name": report.get("name"),
                "status": report.get("status"),
                "error_code": report.get("error_code") or report.get("status") or "unknown",
                "error": str(report.get("error") or "")[:500],
            }
        )
    return errors


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _redacted_log_ref(value: str) -> str:
    text = str(value or "").strip()
    parsed = urlparse(text)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        redacted = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            redacted += "?..."
        return redacted[:240]
    return text[:240] + ("..." if len(text) > 240 else "")


_VISUAL_REGION_FIELD_ALIASES: dict[str, str] = {
    "bbox": "bbox",
    "bboxes": "boxes",
    "box": "bbox",
    "boxes": "boxes",
    "rect": "bbox",
    "rects": "boxes",
    "rectangle": "bbox",
    "rectangles": "boxes",
    "alarmBox": "alarm_box",
    "alarm_box": "alarm_box",
    "alarmBoxes": "alarm_boxes",
    "alarm_boxes": "alarm_boxes",
    "detectBox": "detect_box",
    "detect_box": "detect_box",
    "detectBoxes": "detect_boxes",
    "detect_boxes": "detect_boxes",
    "detectionBox": "detection_box",
    "detection_box": "detection_box",
    "detectionBoxes": "detection_boxes",
    "detection_boxes": "detection_boxes",
    "roi": "roi",
    "rois": "rois",
    "roiArea": "roi",
    "roi_area": "roi",
    "region": "region",
    "regions": "regions",
    "polygon": "polygon",
    "polygons": "polygons",
    "points": "points",
    "pointList": "points",
    "point_list": "points",
    "coordinates": "coordinates",
    "coordinate": "coordinate",
    "area": "area",
    "areas": "areas",
    "mask": "mask",
    "masks": "masks",
}


def _visual_region_payloads(attachments: list[Attachment], prompt_text: str) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, attachment in enumerate(attachments, start=1):
        metadata = attachment.metadata if isinstance(attachment.metadata, dict) else {}
        fields = _extract_visual_region_fields(metadata)
        if not fields:
            continue
        regions.append(
            {
                "source": "attachment_metadata",
                "image_index": index,
                "name": attachment.name,
                "path": attachment.path,
                "url": _first_non_empty(metadata.get("original_uri"), metadata.get("url"), metadata.get("uri"), attachment.path),
                "dataId": _first_non_empty(metadata.get("dataId"), metadata.get("data_id")),
                "sourceId": _first_non_empty(metadata.get("sourceId"), metadata.get("source_id")),
                "timestamp": metadata.get("timestamp"),
                "regions": fields,
            }
        )
    prompt_payload = _parse_json_object(prompt_text)
    prompt_fields = _extract_visual_region_fields(prompt_payload) if prompt_payload is not None else {}
    prompt_objects = _extract_schema_result_objects(prompt_payload) if prompt_payload is not None else []
    if prompt_objects:
        prompt_fields.setdefault("detection_objects", prompt_objects)
    if prompt_fields:
        regions.append({"source": "prompt_json", "regions": prompt_fields})
    return regions


def _extract_visual_region_fields(value: Any) -> dict[str, Any]:
    found: dict[str, Any] = {}

    def visit(current: Any) -> None:
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    continue
                normalized = _VISUAL_REGION_FIELD_ALIASES.get(key)
                if normalized and item not in (None, "", [], {}):
                    found.setdefault(normalized, item)
                if isinstance(item, dict | list):
                    visit(item)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, dict | list):
                    visit(item)

    visit(value)
    objects = _extract_schema_result_objects(value)
    if objects:
        found.setdefault("detection_objects", objects)
    return found


def _extract_schema_result_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    schema_results = value.get("schemaResults")
    if not isinstance(schema_results, dict):
        return []
    objects = schema_results.get("objects")
    if not isinstance(objects, list):
        return []
    extracted: list[dict[str, Any]] = []
    for index, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        box = item.get("box") or item.get("bbox") or item.get("rect")
        if not isinstance(box, list) or len(box) < 4:
            continue
        extracted.append(
            {
                "index": index,
                "box": box[:4],
                "label": item.get("label"),
                "score": item.get("score"),
                "text": _nested_string(item, ("others", "text")),
                "value": _nested_string(item, ("others", "value")),
            }
        )
    return extracted


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
    candidates: dict[str, ReviewSkillCandidate] = {}
    for name in candidate_names:
        try:
            skill = skill_registry.get(name)
        except KeyError:
            continue
        if skill.name in REVIEW_SKILL_EXCLUDED_NAMES:
            continue
        if not _review_skill_allowed(skill.name, objective=objective, prompt_text=prompt_text):
            continue
        context = _skill_context(skill)
        score, reasons = _score_review_skill(skill, context, objective=objective, prompt_text=prompt_text)
        candidates[skill.name] = ReviewSkillCandidate(
            skill=skill,
            context=context,
            score=score,
            score_reasons=tuple((*reasons, "template_candidate")),
        )
    if candidates:
        return sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    list_skills = getattr(skill_registry, "list", None)
    if callable(list_skills):
        for skill in list_skills(executable_only=True):
            if skill.name in REVIEW_SKILL_EXCLUDED_NAMES:
                continue
            if skill.name in candidates:
                continue
            if not _review_skill_allowed(skill.name, objective=objective, prompt_text=prompt_text):
                continue
            context = _skill_context(skill)
            score, reasons = _score_review_skill(skill, context, objective=objective, prompt_text=prompt_text)
            if score < SKILL_AUTO_CANDIDATE_MIN_SCORE:
                continue
            candidates[skill.name] = ReviewSkillCandidate(
                skill=skill,
                context=context,
                score=score,
                score_reasons=tuple((*reasons, "auto_candidate")),
            )
    return sorted(candidates.values(), key=lambda item: item.score, reverse=True)


def _review_skill_allowed(skill_name: str, *, objective: str, prompt_text: str) -> bool:
    if skill_name != SMOKING_REVIEW_SKILL_NAME:
        return True
    normalized_objective = _normalize_match_text(objective)
    if normalized_objective == SMOKING_OBJECTIVE:
        return True
    if _prompt_explicitly_targets_smoking(prompt_text):
        return True
    return normalized_objective not in SMOKING_EXCLUDED_OBJECTIVES



def _prompt_explicitly_targets_smoking(prompt_text: str) -> bool:
    payload = _parse_json_object(prompt_text)
    if payload is not None:
        for term in _target_terms_from_payload(payload):
            if _text_has_smoking_target(term):
                return True
    if _text_has_smoking_target(_labeled_objective_text(prompt_text)):
        return True
    bracket_targets = re.findall(r"(?:识别目标|检测类型)为\[([^\]]+)\]", prompt_text)
    return any(_text_has_smoking_target(term) for term in bracket_targets)


def _labeled_objective_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    normalized_labels = {_normalize_match_text(item) for item in OBJECTIVE_LABEL_ALIASES}
    for index, line in enumerate(lines):
        if not line:
            continue
        normalized_line = _normalize_match_text(line).strip(":：")
        if normalized_line not in normalized_labels:
            continue
        for candidate in lines[index + 1 : index + 4]:
            if candidate:
                return candidate
    for pattern in (
        r"(?:场景|识别目标|检测目标|算法|算法模型|模型)\s*[:：]\s*([^\n，,;；]+)",
        r"(?:置信度|标签|类别)\s*[:：]?\s*([A-Za-z][A-Za-z0-9_-]+)",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _text_has_smoking_target(value: str) -> bool:
    normalized = _normalize_match_text(value)
    return any(_normalize_match_text(term) in normalized for term in SMOKING_EXPLICIT_TERMS)


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
        "clutterdetection": [
            "clutter",
            "杂物",
            "杂物检测",
            "堆积",
            "占道",
            "后厨通道卫生",
            "后厨通道卫生检测",
            "后厨通道卫生安全监管",
            "通道卫生",
            "垃圾袋",
            "乱堆乱放",
            "塑料筐",
            "托盘",
        ],
        "smokingdetection": ["smoking", "smoke", "抽烟", "吸烟"],
        "fightdetection": ["fight", "fighting", "argument", "争吵", "争吵检测", "打架", "斗殴", "冲突", "肢体冲突"],
        "falldetection": ["fall", "fallen", "跌倒", "摔倒", "倒地", "人员跌倒", "人员跌倒/倒地检测"],
        "customerbehaviordetection": [
            "顾客行为",
            "顾客行为监管",
            "顾客行为检测",
            "客户行为",
            "客户行为监管",
            "客户行为检测",
            "抽烟",
            "吸烟",
            "跌倒",
            "倒地",
            "争吵",
            "打架",
        ],
        "garbageoverflowdetection": [
            "garbage",
            "overflow",
            "trashoverflow",
            "binoverflow",
            "垃圾满溢",
            "垃圾满溢检测",
            "垃圾桶满溢",
            "垃圾外溢",
            "垃圾堆积",
            "垃圾散落",
            "垃圾超出桶口",
        ],
        "firedetection": ["fire", "flame", "火焰", "明火", "火焰检测", "明火检测", "火焰/明火检测"],
        "firelanecompliancedetection": [
            "FireLane",
            "消防通道",
            "消防通道占用",
            "消防通道占用检测",
            "消防通道堵塞",
            "消防通道堵塞检测",
            "消防通道监管",
            "消防通道检测",
            "消防通道合规检测",
        ],
        "parkingabnormaldetection": [
            "车场异常",
        "车场异常事件监控",
        "车场异常事件监管",
        "车场异常监控",
        "车场异常监管",
        "停车场异常事件监控",
        "停车场异常事件监管",
        "停车场异常监控",
        "停车场异常监管",
        "违规停车",
        "车辆违停",
            "车辆违停检测",
            "停车场通道拥堵检测",
            "车辆拥堵",
            "通道拥堵",
        ],
        "parkingviolationdetection": [
            "ParkingViolation",
            "IllegalParking",
            "违规停车",
            "违规停车检测",
            "车辆违停",
            "车辆违停检测",
            "违停",
            "非法停车",
            "禁停",
            "占用通道",
            "占用车道",
        ],
        "parkingcongestiondetection": [
            "ParkingCongestion",
            "TrafficCongestion",
            "停车场通道拥堵检测",
            "车辆拥堵检测",
            "车辆拥堵",
            "通道拥堵",
            "车道拥堵",
            "车场拥堵",
            "停车场拥堵",
            "排队",
            "堵塞",
            "拥堵",
        ],
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
        ("sourceTaskName",),
        ("edgeTaskName",),
        ("headers", "target"),
        ("headers", "taskName"),
        ("headers", "sourceTaskName"),
        ("headers", "edgeTaskName"),
        ("headers", "modelName"),
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
    client = OpenAICompatibleClient(agent_config, runtime_options=_review_llm_runtime_options(runtime_options))
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
    visual_regions: list[dict[str, Any]],
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
        "visual_regions": visual_regions,
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


def _objective_review_prompt_context(
    objective: str,
    *,
    prompt_text: str,
    selection: ReviewSkillSelection | None,
) -> str:
    selected_skill_name = selection.skill_name if selection is not None else ""
    if (
        _normalize_match_text(objective) != "parkingviolationdetection"
        and selected_skill_name != "parking-violation-review"
    ):
        if (
            _normalize_match_text(objective) != "garbageoverflowdetection"
            and selected_skill_name != "garbage-overflow-review"
        ):
            return ""
        return (
            "\n垃圾满溢专项硬规则：\n"
            "- 本次只复判“垃圾满溢/垃圾外溢/垃圾散落/垃圾堆积”是否命中；"
            "不能把行人、车辆、普通墙面、门、固定设施、清洁工具或正常垃圾桶当作命中证据。\n"
            "- 只有清楚看到垃圾桶已满、垃圾超过桶口、垃圾袋破裂散落、垃圾在桶外地面明显堆积，"
            "或垃圾堆积影响卫生/通行时，hit 才能为 1。\n"
            "- 如果检测框内主要是人员肢体、行走人员、空/正常垃圾桶、墙边固定垃圾桶，"
            "或无法确认垃圾已满溢/外溢，必须判定 hit=0。\n"
            "- 告警摘要里出现“人员”“车辆”等其他目标时，只能作为上游误检线索；"
            "不得因此判定垃圾满溢命中。\n"
        )
    return (
        "\n车辆违停专项硬规则：\n"
        "- 只有能清楚看到车辆停放在禁止停车区域、通行车道、出入口、消防通道、坡道口、转弯口，"
        "或明确阻碍车辆/人员通行时，hit 才能为 1。\n"
        "- 如果车辆位于正常停车位、停车线内、划定停车区域内，必须判定 hit=0；"
        "即使上游目标为“车辆违停”或存在告警框，也不能判定命中。\n"
        "- 告警框、检测框、bbox 通常只是车辆检测框；ROI/area 可能只是算法识别范围。"
        "除非区域信息或画面清晰表明该区域是禁停区/通行通道，否则不能把框内有车当作违停证据。\n"
        "- 如果无法确认车辆是否越出车位、占用通道、位于禁停区或阻碍通行，必须判定 hit=0，"
        "result 写“未发现明确违规停车证据”。\n"
        f"{_cross_objective_summary_prompt_context(objective, prompt_text)}"
    )


def _cross_objective_summary_prompt_context(objective: str, prompt_text: str) -> str:
    normalized_objective = _normalize_match_text(objective)
    normalized_prompt = _normalize_match_text(prompt_text)
    if normalized_objective != "parkingviolationdetection":
        return ""
    clutter_terms = ("杂物检测", "杂物堆积", "杂物", "clutterdetection")
    garbage_terms = ("垃圾满溢", "垃圾桶满溢", "garbageoverflowdetection")
    if not any(term in normalized_prompt for term in tuple(_normalize_match_text(item) for item in (*clutter_terms, *garbage_terms))):
        return ""
    return (
        "- 如果输入文本里出现“杂物检测”“杂物堆积”“垃圾满溢”等与车辆违停目标不一致的历史摘要或旧结果，"
        "这些文本只能视为历史上下文，不可作为本次命中的判定依据；仍必须只依据当前图片/视频中可见的车辆违停证据判断。\n"
    )


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
                timeout_seconds=VIDEO_FRAME_TIMEOUT_SECONDS,
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


def _video_frame_attachments_for_review(
    image_attachments: list[Attachment],
    video_attachments: list[Attachment],
    paths: ThreadPaths,
) -> tuple[list[Attachment], list[dict[str, Any]]]:
    if not video_attachments:
        return [], []
    if (
        VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST > 0
        and len(image_attachments) >= VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST
    ):
        return [], [
            {
                "name": attachment.name,
                "mime_type": attachment.mime_type,
                "path": attachment.path,
                "status": "skipped",
                "error_code": "image_attachments_preferred",
                "image_attachment_count": len(image_attachments),
                "threshold": VIDEO_FRAME_SKIP_WHEN_IMAGE_COUNT_AT_LEAST,
            }
            for attachment in video_attachments
        ]
    return _video_frame_attachments(video_attachments, paths)


def _extract_video_frames(
    video_path: Path,
    output_dir: Path,
    *,
    prefix: str,
    max_frames: int = VIDEO_FRAME_MAX_ATTACHMENTS,
    max_candidates: int = VIDEO_FRAME_MAX_CANDIDATES,
    timeout_seconds: int = VIDEO_FRAME_TIMEOUT_SECONDS,
) -> list[ExtractedVideoFrame]:
    deadline = time.monotonic() + max(1, timeout_seconds)
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
        target_candidates = _effective_video_candidate_count(max_frames=max_frames, max_candidates=max_candidates)
        candidates = _read_video_frame_candidates(
            cv2,
            capture,
            fps=fps,
            frame_count=frame_count,
            max_candidates=target_candidates,
            deadline=deadline,
        )
    finally:
        capture.release()

    if not candidates:
        raise ValueError(f"No readable frames extracted from video: {video_path}")

    selected = _select_video_frame_candidates(cv2, candidates, max_frames=max_frames)
    extracted: list[ExtractedVideoFrame] = []
    for output_index, candidate in enumerate(selected, start=1):
        _raise_if_video_frame_timeout(deadline)
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
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError("OpenCV is unavailable; install opencv-python to enable video frame extraction.") from exc
    _suppress_cv2_video_logs(cv2)
    return cv2


def _suppress_cv2_video_logs(cv2: Any) -> None:
    set_log_level = getattr(cv2, "setLogLevel", None)
    if not callable(set_log_level):
        return
    level = getattr(cv2, "LOG_LEVEL_ERROR", None)
    if level is None:
        level = getattr(cv2, "LOG_LEVEL_SILENT", None)
    if level is None:
        return
    try:
        set_log_level(level)
    except Exception:
        return


def _read_video_frame_candidates(
    cv2: Any,
    capture: Any,
    *,
    fps: float,
    frame_count: int,
    max_candidates: int,
    deadline: float,
) -> list[VideoFrameCandidate]:
    if frame_count > 0:
        if frame_count <= VIDEO_FRAME_SEQUENTIAL_SCAN_MAX_FRAMES:
            return _read_target_video_frame_candidates(
                cv2,
                capture,
                fps=fps,
                frame_count=frame_count,
                max_candidates=max_candidates,
                deadline=deadline,
            )
        candidates: list[VideoFrameCandidate] = []
        for index in _candidate_frame_indices(frame_count, max_candidates):
            _raise_if_video_frame_timeout(deadline)
            candidate = _read_video_frame_at(cv2, capture, index, fps=fps, deadline=deadline)
            if candidate is not None:
                candidates.append(candidate)
        return candidates
    return _read_sequential_video_frame_candidates(cv2, capture, fps=fps, max_candidates=max_candidates, deadline=deadline)


def _read_target_video_frame_candidates(
    cv2: Any,
    capture: Any,
    *,
    fps: float,
    frame_count: int,
    max_candidates: int,
    deadline: float,
) -> list[VideoFrameCandidate]:
    targets = _candidate_frame_indices(frame_count, max_candidates)
    if not targets:
        return []
    candidates: list[VideoFrameCandidate] = []
    current_index = 0
    for target in targets:
        while current_index < target:
            _raise_if_video_frame_timeout(deadline)
            if not capture.grab():
                return candidates
            current_index += 1
        _raise_if_video_frame_timeout(deadline)
        ok, frame = capture.read()
        if not ok or frame is None:
            current_index += 1
            continue
        timestamp_ms = (target / fps * 1000.0) if fps > 0 else float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        candidates.append(_video_frame_candidate(cv2, frame, frame_index=target, timestamp_ms=timestamp_ms))
        current_index += 1
    return candidates


def _read_video_frame_at(cv2: Any, capture: Any, frame_index: int, *, fps: float, deadline: float) -> VideoFrameCandidate | None:
    _raise_if_video_frame_timeout(deadline)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _raise_if_video_frame_timeout(deadline)
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
    deadline: float,
) -> list[VideoFrameCandidate]:
    candidates: list[VideoFrameCandidate] = []
    frame_index = 0
    stride = 15
    max_reads = max_candidates * stride * 2
    while len(candidates) < max_candidates and frame_index < max_reads:
        _raise_if_video_frame_timeout(deadline)
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


def _raise_if_video_frame_timeout(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError("video frame extraction timed out")


def _video_frame_candidate(cv2: Any, frame: Any, *, frame_index: int, timestamp_ms: float) -> VideoFrameCandidate:
    analysis_frame = _resize_frame(cv2, frame, max_width=VIDEO_FRAME_SCORE_MAX_WIDTH)
    gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
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


def _effective_video_candidate_count(*, max_frames: int, max_candidates: int) -> int:
    if max_candidates <= 0:
        return 0
    target = max_frames * VIDEO_FRAME_CANDIDATE_MULTIPLIER
    if max_frames >= VIDEO_FRAME_MIN_CANDIDATES:
        target = max(VIDEO_FRAME_MIN_CANDIDATES, target)
    return min(max_candidates, target)


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


def _visual_region_prompt_context(visual_regions: list[dict[str, Any]]) -> str:
    if not visual_regions:
        return ""
    return (
        "\n告警框/检测框/ROI 区域信息：\n"
        f"{_pretty_payload_json(visual_regions)}\n"
        "使用要求：以上区域是前端/上游算法提供的关注区域。复判时应重点查看对应图片区域；"
        "如果区域内没有清晰可见的目标证据，仍应判定为不命中。\n"
    )


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


def _compact_review_json(value: str) -> str:
    parsed = _parse_json_array(value)
    text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")) if parsed is not None else value.strip()
    if len(text) <= REVIEW_LLM_REPLY_LOG_MAX_CHARS:
        return text
    return f"{text[:REVIEW_LLM_REPLY_LOG_MAX_CHARS]}...<truncated chars={len(text) - REVIEW_LLM_REPLY_LOG_MAX_CHARS}>"


def _compact_payload_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= REVIEW_LLM_REPLY_LOG_MAX_CHARS:
        return text
    return f"{text[:REVIEW_LLM_REPLY_LOG_MAX_CHARS]}...<truncated chars={len(text) - REVIEW_LLM_REPLY_LOG_MAX_CHARS}>"


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


def _sanitize_review_prompt_text(prompt_text: str) -> str:
    text = str(prompt_text or "").strip()
    if not text:
        return ""
    sanitized_json = _sanitize_prompt_json_text(text)
    if sanitized_json is not None:
        return sanitized_json
    sanitized_text = _strip_summary_sections_from_text(text)
    return sanitized_text.strip() or text


def _sanitize_prompt_json_text(text: str) -> str | None:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).strip() if fenced else stripped
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    sanitized = _remove_summary_fields(payload)
    return json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))


def _remove_summary_fields(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = _normalize_prompt_label(str(key))
            if normalized_key in {_normalize_prompt_label(name) for name in SUMMARY_FIELD_KEYS}:
                continue
            sanitized[key] = _remove_summary_fields(item)
        return sanitized
    if isinstance(value, list):
        return [_remove_summary_fields(item) for item in value]
    return value


def _strip_summary_sections_from_text(text: str) -> str:
    lines = text.splitlines()
    sanitized: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _line_starts_summary_section(line):
            index += 1
            while index < len(lines):
                candidate = lines[index]
                stripped = candidate.strip()
                if not stripped:
                    index += 1
                    continue
                if _line_is_prompt_section_boundary(candidate):
                    break
                index += 1
            continue
        sanitized.append(line)
        index += 1
    return "\n".join(sanitized)


def _line_starts_summary_section(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    match = re.match(r"^([A-Za-z\u4e00-\u9fff _-]+)\s*[:：]?(.*)$", stripped)
    if match is None:
        return False
    label = _normalize_prompt_label(match.group(1))
    if label not in SUMMARY_SECTION_LABELS:
        return False
    remainder = match.group(2).strip()
    return not remainder or True


def _line_is_prompt_section_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    normalized = _normalize_prompt_label(stripped)
    if normalized in PROMPT_SECTION_BOUNDARY_LABELS:
        return True
    match = re.match(r"^([A-Za-z\u4e00-\u9fff _-]+)\s*[:：]", stripped)
    if match is None:
        return False
    return _normalize_prompt_label(match.group(1)) in PROMPT_SECTION_BOUNDARY_LABELS


def _normalize_prompt_label(value: str) -> str:
    return re.sub(r"[\s:：_-]+", "", value).lower()


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
