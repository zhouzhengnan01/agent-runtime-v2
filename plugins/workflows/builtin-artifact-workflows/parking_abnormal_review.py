from __future__ import annotations

import hashlib
import importlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


WORKFLOW_NAME = "parking_abnormal_review"
OUTPUT_NAME = "parking_abnormal_review_result.json"
VIDEO_FRAME_OUTPUT_DIR = "parking_abnormal_review_video_frames"
VIDEO_FRAME_MAX_ATTACHMENTS = 6
VIDEO_FRAME_MAX_CANDIDATES = 24
VIDEO_FRAME_MAX_WIDTH = 1280
VIDEO_FRAME_MIN_DIFFERENCE = 8.0


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


class ParkingAbnormalReviewWorkflow:
    """Deterministic parking abnormal review workflow that returns only JSON."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store

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
        recorder.emit(
            "review.input",
            {
                "review_source_id": review_source_id,
                "objective": objective,
                "attachment_count": len(attachments),
                "image_attachment_count": len(image_attachments),
                "video_attachment_count": len(video_attachments),
                "video_frame_attachment_count": len(video_frame_attachments),
                "video_frame_reports": video_frame_reports,
            },
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
            )
            reply = _normalize_json_reply(raw_reply, review_source_id=review_source_id, objective=objective)

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

    @staticmethod
    def _complete_review(
        agent_config: AgentConfig,
        runtime_options: RuntimeOptions,
        prompt_text: str,
        image_attachments: list[Attachment],
        paths: ThreadPaths,
        review_source_id: str,
        objective: str,
    ) -> str:
        client = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
        evidence_context = _attachment_evidence_context(image_attachments)
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
            "- 不要使用“空旷路面”“停车位”“商场通道”“墙边堆放”等场景描述，除非这些内容在图片中清晰可见。\n"
            f"- 本次目标：{objective}\n"
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
        return match.group(1).strip()
    return "ClutterDetection"


def _image_attachments(attachments: list[Attachment]) -> list[Attachment]:
    return [
        attachment
        for attachment in attachments
        if (attachment.mime_type or "").split(";", 1)[0].strip().lower().startswith("image/")
    ]


def _video_attachments(attachments: list[Attachment]) -> list[Attachment]:
    return [
        attachment
        for attachment in attachments
        if (attachment.mime_type or "").split(";", 1)[0].strip().lower().startswith("video/")
    ]


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
