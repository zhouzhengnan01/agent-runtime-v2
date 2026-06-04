from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from app.core.artifacts import ArtifactStore
from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions


WORKFLOW_NAME = "parking_abnormal_review"
OUTPUT_NAME = "parking_abnormal_review_result.json"


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
        recorder.emit(
            "review.input",
            {
                "review_source_id": review_source_id,
                "objective": objective,
                "attachment_count": len(attachments),
                "image_attachment_count": len(image_attachments),
            },
        )

        if not image_attachments:
            reply = _json_reply(
                [
                    {
                        "reviewSourceId": review_source_id,
                        "reviewEventId": _review_event_id(review_source_id, objective),
                        "hit": 0,
                        "result": "无法完成复判：当前请求未提供可访问的图片/视频文件，无法依据画面内容判断是否命中。",
                    }
                ]
            )
        else:
            raw_reply = self._complete_review(
                agent_config,
                runtime_options,
                prompt_text,
                image_attachments,
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
        review_source_id: str,
        objective: str,
    ) -> str:
        client = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
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
            "- 不要使用“空旷路面”“停车位”“商场通道”“墙边堆放”等场景描述，除非这些内容在图片中清晰可见。\n"
            f"- 本次目标：{objective}\n"
        )
        payload = {
            "role": "user",
            "content": user_prompt,
            "_attachments": [attachment.model_dump(mode="python") for attachment in image_attachments],
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


def _json_reply(items: list[dict[str, Any]]) -> str:
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))
