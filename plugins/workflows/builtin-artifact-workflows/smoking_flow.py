from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.core.skills import SkillRunner
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions, VerificationResult

_DATASET_PACKAGE_EXTS = (".zip", ".tar", ".tar.gz")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class SmokingDetectionTrainingWorkflow:
    """Two-step flow: upload dataset+reference image, then prompt; run all automatically."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.skill_runner = SkillRunner(artifact_store)

    def run_with_events(
        self,
        agent_config: Any,
        messages: list[Message],
        attachments: list[Attachment],
        thread_id: str | None,
        on_event: Any = None,
        workflow_name: str | None = None,
        runtime_options: RuntimeOptions | None = None,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        runtime_options = runtime_options or RuntimeOptions()
        workflow = workflow_name or "smoking_detection_training_flow"
        paths = self.artifact_store.prepare_thread(thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id, on_emit=on_event)
        recorder.emit("run.started", {"workflow": workflow})

        user_text = _last_user_text(messages)
        clean_user_text = _strip_uploaded_files_context(user_text)
        waiting_prompt = _is_waiting_prompt(paths)
        prompt_text = _extract_generation_prompt(clean_user_text, allow_free_text=waiting_prompt)
        if prompt_text:
            _save_generation_prompt(paths, prompt_text)
        else:
            prompt_text = _load_generation_prompt(paths)
        labels = _extract_annotation_labels(clean_user_text)
        if labels:
            _save_annotation_labels(paths, labels)
        else:
            labels = _load_annotation_labels(paths)
        generated_dir = paths.workspace / "generated_images"
        generated_dir.mkdir(parents=True, exist_ok=True)

        dataset_attachment = _find_dataset_package_attachment(attachments)
        image_attachment = _find_image_attachment(attachments, include_thread_files=False)
        if image_attachment is None and waiting_prompt:
            image_attachment = _find_image_attachment(attachments, include_thread_files=True)

        if dataset_attachment and dataset_attachment.path:
            _save_dataset_package_path(paths, dataset_attachment.path)
        if image_attachment and image_attachment.path:
            _save_reference_image_path(paths, image_attachment.path)

        dataset_pkg = _load_dataset_package_path(paths)
        reference_image = _load_reference_image_path(paths)

        if not dataset_pkg or not reference_image:
            return self._input_required_result(
                recorder,
                agent_config.name,
                paths.thread_id,
                workflow,
                [
                    {"type": "dataset", "accept": ".zip,.tar,.tar.gz", "required": True, "reason": "需要上传数据集压缩包"},
                    {"type": "image", "accept": "image/*", "required": True, "reason": "需要上传参考图片"},
                ],
            )

        if not prompt_text:
            _set_waiting_prompt(paths, True)
            reply = (
                "已收到数据集和参考图。请继续输入生图提示词、标注类别和训练参数后开始流程。\n\n"
                "请使用如下格式：\n"
                "prompt: 你的生图描述\n"
                "labels=person,cigarette\n"
                "conda_env_name=yolo_jetson model=yolo11n.pt epochs=10 imgsz=640 batch=8 "
                "device=0 workers=4 patience=20 dataset.split.train=0.7 dataset.split.val=0.2 dataset.split.test=0.1\n\n"
                "收到后我会自动：基于参考图生成图片、合并上传数据集、自动标注生成 COCO JSON，然后启动 YOLO 训练。"
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="completed",
                reply=reply,
                metadata={
                    "workflow": workflow,
                    "phase": "await_prompt",
                    "requires_prompt_text": True,
                    "requires_labels": True,
                    "requires_training_config": True,
                },
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.completed", {"result": result.model_dump()})
            return result, recorder.events

        if not labels:
            _set_waiting_prompt(paths, True)
            reply = (
                "已收到生图提示词。请补充自动标注类别后继续，例如：\n"
                "labels=person,cigarette\n"
                "也支持 label=person cigarette、classes=person,cigarette、标注类别：person，cigarette"
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="completed",
                reply=reply,
                metadata={"workflow": workflow, "phase": "await_labels", "requires_prompt_text": True, "requires_labels": True},
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.completed", {"result": result.model_dump()})
            return result, recorder.events

        if not _has_explicit_training_config(clean_user_text):
            _set_waiting_prompt(paths, True)
            reply = (
                "已收到生图提示词。为避免使用默认训练配置，请在同一条消息补充训练参数后继续，例如：\n"
                "conda_env_name=yolo_jetson model=yolo11n.pt epochs=10 imgsz=640 batch=8 "
                "device=0 workers=4 patience=20 dataset.split.train=0.7 dataset.split.val=0.2 dataset.split.test=0.1"
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="completed",
                reply=reply,
                metadata={"workflow": workflow, "phase": "await_prompt", "requires_prompt_text": True, "requires_training_config": True},
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.completed", {"result": result.model_dump()})
            return result, recorder.events

        _reset_generated_dir(generated_dir)
        _set_waiting_prompt(paths, False)
        recorder.emit("spec.started", {"selected_skills": ["image-dataset-generation", "data-auto-annotation", "gpu-training-orchestrator"], "attachment_count": len(attachments)})

        generation_spec = {
            "skill_name": "image-dataset-generation",
            "input_image": reference_image,
            "overrides_text": clean_user_text,
            "attachments": [_attachment_payload(image_attachment)] if image_attachment is not None else [],
            "workflow_context": {
                "output_dir": str(generated_dir.resolve()),
                "phase": "generation",
            },
        }
        recorder.emit("skill.started", {"skill_name": "image-dataset-generation", "attempt": 0})
        generation_result = self.skill_runner.run("image-dataset-generation", generation_spec, paths)
        recorder.emit("skill.completed", {"skill_name": "image-dataset-generation", "output_count": len(generation_result.outputs)})

        unpack_root = paths.workspace / "uploaded_dataset"
        dataset_root = _unpack_dataset_archive(dataset_pkg, unpack_root)
        reference_name = Path(reference_image).name if reference_image else ""
        merged_dataset_root = _merge_images_only(
            dataset_root,
            generated_dir,
            paths.workspace / "merged_dataset",
            excluded_image_names={reference_name} if reference_name else None,
        )
        annotation_spec = {
            "skill_name": "data-auto-annotation",
            "labels": labels,
            "overrides_text": clean_user_text,
            "attachments": [_attachment_payload(item) for item in attachments],
            "workflow_context": {
                "image_path": str((merged_dataset_root / "images").resolve()),
                "labels": labels,
                "phase": "annotation",
            },
        }
        recorder.emit("skill.started", {"skill_name": "data-auto-annotation", "attempt": 0})
        annotation_result = self.skill_runner.run("data-auto-annotation", annotation_spec, paths)
        recorder.emit("skill.completed", {"skill_name": "data-auto-annotation", "output_count": len(annotation_result.outputs)})
        try:
            merged_coco = _resolve_coco_output(paths, annotation_result.outputs)
        except FileNotFoundError:
            merged_coco = _run_annotation_fallback(merged_dataset_root, paths, labels)

        run_name = f"smoking_yolo_{paths.thread_id}"
        project_dir = str((paths.outputs / "training_runs").resolve())
        training_spec = {
            "skill_name": "gpu-training-orchestrator",
            "overrides_text": clean_user_text,
            "attachments": [_attachment_payload(item) for item in attachments],
            "workflow_context": {
                "dataset_root": str(merged_dataset_root),
                "coco_json": str(merged_coco),
                "class_names": labels,
                "project_dir": project_dir,
                "run_name": run_name,
                "phase": "training",
            },
        }

        recorder.emit("skill.started", {"skill_name": "gpu-training-orchestrator", "attempt": 0})
        training_result = self.skill_runner.run("gpu-training-orchestrator", training_spec, paths)
        recorder.emit("skill.completed", {"skill_name": "gpu-training-orchestrator", "output_count": len(training_result.outputs)})

        outputs = [*generation_result.outputs, *annotation_result.outputs, *training_result.outputs]
        for artifact in outputs:
            recorder.emit("artifact.created", {"artifact": artifact.model_dump()})
            recorder.emit("preview.ready", {"artifact": artifact.model_dump()})

        summary = _read_run_summary(paths)
        best_pt = _find_best_pt(paths)
        template_reply = ""
        if isinstance(training_result.data, dict):
            template_reply = str(training_result.data.get("final_reply") or "").strip()
        reply = _generate_model_controlled_reply(
            agent_config=agent_config,
            runtime_options=runtime_options,
            recorder=recorder,
            user_text=clean_user_text,
            workflow=workflow,
            labels=labels,
            summary=summary,
            best_pt=best_pt,
            merged_dataset_root=str(merged_dataset_root),
            merged_coco=str(merged_coco),
            generation_result=generation_result.data if isinstance(generation_result.data, dict) else {},
            annotation_result=annotation_result.data if isinstance(annotation_result.data, dict) else {},
            training_result=training_result.data if isinstance(training_result.data, dict) else {},
            fallback_reply=template_reply,
        )
        if not reply:
            reply = template_reply or (
                "训练已完成\n"
                f"- 保存目录: {summary.get('train_save_dir') or project_dir}\n"
                f"- best.pt: {best_pt if best_pt else '未生成'}"
            )

        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=paths.thread_id,
            status="completed",
            reply=reply,
            artifacts=outputs,
            verification=VerificationResult(passed=True, retry_count=0, checks=[], failed_checks=[]),
            metadata={
                "workflow": workflow,
                "phase": "training_completed",
                "training_spec": training_spec,
                "training_summary": summary,
                "best_pt": best_pt,
                "merged_dataset_root": str(merged_dataset_root),
                "merged_coco_json": str(merged_coco),
                "merge_summary": _read_merge_summary(paths),
                "labels": labels,
            },
        )
        recorder.emit("agent.message", {"text": reply})
        recorder.emit("run.completed", {"result": result.model_dump()})
        return result, recorder.events

    def _input_required_result(
        self,
        recorder: EventRecorder,
        agent_name: str,
        thread_id: str,
        workflow: str,
        required_inputs: list[dict[str, Any]],
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        labels = "?".join(_input_label(item.get("type")) for item in required_inputs)
        reply = f"请补充{labels}"
        result = AgentRunResult(
            agent=agent_name,
            thread_id=thread_id,
            status="completed",
            reply=reply,
            metadata={"workflow": workflow, "requires_input": True, "required_inputs": required_inputs},
        )
        recorder.emit("agent.message", {"text": reply})
        recorder.emit("run.completed", {"result": result.model_dump()})
        return result, recorder.events


def _find_dataset_package_attachment(attachments: list[Attachment]) -> Attachment | None:
    for item in attachments:
        name = item.name.lower()
        path = (item.path or "").lower()
        if any(name.endswith(ext) or path.endswith(ext) for ext in _DATASET_PACKAGE_EXTS):
            return item
    return None


def _attachment_payload(attachment: Any) -> dict[str, Any]:
    if isinstance(attachment, dict):
        return dict(attachment)
    model_dump = getattr(attachment, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _find_image_attachment(attachments: list[Attachment], *, include_thread_files: bool = True) -> Attachment | None:
    for item in attachments:
        if not include_thread_files and bool(item.metadata.get("thread_file")):
            continue
        mime = (item.mime_type or "").lower()
        name = item.name.lower()
        path = (item.path or "").lower()
        if mime.startswith("image/") or any(name.endswith(ext) or path.endswith(ext) for ext in _IMAGE_EXTS):
            return item
    return None


def _generate_model_controlled_reply(
    *,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
    user_text: str,
    workflow: str,
    labels: list[str],
    summary: dict[str, Any],
    best_pt: str,
    merged_dataset_root: str,
    merged_coco: str,
    generation_result: dict[str, Any],
    annotation_result: dict[str, Any],
    training_result: dict[str, Any],
    fallback_reply: str,
) -> str:
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit(
        "llm.started",
        {
            "model": llm.model,
            "temperature": llm.temperature,
            "top_p": llm.top_p,
            "max_tokens": llm.max_tokens,
            "request_timeout_seconds": llm.request_timeout_seconds,
            "configured": llm.configured,
            "purpose": "workflow_final_reply",
        },
    )
    if not llm.configured:
        recorder.emit("llm.completed", {"purpose": "workflow_final_reply", "used_fallback": True, "reason": "not_configured"})
        return _human_fallback_reply(fallback_reply, summary, best_pt)

    facts = {
        "workflow": workflow,
        "user_request": user_text,
        "requested_labels": labels,
        "dataset": {
            "merged_dataset_root": merged_dataset_root,
            "merged_coco_json": merged_coco,
            "num_images": summary.get("num_images"),
            "num_categories": summary.get("num_categories"),
            "class_names": summary.get("class_names"),
            "split_counts": summary.get("split_counts"),
            "merge_summary": _read_merge_summary_from_root(Path(merged_dataset_root)),
        },
        "training": {
            "status": "completed" if best_pt else "unknown",
            "conda_env_name": summary.get("conda_env_name"),
            "task": summary.get("task"),
            "model": summary.get("model"),
            "run_root": summary.get("run_root"),
            "train_save_dir": summary.get("train_save_dir"),
            "best_pt": best_pt,
            "eval_error": summary.get("eval_error"),
        },
        "evaluation": _extract_evaluation_facts(summary),
        "skill_results": {
            "image_dataset_generation": _small_dict(generation_result),
            "data_auto_annotation": _small_dict(annotation_result),
            "gpu_training_orchestrator": _small_dict(training_result),
        },
        "raw_training_reply_for_reference": fallback_reply,
    }
    system_prompt = (
        "你是算法工程师 Agent 的最终回复生成器。"
        "上游工作流已经完成技能调用，你只负责基于事实组织输出样式和表达。"
        "要求：使用中文；不要编造事实；保留关键路径、best.pt、数据划分、类别和评估指标；"
        "必须说明生成图片与上传数据集已经合并后的训练数据规模；"
        "如果 evaluation.results_dict 或 eval_block 中有 precision、recall、mAP50、mAP50-95，要在回复中明确列出；"
        "如果某个类别指标很差或为 0，要温和指出可能是样本/标注不足；"
        "输出应像专业算法训练报告，但不要机械复述 JSON。"
    )
    messages = [
        {
            "role": "user",
            "content": (
                "请根据以下工作流事实生成最终 Agent 回复。"
                "不要说“我是模型”；不要说无法访问文件；不要输出调试 JSON。\n\n"
                + json.dumps(facts, ensure_ascii=False, indent=2, default=str)
            ),
        }
    ]
    try:
        reply = llm.complete_sync(system_prompt, messages).strip()
    except Exception as exc:
        recorder.emit(
            "llm.completed",
            {"purpose": "workflow_final_reply", "used_fallback": True, "error": str(exc)[:1000]},
        )
        return _human_fallback_reply(fallback_reply, summary, best_pt)
    recorder.emit(
        "llm.completed",
        {
            "purpose": "workflow_final_reply",
            "used_fallback": not bool(reply),
            "reply_chars": len(reply),
        },
    )
    return reply or _human_fallback_reply(fallback_reply, summary, best_pt)


def _human_fallback_reply(fallback_reply: str, summary: dict[str, Any], best_pt: str) -> str:
    facts: dict[str, Any] = {}
    try:
        parsed = json.loads(fallback_reply) if fallback_reply.strip().startswith("{") else {}
        if isinstance(parsed, dict):
            facts = parsed
    except Exception:
        facts = {}
    if not facts:
        facts = {
            "model": summary.get("model"),
            "task": summary.get("task"),
            "num_images": summary.get("num_images"),
            "num_categories": summary.get("num_categories"),
            "split": summary.get("split_counts"),
            "train_dir": summary.get("train_save_dir"),
            "best_pt": best_pt,
            "eval_block": summary.get("eval_block"),
            "results_dict": _extract_results_dict(summary),
            "eval_error": summary.get("eval_error"),
        }

    split = facts.get("split")
    if isinstance(split, dict):
        split_text = f"训练 {split.get('train', '-')} / 验证 {split.get('val', '-')} / 测试 {split.get('test', '-')}"
    else:
        split_text = str(split or "-")

    lines = [
        "训练流程已完成，但模型总结暂时未返回内容，以下是工作流结果摘要：",
        "",
        f"- 模型：`{facts.get('model') or '-'}`",
        f"- 任务：`{facts.get('task') or '-'}`",
        f"- 数据规模：images={facts.get('num_images') or '-'} classes={facts.get('num_categories') or '-'}",
        f"- 数据划分：{split_text}",
        f"- 训练目录：`{facts.get('train_dir') or '-'}`",
        f"- best.pt：`{facts.get('best_pt') or best_pt or '-'}`",
    ]
    eval_block = str(facts.get("eval_block") or "").strip()
    if eval_block:
        lines.extend(["", "评估指标：", "```text", eval_block, "```"])
    results_dict = facts.get("results_dict")
    if isinstance(results_dict, dict) and results_dict:
        lines.extend(["", "关键指标："])
        for key, value in results_dict.items():
            lines.append(f"- {key}: {value}")
    eval_error = str(facts.get("eval_error") or "").strip()
    if eval_error:
        lines.extend(["", f"评估提示：{eval_error}"])
    return "\n".join(lines).strip()


def _extract_evaluation_facts(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_block": summary.get("eval_block"),
        "results_dict": _extract_results_dict(summary),
        "class_names": summary.get("class_names"),
        "eval_error": summary.get("eval_error"),
    }


def _extract_results_dict(summary: dict[str, Any]) -> dict[str, Any]:
    existing = summary.get("results_dict")
    if isinstance(existing, dict):
        return dict(existing)
    eval_results = str(summary.get("eval_results") or "")
    if not eval_results:
        return {}
    match = re.search(r"results_dict:\s*(\{[^\r\n]+\})", eval_results)
    if not match:
        return {}
    raw = match.group(1)
    try:
        import ast

        parsed = ast.literal_eval(raw)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _small_dict(value: dict[str, Any], *, max_value_chars: int = 1200) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"stdout", "stderr"}:
            text = str(item)
            out[key] = text[-max_value_chars:]
        elif key == "final_reply":
            out[key] = str(item)[:max_value_chars]
        elif isinstance(item, (str, int, float, bool)) or item is None:
            out[key] = item
        elif isinstance(item, (list, dict)):
            raw = json.dumps(item, ensure_ascii=False, default=str)
            out[key] = raw[:max_value_chars]
        else:
            out[key] = str(item)[:max_value_chars]
    return out


def _last_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content.strip()
    return ""


def _strip_uploaded_files_context(user_text: str) -> str:
    text = (user_text or "").strip()
    if not text:
        return ""
    marker = "Uploaded files available to tools:"
    idx = text.lower().find(marker.lower())
    if idx >= 0:
        text = text[:idx].strip()
    return text


def _has_explicit_training_config(user_text: str) -> bool:
    text = (user_text or "").lower()
    required = ("conda_env_name", "model", "epochs", "imgsz", "batch", "device", "workers", "patience")
    return all(k in text for k in required)


def _extract_annotation_labels(user_text: str) -> list[str]:
    text = (user_text or "").strip()
    if not text:
        return []
    match = re.search(
        r"(?:labels?|lables?|lable|classes?|class|标注类别|类别|标签|标注标签)\s*[:=：]\s*([^\r\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    raw = match.group(1)
    raw = re.split(
        r"\s+(?:conda_env_name|model|epochs|imgsz|batch|device|workers|patience|dataset\.split\.)\s*[:=]",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    parts = [item.strip().strip("\"'`，,;；。") for item in re.split(r"[,，;；\s]+", raw)]
    labels: list[str] = []
    seen: set[str] = set()
    for item in parts:
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        labels.append(item)
    return labels


def _extract_generation_prompt(user_text: str, *, allow_free_text: bool = False) -> str:
    text = _strip_uploaded_files_context(user_text)
    if not text:
        return ""
    match = re.search(r"(?:prompt|提示词)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return _strip_config_lines_from_prompt(match.group(1).strip())
    if not allow_free_text:
        return ""
    lowered = text.lower()
    if _is_generic_continue_text(text) or lowered in {"continue", "go on", "next"}:
        return ""
    if _looks_like_config_only_text(text):
        return ""
    return text if len(text) >= 12 else ""


def _is_generic_continue_text(text: str) -> bool:
    normalized = re.sub(r"[\s，,。.!！?？;；:：]+", "", (text or "").strip().lower())
    generic_cmds = {
        "\u7ee7\u7eed",
        "\u7ee7\u7eed\u5904\u7406",
        "\u7ee7\u7eed\u5904\u7406\u521a\u4e0a\u4f20\u7684\u6587\u4ef6",
        "\u5f00\u59cb",
        "\u5f00\u59cb\u5904\u7406",
        "\u7ee7\u7eed\u8bad\u7ec3",
        "\u4e0b\u4e00\u6b65",
        "continue",
        "goon",
        "next",
    }
    return normalized in generic_cmds


def _strip_config_lines_from_prompt(value: str) -> str:
    lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if _looks_like_config_only_text(stripped):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _looks_like_config_only_text(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    config_prefixes = (
        "labels", "label", "lable", "lables", "classes", "class",
        "conda_env_name", "model", "epochs", "imgsz", "batch", "device", "workers", "patience",
        "dataset.split.", "标注类别", "类别", "标签", "标注标签",
    )
    return any(lowered.startswith(prefix) for prefix in config_prefixes)


def _resolve_coco_output(paths: ThreadPaths, outputs: list[Any]) -> Path:
    for artifact in outputs:
        artifact_path = getattr(artifact, "path", "") or ""
        if not isinstance(artifact_path, str):
            continue
        try:
            local = ArtifactStore().resolve_virtual_path(paths.thread_id, artifact_path)
        except Exception:
            continue
        if local.suffix.lower() == ".json" and local.is_file() and _looks_like_coco_json(local):
            return local
    # Fallback 1: outputs json candidates
    json_candidates = sorted([p for p in paths.outputs.rglob("*.json") if p.is_file()])
    for p in reversed(json_candidates):
        n = p.name.lower()
        if (n == "coco.json" or n.endswith(".coco.json")) and _looks_like_coco_json(p):
            return p

    # Fallback 2: scan workspace for common coco filenames
    ws_candidates = sorted([p for p in paths.workspace.rglob("*.json") if p.is_file()])
    for p in reversed(ws_candidates):
        n = p.name.lower()
        if (n == "coco.json" or n.endswith(".coco.json") or "annotation" in n) and _looks_like_coco_json(p):
            return p

    raise FileNotFoundError("未找到可用的 coco.json")


def _looks_like_coco_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return all(key in payload for key in ("images", "annotations", "categories"))


def _run_annotation_fallback(image_dir: Path, paths: ThreadPaths, labels: list[str]) -> Path:
    script = Path(
        "plugins/skills/data-auto-annotation/skills/data-auto-annotation/scripts/sam3-predict.py"
    ).resolve()
    if not script.exists():
        raise FileNotFoundError(f"fallback 标注脚本不存在: {script}")
    out = (paths.outputs / "annotations.coco.json").resolve()
    target_dir = image_dir / "images" if (image_dir / "images").exists() else image_dir
    cmd = [
        "python",
        str(script),
        "--input-dir",
        str(target_dir.resolve()),
        "--text-prompts",
        *labels,
        "--output",
        str(out),
    ]
    env = os.environ.copy()
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if completed.returncode != 0 or not out.exists():
        raise RuntimeError(
            "data-auto-annotation 未产出 coco.json，且 fallback 标注失败。"
            f"\nstdout:\n{completed.stdout[-1200:]}\nstderr:\n{completed.stderr[-1200:]}"
        )
    return out


def _read_run_summary(paths: ThreadPaths) -> dict[str, Any]:
    candidates = list(paths.outputs.rglob("run_summary.json"))
    if not candidates:
        return {}
    latest = sorted(candidates)[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _find_best_pt(paths: ThreadPaths) -> str:
    candidates = sorted(paths.outputs.rglob("best.pt"))
    return str(candidates[-1]) if candidates else ""


def _input_label(value: object) -> str:
    return {"dataset": "数据集", "image": "图片", "model_config": "模型配置"}.get(str(value), "输入")


def _reference_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "reference_image_path.txt"


def _prompt_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "generation_prompt.txt"


def _save_generation_prompt(paths: ThreadPaths, prompt: str) -> None:
    if prompt:
        _prompt_marker_path(paths).write_text(prompt.strip(), encoding="utf-8")


def _load_generation_prompt(paths: ThreadPaths) -> str:
    marker = _prompt_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _labels_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "annotation_labels.json"


def _save_annotation_labels(paths: ThreadPaths, labels: list[str]) -> None:
    if labels:
        _labels_marker_path(paths).write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_annotation_labels(paths: ThreadPaths) -> list[str]:
    marker = _labels_marker_path(paths)
    if not marker.exists():
        return []
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [str(item).strip() for item in payload if str(item).strip()] if isinstance(payload, list) else []


def _save_reference_image_path(paths: ThreadPaths, path: str) -> None:
    if path:
        _reference_marker_path(paths).write_text(path.strip(), encoding="utf-8")


def _load_reference_image_path(paths: ThreadPaths) -> str:
    marker = _reference_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _waiting_prompt_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "awaiting_prompt.flag"


def _set_waiting_prompt(paths: ThreadPaths, waiting: bool) -> None:
    flag = _waiting_prompt_path(paths)
    if waiting:
        flag.write_text("1", encoding="utf-8")
    elif flag.exists():
        flag.unlink()


def _is_waiting_prompt(paths: ThreadPaths) -> bool:
    return _waiting_prompt_path(paths).exists()


def _dataset_package_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "dataset_package_path.txt"


def _save_dataset_package_path(paths: ThreadPaths, path: str) -> None:
    if path:
        _dataset_package_marker_path(paths).write_text(path.strip(), encoding="utf-8")


def _load_dataset_package_path(paths: ThreadPaths) -> str:
    marker = _dataset_package_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _reset_generated_dir(generated_dir: Path) -> None:
    if generated_dir.exists():
        for p in generated_dir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    generated_dir.mkdir(parents=True, exist_ok=True)


def _resolve_uploaded_local_path(thread_root: Path, raw_path: str) -> Path:
    value = (raw_path or "").strip()
    direct = Path(value).expanduser()
    if direct.exists():
        return direct.resolve()
    normalized = value.replace("\\\\", "/")
    uploads_dir = thread_root / "uploads"
    outputs_dir = thread_root / "outputs"
    for prefix, base in (("/mnt/user-data/uploads/", uploads_dir), ("d:/mnt/user-data/uploads/", uploads_dir), ("/mnt/user-data/outputs/", outputs_dir), ("d:/mnt/user-data/outputs/", outputs_dir)):
        if normalized.lower().startswith(prefix):
            rel = normalized[len(prefix):]
            candidate = (base / rel).resolve()
            if candidate.exists():
                return candidate
    basename = Path(normalized).name
    for base in (uploads_dir, outputs_dir):
        candidate = (base / basename).resolve()
        if basename and candidate.exists():
            return candidate
    return direct.resolve()


def _unpack_dataset_archive(archive_path: str, target_dir: Path) -> Path:
    source = _resolve_uploaded_local_path(target_dir.parent.parent, archive_path)
    if not source.exists():
        raise FileNotFoundError(f"找不到上传的文件: {source}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    lower_name = source.name.lower()
    if lower_name.endswith(".tar.gz"):
        shutil.unpack_archive(str(source), str(target_dir), format="gztar")
    elif lower_name.endswith(".tar"):
        shutil.unpack_archive(str(source), str(target_dir), format="tar")
    elif lower_name.endswith(".zip"):
        shutil.unpack_archive(str(source), str(target_dir), format="zip")
    else:
        shutil.unpack_archive(str(source), str(target_dir))
    child_dirs = [p for p in target_dir.iterdir() if p.is_dir()]
    return child_dirs[0] if len(child_dirs) == 1 else target_dir


def _collect_images(root: Path, *, excluded_names: set[str] | None = None) -> list[Path]:
    excluded = {n.lower() for n in (excluded_names or set()) if n}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    if not excluded:
        return files
    return [p for p in files if p.name.lower() not in excluded]


def _find_existing_coco_json(dataset_root: Path) -> Path | None:
    candidates = [p for p in dataset_root.rglob("*.json") if p.is_file()]
    for p in candidates:
        n = p.name.lower()
        if n == "coco.json" or n.endswith(".coco.json"):
            return p
    return candidates[0] if candidates else None


def _merge_coco_payloads(base: dict[str, Any], gen: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {"info": base.get("info") or gen.get("info") or {}, "licenses": base.get("licenses") or gen.get("licenses") or [], "images": [], "annotations": [], "categories": []}
    cat_name_to_newid: dict[str, int] = {}
    for cat in (base.get("categories") or []) + (gen.get("categories") or []):
        name = str(cat.get("name") or "").strip()
        if name and name not in cat_name_to_newid:
            cat_name_to_newid[name] = len(cat_name_to_newid) + 1
            merged["categories"].append({"id": cat_name_to_newid[name], "name": name, "supercategory": cat.get("supercategory", "")})

    def append_ds(payload: dict[str, Any]) -> None:
        id_map: dict[int, int] = {}
        existing = {str(i.get("file_name", "")): int(i.get("id", 0)) for i in merged["images"]}
        for img in payload.get("images") or []:
            old_id = int(img.get("id", 0))
            file_name = Path(str(img.get("file_name", ""))).name
            if file_name in existing:
                new_id = existing[file_name]
            else:
                new_id = len(merged["images"]) + 1
                existing[file_name] = new_id
                new_img = dict(img)
                new_img["id"] = new_id
                new_img["file_name"] = file_name
                merged["images"].append(new_img)
            id_map[old_id] = new_id
        for ann in payload.get("annotations") or []:
            old_img_id = int(ann.get("image_id", 0))
            new_img_id = id_map.get(old_img_id)
            if not new_img_id:
                continue
            old_cat_id = int(ann.get("category_id", 0))
            cat_name = ""
            for c in payload.get("categories") or []:
                if int(c.get("id", 0)) == old_cat_id:
                    cat_name = str(c.get("name") or "")
                    break
            new_cat_id = cat_name_to_newid.get(cat_name)
            if not new_cat_id:
                continue
            new_ann = dict(ann)
            new_ann["id"] = len(merged["annotations"]) + 1
            new_ann["image_id"] = new_img_id
            new_ann["category_id"] = new_cat_id
            merged["annotations"].append(new_ann)

    append_ds(base)
    append_ds(gen)
    return merged


def _merge_images_only(
    dataset_root: Path,
    generated_dir: Path,
    out_root: Path,
    *,
    excluded_image_names: set[str] | None = None,
) -> Path:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    merged_images = out_root / "images"
    merged_images.mkdir(parents=True, exist_ok=True)
    uploaded_count = 0
    generated_count = 0
    skipped_duplicates = 0
    for img in _collect_images(dataset_root):
        target = merged_images / img.name
        if not target.exists():
            shutil.copy2(img, target)
            uploaded_count += 1
        else:
            skipped_duplicates += 1
    for img in _collect_images(generated_dir, excluded_names=excluded_image_names):
        target = merged_images / img.name
        if not target.exists():
            shutil.copy2(img, target)
            generated_count += 1
        else:
            skipped_duplicates += 1
    summary = {
        "uploaded_images": uploaded_count,
        "generated_images": generated_count,
        "merged_images": len(_collect_images(merged_images)),
        "skipped_duplicates": skipped_duplicates,
        "dataset_root": str(dataset_root),
        "generated_dir": str(generated_dir),
        "merged_images_dir": str(merged_images),
    }
    (out_root / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_root


def _read_merge_summary(paths: ThreadPaths) -> dict[str, Any]:
    return _read_merge_summary_from_root(paths.workspace / "merged_dataset")


def _read_merge_summary_from_root(root: Path) -> dict[str, Any]:
    marker = root / "merge_summary.json"
    if not marker.is_file():
        return {}
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _prune_coco_by_existing_images(coco: dict[str, Any], images_dir: Path) -> dict[str, Any]:
    existing_names = {p.name for p in images_dir.glob("*") if p.is_file()}
    kept_images: list[dict[str, Any]] = []
    kept_ids: set[int] = set()
    for img in coco.get("images") or []:
        img_name = Path(str(img.get("file_name", ""))).name
        if img_name in existing_names:
            new_img = dict(img)
            new_img["file_name"] = img_name
            kept_images.append(new_img)
            try:
                kept_ids.add(int(new_img.get("id", 0)))
            except Exception:
                continue

    kept_annotations: list[dict[str, Any]] = []
    used_cat_ids: set[int] = set()
    for ann in coco.get("annotations") or []:
        try:
            image_id = int(ann.get("image_id", 0))
            cat_id = int(ann.get("category_id", 0))
        except Exception:
            continue
        if image_id in kept_ids:
            kept_annotations.append(dict(ann))
            used_cat_ids.add(cat_id)

    kept_categories: list[dict[str, Any]] = []
    for cat in coco.get("categories") or []:
        try:
            cat_id = int(cat.get("id", 0))
        except Exception:
            continue
        if cat_id in used_cat_ids:
            kept_categories.append(dict(cat))

    pruned = dict(coco)
    pruned["images"] = kept_images
    pruned["annotations"] = kept_annotations
    pruned["categories"] = kept_categories if kept_categories else (coco.get("categories") or [])
    return pruned
