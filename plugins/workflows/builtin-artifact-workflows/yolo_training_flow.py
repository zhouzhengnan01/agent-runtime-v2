from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.core.skills import SkillRunner
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions, VerificationResult

_DATASET_PACKAGE_EXTS = (".zip", ".tar", ".tar.gz")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
WORKFLOW_NAME = "yolo_training_flow"
WORKFLOW_OUTPUT_DIR = "yolo_training_flow"
PIPELINE_WORK_DIR = "pipeline_work"
TRAINING_MODEL_REGISTRY = "training_models.json"
DEFAULT_SELECTED_SKILLS = ["image-dataset-generation", "image-dataset-produce", "data-auto-annotation", "gpu-training-orchestrator"]
DEFAULT_TRAINING_SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
MIN_TEST_SPLIT = 0.1

DEFAULT_MAX_SYNTHETIC_IMAGES = 2000
button_epochs = False
FIXED_TRAINING_EPOCHS = 50
AUTO_GENERATE_MISSING_SPEC = True


class YoloTrainingWorkflow:
    """General YOLO training flow with optional synthetic data generation."""

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
        workflow = workflow_name or WORKFLOW_NAME
        paths = self.artifact_store.prepare_thread(thread_id)
        recorder = EventRecorder(agent=agent_config.name, thread_id=paths.thread_id, on_emit=on_event)
        recorder.emit("run.started", {"workflow": workflow})

        user_text = _last_user_text(messages)
        selected_skills = _selected_skills(runtime_options)
        if _has_runtime_selected_skills(runtime_options):
            _save_selected_skills(paths, selected_skills)
        else:
            selected_skills = _load_selected_skills(paths) or selected_skills
        generation_enabled = _capability_enabled(selected_skills, "image_generation")
        training_enabled = _capability_enabled(selected_skills, "training")
        waiting_prompt = _is_waiting_prompt(paths)
        workflow_completed = _is_workflow_completed(paths)
        workflow_output_root = paths.outputs / WORKFLOW_OUTPUT_DIR
        existing_dataset_pkg = _load_dataset_package_path(paths)
        existing_composite_image1 = _load_composite_image1_path(paths)
        existing_composite_image2 = _load_composite_image2_path(paths)
        current_objective = _training_objective_from_user_text(user_text)
        if current_objective:
            _save_training_objective(paths, current_objective)
        stored_objective = _load_training_objective(paths)
        objective_text = current_objective or (stored_objective if not workflow_completed or waiting_prompt else "")
        spec_user_text = _combine_spec_user_text(objective_text, user_text)
        explicit_attachment_roles = _parse_attachment_role_hints(user_text)

        dataset_attachment = _find_dataset_package_attachment(attachments, role_hints=explicit_attachment_roles, include_thread_files=False) if not existing_dataset_pkg else None
        composite_attachments = _find_composite_input_attachments(
            attachments,
            include_thread_files=False,
            dataset_attachment=dataset_attachment,
            allow_archives=bool(existing_dataset_pkg or dataset_attachment),
            role_hints=explicit_attachment_roles,
        )
        if not composite_attachments and waiting_prompt and not (existing_composite_image1 and existing_composite_image2):
            composite_attachments = _find_composite_input_attachments(
                attachments,
                include_thread_files=True,
                dataset_attachment=dataset_attachment,
                allow_archives=bool(existing_dataset_pkg or dataset_attachment),
                role_hints=explicit_attachment_roles,
            )

        if dataset_attachment and dataset_attachment.path:
            _set_workflow_completed(paths, False)
            resolved_dataset = _resolve_uploaded_local_path(paths.root, dataset_attachment.path)
            _save_dataset_package_path(paths, str(resolved_dataset))
            _clear_composite_input_paths(paths)
        if composite_attachments:
            _set_workflow_completed(paths, False)
            _save_composite_input_paths(paths, _prepare_composite_input_paths(paths, composite_attachments, explicit_attachment_roles))

        dataset_pkg = _load_dataset_package_path(paths)
        composite_image1 = _load_composite_image1_path(paths)
        composite_image2 = _load_composite_image2_path(paths)
        if dataset_pkg:
            dataset_pkg = str(_resolve_uploaded_local_path(paths.root, dataset_pkg))
        if composite_image1:
            composite_image1 = str(_resolve_uploaded_local_path(paths.root, composite_image1))
        if composite_image2:
            composite_image2 = str(_resolve_uploaded_local_path(paths.root, composite_image2))

        if not _is_yolo_training_intent(
            user_text,
            attachments,
            agent_config=agent_config,
            runtime_options=runtime_options,
            recorder=recorder,
            has_active_training_state=bool((dataset_pkg or waiting_prompt) and not workflow_completed),
        ):
            reply = _chat_reply_for_non_training_intent(
                user_text,
                agent_config=agent_config,
                runtime_options=runtime_options,
                recorder=recorder,
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="completed",
                reply=reply,
                metadata={"workflow": workflow, "phase": "chat", "training_intent": False},
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.completed", {"result": result.model_dump()})
            return result, recorder.events

        requested_training_model_id = _requested_training_model_id(runtime_options)
        user_training_model, user_training_model_error = _resolve_user_training_model(
            paths,
            requested_training_model_id,
        )
        if user_training_model_error:
            return self._model_spec_failed_result(
                recorder,
                agent_config.name,
                paths.thread_id,
                workflow,
                f"用户上传的训练模型不可用：{user_training_model_error}",
                phase="user_training_model_invalid",
            )

        model_managed_spec = _auto_generate_missing_spec(runtime_options)
        if model_managed_spec:
            generation_enabled = True
            training_enabled = True
            selected_skills = _ensure_full_cycle_skills(selected_skills)
            request_spec = _generate_model_managed_yolo_training_intent_spec(
                user_text=spec_user_text,
                agent_config=agent_config,
                runtime_options=runtime_options,
                recorder=recorder,
            )
            request_spec = _ensure_intent_labels(request_spec, spec_user_text)
            if _spec_string(request_spec, "generation_prompt"):
                request_spec["use_synthetic_generation"] = True
        else:
            request_spec = _extract_yolo_training_request_spec(
                spec_user_text,
                agent_config=agent_config,
                runtime_options=runtime_options,
                recorder=recorder,
            )
            request_spec = _ensure_intent_labels(request_spec, spec_user_text)
        synthetic_generation = _spec_optional_bool(request_spec, "use_synthetic_generation")
        if synthetic_generation is not None:
            _save_synthetic_generation_enabled(paths, synthetic_generation)
        persisted_synthetic_generation = _load_synthetic_generation_enabled(paths)
        if model_managed_spec:
            generation_enabled = True
            _save_synthetic_generation_enabled(paths, True)
        elif persisted_synthetic_generation is not None:
            generation_enabled = generation_enabled and persisted_synthetic_generation

        if model_managed_spec:
            prompt_text = _spec_string(request_spec, "generation_prompt")
            labels = _normalize_detection_labels(_spec_string_list(request_spec, "labels"))
            training_cfg: dict[str, Any] = {}
            training_cfg_available = False
            task_description = _spec_string(request_spec, "task_description")
            if prompt_text:
                _save_generation_prompt(paths, prompt_text)
            if labels:
                _save_annotation_labels(paths, labels)
            if task_description:
                _save_detection_task_description(paths, task_description)
            generation_enabled = True
            _save_synthetic_generation_enabled(paths, True)
        else:
            prompt_text = _spec_string(request_spec, "generation_prompt") or _extract_generation_prompt(spec_user_text, allow_free_text=waiting_prompt)
            if prompt_text:
                _save_generation_prompt(paths, prompt_text)
            else:
                prompt_text = _load_generation_prompt(paths)
            labels = _normalize_detection_labels(_spec_string_list(request_spec, "labels") or _extract_annotation_labels(spec_user_text))
            if labels:
                _save_annotation_labels(paths, labels)
            else:
                labels = _load_annotation_labels(paths)
            training_cfg = _spec_training_config(request_spec)
            training_cfg_available = bool(training_cfg)
            if training_cfg:
                _save_training_config(paths, training_cfg)
            else:
                training_cfg = _load_training_config(paths)
                training_cfg_available = bool(training_cfg)
            if not training_cfg:
                training_cfg = _extract_training_config(spec_user_text)
                training_cfg_available = _has_explicit_training_config(spec_user_text)
                if training_cfg_available:
                    _save_training_config(paths, training_cfg)
            task_description = _spec_string(request_spec, "task_description")
            if task_description:
                _save_detection_task_description(paths, task_description)
            else:
                task_description = _load_detection_task_description(paths)

        if not dataset_pkg or (generation_enabled and (not composite_image1 or not composite_image2)):
            required_inputs = []
            if not dataset_pkg:
                required_inputs.append({"type": "dataset", "accept": ".zip,.tar,.tar.gz", "required": True, "reason": "需要上传 datasets.zip 数据集压缩包"})
            if generation_enabled:
                if not composite_image1:
                    required_inputs.append({"type": "image", "accept": ".zip,.tar,.tar.gz", "required": True, "reason": "需要上传 image1.zip 场景/背景图片压缩包"})
                if not composite_image2:
                    required_inputs.append({"type": "image", "accept": ".zip,.tar,.tar.gz", "required": True, "reason": "需要上传 image2.zip 目标/前景图片压缩包"})
            return self._input_required_result(
                recorder,
                agent_config.name,
                paths.thread_id,
                workflow,
                required_inputs,
            )

        if generation_enabled and not prompt_text:
            _set_waiting_prompt(paths, True)
            if _auto_generate_missing_spec(runtime_options):
                reply = "模型未能生成合成提示词，无法继续自动合成数据。请检查模型配置或重试。"
                status = "failed"
                event_type = "run.failed"
                metadata_phase = "model_spec_completion_failed"
            else:
                reply = "请继续输入合成提示词，格式为 prompt: 你的描述"
                status = "completed"
                event_type = "run.completed"
                metadata_phase = "await_prompt"
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status=status,
                reply=reply,
                metadata={"workflow": workflow, "phase": metadata_phase, "requires_prompt_text": True},
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit(event_type, {"result": result.model_dump()})
            return result, recorder.events

        if not labels:
            _set_waiting_prompt(paths, True)
            if _auto_generate_missing_spec(runtime_options):
                reply = "模型未能生成自动标注类别，无法继续自动标注和训练。请检查模型配置或重试。"
                status = "failed"
                event_type = "run.failed"
                metadata_phase = "model_spec_completion_failed"
            else:
                reply = (
                    "请补充自动标注类别后继续，例如：\n"
                    "labels=person,cigarette\n"
                    "也支持 label=person cigarette、classes=person,cigarette、标注类别：person，cigarette"
                )
                status = "completed"
                event_type = "run.completed"
                metadata_phase = "await_labels"
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status=status,
                reply=reply,
                metadata={"workflow": workflow, "phase": metadata_phase, "requires_prompt_text": True, "requires_labels": True},
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit(event_type, {"result": result.model_dump()})
            return result, recorder.events

        if training_enabled and not training_cfg_available and not model_managed_spec:
            _set_waiting_prompt(paths, True)
            if _auto_generate_missing_spec(runtime_options):
                reply = "模型未能生成完整训练参数，无法继续自动训练。请检查模型配置或重试。"
                status = "failed"
                event_type = "run.failed"
                metadata_phase = "model_spec_completion_failed"
            else:
                reply = (
                    "为避免使用默认训练配置，请在同一条消息补充训练参数后继续，例如：\n"
                    "conda_env_name=yolo_jetson model=yolo11n.pt epochs=10 imgsz=640 batch=8 "
                    "device=0 workers=4 patience=20 dataset.split.train=0.7 dataset.split.val=0.2 dataset.split.test=0.1"
                )
                status = "completed"
                event_type = "run.completed"
                metadata_phase = "await_prompt"
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status=status,
                reply=reply,
                metadata={"workflow": workflow, "phase": metadata_phase, "requires_prompt_text": True, "requires_training_config": True},
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit(event_type, {"result": result.model_dump()})
            return result, recorder.events

        _reset_generated_dir(workflow_output_root / PIPELINE_WORK_DIR)
        _reset_generated_dir(workflow_output_root / "training_run")
        _set_waiting_prompt(paths, False)
        recorder.emit("spec.started", {"selected_skills": selected_skills, "attachment_count": len(attachments)})

        unpack_root = workflow_output_root / "uploaded_dataset"
        dataset_root = _unpack_dataset_archive(dataset_pkg, unpack_root, paths.root)
        dataset_facts = _analyze_uploaded_dataset(dataset_root, workflow_output_root / PIPELINE_WORK_DIR, labels, recorder)
        if model_managed_spec:
            request_spec = _generate_dataset_aware_yolo_training_request_spec(
                base_spec=request_spec,
                dataset_facts=dataset_facts,
                user_text=spec_user_text,
                agent_config=agent_config,
                runtime_options=runtime_options,
                recorder=recorder,
            )
            request_spec = _fallback_model_managed_yolo_training_request_spec(request_spec, spec_user_text, dataset_facts)
            request_spec = _ensure_intent_labels(request_spec, spec_user_text)
            synthetic_generation = _spec_optional_bool(request_spec, "use_synthetic_generation")
            if synthetic_generation is not None:
                generation_enabled = synthetic_generation
                _save_synthetic_generation_enabled(paths, synthetic_generation)
            prompt_text = _spec_string(request_spec, "generation_prompt") or prompt_text
            labels = _normalize_detection_labels(_spec_string_list(request_spec, "labels") or labels)
            training_cfg = _spec_training_config(request_spec)
            training_cfg_available = bool(training_cfg)
            task_description = _spec_string(request_spec, "task_description") or task_description
            if prompt_text:
                _save_generation_prompt(paths, prompt_text)
            if labels:
                _save_annotation_labels(paths, labels)
            if task_description:
                _save_detection_task_description(paths, task_description)
            if training_cfg:
                _apply_epochs_policy(training_cfg, runtime_options)
                request_spec["training"]["epochs"] = training_cfg["training"]["epochs"]
                _force_current_runtime(training_cfg)
                _save_training_config(paths, training_cfg)
            else:
                return self._model_spec_failed_result(
                    recorder,
                    agent_config.name,
                    paths.thread_id,
                    workflow,
                    "模型未能在分析数据集后生成完整训练参数，无法继续自动训练。请检查模型配置或重试。",
                    phase="dataset_aware_training_spec_failed",
                    metadata={"dataset_facts": dataset_facts, "model_generated_spec": _model_generated_spec_payload(request_spec)},
                )
            _emit_model_generated_spec(recorder, request_spec, dataset_facts=dataset_facts)
            _write_model_generated_spec_logs(workflow_output_root, request_spec, dataset_facts=dataset_facts)
        else:
            _apply_epochs_policy(training_cfg, runtime_options)
            _force_current_runtime(training_cfg)

        if user_training_model:
            _apply_user_training_model(training_cfg, user_training_model)
            _save_training_config(paths, training_cfg)
            request_training = request_spec.get("training")
            if isinstance(request_training, dict):
                request_training["model"] = user_training_model["local_path"]
                request_training["model_source"] = "user_upload"
                request_training["strict_model"] = True
                request_training["model_sha256"] = user_training_model["sha256"]
                request_training["model_id"] = user_training_model["modelId"]
            recorder.emit(
                "workflow.user_model_selected",
                {
                    "modelId": user_training_model["modelId"],
                    "name": user_training_model["name"],
                    "path": user_training_model["path"],
                    "sha256": user_training_model["sha256"],
                    "source": "user_upload",
                },
            )
        else:
            _clear_user_training_model_selection(training_cfg, request_spec)
            _save_training_config(paths, training_cfg)

        pipeline_work_dir = str((workflow_output_root / PIPELINE_WORK_DIR).resolve())
        project_dir = str((workflow_output_root / "training_run").resolve())
        run_name = "."
        if not task_description:
            task_description = _extract_detection_task_description(user_text, prompt_text, labels)
        data_prep_output_dir = str((workflow_output_root / "prepared_data").resolve())
        data_prep_spec = {
            "skill_name": "data-auto-annotation",
            "overrides_text": user_text,
            "attachments": [
                _attachment_payload(item)
                for item in attachments
                if not _is_training_model_attachment(item)
            ],
            "dataset_root": str(dataset_root),
            "image1": composite_image1,
            "image2": composite_image2,
            "task": task_description,
            "generation_prompt": prompt_text,
            "labels": labels,
            "work_dir": pipeline_work_dir,
            "output_dir": data_prep_output_dir,
            "skip_generation": not generation_enabled,
            "max_synthetic": _max_synthetic_images(runtime_options),
            "synthetic_count_button": True,
            "produce_count_button": True,
            "register_artifacts": not training_enabled,
            "split_requested": training_enabled,
            "split": training_cfg["split"],
            "training": training_cfg["training"],
            "planner_llm": _planner_llm_config(agent_config, runtime_options),
            "workflow_context": {
                "dataset_root": str(dataset_root),
                "image1": composite_image1,
                "image2": composite_image2,
                "task": task_description,
                "generation_prompt": prompt_text,
                "class_names": labels,
                "labels": labels,
                "skip_generation": not generation_enabled,
                "split_requested": training_enabled,
                "work_dir": pipeline_work_dir,
                "output_dir": data_prep_output_dir,
                "run_name": run_name,
                "phase": "data_preparation",
                "max_synthetic": _max_synthetic_images(runtime_options),
                "synthetic_count_button": True,
                "produce_count_button": True,
                "register_artifacts": not training_enabled,
            },
        }

        recorder.emit("skill.started", {"skill_name": "data-auto-annotation", "attempt": 0})
        annotation_result = self.skill_runner.run("data-auto-annotation", data_prep_spec, paths, on_event=recorder.emit)
        recorder.emit("skill.completed", {"skill_name": "data-auto-annotation", "output_count": len(annotation_result.outputs)})

        data_prep_data = annotation_result.data if isinstance(annotation_result.data, dict) else {}
        data_prep_returncode = int(data_prep_data.get("returncode", 0) or 0)
        if data_prep_returncode != 0:
            outputs = [*annotation_result.outputs]
            for artifact in outputs:
                recorder.emit("artifact.created", {"artifact": artifact.model_dump()})
                recorder.emit("preview.ready", {"artifact": artifact.model_dump()})
            stderr_tail = str(data_prep_data.get("stderr") or "").strip()
            stdout_tail = str(data_prep_data.get("stdout") or "").strip()
            reply = (
                "数据处理流程失败，尚未进入生图和流式标注阶段。\n\n"
                f"- 失败阶段：`data-auto-annotation`\n"
                f"- returncode：`{data_prep_returncode}`\n"
                f"- 主要错误：\n```text\n{(stderr_tail or stdout_tail)[-2000:]}\n```"
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="failed",
                reply=reply,
                artifacts=outputs,
                verification=VerificationResult(passed=False, retry_count=0, checks=[], failed_checks=["data-auto-annotation failed"]),
                metadata={
                    "workflow": workflow,
                    "phase": "data_preparation_failed",
                    "data_preparation_spec": data_prep_spec,
                    "data_preparation_result": data_prep_data,
                    "labels": labels,
                },
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.failed", {"result": result.model_dump(), "error": stderr_tail or stdout_tail})
            return result, recorder.events

        dataset_yaml = _resolve_prepared_dataset_yaml(data_prep_data, Path(data_prep_output_dir))

        if not training_enabled:
            _set_waiting_prompt(paths, False)
            _set_workflow_completed(paths, True)
            outputs = [*annotation_result.outputs]
            for artifact in outputs:
                recorder.emit("artifact.created", {"artifact": artifact.model_dump()})
                recorder.emit("preview.ready", {"artifact": artifact.model_dump()})
            pipeline_paths = _read_pipeline_paths(paths)
            summary = _read_data_preparation_summary(paths)
            reply = (
                "数据处理流程已完成，当前未选择 `gpu-training-orchestrator`，所以不会要求训练参数，也不会启动 YOLO 训练。\n\n"
                f"- prepared_dataset: `{summary.get('prepared_dataset') or data_prep_output_dir}`\n"
                f"- dataset.yaml: `{summary.get('dataset_yaml') or dataset_yaml}`\n"
                f"- synthetic_plan: `{pipeline_paths.get('synthetic_plan') or '未生成或未启用生图'}`"
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
                    "phase": "data_preparation_completed",
                    "data_preparation_spec": data_prep_spec,
                    "data_preparation_summary": summary,
                    "dataset_yaml": dataset_yaml,
                    "labels": labels,
                },
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.completed", {"result": result.model_dump()})
            return result, recorder.events

        training_spec = {
            "skill_name": "gpu-training-orchestrator",
            "overrides_text": user_text,
            "data_yaml": dataset_yaml,
            "project_dir": project_dir,
            "run_name": run_name,
            "training": training_cfg["training"],
            "runtime": _current_runtime_config(training_cfg.get("runtime", {})),
            "workflow_context": {
                "dataset_yaml": dataset_yaml,
                "project_dir": project_dir,
                "run_name": run_name,
                "phase": "training",
            },
        }

        recorder.emit("skill.started", {"skill_name": "gpu-training-orchestrator", "attempt": 0})
        training_result = self.skill_runner.run("gpu-training-orchestrator", training_spec, paths)
        recorder.emit("skill.completed", {"skill_name": "gpu-training-orchestrator", "output_count": len(training_result.outputs)})

        outputs = _filter_training_run_artifacts(training_result.outputs)
        for artifact in outputs:
            recorder.emit("artifact.created", {"artifact": artifact.model_dump()})
            recorder.emit("preview.ready", {"artifact": artifact.model_dump()})

        pipeline_paths = _read_pipeline_paths(paths)
        summary = _read_run_summary(paths)
        best_pt = _find_best_pt(paths)
        template_reply = ""
        if isinstance(training_result.data, dict):
            template_reply = str(training_result.data.get("final_reply") or "").strip()
        training_returncode = 0
        if isinstance(training_result.data, dict):
            try:
                training_returncode = int(training_result.data.get("returncode", 0) or 0)
            except (TypeError, ValueError):
                training_returncode = 1
        if training_returncode != 0:
            _set_waiting_prompt(paths, False)
            _set_workflow_completed(paths, False)
            reply = _training_failed_reply(
                summary=summary,
                fallback_reply=template_reply,
                best_pt=best_pt,
                data_preparation_summary=_read_data_preparation_summary(paths),
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="failed",
                reply=reply,
                artifacts=outputs,
                verification=VerificationResult(passed=False, retry_count=0, checks=[], failed_checks=["gpu-training-orchestrator failed"]),
                metadata={
                    "workflow": workflow,
                    "phase": "training_failed",
                    "training_spec": training_spec,
                    "data_preparation_spec": data_prep_spec,
                    "training_summary": summary,
                    "training_result": training_result.data if isinstance(training_result.data, dict) else {},
                    "best_pt": best_pt,
                    "model_generated_spec": _model_generated_spec_payload(request_spec),
                    "merged_dataset_root": pipeline_paths.get("dataset_root") or str(dataset_root),
                    "merged_coco_json": pipeline_paths.get("coco_json") or "",
                    "synthetic_plan": pipeline_paths.get("synthetic_plan") or "",
                    "training_input": pipeline_paths.get("training_input") or "",
                    "labels": labels,
                },
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.failed", {"result": result.model_dump(), "error": reply[:2000]})
            return result, recorder.events
        reply = _generate_model_controlled_reply(
            agent_config=agent_config,
            runtime_options=runtime_options,
            recorder=recorder,
            user_text=user_text,
            workflow=workflow,
            labels=labels,
            summary=summary,
            best_pt=best_pt,
            merged_dataset_root=pipeline_paths.get("dataset_root") or str(dataset_root),
            merged_coco=pipeline_paths.get("coco_json") or "",
            generation_result={},
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
        _set_waiting_prompt(paths, False)
        _set_workflow_completed(paths, True)

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
                "data_preparation_spec": data_prep_spec,
                "training_summary": summary,
                "best_pt": best_pt,
                "model_generated_spec": _model_generated_spec_payload(request_spec),
                "merged_dataset_root": pipeline_paths.get("dataset_root") or str(dataset_root),
                "merged_coco_json": pipeline_paths.get("coco_json") or "",
                "synthetic_plan": pipeline_paths.get("synthetic_plan") or "",
                "training_input": pipeline_paths.get("training_input") or "",
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

    def _model_spec_failed_result(
        self,
        recorder: EventRecorder,
        agent_name: str,
        thread_id: str,
        workflow: str,
        reply: str,
        *,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[AgentRunResult, list[ChatEvent]]:
        result = AgentRunResult(
            agent=agent_name,
            thread_id=thread_id,
            status="failed",
            reply=reply,
            metadata={"workflow": workflow, "phase": phase, **(metadata or {})},
        )
        recorder.emit("agent.message", {"text": reply})
        recorder.emit("run.failed", {"result": result.model_dump(), "error": reply})
        return result, recorder.events


SmokingDetectionTrainingWorkflow = YoloTrainingWorkflow


def _find_dataset_package_attachment(
    attachments: list[Attachment],
    role_hints: dict[str, str] | None = None,
    *,
    include_thread_files: bool = True,
) -> Attachment | None:
    candidates = [item for item in attachments if include_thread_files or not _is_thread_file_attachment(item)]
    hinted = _attachment_for_role(candidates, "dataset", role_hints or {})
    if hinted is not None:
        return hinted
    named_candidates: list[Attachment] = []
    fallback_candidates: list[Attachment] = []
    for item in candidates:
        name = item.name.lower()
        path = (item.path or "").lower()
        if any(name.endswith(ext) or path.endswith(ext) for ext in _DATASET_PACKAGE_EXTS):
            if _looks_like_dataset_name(name, path):
                named_candidates.append(item)
            elif not _looks_like_composite_role_name(name, path):
                fallback_candidates.append(item)
    return (named_candidates or fallback_candidates or [None])[0]


def _is_training_model_attachment(item: Attachment) -> bool:
    name = str(item.name or "").strip()
    path = str(item.path or "").strip()
    if not (name.lower().endswith(".pt") or path.lower().endswith(".pt")):
        return False
    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    role = str(metadata.get("role") or "").strip().lower()
    source = str(metadata.get("source") or "").strip().lower()
    normalized_path = path.replace("\\", "/").lower()
    return role == "training_model" or source == "user_upload" or "/uploads/models/" in normalized_path


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
        if not include_thread_files and _is_thread_file_attachment(item):
            continue
        mime = (item.mime_type or "").lower()
        name = item.name.lower()
        path = (item.path or "").lower()
        if mime.startswith("image/") or any(name.endswith(ext) or path.endswith(ext) for ext in _IMAGE_EXTS):
            return item
    return None


def _find_composite_input_attachments(
    attachments: list[Attachment],
    *,
    include_thread_files: bool = True,
    dataset_attachment: Attachment | None = None,
    allow_archives: bool = False,
    role_hints: dict[str, str] | None = None,
) -> list[Attachment]:
    dataset_path = str(getattr(dataset_attachment, "path", "") or "")
    hints = role_hints or {}
    candidates = [item for item in attachments if include_thread_files or not _is_thread_file_attachment(item)]
    hinted_image1 = _attachment_for_role(candidates, "image1", hints)
    hinted_image2 = _attachment_for_role(candidates, "image2", hints)
    if hinted_image1 is not None or hinted_image2 is not None:
        return [item for item in (hinted_image1, hinted_image2) if item is not None and _is_composite_input_attachment(item, allow_archives=True)]

    image1_candidates: list[Attachment] = []
    image2_candidates: list[Attachment] = []
    fallback_candidates: list[Attachment] = []
    seen_paths: set[str] = set()
    for item in candidates:
        if dataset_path and str(item.path or "") == dataset_path:
            continue
        path_key = str(item.path or "").strip().lower()
        if path_key and path_key in seen_paths:
            continue
        if _is_composite_input_attachment(item, allow_archives=allow_archives):
            seen_paths.add(path_key)
            name = item.name.lower()
            path = (item.path or "").lower()
            if _looks_like_image1_name(name, path):
                image1_candidates.append(item)
            elif _looks_like_image2_name(name, path):
                image2_candidates.append(item)
            else:
                fallback_candidates.append(item)

    result: list[Attachment] = []
    if image1_candidates:
        result.append(image1_candidates[0])
    if image2_candidates and (not result or str(image2_candidates[0].path or "") != str(result[0].path or "")):
        result.append(image2_candidates[0])
    for item in fallback_candidates:
        if len(result) >= 2:
            break
        if all(str(item.path or "") != str(existing.path or "") for existing in result):
            result.append(item)
    return result[:2]


def _is_composite_input_attachment(item: Attachment, *, allow_archives: bool = False) -> bool:
    mime = (item.mime_type or "").lower()
    name = item.name.lower()
    path_text = (item.path or "").lower()
    if mime.startswith("image/") or any(name.endswith(ext) or path_text.endswith(ext) for ext in _IMAGE_EXTS):
        return True
    if any(name.endswith(ext) or path_text.endswith(ext) for ext in _DATASET_PACKAGE_EXTS):
        return allow_archives
    raw_path = str(item.path or "").strip()
    if not raw_path:
        return False
    try:
        path = Path(raw_path)
        return path.exists() and path.is_dir()
    except Exception:
        return False


def _is_thread_file_attachment(item: Attachment) -> bool:
    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    return bool(metadata.get("thread_file"))


def _parse_attachment_role_hints(user_text: str) -> dict[str, str]:
    text = user_text or ""
    hints: dict[str, str] = {}
    patterns = {
        "dataset": r"(?:dataset|数据集)\s*[:=：]\s*([^\s,，;；]+)",
        "image1": r"(?:image1|图1|图片1|合成图1|背景图|背景)\s*[:=：]\s*([^\s,，;；]+)",
        "image2": r"(?:image2|图2|图片2|合成图2|前景图|目标图|目标)\s*[:=：]\s*([^\s,，;；]+)",
    }
    for role, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hints[role] = match.group(1).strip().strip('"\'')
    return hints


def _attachment_for_role(attachments: list[Attachment], role: str, role_hints: dict[str, str]) -> Attachment | None:
    hint = (role_hints.get(role) or "").lower()
    for item in attachments:
        name = item.name.lower()
        path = (item.path or "").lower()
        if hint and (hint in name or hint in path):
            return item
    if role == "dataset":
        for item in attachments:
            if _looks_like_dataset_name(item.name.lower(), (item.path or "").lower()):
                return item
    if role == "image1":
        for item in attachments:
            if _looks_like_image1_name(item.name.lower(), (item.path or "").lower()):
                return item
    if role == "image2":
        for item in attachments:
            if _looks_like_image2_name(item.name.lower(), (item.path or "").lower()):
                return item
    return None


def _looks_like_dataset_name(name: str, path: str) -> bool:
    text = f"{name} {path}".lower()
    return any(token in text for token in ("dataset", "datasets", "train_dataset", "数据集", "训练集"))


def _looks_like_image1_name(name: str, path: str) -> bool:
    text = f"{name} {path}".lower()
    return any(token in text for token in ("image1", "img1", "background", "bg", "scene", "reference", "ref", "背景", "场景", "参考"))


def _looks_like_image2_name(name: str, path: str) -> bool:
    text = f"{name} {path}".lower()
    return any(token in text for token in ("image2", "img2", "foreground", "fg", "target", "object", "generated", "目标", "前景", "主体"))


def _looks_like_composite_role_name(name: str, path: str) -> bool:
    return _looks_like_image1_name(name, path) or _looks_like_image2_name(name, path)


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

    evaluation = _extract_evaluation_facts(summary)
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
            "source_counts": summary.get("source_counts"),
            "synthetic_generation": _synthetic_generation_facts(summary),
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
        "evaluation": evaluation,
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
        "合成数据状态必须以 dataset.synthetic_generation.status 为准；"
        "如果 status=merged 且 fallback 存在，说明主合成接口失败但 fallback 已成功补救，不要写成数据合成失败；"
        "如果 evaluation.metrics 中存在 precision、recall、mAP50、mAP50_95、fitness，必须在回复中明确列出；"
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


def _is_yolo_training_intent(
    user_text: str,
    attachments: list[Attachment],
    *,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
    has_active_training_state: bool = False,
) -> bool:
    if _has_dataset_attachment(attachments, include_thread_files=False):
        return True
    if has_active_training_state and _find_composite_input_attachments(attachments, include_thread_files=False, allow_archives=True):
        return True
    if _looks_like_yolo_training_request(user_text):
        return True
    if _looks_like_plain_chat(user_text):
        return False
    if not user_text.strip():
        return False

    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_intent_router"})
    if not llm.configured:
        recorder.emit("llm.completed", {"purpose": "workflow_intent_router", "used_fallback": True, "reason": "not_configured"})
        return False

    system_prompt = (
        "你是应用内的意图路由器。判断用户是否明确想启动 YOLO/目标检测模型训练、数据集自动标注、"
        "数据集划分、合成数据生成、训练流程，或是否在继续补充上一次训练流程所需参数。"
        "普通问候、闲聊、能力询问、说明性问题都不是训练意图，即使当前线程之前训练过也不是。"
        "只返回 JSON：{\"is_training_intent\": true/false, \"reason\": \"...\"}。"
    )
    messages = [
        Message(
            role="user",
            content=(
                f"用户消息：{user_text}\n"
                f"附件数量：{len(attachments)}\n"
                f"线程是否存在未清理的训练状态：{has_active_training_state}\n"
                "如果用户只是聊天，请返回 false；如果用户在补 labels、conda_env_name、epochs、数据集、"
                "生图提示词或明确说继续训练，请返回 true。"
            ),
        )
    ]
    try:
        raw = llm.complete_sync(system_prompt, messages)
        payload = _parse_json_object(raw)
        decision = bool(payload.get("is_training_intent"))
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_intent_router",
                "is_training_intent": decision,
                "reason": str(payload.get("reason") or ""),
            },
        )
        return decision
    except Exception as exc:
        recorder.emit("llm.completed", {"purpose": "workflow_intent_router", "used_fallback": True, "error": str(exc)[:1000]})
        return False


def _chat_reply_for_non_training_intent(
    user_text: str,
    *,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
) -> str:
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_chat_reply"})
    if llm.configured:
        try:
            system_prompt = (
                "你是算法工程师全流程应用里的助手。当前用户没有明确请求 YOLO 训练流程，"
                "请按普通聊天自然回复。不要要求上传数据集，除非用户明确提出训练、标注或数据处理。"
            )
            reply = llm.complete_sync(system_prompt, [Message(role="user", content=user_text)]).strip()
            if reply:
                recorder.emit("llm.completed", {"purpose": "workflow_chat_reply", "used_fallback": False})
                return reply
        except Exception as exc:
            recorder.emit("llm.completed", {"purpose": "workflow_chat_reply", "used_fallback": True, "error": str(exc)[:1000]})
    else:
        recorder.emit("llm.completed", {"purpose": "workflow_chat_reply", "used_fallback": True, "reason": "not_configured"})
    return "你好，我在。你可以直接和我聊天；如果需要训练 YOLO 检测模型，再告诉我任务并上传数据集。"


def _has_dataset_attachment(attachments: list[Attachment], *, include_thread_files: bool = True) -> bool:
    return _find_dataset_package_attachment(attachments, include_thread_files=include_thread_files) is not None


def _looks_like_yolo_training_request(user_text: str) -> bool:
    text = (user_text or "").lower()
    positive_markers = (
        "yolo",
        "训练",
        "模型训练",
        "目标检测",
        "检测模型",
        "自动标注",
        "数据集",
        "dataset",
        "labels=",
        "label=",
        "classes=",
        "conda_env_name",
        "epochs",
        "batch",
        "imgsz",
        "data.yaml",
        "best.pt",
        "生图",
        "合成数据",
        "划分",
    )
    return any(marker in text for marker in positive_markers)


def _looks_like_plain_chat(user_text: str) -> bool:
    text = re.sub(r"\s+", " ", (user_text or "").strip().lower())
    if not text:
        return False
    greetings = {
        "hi",
        "hello",
        "hey",
        "你好",
        "您好",
        "嗨",
        "在吗",
        "在么",
        "hello!",
        "hi!",
    }
    return text in greetings


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM did not return a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON was not an object")
    return payload


def _extract_yolo_training_request_spec(
    user_text: str,
    *,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
) -> dict[str, Any]:
    if not (user_text or "").strip():
        return {}
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_request_spec"})
    if not llm.configured:
        recorder.emit("llm.completed", {"purpose": "workflow_request_spec", "used_fallback": True, "reason": "not_configured"})
        return {}
    system_prompt = (
        "你是 YOLO 训练工作流的参数抽取器。请从用户中文或英文消息中抽取结构化规格。"
        "只返回 JSON 对象，不要解释。字段："
        "task_description 字符串，描述要训练的检测任务；"
        "use_synthetic_generation 布尔值或 null，用户要求合成/生图/把 image2 合成到 image1 时为 true，明确不合成时为 false；"
        "generation_prompt 字符串，合成提示词原文；"
        "labels 字符串数组，标注类别；"
        "training 对象，字段可含 task, model, epochs, imgsz, batch, device, workers, patience；"
        "runtime 对象，字段可含 conda_env_name, enforce_conda_env；"
        "split 对象，字段可含 train, val, test。"
        "没有出现的字段返回空字符串、空数组、空对象或 null。不要编造默认训练参数。"
    )
    try:
        raw = llm.complete_sync(system_prompt, [Message(role="user", content=user_text)])
        payload = _parse_json_object(raw)
        spec = payload if isinstance(payload, dict) else {}
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_request_spec",
                "used_fallback": False,
                "extracted_keys": sorted(str(key) for key in spec.keys()),
            },
        )
        return spec
    except Exception as exc:
        recorder.emit("llm.completed", {"purpose": "workflow_request_spec", "used_fallback": True, "error": str(exc)[:1000]})
        return {}


def _generate_model_managed_yolo_training_intent_spec(
    *,
    user_text: str,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
) -> dict[str, Any]:
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_model_managed_intent_spec"})
    if not llm.configured:
        recorder.emit("llm.completed", {"purpose": "workflow_model_managed_intent_spec", "used_fallback": True, "reason": "not_configured"})
        return {}
    system_prompt = (
        "你是 YOLO 训练工作流的前置意图规划器。"
        "此阶段还没有分析数据集，因此禁止输出训练参数、batch、epochs、imgsz、split 等依赖数据集的字段。"
        "你只能根据用户业务目标提取任务意图、候选类别和合成提示词。"
        "必须只返回 JSON 对象，不要解释。JSON 字段包含："
        "task_description 字符串；"
        "use_synthetic_generation 布尔值，必须为 true；"
        "generation_prompt 字符串；"
        "labels 字符串数组；"
        "training 空对象；runtime 空对象；split 空对象。"
        "合成设定：用户会上传 dataset.zip、image1.zip、image2.zip；"
        "image1.zip 固定是场景/背景文件夹，image2.zip 固定是目标物文件夹；"
        "合成目的必须是把 image2 中目标自然合成到 image1 场景中，形成真实、可标注的训练图片。"
        "generation_prompt 必须包含上述 image1/image2 角色、自然融合、光照/尺度/遮挡一致、适合检测标注等要求。"
        "labels 必须围绕用户要训练的检测目标，不要固定套用示例类别；"
        "labels 必须是 YOLO 训练可直接使用的英文 ASCII 类名，只能使用小写英文、数字和下划线，"
        "禁止输出中文、空格或自然语言短语；例如人脸检测输出 face，不要输出 人脸；"
        "严禁返回 object、target、thing、foreground、目标、物体、对象 等泛化类别；"
        "必须从用户业务目标里解析具体对象作为 labels，例如：车辆检测输出 car，瓶子检测输出 bottle，钢材检测输出 steel；"
        "只有抽烟检测通常应包含 person 和 cigarette；车辆检测应输出车辆相关类别。"
    )
    messages = [
        Message(
            role="user",
            content=(
                f"用户业务目标：{user_text}\n"
                "请输出完整 YOLO 训练托管规格。"
            ),
        )
    ]
    try:
        raw = llm.complete_sync(system_prompt, messages)
        payload = _parse_json_object(raw)
        spec = payload if isinstance(payload, dict) else {}
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_model_managed_intent_spec",
                "used_fallback": False,
                "generated_keys": sorted(str(key) for key in spec.keys()),
            },
        )
        return spec
    except Exception as exc:
        recorder.emit("llm.completed", {"purpose": "workflow_model_managed_intent_spec", "used_fallback": True, "error": str(exc)[:1000]})
        return {}


def _generate_dataset_aware_yolo_training_request_spec(
    *,
    base_spec: dict[str, Any],
    dataset_facts: dict[str, Any],
    user_text: str,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
) -> dict[str, Any]:
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_dataset_aware_training_spec"})
    if not llm.configured:
        recorder.emit("llm.completed", {"purpose": "workflow_dataset_aware_training_spec", "used_fallback": True, "reason": "not_configured"})
        return base_spec
    system_prompt = (
        "你是资深计算机视觉算法工程师。现在必须基于 dataset_facts 分析结果，而不是只基于用户一句话，"
        "补全 YOLO 检测训练参数。只返回 JSON 对象，不要解释。"
        "必须包含字段：task_description, use_synthetic_generation, generation_prompt, labels, training, runtime, split。"
        "training 必须含 task, model, epochs, imgsz, batch, device, workers, patience。"
        "runtime 必须含 enforce_conda_env=false；不要指定 conda_env_name，或置为空字符串。"
        "split 必须含 train, val, test，三项相加约等于 1。"
        "决策规则："
        "根据 image_count、format、label_count、category_counts、bbox_size_summary、image_size_summary 判断训练强度；"
        "小目标多时提高 imgsz；图片少或类别不均衡时增加 epochs/patience 并启用合成；"
        "val/test 必须优先使用真实图，合成图只能进入 train；"
        "如果数据规模很小，test 比例要保守，避免每类在验证集缺失；"
        "labels 必须是英文 ASCII 类名，使用小写英文、数字、下划线，禁止中文和泛化 object/target。"
    )
    messages = [
        Message(
            role="user",
            content=json.dumps(
                {
                    "user_text": user_text,
                    "base_spec": base_spec,
                    "dataset_facts": dataset_facts,
                    "requirements": [
                        "先解释性地利用 dataset_facts 决定训练参数，但输出只要 JSON。",
                        "不要覆盖 base_spec 中合理的 task_description、labels、generation_prompt，除非 dataset_facts 证明需要修正。",
                        "训练必须使用当前运行环境。",
                    ],
                },
                ensure_ascii=False,
                default=str,
            ),
        )
    ]
    try:
        raw = llm.complete_sync(system_prompt, messages)
        payload = _parse_json_object(raw)
        generated = payload if isinstance(payload, dict) else {}
        merged = _merge_request_specs(base_spec, generated)
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_dataset_aware_training_spec",
                "used_fallback": False,
                "generated_keys": sorted(str(key) for key in generated.keys()),
                "dataset_image_count": dataset_facts.get("image_count"),
                "dataset_format": dataset_facts.get("format"),
            },
        )
        return merged
    except Exception as exc:
        recorder.emit("llm.completed", {"purpose": "workflow_dataset_aware_training_spec", "used_fallback": True, "error": str(exc)[:1000]})
        return base_spec


def _fallback_model_managed_yolo_training_request_spec(
    spec: dict[str, Any],
    user_text: str,
    dataset_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the upload contract deterministic when model-managed planning fails."""
    if _request_spec_complete(spec):
        return spec
    labels = _spec_string_list(spec, "labels") or _infer_labels_from_training_intent(user_text) or ["object"]
    labels = _normalize_detection_labels(labels) or ["object"]
    label_text = ", ".join(labels)
    fallback_training = _fallback_training_from_dataset_facts(dataset_facts or {})
    fallback = {
        "task_description": _spec_string(spec, "task_description") or f"{label_text} detection",
        "use_synthetic_generation": True,
        "generation_prompt": _spec_string(spec, "generation_prompt")
        or (
            f"Use image1.zip as background/scene images and image2.zip as foreground target images for {label_text}. "
            "Naturally composite the targets from image2 into image1 scenes with consistent lighting, scale, perspective, "
            "occlusion, and realistic camera appearance, producing images suitable for object-detection annotation."
        ),
        "labels": labels,
        "training": fallback_training["training"],
        "runtime": {"conda_env_name": "", "enforce_conda_env": False},
        "split": fallback_training["split"],
    }
    return _merge_request_specs(fallback, spec)


def _complete_yolo_training_request_spec(
    spec: dict[str, Any],
    *,
    user_text: str,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
) -> dict[str, Any]:
    if _request_spec_complete(spec):
        return spec
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_request_spec_completion"})
    if not llm.configured:
        recorder.emit("llm.completed", {"purpose": "workflow_request_spec_completion", "used_fallback": True, "reason": "not_configured"})
        return spec
    system_prompt = (
        "你是资深计算机视觉算法工程师，负责为 YOLO 目标检测训练工作流补全缺失参数。"
        "用户可能只给一句业务目标，你需要给出合理、可执行、保守的默认规格。"
        "只返回 JSON 对象，不要解释。字段："
        "task_description 字符串；use_synthetic_generation 布尔值；generation_prompt 字符串；"
        "labels 字符串数组；training 对象，含 task, model, epochs, imgsz, batch, device, workers, patience；"
        "runtime 对象，含 enforce_conda_env；不要输出 conda_env_name，或将 conda_env_name 置为空字符串；"
        "split 对象，含 train, val, test。"
        "原则：不要覆盖 existing_spec 里已经有的非空值；labels 必须跟随用户目标变化，不要固定套用示例类别；"
        "labels 必须是 YOLO 训练可直接使用的英文 ASCII 类名，只能使用小写英文、数字和下划线，"
        "禁止输出中文、空格或自然语言短语；例如人脸检测输出 face，不要输出 人脸；"
        "严禁返回 object、target、thing、foreground、目标、物体、对象 等泛化类别；"
        "必须从用户业务目标里解析具体对象作为 labels，例如：车辆检测输出 car，瓶子检测输出 bottle，钢材检测输出 steel；"
        "只有任务是抽烟检测时 labels 才优先包含 person 和 cigarette；车辆检测应输出车辆相关类别；"
        "合成提示词要适合 image2 目标自然合成到 image1 场景，并强调真实监控画面、可标注；"
        "训练参数必须由你根据任务目标、YOLO 训练常识和快速验证需求自行选择，不要照抄用户未提供的固定模板；"
        "训练必须使用当前运行环境，不要推理或指定 Conda 环境；runtime.enforce_conda_env 必须为 false；"
        "split 比例、模型大小、epochs、batch、patience 也必须由你合理选择。"
        "如果用户提供了 image1/image2 或上下文暗示需要合成数据，use_synthetic_generation 必须为 true。"
    )
    messages = [
        Message(
            role="user",
            content=(
                "请补全这个 YOLO 训练请求。\n"
                f"用户原文：{user_text}\n"
                f"existing_spec：{json.dumps(spec, ensure_ascii=False, default=str)}"
            ),
        )
    ]
    try:
        raw = llm.complete_sync(system_prompt, messages)
        payload = _parse_json_object(raw)
        completed = payload if isinstance(payload, dict) else {}
        merged = _merge_request_specs(spec, completed)
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_request_spec_completion",
                "used_fallback": False,
                "completed_keys": sorted(str(key) for key in completed.keys()),
            },
        )
        return merged
    except Exception as exc:
        recorder.emit("llm.completed", {"purpose": "workflow_request_spec_completion", "used_fallback": True, "error": str(exc)[:1000]})
        return spec


def _request_spec_complete(spec: dict[str, Any]) -> bool:
    return bool(
        _spec_string(spec, "generation_prompt")
        and _spec_string_list(spec, "labels")
        and _spec_training_config(spec)
        and _spec_optional_bool(spec, "use_synthetic_generation") is not None
    )


def _merge_request_specs(base: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("task_description", "generation_prompt"):
        if not _spec_string(merged, key) and _spec_string(generated, key):
            merged[key] = _spec_string(generated, key)
    if _spec_optional_bool(merged, "use_synthetic_generation") is None:
        generated_bool = _spec_optional_bool(generated, "use_synthetic_generation")
        if generated_bool is not None:
            merged["use_synthetic_generation"] = generated_bool
    if not _spec_string_list(merged, "labels"):
        labels = _spec_string_list(generated, "labels")
        if labels:
            merged["labels"] = labels
    if not _spec_training_config(merged):
        for key in ("training", "runtime", "split"):
            value = generated.get(key)
            if isinstance(value, dict):
                merged[key] = dict(value)
    return merged


def _emit_model_generated_spec(
    recorder: EventRecorder,
    spec: dict[str, Any],
    *,
    dataset_facts: dict[str, Any] | None = None,
) -> None:
    payload = _model_generated_spec_payload(spec)
    if dataset_facts:
        payload["dataset_facts"] = dataset_facts
    recorder.emit("workflow.generated_spec", payload)
    recorder.emit(
        "agent.log",
        {
            "message": (
                "模型思考生成参数："
                f"合成提示词={payload['generation_prompt']}; "
                f"labels={payload['labels']}; "
                f"training={payload['training']}; "
                f"runtime={payload['runtime']}; "
                f"split={payload['split']}"
            )
        },
    )


def _write_model_generated_spec_logs(
    workflow_output_root: Path,
    spec: dict[str, Any],
    *,
    dataset_facts: dict[str, Any] | None = None,
) -> None:
    payload = _model_generated_spec_payload(spec)
    if dataset_facts:
        payload["dataset_facts"] = dataset_facts
    log_dir = workflow_output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "model-generated-spec.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "模型思考生成的 YOLO 训练参数",
        "",
        "dataset_facts:",
        json.dumps(dataset_facts or {}, ensure_ascii=False, indent=2, default=str),
        "",
        f"task_description: {payload.get('task_description') or ''}",
        f"use_synthetic_generation: {payload.get('use_synthetic_generation')}",
        "",
        "generation_prompt:",
        str(payload.get("generation_prompt") or ""),
        "",
        "labels:",
        ", ".join(str(item) for item in payload.get("labels", [])) if isinstance(payload.get("labels"), list) else str(payload.get("labels") or ""),
        "",
        "training:",
        json.dumps(payload.get("training") or {}, ensure_ascii=False, indent=2),
        "",
        "runtime:",
        json.dumps(payload.get("runtime") or {}, ensure_ascii=False, indent=2),
        "",
        "split:",
        json.dumps(payload.get("split") or {}, ensure_ascii=False, indent=2),
        "",
    ]
    (log_dir / "model-generated-spec.txt").write_text("\n".join(lines), encoding="utf-8")


def _model_generated_spec_payload(spec: dict[str, Any]) -> dict[str, Any]:
    training_cfg = _spec_training_config(spec)
    _force_current_runtime(training_cfg)
    return {
        "title": "模型思考生成的 YOLO 训练参数",
        "task_description": _spec_string(spec, "task_description"),
        "use_synthetic_generation": _spec_optional_bool(spec, "use_synthetic_generation"),
        "generation_prompt": _spec_string(spec, "generation_prompt"),
        "labels": _spec_string_list(spec, "labels"),
        "training": training_cfg.get("training", {}),
        "runtime": training_cfg.get("runtime", {}),
        "split": training_cfg.get("split", {}),
    }


def _spec_string(spec: dict[str, Any], key: str) -> str:
    value = spec.get(key)
    return str(value).strip() if isinstance(value, str) else ""


def _spec_optional_bool(spec: dict[str, Any], key: str) -> bool | None:
    value = spec.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on", "需要", "是", "启用"}:
            return True
        if lowered in {"false", "0", "no", "n", "off", "不需要", "否", "禁用"}:
            return False
    return None


def _bool_from_any(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on", "需要", "是", "启用"}:
            return True
        if lowered in {"false", "0", "no", "n", "off", "不需要", "否", "禁用"}:
            return False
    return None


def _int_from_any(value: Any, default: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _workflow_skill_parameters(runtime_options: RuntimeOptions) -> dict[str, Any]:
    params = getattr(runtime_options, "skill_parameters", None)
    if not isinstance(params, dict):
        return {}
    workflow_params = params.get(WORKFLOW_NAME)
    if isinstance(workflow_params, dict):
        return workflow_params
    app_params = params.get("algorithm-engineer-full-cycle-test")
    return app_params if isinstance(app_params, dict) else {}


def _max_synthetic_images(runtime_options: RuntimeOptions) -> int:
    direct = getattr(runtime_options, "max_synthetic_images", None)
    if direct is not None:
        return _int_from_any(direct, DEFAULT_MAX_SYNTHETIC_IMAGES)
    params = _workflow_skill_parameters(runtime_options)
    return _int_from_any(
        params.get("maxSyntheticImages") or params.get("max_synthetic_images") or params.get("max_synthetic"),
        DEFAULT_MAX_SYNTHETIC_IMAGES,
    )


def _epochs_button(runtime_options: RuntimeOptions) -> bool:
    params = _workflow_skill_parameters(runtime_options)
    value = _bool_from_any(params.get("button_epochs"))
    return button_epochs if value is None else value


def _apply_epochs_policy(training_cfg: dict[str, Any], runtime_options: RuntimeOptions) -> None:
    if _epochs_button(runtime_options):
        return
    training = training_cfg.get("training")
    if isinstance(training, dict):
        training["epochs"] = FIXED_TRAINING_EPOCHS


def _auto_generate_missing_spec(runtime_options: RuntimeOptions) -> bool:
    params = _workflow_skill_parameters(runtime_options)
    value = _bool_from_any(params.get("auto_generate_missing_spec"))
    return AUTO_GENERATE_MISSING_SPEC if value is None else value


def _current_runtime_config(runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(runtime or {})
    payload["conda_env_name"] = ""
    payload["enforce_conda_env"] = False
    return payload


def _force_current_runtime(training_cfg: dict[str, Any]) -> None:
    if not isinstance(training_cfg, dict):
        return
    runtime = training_cfg.get("runtime") if isinstance(training_cfg.get("runtime"), dict) else {}
    training_cfg["runtime"] = _current_runtime_config(runtime)


def _spec_string_list(spec: dict[str, Any], key: str) -> list[str]:
    value = spec.get(key)
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        label = str(item).strip().strip("\"'`，,;；。")
        if not label:
            continue
        lower = label.lower()
        if lower in seen:
            continue
        seen.add(lower)
        labels.append(label)
    return labels


_GENERIC_DETECTION_LABELS = {
    "object",
    "objects",
    "target",
    "targets",
    "thing",
    "things",
    "foreground",
    "foreground_object",
    "item",
    "items",
    "class",
    "classes",
    "unknown",
    "目标",
    "物体",
    "对象",
    "前景",
    "主体",
    "类别",
    "待检测目标",
}


_LABEL_ALIAS_TO_CANONICAL: dict[str, tuple[str, ...]] = {
    "smoking": ("person", "cigarette"),
    "抽烟": ("person", "cigarette"),
    "吸烟": ("person", "cigarette"),
    "cigarette": ("cigarette",),
    "香烟": ("cigarette",),
    "烟支": ("cigarette",),
    "烟头": ("cigarette",),
    "license_plate": ("license_plate",),
    "license plate": ("license_plate",),
    "车牌": ("license_plate",),
    "牌照": ("license_plate",),
    "forklift": ("forklift",),
    "叉车": ("forklift",),
    "electric_bicycle": ("electric_bicycle",),
    "e_bike": ("electric_bicycle",),
    "ebike": ("electric_bicycle",),
    "电动车": ("electric_bicycle",),
    "电瓶车": ("electric_bicycle",),
    "motorcycle": ("motorcycle",),
    "摩托车": ("motorcycle",),
    "摩托": ("motorcycle",),
    "bicycle": ("bicycle",),
    "bike": ("bicycle",),
    "自行车": ("bicycle",),
    "单车": ("bicycle",),
    "truck": ("truck",),
    "货车": ("truck",),
    "卡车": ("truck",),
    "bus": ("bus",),
    "公交车": ("bus",),
    "公交": ("bus",),
    "巴士": ("bus",),
    "客车": ("bus",),
    "car": ("car",),
    "vehicle": ("car",),
    "vehicles": ("car",),
    "车辆": ("car",),
    "汽车": ("car",),
    "小汽车": ("car",),
    "轿车": ("car",),
    "机动车": ("car",),
    "车": ("car",),
    "person": ("person",),
    "people": ("person",),
    "pedestrian": ("person",),
    "行人": ("person",),
    "人员": ("person",),
    "人": ("person",),
    "bottle": ("bottle",),
    "瓶子": ("bottle",),
    "水瓶": ("bottle",),
    "矿泉水瓶": ("bottle",),
    "瓶": ("bottle",),
    "steel": ("steel",),
    "钢材": ("steel",),
    "钢筋": ("steel",),
    "钢板": ("steel",),
    "rebar": ("steel",),
    "hard_hat": ("hard_hat",),
    "helmet": ("hard_hat",),
    "安全帽": ("hard_hat",),
    "mask": ("mask",),
    "口罩": ("mask",),
    "fire": ("fire",),
    "火焰": ("fire",),
    "flame": ("fire",),
    "smoke": ("smoke",),
    "烟雾": ("smoke",),
    "face": ("face",),
    "human_face": ("face",),
    "人脸": ("face",),
    "脸": ("face",),
    "脸部": ("face",),
    "面部": ("face",),
    "人面部": ("face",),
    "head": ("head",),
    "头部": ("head",),
    "hand": ("hand",),
    "手": ("hand",),
    "glove": ("glove",),
    "手套": ("glove",),
    "phone": ("phone",),
    "mobile_phone": ("phone",),
    "手机": ("phone",),
    "cup": ("cup",),
    "杯子": ("cup",),
    "box": ("box",),
    "纸箱": ("box",),
    "箱子": ("box",),
    "package": ("package",),
    "包裹": ("package",),
    "parcel": ("package",),
    "bag": ("bag",),
    "袋子": ("bag",),
    "knife": ("knife",),
    "刀": ("knife",),
    "helmet": ("hard_hat",),
    "vest": ("safety_vest",),
    "safety_vest": ("safety_vest",),
    "反光衣": ("safety_vest",),
    "安全背心": ("safety_vest",),
}


_INTENT_LABEL_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("抽烟", "吸烟", "smoking"), ("person", "cigarette")),
    (("烟头", "香烟", "烟支", "cigarette"), ("cigarette",)),
    (("车牌", "牌照", "license plate", "license_plate"), ("license_plate",)),
    (("叉车", "forklift"), ("forklift",)),
    (("电动车", "电瓶车", "electric bicycle", "e-bike", "ebike"), ("electric_bicycle",)),
    (("摩托车", "摩托", "motorcycle"), ("motorcycle",)),
    (("自行车", "单车", "bicycle", "bike"), ("bicycle",)),
    (("货车", "卡车", "truck"), ("truck",)),
    (("公交车", "公交", "巴士", "客车", "bus"), ("bus",)),
    (("车辆", "汽车", "小汽车", "轿车", "机动车", "vehicle", "car"), ("car",)),
    (("瓶子", "水瓶", "矿泉水瓶", "bottle"), ("bottle",)),
    (("钢材", "钢筋", "钢板", "steel", "rebar"), ("steel",)),
    (("安全帽", "helmet", "hard hat", "hard_hat"), ("hard_hat",)),
    (("口罩", "mask"), ("mask",)),
    (("火焰", "fire", "flame"), ("fire",)),
    (("人脸", "脸部", "面部", "human face", "face"), ("face",)),
    (("头部", "head"), ("head",)),
    (("手套", "glove"), ("glove",)),
    (("手机", "phone", "mobile phone"), ("phone",)),
    (("杯子", "cup"), ("cup",)),
    (("纸箱", "箱子", "box"), ("box",)),
    (("包裹", "package", "parcel"), ("package",)),
    (("袋子", "bag"), ("bag",)),
    (("刀具", "刀", "knife"), ("knife",)),
    (("反光衣", "安全背心", "safety vest", "safety_vest"), ("safety_vest",)),
    (("烟雾", "smoke"), ("smoke",)),
    (("行人", "人员", "person", "pedestrian"), ("person",)),
)


def _ensure_intent_labels(spec: dict[str, Any], user_text: str) -> dict[str, Any]:
    merged = dict(spec or {})
    explicit_labels = _normalize_detection_labels(_extract_annotation_labels(user_text))
    if explicit_labels:
        merged["labels"] = explicit_labels
        _align_spec_text_with_labels(merged, explicit_labels)
        return merged

    current_labels = _spec_string_list(merged, "labels")
    normalized_current = _normalize_detection_labels(current_labels)
    inferred_labels = _infer_labels_from_training_intent(user_text)
    if inferred_labels and (not normalized_current or _labels_are_generic(current_labels) or normalized_current != inferred_labels):
        merged["labels"] = inferred_labels
    elif normalized_current:
        merged["labels"] = normalized_current
    elif inferred_labels:
        merged["labels"] = inferred_labels
    _align_spec_text_with_labels(merged, _spec_string_list(merged, "labels"))
    return merged


def _infer_labels_from_training_intent(user_text: str) -> list[str]:
    text = (user_text or "").strip()
    if not text:
        return []
    lowered = text.lower()
    for aliases, labels in _INTENT_LABEL_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            return _dedupe_detection_labels(labels)
    for target in _extract_intent_target_terms(text):
        labels = _normalize_detection_labels([target])
        if labels:
            return labels
    return []


def _extract_intent_target_terms(text: str) -> list[str]:
    patterns = (
        r"YOLO\s*([^，,。；;\n]{1,32}?)(?:目标检测|检测|识别|分割)?模型",
        r"(?:训练|构建|开发|做|生成)(?:一个|一套|一种)?\s*([^，,。；;\n]{1,32}?)(?:YOLO|yolo)(?:目标检测|检测|识别|分割)?模型",
        r"([^，,。；;\n]{1,32}?)(?:目标检测|检测|识别|分割)模型",
    )
    terms: list[str] = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        term = _clean_intent_target_term(match.group(1))
        if term:
            terms.append(term)
    return _dedupe_detection_labels(terms)


def _clean_intent_target_term(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(?i)yolo\d*[a-z]*|cv|目标检测|检测|识别|分割|模型|算法|训练|帮我|请|一个|一套|一种|的|用于", "", text)
    return re.sub(r"[\s:：，,。；;]+", "", text).strip()


def _normalize_detection_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for raw in labels:
        label = str(raw or "").strip().strip("\"'`，,;；。")
        if not label or _is_generic_detection_label(label):
            continue
        canonical = _LABEL_ALIAS_TO_CANONICAL.get(_label_lookup_key(label))
        if canonical:
            normalized.extend(canonical)
            continue
        clean_label = _clean_detection_label(label)
        if clean_label and not _is_generic_detection_label(clean_label):
            normalized.append(clean_label)
    return _dedupe_detection_labels([label for label in normalized if _is_yolo_safe_label(label)])


def _labels_are_generic(labels: list[str]) -> bool:
    return bool(labels) and all(_is_generic_detection_label(label) for label in labels)


def _is_generic_detection_label(label: str) -> bool:
    key = _label_lookup_key(label)
    raw = str(label or "").strip()
    return key in _GENERIC_DETECTION_LABELS or raw in _GENERIC_DETECTION_LABELS


def _label_lookup_key(label: str) -> str:
    text = str(label or "").strip().strip("\"'`，,;；。").lower()
    return re.sub(r"[\s\-]+", "_", text).strip("_")


def _clean_detection_label(label: str) -> str:
    text = str(label or "").strip().strip("\"'`，,;；。")
    if re.search(r"[A-Za-z0-9]", text):
        return re.sub(r"[^0-9A-Za-z_]+", "_", text.lower()).strip("_")
    return ""


def _is_yolo_safe_label(label: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*", str(label or "").strip()))


def _dedupe_detection_labels(labels: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in labels:
        label = str(item or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(label)
    return result


def _align_spec_text_with_labels(spec: dict[str, Any], labels: list[str]) -> None:
    if not labels:
        return
    label_text = ", ".join(labels)
    for key in ("task_description", "generation_prompt"):
        value = _spec_string(spec, key)
        if not value:
            continue
        if re.search(r"\bdefect\b|缺陷|异物", value, flags=re.IGNORECASE):
            spec[key] = re.sub(r"\bdefect\b", label_text, value, flags=re.IGNORECASE)
            spec[key] = str(spec[key]).replace("缺陷/异物", label_text).replace("缺陷", label_text).replace("异物", label_text)


def _spec_training_config(spec: dict[str, Any]) -> dict[str, Any]:
    raw_training = spec.get("training") if isinstance(spec.get("training"), dict) else {}
    raw_runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    raw_split = spec.get("split") if isinstance(spec.get("split"), dict) else {}
    if not (raw_training or raw_runtime or raw_split):
        return {}
    cfg = _extract_training_config("")
    for key in ("task", "model", "device"):
        value = raw_training.get(key)
        if value not in (None, ""):
            cfg["training"][key] = str(value).strip()
    cfg["training"]["task"] = _normalize_yolo_task(cfg["training"].get("task"))
    for key in ("epochs", "imgsz", "batch", "workers", "patience"):
        value = _coerce_int(raw_training.get(key))
        if value is not None:
            cfg["training"][key] = value
    value = raw_runtime.get("conda_env_name")
    if value not in (None, ""):
        cfg["runtime"]["conda_env_name"] = str(value).strip()
    value = _coerce_bool(raw_runtime.get("enforce_conda_env"))
    if value is not None:
        cfg["runtime"]["enforce_conda_env"] = value
    for key in ("train", "val", "test"):
        value = _coerce_float(raw_split.get(key))
        if value is not None:
            cfg["split"][key] = value
    cfg["split"] = _normalize_training_split(cfg["split"])
    return cfg if _training_config_has_required_fields(raw_training, raw_runtime) else {}


def _normalize_training_split(split: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in ("train", "val", "test"):
        value = _coerce_float(split.get(key))
        if value is None or value < 0:
            return dict(DEFAULT_TRAINING_SPLIT)
        values[key] = value

    total = values["train"] + values["val"] + values["test"]
    if total <= 0:
        return dict(DEFAULT_TRAINING_SPLIT)
    values = {key: value / total for key, value in values.items()}

    if values["test"] < MIN_TEST_SPLIT:
        deficit = MIN_TEST_SPLIT - values["test"]
        values["test"] = MIN_TEST_SPLIT
        values["train"] -= deficit

    if values["train"] <= 0 or values["val"] <= 0:
        return dict(DEFAULT_TRAINING_SPLIT)

    normalized_total = values["train"] + values["val"] + values["test"]
    values = {key: round(value / normalized_total, 6) for key, value in values.items()}
    values["train"] = round(1.0 - values["val"] - values["test"], 6)
    return values


def _training_config_has_required_fields(raw_training: dict[str, Any], raw_runtime: dict[str, Any]) -> bool:
    del raw_runtime
    required_training = ("model", "epochs", "imgsz", "batch", "device", "workers", "patience")
    return all(raw_training.get(key) not in (None, "") for key in required_training)


def _normalize_yolo_task(value: Any) -> str:
    text = str(value or "detect").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"segment", "seg", "segmentation", "instance_segmentation"}:
        return "segment"
    return "detect"


def _coerce_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _training_failed_reply(
    *,
    summary: dict[str, Any],
    fallback_reply: str,
    best_pt: str,
    data_preparation_summary: dict[str, Any],
) -> str:
    parsed = _parse_json_object(fallback_reply)
    stderr_tail = str(parsed.get("stderr_tail") or "").strip()
    stdout_tail = str(parsed.get("stdout_tail") or "").strip()
    error_tail = stderr_tail or stdout_tail
    returncode = parsed.get("returncode")
    prep = data_preparation_summary or summary
    synthetic = _synthetic_generation_facts(prep)

    lines = ["YOLO 训练流程执行失败。", ""]
    if prep:
        lines.extend(
            [
                "数据处理阶段已完成：",
                f"- prepared_dataset: `{prep.get('prepared_dataset') or '-'}`",
                f"- dataset.yaml: `{prep.get('dataset_yaml') or summary.get('dataset_yaml') or '-'}`",
            ]
        )
        synthetic_line = _synthetic_generation_summary_line(synthetic)
        if synthetic_line:
            lines.append(f"- 合成数据：{synthetic_line}")
        split_counts = prep.get("split_counts")
        if isinstance(split_counts, dict):
            lines.append(
                "- 数据划分："
                f"train={split_counts.get('train', '-')} "
                f"val={split_counts.get('val', '-')} "
                f"test={split_counts.get('test', '-')}"
            )
        lines.append("")

    lines.extend(
        [
            "训练阶段失败：",
            f"- returncode: `{returncode if returncode not in (None, '') else '-'}`",
            f"- best.pt: `{best_pt or '未生成'}`",
        ]
    )
    if error_tail:
        lines.extend(["- 主要错误：", "```text", error_tail[-2000:], "```"])
    return "\n".join(lines).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text) if text.strip().startswith("{") else {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _synthetic_generation_facts(summary: dict[str, Any]) -> dict[str, Any]:
    source_counts = summary.get("source_counts")
    synthetic_count = 0
    if isinstance(source_counts, dict):
        for split_info in source_counts.values():
            if isinstance(split_info, dict):
                try:
                    synthetic_count += int(split_info.get("synthetic", 0) or 0)
                except (TypeError, ValueError):
                    continue
    return {
        "status": summary.get("synthetic_generation_status"),
        "fallback": summary.get("synthetic_generation_fallback"),
        "error": summary.get("synthetic_generation_error"),
        "primary_error": summary.get("synthetic_generation_primary_error"),
        "source_counts": source_counts,
        "synthetic_count": synthetic_count,
    }


def _synthetic_generation_summary_line(facts: dict[str, Any]) -> str:
    status = str(facts.get("status") or "").strip()
    fallback = str(facts.get("fallback") or "").strip()
    synthetic_count = facts.get("synthetic_count")
    primary_error = str(facts.get("primary_error") or "").strip()
    error = str(facts.get("error") or "").strip()
    if status == "merged":
        if fallback:
            text = f"主合成接口不可用后已切换到 `{fallback}`，并成功合并 {synthetic_count or 0} 张合成图"
            if primary_error:
                text += f"（主接口错误：{primary_error[:180]}）"
            return text
        return f"已成功合并 {synthetic_count or 0} 张合成图"
    if status:
        return f"状态 `{status}`" + (f"，错误：{error[:240]}" if error else "")
    return ""


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
            "metrics": _extract_metric_summary(summary),
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
    metrics = facts.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        metrics = _extract_metric_summary(summary)
    if isinstance(metrics, dict) and metrics:
        lines.extend(["", "关键评估指标："])
        for key in ("precision", "recall", "mAP50", "mAP50_95", "fitness"):
            if key in metrics:
                lines.append(f"- {key}: {_format_metric(metrics.get(key))}")
    eval_block = str(facts.get("eval_block") or "").strip()
    if eval_block:
        lines.extend(["", "评估指标：", "```text", eval_block, "```"])
    eval_error = str(facts.get("eval_error") or "").strip()
    if eval_error:
        lines.extend(["", f"评估提示：{eval_error}"])
    return "\n".join(lines).strip()


def _extract_evaluation_facts(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": _extract_metric_summary(summary),
        "results_dict": _extract_results_dict(summary),
        "class_names": summary.get("class_names"),
        "eval_error": summary.get("eval_error"),
        "eval_results_text": str(summary.get("eval_results") or "")[:2000],
    }


def _extract_metric_summary(summary: dict[str, Any]) -> dict[str, Any]:
    results = _extract_results_dict(summary)
    mapping = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50_95": "metrics/mAP50-95(B)",
        "fitness": "fitness",
    }
    metrics: dict[str, Any] = {}
    for out_key, source_key in mapping.items():
        if source_key in results:
            metrics[out_key] = results[source_key]
    return metrics


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
    try:
        import ast

        parsed = ast.literal_eval(match.group(1))
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _format_metric(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}"


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
            return _clean_user_visible_text(message.content)
    return ""


def _clean_user_visible_text(text: str) -> str:
    cleaned = _strip_uploaded_files_context(text)
    cleaned = re.split(r"\n+\[Workbench selected capabilities\]\n", cleaned, maxsplit=1)[0]
    return cleaned.strip()


def _strip_uploaded_files_context(text: str) -> str:
    return re.split(r"\n\nUploaded files available to tools:\n", text or "", maxsplit=1)[0].strip()


def _training_objective_from_user_text(user_text: str) -> str:
    text = _clean_user_visible_text(user_text)
    if not text or _is_generic_continue_text(text):
        return ""
    if _looks_like_yolo_training_request(text):
        return text
    return ""


def _combine_spec_user_text(objective_text: str, user_text: str) -> str:
    current = _clean_user_visible_text(user_text)
    objective = _clean_user_visible_text(objective_text)
    if not objective:
        return current
    if not current or current == objective or _is_generic_continue_text(current):
        return objective
    return f"{objective}\n\n当前补充信息：{current}"


def _is_generic_continue_text(text: str) -> bool:
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", (text or "").strip().lower())
    return normalized in {
        "继续",
        "继续处理",
        "继续处理刚上传的文件",
        "继续处理刚上传文件",
        "开始",
        "开始训练",
        "继续训练",
        "下一步",
        "continue",
        "goon",
        "next",
    }


def _selected_skills(runtime_options: RuntimeOptions) -> list[str]:
    selected = getattr(runtime_options, "selected_skills", None) or []
    return [str(item).strip() for item in selected if str(item).strip()] or [*DEFAULT_SELECTED_SKILLS]


def _ensure_full_cycle_skills(selected_skills: list[str]) -> list[str]:
    merged = [str(item).strip() for item in selected_skills if str(item).strip()]
    seen = set(merged)
    for skill_name in DEFAULT_SELECTED_SKILLS:
        if skill_name not in seen:
            merged.append(skill_name)
            seen.add(skill_name)
    return merged


def _has_runtime_selected_skills(runtime_options: RuntimeOptions) -> bool:
    selected = getattr(runtime_options, "selected_skills", None) or []
    return any(str(item).strip() for item in selected)


def _skill_enabled(selected_skills: list[str], skill_name: str) -> bool:
    return skill_name in set(selected_skills)


def _capability_enabled(selected_skills: list[str], capability: str) -> bool:
    selected = {str(item).strip() for item in selected_skills if str(item).strip()}
    if not selected:
        return False
    for skill_name in selected:
        if capability in _workflow_capabilities_for_skill(skill_name):
            return True
    return False


@lru_cache(maxsize=256)
def _workflow_capabilities_for_skill(skill_name: str) -> frozenset[str]:
    capabilities: set[str] = set()
    for path in _skill_metadata_paths(skill_name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        raw = payload.get("workflow_capabilities")
        if isinstance(raw, list):
            capabilities.update(str(item).strip() for item in raw if str(item).strip())
        raw_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        raw_meta_caps = raw_meta.get("workflow_capabilities")
        if isinstance(raw_meta_caps, list):
            capabilities.update(str(item).strip() for item in raw_meta_caps if str(item).strip())
    if not capabilities:
        capabilities.update(_fallback_workflow_capabilities(skill_name))
    return frozenset(capabilities)


def _skill_metadata_paths(skill_name: str) -> list[Path]:
    workflow_dir = Path(__file__).resolve().parent
    project_root = workflow_dir.parents[2]
    return [
        project_root / "plugins" / "skills" / skill_name / "plugin.json",
        project_root / "config" / "skills" / f"{skill_name}.json",
    ]


def _fallback_workflow_capabilities(skill_name: str) -> set[str]:
    fallback = {
        "data-auto-annotation": {"data_preparation", "auto_annotation"},
        "image-dataset-generation": {"image_generation"},
        "image-dataset-produce": {"image_generation"},
        "gpu-training-orchestrator": {"training"},
    }
    return set(fallback.get(skill_name, set()))


def _has_explicit_training_config(user_text: str) -> bool:
    text = (user_text or "").lower()
    required = ("conda_env_name", "model", "epochs", "imgsz", "batch", "device", "workers", "patience")
    return all(k in text for k in required)


def _extract_training_config(user_text: str) -> dict[str, Any]:
    return {
        "split": {
            "train": _float_param(user_text, "dataset.split.train", 0.7),
            "val": _float_param(user_text, "dataset.split.val", 0.2),
            "test": _float_param(user_text, "dataset.split.test", 0.1),
        },
        "training": {
            "task": _normalize_yolo_task(_string_param(user_text, "training_task", "detect")),
            "model": _string_param(user_text, "model", "yolo11n.pt"),
            "epochs": _int_param(user_text, "epochs", 50),
            "imgsz": _int_param(user_text, "imgsz", 640),
            "batch": _int_param(user_text, "batch", 16),
            "device": _string_param(user_text, "device", "0"),
            "workers": _int_param(user_text, "workers", 4),
            "patience": _int_param(user_text, "patience", 8),
        },
        "runtime": {
            "conda_env_name": _string_param(user_text, "conda_env_name", "yolo"),
            "enforce_conda_env": _bool_param(user_text, "enforce_conda_env", False),
        },
    }


def _extract_detection_task_description(user_text: str, prompt_text: str, labels: list[str]) -> str:
    text = user_text or ""
    for key in ("task_description", "detection_task", "detect_task", "任务描述", "检测任务", "任务"):
        value = _string_param(text, key, "")
        if value and value.lower() not in {"detect", "segment", "train"}:
            return value

    match = re.search(
        r"训练(?:一个|一個)?\s*(.{1,80}?)(?:YOLO|yolo).{0,40}?(?:检测|detect)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group(1).strip().strip(":：,，。；; ")
        if value:
            return value

    if labels:
        return f"{', '.join(labels)} detection"

    cleaned_prompt = _strip_config_lines_from_prompt(prompt_text).strip()
    if cleaned_prompt:
        return cleaned_prompt[:120]
    return "generic object detection"


def _planner_llm_config(agent_config: Any, runtime_options: RuntimeOptions) -> dict[str, Any]:
    model_config = getattr(agent_config, "model", None)
    base_url = getattr(runtime_options, "base_url", None) or getattr(model_config, "base_url", "")
    api_key = getattr(runtime_options, "api_key", None) or getattr(model_config, "api_key", "")
    model = (
        getattr(runtime_options, "model_name", None)
        or getattr(model_config, "model", "")
        or getattr(model_config, "default_model", "")
    )
    temperature = getattr(runtime_options, "temperature", None)
    if temperature is None:
        temperature = getattr(model_config, "temperature", 0.2)
    max_tokens = getattr(runtime_options, "max_tokens", None)
    if max_tokens is None:
        max_tokens = min(int(getattr(model_config, "max_tokens", 2048) or 2048), 2048)
    timeout = getattr(runtime_options, "request_timeout_seconds", None)
    if timeout is None:
        timeout = getattr(model_config, "request_timeout_seconds", 120)
    return {
        "base_url": str(base_url or "").rstrip("/"),
        "api_key": str(api_key or ""),
        "model": str(model or ""),
        "temperature": float(temperature if temperature is not None else 0.2),
        "max_tokens": int(max_tokens or 2048),
        "timeout": int(timeout or 120),
    }


def _string_param(text: str, key: str, default: str) -> str:
    match = re.search(rf"(?<![\w.]){re.escape(key)}\s*[:=]\s*([^\s,，;；\r\n]+)", text or "", flags=re.IGNORECASE)
    return match.group(1).strip().strip("\"'`") if match else default


def _int_param(text: str, key: str, default: int) -> int:
    value = _string_param(text, key, "")
    try:
        return int(value)
    except ValueError:
        return default


def _float_param(text: str, key: str, default: float) -> float:
    value = _string_param(text, key, "")
    try:
        return float(value)
    except ValueError:
        return default


def _bool_param(text: str, key: str, default: bool) -> bool:
    value = _string_param(text, key, "")
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _extract_annotation_labels(user_text: str) -> list[str]:
    text = (user_text or "").strip()
    if not text:
        return []
    raw = _extract_labeled_section(
        text,
        ("labels", "label", "lables", "lable", "classes", "class", "标注类别", "类别", "标签", "标注标签"),
        ("训练参数", "conda_env_name", "model", "epochs", "imgsz", "batch", "device", "workers", "patience", "dataset.split."),
    )
    if not raw:
        return []
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
    text = (user_text or "").strip()
    if not text:
        return ""
    raw = _extract_labeled_section(
        text,
        ("prompt", "提示词", "合成提示词", "生图提示词"),
        ("标注类别", "类别", "标签", "标注标签", "labels", "label", "classes", "class", "训练参数", "conda_env_name"),
    )
    if raw:
        return _strip_config_lines_from_prompt(raw)
    if not allow_free_text:
        return ""
    lowered = text.lower()
    generic_cmds = {"继续", "继续处理", "继续处理刚上传的文件", "开始", "继续训练", "下一步"}
    if text in generic_cmds or lowered in {"continue", "go on", "next"}:
        return ""
    if _looks_like_config_only_text(text):
        return ""
    return text if len(text) >= 12 else ""


def _extract_labeled_section(text: str, starts: tuple[str, ...], stops: tuple[str, ...]) -> str:
    start = -1
    for label in sorted(starts, key=len, reverse=True):
        candidate = _find_label_value_start(text, label)
        if candidate >= 0 and (start < 0 or candidate < start):
            start = candidate
    if start < 0:
        return ""
    end = len(text)
    for label in sorted(stops, key=len, reverse=True):
        candidate = _find_label_prefix_start(text, label, start)
        if candidate >= 0:
            end = min(end, candidate)
    return text[start:end].strip().strip(",，;；。 ")


def _find_label_value_start(text: str, label: str, search_from: int = 0) -> int:
    prefix = _find_label_prefix_start(text, label, search_from)
    if prefix < 0:
        return -1
    index = prefix + len(label)
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] in {"为", "是"}:
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
    if index < len(text) and text[index] in {":", "=", "："}:
        return index + 1
    return -1


def _find_label_prefix_start(text: str, label: str, search_from: int = 0) -> int:
    lower_text = text.lower()
    lower_label = label.lower()
    pos = lower_text.find(lower_label, search_from)
    while pos >= 0:
        index = pos + len(label)
        while index < len(text) and text[index].isspace():
            index += 1
        if index < len(text) and text[index] in {"为", "是"}:
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
        if index < len(text) and text[index] in {":", "=", "："}:
            return pos
        pos = lower_text.find(lower_label, pos + 1)
    return -1


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
        "--per-image-output-dir",
        str(target_dir.resolve()),
        "--per-image-base-dir",
        str(target_dir.resolve()),
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
    preferred = paths.outputs / WORKFLOW_OUTPUT_DIR / "training_run" / "run_summary.json"
    if preferred.is_file():
        candidates = [preferred]
    else:
        candidates = list(paths.outputs.rglob("run_summary.json"))
    if not candidates:
        return {}
    latest = sorted(candidates)[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    prep = _read_data_preparation_summary(paths)
    if prep:
        payload.setdefault("num_images", prep.get("num_images"))
        payload.setdefault("num_categories", prep.get("num_categories"))
        payload.setdefault("class_names", prep.get("class_names"))
        payload.setdefault("split_counts", prep.get("split_counts"))
        payload.setdefault("source_counts", prep.get("source_counts"))
        payload.setdefault("prepared_dataset", prep.get("prepared_dataset"))
        payload.setdefault("synthetic_generation_status", prep.get("synthetic_generation_status"))
        payload.setdefault("synthetic_generation_fallback", prep.get("synthetic_generation_fallback"))
        payload.setdefault("synthetic_generation_error", prep.get("synthetic_generation_error"))
        payload.setdefault("synthetic_generation_primary_error", prep.get("synthetic_generation_primary_error"))
    return payload


def _read_data_preparation_summary(paths: ThreadPaths) -> dict[str, Any]:
    preferred = paths.outputs / WORKFLOW_OUTPUT_DIR / "prepared_data" / "data_preparation_summary.json"
    candidates = [preferred] if preferred.is_file() else sorted(paths.outputs.rglob("data_preparation_summary.json"))
    if not candidates:
        return {}
    try:
        payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _analyze_uploaded_dataset(
    dataset_root: Path,
    work_dir: Path,
    labels: list[str],
    recorder: EventRecorder,
) -> dict[str, Any]:
    facts = _inspect_dataset_structure(dataset_root)
    images_dir = Path(str(facts.get("images_dir") or "")) if facts.get("images_dir") else None
    images = _collect_images(images_dir or dataset_root)
    facts["image_count"] = len(images)
    facts["image_size_summary"] = _image_size_summary(images)
    facts["labels_requested"] = labels
    coco_path = _string_path(facts.get("coco_json"))
    if coco_path:
        facts["coco_summary"] = _summarize_coco_dataset(Path(coco_path), images_dir)
    elif facts.get("format") == "yolo" and images_dir and facts.get("labels_dir"):
        facts["yolo_summary"] = _summarize_yolo_dataset(images_dir, Path(str(facts["labels_dir"])))
    work_dir.mkdir(parents=True, exist_ok=True)
    facts_path = work_dir / "dataset_facts.json"
    facts_path.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recorder.emit(
        "workflow.dataset_analyzed",
        {
            "dataset_facts": facts,
            "dataset_facts_path": str(facts_path),
            "image_count": facts.get("image_count"),
            "format": facts.get("format"),
        },
    )
    return facts


def _inspect_dataset_structure(dataset_root: Path) -> dict[str, Any]:
    images_dir = _find_dataset_images_dir(dataset_root)
    labels_dir = _find_dataset_labels_dir(dataset_root)
    label_info = _detect_dataset_label_format(labels_dir)
    return {
        "status": "ok" if images_dir else "error",
        "dataset_root": str(dataset_root),
        "images_dir": str(images_dir) if images_dir else "",
        "labels_dir": str(labels_dir) if labels_dir else "",
        **label_info,
    }


def _find_dataset_images_dir(dataset_root: Path) -> Path | None:
    for candidate in (dataset_root / "images", dataset_root / "image", dataset_root / "JPEGImages"):
        if _collect_images(candidate):
            return candidate.resolve()
    return dataset_root.resolve() if _collect_images(dataset_root) else None


def _find_dataset_labels_dir(dataset_root: Path) -> Path | None:
    for name in ("labels", "label", "annotations", "annotation"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _detect_dataset_label_format(labels_dir: Path | None) -> dict[str, Any]:
    if labels_dir is None or not labels_dir.exists():
        return {"format": "unlabeled", "label_count": 0, "coco_json": ""}
    label_files = [p for p in labels_dir.rglob("*") if p.is_file()]
    if not label_files:
        return {"format": "unlabeled", "label_count": 0, "coco_json": ""}
    coco_files = [p for p in sorted(labels_dir.rglob("*.json")) if _looks_like_coco_json(p)]
    if coco_files:
        return {"format": "coco", "label_count": len(coco_files), "coco_json": str(coco_files[0].resolve())}
    txt_files = [p for p in labels_dir.rglob("*.txt") if p.name.lower() not in {"classes.txt", "obj.names"}]
    if txt_files:
        return {"format": "yolo", "label_count": len(txt_files), "classes_file": str((labels_dir / "classes.txt").resolve()) if (labels_dir / "classes.txt").is_file() else ""}
    xml_files = list(labels_dir.rglob("*.xml"))
    if xml_files:
        return {"format": "pascal_voc", "label_count": len(xml_files), "supported": False}
    return {"format": "unknown", "label_count": len(label_files), "supported": False}


def _looks_like_coco_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return isinstance(payload, dict) and all(key in payload for key in ("images", "annotations", "categories"))


def _image_size_summary(images: list[Path]) -> dict[str, Any]:
    sizes: list[tuple[int, int]] = []
    for image in images[:500]:
        size = _read_image_size(image)
        if size is not None:
            sizes.append(size)
    if not sizes:
        return {"sampled": 0}
    widths = [item[0] for item in sizes]
    heights = [item[1] for item in sizes]
    return {
        "sampled": len(sizes),
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": round(sum(widths) / len(widths), 1),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": round(sum(heights) / len(heights), 1),
        "portrait_count": sum(1 for w, h in sizes if h > w),
        "landscape_count": sum(1 for w, h in sizes if w >= h),
    }


def _read_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _summarize_coco_dataset(coco_path: Path, images_dir: Path | None) -> dict[str, Any]:
    try:
        coco = json.loads(coco_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"error": str(exc)[:500]}
    images = coco.get("images") if isinstance(coco.get("images"), list) else []
    annotations = coco.get("annotations") if isinstance(coco.get("annotations"), list) else []
    categories = coco.get("categories") if isinstance(coco.get("categories"), list) else []
    cat_by_id = {int(cat.get("id")): str(cat.get("name") or "") for cat in categories if isinstance(cat, dict) and cat.get("id") is not None}
    counts: dict[str, int] = {name: 0 for name in cat_by_id.values() if name}
    bbox_areas: list[float] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        name = cat_by_id.get(_safe_int(ann.get("category_id")))
        if name:
            counts[name] = counts.get(name, 0) + 1
        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                bbox_areas.append(max(0.0, float(bbox[2])) * max(0.0, float(bbox[3])))
            except (TypeError, ValueError):
                pass
    return {
        "coco_json": str(coco_path),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "category_counts": counts,
        "bbox_size_summary": _bbox_area_summary(bbox_areas),
        "images_dir": str(images_dir) if images_dir else "",
    }


def _summarize_yolo_dataset(images_dir: Path, labels_dir: Path) -> dict[str, Any]:
    txt_files = [p for p in labels_dir.rglob("*.txt") if p.name.lower() not in {"classes.txt", "obj.names"}]
    class_names = _read_yolo_class_names(labels_dir)
    counts: dict[str, int] = {}
    box_count = 0
    for label_file in txt_files:
        try:
            lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = _safe_int(parts[0])
            name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
            counts[name] = counts.get(name, 0) + 1
            box_count += 1
    return {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "label_file_count": len(txt_files),
        "annotation_count": box_count,
        "category_counts": counts,
        "class_names": class_names,
    }


def _read_yolo_class_names(labels_dir: Path) -> list[str]:
    for name in ("classes.txt", "obj.names"):
        path = labels_dir / name
        if path.is_file():
            return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    return []


def _bbox_area_summary(areas: list[float]) -> dict[str, Any]:
    if not areas:
        return {"count": 0}
    sorted_areas = sorted(areas)
    mid = len(sorted_areas) // 2
    return {
        "count": len(areas),
        "min": round(sorted_areas[0], 2),
        "median": round(sorted_areas[mid], 2),
        "max": round(sorted_areas[-1], 2),
        "small_area_count": sum(1 for area in areas if area < 32 * 32),
    }


def _fallback_training_from_dataset_facts(dataset_facts: dict[str, Any]) -> dict[str, Any]:
    image_count = _safe_int(dataset_facts.get("image_count"))
    image_summary = dataset_facts.get("image_size_summary") if isinstance(dataset_facts.get("image_size_summary"), dict) else {}
    coco_summary = dataset_facts.get("coco_summary") if isinstance(dataset_facts.get("coco_summary"), dict) else {}
    bbox_summary = coco_summary.get("bbox_size_summary") if isinstance(coco_summary.get("bbox_size_summary"), dict) else {}
    small_boxes = _safe_int(bbox_summary.get("small_area_count"))
    avg_width = float(image_summary.get("avg_width") or 0)
    avg_height = float(image_summary.get("avg_height") or 0)
    imgsz = 640
    if small_boxes > 0 or max(avg_width, avg_height) >= 1000:
        imgsz = 960
    if image_count <= 30:
        epochs, patience = 80, 30
        split = dict(DEFAULT_TRAINING_SPLIT)
    elif image_count <= 100:
        epochs, patience = 60, 25
        split = {"train": 0.75, "val": 0.2, "test": 0.05}
    else:
        epochs, patience = 40, 20
        split = {"train": 0.7, "val": 0.2, "test": 0.1}
    batch = 4 if imgsz >= 960 else 8
    return {
        "training": {
            "task": "detect",
            "model": "yolo11n.pt",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": "0",
            "workers": 4,
            "patience": patience,
        },
        "split": split,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _string_path(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _resolve_prepared_dataset_yaml(data_prep_data: dict[str, Any], output_dir: Path) -> str:
    explicit = str(data_prep_data.get("dataset_yaml") or data_prep_data.get("data_yaml") or "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.is_file():
            return str(explicit_path.resolve())

    summary_path = output_dir / "data_preparation_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        if isinstance(summary, dict):
            for key in ("dataset_yaml", "data_yaml"):
                value = str(summary.get(key) or "").strip()
                if value and Path(value).expanduser().is_file():
                    return str(Path(value).expanduser().resolve())

    candidates = [output_dir / "dataset.yaml", output_dir / "data.yaml", *sorted(output_dir.rglob("dataset.yaml")), *sorted(output_dir.rglob("data.yaml"))]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return str((output_dir / "dataset.yaml").resolve())


def _read_pipeline_paths(paths: ThreadPaths) -> dict[str, str]:
    result: dict[str, str] = {}
    prep_summary = _read_data_preparation_summary(paths)
    if prep_summary:
        if prep_summary.get("prepared_dataset"):
            result["dataset_root"] = str(prep_summary.get("prepared_dataset"))
        if prep_summary.get("training_coco"):
            result["coco_json"] = str(prep_summary.get("training_coco"))
        if prep_summary.get("dataset_yaml"):
            result["dataset_yaml"] = str(prep_summary.get("dataset_yaml"))
        if prep_summary.get("synthetic_plan"):
            result["synthetic_plan"] = str(prep_summary.get("synthetic_plan"))

    preferred_training_input = paths.workspace / "gpu-training-orchestrator-training-input.json"
    if preferred_training_input.is_file():
        training_input = preferred_training_input
    else:
        legacy_training_input = paths.outputs / WORKFLOW_OUTPUT_DIR / PIPELINE_WORK_DIR / "training_input.json"
        candidates = ([legacy_training_input] if legacy_training_input.is_file() else [])
        candidates += sorted(paths.workspace.rglob("*training-input.json")) + sorted(paths.outputs.rglob("training_input.json"))
        training_input = candidates[-1] if candidates else None
    if training_input and training_input.exists():
        result["training_input"] = str(training_input)
        try:
            payload = json.loads(training_input.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
        if dataset.get("data_yaml"):
            result["dataset_yaml"] = str(dataset.get("data_yaml"))
    preferred_plan = paths.outputs / WORKFLOW_OUTPUT_DIR / PIPELINE_WORK_DIR / "synthetic_plan.json"
    if "synthetic_plan" not in result and preferred_plan.is_file():
        result["synthetic_plan"] = str(preferred_plan)
    elif "synthetic_plan" not in result:
        legacy_plan = paths.outputs / WORKFLOW_OUTPUT_DIR / "smoking_pipeline" / "synthetic_plan.json"
        plan_candidates = ([legacy_plan] if legacy_plan.is_file() else [])
        plan_candidates += sorted(paths.workspace.rglob("synthetic_plan.json")) + sorted(paths.outputs.rglob("synthetic_plan.json"))
        if plan_candidates:
            result["synthetic_plan"] = str(plan_candidates[-1])
    return result


def _find_best_pt(paths: ThreadPaths) -> str:
    preferred = paths.outputs / WORKFLOW_OUTPUT_DIR / "training_run" / "train" / "weights" / "best.pt"
    if preferred.is_file():
        return str(preferred)
    candidates = sorted(paths.outputs.rglob("best.pt"))
    return str(candidates[-1]) if candidates else ""


def _input_label(value: object) -> str:
    return {"dataset": "数据集", "image": "图片", "model_config": "模型配置"}.get(str(value), "输入")


def _filter_training_run_artifacts(outputs: list[Any]) -> list[Any]:
    result: list[Any] = []
    marker = f"/{WORKFLOW_OUTPUT_DIR}/training_run/"
    for artifact in outputs:
        path = str(getattr(artifact, "path", "") or "").replace("\\", "/")
        if marker in path:
            result.append(artifact)
    return result


def _reference_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "reference_image_path.txt"


def _composite_image1_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "composite_image1_path.txt"


def _composite_image2_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "composite_image2_path.txt"


def _prompt_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "generation_prompt.txt"


def _training_objective_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "training_objective.txt"


def _save_training_objective(paths: ThreadPaths, objective: str) -> None:
    if objective:
        _training_objective_marker_path(paths).write_text(objective.strip(), encoding="utf-8")


def _load_training_objective(paths: ThreadPaths) -> str:
    marker = _training_objective_marker_path(paths)
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    objective = _load_training_objective_from_memory(paths)
    if objective:
        _save_training_objective(paths, objective)
    return objective


def _load_training_objective_from_memory(paths: ThreadPaths) -> str:
    memory_file = paths.root / "memory" / "conversation.jsonl"
    if not memory_file.exists():
        return ""
    try:
        lines = memory_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return ""
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:
            continue
        message = payload.get("message") if isinstance(payload, dict) else {}
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        objective = _training_objective_from_user_text(str(message.get("content") or ""))
        if objective:
            return objective
    return ""


def _save_generation_prompt(paths: ThreadPaths, prompt: str) -> None:
    if prompt:
        _prompt_marker_path(paths).write_text(prompt.strip(), encoding="utf-8")


def _load_generation_prompt(paths: ThreadPaths) -> str:
    marker = _prompt_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _selected_skills_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "selected_skills.json"


def _save_selected_skills(paths: ThreadPaths, selected_skills: list[str]) -> None:
    if selected_skills:
        _selected_skills_marker_path(paths).write_text(json.dumps(selected_skills, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_selected_skills(paths: ThreadPaths) -> list[str]:
    marker = _selected_skills_marker_path(paths)
    if not marker.exists():
        return []
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item).strip() for item in payload if str(item).strip()]


def _labels_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "annotation_labels.json"


def _save_annotation_labels(paths: ThreadPaths, labels: list[str]) -> None:
    normalized = _normalize_detection_labels(labels)
    if normalized:
        _labels_marker_path(paths).write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_annotation_labels(paths: ThreadPaths) -> list[str]:
    marker = _labels_marker_path(paths)
    if not marker.exists():
        return []
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return []
    return _normalize_detection_labels([str(item).strip() for item in payload if str(item).strip()]) if isinstance(payload, list) else []


def _training_config_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "training_config.json"


def _training_model_registry_path(paths: ThreadPaths) -> Path:
    return paths.workspace / TRAINING_MODEL_REGISTRY


def _requested_training_model_id(runtime_options: RuntimeOptions) -> str:
    direct = str(getattr(runtime_options, "training_model_id", None) or "").strip()
    if direct:
        return direct
    params = _workflow_skill_parameters(runtime_options)
    for key in ("modelId", "model_id", "trainingModelId", "training_model_id"):
        value = str(params.get(key) or "").strip()
        if value:
            return value
    return ""


def _resolve_user_training_model(
    paths: ThreadPaths,
    model_id: str,
) -> tuple[dict[str, Any] | None, str]:
    requested_model_id = str(model_id or "").strip()
    if not requested_model_id:
        return None, ""
    registry_path = _training_model_registry_path(paths)
    if not registry_path.exists():
        return None, f"当前线程没有已上传模型，找不到 modelId={requested_model_id}"
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"训练模型注册表无法读取：{exc}"
    if not isinstance(payload, dict):
        return None, "训练模型注册表格式无效"
    models = payload.get("models")
    if not isinstance(models, dict):
        return None, "训练模型注册表中缺少 models"
    record = models.get(requested_model_id)
    if not isinstance(record, dict):
        return None, f"当前线程找不到 modelId={requested_model_id}"
    if str(record.get("modelId") or "").strip() != requested_model_id:
        return None, f"模型记录与请求的 modelId={requested_model_id} 不一致"
    return _validate_user_training_model_record(paths, record)


def _validate_user_training_model_record(
    paths: ThreadPaths,
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    raw_path = str(record.get("local_path") or record.get("path") or "").strip()
    if not raw_path:
        return None, "训练模型路径为空"
    model_path = _resolve_uploaded_local_path(paths.root, raw_path)
    try:
        resolved = model_path.resolve(strict=True)
    except OSError:
        return None, f"找不到模型文件：{raw_path}"
    models_root = (paths.uploads / "models").resolve()
    try:
        resolved.relative_to(models_root)
    except ValueError:
        return None, "模型文件不在当前线程的 uploads/models 目录中"
    if not resolved.is_file() or resolved.suffix.lower() != ".pt":
        return None, "模型文件必须是有效的 .pt 文件"
    if resolved.stat().st_size <= 0:
        return None, "模型文件为空"

    actual_sha256 = _file_sha256(resolved)
    expected_sha256 = str(record.get("sha256") or "").strip().lower()
    if expected_sha256 and expected_sha256 != actual_sha256:
        return None, "模型文件 SHA256 与上传记录不一致"
    relative = resolved.relative_to(paths.uploads).as_posix()
    return {
        **record,
        "source": "user_upload",
        "name": str(record.get("name") or resolved.name),
        "path": f"/mnt/user-data/uploads/{relative}",
        "local_path": str(resolved),
        "mime_type": "application/octet-stream",
        "size": resolved.stat().st_size,
        "sha256": actual_sha256,
    }, ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_user_training_model(training_cfg: dict[str, Any], model_record: dict[str, Any]) -> None:
    training = training_cfg.get("training")
    if not isinstance(training, dict):
        return
    training["model"] = str(model_record["local_path"])
    training["model_source"] = "user_upload"
    training["strict_model"] = True
    training["model_sha256"] = str(model_record["sha256"])
    training["model_original_name"] = str(model_record.get("name") or "")
    training["model_id"] = str(model_record.get("modelId") or "")


def _clear_user_training_model_selection(
    training_cfg: dict[str, Any],
    request_spec: dict[str, Any],
) -> None:
    training = training_cfg.get("training")
    if not isinstance(training, dict):
        return
    has_user_selection = bool(
        str(training.get("model_source") or "").strip() == "user_upload"
        or training.get("strict_model")
        or str(training.get("model_id") or "").strip()
    )
    current_model = str(training.get("model") or "").replace("\\", "/").lower()
    if "/uploads/models/" in current_model:
        has_user_selection = True
    if not has_user_selection:
        return

    request_training = request_spec.get("training")
    llm_model = ""
    if isinstance(request_training, dict):
        llm_model = str(request_training.get("model") or "").strip()
        normalized_llm_model = llm_model.replace("\\", "/").lower()
        if (
            str(request_training.get("model_source") or "").strip() == "user_upload"
            or request_training.get("strict_model")
            or str(request_training.get("model_id") or "").strip()
            or "/uploads/models/" in normalized_llm_model
        ):
            llm_model = ""
    training["model"] = llm_model or "yolo11n.pt"
    for key in ("model_source", "strict_model", "model_sha256", "model_original_name", "model_id"):
        training.pop(key, None)


def _save_training_config(paths: ThreadPaths, training_cfg: dict[str, Any]) -> None:
    if training_cfg:
        _training_config_marker_path(paths).write_text(json.dumps(training_cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_training_config(paths: ThreadPaths) -> dict[str, Any]:
    marker = _training_config_marker_path(paths)
    if not marker.exists():
        return {}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _detection_task_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "detection_task_description.txt"


def _save_detection_task_description(paths: ThreadPaths, task_description: str) -> None:
    if task_description:
        _detection_task_marker_path(paths).write_text(task_description.strip(), encoding="utf-8")


def _load_detection_task_description(paths: ThreadPaths) -> str:
    marker = _detection_task_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _synthetic_generation_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "synthetic_generation_enabled.json"


def _save_synthetic_generation_enabled(paths: ThreadPaths, enabled: bool) -> None:
    _synthetic_generation_marker_path(paths).write_text(json.dumps(bool(enabled)), encoding="utf-8")


def _load_synthetic_generation_enabled(paths: ThreadPaths) -> bool | None:
    marker = _synthetic_generation_marker_path(paths)
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, bool) else None


def _save_reference_image_path(paths: ThreadPaths, path: str) -> None:
    if path:
        _reference_marker_path(paths).write_text(path.strip(), encoding="utf-8")


def _load_reference_image_path(paths: ThreadPaths) -> str:
    marker = _reference_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _save_composite_input_paths(paths: ThreadPaths, paths_to_save: list[Path]) -> None:
    values = [str(path.resolve()) for path in paths_to_save if str(path).strip()]
    if not values:
        return
    if len(values) >= 2:
        _composite_image1_marker_path(paths).write_text(values[0], encoding="utf-8")
        _composite_image2_marker_path(paths).write_text(values[1], encoding="utf-8")
        return
    if not _load_composite_image1_path(paths):
        _composite_image1_marker_path(paths).write_text(values[0], encoding="utf-8")
    else:
        _composite_image2_marker_path(paths).write_text(values[0], encoding="utf-8")


def _prepare_composite_input_path(paths: ThreadPaths, raw_path: str, index: int) -> Path:
    source = _resolve_uploaded_local_path(paths.workspace.parent, raw_path)
    if source.is_file() and _is_archive_path(source):
        target_dir = paths.outputs / WORKFLOW_OUTPUT_DIR / "composite_inputs" / f"image{index}"
        return _unpack_archive_to_dir(source, target_dir)
    return source


def _prepare_composite_input_paths(paths: ThreadPaths, attachments: list[Attachment], role_hints: dict[str, str]) -> list[Path]:
    if not attachments:
        return []
    role_by_path: dict[str, int] = {}
    for role, index in (("image1", 1), ("image2", 2)):
        hinted = _attachment_for_role(attachments, role, role_hints)
        if hinted is not None and hinted.path:
            role_by_path[str(hinted.path)] = index

    prepared: list[Path | None] = [None, None]
    next_index = 1
    for item in attachments:
        if not item.path:
            continue
        index = role_by_path.get(str(item.path))
        if index is None:
            while next_index <= 2 and prepared[next_index - 1] is not None:
                next_index += 1
            if next_index > 2:
                break
            index = next_index
        prepared[index - 1] = _prepare_composite_input_path(paths, item.path, index)
    return [path for path in prepared if path is not None]


def _clear_composite_input_paths(paths: ThreadPaths) -> None:
    for marker in (_composite_image1_marker_path(paths), _composite_image2_marker_path(paths), _reference_marker_path(paths)):
        if marker.exists():
            marker.unlink()


def _load_composite_image1_path(paths: ThreadPaths) -> str:
    marker = _composite_image1_marker_path(paths)
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    return _load_reference_image_path(paths)


def _load_composite_image2_path(paths: ThreadPaths) -> str:
    marker = _composite_image2_marker_path(paths)
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else ""


def _waiting_prompt_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "awaiting_prompt.flag"


def _workflow_completed_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "workflow_completed.flag"


def _set_workflow_completed(paths: ThreadPaths, completed: bool) -> None:
    flag = _workflow_completed_path(paths)
    if completed:
        flag.write_text("1", encoding="utf-8")
    elif flag.exists():
        flag.unlink()


def _is_workflow_completed(paths: ThreadPaths) -> bool:
    if _workflow_completed_path(paths).exists():
        return True
    output_root = paths.outputs / WORKFLOW_OUTPUT_DIR
    completion_markers = (
        output_root / "training_run" / "run_summary.json",
        output_root / "training_run" / "train" / "weights" / "best.pt",
        output_root / "prepared_data" / "data_preparation_summary.json",
    )
    return any(marker.exists() for marker in completion_markers)


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
    normalized = value.replace("\\", "/")
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


def _unpack_dataset_archive(archive_path: str, target_dir: Path, thread_root: Path) -> Path:
    source = _resolve_uploaded_local_path(thread_root, archive_path)
    if not source.exists():
        raise FileNotFoundError(f"找不到上传的文件: {source}")
    return _unpack_archive_to_dir(source, target_dir)


def _is_archive_path(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".tar.gz") or lower_name.endswith(".tar") or lower_name.endswith(".zip")


def _unpack_archive_to_dir(source: Path, target_dir: Path) -> Path:
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
    for img in _collect_images(dataset_root):
        target = merged_images / img.name
        if not target.exists():
            shutil.copy2(img, target)
    for img in _collect_images(generated_dir, excluded_names=excluded_image_names):
        target = merged_images / img.name
        if not target.exists():
            shutil.copy2(img, target)
    return out_root


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
