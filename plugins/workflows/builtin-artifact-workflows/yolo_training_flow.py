from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.events import EventRecorder
from app.core.http_training_jobs import read_current_http_training_job_marker
from app.core.llm.openai_compatible import OpenAICompatibleClient
from app.core.skills import SkillRunner
from app.core.training_status import TrainingProgressWriter, chain_event_hooks
from app.schemas import AgentRunResult, Attachment, ChatEvent, Message, RuntimeOptions, VerificationResult

_DATASET_PACKAGE_EXTS = (".zip", ".tar", ".tar.gz")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
WORKFLOW_NAME = "yolo_training_flow"
WORKFLOW_OUTPUT_DIR = "yolo_training_flow"
PIPELINE_WORK_DIR = "pipeline_work"
TRAINING_MODEL_REGISTRY = "training_models.json"
SUPPORTED_TRAINING_MODEL_SUFFIXES = {".pt", ".pth"}
DEFAULT_SELECTED_SKILLS = ["image-dataset-generation", "image-dataset-produce", "data-auto-annotation", "gpu-training-orchestrator"]
DEIMV2_SELECTED_SKILLS = ["image-dataset-generation", "image-dataset-produce", "data-auto-annotation", "deimv2-auto-training"]
YOLO_TRAINING_SKILL = "gpu-training-orchestrator"
DEIMV2_TRAINING_SKILL = "deimv2-auto-training"
DEIMV2_DEFAULT_MODEL_VARIANT = "deimv2-dinov3-s"
DEIMV2_BACKBONE_CHECKPOINT = "models/deimv2/vitt_distill.pt"
DEIMV2_MODEL_VARIANTS = {
    "deimv2-dinov3-s": {
        "template_config": "configs/deimv2/deimv2_dinov3_s_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_s_coco.pth",
    },
    "deimv2-dinov3-m": {
        "template_config": "configs/deimv2/deimv2_dinov3_m_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_m_coco.pth",
    },
    "deimv2-dinov3-l": {
        "template_config": "configs/deimv2/deimv2_dinov3_l_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_l_coco.pth",
    },
    "deimv2-dinov3-x": {
        "template_config": "configs/deimv2/deimv2_dinov3_x_coco.yml",
        "tuning_checkpoint": "models/deimv2/deimv2_dinov3_x_coco.pth",
    },
}
DEFAULT_TRAINING_SPLIT = {"train": 0.7, "val": 0.2, "test": 0.1}
MIN_TEST_SPLIT = 0.1

DEFAULT_MAX_SYNTHETIC_IMAGES = 10
DEFAULT_ANNOTATION_PROVIDER = "locate_sam3"
# button_epochs=False 是本地冒烟测试路径：忽略模型思考出的 epochs，
# 强制使用较短轮数。button_epochs=True 时允许模型结合数据集思考轮数，
# 但 _cap_training_epochs 仍会兜底限制最大值。
button_epochs = True
button_export_onnx = True
FIXED_TRAINING_EPOCHS = 10
MAX_TRAINING_EPOCHS = 200
AUTO_GENERATE_MISSING_SPEC = True
DATA_REUSE_MANIFEST_NAME = "data_preparation_reuse_manifest.json"
REAL_DATA_REUSE_COPY_PATHS = (
    "uploaded_dataset",
    "pipeline_work/real_coco.json",
)
SYNTHETIC_DATA_REUSE_COPY_PATHS = (
    "pipeline_work/synthetic_coco.json",
    "pipeline_work/merged_coco.json",
    "pipeline_work/merged_images",
    "pipeline_work/synthetic_images",
    "pipeline_work/synthetic_annotations",
    "pipeline_work/synthetic_plan.json",
)
DATA_REUSE_COPY_PATHS = REAL_DATA_REUSE_COPY_PATHS + SYNTHETIC_DATA_REUSE_COPY_PATHS


@dataclass(frozen=True)
class TrainingRunPaths:
    thread: ThreadPaths
    run_id: str
    workspace: Path
    outputs: Path

    @property
    def root(self) -> Path:
        return self.thread.root

    @property
    def uploads(self) -> Path:
        return self.thread.uploads

    @property
    def thread_id(self) -> str:
        return self.thread.thread_id


@dataclass(frozen=True)
class DataPreparationReuse:
    source_run_id: str
    dataset_root: Path
    coco_json: Path
    real_images: int
    synthetic_images: int
    annotations: int
    copied_files: int
    reuse_level: str
    synthetic_reused: bool


class YoloTrainingWorkflow:
    """通用训练工作流，支持 YOLO/DEIMv2 分支和可选合成数据。"""

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
        progress_writer: TrainingProgressWriter | None = None

        def _install_progress_writer(active_run_paths: TrainingRunPaths) -> None:
            nonlocal progress_writer
            if progress_writer is not None:
                return
            progress_writer = TrainingProgressWriter(
                thread_id=paths.thread_id,
                run_id=active_run_paths.run_id,
                run_dir=active_run_paths.outputs,
                workflow=workflow,
                backend=training_backend,
                app_template_name=runtime_options.app_template_name,
                selected_skills=selected_skills,
                user_text=user_text,
            )
            progress_writer.initialize()
            recorder.on_emit = chain_event_hooks(on_event, progress_writer.handle_event)

        user_text = _last_user_text(messages)
        selected_skills = _selected_skills(runtime_options)
        training_backend = _selected_training_backend(selected_skills, runtime_options, workflow)
        # ACP 客户端的后续请求可能不再携带 selectedSkills。
        # 持久化上一次技能选择，可以让同一会话继续留在 YOLO/DEIMv2 分支，
        # 直到新请求明确切换分支。
        if _has_runtime_selected_skills(runtime_options):
            _save_selected_skills(paths, selected_skills)
        else:
            selected_skills = _load_selected_skills(paths) or selected_skills
            training_backend = _selected_training_backend(selected_skills, runtime_options, workflow)
        deimv2_model_selection = _select_deimv2_model_variant(runtime_options, recorder) if training_backend == "deimv2" else {}
        generation_enabled = _capability_enabled(selected_skills, "image_generation")
        training_enabled = _capability_enabled(selected_skills, "training")
        active_run_paths = _load_active_training_run_paths(paths)
        existing_dataset_pkg = _load_dataset_package_path(paths)
        existing_composite_image1 = _load_composite_image1_path(paths)
        existing_composite_image2 = _load_composite_image2_path(paths)
        current_objective = _training_objective_from_user_text(user_text)
        waiting_prompt = _is_waiting_prompt(active_run_paths or paths)
        # 当用户提出明确的新训练目标时，创建独立 run 工作区。
        # 这样同一线程二次训练时，旧的 generation_prompt/training_config
        # 不会污染新的训练请求。
        start_new_run = bool(current_objective and not waiting_prompt)
        reuse_source_run_paths = (
            _select_data_reuse_source(paths, active_run_paths, runtime_options)
            if start_new_run
            else None
        )
        if start_new_run:
            run_paths = _create_training_run_paths(paths, training_backend)
            _save_active_training_run(paths, run_paths, training_backend=training_backend, objective=current_objective)
            _set_waiting_prompt(run_paths, False)
            _set_workflow_completed(run_paths, False)
            _set_workflow_completed(paths, False)
        elif active_run_paths is not None:
            run_paths = active_run_paths
        else:
            run_paths = None
        state_paths = run_paths or paths
        waiting_prompt = _is_waiting_prompt(state_paths)
        workflow_completed = _is_workflow_completed(state_paths)
        workflow_output_root = _workflow_output_root(state_paths)
        if run_paths is not None:
            _install_progress_writer(run_paths)
            recorder.emit(
                "workflow.training_run.selected",
                {
                    "run_id": run_paths.run_id,
                    "workspace": str(run_paths.workspace),
                    "outputs": str(run_paths.outputs),
                    "training_backend": training_backend,
                    "new_run": start_new_run or active_run_paths is None,
                },
            )
        if current_objective:
            _save_training_objective(paths, current_objective)
        stored_objective = _load_training_objective(paths)
        objective_text = current_objective or (stored_objective if not workflow_completed or waiting_prompt else "")
        spec_user_text = _combine_spec_user_text(objective_text, user_text)
        explicit_attachment_roles = _parse_attachment_role_hints(user_text)

        dataset_attachment = _find_dataset_package_attachment(
            attachments,
            role_hints=explicit_attachment_roles,
            include_thread_files=False,
        )
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

        attachment_thread_error = _validate_attachment_thread_paths(paths, [item for item in [dataset_attachment, *composite_attachments] if item is not None])
        if attachment_thread_error:
            return self._model_spec_failed_result(
                recorder,
                agent_config.name,
                paths.thread_id,
                workflow,
                attachment_thread_error,
                phase="attachment_thread_mismatch",
            )

        if dataset_attachment and dataset_attachment.path:
            _set_workflow_completed(paths, False)
            if run_paths is not None:
                _set_workflow_completed(run_paths, False)
            resolved_dataset = _resolve_uploaded_local_path(paths.root, dataset_attachment.path)
            _save_dataset_package_path(paths, str(resolved_dataset))
            # 新数据集会让上一轮 image1/image2 合成输入失效。
            # 清理旧合成输入，避免后续训练把旧合成素材和新真实数据混在一起。
            _clear_composite_input_paths(paths)
        if composite_attachments:
            _set_workflow_completed(paths, False)
            if run_paths is not None:
                _set_workflow_completed(run_paths, False)
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

        if run_paths is None:
            run_paths = _create_training_run_paths(paths, training_backend)
            _save_active_training_run(paths, run_paths, training_backend=training_backend, objective=current_objective or stored_objective)
            workflow_output_root = run_paths.outputs
            workflow_completed = _is_workflow_completed(run_paths)
            waiting_prompt = _is_waiting_prompt(run_paths)
            _install_progress_writer(run_paths)
            recorder.emit(
                "workflow.training_run.selected",
                {
                    "run_id": run_paths.run_id,
                    "workspace": str(run_paths.workspace),
                    "outputs": str(run_paths.outputs),
                    "training_backend": training_backend,
                    "new_run": True,
                },
            )

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
            selected_skills = _ensure_full_cycle_skills(selected_skills, training_backend)
            request_spec = _generate_model_managed_training_intent_spec(
                user_text=spec_user_text,
                agent_config=agent_config,
                runtime_options=runtime_options,
                recorder=recorder,
                training_backend=training_backend,
                deimv2_model_selection=deimv2_model_selection,
                workflow_output_root=workflow_output_root,
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
        if _http_training_cancel_requested(paths, run_paths):
            return _cancelled_training_result(
                recorder=recorder,
                agent_name=agent_config.name,
                thread_id=paths.thread_id,
                workflow=workflow,
                run_paths=run_paths,
                phase="cancelled",
                reply="训练任务已收到取消请求，已停止后续训练流程。",
            )
        synthetic_generation = _spec_optional_bool(request_spec, "use_synthetic_generation")
        if synthetic_generation is not None:
            _save_synthetic_generation_enabled(run_paths, synthetic_generation)
        persisted_synthetic_generation = _load_synthetic_generation_enabled(run_paths)
        if model_managed_spec:
            generation_enabled = True
            _save_synthetic_generation_enabled(run_paths, True)
        elif persisted_synthetic_generation is not None:
            generation_enabled = generation_enabled and persisted_synthetic_generation

        if model_managed_spec:
            prompt_text = _spec_string(request_spec, "generation_prompt")
            labels = _normalize_detection_labels(_spec_string_list(request_spec, "labels"))
            training_cfg: dict[str, Any] = {}
            training_cfg_available = False
            task_description = _spec_string(request_spec, "task_description")
            if not prompt_text:
                prompt_text = _fallback_generation_prompt(labels, task_description=task_description, user_text=spec_user_text)
                if prompt_text:
                    request_spec["generation_prompt"] = prompt_text
                    request_spec["use_synthetic_generation"] = True
                    recorder.emit(
                        "workflow.generation_prompt_fallback",
                        {
                            "reason": "model_managed_spec_missing_generation_prompt",
                            "labels": labels,
                            "generation_prompt": prompt_text,
                        },
                    )
            if prompt_text:
                _save_generation_prompt(run_paths, prompt_text)
            if labels:
                _save_annotation_labels(run_paths, labels)
            if task_description:
                _save_detection_task_description(run_paths, task_description)
            generation_enabled = True
            _save_synthetic_generation_enabled(run_paths, True)
        else:
            prompt_text = _spec_string(request_spec, "generation_prompt") or _extract_generation_prompt(spec_user_text, allow_free_text=waiting_prompt)
            if prompt_text:
                _save_generation_prompt(run_paths, prompt_text)
            else:
                prompt_text = _load_generation_prompt(run_paths)
            labels = _normalize_detection_labels(_spec_string_list(request_spec, "labels") or _extract_annotation_labels(spec_user_text))
            if labels:
                _save_annotation_labels(run_paths, labels)
            else:
                labels = _load_annotation_labels(run_paths)
            training_cfg = _spec_training_config(request_spec)
            if training_backend == "deimv2":
                _apply_deimv2_model_selection_to_training_config(training_cfg, deimv2_model_selection)
                _apply_deimv2_export_onnx_policy(training_cfg)
            training_cfg_available = bool(training_cfg)
            if training_cfg:
                _save_training_config(run_paths, training_cfg)
            else:
                training_cfg = _load_training_config(run_paths)
                training_cfg_available = bool(training_cfg)
            if not training_cfg:
                training_cfg = _extract_training_config(spec_user_text)
                training_cfg_available = _has_explicit_training_config(spec_user_text)
                if training_cfg_available:
                    _save_training_config(run_paths, training_cfg)
            task_description = _spec_string(request_spec, "task_description")
            if task_description:
                _save_detection_task_description(run_paths, task_description)
            else:
                task_description = _load_detection_task_description(run_paths)

        generation_requested = bool(generation_enabled)
        generation_enabled, generation_skip_reason = _effective_synthetic_generation(
            generation_requested,
            composite_image1,
            composite_image2,
        )

        if not dataset_pkg:
            required_inputs = []
            required_inputs.append({"type": "dataset", "accept": ".zip,.tar,.tar.gz", "required": True, "reason": "需要上传 datasets.zip 数据集压缩包"})
            return self._input_required_result(
                recorder,
                agent_config.name,
                paths.thread_id,
                workflow,
                required_inputs,
            )

        if generation_enabled and not prompt_text:
            _set_waiting_prompt(run_paths, True)
            if _auto_generate_missing_spec(runtime_options):
                reply = _model_generation_failed_reply(
                    "模型未能生成合成提示词，无法继续自动合成数据。",
                    recorder,
                    missing_field="generation_prompt",
                )
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
            _set_waiting_prompt(run_paths, True)
            if _auto_generate_missing_spec(runtime_options):
                reply = _model_generation_failed_reply(
                    "模型未能生成自动标注类别，无法继续自动标注和训练。",
                    recorder,
                    missing_field="labels",
                )
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
            _set_waiting_prompt(run_paths, True)
            if _auto_generate_missing_spec(runtime_options):
                reply = _model_generation_failed_reply(
                    "模型未能生成完整训练参数，无法继续自动训练。",
                    recorder,
                    missing_field="training",
                )
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

        if _http_training_cancel_requested(paths, run_paths):
            return _cancelled_training_result(
                recorder=recorder,
                agent_name=agent_config.name,
                thread_id=paths.thread_id,
                workflow=workflow,
                run_paths=run_paths,
                phase="cancelled",
                reply="训练任务已收到取消请求，已在数据准备前停止。",
            )

        _reset_generated_dir(workflow_output_root / PIPELINE_WORK_DIR)
        _reset_generated_dir(workflow_output_root / "training_run")
        _reset_generated_dir(workflow_output_root / "deimv2_training_run")
        _set_waiting_prompt(run_paths, False)
        recorder.emit("spec.started", {"selected_skills": selected_skills, "attachment_count": len(attachments)})

        unpack_root = workflow_output_root / "uploaded_dataset"
        dataset_root = _unpack_dataset_archive(dataset_pkg, unpack_root, paths.root)
        dataset_facts = _analyze_uploaded_dataset(dataset_root, workflow_output_root / PIPELINE_WORK_DIR, labels, recorder)
        if _http_training_cancel_requested(paths, run_paths):
            return _cancelled_training_result(
                recorder=recorder,
                agent_name=agent_config.name,
                thread_id=paths.thread_id,
                workflow=workflow,
                run_paths=run_paths,
                phase="cancelled",
                reply="训练任务已收到取消请求，已在数据集分析后停止。",
                metadata={"dataset_facts": dataset_facts},
            )
        if model_managed_spec:
            request_spec = _generate_dataset_aware_training_request_spec(
                base_spec=request_spec,
                dataset_facts=dataset_facts,
                user_text=spec_user_text,
                agent_config=agent_config,
                runtime_options=runtime_options,
                recorder=recorder,
                training_backend=training_backend,
                deimv2_model_selection=deimv2_model_selection,
            )
            request_spec = _fallback_model_managed_training_request_spec(
                request_spec,
                spec_user_text,
                dataset_facts,
                training_backend=training_backend,
                deimv2_model_selection=deimv2_model_selection,
            )
            request_spec = _ensure_intent_labels(request_spec, spec_user_text)
            if training_backend == "deimv2":
                _apply_deimv2_model_selection_to_spec(request_spec, deimv2_model_selection)
            synthetic_generation = _spec_optional_bool(request_spec, "use_synthetic_generation")
            if synthetic_generation is not None:
                generation_requested = synthetic_generation
                _save_synthetic_generation_enabled(run_paths, synthetic_generation)
            prompt_text = _spec_string(request_spec, "generation_prompt") or prompt_text
            labels = _normalize_detection_labels(_spec_string_list(request_spec, "labels") or labels)
            training_cfg = _spec_training_config(request_spec)
            if training_backend == "deimv2":
                _apply_deimv2_model_selection_to_training_config(training_cfg, deimv2_model_selection)
                _apply_deimv2_export_onnx_policy(training_cfg)
            training_cfg_available = bool(training_cfg)
            task_description = _spec_string(request_spec, "task_description") or task_description
            if prompt_text:
                _save_generation_prompt(run_paths, prompt_text)
            if labels:
                _save_annotation_labels(run_paths, labels)
            if task_description:
                _save_detection_task_description(run_paths, task_description)
            if training_cfg:
                _apply_epochs_policy(training_cfg, runtime_options)
                request_spec["training"]["epochs"] = training_cfg["training"]["epochs"]
                _force_current_runtime(training_cfg)
                _save_training_config(run_paths, training_cfg)
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

        generation_enabled, generation_skip_reason = _effective_synthetic_generation(
            generation_requested,
            composite_image1,
            composite_image2,
        )

        if _http_training_cancel_requested(paths, run_paths):
            return _cancelled_training_result(
                recorder=recorder,
                agent_name=agent_config.name,
                thread_id=paths.thread_id,
                workflow=workflow,
                run_paths=run_paths,
                phase="cancelled",
                reply="训练任务已收到取消请求，已在启动数据处理前停止。",
                metadata={"dataset_facts": dataset_facts, "model_generated_spec": _model_generated_spec_payload(request_spec)},
            )

        if user_training_model:
            user_model_apply_error = _apply_user_training_model(training_cfg, user_training_model, training_backend)
            if user_model_apply_error:
                return self._model_spec_failed_result(
                    recorder,
                    agent_config.name,
                    paths.thread_id,
                    workflow,
                    f"用户上传的训练模型不可用：{user_model_apply_error}",
                    phase="user_training_model_invalid",
                )
            _save_training_config(run_paths, training_cfg)
            request_training = request_spec.get("training")
            if isinstance(request_training, dict):
                _apply_user_training_model({"training": request_training}, user_training_model, training_backend)
            recorder.emit(
                "workflow.user_model_selected",
                {
                    "modelId": user_training_model["modelId"],
                    "name": user_training_model["name"],
                    "path": user_training_model["path"],
                    "sha256": user_training_model["sha256"],
                    "source": "user_upload",
                    "training_backend": training_backend,
                    "usage": _user_training_model_usage(training_backend, user_training_model),
                },
            )
        else:
            _clear_user_training_model_selection(training_cfg, request_spec, training_backend)
            _save_training_config(run_paths, training_cfg)

        pipeline_work_dir = str((workflow_output_root / PIPELINE_WORK_DIR).resolve())
        project_dir_name = "deimv2_training_run" if training_backend == "deimv2" else "training_run"
        project_dir = str((workflow_output_root / project_dir_name).resolve())
        run_name = "."
        if not task_description:
            task_description = _extract_detection_task_description(user_text, prompt_text, labels)
        annotation_prompt_map = _annotation_prompt_map_from_spec(request_spec, labels)
        annotation_prompts = list(annotation_prompt_map.keys())
        annotation_provider = _default_annotation_provider()
        intent_items = _spec_intent_items(request_spec)
        reuse_fingerprint = _data_preparation_fingerprint(
            dataset_package=Path(dataset_pkg),
            image1=Path(composite_image1) if composite_image1 else None,
            image2=Path(composite_image2) if composite_image2 else None,
            labels=labels,
            annotation_provider=annotation_provider,
            generation_enabled=generation_enabled,
            max_synthetic_images=_max_synthetic_images(runtime_options),
        )
        reused_data, reuse_reason = _reuse_previous_data_preparation(
            source=reuse_source_run_paths,
            target=run_paths,
            fingerprint=reuse_fingerprint,
            mode=_data_reuse_mode(runtime_options),
        )
        if reused_data is not None:
            dataset_root = reused_data.dataset_root
            recorder.emit(
                "data_preparation.artifacts_reused",
                {
                    "source_run_id": reused_data.source_run_id,
                    "target_run_id": run_paths.run_id,
                    "real_images": reused_data.real_images,
                    "synthetic_images": reused_data.synthetic_images,
                    "annotations": reused_data.annotations,
                    "copied_files": reused_data.copied_files,
                    "reuse_level": reused_data.reuse_level,
                    "synthetic_reused": reused_data.synthetic_reused,
                },
            )
        elif reuse_source_run_paths is not None and _data_reuse_mode(runtime_options) != "never":
            recorder.emit(
                "data_preparation.reuse_skipped",
                {
                    "source_run_id": reuse_source_run_paths.run_id,
                    "target_run_id": run_paths.run_id,
                    "reason": reuse_reason,
                },
            )
        if reused_data is None and _data_reuse_mode(runtime_options) == "required":
            return self._model_spec_failed_result(
                recorder,
                agent_config.name,
                paths.thread_id,
                workflow,
                f"指定必须复用上一轮数据准备产物，但复用条件不满足：{reuse_reason}",
                phase="data_reuse_required_failed",
                metadata={
                    "source_run_id": reuse_source_run_paths.run_id if reuse_source_run_paths is not None else None,
                    "target_run_id": run_paths.run_id,
                    "reuse_reason": reuse_reason,
                },
            )
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
            "coco_json": str(reused_data.coco_json) if reused_data is not None else "",
            "reuse_synthetic_data": bool(reused_data and reused_data.synthetic_reused),
            "image1": composite_image1,
            "image2": composite_image2,
            "task": task_description,
            "generation_prompt": prompt_text,
            "labels": labels,
            "intent_items": intent_items,
            "annotation_prompts": annotation_prompts,
            "annotation_prompt_map": annotation_prompt_map,
            "annotation_provider": annotation_provider,
            "work_dir": pipeline_work_dir,
            "output_dir": data_prep_output_dir,
            "skip_generation": bool(reused_data and reused_data.synthetic_reused) or not generation_enabled,
            "generation_skip_reason": "reused" if reused_data and reused_data.synthetic_reused else generation_skip_reason,
            "max_synthetic": _max_synthetic_images(runtime_options),
            "synthetic_count_button": True,
            "produce_count_button": True,
            "register_artifacts": not training_enabled,
            "split_requested": training_enabled,
            "progress_state_path": str((workflow_output_root / "progress_state.json").resolve()),
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
                "intent_items": intent_items,
                "annotation_prompts": annotation_prompts,
                "annotation_prompt_map": annotation_prompt_map,
                "annotation_provider": annotation_provider,
                "skip_generation": bool(reused_data and reused_data.synthetic_reused) or not generation_enabled,
                "generation_skip_reason": "reused" if reused_data and reused_data.synthetic_reused else generation_skip_reason,
                "split_requested": training_enabled,
                "work_dir": pipeline_work_dir,
                "output_dir": data_prep_output_dir,
                "run_name": run_name,
                "phase": "data_preparation",
                "progress_state_path": str((workflow_output_root / "progress_state.json").resolve()),
                "max_synthetic": _max_synthetic_images(runtime_options),
                "synthetic_count_button": True,
                "produce_count_button": True,
                "register_artifacts": not training_enabled,
            },
        }

        recorder.emit(
            "workflow.request_spec.resolved",
            {
                "labels": labels,
                "intent_items": intent_items,
                "annotation_prompts": annotation_prompts,
                "annotation_prompt_map": annotation_prompt_map,
                "annotation_provider": annotation_provider,
                "generation_prompt": prompt_text,
                "task_description": task_description,
                "max_synthetic_images": _max_synthetic_images(runtime_options),
                "training_backend": training_backend,
                "training_config": training_cfg,
            },
        )

        recorder.emit("skill.started", {"skill_name": "data-auto-annotation", "attempt": 0})
        annotation_result = self.skill_runner.run("data-auto-annotation", data_prep_spec, paths, on_event=recorder.emit)
        recorder.emit("skill.completed", {"skill_name": "data-auto-annotation", "output_count": len(annotation_result.outputs)})

        data_prep_data = annotation_result.data if isinstance(annotation_result.data, dict) else {}
        data_prep_returncode = int(data_prep_data.get("returncode", 0) or 0)
        if data_prep_returncode != 0:
            outputs = [*annotation_result.outputs]
            for artifact in outputs:
                recorder.emit("artifact.created", {"artifact": artifact.model_dump()})
            stderr_tail = str(data_prep_data.get("stderr") or "").strip()
            stdout_tail = str(data_prep_data.get("stdout") or "").strip()
            failure_reason_lines = _data_preparation_failure_reason_lines(stderr_tail, stdout_tail)
            failure_reason_text = "\n".join(f"- {line}" for line in failure_reason_lines)
            failure_reason_block = f"{failure_reason_text}\n" if failure_reason_text else ""
            reply = (
                "数据处理流程失败，尚未进入生图和流式标注阶段。\n\n"
                f"- 失败阶段：`data-auto-annotation`\n"
                f"- returncode：`{data_prep_returncode}`\n"
                f"{failure_reason_block}"
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
                    "run_id": run_paths.run_id,
                    "run_output_dir": str(run_paths.outputs),
                    "phase": "data_preparation_failed",
                    "data_preparation_spec": data_prep_spec,
                    "data_preparation_result": data_prep_data,
                    "labels": labels,
                },
            )
            recorder.emit("agent.message", {"text": reply})
            recorder.emit("run.failed", {"result": result.model_dump(), "error": stderr_tail or stdout_tail})
            return result, recorder.events

        if reused_data is not None:
            _mark_reused_data_preparation_summary(run_paths, reused_data)
        _write_data_preparation_reuse_manifest(
            run_paths,
            fingerprint=reuse_fingerprint,
            source_run_id=reused_data.source_run_id if reused_data is not None else "",
        )

        dataset_yaml = _resolve_prepared_dataset_yaml(data_prep_data, Path(data_prep_output_dir))
        training_skill_name = _training_skill_for_backend(training_backend)

        if not training_enabled:
            _set_waiting_prompt(run_paths, False)
            _set_workflow_completed(run_paths, True)
            outputs = [*annotation_result.outputs]
            for artifact in outputs:
                recorder.emit("artifact.created", {"artifact": artifact.model_dump()})
            pipeline_paths = _read_pipeline_paths(run_paths)
            summary = _read_data_preparation_summary(run_paths)
            reply = (
                f"数据处理流程已完成，当前未选择 `{training_skill_name}`，所以不会要求训练参数，也不会启动训练。\n\n"
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
                    "run_id": run_paths.run_id,
                    "run_output_dir": str(run_paths.outputs),
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

        if training_backend == "deimv2":
            prep_summary = _read_data_preparation_summary(run_paths)
            training_coco = str(prep_summary.get("training_coco") or "")
            training_root = str(prep_summary.get("training_root") or "")
            prepared_class_names = [
                str(item)
                for item in (prep_summary.get("class_names") or prep_summary.get("labels_final") or labels)
                if str(item).strip()
            ]
            if prepared_class_names:
                labels = prepared_class_names
            if not training_coco or not training_root:
                return self._model_spec_failed_result(
                    recorder,
                    agent_config.name,
                    paths.thread_id,
                    workflow,
                    "数据准备结果缺少 training_coco 或 training_root，无法启动 DEIMv2 训练。",
                    phase="deimv2_training_input_missing",
                    metadata={"data_preparation_summary": prep_summary},
                )
            training_spec = {
                "skill_name": training_skill_name,
                "overrides_text": user_text,
                "deimv2_root": "",
                "dataset": {
                    "root_dir": training_root,
                    "coco_json": training_coco,
                    "class_names": labels,
                    "split": training_cfg["split"],
                },
                "project_dir": project_dir,
                "training": training_cfg["training"],
                "runtime": _current_runtime_config(training_cfg.get("runtime", {})),
                "workflow_context": {
                    "training_backend": "deimv2",
                    "dataset_root": training_root,
                    "coco_json": training_coco,
                    "project_dir": project_dir,
                    "run_name": run_name,
                    "phase": "training",
                    "deimv2_model_selection": deimv2_model_selection,
                    "model_generated_spec": _model_generated_spec_payload(request_spec),
                },
            }
        else:
            training_spec = {
                "skill_name": training_skill_name,
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

        if _http_training_cancel_requested(paths, run_paths):
            return _cancelled_training_result(
                recorder=recorder,
                agent_name=agent_config.name,
                thread_id=paths.thread_id,
                workflow=workflow,
                run_paths=run_paths,
                phase="cancelled",
                reply="训练任务已收到取消请求，已在启动模型训练前停止。",
                artifacts=[*annotation_result.outputs],
                metadata={
                    "data_preparation_spec": data_prep_spec,
                    "training_spec": training_spec,
                    "labels": labels,
                },
            )

        recorder.emit("skill.started", {"skill_name": training_skill_name, "attempt": 0})
        training_result = self.skill_runner.run(training_skill_name, training_spec, paths)
        recorder.emit("skill.completed", {"skill_name": training_skill_name, "output_count": len(training_result.outputs)})

        outputs = _filter_training_run_artifacts(training_result.outputs, training_backend=training_backend)
        for artifact in outputs:
            recorder.emit("artifact.created", {"artifact": artifact.model_dump()})

        pipeline_paths = _read_pipeline_paths(run_paths)
        summary = _read_run_summary(run_paths, training_backend=training_backend)
        best_pt = _find_best_checkpoint(run_paths, training_backend=training_backend)
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
            _set_waiting_prompt(run_paths, False)
            _set_workflow_completed(run_paths, False)
            reply = _training_failed_reply(
                summary=summary,
                fallback_reply=template_reply,
                best_pt=best_pt,
                data_preparation_summary=_read_data_preparation_summary(run_paths),
                training_backend=training_backend,
            )
            result = AgentRunResult(
                agent=agent_config.name,
                thread_id=paths.thread_id,
                status="failed",
                reply=reply,
                artifacts=outputs,
                verification=VerificationResult(passed=False, retry_count=0, checks=[], failed_checks=[f"{training_skill_name} failed"]),
                metadata={
                    "workflow": workflow,
                    "run_id": run_paths.run_id,
                    "run_output_dir": str(run_paths.outputs),
                    "phase": "training_failed",
                    "training_spec": training_spec,
                    "data_preparation_spec": data_prep_spec,
                    "training_summary": summary,
                    "training_result": training_result.data if isinstance(training_result.data, dict) else {},
                    "best_pt": best_pt,
                    "best_checkpoint": best_pt,
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
                f"- best checkpoint: {best_pt if best_pt else '未生成'}"
            )
        _set_waiting_prompt(run_paths, False)
        _set_workflow_completed(run_paths, True)

        result = AgentRunResult(
            agent=agent_config.name,
            thread_id=paths.thread_id,
            status="completed",
            reply=reply,
            artifacts=outputs,
            verification=VerificationResult(passed=True, retry_count=0, checks=[], failed_checks=[]),
            metadata={
                "workflow": workflow,
                "run_id": run_paths.run_id,
                "run_output_dir": str(run_paths.outputs),
                "phase": "training_completed",
                "training_spec": training_spec,
                "data_preparation_spec": data_prep_spec,
                "training_summary": summary,
                "best_pt": best_pt,
                "best_checkpoint": best_pt,
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
        failure_reason = "当前应用没有可用的大模型配置，或 agent/app/runtimeOptions 中缺少 `base_url`、`model` 等必要配置。"
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_final_reply",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": failure_reason,
            },
        )
        return _human_fallback_reply(
            fallback_reply,
            summary,
            best_pt,
            summary_failure_reason=failure_reason,
            llm_failures=_llm_failure_summaries(recorder),
        )

    evaluation = _extract_evaluation_facts(summary)
    llm_failures_before_final = _llm_failure_summaries(recorder)
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
        "llm_call_failures": llm_failures_before_final,
        "raw_training_reply_for_reference": fallback_reply,
    }
    system_prompt = (
        "你是算法工程师 Agent 的最终回复生成器。"
        "上游工作流已经完成技能调用，你只负责基于事实组织输出样式和表达。"
        "要求：使用中文；不要编造事实；保留关键路径、最佳模型权重、数据划分、类别和评估指标；"
        "合成数据状态必须以 dataset.synthetic_generation.status 为准；"
        "如果 status=merged 且 fallback 存在，说明主合成接口失败但 fallback 已成功补救，不要写成数据合成失败；"
        "如果 evaluation.metrics 中存在 precision、recall、mAP50、mAP50_95、fitness、mAP75、AR100、best_coco_eval_bbox、best_epoch，必须在回复中明确列出；"
        "DEIMv2 使用 COCO AP/AR 指标，precision 可能没有日志输出；不要因为 precision 不存在就说指标为空。"
        "如果 llm_call_failures 非空，必须在回复中增加“大模型调用告警”，逐条说明失败阶段和原因；"
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
        failure_reason = _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc)))
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_final_reply",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": failure_reason,
            },
        )
        return _human_fallback_reply(
            fallback_reply,
            summary,
            best_pt,
            summary_failure_reason=failure_reason,
            llm_failures=_llm_failure_summaries(recorder),
        )
    recorder.emit(
        "llm.completed",
        {
            "purpose": "workflow_final_reply",
            "used_fallback": not bool(reply),
            "reply_chars": len(reply),
            **({"failure_reason": "大模型总结接口返回空内容。"} if not reply else {}),
        },
    )
    return reply or _human_fallback_reply(
        fallback_reply,
        summary,
        best_pt,
        summary_failure_reason="大模型总结接口返回空内容。",
        llm_failures=_llm_failure_summaries(recorder),
    )


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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_intent_router",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_intent_router",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
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
    failure_reason = ""
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
            failure_reason = _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc)))
            recorder.emit(
                "llm.completed",
                {
                    "purpose": "workflow_chat_reply",
                    "used_fallback": True,
                    "error": str(exc)[:1000],
                    "failure_reason": failure_reason,
                },
            )
    else:
        failure_reason = _llm_not_configured_failure_reason()
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_chat_reply",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": failure_reason,
            },
        )
    if failure_reason:
        return f"普通聊天回复的大模型调用失败：{failure_reason}"
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
    text = (raw or "").strip().lstrip("\ufeff")
    if not text:
        raise ValueError("LLM returned empty content")

    candidates = _dedupe_text_values([
        text,
        _strip_markdown_json_fence(text),
        _extract_first_json_object(text),
    ])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if not isinstance(payload, dict):
                raise ValueError("LLM JSON was not an object")
            return payload
        except Exception as exc:
            last_error = exc
    raise ValueError(f"LLM did not return a valid JSON object: {last_error}")


def _strip_markdown_json_fence(text: str) -> str:
    value = (text or "").strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_first_json_object(text: str) -> str:
    value = text or ""
    start = value.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return ""


def _write_model_intent_spec_raw_log(
    workflow_output_root: Path | None,
    *,
    purpose: str,
    system_prompt: str,
    messages: list[Message],
    raw_response: str,
    parsed_spec: dict[str, Any],
    parse_error: str = "",
) -> None:
    if workflow_output_root is None:
        return
    log_dir = workflow_output_root / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "purpose": purpose,
            "system_prompt": system_prompt,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "raw_response": raw_response,
            "parse_error": parse_error,
            "parsed_spec": parsed_spec,
            "generated_keys": sorted(str(key) for key in parsed_spec.keys()),
        }
        (log_dir / "model-intent-spec-raw.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except Exception:
        return


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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_request_spec",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_request_spec",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
        return {}


def _intent_semantic_planning_contract(backend_name: str) -> str:
    return (
        f"你是 {backend_name} 目标检测训练工作流的前置语义规划器。"
        "先理解用户最终想检测的业务目标，再把目标拆分为基础实体、属性、行为、状态、辅助实体和实体关系。"
        "此阶段尚未分析数据集，禁止输出 batch、epochs、img_size、imgsz、split 等训练参数。"
        "只返回 JSON 对象，不要解释，也不要输出思维过程。"
        "顶层字段包含 task_description、task_type、intent_items、use_synthetic_generation、generation_prompt、labels、training、runtime、split。"
        "intent_items 中每个业务目标必须包含："
        "label、type、subject、description、business_label、training_labels、bbox_target、base_entity、"
        "required_attributes、required_actions、required_states、required_relations、auxiliary_entities、observable_entities、"
        "annotation_strategy、primary_sam3_prompt、sam3_prompts、sam3_prompt_map、forbidden_direct_prompts、confidence、assumption。"
        "type 使用 object_detection、person_attribute_detection、behavior_detection、state_detection、entity_interaction 或 anomaly_detection。"
        "label 是用户业务目标的英文 ASCII 规范类名，business_label 保留用户原始业务名称；"
        "training_labels 是真正写入 COCO 并参与训练的英文 ASCII 类别。"
        "primary_sam3_prompt 必须是一个简短英文可视短语，描述需要被边界框框住的目标，并保留用户要求的全部属性、行为或状态约束。"
        "只把用户明确要求或识别该目标所需的最小视觉条件放入 required_* 和 primary_sam3_prompt；"
        "帽子、臂章、工具等可能出现但非必需的线索只能放入 auxiliary_entities，禁止使用 and 把可选线索变成强制条件。"
        "sam3_prompt_map 的键是实际发送给 SAM3 的 prompt，值是该 prompt 对应的 training_label。"
        "observable_entities 和 base_entity 只用于语义分析，绝对不能因为它们可见就自动加入 sam3_prompts。"
        "如果目标带属性、行为、状态或关系约束，annotation_strategy 使用 constrained_target，"
        "sam3_prompts 只包含 primary_sam3_prompt，不得加入单独的 person、vehicle 或其他宽泛基础实体；"
        "forbidden_direct_prompts 必须列出会丢失业务约束的宽泛 prompt。"
        "如果用户要训练的是纯实体类别，annotation_strategy 使用 direct_entity，prompt 使用该实体的简短英文名称。"
        "如果任务需要分别训练多个可见实体，annotation_strategy 使用 entity_interaction，training_labels 列出各实体，"
        "sam3_prompt_map 为每个训练类别分别提供一个实体 prompt；不要把多个 prompt 映射到同一个受约束业务类别。"
        "如果概念无法从单帧图像稳定判断，annotation_strategy 使用 requires_review，并在 assumption 中说明视觉假设，"
        "但仍应给出最贴近用户目标的可见训练类别，禁止退化为 object、target、thing 或 custom_entity 等占位类别。"
        "labels 必须等于所有 training_labels 的去重合集，只能包含小写英文、数字和下划线。"
        "training、runtime、split 返回空对象。"
    )


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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_model_managed_intent_spec",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
        return {}
    system_prompt = (
        _intent_semantic_planning_contract("YOLO")
        +
        "use_synthetic_generation 布尔值，表示是否请求合成数据；"
        "generation_prompt 字符串；"
        "输入设定：dataset.zip 必须上传，image1.zip 和 image2.zip 是可选但必须成对提供的合成输入；"
        "缺少任意一个合成输入时工作流会跳过合成并只使用真实数据训练；"
        "image1.zip 固定是场景/背景文件夹，image2.zip 固定是目标物文件夹；"
        "合成目的必须是把 image2 中目标自然合成到 image1 场景中，形成真实、可标注的训练图片。"
        "generation_prompt 必须包含上述 image1/image2 角色、自然融合、光照/尺度/遮挡一致、适合检测标注等要求。"
        "用户包含多个检测目标时必须全部解析，不能只保留第一个。"
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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_model_managed_intent_spec",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
        return {}


def _generate_model_managed_training_intent_spec(
    *,
    user_text: str,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
    training_backend: str,
    deimv2_model_selection: dict[str, Any] | None = None,
    workflow_output_root: Path | None = None,
) -> dict[str, Any]:
    if training_backend != "deimv2":
        return _generate_model_managed_yolo_training_intent_spec(
            user_text=user_text,
            agent_config=agent_config,
            runtime_options=runtime_options,
            recorder=recorder,
        )
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_deimv2_model_managed_intent_spec"})
    if not llm.configured:
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_deimv2_model_managed_intent_spec",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
        return {}
    system_prompt = (
        _intent_semantic_planning_contract("DEIMv2 DINOv3")
        +
        "use_synthetic_generation 布尔值，表示是否请求合成数据；"
        "generation_prompt 字符串。用户包含多个检测目标时必须全部解析。"
        "输入设定：datasets.zip 必须上传，image1.zip 和 image2.zip 是可选但必须成对提供的合成输入；"
        "缺少任意一个合成输入时工作流会跳过合成并只使用真实数据训练；"
        "image1.zip 固定是参考图/场景背景文件夹，image2.zip 固定是目标图/前景目标文件夹；"
        "generation_prompt 必须明确写出：使用 image1.zip 作为参考图/场景背景，使用 image2.zip 作为目标图/前景目标，"
        "把 image2.zip 中的目标自然合成到 image1.zip 参考图的场景中；"
        "同时必须要求光照、尺度、透视、遮挡、阴影一致，且适合后续 SAM3/COCO 边界框标注。"
    )
    messages = [
        Message(role="user", content=f"用户业务目标：{user_text}\n请输出 DEIMv2 自动训练托管规格。")
    ]
    raw = ""
    parse_error = ""
    spec: dict[str, Any] = {}
    try:
        raw = llm.complete_sync(system_prompt, messages)
        try:
            payload = _parse_json_object(raw)
            spec = payload if isinstance(payload, dict) else {}
        except Exception as exc:
            parse_error = str(exc)
            spec = {}
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_deimv2_model_managed_intent_spec",
                "used_fallback": False,
                "generated_keys": sorted(str(key) for key in spec.keys()),
                **({"parse_error": parse_error[:1000]} if parse_error else {}),
            },
        )
        _write_model_intent_spec_raw_log(
            workflow_output_root,
            purpose="workflow_deimv2_model_managed_intent_spec",
            system_prompt=system_prompt,
            messages=messages,
            raw_response=raw,
            parsed_spec=spec,
            parse_error=parse_error,
        )
        return spec
    except Exception as exc:
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_deimv2_model_managed_intent_spec",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
        _write_model_intent_spec_raw_log(
            workflow_output_root,
            purpose="workflow_deimv2_model_managed_intent_spec",
            system_prompt=system_prompt,
            messages=messages,
            raw_response=raw,
            parsed_spec=spec,
            parse_error=str(exc),
        )
        return {}


def _generate_dataset_aware_training_request_spec(
    *,
    base_spec: dict[str, Any],
    dataset_facts: dict[str, Any],
    user_text: str,
    agent_config: Any,
    runtime_options: RuntimeOptions,
    recorder: EventRecorder,
    training_backend: str,
    deimv2_model_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if training_backend != "deimv2":
        return _generate_dataset_aware_yolo_training_request_spec(
            base_spec=base_spec,
            dataset_facts=dataset_facts,
            user_text=user_text,
            agent_config=agent_config,
            runtime_options=runtime_options,
            recorder=recorder,
        )
    llm = OpenAICompatibleClient(agent_config, runtime_options=runtime_options)
    recorder.emit("llm.started", {"model": llm.model, "configured": llm.configured, "purpose": "workflow_deimv2_dataset_aware_training_spec"})
    if not llm.configured:
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_deimv2_dataset_aware_training_spec",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
        return base_spec
    readme_excerpt = _deimv2_readme_for_prompt()
    model_selection = deimv2_model_selection or _default_deimv2_model_selection()
    # DEIMv2 的 train.yml 最终由 runner 生成，但模型思考训练参数时
    # 必须参考真实 vendor 模板及其递归 __include__ 文件，
    # 避免生成的 epochs/batch/img_size 脱离实际模型配置。
    vendor_config_context = _deimv2_vendor_config_context_for_prompt(model_selection)
    model_variant = str(model_selection.get("effective_model_variant") or DEIMV2_DEFAULT_MODEL_VARIANT)
    template_config = str(model_selection.get("template_config") or DEIMV2_MODEL_VARIANTS[DEIMV2_DEFAULT_MODEL_VARIANT]["template_config"])
    tuning_checkpoint = str(model_selection.get("tuning_checkpoint") or "")
    system_prompt = (
        "你是资深计算机视觉算法工程师。现在要为 DEIMv2 DINOv3 目标检测训练生成可执行训练规格，"
        "必须基于 dataset_facts、用户业务目标、DEIMv2 README 和 vendor/deimv2/configs 下的真实配置。只返回 JSON 对象，不要解释。"
        "必须包含字段：task_description, task_type, intent_items, use_synthetic_generation, generation_prompt, labels, training, runtime, split。"
        f"training 必须含 model_variant='{model_variant}', template_config, epochs, img_size, batch, device, workers, "
        "backbone_checkpoint, tuning_checkpoint。"
        f"template_config 必须使用 {template_config}；"
        f"backbone_checkpoint 默认 {DEIMV2_BACKBONE_CHECKPOINT}；tuning_checkpoint 使用 {tuning_checkpoint or '空字符串'}；"
        "device 必须为 auto，表示插件按 CUDA GPU、NPU、CPU 顺序选择；runtime.enforce_conda_env=false。"
        "split 必须含 train/val/test，合成图只能进入 train，val/test 必须用真实图。"
        "batch 表示 train_dataloader.total_batch_size，必须结合 vendor 配置、数据量、img_size、类别数、目标大小和硬件选择；"
        "CUDA 且 300 张以上图片不应机械固定为 1 或 2，可优先考虑 4 或 8；CPU/NPU 可更保守。"
        "epochs 必须根据数据量、类别数、目标大小、用户是否明确要求快速测试来决定。"
        f"epochs 最大不能超过 {MAX_TRAINING_EPOCHS}；即使用户目标较复杂，也必须在该上限内选择合理轮数。"
        "如果用户没有明确指定训练轮数，不要机械使用 10 epoch；小数据集正式训练通常至少 50 epoch。"
        "img_size 通常 640。"
        "如果 base_spec 或用户目标表达的是行为/状态检测，例如人员摔倒、人员玩手机、人员睡岗、人员攀爬，"
        "labels 不能退化为 person，必须保留 person_fall、person_use_phone、person_sleep、person_climb 这类业务状态类名；"
        "intent_items 必须保留这些业务意图，训练参数再基于数据集规模和模板配置推理。"
        "README 关键内容如下：\n"
        f"{readme_excerpt}\n\n"
        "vendor/deimv2/configs 关键配置如下，包含选中模板及其 __include__ 递归依赖；生成 training 时必须以这些真实字段和结构为约束：\n"
        f"{vendor_config_context}"
    )
    messages = [
        Message(
            role="user",
            content=json.dumps(
                {
                    "user_text": user_text,
                    "base_spec": base_spec,
                    "dataset_facts": dataset_facts,
                    "deimv2_model_selection": model_selection,
                    "requirements": [
                        "根据标注类别和合成图数量生成当前应用的 DEIMv2 dataset/training yml 所需规格。",
                        "training 字段必须能映射到 runner 生成的 configs/train.yml 与 configs/dataset.yml，不能臆造 DEIMv2 不支持的配置项。",
                        "优先依据 vendor config 的 total_batch_size、epoches、optimizer/lr 结构和 dataset loader 结构做推理。",
                        "不要改成 YOLO 参数；不要输出 YOLO model/best.pt 字段。",
                        "输出只要 JSON，不要 Markdown。",
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
        merged = _merge_deimv2_request_specs(base_spec, generated, override_nested=True)
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_deimv2_dataset_aware_training_spec",
                "used_fallback": False,
                "generated_keys": sorted(str(key) for key in generated.keys()),
                "dataset_image_count": dataset_facts.get("image_count"),
                "dataset_format": dataset_facts.get("format"),
            },
        )
        return _apply_deimv2_dataset_epoch_reasoning(merged, dataset_facts, user_text)
    except Exception as exc:
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_deimv2_dataset_aware_training_spec",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
        return _apply_deimv2_dataset_epoch_reasoning(base_spec, dataset_facts, user_text)


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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_dataset_aware_training_spec",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
        return base_spec
    system_prompt = (
        "你是资深计算机视觉算法工程师。现在必须基于 dataset_facts 分析结果，而不是只基于用户一句话，"
        "补全 YOLO 检测训练参数。只返回 JSON 对象，不要解释。"
        "必须包含字段：task_description, task_type, intent_items, use_synthetic_generation, generation_prompt, labels, training, runtime, split。"
        "training 必须含 task, model, epochs, imgsz, batch, device, workers, patience。"
        "runtime 必须含 enforce_conda_env=false；不要指定 conda_env_name，或置为空字符串。"
        "split 必须含 train, val, test，三项相加约等于 1。"
        "决策规则："
        "根据 image_count、format、label_count、category_counts、bbox_size_summary、image_size_summary 判断训练强度；"
        "小目标多时提高 imgsz；图片少或类别不均衡时增加 epochs/patience 并启用合成；"
        f"epochs 最大不能超过 {MAX_TRAINING_EPOCHS}；禁止输出超过该上限的训练轮数；"
        "val/test 必须优先使用真实图，合成图只能进入 train；"
        "如果数据规模很小，test 比例要保守，避免每类在验证集缺失；"
        "labels 必须是英文 ASCII 类名，使用小写英文、数字、下划线，禁止中文和泛化 object/target；"
        "复合任务必须保留所有类别，不要覆盖或丢弃 base_spec 中合理的 labels。"
        "如果 base_spec 或用户目标表达的是行为/状态检测，例如人员摔倒、人员玩手机、人员睡岗、人员攀爬，"
        "labels 不能退化为 person，必须保留 person_fall、person_use_phone、person_sleep、person_climb 这类业务状态类名。"
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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_dataset_aware_training_spec",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
        return base_spec


def _fallback_model_managed_yolo_training_request_spec(
    spec: dict[str, Any],
    user_text: str,
    dataset_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """模型规划失败时，保证上传数据的训练契约仍然确定。"""
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
        or _fallback_generation_prompt(
            labels,
            task_description=_spec_string(spec, "task_description") or f"{label_text} detection",
            user_text=user_text,
        ),
        "labels": labels,
        "training": fallback_training["training"],
        "runtime": {"conda_env_name": "", "enforce_conda_env": False},
        "split": fallback_training["split"],
    }
    return _apply_behavior_intent_guard(_merge_request_specs(fallback, spec), user_text)


def _fallback_model_managed_training_request_spec(
    spec: dict[str, Any],
    user_text: str,
    dataset_facts: dict[str, Any] | None = None,
    *,
    training_backend: str,
    deimv2_model_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if training_backend != "deimv2":
        return _fallback_model_managed_yolo_training_request_spec(spec, user_text, dataset_facts)
    spec = _apply_deimv2_dataset_epoch_reasoning(spec, dataset_facts or {}, user_text)
    if _request_spec_complete(spec):
        return spec
    labels = _spec_string_list(spec, "labels") or _infer_labels_from_training_intent(user_text) or ["object"]
    labels = _normalize_detection_labels(labels) or ["object"]
    fallback_training = _fallback_deimv2_training_from_dataset_facts(dataset_facts or {}, deimv2_model_selection)
    fallback = {
        "task_description": _spec_string(spec, "task_description") or f"{', '.join(labels)} detection",
        "use_synthetic_generation": True,
        "generation_prompt": _spec_string(spec, "generation_prompt")
        or _fallback_generation_prompt(
            labels,
            task_description=_spec_string(spec, "task_description") or f"{', '.join(labels)} detection",
            user_text=user_text,
        ),
        "labels": labels,
        "training": fallback_training["training"],
        "runtime": {"conda_env_name": "", "enforce_conda_env": False},
        "split": fallback_training["split"],
    }
    return _apply_behavior_intent_guard(_merge_deimv2_request_specs(fallback, spec), user_text)


def _apply_deimv2_dataset_epoch_reasoning(spec: dict[str, Any], dataset_facts: dict[str, Any] | None, user_text: str) -> dict[str, Any]:
    training = spec.get("training") if isinstance(spec.get("training"), dict) else {}
    if not _looks_like_deimv2_training(training):
        return spec
    training.pop("smoke_epochs", None)
    if not _user_mentions_epoch_count(user_text):
        desired_epochs = _deimv2_epochs_from_dataset_facts(dataset_facts or {})
        current_epochs = _coerce_int(training.get("epochs"))
        if current_epochs is None or current_epochs < desired_epochs:
            training["epochs"] = desired_epochs
    if not _user_mentions_batch_count(user_text):
        desired_batch = _deimv2_batch_from_dataset_facts(dataset_facts or {}, training)
        current_batch = _coerce_int(training.get("batch"))
        if current_batch is None or current_batch < desired_batch:
            training["batch"] = desired_batch
    return spec


def _user_mentions_batch_count(user_text: str) -> bool:
    text = str(user_text or "")
    return bool(re.search(r"(?i)(?:\bbatch(?:_size)?\b|批大小|批次大小|batch\s*=\s*\d+)", text))


def _user_mentions_epoch_count(user_text: str) -> bool:
    text = str(user_text or "")
    return bool(
        re.search(
            r"(?i)(?:\bepochs?\b|训练轮数|训练\s*\d+\s*轮|跑\s*\d+\s*(?:个)?\s*epochs?|\d+\s*(?:个)?\s*epochs?|\d+\s*轮)",
            text,
        )
    )


def _deimv2_epochs_from_dataset_facts(dataset_facts: dict[str, Any]) -> int:
    image_count = _safe_int(dataset_facts.get("image_count"))
    label_count = _safe_int(dataset_facts.get("label_count") or dataset_facts.get("num_categories"))
    bbox_summary = dataset_facts.get("bbox_size_summary") if isinstance(dataset_facts.get("bbox_size_summary"), dict) else {}
    small_ratio = _coerce_float(bbox_summary.get("small_ratio")) or 0.0
    if image_count <= 30:
        epochs = 50
    elif image_count <= 100:
        epochs = 60
    else:
        epochs = 80
    if label_count >= 4 or small_ratio >= 0.35:
        epochs += 20
    return min(120, epochs)


def _deimv2_batch_from_dataset_facts(dataset_facts: dict[str, Any], training: dict[str, Any] | None = None) -> int:
    training = training or {}
    device = str(training.get("device") or "auto").strip().lower()
    image_count = _safe_int(dataset_facts.get("image_count"))
    img_size = _coerce_int(training.get("img_size") or training.get("imgsz")) or 640
    bbox_summary = dataset_facts.get("bbox_size_summary") if isinstance(dataset_facts.get("bbox_size_summary"), dict) else {}
    small_ratio = _coerce_float(bbox_summary.get("small_ratio")) or 0.0
    if device == "cpu":
        return 1
    if device in {"npu", "ascend"}:
        if image_count <= 100 or img_size >= 960:
            return 2
        return 4
    max_batch = 8 if device in {"cuda", "gpu", "0"} else 4
    if image_count <= 30:
        batch = 1
    elif image_count <= 100:
        batch = 2
    elif image_count <= 300:
        batch = 4
    else:
        batch = 8
    if img_size >= 960 or small_ratio >= 0.45:
        batch = max(1, batch // 2)
    return max(1, min(batch, max_batch))


def _merge_deimv2_request_specs(base: dict[str, Any], generated: dict[str, Any], *, override_nested: bool = False) -> dict[str, Any]:
    merged = dict(base)
    for key in ("task_description", "generation_prompt", "task_type"):
        if not _spec_string(merged, key) and _spec_string(generated, key):
            merged[key] = _spec_string(generated, key)
    if not _spec_intent_items(merged) and _spec_intent_items(generated):
        merged["intent_items"] = _spec_intent_items(generated)
    if not _spec_string_list(merged, "annotation_prompts") and _spec_string_list(generated, "annotation_prompts"):
        merged["annotation_prompts"] = _spec_string_list(generated, "annotation_prompts")
    if _spec_optional_bool(merged, "use_synthetic_generation") is None:
        generated_bool = _spec_optional_bool(generated, "use_synthetic_generation")
        if generated_bool is not None:
            merged["use_synthetic_generation"] = generated_bool
    if not _spec_string_list(merged, "labels"):
        labels = _spec_string_list(generated, "labels")
        if labels:
            merged["labels"] = labels
    for key in ("training", "runtime", "split"):
        current = merged.get(key)
        generated_value = generated.get(key)
        if not isinstance(generated_value, dict):
            continue
        if override_nested and isinstance(current, dict):
            nested = dict(current)
            for nested_key, nested_value in generated_value.items():
                if nested_value not in (None, "", [], {}):
                    nested[nested_key] = nested_value
            merged[key] = nested
        elif (not isinstance(current, dict) or not current):
            merged[key] = dict(generated_value)
    return merged


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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_request_spec_completion",
                "used_fallback": True,
                "reason": "not_configured",
                "failure_reason": _llm_not_configured_failure_reason(),
            },
        )
        return spec
    system_prompt = (
        "你是资深计算机视觉算法工程师，负责为 YOLO 目标检测训练工作流补全缺失参数。"
        "用户可能只给一句业务目标，你需要给出合理、可执行、保守的默认规格。"
        "只返回 JSON 对象，不要解释。字段："
        "task_description 字符串；task_type 字符串；intent_items 数组；use_synthetic_generation 布尔值；generation_prompt 字符串；"
        "labels 字符串数组；training 对象，含 task, model, epochs, imgsz, batch, device, workers, patience；"
        "runtime 对象，含 enforce_conda_env；不要输出 conda_env_name，或将 conda_env_name 置为空字符串；"
        "split 对象，含 train, val, test。"
        "原则：不要覆盖 existing_spec 里已经有的非空值；labels 必须跟随用户目标变化，不要固定套用示例类别；"
        "labels 必须是 YOLO 训练可直接使用的英文 ASCII 类名，只能使用小写英文、数字和下划线，"
        "禁止输出中文、空格或自然语言短语；例如人脸检测输出 face，不要输出 人脸；"
        "严禁返回 object、target、thing、foreground、目标、物体、对象 等泛化类别；"
        "必须从用户业务目标里解析具体对象作为 labels，例如：车辆检测输出 car，瓶子检测输出 bottle，钢材检测输出 steel；"
        "复合任务必须提取全部目标类别，例如“抽烟检测和人脸检测”输出 person,cigarette,face；"
        "行为/状态检测不能降级为 person，例如人员摔倒输出 person_fall，人员玩手机输出 person_use_phone；"
        "只有任务是抽烟检测时 labels 才优先包含 person 和 cigarette；车辆检测应输出车辆相关类别；"
        "合成提示词要适合 image2 目标自然合成到 image1 场景，并强调真实监控画面、可标注；"
        "训练参数必须由你根据任务目标、YOLO 训练常识和快速验证需求自行选择，不要照抄用户未提供的固定模板；"
        f"epochs 最大不能超过 {MAX_TRAINING_EPOCHS}；禁止输出超过该上限的训练轮数；"
        "训练必须使用当前运行环境，不要推理或指定 Conda 环境；runtime.enforce_conda_env 必须为 false；"
        "split 比例、模型大小、epochs、batch、patience 也必须由你合理选择。"
        "如果用户提供了完整的 image1/image2 配对或上下文明确要求合成数据，use_synthetic_generation 必须为 true。"
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
        recorder.emit(
            "llm.completed",
            {
                "purpose": "workflow_request_spec_completion",
                "used_fallback": True,
                "error": str(exc)[:1000],
                "failure_reason": _strip_failure_reason_prefix(_classify_llm_error_reason(str(exc))),
            },
        )
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
    for key in ("task_description", "generation_prompt", "task_type"):
        if not _spec_string(merged, key) and _spec_string(generated, key):
            merged[key] = _spec_string(generated, key)
    if not _spec_intent_items(merged) and _spec_intent_items(generated):
        merged["intent_items"] = _spec_intent_items(generated)
    if not _spec_string_list(merged, "annotation_prompts") and _spec_string_list(generated, "annotation_prompts"):
        merged["annotation_prompts"] = _spec_string_list(generated, "annotation_prompts")
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
                f"task_type={payload.get('task_type')}; "
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
        f"task_type: {payload.get('task_type') or ''}",
        "intent_items:",
        json.dumps(payload.get("intent_items") or [], ensure_ascii=False, indent=2),
        "",
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
    title = "模型思考生成的 DEIMv2 训练参数" if _looks_like_deimv2_training(training_cfg.get("training", {})) else "模型思考生成的 YOLO 训练参数"
    intent_items = _spec_intent_items(spec)
    annotation_prompt_map = _annotation_prompt_map_from_spec(spec, _spec_string_list(spec, "labels"))
    annotation_prompts = _dedupe_text_values([
        *_annotation_prompts_from_intent_items(intent_items),
        *_spec_string_list(spec, "annotation_prompts"),
        *annotation_prompt_map.keys(),
    ])
    return {
        "title": title,
        "task_description": _spec_string(spec, "task_description"),
        "task_type": _spec_string(spec, "task_type"),
        "intent_items": intent_items,
        "annotation_prompts": annotation_prompts,
        "annotation_prompt_map": annotation_prompt_map,
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
    if isinstance(app_params, dict):
        return app_params
    deimv2_app_params = params.get("algorithm-engineer-full-cycle-deimv2-test")
    return deimv2_app_params if isinstance(deimv2_app_params, dict) else {}


def _default_annotation_provider() -> str:
    return _normalize_annotation_provider(os.getenv("ANNOTATION_PROVIDER", DEFAULT_ANNOTATION_PROVIDER))


def _normalize_annotation_provider(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"locate", "locateanything", "locate_anything", "locate_sam3"}:
        return "locate_sam3"
    return "sam3"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deimv2_vendor_root() -> Path:
    return _project_root() / "plugins" / "skills" / "deimv2-auto-training" / "vendor" / "deimv2"


def _normalize_deimv2_model_variant(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return DEIMV2_DEFAULT_MODEL_VARIANT
    normalized = raw.replace("_", "-").replace(" ", "")
    normalized = re.sub(r"-coco$", "", normalized)
    if normalized in {"s", "m", "l", "x"}:
        return f"deimv2-dinov3-{normalized}"
    if normalized in {"dinov3-s", "dinov3-m", "dinov3-l", "dinov3-x"}:
        return f"deimv2-{normalized}"
    return normalized


def _requested_deimv2_model_variant(runtime_options: RuntimeOptions) -> str:
    direct = (
        getattr(runtime_options, "deimv2_model_variant", None)
        or getattr(runtime_options, "deimv2ModelVariant", None)
    )
    if direct:
        return str(direct).strip()
    config_options = getattr(runtime_options, "config_options", None)
    if isinstance(config_options, dict):
        for key in ("deimv2ModelVariant", "deimv2_model_variant"):
            value = str(config_options.get(key) or "").strip()
            if value:
                return value
    params = _workflow_skill_parameters(runtime_options)
    for key in ("deimv2ModelVariant", "deimv2_model_variant", "model_variant"):
        value = str(params.get(key) or "").strip()
        if value:
            return value
    return DEIMV2_DEFAULT_MODEL_VARIANT


def _deimv2_model_spec(model_variant: str) -> dict[str, str]:
    return dict(DEIMV2_MODEL_VARIANTS.get(model_variant) or DEIMV2_MODEL_VARIANTS[DEIMV2_DEFAULT_MODEL_VARIANT])


def _deimv2_checkpoint_exists(relative_path: str) -> bool:
    if not relative_path:
        return False
    requested = Path(relative_path)
    candidates = [requested] if requested.is_absolute() else [
        _project_root() / requested,
        _project_root() / "models" / requested.name,
        _deimv2_vendor_root() / "ckpts" / requested.name,
        Path("/models/deimv2") / requested.name,
    ]
    return any(path.is_file() for path in candidates)


def _deimv2_variant_unavailable_reason(model_variant: str) -> str:
    spec = DEIMV2_MODEL_VARIANTS.get(model_variant)
    if spec is None:
        allowed = ", ".join(sorted(DEIMV2_MODEL_VARIANTS))
        return f"unsupported model variant, allowed variants: {allowed}"
    template = _deimv2_vendor_root() / str(spec.get("template_config") or "")
    if not template.is_file():
        return f"template config not found: {template}"
    if model_variant != DEIMV2_DEFAULT_MODEL_VARIANT and not _deimv2_checkpoint_exists(str(spec.get("tuning_checkpoint") or "")):
        return f"tuning checkpoint not found: {spec.get('tuning_checkpoint')}"
    return ""


def _default_deimv2_model_selection(requested: str | None = None, fallback_reason: str = "") -> dict[str, Any]:
    spec = _deimv2_model_spec(DEIMV2_DEFAULT_MODEL_VARIANT)
    return {
        "requested_model_variant": requested or DEIMV2_DEFAULT_MODEL_VARIANT,
        "effective_model_variant": DEIMV2_DEFAULT_MODEL_VARIANT,
        "fallback_used": bool(fallback_reason),
        "fallback_reason": fallback_reason,
        "template_config": spec["template_config"],
        "backbone_checkpoint": DEIMV2_BACKBONE_CHECKPOINT,
        "tuning_checkpoint": spec["tuning_checkpoint"] if _deimv2_checkpoint_exists(spec["tuning_checkpoint"]) else "",
        "available_variants": sorted(DEIMV2_MODEL_VARIANTS),
    }


def _select_deimv2_model_variant(runtime_options: RuntimeOptions, recorder: EventRecorder) -> dict[str, Any]:
    requested_raw = _requested_deimv2_model_variant(runtime_options)
    requested = _normalize_deimv2_model_variant(requested_raw)
    reason = _deimv2_variant_unavailable_reason(requested)
    if reason:
        # 模型规格是面向用户的选择项。规格不存在或缺少 checkpoint 时记录原因，
        # 但回退到默认 S，保证专用 DEIMv2 应用仍可运行。
        selection = _default_deimv2_model_selection(requested=requested, fallback_reason=reason)
        recorder.emit("deimv2.model_variant.fallback", selection)
        return selection
    spec = _deimv2_model_spec(requested)
    selection = {
        "requested_model_variant": requested,
        "effective_model_variant": requested,
        "fallback_used": False,
        "fallback_reason": "",
        "template_config": spec["template_config"],
        "backbone_checkpoint": DEIMV2_BACKBONE_CHECKPOINT,
        "tuning_checkpoint": spec["tuning_checkpoint"] if _deimv2_checkpoint_exists(spec["tuning_checkpoint"]) else "",
        "available_variants": sorted(DEIMV2_MODEL_VARIANTS),
    }
    recorder.emit("deimv2.model_variant.selected", selection)
    return selection


def _apply_deimv2_model_selection_to_spec(spec: dict[str, Any], selection: dict[str, Any] | None) -> None:
    if not selection:
        return
    training = spec.get("training") if isinstance(spec.get("training"), dict) else {}
    spec["training"] = training
    training["model_variant"] = selection.get("effective_model_variant") or DEIMV2_DEFAULT_MODEL_VARIANT
    training["template_config"] = selection.get("template_config") or DEIMV2_MODEL_VARIANTS[DEIMV2_DEFAULT_MODEL_VARIANT]["template_config"]
    training["backbone_checkpoint"] = selection.get("backbone_checkpoint") or DEIMV2_BACKBONE_CHECKPOINT
    training["tuning_checkpoint"] = selection.get("tuning_checkpoint") or ""
    training["requested_model_variant"] = selection.get("requested_model_variant") or training["model_variant"]
    training["model_variant_fallback_reason"] = selection.get("fallback_reason") or ""


def _apply_deimv2_model_selection_to_training_config(training_cfg: dict[str, Any], selection: dict[str, Any] | None) -> None:
    if not training_cfg or not selection:
        return
    training = training_cfg.get("training") if isinstance(training_cfg.get("training"), dict) else {}
    if not training:
        return
    training["model_variant"] = selection.get("effective_model_variant") or DEIMV2_DEFAULT_MODEL_VARIANT
    training["template_config"] = selection.get("template_config") or DEIMV2_MODEL_VARIANTS[DEIMV2_DEFAULT_MODEL_VARIANT]["template_config"]
    training["backbone_checkpoint"] = selection.get("backbone_checkpoint") or DEIMV2_BACKBONE_CHECKPOINT
    training["tuning_checkpoint"] = selection.get("tuning_checkpoint") or ""
    training["requested_model_variant"] = selection.get("requested_model_variant") or training["model_variant"]
    training["model_variant_fallback_reason"] = selection.get("fallback_reason") or ""


def _apply_deimv2_export_onnx_policy(training_cfg: dict[str, Any]) -> None:
    training = training_cfg.get("training") if isinstance(training_cfg.get("training"), dict) else {}
    if not training:
        return
    # 后端按钮：为 True 时，DEIMv2 训练完成后自动导出 ONNX 并注册为产物。
    training["export_onnx"] = bool(button_export_onnx)


def _max_synthetic_images(runtime_options: RuntimeOptions) -> int:
    direct = getattr(runtime_options, "max_synthetic_images", None)
    if direct is not None:
        return min(_int_from_any(direct, DEFAULT_MAX_SYNTHETIC_IMAGES), DEFAULT_MAX_SYNTHETIC_IMAGES)
    params = _workflow_skill_parameters(runtime_options)
    requested = _int_from_any(
        params.get("maxSyntheticImages") or params.get("max_synthetic_images") or params.get("max_synthetic"),
        DEFAULT_MAX_SYNTHETIC_IMAGES,
    )
    return min(requested, DEFAULT_MAX_SYNTHETIC_IMAGES)


def _epochs_button(runtime_options: RuntimeOptions) -> bool:
    params = _workflow_skill_parameters(runtime_options)
    value = _bool_from_any(params.get("button_epochs"))
    return button_epochs if value is None else value


def _apply_epochs_policy(training_cfg: dict[str, Any], runtime_options: RuntimeOptions) -> None:
    training = training_cfg.get("training")
    if not isinstance(training, dict):
        return
    if not _epochs_button(runtime_options):
        # 开发/测试模式：无论模型思考出多少轮，都强制短轮数。
        # 生产环境应保持 button_epochs=True。
        training["epochs"] = FIXED_TRAINING_EPOCHS
    _cap_training_epochs(training)


def _cap_training_epochs(training: dict[str, Any]) -> None:
    for key in ("epochs", "epoches"):
        value = _coerce_int(training.get(key))
        if value is not None and value > MAX_TRAINING_EPOCHS:
            training[key] = MAX_TRAINING_EPOCHS


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
    "custom_detection_entity",
    "detection_entity",
    "target_entity",
    "custom_object",
    "custom_entity",
    "placeholder_entity",
    "unspecified_entity",
    "generic_entity",
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
    "socket": ("socket",),
    "power_socket": ("socket",),
    "插座": ("socket",),
    "电源插座": ("socket",),
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
    "volleyball": ("volleyball",),
    "排球": ("volleyball",),
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
    (("插座", "电源插座", "socket", "power socket", "power_socket"), ("socket",)),
    (("钢材", "钢筋", "钢板", "steel", "rebar"), ("steel",)),
    (("排球", "volleyball"), ("volleyball",)),
    (("安全帽", "helmet", "hard hat", "hard_hat"), ("hard_hat",)),
    (("口罩", "mask"), ("mask",)),
    (("火焰", "fire", "flame"), ("fire",)),
    (("人脸", "脸部", "面部", "human face", "face"), ("face",)),
    (("头部", "head"), ("head",)),
    (("手套", "glove"), ("glove",)),
    (("手机", "phone", "mobile phone"), ("phone",)),
    (("鱼竿", "钓鱼竿", "fishing rod", "fishing_rod"), ("fishing_rod",)),
    (("杯子", "cup"), ("cup",)),
    (("纸箱", "箱子", "box"), ("box",)),
    (("包裹", "package", "parcel"), ("package",)),
    (("袋子", "bag"), ("bag",)),
    (("刀具", "刀", "knife"), ("knife",)),
    (("反光衣", "安全背心", "safety vest", "safety_vest"), ("safety_vest",)),
    (("烟雾", "smoke"), ("smoke",)),
    (("行人", "人员", "person", "pedestrian"), ("person",)),
)


_PERSON_SUBJECT_LABELS = {"person", "people", "pedestrian", "human"}
_BEHAVIOR_AUXILIARY_LABELS = {"phone", "cigarette", "volleyball", "fishing_rod"}
_ENTITY_INTERACTION_TYPES = {"entity_interaction", "object_interaction", "person_object_interaction"}
_PERSON_ATTRIBUTE_TYPES = {"person_attribute_detection", "person_attribute", "appearance_detection", "wearing_detection"}
_CONSTRAINED_INTENT_TYPES = {
    "behavior_detection",
    "behavior",
    "behaviour",
    "action",
    "activity",
    "state_detection",
    "state",
    "status",
    "anomaly_detection",
    *_PERSON_ATTRIBUTE_TYPES,
}
_CONSTRAINT_FIELDS = (
    "required_attributes",
    "required_actions",
    "required_states",
    "required_relations",
)


_PERSON_ATTRIBUTE_RULES: tuple[dict[str, Any], ...] = (
    {
        "aliases": ("保安服", "保安制服", "安保服", "security uniform", "security guard uniform"),
        "attribute": "security_uniform",
        "phrase": "security uniform",
        "description": "检测画面中穿保安服的人",
    },
    {
        "aliases": ("制服", "工作服", "工服", "uniform", "work uniform", "workwear"),
        "attribute": "uniform",
        "phrase": "uniform",
        "description": "检测画面中穿制服或工作服的人",
    },
    {
        "aliases": ("反光衣", "安全背心", "safety vest", "reflective vest"),
        "attribute": "safety_vest",
        "phrase": "safety vest",
        "description": "检测画面中穿反光衣的人",
    },
    {
        "aliases": ("安全帽", "helmet", "hard hat", "hard_hat"),
        "attribute": "hard_hat",
        "phrase": "hard hat",
        "description": "检测画面中戴安全帽的人",
    },
    {
        "aliases": ("口罩", "mask", "face mask"),
        "attribute": "mask",
        "phrase": "face mask",
        "description": "检测画面中戴口罩的人",
    },
)


_BEHAVIOR_INTENT_RULES: tuple[dict[str, Any], ...] = (
    {
        "aliases": ("摔倒", "跌倒", "倒地", "fall down", "fallen person", "person fall", "person_fall"),
        "label": "person_fall",
        "type": "state_detection",
        "subject": "person",
        "behavior": "fall",
        "description": "检测画面中摔倒、跌倒或倒地的人员",
        "annotation_prompts": ("fallen person", "person falling", "person lying on ground", "person"),
    },
    {
        "aliases": ("玩手机", "看手机", "使用手机", "打电话", "接打电话", "using phone", "use phone", "phone use", "person_use_phone"),
        "label": "person_use_phone",
        "training_labels": ("person", "phone"),
        "type": "entity_interaction",
        "subject": "person",
        "behavior": "use_phone",
        "description": "检测画面中正在使用手机的人员",
        "annotation_prompts": ("person", "phone"),
    },
    {
        "aliases": ("抽烟", "吸烟", "smoking", "person smoking", "person_smoking"),
        "label": "person_smoking",
        "training_labels": ("person", "cigarette"),
        "type": "entity_interaction",
        "subject": "person",
        "behavior": "smoking",
        "description": "检测画面中正在抽烟的人员",
        "annotation_prompts": ("person", "cigarette"),
    },
    {
        "aliases": ("钓鱼", "垂钓", "fishing", "person fishing", "person_fishing"),
        "label": "person_fishing",
        "training_labels": ("person", "fishing_rod"),
        "type": "entity_interaction",
        "subject": "person",
        "behavior": "fishing",
        "description": "检测画面中正在钓鱼的人员",
        "annotation_prompts": ("person", "fishing rod"),
    },
    {
        "aliases": ("睡岗", "睡觉", "打瞌睡", "sleeping", "person sleeping", "person_sleep"),
        "label": "person_sleep",
        "type": "state_detection",
        "subject": "person",
        "behavior": "sleep",
        "description": "检测画面中睡岗或睡觉的人员",
        "annotation_prompts": ("sleeping person", "person sleeping", "person"),
    },
    {
        "aliases": ("跑步", "奔跑", "running", "person running", "person_running"),
        "label": "person_running",
        "type": "behavior_detection",
        "subject": "person",
        "behavior": "running",
        "description": "检测画面中正在跑步或奔跑的人员",
        "annotation_prompts": ("person running", "running person", "person"),
    },
    {
        "aliases": ("攀爬", "翻越", "爬墙", "climbing", "person climbing", "person_climb"),
        "label": "person_climb",
        "type": "behavior_detection",
        "subject": "person",
        "behavior": "climb",
        "description": "检测画面中攀爬或翻越的人员",
        "annotation_prompts": ("person climbing", "climbing person", "person"),
    },
    {
        "aliases": ("打架", "斗殴", "fight", "fighting", "person_fight"),
        "label": "person_fight",
        "type": "behavior_detection",
        "subject": "person",
        "behavior": "fight",
        "description": "检测画面中打架或斗殴的人员",
        "annotation_prompts": ("fighting person", "person fighting", "person"),
    },
)


def _ensure_intent_labels(spec: dict[str, Any], user_text: str) -> dict[str, Any]:
    merged = dict(spec or {})
    explicit_labels = _normalize_detection_labels(_extract_annotation_labels(user_text))
    if explicit_labels:
        merged["labels"] = explicit_labels
        merged = _apply_behavior_intent_guard(merged, user_text, preserve_explicit_labels=True)
        _align_spec_text_with_labels(merged, explicit_labels)
        return merged

    current_labels = _spec_string_list(merged, "labels")
    normalized_current = _normalize_detection_labels(current_labels)
    inferred_labels = _infer_labels_from_training_intent(user_text)
    if normalized_current and not _labels_are_generic(current_labels):
        # LLM 解析出的 labels 优先保留；规则只补全用户原文中非常明确、
        # 但模型漏掉的类别，避免复合任务被规则覆盖成单一任务。
        merged["labels"] = _dedupe_detection_labels([*normalized_current, *inferred_labels])
    elif inferred_labels:
        merged["labels"] = inferred_labels
    elif normalized_current:
        merged["labels"] = normalized_current
    merged = _apply_behavior_intent_guard(merged, user_text)
    merged = _apply_smoking_entity_annotation_policy(merged, user_text)
    _align_spec_text_with_labels(merged, _spec_string_list(merged, "labels"))
    return merged


_SMOKING_INTENT_TOKENS = (
    "抽烟",
    "吸烟",
    "smoking",
    "person_smoking",
    "smoking_person",
)


def _apply_smoking_entity_annotation_policy(spec: dict[str, Any], user_text: str) -> dict[str, Any]:
    # 抽烟检测在当前 SAM3 标注链路中按可见实体拆分：人和香烟分别标注。
    # 这样训练标签稳定为 person/cigarette，避免 LLM 返回 smoking_person 后
    # 使用行为短语标注，导致 person 漏标或只学到行为框。
    if not _has_smoking_intent(spec, user_text):
        return spec

    smoking_item = {
        "type": "entity_interaction",
        "subject": "person",
        "behavior": "smoking",
        "label": "person_smoking",
        "business_label": "person_smoking",
        "training_labels": ["person", "cigarette"],
        "business_labels": ["person_smoking"],
        "base_entity": "person",
        "bbox_target": "visible smoking-related entities",
        "annotation_strategy": "entity_interaction",
        "observable_entities": ["person", "cigarette"],
        "primary_sam3_prompt": "person",
        "sam3_prompts": ["person", "cigarette"],
        "annotation_prompts": ["person", "cigarette"],
        "sam3_prompt_map": {"person": "person", "cigarette": "cigarette"},
        "description": "检测画面中人员抽烟相关的可见实体",
        "source": "smoking_entity_policy",
    }
    kept_items = [
        item
        for item in _spec_intent_items(spec)
        if not _is_smoking_intent_item(item)
    ]
    items = _dedupe_intent_items([smoking_item, *kept_items])
    labels = _dedupe_detection_labels([
        "person",
        "cigarette",
        *_labels_from_intent_items(kept_items),
        *[
            label
            for label in _normalize_detection_labels(_spec_string_list(spec, "labels"))
            if not _is_smoking_label(label)
        ],
    ])

    merged = dict(spec or {})
    merged["labels"] = labels or ["person", "cigarette"]
    merged["intent_items"] = items
    merged["annotation_prompts"] = _annotation_prompts_from_intent_items(items) or ["person", "cigarette"]
    if not _spec_string(merged, "task_type") or _spec_string(merged, "task_type") in {"behavior_detection", "object_detection"}:
        merged["task_type"] = "entity_interaction"
    return merged


def _has_smoking_intent(spec: dict[str, Any], user_text: str) -> bool:
    text = str(user_text or "").lower()
    if any(token in text for token in _SMOKING_INTENT_TOKENS):
        return True
    labels = _normalize_detection_labels(_spec_string_list(spec, "labels"))
    if any(_is_smoking_label(label) for label in labels):
        return True
    return any(_is_smoking_intent_item(item) for item in _spec_intent_items(spec))


def _is_smoking_intent_item(item: dict[str, Any]) -> bool:
    values = _raw_string_values(
        item.get("label"),
        item.get("business_label"),
        item.get("behavior"),
        item.get("description"),
        item.get("business_labels"),
        item.get("sam3_prompts"),
        item.get("annotation_prompts"),
    )
    return any(_is_smoking_label(value) for value in values)


def _is_smoking_label(value: str) -> bool:
    key = _label_lookup_key(value)
    text = str(value or "").strip().lower()
    return key in {"smoking", "person_smoking", "smoking_person"} or any(
        token in text for token in _SMOKING_INTENT_TOKENS
    )


def _infer_labels_from_training_intent(user_text: str) -> list[str]:
    text = (user_text or "").strip()
    if not text:
        return []
    lowered = text.lower()
    inferred: list[str] = []
    inferred.extend(_labels_from_intent_items(_infer_behavior_intent_items(text)))
    for aliases, labels in _INTENT_LABEL_RULES:
        if any(alias.lower() in lowered for alias in aliases):
            inferred.extend(labels)
    for target in _extract_intent_target_terms(text):
        labels = _normalize_detection_labels([target])
        if labels:
            inferred.extend(labels)
    return _dedupe_detection_labels(inferred)


def _infer_behavior_intent_items(user_text: str) -> list[dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return []
    lowered = text.lower()
    items: list[dict[str, Any]] = []
    for rule in _BEHAVIOR_INTENT_RULES:
        aliases = tuple(str(alias).lower() for alias in rule.get("aliases", ()) if str(alias).strip())
        if not any(alias in lowered for alias in aliases):
            continue
        label = _normalize_detection_labels([str(rule.get("label") or "")])
        if not label:
            continue
        items.append(
            {
                "type": str(rule.get("type") or "behavior_detection"),
                "subject": str(rule.get("subject") or "person"),
                "behavior": str(rule.get("behavior") or label[0]),
                "label": label[0],
                "training_labels": _normalize_detection_labels(rule.get("training_labels") or []),
                "description": str(rule.get("description") or label[0]),
                "annotation_prompts": [str(item).strip() for item in rule.get("annotation_prompts", ()) if str(item).strip()],
                "sam3_prompts": [str(item).strip() for item in rule.get("annotation_prompts", ()) if str(item).strip()],
                "observable_entities": [str(item).strip() for item in rule.get("annotation_prompts", ()) if str(item).strip()],
                "source": "rule_fallback",
            }
        )
    items.extend(_infer_person_attribute_intent_items(text))
    items.extend(_infer_generic_person_entity_behavior_items(text))
    return _dedupe_intent_items(items)


def _infer_person_attribute_intent_items(user_text: str) -> list[dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return []
    lowered = text.lower()
    if not any(token in lowered for token in ("人员", "行人", "人", "person", "people", "pedestrian")):
        return []
    if not any(token in lowered for token in ("穿", "戴", "佩戴", "身穿", "穿着", "wear", "wearing", "with")):
        return []

    items: list[dict[str, Any]] = []
    for rule in _PERSON_ATTRIBUTE_RULES:
        aliases = tuple(str(alias).lower() for alias in rule.get("aliases", ()) if str(alias).strip())
        if not any(alias in lowered for alias in aliases):
            continue
        attribute = _normalize_detection_labels([str(rule.get("attribute") or "")])
        if not attribute:
            continue
        label = _normalize_detection_labels([f"person_wearing_{attribute[0]}"])
        if not label:
            continue
        phrase = str(rule.get("phrase") or attribute[0].replace("_", " ")).strip()
        prompt = f"person wearing {phrase}"
        items.append(
            {
                "type": "person_attribute_detection",
                "subject": "person",
                "attribute": attribute[0],
                "label": label[0],
                "training_labels": label,
                "business_labels": label,
                "description": str(rule.get("description") or f"检测画面中穿戴 {attribute[0]} 的人"),
                "observable_entities": [prompt],
                "sam3_prompts": [prompt],
                "annotation_prompts": [prompt],
                "source": "person_attribute_fallback",
            }
        )
    return _dedupe_intent_items(items)


def _infer_generic_person_entity_behavior_items(user_text: str) -> list[dict[str, Any]]:
    text = str(user_text or "").strip()
    if not text:
        return []
    lowered = text.lower()
    if not any(token in lowered for token in ("人员", "行人", "人", "person", "people", "pedestrian")):
        return []
    if not any(token in lowered for token in ("打", "玩", "使用", "拿", "持", "骑", "穿", "戴", "钓", "play", "use", "hold", "ride", "wear")):
        return []

    auxiliary_labels: list[str] = []
    for aliases, labels in _INTENT_LABEL_RULES:
        if any(str(alias).lower() in lowered for alias in aliases):
            auxiliary_labels.extend(label for label in labels if not _is_person_subject_label(label))
    auxiliary_labels = _dedupe_detection_labels(auxiliary_labels)
    if not auxiliary_labels:
        return []

    action = "interact"
    if any(token in lowered for token in ("打", "玩", "play")):
        action = "play"
    elif any(token in lowered for token in ("使用", "拿", "持", "use", "hold")):
        action = "use"
    elif any(token in lowered for token in ("骑", "ride")):
        action = "ride"
    elif any(token in lowered for token in ("穿", "戴", "wear")):
        action = "wear"
    elif any(token in lowered for token in ("钓", "fish")):
        action = "fish"

    primary = auxiliary_labels[0]
    label = _normalize_detection_labels([f"person_{action}_{primary}"])
    if not label:
        return []
    prompts = _dedupe_text_values(["person", *auxiliary_labels])
    return [
        {
            "type": "entity_interaction",
            "subject": "person",
            "behavior": action,
            "label": label[0],
            "training_labels": _dedupe_detection_labels(["person", *auxiliary_labels]),
            "business_labels": label,
            "observable_entities": prompts,
            "sam3_prompts": prompts,
            "annotation_prompts": prompts,
            "description": f"检测画面中人员与 {', '.join(auxiliary_labels)} 相关的行为",
            "source": "generic_person_entity_fallback",
        }
    ]


def _apply_behavior_intent_guard(
    spec: dict[str, Any],
    user_text: str,
    *,
    preserve_explicit_labels: bool = False,
) -> dict[str, Any]:
    # LLM 产出的结构化语义计划是主路径。固定词表只在模型没有给出任何
    # 可用 intent_items 时兜底，避免旧规则向有效计划追加宽泛 prompt。
    existing_items = _spec_intent_items(spec)
    inferred_items = [] if existing_items else _infer_behavior_intent_items(user_text)
    all_items = _dedupe_intent_items([*existing_items, *inferred_items])
    behavior_labels = _labels_from_intent_items(all_items)
    if not behavior_labels:
        return spec

    merged = dict(spec or {})
    current_labels = _normalize_detection_labels(_spec_string_list(merged, "labels"))
    if preserve_explicit_labels and current_labels:
        labels = current_labels
    else:
        # 行为检测以 intent_items 的训练 label 为准。LLM 有时会把行为里的辅助物体
        # 也放进顶层 labels，例如“人员玩手机检测”返回 person_use_phone + phone。
        # 这会把训练任务从行为检测扩大成额外的物体检测，因此默认只保留行为/状态
        # 意图 label；显式 labels=... 的场景由 preserve_explicit_labels 分支保留。
        labels = behavior_labels
    if labels:
        merged["labels"] = labels
    merged["intent_items"] = all_items
    if not _spec_string(merged, "task_type") or _spec_string(merged, "task_type") == "object_detection":
        merged["task_type"] = _behavior_task_type(all_items)
    if not _spec_string(merged, "annotation_prompts"):
        prompts = _annotation_prompts_from_intent_items(all_items)
        if prompts:
            merged["annotation_prompts"] = prompts

    description = _spec_string(merged, "task_description")
    if not description or _looks_like_generic_person_description(description):
        merged["task_description"] = _behavior_task_description_with_extra_labels(all_items, _spec_string_list(merged, "labels"))

    prompt = _spec_string(merged, "generation_prompt")
    if not prompt or _looks_like_generic_person_description(prompt):
        merged["generation_prompt"] = _fallback_generation_prompt(
            _spec_string_list(merged, "labels"),
            task_description=_spec_string(merged, "task_description"),
            user_text=user_text,
        )
        merged["use_synthetic_generation"] = True
    return merged


def _raw_string_values(*values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, list | tuple):
            result.extend(str(item).strip() for item in value if str(item).strip())
        elif isinstance(value, str) and value.strip():
            result.append(value.strip())
    return result


def _intent_is_constrained_target(item: dict[str, Any]) -> bool:
    strategy = str(item.get("annotation_strategy") or "").strip().lower()
    if strategy in {"constrained_target", "requires_review", "candidate_and_verify"}:
        return True
    task_type = str(item.get("type") or item.get("task_type") or "").strip().lower()
    if task_type in _CONSTRAINED_INTENT_TYPES:
        return True
    return any(bool(_raw_string_values(item.get(field))) for field in _CONSTRAINT_FIELDS)


def _intent_prompt_is_forbidden(item: dict[str, Any], prompt: str) -> bool:
    prompt_key = str(prompt or "").strip().lower()
    if not prompt_key:
        return True
    forbidden = {
        str(value).strip().lower()
        for value in _raw_string_values(item.get("forbidden_direct_prompts"))
        if str(value).strip()
    }
    if _intent_is_constrained_target(item):
        primary = str(item.get("primary_sam3_prompt") or "").strip().lower()
        broad_values = _raw_string_values(item.get("base_entity"), item.get("bbox_target"), item.get("subject"))
        forbidden.update(str(value).strip().lower() for value in broad_values if str(value).strip())
        if primary:
            forbidden.discard(primary)
    return prompt_key in forbidden


def _intent_sam3_prompts(item: dict[str, Any]) -> list[str]:
    prompts = _dedupe_text_values(
        _raw_string_values(
            item.get("primary_sam3_prompt"),
            item.get("sam3_prompts"),
            item.get("sam_prompts"),
        )
    )
    if not prompts:
        prompts = _dedupe_text_values(_raw_string_values(item.get("annotation_prompts")))
    if not prompts and not _intent_is_constrained_target(item):
        prompts = _dedupe_text_values(_raw_string_values(item.get("observable_entities")))
    return [prompt for prompt in prompts if not _intent_prompt_is_forbidden(item, prompt)]


def _spec_intent_items(spec: dict[str, Any]) -> list[dict[str, Any]]:
    value = spec.get("intent_items")
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        raw_business_label = str(raw.get("business_label") or raw.get("label") or "").strip()
        label = _normalize_detection_labels([str(raw.get("label") or raw.get("train_label") or "")])
        if not label:
            raw_training_labels = raw.get("training_labels")
            fallback_labels = (
                list(raw_training_labels)
                if isinstance(raw_training_labels, (list, tuple))
                else [raw_training_labels]
                if isinstance(raw_training_labels, str) and raw_training_labels.strip()
                else []
            )
            label = _normalize_detection_labels(
                fallback_labels
            )
        if not label:
            continue
        item = dict(raw)
        item["label"] = label[0]
        task_type = str(item.get("type") or item.get("task_type") or "object_detection").strip().lower()
        # 大模型常把行为类写成 behavior/action，把状态类写成 state/status。
        # 工作流内部统一使用 *_detection，避免后续提取训练 label 时漏掉这些意图。
        if task_type in {"behavior", "action", "activity", "behaviour"}:
            task_type = "behavior_detection"
        elif task_type in {"state", "status"}:
            task_type = "state_detection"
        elif task_type in {"person_attribute", "appearance", "wearing", "wearing_detection"}:
            task_type = "person_attribute_detection"
        item["type"] = task_type
        item["subject"] = str(item.get("subject") or "")
        item["description"] = str(item.get("description") or "")
        item["business_label"] = raw_business_label or item["label"]
        item["bbox_target"] = str(item.get("bbox_target") or "")
        item["base_entity"] = str(item.get("base_entity") or item.get("subject") or "")
        item["annotation_strategy"] = str(item.get("annotation_strategy") or "").strip().lower()
        item["assumption"] = str(item.get("assumption") or "")
        for field in (*_CONSTRAINT_FIELDS, "auxiliary_entities", "forbidden_direct_prompts"):
            item[field] = _dedupe_text_values(_raw_string_values(item.get(field)))
        training_labels = item.get("training_labels")
        if isinstance(training_labels, list | tuple):
            item["training_labels"] = _normalize_detection_labels(training_labels)
        elif isinstance(training_labels, str) and training_labels.strip():
            item["training_labels"] = _normalize_detection_labels([training_labels])
        else:
            item["training_labels"] = []
        item["business_labels"] = _normalize_detection_labels(
            _raw_string_values(
                item.get("business_labels"),
                item.get("business_label"),
                item.get("label"),
            )
        )
        item["observable_entities"] = _dedupe_text_values(
            _raw_string_values(
                item.get("observable_entities"),
                item.get("visible_entities"),
                item.get("entities"),
            )
        )
        raw_prompt_map = item.get("sam3_prompt_map")
        item["sam3_prompt_map"] = {
            str(prompt).strip(): label_name[0]
            for prompt, mapped_label in (raw_prompt_map.items() if isinstance(raw_prompt_map, dict) else [])
            if str(prompt).strip()
            if (label_name := _normalize_detection_labels([str(mapped_label or "")]))
        }
        item["primary_sam3_prompt"] = str(item.get("primary_sam3_prompt") or "").strip()
        item["sam3_prompts"] = _intent_sam3_prompts(item)
        if not item["primary_sam3_prompt"] and item["sam3_prompts"]:
            item["primary_sam3_prompt"] = item["sam3_prompts"][0]
        if not item["training_labels"] and _is_entity_interaction_intent_item(item):
            item["training_labels"] = _normalize_detection_labels(item["observable_entities"] or item["sam3_prompts"])
        item["annotation_prompts"] = item["sam3_prompts"]
        items.append(item)
    return _dedupe_intent_items(items)


def _dedupe_intent_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for item in items:
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in by_label:
            existing = by_label[key]
            # 同一个训练 label 可能同时来自 LLM 和本地行为规则。这里合并字段，
            # 避免 LLM 只给 person_fall 而规则里的自然语言标注提示词被去重丢掉。
            for field in (
                "type",
                "subject",
                "behavior",
                "description",
                "source",
                "business_label",
                "bbox_target",
                "base_entity",
                "annotation_strategy",
                "primary_sam3_prompt",
                "assumption",
            ):
                if not str(existing.get(field) or "").strip() and str(item.get(field) or "").strip():
                    existing[field] = item.get(field)
            existing["annotation_prompts"] = _dedupe_text_values([
                *(existing.get("annotation_prompts") if isinstance(existing.get("annotation_prompts"), list) else []),
                *(item.get("annotation_prompts") if isinstance(item.get("annotation_prompts"), list) else []),
            ])
            existing["sam3_prompts"] = _dedupe_text_values([
                *(existing.get("sam3_prompts") if isinstance(existing.get("sam3_prompts"), list) else []),
                *(item.get("sam3_prompts") if isinstance(item.get("sam3_prompts"), list) else []),
            ])
            existing["observable_entities"] = _dedupe_text_values([
                *(existing.get("observable_entities") if isinstance(existing.get("observable_entities"), list) else []),
                *(item.get("observable_entities") if isinstance(item.get("observable_entities"), list) else []),
            ])
            for field in (*_CONSTRAINT_FIELDS, "auxiliary_entities", "forbidden_direct_prompts"):
                existing[field] = _dedupe_text_values([
                    *(existing.get(field) if isinstance(existing.get(field), list) else []),
                    *(item.get(field) if isinstance(item.get(field), list) else []),
                ])
            existing["sam3_prompt_map"] = {
                **(existing.get("sam3_prompt_map") if isinstance(existing.get("sam3_prompt_map"), dict) else {}),
                **(item.get("sam3_prompt_map") if isinstance(item.get("sam3_prompt_map"), dict) else {}),
            }
            existing["business_labels"] = _dedupe_detection_labels([
                *(existing.get("business_labels") if isinstance(existing.get("business_labels"), list) else []),
                *(item.get("business_labels") if isinstance(item.get("business_labels"), list) else []),
            ])
            existing["training_labels"] = _dedupe_detection_labels([
                *(existing.get("training_labels") if isinstance(existing.get("training_labels"), list) else []),
                *(item.get("training_labels") if isinstance(item.get("training_labels"), list) else []),
            ])
            continue
        copied = dict(item)
        copied["annotation_prompts"] = _dedupe_text_values(
            copied.get("annotation_prompts") if isinstance(copied.get("annotation_prompts"), list) else []
        )
        copied["sam3_prompts"] = _dedupe_text_values(
            copied.get("sam3_prompts") if isinstance(copied.get("sam3_prompts"), list) else []
        )
        copied["observable_entities"] = _dedupe_text_values(
            copied.get("observable_entities") if isinstance(copied.get("observable_entities"), list) else []
        )
        copied["business_labels"] = _dedupe_detection_labels(
            copied.get("business_labels") if isinstance(copied.get("business_labels"), list) else []
        )
        copied["training_labels"] = _dedupe_detection_labels(
            copied.get("training_labels") if isinstance(copied.get("training_labels"), list) else []
        )
        by_label[key] = copied
        result.append(copied)
    return result


def _labels_from_intent_items(items: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for item in items:
        training_labels = item.get("training_labels")
        if isinstance(training_labels, list) and training_labels:
            labels.extend(_normalize_detection_labels(training_labels))
        else:
            labels.extend(_normalize_detection_labels([str(item.get("label") or item.get("train_label") or "")]))
    return _dedupe_detection_labels(labels)


def _annotation_prompts_from_intent_items(items: list[dict[str, Any]]) -> list[str]:
    prompts: list[str] = []
    for item in items:
        raw_prompts = item.get("sam3_prompt_map", {}).keys() if _is_entity_interaction_intent_item(item) else None
        if not raw_prompts and _is_entity_interaction_intent_item(item):
            raw_prompts = item.get("observable_entities")
        if not raw_prompts:
            raw_prompts = _intent_sam3_prompts(item)
        if raw_prompts and not isinstance(raw_prompts, str):
            prompts.extend(str(prompt).strip() for prompt in raw_prompts if str(prompt).strip())
    return _dedupe_text_values(prompts)


def _is_entity_interaction_intent_item(item: dict[str, Any]) -> bool:
    task_type = str(item.get("type") or item.get("task_type") or "").strip().lower()
    if task_type in _PERSON_ATTRIBUTE_TYPES:
        return False
    if task_type in _ENTITY_INTERACTION_TYPES:
        return True
    strategy = str(item.get("annotation_strategy") or "").strip().lower()
    if strategy in {"constrained_target", "requires_review", "candidate_and_verify"}:
        return False
    if task_type in _CONSTRAINED_INTENT_TYPES:
        return False
    entity_labels = _normalize_detection_labels(item.get("observable_entities") if isinstance(item.get("observable_entities"), list) else [])
    training_labels = _normalize_detection_labels(item.get("training_labels") if isinstance(item.get("training_labels"), list) else [])
    non_person_entities = [label for label in entity_labels if _is_stable_auxiliary_entity_label(label)]
    non_person_training = [label for label in training_labels if _is_stable_auxiliary_entity_label(label)]
    return bool(non_person_entities and ("person" in entity_labels or len(entity_labels) > 1)) or bool(non_person_training and "person" in training_labels)


def _is_stable_auxiliary_entity_label(label: str) -> bool:
    key = _label_lookup_key(label)
    return bool(key and not _is_person_subject_label(key) and "person" not in key)


def _annotation_prompt_map_from_spec(spec: dict[str, Any], labels: list[str]) -> dict[str, str]:
    class_names = _normalize_detection_labels(labels)
    if not class_names:
        return {}
    class_set = {label.lower() for label in class_names}
    prompt_map: dict[str, str] = {}
    for item in _spec_intent_items(spec):
        business_label = _normalize_detection_labels([str(item.get("label") or "")])
        if not business_label:
            continue
        mapped_labels = _normalize_detection_labels(item.get("training_labels") or []) or business_label
        mapped_labels = [label for label in mapped_labels if label.lower() in class_set]
        if not mapped_labels:
            continue
        declared_prompt_map = item.get("sam3_prompt_map") if isinstance(item.get("sam3_prompt_map"), dict) else {}
        for prompt, mapped_label in declared_prompt_map.items():
            normalized_label = _normalize_detection_labels([str(mapped_label or "")])
            if not normalized_label or normalized_label[0].lower() not in class_set:
                continue
            if _intent_prompt_is_forbidden(item, str(prompt)):
                continue
            prompt_map[str(prompt).strip()] = normalized_label[0]
        if _is_entity_interaction_intent_item(item):
            raw_prompts = item.get("observable_entities") if isinstance(item.get("observable_entities"), list) else []
            if not raw_prompts:
                raw_prompts = item.get("training_labels") if isinstance(item.get("training_labels"), list) else []
            prompts = [*raw_prompts]
        else:
            prompts = _intent_sam3_prompts(item)
            if not _intent_is_constrained_target(item):
                prompts = [*prompts, _label_to_annotation_prompt(business_label[0])]
        for prompt in prompts:
            text = str(prompt or "").strip()
            if not text or _intent_prompt_is_forbidden(item, text):
                continue
            prompt_label = _best_prompt_mapped_label(text, mapped_labels)
            if prompt_label:
                prompt_map[text] = prompt_label
    top_level_prompts = _spec_string_list(spec, "annotation_prompts")
    if len(class_names) == 1:
        constrained_items = [
            item
            for item in _spec_intent_items(spec)
            if class_names[0] in _normalize_detection_labels(item.get("training_labels") or [item.get("label")])
            and _intent_is_constrained_target(item)
        ]
        for prompt in top_level_prompts:
            if constrained_items and any(_intent_prompt_is_forbidden(item, prompt) for item in constrained_items):
                continue
            prompt_map.setdefault(prompt, class_names[0])
    for label in class_names:
        related_items = [
            item
            for item in _spec_intent_items(spec)
            if label in _normalize_detection_labels(item.get("training_labels") or [item.get("label")])
        ]
        if any(_intent_is_constrained_target(item) for item in related_items):
            continue
        prompt_map.setdefault(_label_to_annotation_prompt(label), label)
        prompt_map.setdefault(label, label)
    return prompt_map


def _best_prompt_mapped_label(prompt: str, labels: list[str]) -> str:
    normalized_labels = _normalize_detection_labels(labels)
    if not normalized_labels:
        return ""
    if len(normalized_labels) == 1:
        return normalized_labels[0]
    prompt_key = _label_lookup_key(prompt)
    prompt_canonical = _normalize_detection_labels([prompt])
    if prompt_canonical:
        prompt_key = prompt_canonical[0]
    for label in normalized_labels:
        if prompt_key == _label_lookup_key(label):
            return label
    for label in normalized_labels:
        if _looks_related_prompt(prompt, label):
            return label
    return ""


def _looks_related_prompt(prompt: str, label: str) -> bool:
    prompt_tokens = {item for item in re.split(r"[^a-z0-9]+", str(prompt).lower()) if item}
    label_tokens = {item for item in re.split(r"[^a-z0-9]+", _label_to_annotation_prompt(label).lower()) if item}
    return bool(prompt_tokens and label_tokens and prompt_tokens.intersection(label_tokens))


def _label_to_annotation_prompt(label: str) -> str:
    text = str(label or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    if not text:
        return ""
    if text.startswith("person_"):
        return "person " + text.removeprefix("person_").replace("_", " ")
    return text.replace("_", " ")


def _behavior_annotation_prompts_for_label(label: str) -> list[str]:
    normalized = _normalize_detection_labels([label])
    if not normalized:
        return []
    target = normalized[0]
    for rule in _BEHAVIOR_INTENT_RULES:
        rule_label = _normalize_detection_labels([str(rule.get("label") or "")])
        if rule_label and rule_label[0] == target:
            return _dedupe_text_values([str(item).strip() for item in rule.get("annotation_prompts", ()) if str(item).strip()])
    return []


def _dedupe_text_values(values: list[Any] | tuple[Any, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip().strip("\"'`，,;；。")
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _behavior_task_type(items: list[dict[str, Any]]) -> str:
    types = {str(item.get("type") or "").strip().lower() for item in items}
    if types and types.issubset(_PERSON_ATTRIBUTE_TYPES):
        return "person_attribute_detection"
    if types.intersection(_ENTITY_INTERACTION_TYPES):
        return "entity_interaction"
    if "anomaly_detection" in types:
        return "anomaly_detection"
    if types.intersection({"behavior_detection", "behavior", "behaviour", "action", "activity"}):
        return "behavior_detection"
    if types.intersection({"state_detection", "state", "status"}):
        return "state_detection"
    return "object_detection"


def _behavior_task_description(items: list[dict[str, Any]]) -> str:
    descriptions = [str(item.get("description") or item.get("label") or "").strip() for item in items]
    descriptions = [item for item in descriptions if item]
    return "；".join(descriptions) if descriptions else "人员行为状态检测"


def _behavior_task_description_with_extra_labels(items: list[dict[str, Any]], labels: list[str]) -> str:
    description = _behavior_task_description(items)
    behavior_labels = set(_labels_from_intent_items(items))
    extra_labels = [
        label
        for label in _normalize_detection_labels(labels)
        if label not in behavior_labels and not _is_person_subject_label(label) and label not in _BEHAVIOR_AUXILIARY_LABELS
    ]
    if extra_labels:
        return f"{description}；检测 {', '.join(extra_labels)}"
    return description


def _is_person_subject_label(label: str) -> bool:
    return _label_lookup_key(label) in _PERSON_SUBJECT_LABELS


def _looks_like_generic_person_description(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return bool(
        re.search(r"\b(?:person|people|pedestrian|human)\s+detection\b", text)
        or re.search(r"(人员|行人|人体|人)\s*(目标)?(检测|识别)", text)
    )


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
    text = re.sub(
        r"(?i)yolo\d*[a-z]*|deimv?\d*[a-z0-9_\-]*|dino(?:v?\d+)?|cv|目标检测|检测|识别|分割|模型|算法|训练|帮我|请|一个|一套|一种|的|用于",
        "",
        text,
    )
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
    if _looks_like_deimv2_training(raw_training):
        model_variant = _normalize_deimv2_model_variant(raw_training.get("model_variant"))
        model_spec = _deimv2_model_spec(model_variant)
        cfg = {
            "split": dict(DEFAULT_TRAINING_SPLIT),
            "training": {
                "model_variant": model_variant,
                "template_config": str(raw_training.get("template_config") or model_spec["template_config"]),
                "epochs": 10,
                "img_size": 640,
                "batch": 1,
                "device": "auto",
                "workers": 0,
                "backbone_checkpoint": str(raw_training.get("backbone_checkpoint") or DEIMV2_BACKBONE_CHECKPOINT),
                "tuning_checkpoint": str(raw_training.get("tuning_checkpoint") or model_spec["tuning_checkpoint"]),
            },
            "runtime": {
                "conda_env_name": "",
                "enforce_conda_env": False,
            },
        }
        for raw_key, out_key in (("epochs", "epochs"), ("img_size", "img_size"), ("imgsz", "img_size"), ("batch", "batch"), ("workers", "workers"), ("flat_epoch", "flat_epoch"), ("no_aug_epoch", "no_aug_epoch"), ("warmup_iter", "warmup_iter"), ("checkpoint_freq", "checkpoint_freq"), ("num_top_queries", "num_top_queries")):
            value = _coerce_int(raw_training.get(raw_key))
            if value is not None:
                cfg["training"][out_key] = value

        for raw_key, out_key in (("lr", "lr"), ("learning_rate", "lr"), ("weight_decay", "weight_decay")):
            value = _coerce_float(raw_training.get(raw_key))
            if value is not None:
                cfg["training"][out_key] = value
        if isinstance(raw_training.get("betas"), list):
            cfg["training"]["betas"] = raw_training["betas"]
        if isinstance(raw_training.get("optimizer"), dict):
            cfg["training"]["optimizer"] = dict(raw_training["optimizer"])
        for key in ("device", "template_config", "model_variant", "backbone_checkpoint", "tuning_checkpoint", "requested_model_variant", "model_variant_fallback_reason"):
            value = raw_training.get(key)
            if value not in (None, ""):
                cfg["training"][key] = str(value).strip()
        value = raw_runtime.get("conda_env_name") or raw_training.get("conda_env") or raw_training.get("conda_env_name")
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
        return cfg
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


def _looks_like_deimv2_training(raw_training: dict[str, Any]) -> bool:
    text = " ".join(
        str(raw_training.get(key) or "")
        for key in ("model_variant", "requested_model_variant", "template_config", "backbone_checkpoint", "tuning_checkpoint")
    ).lower()
    return "deimv2" in text or "dinov3" in text or "vitt_distill" in text


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
    training_backend: str = "yolo",
) -> str:
    parsed = _parse_json_object(fallback_reply)
    stderr_tail = str(parsed.get("stderr_tail") or "").strip()
    stdout_tail = str(parsed.get("stdout_tail") or "").strip()
    error_tail = stderr_tail or stdout_tail
    training_error = parsed.get("training_error") if isinstance(parsed.get("training_error"), dict) else {}
    training_error_message = str(training_error.get("message") or "").strip()
    training_error_details = str(training_error.get("details") or "").strip()
    training_error_hint = str(training_error.get("hint") or "").strip()
    training_log_path = str(training_error.get("log_path") or "").strip()
    returncode = parsed.get("returncode")
    prep = data_preparation_summary or summary
    synthetic = _synthetic_generation_facts(prep)

    backend_name = "DEIMv2" if training_backend == "deimv2" else "YOLO"
    checkpoint_label = "best checkpoint" if training_backend == "deimv2" else "best.pt"
    lines = [f"{backend_name} 训练流程执行失败。", ""]
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
            f"- {checkpoint_label}: `{best_pt or '未生成'}`",
        ]
    )
    if training_error_message:
        lines.append(f"- 主要错误：{training_error_message}")
        if training_error_hint:
            lines.append(f"- 原因提示：{training_error_hint}")
        if training_log_path:
            lines.append(f"- 训练日志：`{training_log_path}`")
        if training_error_details:
            lines.extend(["- 错误详情：", "```text", training_error_details[-6000:], "```"])
    elif error_tail:
        lines.extend(["- 主要错误：", "```text", error_tail[-2000:], "```"])
    return "\n".join(lines).strip()


def _model_generation_failed_reply(title: str, recorder: EventRecorder, *, missing_field: str) -> str:
    reason_lines = _model_generation_failure_reason_lines(recorder, missing_field=missing_field)
    if not reason_lines:
        reason_lines = [f"失败原因：模型返回内容中缺少 `{missing_field}` 字段，且无法根据当前请求生成兜底值。"]
    return "\n".join([title, "", *(f"- {line}" for line in reason_lines)]).strip()


def _model_generation_failure_reason_lines(recorder: EventRecorder, *, missing_field: str) -> list[str]:
    event = _latest_llm_completion_event(recorder)
    if event is None:
        return [
            f"失败原因：模型没有产出 `{missing_field}`，且当前运行记录中没有找到可用的 LLM 调用完成事件。",
            "阶段说明：流程停在模型生成训练意图/合成提示词阶段，尚未进入数据处理、生图或训练阶段。",
        ]

    data = event.data if isinstance(event.data, dict) else {}
    purpose = str(data.get("purpose") or "").strip()
    reason = str(data.get("reason") or "").strip()
    error = str(data.get("error") or "").strip()
    failure_reason = str(data.get("failure_reason") or "").strip()
    generated_keys = _llm_event_generated_keys(data)

    lines: list[str] = []
    if failure_reason:
        lines.append(f"失败原因：{_strip_failure_reason_prefix(failure_reason)}")
    elif reason == "not_configured":
        lines.append(f"失败原因：{_llm_not_configured_failure_reason()}")
    elif error:
        lines.append(_classify_llm_error_reason(error))
    elif generated_keys is not None:
        if generated_keys:
            lines.append(
                f"失败原因：大模型已返回结构化结果，但结果缺少 `{missing_field}` 字段；"
                f"本次返回字段为：{', '.join(generated_keys)}。"
            )
        else:
            lines.append(f"失败原因：大模型返回为空、不是有效 JSON，或没有生成任何可用字段，因此缺少 `{missing_field}`。")
    else:
        lines.append(f"失败原因：大模型没有生成 `{missing_field}`，且没有返回明确的错误原因。")

    if purpose:
        lines.append(f"模型调用阶段：`{purpose}`。")
    lines.append("阶段说明：流程停在模型生成训练意图/合成提示词阶段，尚未进入数据处理、生图或训练阶段。")
    lines.append("处理建议：检查应用模型配置、base_url 是否可访问、api_key 是否有效、模型名称是否正确，以及模型返回是否为包含 `generation_prompt` 的 JSON。")
    return lines


def _latest_llm_completion_event(recorder: EventRecorder) -> ChatEvent | None:
    for event in reversed(recorder.events):
        if event.type != "llm.completed":
            continue
        data = event.data if isinstance(event.data, dict) else {}
        purpose = str(data.get("purpose") or "")
        if purpose.startswith("workflow_"):
            return event
    return None


def _llm_failure_summaries(recorder: EventRecorder) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for event in recorder.events:
        if event.type != "llm.completed":
            continue
        data = event.data if isinstance(event.data, dict) else {}
        if not data.get("used_fallback"):
            continue
        purpose = str(data.get("purpose") or "unknown").strip()
        failure_reason = str(data.get("failure_reason") or "").strip()
        reason = str(data.get("reason") or "").strip()
        error = str(data.get("error") or "").strip()
        if failure_reason:
            detail = _strip_failure_reason_prefix(failure_reason)
        elif reason == "not_configured":
            detail = _llm_not_configured_failure_reason()
        elif error:
            detail = _strip_failure_reason_prefix(_classify_llm_error_reason(error))
        else:
            detail = "大模型调用失败或使用了 fallback，但事件中没有记录更具体的错误。"
        key = (purpose, detail)
        if key in seen:
            continue
        seen.add(key)
        failures.append({"purpose": purpose, "reason": detail})
        if len(failures) >= 8:
            break
    return failures


def _llm_event_generated_keys(data: dict[str, Any]) -> list[str] | None:
    for key in ("generated_keys", "extracted_keys", "completed_keys"):
        raw = data.get(key)
        if isinstance(raw, list):
            return sorted(str(item) for item in raw)
    return None


def _llm_not_configured_failure_reason() -> str:
    return "当前应用没有可用的大模型配置，或 agent/app/runtimeOptions 中缺少 `base_url`、`model` 等必要配置。"


def _classify_llm_error_reason(error: str) -> str:
    lowered = error.lower()
    if _contains_any(lowered, ("connecttimeout", "readtimeout", "timeout", "timed out", "deadline exceeded")):
        return "失败原因：访问大模型服务超时；请检查模型服务响应速度、网络链路、request_timeout_seconds 配置或中转服务是否卡住。"
    if _contains_any(lowered, ("connection refused", "failed to establish a new connection", "connecterror", "connectionerror", "name or service not known", "nodename nor servname", "network is unreachable", "no route to host")):
        return "失败原因：大模型服务地址不可达或网络不通；请检查 base_url、端口、防火墙、DNS、代理/VPN 路由和中转服务状态。"
    if _contains_any(lowered, ("proxyerror", "proxy error", "tunnel connection failed")):
        return "失败原因：访问大模型服务时受到代理影响；请检查代理配置，内网/中转地址需要时应配置 NO_PROXY 或绕过代理。"
    if _contains_any(lowered, ("invalid_token", "invalid token", "invalid api key", "invalid_api_key", "incorrect api key")):
        return "失败原因：大模型服务返回 token/api_key 无效；请检查上传应用 JSON、runtimeOptions 或中转服务中的 api_key 是否正确、是否过期。"
    if _contains_any(lowered, ("insufficient_user_quota", "insufficient quota", "quota exceeded", "no quota", "余额不足", "额度不足")):
        return "失败原因：大模型账号额度不足或配额耗尽；请检查云平台/中转服务账号余额、调用额度和模型配额。"
    if _contains_any(lowered, ("rate limit", "ratelimit", "too many requests", "status code: 429", "client error '429", "httpstatuserror: client error '429")):
        return "失败原因：大模型服务返回 429 限流；请降低请求频率、缩短上下文或检查中转服务限流策略。"
    if _contains_any(lowered, ("401 unauthorized", "status code: 401", "client error '401", "httpstatuserror: client error '401")):
        return "失败原因：大模型服务返回 401，通常是 api_key 缺失、无效或鉴权头不符合中转服务要求。"
    if _contains_any(lowered, ("403 forbidden", "status code: 403", "client error '403", "httpstatuserror: client error '403")):
        return "失败原因：大模型服务返回 403，当前 api_key 或账号没有调用该模型/接口的权限。"
    if _contains_any(lowered, ("404 not found", "status code: 404", "client error '404", "httpstatuserror: client error '404")):
        return "失败原因：大模型服务返回 404，通常是 base_url 路径、`/chat/completions` 中转路径或模型名称配置错误。"
    if _contains_any(lowered, ("400 bad request", "status code: 400", "client error '400", "httpstatuserror: client error '400")):
        return "失败原因：大模型服务返回 400，请检查模型名称、请求格式、max_tokens、messages 或中转服务参数兼容性。"
    if _contains_any(lowered, ("500 internal server error", "502 bad gateway", "503 service unavailable", "504 gateway timeout", "server error '500", "server error '502", "server error '503", "server error '504")):
        return "失败原因：大模型服务或中转服务返回 5xx，说明请求已到达服务端但后端处理失败或网关不可用；请检查中转服务和模型服务日志。"
    return f"失败原因：大模型调用异常：{error[:500]}"


def _data_preparation_failure_reason_lines(stderr_text: str, stdout_text: str) -> list[str]:
    text = f"{stderr_text or ''}\n{stdout_text or ''}"
    lowered = text.lower()
    if not text.strip():
        return []

    lines: list[str] = []
    if _contains_any(
        lowered,
        (
            "sam3",
            "sam3-predict.py",
            "/v1/sam3/predict",
            "real dataset auto-annotation failed",
            "auto-annotation failed",
        ),
    ):
        endpoint = _extract_sam3_endpoint(text)
        if _contains_any(lowered, ("connection refused", "failed to establish a new connection", "no route to host", "network is unreachable", "connectionerror", "sam3_connection_error")):
            lines.append(
                "失败原因：SAM3 标注服务端口不可达或网络不通"
                f"{f'（{endpoint}）' if endpoint else ''}；请检查 SAM3 服务是否启动、端口/防火墙/代理/VPN 路由是否正常。"
            )
        elif _contains_any(lowered, ("connecttimeout", "readtimeout", "timed out", "timeout", "sam3_connect_timeout", "sam3_read_timeout")):
            lines.append(
                "失败原因：SAM3 标注服务请求超时"
                f"{f'（{endpoint}）' if endpoint else ''}；请检查端口连通性、服务负载、单张图片是否过大或模型推理是否卡住。"
            )
        elif _contains_any(lowered, ("proxyerror", "proxy error", "tunnel connection failed")):
            lines.append(
                "失败原因：访问 SAM3 标注服务时受到代理影响；请确认访问内网 SAM3 地址时已绕过代理，或配置 NO_PROXY。"
            )
        elif _contains_any(lowered, ("invalid_token", "invalid token", "401 unauthorized", "status_code\": 401", '"status_code": 401', "401 client error")):
            lines.append(
                "失败原因：SAM3 标注服务鉴权失败，通常是 Authorization token 缺失、错误或已过期；请检查 SAM3 请求头和服务端 token 配置。"
            )
        elif _contains_any(lowered, ("rate limit", "too many requests", "status_code\": 429", '"status_code": 429', "429 client error", "quota", "额度不足")):
            lines.append(
                "失败原因：SAM3 标注服务限流或额度/资源不足；请降低并发、稍后重试，或检查 SAM3 服务资源和配额。"
            )
        elif _contains_any(lowered, ("500 server error", "internal server error", '"status_code": 500', "http 500")):
            lines.append(
                "失败原因：SAM3 标注服务接口返回 HTTP 500，说明端口已响应但服务内部处理失败；请检查 SAM3 服务日志、模型是否加载成功、当前图片或类别是否触发服务端异常。"
            )
        elif _contains_any(lowered, ("502 server error", "503 server error", "504 server error", '"status_code": 502', '"status_code": 503', '"status_code": 504', "bad gateway", "service unavailable", "gateway timeout")):
            lines.append(
                "失败原因：SAM3 标注服务网关或后端临时不可用；请检查 SAM3 服务进程、反向代理和模型服务健康状态。"
            )
        elif _contains_any(lowered, ("400 client error", "401 client error", "403 client error", "404 client error", '"status_code": 400', '"status_code": 401', '"status_code": 403', '"status_code": 404')):
            lines.append(
                "失败原因：SAM3 标注接口返回客户端错误；请检查接口地址、Authorization token、请求格式和类别文本是否符合 SAM3 接口约定。"
            )
        else:
            lines.append("失败原因：真实图片自动标注阶段调用 SAM3 失败；请检查 SAM3 服务、上传图片和检测类别配置。")

        if _contains_any(lowered, ("per_image_coco_path:", "[data-prep] per-image real coco written:")):
            lines.append("阶段说明：流程已经进入真实图片 SAM3 自动标注，并且部分图片已生成 COCO sidecar；失败发生在后续某张图片标注时，所以尚未进入生图、数据划分或训练阶段。")
        else:
            lines.append("阶段说明：流程在真实图片 SAM3 自动标注阶段失败，所以尚未进入生图、数据划分或训练阶段。")

        if "format': 'unlabeled'" in lowered or "images_only" in lowered or "unlabeled" in lowered:
            lines.append("数据提示：当前数据集被识别为未标注图片；如果 zip 内已有 COCO JSON，请确认放在 `annotations/coco.json` 或 `labels/coco.json`，否则会重新触发 SAM3 标注。")
    elif _contains_any(
        lowered,
        (
            "synthetic",
            "image-dataset-generation",
            "image-dataset-produce",
            "image_dataset_generation",
            "image_dataset_produce",
            "composite",
            "generation_prompt",
            "生图",
            "合成图",
        ),
    ):
        reason = _classify_synthetic_generation_error_reason(text)
        if reason:
            lines.append(f"失败原因：{reason}")
        else:
            lines.append("失败原因：合成图模型调用失败，但日志中没有返回明确的网络、鉴权、额度或服务端错误类型。")
        lines.append("阶段说明：流程在合成图模型调用或合成数据处理阶段失败，所以没有继续进入后续数据划分或训练阶段。")
    return lines


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _extract_sam3_endpoint(text: str) -> str:
    match = re.search(r"https?://[^\s\"'<>]+/v1/sam3/predict", text)
    return match.group(0).rstrip(".,);") if match else ""


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


def _strip_failure_reason_prefix(text: str) -> str:
    return re.sub(r"^\s*失败原因[:：]\s*", "", str(text or "").strip())


def _classify_synthetic_generation_error_reason(error: str) -> str:
    raw = str(error or "").strip()
    lowered = raw.lower()
    if not raw:
        return ""
    if _contains_any(lowered, ("connecttimeout", "readtimeout", "timeout", "timed out", "deadline exceeded")):
        return "合成图模型请求超时；请检查合成模型服务响应速度、网络链路、图片大小或请求超时配置。"
    if _contains_any(lowered, ("connection refused", "failed to establish a new connection", "connecterror", "connectionerror", "name or service not known", "network is unreachable", "no route to host")):
        return "合成图模型服务地址不可达或网络不通；请检查 base_url、端口、防火墙、DNS、代理/VPN 路由和模型服务状态。"
    if _contains_any(lowered, ("proxyerror", "proxy error", "tunnel connection failed")):
        return "访问合成图模型时受到代理影响；请检查代理配置，内网模型地址需要时应配置 NO_PROXY 或绕过代理。"
    if _contains_any(lowered, ("invalid_token", "invalid token", "invalid api key", "invalid_api_key", "incorrect api key", "401 unauthorized", "status code: 401", "client error '401")):
        return "合成图模型鉴权失败；请检查 api_key、token、Authorization 请求头或中转服务鉴权配置。"
    if _contains_any(lowered, ("403 forbidden", "status code: 403", "client error '403")):
        return "合成图模型返回 403；当前账号或 token 没有调用该模型/接口的权限。"
    if _contains_any(lowered, ("insufficient_user_quota", "insufficient quota", "quota exceeded", "no quota", "余额不足", "额度不足")):
        return "合成图模型账号额度不足或配额耗尽；请检查云平台/中转服务账号余额、调用额度和模型配额。"
    if _contains_any(lowered, ("rate limit", "ratelimit", "too many requests", "status code: 429", "client error '429")):
        return "合成图模型返回 429 限流；请降低请求频率、减少合成数量或检查中转服务限流策略。"
    if _contains_any(lowered, ("400 bad request", "status code: 400", "client error '400")):
        return "合成图模型返回 400；请检查请求格式、提示词、输入图片编码/路径、图片大小或模型参数是否符合接口要求。"
    if _contains_any(lowered, ("404 not found", "status code: 404", "client error '404")):
        return "合成图模型返回 404；通常是 base_url 路径、接口路由或模型名称配置错误。"
    if _contains_any(lowered, ("500 internal server error", "502 bad gateway", "503 service unavailable", "504 gateway timeout", "server error '500", "server error '502", "server error '503", "server error '504")):
        return "合成图模型或中转服务返回 5xx；说明请求已到达服务端但后端处理失败或网关不可用，请检查服务端日志。"
    return f"合成图模型调用异常：{raw[:500]}"


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
                reason = _classify_synthetic_generation_error_reason(primary_error)
                text += f"（主合成模型调用失败原因：{reason or primary_error[:180]}）"
            return text
        return f"已成功合并 {synthetic_count or 0} 张合成图"
    if status:
        reason = _classify_synthetic_generation_error_reason(error or primary_error)
        return f"状态 `{status}`" + (f"，合成图模型调用失败原因：{reason}" if reason else "")
    return ""


def _human_fallback_reply(
    fallback_reply: str,
    summary: dict[str, Any],
    best_pt: str,
    *,
    summary_failure_reason: str = "",
    llm_failures: list[dict[str, str]] | None = None,
) -> str:
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
            "best_pt": best_pt or summary.get("best_checkpoint"),
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
        f"- 最佳模型权重：`{facts.get('best_pt') or best_pt or '-'}`",
    ]
    if summary_failure_reason:
        lines[2:2] = [f"- 模型总结调用失败原因：{summary_failure_reason}", ""]
    if llm_failures:
        lines.extend(["", "大模型调用告警："])
        for item in llm_failures:
            if not isinstance(item, dict):
                continue
            purpose = str(item.get("purpose") or "unknown")
            reason = str(item.get("reason") or "").strip()
            lines.append(f"- `{purpose}`：{reason or '未记录明确原因'}")
    metrics = facts.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        metrics = _extract_metric_summary(summary)
    if isinstance(metrics, dict) and metrics:
        lines.extend(["", "关键评估指标："])
        for key in ("precision", "recall", "mAP50", "mAP50_95", "mAP75", "AR100", "best_coco_eval_bbox", "best_epoch", "fitness"):
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
    existing_metrics = summary.get("metrics")
    metrics: dict[str, Any] = {}
    if isinstance(existing_metrics, dict):
        for key in ("precision", "recall", "mAP50", "mAP50_95", "mAP75", "AR1", "AR10", "AR100", "best_coco_eval_bbox", "best_epoch", "fitness"):
            if key in existing_metrics:
                metrics[key] = existing_metrics[key]
        if "recall" not in metrics and "AR100" in metrics:
            metrics["recall"] = metrics["AR100"]
    results = _extract_results_dict(summary)
    mapping = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50_95": "metrics/mAP50-95(B)",
        "fitness": "fitness",
    }
    for out_key, source_key in mapping.items():
        if out_key not in metrics and source_key in results:
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


def _ensure_full_cycle_skills(selected_skills: list[str], training_backend: str = "yolo") -> list[str]:
    merged = [str(item).strip() for item in selected_skills if str(item).strip()]
    seen = set(merged)
    defaults = DEIMV2_SELECTED_SKILLS if training_backend == "deimv2" else DEFAULT_SELECTED_SKILLS
    for skill_name in defaults:
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


def _selected_training_backend(selected_skills: list[str], runtime_options: RuntimeOptions, workflow: str) -> str:
    del workflow
    app_template_name = str(getattr(runtime_options, "app_template_name", "") or "").strip().lower()
    if app_template_name == "algorithm-engineer-full-cycle-deimv2-test":
        return "deimv2"
    params = _workflow_skill_parameters(runtime_options)
    raw = str(params.get("training_backend") or params.get("trainingBackend") or "").strip().lower()
    if raw in {"deim", "deimv2", "dino", "dinov3"}:
        return "deimv2"
    if DEIMV2_TRAINING_SKILL in {str(item).strip() for item in selected_skills if str(item).strip()}:
        return "deimv2"
    return "yolo"


def _training_skill_for_backend(training_backend: str) -> str:
    return DEIMV2_TRAINING_SKILL if training_backend == "deimv2" else YOLO_TRAINING_SKILL


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
    project_root = _project_root()
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
        "deimv2-auto-training": {"training", "deimv2_training"},
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
    behavior_items = _infer_behavior_intent_items(text)
    if behavior_items:
        return _behavior_task_description(behavior_items)
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


def _fallback_generation_prompt(
    labels: list[str] | tuple[str, ...],
    *,
    task_description: str = "",
    user_text: str = "",
) -> str:
    label_text = _generation_prompt_label_text(labels)
    if not label_text:
        return ""
    objective = _clean_user_visible_text(task_description) or _training_objective_from_user_text(user_text)
    objective_clause = f"，任务目标为{objective}" if objective else ""
    return (
        f"真实监控画面，使用 image1.zip 作为参考图/场景背景，使用 image2.zip 作为目标图/前景目标；"
        f"将 image2.zip 中的 {label_text} 目标自然合成到 image1.zip 参考图的场景中"
        f"{objective_clause}，保持光照方向、色彩、尺度比例、透视关系和遮挡关系一致，"
        "目标轮廓清晰、完整可见、可精确标注，适合 COCO 目标检测边界框标注。"
    )


def _generation_prompt_label_text(labels: list[str] | tuple[str, ...]) -> str:
    normalized_labels = _normalize_detection_labels(list(labels))
    return ", ".join(normalized_labels)


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
    # 兜底 1：优先从 outputs 中找 JSON 候选文件
    json_candidates = sorted([p for p in paths.outputs.rglob("*.json") if p.is_file()])
    for p in reversed(json_candidates):
        n = p.name.lower()
        if (n == "coco.json" or n.endswith(".coco.json")) and _looks_like_coco_json(p):
            return p

    # 兜底 2：扫描 workspace 中常见的 COCO 文件名
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


def _workflow_output_root(paths: ThreadPaths | TrainingRunPaths) -> Path:
    if isinstance(paths, TrainingRunPaths):
        return paths.outputs
    return paths.outputs / WORKFLOW_OUTPUT_DIR


def _read_run_summary(paths: ThreadPaths | TrainingRunPaths, training_backend: str = "yolo") -> dict[str, Any]:
    run_dir = "deimv2_training_run" if training_backend == "deimv2" else "training_run"
    output_root = _workflow_output_root(paths)
    preferred = output_root / run_dir / "run_summary.json"
    if preferred.is_file():
        candidates = [preferred]
    else:
        candidates = list(output_root.rglob("run_summary.json"))
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


def _read_data_preparation_summary(paths: ThreadPaths | TrainingRunPaths) -> dict[str, Any]:
    output_root = _workflow_output_root(paths)
    preferred = output_root / "prepared_data" / "data_preparation_summary.json"
    candidates = [preferred] if preferred.is_file() else sorted(output_root.rglob("data_preparation_summary.json"))
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


def _fallback_deimv2_training_from_dataset_facts(
    dataset_facts: dict[str, Any],
    model_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_selection = model_selection or _default_deimv2_model_selection()
    image_count = _safe_int(dataset_facts.get("image_count"))
    image_summary = dataset_facts.get("image_size_summary") if isinstance(dataset_facts.get("image_size_summary"), dict) else {}
    avg_width = float(image_summary.get("avg_width") or 0)
    avg_height = float(image_summary.get("avg_height") or 0)
    img_size = 640 if max(avg_width, avg_height) >= 512 else 320
    epochs = _deimv2_epochs_from_dataset_facts(dataset_facts)
    batch = _deimv2_batch_from_dataset_facts(dataset_facts, {"device": "auto", "img_size": img_size})
    if image_count <= 30:
        split = dict(DEFAULT_TRAINING_SPLIT)
    elif image_count <= 100:
        split = {"train": 0.75, "val": 0.2, "test": 0.05}
    else:
        split = {"train": 0.7, "val": 0.2, "test": 0.1}
    return {
        "training": {
            "model_variant": model_selection.get("effective_model_variant") or DEIMV2_DEFAULT_MODEL_VARIANT,
            "template_config": model_selection.get("template_config") or DEIMV2_MODEL_VARIANTS[DEIMV2_DEFAULT_MODEL_VARIANT]["template_config"],
            "epochs": epochs,
            "img_size": img_size,
            "batch": batch,
            "device": "auto",
            "workers": 0,
            "backbone_checkpoint": model_selection.get("backbone_checkpoint") or DEIMV2_BACKBONE_CHECKPOINT,
            "tuning_checkpoint": model_selection.get("tuning_checkpoint") or "",
            "requested_model_variant": model_selection.get("requested_model_variant") or model_selection.get("effective_model_variant") or DEIMV2_DEFAULT_MODEL_VARIANT,
            "model_variant_fallback_reason": model_selection.get("fallback_reason") or "",
        },
        "split": split,
    }


def _deimv2_readme_for_prompt() -> str:
    project_root = _project_root()
    candidates = [
        project_root / "plugins" / "skills" / "deimv2-auto-training" / "references" / "DEIMv2_README.md",
        project_root / "plugins" / "skills" / "deimv2-auto-training" / "vendor" / "deimv2" / "README.md",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        wanted_markers = ("## 1. Model Zoo", "### 2.2 Data Preparation", "### 2.3 Backbone Preparation", "## 3. Usage")
        chunks: list[str] = []
        for marker in wanted_markers:
            start = text.find(marker)
            if start < 0:
                continue
            end = text.find("\n## ", start + 1)
            chunk = text[start:end if end > start else min(len(text), start + 5000)]
            chunks.append(chunk[:5000])
        excerpt = "\n\n".join(chunks).strip() or text[:12000]
        return excerpt[:16000]
    return (
        "DEIMv2 DINOv3 uses configs/deimv2/deimv2_dinov3_{s,m,l,x}_coco.yml, DINOv3STAs vit_tiny, "
        "ckpts/vitt_distill.pt backbone, COCO layout images/train|val and annotations/instances_train|val.json. "
        "Use train.py -c config.yml --use-amp on CUDA and -t checkpoint.pth for tuning."
    )


def _deimv2_vendor_config_context_for_prompt(model_selection: dict[str, Any] | None = None) -> str:
    model_selection = model_selection or _default_deimv2_model_selection()
    config_root = _deimv2_vendor_root() / "configs"
    template_config = str(model_selection.get("template_config") or DEIMV2_MODEL_VARIANTS[DEIMV2_DEFAULT_MODEL_VARIANT]["template_config"])
    template_rel = template_config.removeprefix("configs/").replace("\\", "/")
    relative_paths = _deimv2_template_config_context_paths(config_root, template_rel)
    chunks: list[str] = []
    for rel in relative_paths:
        path = config_root / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text = "\n".join(line.rstrip() for line in text.splitlines())
        chunks.append(f"# configs/{rel}\n{text[:5000]}")
    if chunks:
        header = json.dumps(
            {
                "requested_model_variant": model_selection.get("requested_model_variant"),
                "effective_model_variant": model_selection.get("effective_model_variant"),
                "fallback_used": model_selection.get("fallback_used"),
                "fallback_reason": model_selection.get("fallback_reason"),
                "allowed_variants": model_selection.get("available_variants") or sorted(DEIMV2_MODEL_VARIANTS),
            },
            ensure_ascii=False,
        )
        return (f"# deimv2_model_selection\n{header}\n\n" + "\n\n".join(chunks))[:20000]
    return (
        f"{template_config} includes the selected DEIMv2 DINOv3 defaults; "
        "dataset.yml overrides train_dataloader/val_dataloader total_batch_size and COCO paths; "
        "train.yml overrides epoches, output_dir, evaluator, checkpoint paths, and worker/device runtime settings."
    )


def _deimv2_template_config_context_paths(config_root: Path, template_rel: str) -> list[str]:
    template_path = config_root / template_rel
    ordered: list[str] = []
    seen: set[str] = set()

    def add_path(path: Path) -> None:
        try:
            rel = path.resolve().relative_to(config_root.resolve()).as_posix()
        except ValueError:
            return
        if rel in seen:
            return
        seen.add(rel)
        ordered.append(rel)

    def walk(path: Path) -> None:
        add_path(path)
        if not path.is_file():
            return
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        for raw_include in _deimv2_yaml_include_entries(text):
            include_path = Path(raw_include)
            if not include_path.is_absolute():
                include_path = path.parent / include_path
            walk(include_path)

    walk(template_path)
    return ordered or [template_rel]


def _deimv2_yaml_include_entries(text: str) -> list[str]:
    match = re.search(r"(?ms)^__include__\s*:\s*(\[[^\]]*\]|[^\n]+)", text or "")
    if not match:
        return []
    block = match.group(1)
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", block)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]
    return [item.strip().strip(",") for item in block.splitlines() if item.strip().strip(",")]


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


def _read_pipeline_paths(paths: ThreadPaths | TrainingRunPaths) -> dict[str, str]:
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

    output_root = _workflow_output_root(paths)
    thread_workspace = paths.thread.workspace if isinstance(paths, TrainingRunPaths) else paths.workspace
    preferred_deimv2_input = paths.workspace / "deimv2-training-input.json"
    preferred_training_input = paths.workspace / "gpu-training-orchestrator-training-input.json"
    if preferred_deimv2_input.is_file():
        training_input = preferred_deimv2_input
    elif preferred_training_input.is_file():
        training_input = preferred_training_input
    else:
        legacy_training_input = output_root / PIPELINE_WORK_DIR / "training_input.json"
        candidates = ([legacy_training_input] if legacy_training_input.is_file() else [])
        candidates += sorted(paths.workspace.rglob("*training-input.json"))
        if thread_workspace != paths.workspace:
            candidates += sorted(thread_workspace.rglob("*training-input.json"))
        candidates += sorted(output_root.rglob("training_input.json"))
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
    preferred_plan = output_root / PIPELINE_WORK_DIR / "synthetic_plan.json"
    if "synthetic_plan" not in result and preferred_plan.is_file():
        result["synthetic_plan"] = str(preferred_plan)
    elif "synthetic_plan" not in result:
        legacy_plan = output_root / "smoking_pipeline" / "synthetic_plan.json"
        plan_candidates = ([legacy_plan] if legacy_plan.is_file() else [])
        plan_candidates += sorted(paths.workspace.rglob("synthetic_plan.json"))
        if thread_workspace != paths.workspace:
            plan_candidates += sorted(thread_workspace.rglob("synthetic_plan.json"))
        plan_candidates += sorted(output_root.rglob("synthetic_plan.json"))
        if plan_candidates:
            result["synthetic_plan"] = str(plan_candidates[-1])
    return result


def _find_best_pt(paths: ThreadPaths | TrainingRunPaths) -> str:
    output_root = _workflow_output_root(paths)
    preferred = output_root / "training_run" / "train" / "weights" / "best.pt"
    if preferred.is_file():
        return str(preferred)
    candidates = sorted(output_root.rglob("best.pt"))
    return str(candidates[-1]) if candidates else ""


def _find_best_checkpoint(paths: ThreadPaths | TrainingRunPaths, training_backend: str = "yolo") -> str:
    if training_backend != "deimv2":
        return _find_best_pt(paths)
    run_dir = _workflow_output_root(paths) / "deimv2_training_run"
    for name in ("best_stg2.pth", "best_stg1.pth", "last.pth"):
        candidates = sorted(run_dir.rglob(name))
        if candidates:
            return str(candidates[-1])
    candidates = sorted(run_dir.rglob("*.pth"))
    return str(candidates[-1]) if candidates else ""


def _input_label(value: object) -> str:
    return {"dataset": "数据集", "image": "图片", "model_config": "模型配置"}.get(str(value), "输入")


def _filter_training_run_artifacts(outputs: list[Any], training_backend: str = "yolo") -> list[Any]:
    result: list[Any] = []
    run_dir = "deimv2_training_run" if training_backend == "deimv2" else "training_run"
    for artifact in outputs:
        path = str(getattr(artifact, "path", "") or "").replace("\\", "/")
        if f"/{WORKFLOW_OUTPUT_DIR}/" in path and f"/{run_dir}/" in path:
            result.append(artifact)
    return result


def _reference_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "reference_image_path.txt"


def _composite_image1_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "composite_image1_path.txt"


def _composite_image2_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "composite_image2_path.txt"


def _active_training_run_marker_path(paths: ThreadPaths) -> Path:
    return paths.workspace / "current_training_run.json"


def _data_reuse_mode(runtime_options: RuntimeOptions) -> str:
    value = getattr(runtime_options, "reuse_previous_data_preparation", "auto")
    if isinstance(value, bool):
        return "auto" if value else "never"
    normalized = str(value or "auto").strip().lower()
    return normalized if normalized in {"auto", "never", "required"} else "auto"


def _select_data_reuse_source(
    paths: ThreadPaths,
    active_run_paths: TrainingRunPaths | None,
    runtime_options: RuntimeOptions,
) -> TrainingRunPaths | None:
    if _data_reuse_mode(runtime_options) == "never":
        return None
    requested = str(getattr(runtime_options, "reuse_from_run_id", None) or "").strip()
    if requested:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", requested) or ".." in requested:
            return None
        return _load_training_run_paths(paths, requested)
    return active_run_paths


def _data_preparation_fingerprint(
    *,
    dataset_package: Path,
    image1: Path | None,
    image2: Path | None,
    labels: list[str],
    annotation_provider: str,
    generation_enabled: bool,
    max_synthetic_images: int,
) -> dict[str, Any]:
    payload = {
        "dataset_sha256": _file_sha256(dataset_package.resolve()) if dataset_package.is_file() else "",
        "image1_sha256": _file_sha256(image1.resolve()) if image1 is not None and image1.is_file() else "",
        "image2_sha256": _file_sha256(image2.resolve()) if image2 is not None and image2.is_file() else "",
        "labels": sorted({str(label).strip().casefold() for label in labels if str(label).strip()}),
        "annotation_provider": str(annotation_provider or "").strip().casefold(),
        "generation_enabled": bool(generation_enabled),
        "max_synthetic_images": int(max_synthetic_images),
    }
    real_inputs = _data_reuse_fingerprint_inputs(payload, "real")
    synthetic_inputs = _data_reuse_fingerprint_inputs(payload, "synthetic")
    return {
        "digest": _data_reuse_fingerprint_digest(payload),
        "inputs": payload,
        "real": {
            "digest": _data_reuse_fingerprint_digest(real_inputs),
            "inputs": real_inputs,
        },
        "synthetic": {
            "digest": _data_reuse_fingerprint_digest(synthetic_inputs),
            "inputs": synthetic_inputs,
        },
    }


def _data_reuse_fingerprint_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _data_reuse_fingerprint_inputs(payload: dict[str, Any], component: str) -> dict[str, Any]:
    real_keys = ("dataset_sha256", "labels", "annotation_provider")
    synthetic_keys = (
        "dataset_sha256",
        "image1_sha256",
        "image2_sha256",
        "labels",
        "annotation_provider",
        "generation_enabled",
        "max_synthetic_images",
    )
    keys = real_keys if component == "real" else synthetic_keys
    return {key: payload.get(key) for key in keys}


def _data_reuse_fingerprint_component(fingerprint: dict[str, Any], component: str) -> dict[str, Any]:
    stored = fingerprint.get(component)
    if isinstance(stored, dict) and stored.get("digest"):
        return stored
    inputs = fingerprint.get("inputs") if isinstance(fingerprint.get("inputs"), dict) else {}
    component_inputs = _data_reuse_fingerprint_inputs(inputs, component)
    return {
        "digest": _data_reuse_fingerprint_digest(component_inputs),
        "inputs": component_inputs,
    }


def _reuse_previous_data_preparation(
    *,
    source: TrainingRunPaths | None,
    target: TrainingRunPaths,
    fingerprint: dict[str, Any],
    mode: str,
) -> tuple[DataPreparationReuse | None, str]:
    if mode == "never":
        return None, "reuse_disabled"
    if source is None or source.run_id == target.run_id:
        return None, "previous_run_not_found"
    manifest_path = source.outputs / DATA_REUSE_MANIFEST_NAME
    manifest = _read_json_file(manifest_path)
    if not manifest:
        return None, "previous_data_preparation_not_reusable"
    previous_fingerprint = manifest.get("fingerprint") if isinstance(manifest.get("fingerprint"), dict) else {}
    previous_real_fingerprint = _data_reuse_fingerprint_component(previous_fingerprint, "real")
    current_real_fingerprint = _data_reuse_fingerprint_component(fingerprint, "real")
    if str(previous_real_fingerprint.get("digest") or "") != str(current_real_fingerprint.get("digest") or ""):
        return None, "real_annotation_fingerprint_mismatch"

    source_real_coco = source.outputs / "pipeline_work" / "real_coco.json"
    source_synthetic_coco = source.outputs / "pipeline_work" / "synthetic_coco.json"
    source_merged_coco = source.outputs / "pipeline_work" / "merged_coco.json"
    source_real_root = source.outputs / "uploaded_dataset"
    source_synthetic_root = source.outputs / "pipeline_work" / "synthetic_images"
    source_merged_root = source.outputs / "pipeline_work" / "merged_images"
    real_facts = _validated_coco_facts(source_real_coco, source_real_root)
    if real_facts is None or real_facts[0] <= 0:
        return None, "previous_real_coco_or_images_incomplete"
    synthetic_facts = _validated_coco_facts(source_synthetic_coco, source_synthetic_root)
    merged_facts = _validated_coco_facts(source_merged_coco, source_merged_root)
    synthetic_complete = bool(
        synthetic_facts
        and merged_facts
        and synthetic_facts[0] > 0
        and merged_facts[0] == real_facts[0] + synthetic_facts[0]
    )
    fingerprint_inputs = fingerprint.get("inputs") if isinstance(fingerprint.get("inputs"), dict) else {}
    generation_enabled = bool(fingerprint_inputs.get("generation_enabled"))
    previous_synthetic_fingerprint = _data_reuse_fingerprint_component(previous_fingerprint, "synthetic")
    current_synthetic_fingerprint = _data_reuse_fingerprint_component(fingerprint, "synthetic")
    synthetic_fingerprint_matches = (
        str(previous_synthetic_fingerprint.get("digest") or "")
        == str(current_synthetic_fingerprint.get("digest") or "")
    )
    synthetic_reused = generation_enabled and synthetic_complete and synthetic_fingerprint_matches
    reuse_level = "real_and_synthetic" if synthetic_reused else "real_only"
    selected_coco = source_merged_coco if synthetic_reused else source_real_coco
    selected_facts = merged_facts if synthetic_reused else real_facts
    expected_labels = {
        str(label).strip().casefold()
        for label in fingerprint_inputs.get("labels", [])
        if str(label).strip()
    }
    if expected_labels and _coco_category_names_for_reuse(selected_coco) != expected_labels:
        return None, "previous_coco_categories_mismatch"

    copy_paths = DATA_REUSE_COPY_PATHS if synthetic_reused else REAL_DATA_REUSE_COPY_PATHS

    staging_root = target.workspace / f".data-reuse-{os.urandom(4).hex()}"
    backup_root = target.workspace / f".data-reuse-backup-{os.urandom(4).hex()}"
    copied_files = 0
    committed: list[tuple[Path, Path | None]] = []
    try:
        for relative in copy_paths:
            source_path = source.outputs / relative
            if not source_path.exists():
                if relative.endswith("synthetic_plan.json"):
                    continue
                raise FileNotFoundError(str(source_path))
            staged_path = staging_root / relative
            copied_files += _copy_reuse_path(source_path, staged_path)

        for relative in copy_paths:
            staged_path = staging_root / relative
            if not staged_path.exists():
                continue
            target_path = target.outputs / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path: Path | None = None
            if target_path.exists():
                backup_path = backup_root / relative
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target_path), str(backup_path))
            shutil.move(str(staged_path), str(target_path))
            committed.append((target_path, backup_path))
    except (OSError, ValueError) as exc:
        for target_path, backup_path in reversed(committed):
            _remove_reuse_path(target_path)
            if backup_path is not None and backup_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(backup_path), str(target_path))
        return None, f"reuse_copy_failed:{exc}"
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)

    current_dataset_root = (
        target.outputs / "pipeline_work" / "merged_images"
        if synthetic_reused
        else target.outputs / "uploaded_dataset"
    )
    current_coco = (
        target.outputs / "pipeline_work" / "merged_coco.json"
        if synthetic_reused
        else target.outputs / "pipeline_work" / "real_coco.json"
    )
    provenance = {
        "schema": "jetlinks-data-preparation-reuse.v1",
        "source_run_id": source.run_id,
        "target_run_id": target.run_id,
        "fingerprint": fingerprint,
        "real_images": real_facts[0],
        "synthetic_images": synthetic_facts[0] if synthetic_reused and synthetic_facts else 0,
        "annotations": selected_facts[1],
        "copied_files": copied_files,
        "reuse_level": reuse_level,
        "synthetic_reused": synthetic_reused,
        "real_fingerprint_matched": True,
        "synthetic_fingerprint_matched": synthetic_fingerprint_matches,
        "synthetic_reuse_reason": (
            "reused"
            if synthetic_reused
            else (
                "generation_disabled"
                if not generation_enabled
                else ("synthetic_artifacts_incomplete" if not synthetic_complete else "synthetic_fingerprint_mismatch")
            )
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (target.outputs / "data_reuse_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return (
        DataPreparationReuse(
            source_run_id=source.run_id,
            dataset_root=current_dataset_root.resolve(),
            coco_json=current_coco.resolve(),
            real_images=real_facts[0],
            synthetic_images=synthetic_facts[0] if synthetic_reused and synthetic_facts else 0,
            annotations=selected_facts[1],
            copied_files=copied_files,
            reuse_level=reuse_level,
            synthetic_reused=synthetic_reused,
        ),
        "reused",
    )


def _copy_reuse_path(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, copy_function=shutil.copy2)
        return sum(1 for path in target.rglob("*") if path.is_file())
    shutil.copy2(source, target)
    return 1


def _remove_reuse_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _validated_coco_facts(coco_path: Path, image_root: Path) -> tuple[int, int] | None:
    payload = _read_json_file(coco_path)
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    annotations = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    categories = payload.get("categories") if isinstance(payload.get("categories"), list) else []
    if not coco_path.is_file() or not image_root.is_dir() or not images or not categories:
        return None
    available_names = {
        path.name.casefold()
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS
    }
    for image in images:
        if not isinstance(image, dict):
            return None
        file_name = str(image.get("file_name") or "").replace("\\", "/").strip()
        if not file_name:
            return None
        direct = image_root / file_name
        if not direct.is_file() and Path(file_name).name.casefold() not in available_names:
            return None
    return len(images), len([item for item in annotations if isinstance(item, dict)])


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coco_category_names_for_reuse(path: Path) -> set[str]:
    payload = _read_json_file(path)
    categories = payload.get("categories") if isinstance(payload.get("categories"), list) else []
    return {
        str(item.get("name") or "").strip().casefold()
        for item in categories
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }


def _effective_synthetic_generation(
    requested: bool,
    image1: str | None,
    image2: str | None,
) -> tuple[bool, str]:
    if not requested:
        return False, "not_requested"
    if not image1 or not image2:
        return False, "sufficient_data_samples"
    return True, ""


def _mark_reused_data_preparation_summary(run_paths: TrainingRunPaths, reused: DataPreparationReuse) -> None:
    summary_path = run_paths.outputs / "prepared_data" / "data_preparation_summary.json"
    summary = _read_json_file(summary_path)
    if not summary:
        return
    summary["real_coco"] = str((run_paths.outputs / "pipeline_work" / "real_coco.json").resolve())
    summary["data_reuse"] = {
        "source_run_id": reused.source_run_id,
        "real_images": reused.real_images,
        "synthetic_images": reused.synthetic_images,
        "annotations": reused.annotations,
        "copied_files": reused.copied_files,
        "reuse_level": reused.reuse_level,
        "synthetic_reused": reused.synthetic_reused,
    }
    if reused.synthetic_reused:
        summary.update(
            {
                "synthetic_coco": str((run_paths.outputs / "pipeline_work" / "synthetic_coco.json").resolve()),
                "synthetic_images": str((run_paths.outputs / "pipeline_work" / "synthetic_images").resolve()),
                "synthetic_plan": str((run_paths.outputs / "pipeline_work" / "synthetic_plan.json").resolve())
                if (run_paths.outputs / "pipeline_work" / "synthetic_plan.json").is_file()
                else "",
                "synthetic_generation_status": "reused",
            }
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_data_preparation_reuse_manifest(
    run_paths: TrainingRunPaths,
    *,
    fingerprint: dict[str, Any],
    source_run_id: str,
) -> None:
    real_facts = _validated_coco_facts(
        run_paths.outputs / "pipeline_work" / "real_coco.json",
        run_paths.outputs / "uploaded_dataset",
    )
    synthetic_facts = _validated_coco_facts(
        run_paths.outputs / "pipeline_work" / "synthetic_coco.json",
        run_paths.outputs / "pipeline_work" / "synthetic_images",
    )
    merged_facts = _validated_coco_facts(
        run_paths.outputs / "pipeline_work" / "merged_coco.json",
        run_paths.outputs / "pipeline_work" / "merged_images",
    )
    real_complete = bool(real_facts and real_facts[0] > 0)
    synthetic_complete = bool(
        real_complete
        and synthetic_facts
        and merged_facts
        and synthetic_facts[0] > 0
        and merged_facts[0] == real_facts[0] + synthetic_facts[0]
    )
    reusable = real_complete
    reuse_level = "real_and_synthetic" if synthetic_complete else ("real_only" if real_complete else "none")
    payload = {
        "schema": "jetlinks-data-preparation-reuse-manifest.v1",
        "run_id": run_paths.run_id,
        "source_run_id": source_run_id or None,
        "reusable": reusable,
        "reason": "complete" if synthetic_complete else ("real_annotations_complete" if real_complete else "real_artifacts_incomplete"),
        "reuse_level": reuse_level,
        "fingerprint": fingerprint,
        "counts": {
            "real_images": real_facts[0] if real_facts else 0,
            "synthetic_images": synthetic_facts[0] if synthetic_facts else 0,
            "merged_images": merged_facts[0] if merged_facts else 0,
            "annotations": merged_facts[1] if synthetic_complete and merged_facts else (real_facts[1] if real_facts else 0),
        },
        "artifacts": list(DATA_REUSE_COPY_PATHS if synthetic_complete else REAL_DATA_REUSE_COPY_PATHS),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest_path = run_paths.outputs / DATA_REUSE_MANIFEST_NAME
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)


def _training_runs_workspace_root(paths: ThreadPaths) -> Path:
    return paths.workspace / "runs"


def _training_runs_output_root(paths: ThreadPaths) -> Path:
    return paths.outputs / WORKFLOW_OUTPUT_DIR / "runs"


def _make_training_run_id(training_backend: str) -> str:
    prefix = "deimv2" if training_backend == "deimv2" else "yolo"
    return f"run-{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"


def _create_training_run_paths(paths: ThreadPaths, training_backend: str) -> TrainingRunPaths:
    run_id = _make_training_run_id(training_backend)
    run_paths = TrainingRunPaths(
        thread=paths,
        run_id=run_id,
        workspace=_training_runs_workspace_root(paths) / run_id,
        outputs=_training_runs_output_root(paths) / run_id,
    )
    run_paths.workspace.mkdir(parents=True, exist_ok=True)
    run_paths.outputs.mkdir(parents=True, exist_ok=True)
    return run_paths


def _load_training_run_paths(paths: ThreadPaths, run_id: str) -> TrainingRunPaths | None:
    value = str(run_id or "").strip()
    if not value:
        return None
    run_paths = TrainingRunPaths(
        thread=paths,
        run_id=value,
        workspace=_training_runs_workspace_root(paths) / value,
        outputs=_training_runs_output_root(paths) / value,
    )
    if not run_paths.workspace.exists() and not run_paths.outputs.exists():
        return None
    return run_paths


def _save_active_training_run(paths: ThreadPaths, run_paths: TrainingRunPaths, *, training_backend: str, objective: str = "") -> None:
    payload = {
        "run_id": run_paths.run_id,
        "training_backend": training_backend,
        "objective": objective,
        "workspace": str(run_paths.workspace),
        "outputs": str(run_paths.outputs),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _active_training_run_marker_path(paths).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_active_training_run_paths(paths: ThreadPaths) -> TrainingRunPaths | None:
    marker = _active_training_run_marker_path(paths)
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _load_training_run_paths(paths, str(payload.get("run_id") or ""))


def _http_training_cancel_requested(paths: ThreadPaths, run_paths: TrainingRunPaths | None = None) -> bool:
    try:
        marker = read_current_http_training_job_marker(paths.thread_id, require_active=False)
    except Exception:
        return False
    if not isinstance(marker, dict):
        return False
    status = str(marker.get("status") or "").strip().lower()
    if status not in {"cancelling", "cancelled"}:
        return False
    marker_run_id = str(marker.get("run_id") or "").strip()
    current_run_id = str(getattr(run_paths, "run_id", "") or "").strip()
    if marker_run_id:
        return marker_run_id == current_run_id
    return status == "cancelling"


def _cancelled_training_result(
    *,
    recorder: EventRecorder,
    agent_name: str,
    thread_id: str,
    workflow: str,
    run_paths: TrainingRunPaths,
    phase: str,
    reply: str,
    artifacts: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[AgentRunResult, list[ChatEvent]]:
    _set_waiting_prompt(run_paths, False)
    _set_workflow_completed(run_paths, False)
    result = AgentRunResult(
        agent=agent_name,
        thread_id=thread_id,
        status="failed",
        reply=reply,
        artifacts=list(artifacts or []),
        verification=VerificationResult(passed=False, retry_count=0, checks=[], failed_checks=["training cancelled"]),
        metadata={
            "workflow": workflow,
            "run_id": run_paths.run_id,
            "run_output_dir": str(run_paths.outputs),
            "phase": phase,
            "cancelled": True,
            **(metadata or {}),
        },
    )
    recorder.emit("agent.message", {"text": reply})
    recorder.emit("run.failed", {"result": result.model_dump(), "error": reply})
    return result, recorder.events


def _run_aware_marker_path(paths: ThreadPaths | TrainingRunPaths, filename: str) -> Path:
    return paths.workspace / filename


def _legacy_thread_marker_path(paths: ThreadPaths | TrainingRunPaths, filename: str) -> Path:
    thread = paths.thread if isinstance(paths, TrainingRunPaths) else paths
    return thread.workspace / filename


def _read_text_marker(paths: ThreadPaths | TrainingRunPaths, filename: str) -> str:
    marker = _run_aware_marker_path(paths, filename)
    if marker.exists():
        return marker.read_text(encoding="utf-8").strip()
    legacy = _legacy_thread_marker_path(paths, filename)
    # 兼容早期还没有 run 级工作区时创建的线程。
    # 新写入始终进入当前 run 工作区。
    if legacy != marker and legacy.exists():
        return legacy.read_text(encoding="utf-8").strip()
    return ""


def _read_json_marker(paths: ThreadPaths | TrainingRunPaths, filename: str) -> Any:
    marker = _run_aware_marker_path(paths, filename)
    if not marker.exists():
        legacy = _legacy_thread_marker_path(paths, filename)
        marker = legacy if legacy != marker and legacy.exists() else marker
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None


def _prompt_marker_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
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


def _save_generation_prompt(paths: ThreadPaths | TrainingRunPaths, prompt: str) -> None:
    if prompt:
        _prompt_marker_path(paths).write_text(prompt.strip(), encoding="utf-8")


def _load_generation_prompt(paths: ThreadPaths | TrainingRunPaths) -> str:
    return _read_text_marker(paths, "generation_prompt.txt")


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


def _labels_marker_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
    return paths.workspace / "annotation_labels.json"


def _save_annotation_labels(paths: ThreadPaths | TrainingRunPaths, labels: list[str]) -> None:
    normalized = _normalize_detection_labels(labels)
    if normalized:
        _labels_marker_path(paths).write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_annotation_labels(paths: ThreadPaths | TrainingRunPaths) -> list[str]:
    payload = _read_json_marker(paths, "annotation_labels.json")
    if payload is None:
        return []
    return _normalize_detection_labels([str(item).strip() for item in payload if str(item).strip()]) if isinstance(payload, list) else []


def _training_config_marker_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
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
    uploads_root = paths.uploads.resolve()
    models_root = (uploads_root / "models").resolve()
    try:
        resolved.relative_to(models_root)
    except ValueError:
        return None, "模型文件不在当前线程的 uploads/models 目录中"
    suffix = resolved.suffix.lower()
    if not resolved.is_file() or suffix not in SUPPORTED_TRAINING_MODEL_SUFFIXES:
        return None, "模型文件必须是有效的 .pt 或 .pth 文件"
    if resolved.stat().st_size <= 0:
        return None, "模型文件为空"

    actual_sha256 = _file_sha256(resolved)
    expected_sha256 = str(record.get("sha256") or "").strip().lower()
    if expected_sha256 and expected_sha256 != actual_sha256:
        return None, "模型文件 SHA256 与上传记录不一致"
    relative = resolved.relative_to(uploads_root).as_posix()
    return {
        **record,
        "source": "user_upload",
        "name": str(record.get("name") or resolved.name),
        "path": f"/mnt/user-data/uploads/{relative}",
        "local_path": str(resolved),
        "mime_type": "application/octet-stream",
        "size": resolved.stat().st_size,
        "sha256": actual_sha256,
        "extension": suffix,
    }, ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _user_training_model_suffix(model_record: dict[str, Any]) -> str:
    value = str(model_record.get("extension") or Path(str(model_record.get("local_path") or model_record.get("path") or "")).suffix)
    return value.lower()


def _apply_user_training_model(
    training_cfg: dict[str, Any],
    model_record: dict[str, Any],
    training_backend: str = "yolo",
) -> str:
    if training_backend == "deimv2":
        return _apply_user_training_model_for_deimv2(training_cfg, model_record)
    return _apply_user_training_model_for_yolo(training_cfg, model_record)


def _apply_common_user_training_model_metadata(training: dict[str, Any], model_record: dict[str, Any]) -> None:
    training["model_source"] = "user_upload"
    training["strict_model"] = True
    training["model_sha256"] = str(model_record["sha256"])
    training["model_original_name"] = str(model_record.get("name") or "")
    training["model_id"] = str(model_record.get("modelId") or "")
    training["model_extension"] = _user_training_model_suffix(model_record)


def _apply_user_training_model_for_yolo(training_cfg: dict[str, Any], model_record: dict[str, Any]) -> str:
    training = training_cfg.get("training")
    if not isinstance(training, dict):
        return ""
    if _user_training_model_suffix(model_record) != ".pt":
        return "YOLO 训练模型必须是 .pt 文件"
    # Ultralytics 可以直接把检测器 .pt 作为训练入口模型。
    training["model"] = str(model_record["local_path"])
    _apply_common_user_training_model_metadata(training, model_record)
    return ""


def _apply_user_training_model_for_deimv2(training_cfg: dict[str, Any], model_record: dict[str, Any]) -> str:
    training = training_cfg.get("training")
    if not isinstance(training, dict):
        return ""
    suffix = _user_training_model_suffix(model_record)
    local_path = str(model_record["local_path"])
    if suffix == ".pt":
        # 在 DEIMv2 DINOv3 中，.pt 表示 backbone checkpoint，不是完整检测器。
        # 因此关闭 tuning，避免 runner 又加载旧的 .pth 检测器。
        training["backbone_checkpoint"] = local_path
        training["tuning_checkpoint"] = ""
        training["disable_tuning_checkpoint"] = True
        training["deimv2_upload_model_usage"] = "backbone_checkpoint"
    elif suffix == ".pth":
        # .pth 表示完整检测器 checkpoint，用于微调。
        # DINOv3 backbone 仍由 runner 配置单独解析。
        training["tuning_checkpoint"] = local_path
        training["disable_tuning_checkpoint"] = False
        training["deimv2_upload_model_usage"] = "tuning_checkpoint"
    else:
        return "DEIMv2 训练模型必须是 .pt 或 .pth 文件"
    training.pop("model", None)
    _apply_common_user_training_model_metadata(training, model_record)
    return ""


def _user_training_model_usage(training_backend: str, model_record: dict[str, Any]) -> str:
    suffix = _user_training_model_suffix(model_record)
    if training_backend == "deimv2":
        if suffix == ".pt":
            return "backbone_checkpoint"
        if suffix == ".pth":
            return "tuning_checkpoint"
    return "model"


def _clear_user_training_model_selection(
    training_cfg: dict[str, Any],
    request_spec: dict[str, Any],
    training_backend: str = "yolo",
) -> None:
    if training_backend == "deimv2":
        _clear_deimv2_user_training_model_selection(training_cfg, request_spec)
        return
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
    for key in ("model_source", "strict_model", "model_sha256", "model_original_name", "model_id", "model_extension"):
        training.pop(key, None)


def _clear_deimv2_user_training_model_selection(
    training_cfg: dict[str, Any],
    request_spec: dict[str, Any],
) -> None:
    training = training_cfg.get("training")
    if not isinstance(training, dict):
        return

    def is_upload_path(value: Any) -> bool:
        return "/uploads/models/" in str(value or "").replace("\\", "/").lower()

    has_user_selection = bool(
        str(training.get("model_source") or "").strip() == "user_upload"
        or training.get("strict_model")
        or str(training.get("model_id") or "").strip()
        or str(training.get("deimv2_upload_model_usage") or "").strip()
        or training.get("disable_tuning_checkpoint")
        or is_upload_path(training.get("backbone_checkpoint"))
        or is_upload_path(training.get("tuning_checkpoint"))
    )
    if not has_user_selection:
        return

    request_training = request_spec.get("training")
    if not isinstance(request_training, dict):
        request_training = {}

    request_has_user_selection = bool(
        str(request_training.get("model_source") or "").strip() == "user_upload"
        or request_training.get("strict_model")
        or str(request_training.get("model_id") or "").strip()
        or is_upload_path(request_training.get("backbone_checkpoint"))
        or is_upload_path(request_training.get("tuning_checkpoint"))
    )
    if request_has_user_selection:
        request_training = {}

    if is_upload_path(training.get("backbone_checkpoint")):
        training["backbone_checkpoint"] = str(request_training.get("backbone_checkpoint") or DEIMV2_BACKBONE_CHECKPOINT)
    if is_upload_path(training.get("tuning_checkpoint")) or training.get("disable_tuning_checkpoint"):
        training["tuning_checkpoint"] = str(request_training.get("tuning_checkpoint") or "")
    for key in (
        "model_source",
        "strict_model",
        "model_sha256",
        "model_original_name",
        "model_id",
        "model_extension",
        "deimv2_upload_model_usage",
        "disable_tuning_checkpoint",
    ):
        training.pop(key, None)


def _save_training_config(paths: ThreadPaths | TrainingRunPaths, training_cfg: dict[str, Any]) -> None:
    if training_cfg:
        _training_config_marker_path(paths).write_text(json.dumps(training_cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_training_config(paths: ThreadPaths | TrainingRunPaths) -> dict[str, Any]:
    payload = _read_json_marker(paths, "training_config.json")
    if payload is None:
        return {}
    return payload if isinstance(payload, dict) else {}


def _detection_task_marker_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
    return paths.workspace / "detection_task_description.txt"


def _save_detection_task_description(paths: ThreadPaths | TrainingRunPaths, task_description: str) -> None:
    if task_description:
        _detection_task_marker_path(paths).write_text(task_description.strip(), encoding="utf-8")


def _load_detection_task_description(paths: ThreadPaths | TrainingRunPaths) -> str:
    return _read_text_marker(paths, "detection_task_description.txt")


def _synthetic_generation_marker_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
    return paths.workspace / "synthetic_generation_enabled.json"


def _save_synthetic_generation_enabled(paths: ThreadPaths | TrainingRunPaths, enabled: bool) -> None:
    _synthetic_generation_marker_path(paths).write_text(json.dumps(bool(enabled)), encoding="utf-8")


def _validate_attachment_thread_paths(paths: ThreadPaths, attachments: list[Attachment]) -> str:
    current_thread_id = str(paths.thread_id or "").strip()
    if not current_thread_id:
        return ""
    for item in attachments:
        raw_path = str(getattr(item, "path", "") or "").strip()
        if not raw_path:
            continue
        normalized = raw_path.replace("\\", "/")
        match = re.search(r"(?:^|/)\.runtime/threads/([^/]+)/", normalized, flags=re.IGNORECASE)
        if match and match.group(1) != current_thread_id:
            return (
                "上传文件路径与当前会话 threadId 不一致："
                f"当前 threadId={current_thread_id}，文件路径属于 threadId={match.group(1)}，"
                "请重新上传文件或把 runtimeOptions.files 中的 path/fileUrl 改为当前线程目录。"
            )
    return ""


def _load_synthetic_generation_enabled(paths: ThreadPaths | TrainingRunPaths) -> bool | None:
    payload = _read_json_marker(paths, "synthetic_generation_enabled.json")
    if payload is None:
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


def _waiting_prompt_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
    return paths.workspace / "awaiting_prompt.flag"


def _workflow_completed_path(paths: ThreadPaths | TrainingRunPaths) -> Path:
    return paths.workspace / "workflow_completed.flag"


def _set_workflow_completed(paths: ThreadPaths | TrainingRunPaths, completed: bool) -> None:
    flag = _workflow_completed_path(paths)
    if completed:
        flag.write_text("1", encoding="utf-8")
    elif flag.exists():
        flag.unlink()


def _is_workflow_completed(paths: ThreadPaths | TrainingRunPaths) -> bool:
    if _workflow_completed_path(paths).exists():
        return True
    output_root = _workflow_output_root(paths)
    completion_markers = (
        output_root / "training_run" / "run_summary.json",
        output_root / "training_run" / "train" / "weights" / "best.pt",
        output_root / "deimv2_training_run" / "run_summary.json",
        output_root / "deimv2_training_run" / "training_summary.json",
        output_root / "prepared_data" / "data_preparation_summary.json",
    )
    return any(marker.exists() for marker in completion_markers)


def _set_waiting_prompt(paths: ThreadPaths | TrainingRunPaths, waiting: bool) -> None:
    flag = _waiting_prompt_path(paths)
    if waiting:
        flag.write_text("1", encoding="utf-8")
    elif flag.exists():
        flag.unlink()


def _is_waiting_prompt(paths: ThreadPaths | TrainingRunPaths) -> bool:
    if _waiting_prompt_path(paths).exists():
        return True
    if isinstance(paths, TrainingRunPaths):
        legacy = _waiting_prompt_path(paths.thread)
        return legacy.exists()
    return False


def _reset_completed_request_state_for_new_training(paths: ThreadPaths) -> None:
    _set_workflow_completed(paths, False)
    _set_waiting_prompt(paths, False)
    for marker in (
        _training_config_marker_path(paths),
        _prompt_marker_path(paths),
        _labels_marker_path(paths),
        _detection_task_marker_path(paths),
        _synthetic_generation_marker_path(paths),
    ):
        if marker.exists():
            marker.unlink()


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
