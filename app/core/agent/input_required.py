from __future__ import annotations

from typing import Any

from app.schemas import AgentRunResult, Attachment, ChatRequest


# This module centralizes "stop and ask for more input" decisions. The runtime
# uses it both before execution starts and after a run completes so the same
# input contract applies across workflows, tools, and direct LLM replies.
_DEFAULT_ACCEPT_BY_TYPE = {
    "file": "*/*",
    "image": "image/*",
    "video": "video/*",
    "audio": "audio/*",
    "dataset": ".zip,.tar,.tar.gz,.csv,.json,.jsonl,.parquet,.yaml,.yml",
    "model": ".onnx,.pt,.pth,.bin,.safetensors,.gguf,.pkl,.joblib",
    "model_config": ".json,.yaml,.yml,.toml",
    "text": "text/plain",
    "json": "application/json",
    "number": "number",
    "boolean": "boolean",
    "secret": "password",
}
_KNOWN_TYPES = set(_DEFAULT_ACCEPT_BY_TYPE)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_DATASET_EXTENSIONS = {".zip", ".tar", ".gz", ".csv", ".json", ".jsonl", ".parquet", ".yaml", ".yml"}
_MODEL_EXTENSIONS = {".onnx", ".pt", ".pth", ".bin", ".safetensors", ".gguf", ".pkl", ".joblib"}
_MODEL_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml"}


def required_inputs_for_result(result: AgentRunResult, request: ChatRequest) -> list[dict[str, Any]]:
    # Prefer explicit tool/workflow metadata, then fall back to heuristics over
    # the reply text when older skills still return plain-language prompts.
    explicit = _explicit_required_inputs(result.metadata)
    if explicit:
        return explicit
    inferred = _infer_required_inputs(result, request)
    if inferred:
        return inferred
    if result.metadata.get("requires_input") is True:
        return [_requirement("file", reason="The agent requires additional user input.")]
    return []


def required_inputs_for_request(request: ChatRequest) -> list[dict[str, Any]]:
    # Preflight guards catch common long-running flows before the agent spends
    # tool rounds only to discover that basic files were never provided.
    selected_skills = {name.strip() for name in request.runtime_options.selected_skills if name.strip()}
    if _requires_algorithm_training_inputs(request, selected_skills) and not _has_algorithm_training_input(request):
        return [
            _requirement(
                "dataset",
                reason=(
                    "Algorithm training needs base data before execution: upload a YOLO dataset/data.yaml, "
                    "a labeled image dataset archive, or provide an accessible data.yaml/dataset path."
                ),
            )
        ]
    if _requires_data_auto_annotation(request, selected_skills) and not _has_attachment(request.attachments, "image"):
        return [_requirement("image", reason="Data auto annotation requires an uploaded image.")]
    return []


def _requires_data_auto_annotation(request: ChatRequest, selected_skills: set[str]) -> bool:
    if "data-auto-annotation" in selected_skills:
        if _is_algorithm_training_flow(selected_skills):
            text = _last_user_text(request).lower()
            annotation_markers: tuple[str, ...] = ("自动标注", "预标注", "sam3", "coco", "label", "labels", "annotation")
            image_markers: tuple[str, ...] = ("图片", "图像", "照片", "截图", "image", "photo", "picture")
            return _mentions_any(text, annotation_markers) and _mentions_any(text, image_markers)
        return True
    text = _last_user_text(request).lower()
    if not text:
        return False
    annotation_markers = ("自动标注", "预标注", "目标检测", "sam3", "coco", "label", "labels", "annotation")
    image_markers = ("图片", "图像", "照片", "截图", "image", "photo", "picture")
    return _mentions_any(text, annotation_markers) and _mentions_any(text, image_markers)


def _requires_algorithm_training_inputs(request: ChatRequest, selected_skills: set[str]) -> bool:
    if not _is_algorithm_training_flow(selected_skills):
        return False
    if _is_algorithm_research_only_flow(selected_skills):
        return False
    if _is_synthetic_smoking_training_app(request, selected_skills):
        return False
    if _is_algorithm_execution_flow(selected_skills):
        return True
    text = _last_user_text(request).lower()
    training_markers = (
        "训练",
        "全流程",
        "benchmark",
        "best.pt",
        "baseline",
        "data.yaml",
        "数据治理",
        "算法工程师",
        "棕榈果",
    )
    return _mentions_any(text, training_markers)


def _is_synthetic_smoking_training_app(request: ChatRequest, selected_skills: set[str]) -> bool:
    if (request.runtime_options.app_template_name or "").strip() != "algorithm-engineer-full-cycle-test":
        return False
    required = {"image-dataset-generation", "data-auto-annotation", "gpu-training-orchestrator"}
    if not required.issubset(selected_skills):
        return False
    raw_contexts = request.runtime_options.config_options.get("composite_skills")
    if not isinstance(raw_contexts, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("name") or "").strip() == "smoking_detection_agent_loop"
        for item in raw_contexts
    )


def _is_algorithm_execution_flow(selected_skills: set[str]) -> bool:
    execution_skills = {
        "algorithm-engineer",
        "dataset-curator",
        "gpu-training-orchestrator",
        "cpu-training-runner",
        "detector-evaluator",
        "deployment-candidate-reviewer",
        "experiment-ledger",
    }
    return bool(selected_skills & execution_skills)


def _is_algorithm_research_only_flow(selected_skills: set[str]) -> bool:
    research_skills = {"algorithm-research-scout", "model-candidate-selector"}
    return bool(selected_skills) and selected_skills <= research_skills


def _is_algorithm_training_flow(selected_skills: set[str]) -> bool:
    training_skills = {
        "algorithm-engineer",
        "dataset-curator",
        "algorithm-research-scout",
        "model-candidate-selector",
        "gpu-training-orchestrator",
        "cpu-training-runner",
        "detector-evaluator",
        "experiment-ledger",
    }
    return bool(selected_skills & training_skills)


def _has_algorithm_training_input(request: ChatRequest) -> bool:
    if _has_attachment(request.attachments, "dataset") or _has_attachment(request.attachments, "image"):
        return True
    text = _last_user_text(request)
    if _mentions_dataset_path(text):
        return True
    return False


def _mentions_dataset_path(text: str) -> bool:
    lowered = text.lower()
    if "data.yaml" in lowered or "data.yml" in lowered:
        return True
    if "mock=true" in lowered:
        return True
    path_markers = ("/data/", "/mnt/", "/home/", "/users/", "/workspace/", "/datasets/")
    return any(marker in lowered for marker in path_markers)


def _last_user_text(request: ChatRequest) -> str:
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content.strip()
    return ""


def _explicit_required_inputs(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_inputs = metadata.get("required_inputs")
    if not isinstance(raw_inputs, list):
        return []
    normalized = [_normalize_requirement(item) for item in raw_inputs]
    return [item for item in normalized if item is not None]


def _normalize_requirement(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_type = value.get("type")
    input_type = raw_type if isinstance(raw_type, str) and raw_type in _KNOWN_TYPES else "file"
    requirement = {key: item for key, item in value.items() if item is not None}
    requirement["type"] = input_type
    requirement.setdefault("accept", _DEFAULT_ACCEPT_BY_TYPE[input_type])
    requirement.setdefault("required", True)
    return requirement


def _infer_required_inputs(result: AgentRunResult, request: ChatRequest) -> list[dict[str, Any]]:
    reply_text = result.reply.lower()
    if not _looks_like_input_request(reply_text):
        return []

    requirements: list[dict[str, Any]] = []
    if _mentions_any(reply_text, ("图片", "图像", "照片", "截图", "image", "photo", "picture")) and not _has_attachment(
        request.attachments, "image"
    ):
        requirements.append(_requirement("image", reason="The agent requires an uploaded image."))
    if _mentions_any(reply_text, ("视频", "video")) and not _has_attachment(request.attachments, "video"):
        requirements.append(_requirement("video", reason="The agent requires an uploaded video."))
    if _mentions_any(reply_text, ("音频", "语音", "audio")) and not _has_attachment(request.attachments, "audio"):
        requirements.append(_requirement("audio", reason="The agent requires an uploaded audio file."))
    if _mentions_any(reply_text, ("数据集", "dataset", "coco json", "coco-json")) and not _has_attachment(
        request.attachments, "dataset"
    ):
        requirements.append(_requirement("dataset", reason="The agent requires an uploaded dataset."))
    if _mentions_any(reply_text, ("模型配置", "model config", "model_config")) and not _has_attachment(
        request.attachments, "model_config"
    ):
        requirements.append(_requirement("model_config", reason="The agent requires a model configuration file."))
    if (
        _mentions_any(reply_text, ("模型文件", "权重", "model weights", "model file", "checkpoint"))
        and not _has_attachment(request.attachments, "model")
    ):
        requirements.append(_requirement("model", reason="The agent requires an uploaded model file."))
    if requirements:
        return requirements
    if _looks_like_generic_file_request(reply_text) and not request.attachments:
        return [_requirement("file", reason="The agent requires an uploaded file.")]
    return []


def _requirement(input_type: str, *, reason: str) -> dict[str, Any]:
    return {
        "type": input_type,
        "accept": _DEFAULT_ACCEPT_BY_TYPE[input_type],
        "required": True,
        "reason": reason,
    }


def _looks_like_input_request(text: str) -> bool:
    return _mentions_any(
        text,
        (
            "待上传",
            "请上传",
            "重新上传",
            "没有检测到",
            "没有收到",
            "没有附件",
            "缺少",
            "需要提供",
            "需要上传",
            "requires",
            "required",
            "missing",
        ),
    )


def _looks_like_generic_file_request(text: str) -> bool:
    return _mentions_any(
        text,
        (
            "请上传文件",
            "请重新上传文件",
            "需要上传文件",
            "需要提供文件",
            "没有收到文件",
            "没有检测到文件",
            "没有附件",
            "缺少文件",
            "upload a file",
            "upload the file",
            "please upload",
            "requires an uploaded file",
            "requires a file",
            "file is required",
            "attachment is required",
            "missing file",
            "missing attachment",
        ),
    )


def _mentions_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _has_attachment(attachments: list[Attachment], input_type: str) -> bool:
    return any(_attachment_matches(attachment, input_type) for attachment in attachments)


def _attachment_matches(attachment: Attachment, input_type: str) -> bool:
    mime_type = (attachment.mime_type or "").lower()
    name = attachment.name.lower()
    metadata_type = str(attachment.metadata.get("acp_type") or attachment.metadata.get("type") or "").lower()
    if input_type == "image":
        return metadata_type == "image" or mime_type.startswith("image/") or _has_extension(name, _IMAGE_EXTENSIONS)
    if input_type == "video":
        return metadata_type == "video" or mime_type.startswith("video/") or _has_extension(name, _VIDEO_EXTENSIONS)
    if input_type == "audio":
        return metadata_type == "audio" or mime_type.startswith("audio/") or _has_extension(name, _AUDIO_EXTENSIONS)
    if input_type == "dataset":
        return metadata_type == "dataset" or _has_extension(name, _DATASET_EXTENSIONS)
    if input_type == "model":
        return metadata_type == "model" or _has_extension(name, _MODEL_EXTENSIONS)
    if input_type == "model_config":
        return metadata_type == "model_config" or ("config" in name and _has_extension(name, _MODEL_CONFIG_EXTENSIONS))
    return bool(attachment.path or attachment.data_base64)


def _has_extension(name: str, extensions: set[str]) -> bool:
    return any(name.endswith(extension) for extension in extensions)
