from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.artifacts import ArtifactStore


PREVIEW_RESIZE_MODE = "normalized_stretch"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_AGGREGATE_COCO_NAMES = {
    "real_coco.json",
    "synthetic_coco.json",
    "merged_coco.json",
    "training_coco.json",
}


def add_annotation_preview_to_update(
    update: dict[str, Any],
    *,
    thread_id: str,
    artifact_store: ArtifactStore,
    preview_width: int | None = None,
    preview_height: int | None = None,
) -> dict[str, Any]:
    artifact = update.get("artifact")
    if not isinstance(artifact, dict):
        return update
    virtual_path = str(artifact.get("path") or "").strip()
    if not virtual_path:
        return update
    try:
        coco_path = artifact_store.resolve_virtual_path(thread_id, virtual_path)
        annotation_preview = build_annotation_preview(
            thread_id=thread_id,
            coco_path=coco_path,
            coco_artifact=artifact,
            artifact_store=artifact_store,
            preview_width=preview_width,
            preview_height=preview_height,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return update
    if annotation_preview is None:
        return update
    enriched = dict(update)
    enriched["annotationPreview"] = annotation_preview
    return enriched


def build_annotation_preview(
    *,
    thread_id: str,
    coco_path: Path,
    coco_artifact: dict[str, Any],
    artifact_store: ArtifactStore,
    preview_width: int | None = None,
    preview_height: int | None = None,
) -> dict[str, Any] | None:
    if not _is_per_image_coco(coco_path):
        return None
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        return None
    file_name = str(images[0].get("file_name") or "").replace("\\", "/").strip()
    if not file_name:
        return None
    source_image = _find_source_image(coco_path, file_name)
    if source_image is None:
        return None

    paths = artifact_store.prepare_thread(thread_id)
    source_image.resolve().relative_to(paths.outputs.resolve())
    with Image.open(source_image) as image:
        actual_width, actual_height = image.size
    if actual_width <= 0 or actual_height <= 0:
        return None

    source_artifact = artifact_store.to_artifact_ref(thread_id, source_image).model_dump()
    original_coco = dict(coco_artifact)
    original_coco.update({"coordinate_width": actual_width, "coordinate_height": actual_height})
    original_image = {
        **source_artifact,
        "width": actual_width,
        "height": actual_height,
    }
    result: dict[str, Any] = {
        "source": _annotation_source(coco_path, payload),
        "original": {
            "image": original_image,
            "coco": original_coco,
        },
    }
    if preview_width is None or preview_height is None:
        return result
    if preview_width <= 0 or preview_height <= 0:
        return result

    resized_image, resized_coco = _ensure_resized_pair(
        coco_path=coco_path,
        source_image=source_image,
        payload=payload,
        source=_annotation_source(coco_path, payload),
        original_width=actual_width,
        original_height=actual_height,
        target_width=preview_width,
        target_height=preview_height,
    )
    resized_image_artifact = artifact_store.to_artifact_ref(thread_id, resized_image).model_dump()
    resized_coco_artifact = artifact_store.to_artifact_ref(thread_id, resized_coco).model_dump()
    result["resized"] = {
        "image": {
            **resized_image_artifact,
            "width": preview_width,
            "height": preview_height,
        },
        "coco": {
            **resized_coco_artifact,
            "coordinate_width": preview_width,
            "coordinate_height": preview_height,
        },
    }
    result["transform"] = {
        "mode": PREVIEW_RESIZE_MODE,
        "scale_x": preview_width / actual_width,
        "scale_y": preview_height / actual_height,
    }
    return result


def _is_per_image_coco(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and name.endswith("_coco.json") and name not in _AGGREGATE_COCO_NAMES


def _find_source_image(coco_path: Path, file_name: str) -> Path | None:
    relative_name = Path(file_name)
    direct_candidates = [coco_path.parent / relative_name, coco_path.parent / relative_name.name]
    for candidate in direct_candidates:
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_EXTENSIONS:
            return candidate.resolve()

    run_dir = _run_dir(coco_path)
    if run_dir is None:
        return None
    matches = [
        path
        for path in run_dir.rglob(relative_name.name)
        if path.is_file()
        and path.suffix.lower() in _IMAGE_EXTENSIONS
        and "annotation_previews" not in {part.lower() for part in path.parts}
    ]
    if not matches:
        return None
    coco_is_real = "uploaded_dataset" in {part.lower() for part in coco_path.parts}
    matches.sort(
        key=lambda path: (
            0 if ("uploaded_dataset" in {part.lower() for part in path.parts}) == coco_is_real else 1,
            len(path.parts),
            path.as_posix(),
        )
    )
    return matches[0].resolve()


def _run_dir(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        if parent.name.startswith("run-"):
            return parent
    return None


def _annotation_source(coco_path: Path, payload: dict[str, Any]) -> str:
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    description = str(info.get("description") or "").lower()
    if "synthetic" in description or "synthetic" in {part.lower() for part in coco_path.parts}:
        return "synthetic"
    return "real"


def _ensure_resized_pair(
    *,
    coco_path: Path,
    source_image: Path,
    payload: dict[str, Any],
    source: str,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> tuple[Path, Path]:
    run_dir = _run_dir(coco_path)
    if run_dir is None:
        raise ValueError(f"COCO file is not inside a training run: {coco_path}")
    source_key = source_image.resolve().relative_to(run_dir.resolve()).as_posix()
    digest = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:12]
    output_dir = run_dir / "annotation_previews" / f"{target_width}x{target_height}" / source / digest
    output_dir.mkdir(parents=True, exist_ok=True)
    resized_image = output_dir / f"{source_image.stem}_{target_width}x{target_height}{source_image.suffix.lower()}"
    resized_coco = output_dir / f"{source_image.stem}_{target_width}x{target_height}_coco.json"
    newest_input = max(source_image.stat().st_mtime, coco_path.stat().st_mtime)
    if (
        resized_image.is_file()
        and resized_coco.is_file()
        and min(resized_image.stat().st_mtime, resized_coco.stat().st_mtime) >= newest_input
    ):
        return resized_image, resized_coco

    with Image.open(source_image) as image:
        resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        if resized_image.suffix in {".jpg", ".jpeg"} and resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")
        resized.save(resized_image)

    transformed = _transform_coco(
        payload,
        resized_image_name=resized_image.name,
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
    )
    resized_coco.write_text(json.dumps(transformed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return resized_image, resized_coco


def _transform_coco(
    payload: dict[str, Any],
    *,
    resized_image_name: str,
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    transformed = copy.deepcopy(payload)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    image = transformed["images"][0]
    image["file_name"] = resized_image_name
    image["width"] = target_width
    image["height"] = target_height

    annotations = transformed.get("annotations")
    if isinstance(annotations, list):
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            bbox = annotation.get("bbox")
            if isinstance(bbox, list) and len(bbox) >= 4:
                annotation["bbox"] = [
                    _rounded(float(bbox[0]) * scale_x),
                    _rounded(float(bbox[1]) * scale_y),
                    _rounded(float(bbox[2]) * scale_x),
                    _rounded(float(bbox[3]) * scale_y),
                ]
            area = annotation.get("area")
            if isinstance(area, (int, float)):
                annotation["area"] = _rounded(float(area) * scale_x * scale_y)
            segmentation = annotation.get("segmentation")
            if isinstance(segmentation, list):
                annotation["segmentation"] = [
                    _scale_polygon(polygon, scale_x, scale_y)
                    if isinstance(polygon, list)
                    else polygon
                    for polygon in segmentation
                ]
            keypoints = annotation.get("keypoints")
            if isinstance(keypoints, list):
                annotation["keypoints"] = _scale_keypoints(keypoints, scale_x, scale_y)

    info = transformed.get("info") if isinstance(transformed.get("info"), dict) else {}
    transformed["info"] = {
        **info,
        "annotation_preview": {
            "mode": PREVIEW_RESIZE_MODE,
            "original_width": original_width,
            "original_height": original_height,
            "target_width": target_width,
            "target_height": target_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
    }
    return transformed


def _scale_polygon(values: list[Any], scale_x: float, scale_y: float) -> list[Any]:
    scaled: list[Any] = []
    for index, value in enumerate(values):
        if isinstance(value, (int, float)):
            scale = scale_x if index % 2 == 0 else scale_y
            scaled.append(_rounded(float(value) * scale))
        else:
            scaled.append(value)
    return scaled


def _scale_keypoints(values: list[Any], scale_x: float, scale_y: float) -> list[Any]:
    scaled = list(values)
    for index in range(0, len(scaled) - 2, 3):
        if isinstance(scaled[index], (int, float)):
            scaled[index] = _rounded(float(scaled[index]) * scale_x)
        if isinstance(scaled[index + 1], (int, float)):
            scaled[index + 1] = _rounded(float(scaled[index + 1]) * scale_y)
    return scaled


def _rounded(value: float) -> float:
    return round(value, 6)
