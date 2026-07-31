from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore, ThreadPaths

COCO_EXPORT_RELATIVE_PATH = Path("artifacts_json") / "annotations" / "instances_all.json"
_COCO_REQUIRED_KEYS = ("images", "annotations", "categories")
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_MODEL_EXTENSIONS = {".bin", ".ckpt", ".engine", ".onnx", ".pt", ".pth", ".safetensors"}


def build_thread_artifacts_json(
    store: ArtifactStore,
    thread_id: str,
    *,
    base_url: str = "",
) -> dict[str, Any]:
    paths = store.prepare_thread(thread_id)
    run_dirs = _list_run_dirs(paths)
    if not run_dirs:
        raise FileNotFoundError(f"No training runs found for thread: {paths.thread_id}")
    rounds = [
        _build_round_artifacts_json(
            store,
            paths.thread_id,
            run_dir,
            round_number=round_number,
            base_url=base_url,
        )
        for round_number, run_dir in enumerate(run_dirs, start=1)
    ]
    return {
        "threadId": paths.thread_id,
        "totalRounds": len(rounds),
        "rounds": rounds,
    }


def _build_round_artifacts_json(
    store: ArtifactStore,
    thread_id: str,
    run_dir: Path,
    *,
    round_number: int,
    base_url: str = "",
) -> dict[str, Any]:
    source_paths: list[Path] = []
    merged_artifact: dict[str, Any] | None = None
    coco_error: str | None = None
    export_path = run_dir / COCO_EXPORT_RELATIVE_PATH
    try:
        source_paths, source_payloads = _load_training_coco_sources(run_dir)
        merged = _merge_coco_payloads(source_payloads)
        _write_json_atomic(export_path, merged)
        merged_artifact = _artifact_payload(store, thread_id, run_dir.name, export_path)
        merged_artifact.update(
            {
                "artifact_role": "merged_coco",
                "image_count": len(merged["images"]),
                "annotation_count": len(merged["annotations"]),
                "category_count": len(merged["categories"]),
                "real_image_count": sum(1 for item in merged["images"] if not _image_is_synthetic(item)),
                "synthetic_image_count": sum(1 for item in merged["images"] if _image_is_synthetic(item)),
            }
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        coco_error = str(exc)

    artifacts: list[dict[str, Any]] = []
    for file_path in sorted(run_dir.rglob("*")):
        if not file_path.is_file() or file_path == export_path:
            continue
        if _is_yolo_label_artifact(run_dir, file_path):
            continue
        if file_path.suffix.lower() == ".json" and _load_coco_or_none(file_path) is not None:
            continue
        artifacts.append(
            _round_artifact_payload(
                store,
                thread_id,
                run_dir,
                file_path,
                round_number=round_number,
                base_url=base_url,
            )
        )

    payload: dict[str, Any] = {
        "round": round_number,
        "runId": run_dir.name,
        "artifacts": artifacts,
        "merged_coco": merged_artifact,
        "source_coco_files": [str(path.relative_to(run_dir).as_posix()) for path in source_paths],
    }
    if coco_error:
        payload["coco_error"] = coco_error
    return payload


def _list_run_dirs(paths: ThreadPaths) -> list[Path]:
    runs_root = paths.outputs / "yolo_training_flow" / "runs"
    if not runs_root.is_dir():
        return []
    return sorted(
        (path.resolve() for path in runs_root.iterdir() if path.is_dir() and path.name.startswith("run-")),
        key=lambda path: path.name,
    )


def _load_training_coco_sources(run_dir: Path) -> tuple[list[Path], list[tuple[str, dict[str, Any]]]]:
    pipeline_work = run_dir / "pipeline_work"
    merged_path = pipeline_work / "merged_coco.json"
    if merged_path.is_file():
        return [merged_path], [("merged", _load_coco(merged_path))]

    real_path = pipeline_work / "real_coco.json"
    synthetic_path = pipeline_work / "synthetic_coco.json"
    sources: list[Path] = []
    payloads: list[tuple[str, dict[str, Any]]] = []
    if real_path.is_file():
        sources.append(real_path)
        payloads.append(("real", _load_coco(real_path)))
    if synthetic_path.is_file():
        sources.append(synthetic_path)
        payloads.append(("synthetic", _load_coco(synthetic_path)))
    if payloads:
        return sources, payloads

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in run_dir.rglob("*.json"):
        if COCO_EXPORT_RELATIVE_PATH.as_posix() in path.as_posix():
            continue
        payload = _load_coco_or_none(path)
        if payload is not None:
            candidates.append((len(payload["images"]), path, payload))
    if not candidates:
        raise FileNotFoundError(f"No COCO annotations found for run: {run_dir.name}")
    candidates.sort(key=lambda item: (-item[0], item[1].as_posix()))
    _, selected_path, selected_payload = candidates[0]
    return [selected_path], [("uploaded", selected_payload)]


def _load_coco(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid COCO JSON: {path}") from exc
    if not isinstance(payload, dict) or not all(isinstance(payload.get(key), list) for key in _COCO_REQUIRED_KEYS):
        raise ValueError(f"COCO JSON must contain array fields images, annotations and categories: {path}")
    return payload


def _load_coco_or_none(path: Path) -> dict[str, Any] | None:
    try:
        return _load_coco(path)
    except ValueError:
        return None


def _merge_coco_payloads(sources: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    if not sources:
        raise ValueError("At least one COCO source is required")

    categories: list[dict[str, Any]] = []
    category_key_to_id: dict[tuple[str, str], int] = {}
    category_id_maps: dict[str, dict[int, int]] = {}
    for source_index, (source_name, payload) in enumerate(sources):
        source_key = f"{source_index}:{source_name}"
        category_map: dict[int, int] = {}
        for category in payload["categories"]:
            if not isinstance(category, dict):
                raise TypeError(f"Invalid category in COCO source: {source_name}")
            old_id = _required_int(category.get("id"), f"category.id in {source_name}")
            name = str(category.get("name") or "").strip()
            if not name:
                raise ValueError(f"Empty category name in COCO source: {source_name}")
            key = (name, str(category.get("supercategory") or "object"))
            new_id = category_key_to_id.get(key)
            if new_id is None:
                new_id = len(categories) + 1
                category_key_to_id[key] = new_id
                copied = deepcopy(category)
                copied["id"] = new_id
                copied["name"] = name
                copied.setdefault("supercategory", key[1])
                categories.append(copied)
            category_map[old_id] = new_id
        category_id_maps[source_key] = category_map

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for source_index, (source_name, payload) in enumerate(sources):
        source_key = f"{source_index}:{source_name}"
        image_map: dict[int, int] = {}
        for image in payload["images"]:
            if not isinstance(image, dict):
                raise TypeError(f"Invalid image in COCO source: {source_name}")
            old_id = _required_int(image.get("id"), f"image.id in {source_name}")
            if old_id in image_map:
                raise ValueError(f"Duplicate image id {old_id} in COCO source: {source_name}")
            file_name = str(image.get("file_name") or "").replace("\\", "/").strip()
            if not file_name:
                raise ValueError(f"Empty image file_name in COCO source: {source_name}")
            new_id = len(images) + 1
            image_map[old_id] = new_id
            copied = deepcopy(image)
            copied["id"] = new_id
            copied["file_name"] = file_name
            copied.pop("jetlinks_artifact", None)
            if source_name in {"real", "synthetic"}:
                copied.setdefault("source", source_name)
                copied.setdefault("is_synthetic", source_name == "synthetic")
            images.append(copied)
        seen_annotation_ids: set[int] = set()
        for annotation in payload["annotations"]:
            if not isinstance(annotation, dict):
                raise TypeError(f"Invalid annotation in COCO source: {source_name}")
            old_annotation_id = _required_int(annotation.get("id"), f"annotation.id in {source_name}")
            if old_annotation_id in seen_annotation_ids:
                raise ValueError(f"Duplicate annotation id {old_annotation_id} in COCO source: {source_name}")
            seen_annotation_ids.add(old_annotation_id)
            old_image_id = _required_int(annotation.get("image_id"), f"annotation.image_id in {source_name}")
            old_category_id = _required_int(annotation.get("category_id"), f"annotation.category_id in {source_name}")
            if old_image_id not in image_map:
                raise ValueError(f"Annotation references missing image id {old_image_id} in {source_name}")
            if old_category_id not in category_id_maps[source_key]:
                raise ValueError(f"Annotation references missing category id {old_category_id} in {source_name}")
            bbox = _validate_bbox(annotation.get("bbox"), source_name)
            copied = deepcopy(annotation)
            copied["id"] = len(annotations) + 1
            copied["image_id"] = image_map[old_image_id]
            copied["category_id"] = category_id_maps[source_key][old_category_id]
            copied["bbox"] = list(bbox)
            copied["area"] = bbox[2] * bbox[3]
            copied.setdefault("iscrowd", 0)
            annotations.append(copied)

    first_payload = sources[0][1]
    return {
        "info": deepcopy(first_payload.get("info") or {"description": "JetLinks merged COCO dataset"}),
        "licenses": deepcopy(first_payload.get("licenses") or []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _required_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _validate_bbox(value: Any, source_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Annotation bbox must contain four values in {source_name}")
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Annotation bbox contains a non-numeric value in {source_name}") from exc
    if not all(math.isfinite(item) for item in (x, y, width, height)) or width < 0 or height < 0:
        raise ValueError(f"Annotation bbox is invalid in {source_name}")
    return x, y, width, height


def _image_is_synthetic(image: dict[str, Any]) -> bool:
    if bool(image.get("is_synthetic") or image.get("synthetic")):
        return True
    return str(image.get("source") or "").strip().lower() in {"synthetic", "generated", "gen"}


def _is_yolo_label_artifact(run_dir: Path, path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    parts = {part.casefold() for part in path.relative_to(run_dir).parts}
    return "prepared_dataset" in parts and "labels" in parts


def _round_artifact_payload(
    store: ArtifactStore,
    thread_id: str,
    run_dir: Path,
    path: Path,
    *,
    round_number: int,
    base_url: str,
) -> dict[str, Any]:
    artifact = _artifact_payload(store, thread_id, run_dir.name, path)
    return {
        "name": artifact["name"],
        "path": artifact["path"],
        "stage": _artifact_stage(run_dir, path),
        "artifact_type": _artifact_type(path),
        "mimeType": artifact["mime_type"],
        "size": artifact["size"],
        "downloadUrl": _absolute_url(base_url, artifact["download_url"]),
        "round": round_number,
        "runId": run_dir.name,
        "threadId": thread_id,
    }


def _artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _MODEL_EXTENSIONS:
        return "model"
    return "summary"


def _artifact_stage(run_dir: Path, path: Path) -> str:
    relative = path.relative_to(run_dir)
    parts = {part.casefold() for part in relative.parts}
    name = path.name.casefold()
    suffix = path.suffix.lower()
    if any(token in name for token in ("run_summary", "summary", "evaluation", "report", "metrics")):
        return "summary"
    if suffix in _MODEL_EXTENSIONS or parts.intersection(
        {"checkpoints", "deimv2_training_run", "models", "train", "weights"}
    ):
        return "training"
    if parts.intersection(
        {"composite_inputs", "generated_images", "generation_inputs", "synthetic_images"}
    ):
        return "generation"
    if parts.intersection(
        {"annotations", "pipeline_work", "prepared_data", "uploaded_dataset"}
    ):
        return "annotation"
    return "summary"


def _absolute_url(base_url: str, path: str) -> str:
    if not base_url:
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _artifact_payload(store: ArtifactStore, thread_id: str, run_id: str, path: Path) -> dict[str, Any]:
    payload = store.to_artifact_ref(thread_id, path).model_dump()
    payload["preview_url"] = payload["preview_url"].replace("/api/artifacts/", "/api/artifacts_json/", 1)
    payload["download_url"] = payload["download_url"].replace("/api/artifacts/", "/api/artifacts_json/", 1)
    payload["run_id"] = run_id
    return payload
