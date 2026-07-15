from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.artifacts import ArtifactStore


TRAINING_ARTIFACT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_training_artifact_session_updates(
    *,
    thread_id: str,
    status: dict[str, Any],
    artifact_store: ArtifactStore,
    published_artifacts: set[str],
) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for path in training_status_artifact_paths(status):
        resolved = path.resolve()
        key = str(resolved).casefold()
        if key in published_artifacts:
            continue
        try:
            paths = artifact_store.prepare_thread(thread_id)
            resolved.relative_to(paths.outputs.resolve())
            artifact_store.upsert_artifact(paths, resolved)
            artifact = artifact_store.to_artifact_ref(thread_id, resolved).model_dump()
        except (OSError, ValueError):
            continue
        published_artifacts.add(key)
        updates.append(_artifact_session_update("artifact.created", artifact))
        updates.append(_artifact_session_update("preview.ready", artifact))
    return updates


def training_status_artifact_paths(status: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    dataset = status.get("dataset") if isinstance(status.get("dataset"), dict) else {}
    annotation = status.get("annotation") if isinstance(status.get("annotation"), dict) else {}
    generation = status.get("generation") if isinstance(status.get("generation"), dict) else {}
    training = status.get("training") if isinstance(status.get("training"), dict) else {}
    paths = status.get("paths") if isinstance(status.get("paths"), dict) else {}

    _append_path(candidates, dataset.get("training_coco"))
    real = annotation.get("real") if isinstance(annotation.get("real"), dict) else {}
    synthetic = annotation.get("synthetic") if isinstance(annotation.get("synthetic"), dict) else {}
    _append_path(candidates, real.get("output_coco"))
    _append_path(candidates, synthetic.get("output_coco"))
    _append_path(candidates, training.get("checkpoint"))

    generated_dir = _path_or_none(generation.get("output_dir"))
    if generated_dir is not None and generated_dir.is_dir():
        for path in sorted(generated_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in TRAINING_ARTIFACT_IMAGE_EXTS:
                candidates.append(path)

    run_dir = _path_or_none(paths.get("run_dir"))
    if run_dir is not None and run_dir.is_dir():
        for pattern in ("best.pt", "last.pt", "best_stg2.pth", "best_stg1.pth", "last.pth"):
            candidates.extend(path for path in sorted(run_dir.rglob(pattern)) if path.is_file())

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        key = str(path.resolve()).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _artifact_session_update(event_type: str, artifact: dict[str, Any]) -> dict[str, Any]:
    artifact_name = str(artifact.get("name") or artifact.get("title") or artifact.get("path") or "artifact")
    text = f"artifact: {artifact_name}" if event_type == "artifact.created" else f"preview ready: {artifact_name}"
    return {
        "sessionUpdate": "agent_thought_chunk",
        "content": {"type": "text", "text": text},
        "artifact": artifact,
        "rawOutput": artifact,
        "_meta": {
            "jetlinksRuntimeEvent": {
                "type": event_type,
                "data": {"artifact": artifact},
            },
            "jetlinksDiff": _artifact_diff(artifact) if event_type == "artifact.created" else None,
        },
    }


def _artifact_diff(artifact: dict[str, Any]) -> dict[str, Any] | None:
    path = artifact.get("path")
    if not path:
        return None
    return {
        "type": "artifact_change",
        "status": "completed",
        "path": str(path),
        "source": "artifact",
        "operation": "created",
    }


def _append_path(paths: list[Path], value: Any) -> None:
    path = _path_or_none(value)
    if path is not None:
        paths.append(path)


def _path_or_none(value: Any) -> Path | None:
    if not value:
        return None
    return Path(str(value)).expanduser()
