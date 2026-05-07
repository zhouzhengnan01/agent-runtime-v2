from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.artifacts.preview import artifact_kind, guess_mime_type
from app.schemas import ArtifactRef


VIRTUAL_OUTPUTS_PREFIX = "/mnt/user-data/outputs"
_SAFE_THREAD_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class ThreadPaths:
    thread_id: str
    root: Path
    workspace: Path
    uploads: Path
    outputs: Path


class ArtifactStore:
    """Thread-scoped file container. This is not agent memory."""

    def __init__(self, root_dir: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.root_dir = root_dir or project_root / ".runtime" / "threads"

    def prepare_thread(self, requested_thread_id: str | None = None) -> ThreadPaths:
        thread_id = self._safe_thread_id(requested_thread_id)
        root = self.root_dir / thread_id
        paths = ThreadPaths(
            thread_id=thread_id,
            root=root,
            workspace=root / "user-data" / "workspace",
            uploads=root / "user-data" / "uploads",
            outputs=root / "user-data" / "outputs",
        )
        for path in (paths.workspace, paths.uploads, paths.outputs):
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def write_text_artifact(self, paths: ThreadPaths, filename: str, content: str) -> ArtifactRef:
        target = self._safe_output_path(paths, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return self.to_artifact_ref(paths.thread_id, target)

    def write_bytes_artifact(self, paths: ThreadPaths, filename: str, content: bytes) -> ArtifactRef:
        target = self._safe_output_path(paths, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return self.to_artifact_ref(paths.thread_id, target)

    def list_artifacts(self, thread_id: str) -> list[ArtifactRef]:
        paths = self.prepare_thread(thread_id)
        refs: list[ArtifactRef] = []
        for file_path in sorted(paths.outputs.rglob("*")):
            if file_path.is_file():
                refs.append(self.to_artifact_ref(paths.thread_id, file_path))
        return refs

    def to_artifact_ref(self, thread_id: str, file_path: Path) -> ArtifactRef:
        paths = self.prepare_thread(thread_id)
        resolved = file_path.resolve()
        try:
            relative = resolved.relative_to(paths.outputs.resolve())
        except ValueError as exc:
            raise ValueError(f"Artifact path is outside outputs: {file_path}") from exc
        virtual_path = f"{VIRTUAL_OUTPUTS_PREFIX}/{relative.as_posix()}"
        api_path = virtual_path.lstrip("/")
        mime_type = guess_mime_type(resolved)
        return ArtifactRef(
            thread_id=thread_id,
            path=virtual_path,
            name=resolved.name,
            mime_type=mime_type,
            kind=artifact_kind(resolved, mime_type),
            size=resolved.stat().st_size,
            preview_url=f"/api/artifacts/{thread_id}/{api_path}",
            download_url=f"/api/artifacts/{thread_id}/{api_path}?download=true",
        )

    def resolve_virtual_path(self, thread_id: str, virtual_path: str) -> Path:
        paths = self.prepare_thread(thread_id)
        normalized = "/" + virtual_path.lstrip("/")
        if not normalized.startswith(VIRTUAL_OUTPUTS_PREFIX + "/"):
            raise ValueError(f"Only output artifacts can be served: {virtual_path}")
        rel = normalized[len(VIRTUAL_OUTPUTS_PREFIX) + 1 :]
        candidate = (paths.outputs / rel).resolve()
        outputs = paths.outputs.resolve()
        try:
            candidate.relative_to(outputs)
        except ValueError as exc:
            raise ValueError("Artifact path traversal blocked") from exc
        return candidate

    @staticmethod
    def _safe_thread_id(value: str | None) -> str:
        if not value:
            return f"thread-{uuid.uuid4().hex[:12]}"
        cleaned = _SAFE_THREAD_RE.sub("-", value.strip()).strip(".-")
        return cleaned[:96] or f"thread-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _safe_output_path(paths: ThreadPaths, filename: str) -> Path:
        name = filename.replace("\\", "/").lstrip("/")
        candidate = (paths.outputs / name).resolve()
        outputs = paths.outputs.resolve()
        try:
            candidate.relative_to(outputs)
        except ValueError as exc:
            raise ValueError("Output path traversal blocked") from exc
        return candidate
