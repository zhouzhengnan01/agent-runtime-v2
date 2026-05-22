from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from app.core.artifacts.preview import artifact_kind, guess_mime_type
from app.schemas import ArtifactRef


VIRTUAL_OUTPUTS_PREFIX = "/mnt/user-data/outputs"
_SAFE_THREAD_RE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class ThreadPaths:
    thread_id: str
    root: Path
    memory: Path
    workspace: Path
    uploads: Path
    outputs: Path
    previews: Path
    versions: Path
    manifest: Path


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
            memory=root / "memory",
            workspace=root / "workspace",
            uploads=root / "uploads",
            outputs=root / "outputs",
            previews=root / "previews",
            versions=root / "versions",
            manifest=root / "manifest.json",
        )
        for path in (paths.memory, paths.workspace, paths.uploads, paths.outputs, paths.previews, paths.versions):
            path.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest(paths)
        return paths

    def write_text_artifact(self, paths: ThreadPaths, filename: str, content: str) -> ArtifactRef:
        target = self._safe_output_path(paths, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.upsert_artifact(paths, target)
        return self.to_artifact_ref(paths.thread_id, target)

    def write_bytes_artifact(self, paths: ThreadPaths, filename: str, content: bytes) -> ArtifactRef:
        target = self._safe_output_path(paths, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        self.upsert_artifact(paths, target)
        return self.to_artifact_ref(paths.thread_id, target)

    def list_artifacts(self, thread_id: str) -> list[ArtifactRef]:
        paths = self.prepare_thread(thread_id)
        refs: list[ArtifactRef] = []
        for file_path in sorted(paths.outputs.rglob("*")):
            if file_path.is_file():
                refs.append(self.to_artifact_ref(paths.thread_id, file_path))
        return refs

    def read_manifest(self, thread_id: str) -> dict[str, Any]:
        paths = self.prepare_thread(thread_id)
        return self._read_manifest(paths)

    def write_manifest(self, thread_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
        paths = self.prepare_thread(thread_id)
        payload = {
            **manifest,
            "thread_id": paths.thread_id,
            "updated_at": self._now(),
        }
        self._write_manifest(paths, payload)
        return payload

    def upsert_artifact(
        self,
        paths: ThreadPaths,
        file_path: Path,
        *,
        title: str | None = None,
        preview_path: str | None = None,
        status: str = "ready",
    ) -> dict[str, Any]:
        resolved = file_path.resolve()
        rel = resolved.relative_to(paths.outputs.resolve()).as_posix()
        output_path = f"outputs/{rel}"
        stat = resolved.stat()
        mime_type = guess_mime_type(resolved)
        entry = {
            "path": output_path,
            "name": resolved.name,
            "kind": artifact_kind(resolved, mime_type),
            "mime_type": mime_type,
            "size": stat.st_size,
            "updated_at": self._format_timestamp(stat.st_mtime),
            "status": status,
        }
        if title:
            entry["title"] = title
        if preview_path:
            entry["preview_path"] = self._normalize_manifest_path(preview_path)
        manifest = self._read_manifest(paths)
        artifacts = [item for item in self._manifest_artifacts(manifest) if item.get("path") != output_path]
        artifacts.append(entry)
        artifacts.sort(key=lambda item: str(item.get("path") or ""))
        manifest["artifacts"] = artifacts
        manifest["primary_artifact"] = manifest.get("primary_artifact") or output_path
        manifest["updated_at"] = self._now()
        self._write_manifest(paths, manifest)
        return entry

    def update_manifest(
        self,
        thread_id: str,
        *,
        primary_artifact: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        paths = self.prepare_thread(thread_id)
        manifest = self._read_manifest(paths)
        if primary_artifact is not None:
            manifest["primary_artifact"] = self._normalize_manifest_path(primary_artifact)
        if metadata:
            existing = manifest.get("metadata")
            manifest["metadata"] = {**(existing if isinstance(existing, dict) else {}), **metadata}
        manifest["updated_at"] = self._now()
        self._write_manifest(paths, manifest)
        return manifest

    def output_path(self, paths: ThreadPaths, raw_path: str) -> Path:
        return self._safe_output_path(paths, raw_path)

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
        name = ArtifactStore._normalize_output_name(filename)
        if name.startswith("outputs/"):
            name = name[len("outputs/") :]
        candidate = (paths.outputs / name).resolve()
        outputs = paths.outputs.resolve()
        try:
            candidate.relative_to(outputs)
        except ValueError as exc:
            raise ValueError("Output path traversal blocked") from exc
        return candidate

    @staticmethod
    def _normalize_output_name(filename: str) -> str:
        name = filename.replace("\\", "/").strip()
        if name == VIRTUAL_OUTPUTS_PREFIX or name == VIRTUAL_OUTPUTS_PREFIX.lstrip("/"):
            return ""
        for prefix in (VIRTUAL_OUTPUTS_PREFIX + "/", VIRTUAL_OUTPUTS_PREFIX.lstrip("/") + "/"):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name.lstrip("/")

    def _ensure_manifest(self, paths: ThreadPaths) -> None:
        if paths.manifest.exists():
            return
        self._write_manifest(paths, self._default_manifest(paths))

    def _read_manifest(self, paths: ThreadPaths) -> dict[str, Any]:
        self._ensure_manifest(paths)
        try:
            data = json.loads(paths.manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        manifest = {**self._default_manifest(paths), **data}
        manifest["artifacts"] = self._manifest_artifacts(manifest)
        return manifest

    @staticmethod
    def _write_manifest(paths: ThreadPaths, manifest: dict[str, Any]) -> None:
        paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def _default_manifest(cls, paths: ThreadPaths) -> dict[str, Any]:
        return {
            "thread_id": paths.thread_id,
            "updated_at": cls._now(),
            "primary_artifact": None,
            "artifacts": [],
            "metadata": {},
        }

    @staticmethod
    def _manifest_artifacts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
        raw = manifest.get("artifacts")
        if not isinstance(raw, list):
            return []
        return [dict(item) for item in raw if isinstance(item, dict) and item.get("path")]

    @staticmethod
    def _normalize_manifest_path(raw_path: str) -> str:
        normalized = raw_path.replace("\\", "/").lstrip("/").strip()
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("Invalid artifact manifest path")
        if normalized.startswith("~"):
            raise ValueError("Home-relative artifact paths are not allowed")
        return normalized

    @staticmethod
    def _format_timestamp(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
