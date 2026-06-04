from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from app.core.artifacts import ArtifactStore, ThreadPaths


DEFAULT_DELETE_SCOPES: Final[tuple[str, ...]] = ("workspace", "uploads", "outputs")
VALID_DELETE_SCOPES: Final[frozenset[str]] = frozenset(DEFAULT_DELETE_SCOPES)
VIRTUAL_PREFIXES: Final[dict[str, str]] = {
    "workspace": "/mnt/user-data/workspace",
    "uploads": "/mnt/user-data/uploads",
    "outputs": "/mnt/user-data/outputs",
}

DeletedEntryKind = Literal["file", "directory", "symlink"]


@dataclass(frozen=True)
class DeletedSessionFile:
    scope: str
    path: str
    kind: DeletedEntryKind
    size: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "path": self.path,
            "kind": self.kind,
            "size": self.size,
        }


def delete_thread_files(
    artifact_store: ArtifactStore,
    thread_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Delete ACP-visible files for one runtime thread.

    The method intentionally deletes only the contents of the virtual
    /mnt/user-data scope directories and leaves the thread container itself in
    place so future prompts can reuse the session safely.
    """

    paths = artifact_store.prepare_thread(thread_id)
    scopes = _scopes_from_params(params)
    dry_run = _bool_param(params, "dryRun", "dry_run", default=False)
    store_root = artifact_store.root_dir.resolve()

    entries: list[DeletedSessionFile] = []
    for scope in scopes:
        scope_root = _safe_scope_root(paths, scope, store_root)
        entries.extend(_collect_scope_entries(scope, scope_root))
    entries.sort(key=lambda item: (item.scope, item.path, item.kind))

    if not dry_run:
        for scope in scopes:
            scope_root = _safe_scope_root(paths, scope, store_root)
            _delete_scope_contents(scope_root)

    file_entries = [entry for entry in entries if entry.kind != "directory"]
    return {
        "threadId": paths.thread_id,
        "scopes": scopes,
        "dryRun": dry_run,
        "deleted": [entry.to_payload() for entry in entries],
        "deletedCount": len(entries),
        "deletedFileCount": len(file_entries),
        "deletedBytes": sum(entry.size for entry in file_entries),
    }


def _scopes_from_params(params: dict[str, Any]) -> list[str]:
    raw_scopes = params.get("scopes", params.get("scope"))
    if raw_scopes is None:
        return list(DEFAULT_DELETE_SCOPES)

    if isinstance(raw_scopes, str):
        candidate_scopes = [raw_scopes]
    elif isinstance(raw_scopes, list):
        candidate_scopes = raw_scopes
    else:
        raise ValueError("scopes must be a list of strings")

    scopes: list[str] = []
    for raw_scope in candidate_scopes:
        if not isinstance(raw_scope, str) or not raw_scope.strip():
            raise ValueError("scopes must contain only non-empty strings")
        scope = raw_scope.strip()
        if scope not in VALID_DELETE_SCOPES:
            raise ValueError(f"Unsupported ACP file deletion scope: {scope}")
        if scope not in scopes:
            scopes.append(scope)
    return scopes


def _bool_param(params: dict[str, Any], camel_name: str, snake_name: str, *, default: bool) -> bool:
    value = params.get(camel_name)
    if value is None:
        value = params.get(snake_name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{camel_name} must be a boolean")


def _safe_scope_root(paths: ThreadPaths, scope: str, store_root: Path) -> Path:
    root = _scope_path(paths, scope)
    if root.is_symlink():
        raise ValueError(f"Refusing to delete symlinked ACP file scope: {scope}")
    resolved = root.resolve()
    try:
        resolved.relative_to(store_root)
    except ValueError as exc:
        raise ValueError(f"ACP file deletion scope escaped artifact store: {scope}") from exc
    if not resolved.is_dir():
        raise ValueError(f"ACP file deletion scope is not a directory: {scope}")
    return resolved


def _scope_path(paths: ThreadPaths, scope: str) -> Path:
    if scope == "workspace":
        return paths.workspace
    if scope == "uploads":
        return paths.uploads
    if scope == "outputs":
        return paths.outputs
    raise ValueError(f"Unsupported ACP file deletion scope: {scope}")


def _collect_scope_entries(scope: str, root: Path) -> list[DeletedSessionFile]:
    entries: list[DeletedSessionFile] = []
    for dirpath_name, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        dirpath = Path(dirpath_name)
        for filename in sorted(filenames):
            path = dirpath / filename
            entries.append(_entry_from_path(scope, root, path))
        for dirname in sorted(dirnames):
            path = dirpath / dirname
            entries.append(_entry_from_path(scope, root, path))
    return entries


def _entry_from_path(scope: str, root: Path, path: Path) -> DeletedSessionFile:
    kind = _entry_kind(path)
    return DeletedSessionFile(
        scope=scope,
        path=_virtual_path(scope, root, path),
        kind=kind,
        size=_entry_size(path, kind),
    )


def _entry_kind(path: Path) -> DeletedEntryKind:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    return "file"


def _entry_size(path: Path, kind: DeletedEntryKind) -> int:
    if kind == "directory":
        return 0
    try:
        return path.lstat().st_size
    except FileNotFoundError:
        return 0


def _virtual_path(scope: str, root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("ACP file deletion path traversal blocked") from exc
    return f"{VIRTUAL_PREFIXES[scope]}/{relative}"


def _delete_scope_contents(root: Path) -> None:
    for dirpath_name, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        dirpath = Path(dirpath_name)
        for filename in filenames:
            _unlink_path(dirpath / filename)
        for dirname in dirnames:
            child = dirpath / dirname
            if child.is_symlink() or not child.is_dir():
                _unlink_path(child)
            else:
                _rmdir_path(child)
    root.mkdir(parents=True, exist_ok=True)


def _unlink_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _rmdir_path(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        return
