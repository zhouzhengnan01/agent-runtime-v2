from __future__ import annotations

import difflib
import base64
import json
import mimetypes
import os
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


LOCAL_TOOL_NAMES = {
    "local_read_file",
    "local_patch_file",
    "local_write_file",
    "local_download_url",
    "local_file_to_base64",
    "local_search_text",
    "local_todo",
    "local_shell_command",
    "present_files",
    "extract_archive",
    "validate_yolo_training_inputs",
}


class LocalToolProvider:
    """Thread-workspace local tools inspired by Hermes' file/terminal/todo tools."""

    source_type = "local"

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        thread_id = str(arguments.get("_thread_id") or "mcp-local")
        paths = self.artifact_store.prepare_thread(thread_id)
        operation = str(tool.source.get("operation") or "")
        try:
            if operation == "read_file":
                return self._read_file(paths, arguments)
            if operation == "write_file":
                return self._write_file(paths, arguments)
            if operation == "download_url":
                return self._download_url(paths, arguments)
            if operation == "file_to_base64":
                return self._file_to_base64(paths, arguments)
            if operation == "patch_file":
                return self._patch_file(paths, arguments)
            if operation == "search_text":
                return self._search_text(paths, arguments)
            if operation == "todo":
                return self._todo(paths, arguments)
            if operation == "shell_command":
                return self._shell_command(paths, arguments)
            if operation == "present_files":
                return self._present_files(paths, arguments)
            if operation == "extract_archive":
                return self._extract_archive(paths, arguments)
            if operation == "validate_yolo_training_inputs":
                return self._validate_yolo_training_inputs(paths, arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Unsupported local operation: {operation}"}],
            structured_content={"tool_name": tool.name, "operation": operation},
            is_error=True,
        )

    def _read_file(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._readable_path(paths, str(arguments.get("path") or ""), arguments)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self._display_path(paths, target, arguments)}")
        max_chars = self._bounded_int(arguments.get("max_chars"), default=12000, minimum=100, maximum=50000)
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > max_chars
        content = text[:max_chars]
        return ToolInvocationResult(
            content=[{"type": "text", "text": content}],
            structured_content={
                "path": self._display_path(paths, target, arguments),
                "chars": len(content),
                "truncated": truncated,
            },
            is_error=False,
        )

    def _write_file(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._workspace_path(paths, str(arguments.get("path") or ""), arguments)
        content = str(arguments.get("content") or "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        display_path = self._display_path(paths, target, arguments)
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Wrote {len(content)} characters to {display_path}"}],
            structured_content={
                "path": display_path,
                "chars": len(content),
            },
            is_error=False,
        )

    def _download_url(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        url = str(arguments.get("url") or "").strip()
        if not url:
            raise ValueError("url is required")
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only http:// and https:// URLs are allowed")
        if not parsed.netloc:
            raise ValueError("URL host is required")

        max_bytes = self._bounded_int(
            arguments.get("max_bytes"),
            default=50 * 1024 * 1024,
            minimum=1,
            maximum=500 * 1024 * 1024,
        )
        timeout_seconds = self._bounded_int(
            arguments.get("timeout_seconds"),
            default=30,
            minimum=1,
            maximum=120,
        )
        allowed_prefixes = self._mime_prefixes(
            arguments.get("allowed_mime_prefixes"),
            default=["image/", "video/"],
        )
        overwrite = bool(arguments.get("overwrite", True))

        with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_length = self._content_length(response.headers.get("content-length"))
                if content_length is not None and content_length > max_bytes:
                    raise ValueError(f"Remote file is too large: {content_length} bytes > {max_bytes} bytes")

                header_mime = self._header_mime_type(response.headers.get("content-type"))
                filename = self._download_filename(url, str(arguments.get("filename") or ""), header_mime)
                target = self._upload_target(paths, filename, overwrite=overwrite)
                guessed_mime = self._guess_mime_type(target)
                mime_type = self._download_mime_type(header_mime, guessed_mime)
                if not self._mime_allowed(mime_type, allowed_prefixes):
                    raise ValueError(
                        f"Remote MIME type is not allowed: {mime_type}. "
                        f"Allowed prefixes: {', '.join(allowed_prefixes)}"
                    )

                target.parent.mkdir(parents=True, exist_ok=True)
                bytes_written = 0
                try:
                    with target.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            if not chunk:
                                continue
                            bytes_written += len(chunk)
                            if bytes_written > max_bytes:
                                raise ValueError(
                                    f"Remote file is too large: {bytes_written} bytes > {max_bytes} bytes"
                                )
                            handle.write(chunk)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise

        display_path = self._display_path(paths, target, arguments)
        kind = self._file_kind(target)
        text = f"Downloaded {url} to {display_path} ({bytes_written} bytes, mime_type={mime_type})."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "url": url,
                "path": display_path,
                "_local_path": str(target),
                "filename": target.name,
                "scope": "uploads",
                "bytes": bytes_written,
                "mime_type": mime_type,
                "kind": kind,
            },
            is_error=False,
        )

    def _file_to_base64(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._thread_scoped_path(
            paths,
            str(arguments.get("path") or ""),
            arguments,
            scopes={"uploads", "workspace", "outputs"},
        )
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self._display_path(paths, target, arguments)}")
        max_bytes = self._bounded_int(
            arguments.get("max_bytes"),
            default=2 * 1024 * 1024,
            minimum=1,
            maximum=10 * 1024 * 1024,
        )
        file_size = target.stat().st_size
        if file_size > max_bytes:
            raise ValueError(f"File is too large for base64 encoding: {file_size} bytes > {max_bytes} bytes")
        raw = target.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        mime_type = str(arguments.get("mime_type") or "").strip() or self._guess_mime_type(target)
        include_data_uri = bool(arguments.get("include_data_uri", True))
        data_uri = f"data:{mime_type};base64,{encoded}" if include_data_uri else None
        display_path = self._display_path(paths, target, arguments)
        text = f"Encoded {display_path} to Base64 ({file_size} bytes, mime_type={mime_type})."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "path": display_path,
                "bytes": file_size,
                "mime_type": mime_type,
                "encoding": "base64",
                "base64": encoded,
                "data_uri": data_uri,
            },
            is_error=False,
        )

    def _patch_file(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._workspace_path(paths, str(arguments.get("path") or ""), arguments)
        old_text = str(arguments.get("old_text") or "")
        new_text = str(arguments.get("new_text") or "")
        if not old_text:
            raise ValueError("old_text is required")
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self._display_path(paths, target, arguments)}")
        original = target.read_text(encoding="utf-8", errors="replace")
        occurrences = original.count(old_text)
        display_path = self._display_path(paths, target, arguments)
        if occurrences == 0:
            raise ValueError(f"old_text was not found in {display_path}")
        if occurrences > 1:
            raise ValueError(f"old_text matched {occurrences} times in {display_path}; expected exactly one")
        updated = original.replace(old_text, new_text, 1)
        target.write_text(updated, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{display_path}",
                tofile=f"b/{display_path}",
                n=3,
            )
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Patched {display_path}\n{diff[:12000]}"}],
            structured_content={
                "path": display_path,
                "replacements": 1,
                "diff": diff[:50000],
                "diff_truncated": len(diff) > 50000,
            },
            is_error=False,
        )

    def _search_text(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        pattern = str(arguments.get("pattern") or "")
        if not pattern:
            raise ValueError("pattern is required")
        root = self._workspace_path(paths, str(arguments.get("path") or "."), arguments)
        if not root.exists():
            raise FileNotFoundError(f"Path not found: {self._display_path(paths, root, arguments)}")
        max_results = self._bounded_int(arguments.get("max_results"), default=50, minimum=1, maximum=200)
        case_sensitive = bool(arguments.get("case_sensitive", False))
        needle = pattern if case_sensitive else pattern.lower()
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        results: list[dict[str, Any]] = []
        for file_path in files:
            if len(results) >= max_results:
                break
            if self._skip_file(file_path):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.lower()
                if needle in haystack:
                    results.append(
                        {
                            "path": self._display_path(paths, file_path, arguments),
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(results) >= max_results:
                        break
        text = "\n".join(f"{item['path']}:{item['line']}: {item['text']}" for item in results) or "No matches."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={"pattern": pattern, "matches": results, "truncated": len(results) >= max_results},
            is_error=False,
        )

    def _todo(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        action = str(arguments.get("action") or "list").strip().lower()
        todo_path = paths.workspace / ".todos.json"
        todos = self._read_todos(todo_path)
        if action == "add":
            text = str(arguments.get("text") or "").strip()
            if not text:
                raise ValueError("text is required for add")
            next_id = max((int(item["id"]) for item in todos), default=0) + 1
            todos.append({"id": next_id, "text": text, "done": False})
            self._write_todos(todo_path, todos)
        elif action == "complete":
            todo_id = self._bounded_int(arguments.get("id"), default=0, minimum=1, maximum=1_000_000)
            matched = False
            for item in todos:
                if int(item["id"]) == todo_id:
                    item["done"] = True
                    matched = True
                    break
            if not matched:
                raise ValueError(f"todo id not found: {todo_id}")
            self._write_todos(todo_path, todos)
        elif action == "clear":
            todos = []
            self._write_todos(todo_path, todos)
        elif action != "list":
            raise ValueError("action must be one of: list, add, complete, clear")
        lines = [f"{item['id']}. [{'x' if item['done'] else ' '}] {item['text']}" for item in todos]
        return ToolInvocationResult(
            content=[{"type": "text", "text": "\n".join(lines) or "No todos."}],
            structured_content={"todos": todos},
            is_error=False,
        )

    def _shell_command(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        if os.getenv("LOCAL_SHELL_TOOL_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
            raise PermissionError("local_shell_command is disabled. Set LOCAL_SHELL_TOOL_ENABLED=true to enable it.")
        command = str(arguments.get("command") or "").strip()
        if not command:
            raise ValueError("command is required")
        cwd = self._workspace_path(paths, str(arguments.get("cwd") or "."), arguments)
        if not cwd.is_dir():
            raise NotADirectoryError(f"cwd is not a directory: {self._display_path(paths, cwd, arguments)}")
        timeout = self._bounded_int(arguments.get("timeout_seconds"), default=20, minimum=1, maximum=60)
        max_chars = self._bounded_int(arguments.get("max_chars"), default=12000, minimum=100, maximum=50000)
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout[:max_chars]
        stderr = completed.stderr[:max_chars]
        text = "\n".join(
            part
            for part in [
                f"exit_code: {completed.returncode}",
                f"stdout:\n{stdout}" if stdout else "",
                f"stderr:\n{stderr}" if stderr else "",
            ]
            if part
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "command": command,
                "cwd": self._display_path(paths, cwd, arguments),
                "exit_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            is_error=completed.returncode != 0,
        )

    def _present_files(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        include_workspace = bool(arguments.get("include_workspace", False))
        max_results = self._bounded_int(arguments.get("max_results"), default=100, minimum=1, maximum=500)
        outputs = self._file_entries(paths, paths.outputs, "outputs", max_results)
        remaining = max(0, max_results - len(outputs))
        uploads = self._file_entries(paths, paths.uploads, "uploads", remaining) if remaining else []
        remaining = max(0, max_results - len(outputs) - len(uploads))
        workspace = self._file_entries(paths, paths.workspace, "workspace", remaining) if include_workspace and remaining else []
        files = outputs + uploads + workspace
        text = "\n".join(f"- {item['scope']}/{item['path']} ({item['size']} bytes)" for item in files) or "No files."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "thread_id": paths.thread_id,
                "files": files,
                "truncated": len(files) >= max_results,
            },
            is_error=False,
        )

    def _extract_archive(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        archive = self._thread_scoped_path(
            paths,
            str(arguments.get("path") or arguments.get("archive_path") or ""),
            arguments,
            scopes={"uploads", "workspace", "outputs"},
        )
        if not archive.is_file():
            raise FileNotFoundError(f"Archive not found: {self._display_path(paths, archive, arguments)}")
        if not self._is_supported_archive(archive):
            raise ValueError("Archive must be .zip, .tar, .tar.gz, or .tgz")
        default_output = f"extracted/{archive.stem.removesuffix('.tar')}"
        output_dir = self._thread_scoped_path(
            paths,
            str(arguments.get("output_dir") or arguments.get("target_dir") or default_output),
            arguments,
            scopes={"workspace", "outputs"},
        )
        overwrite = bool(arguments.get("overwrite", True))
        max_files = self._bounded_int(arguments.get("max_files"), default=10000, minimum=1, maximum=100000)
        max_bytes = self._bounded_int(
            arguments.get("max_bytes"),
            default=512 * 1024 * 1024,
            minimum=1024,
            maximum=2 * 1024 * 1024 * 1024,
        )
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory is not empty: {self._display_path(paths, output_dir, arguments)}")
        output_dir.mkdir(parents=True, exist_ok=True)

        extracted = self._extract_archive_to(archive, output_dir, max_files=max_files, max_bytes=max_bytes)
        image_files = [path for path in extracted if self._is_image_file(path)]
        dataset_root = self._best_dataset_root(output_dir, image_files)
        text = (
            f"Extracted {len(extracted)} files from {self._display_path(paths, archive, arguments)} "
            f"to {self._display_path(paths, output_dir, arguments)}. "
            f"Detected {len(image_files)} image files."
        )
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "status": "completed",
                "archive_path": self._display_path(paths, archive, arguments),
                "output_dir": self._display_path(paths, output_dir, arguments),
                "dataset_root": self._display_path(paths, dataset_root, arguments),
                "file_count": len(extracted),
                "image_count": len(image_files),
                "top_level_entries": self._top_level_entries(output_dir),
                "sample_images": [self._display_path(paths, path, arguments) for path in image_files[:20]],
            },
            is_error=False,
        )

    def _validate_yolo_training_inputs(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        dataset_value = str(arguments.get("dataset_root") or arguments.get("data") or "").strip()
        ref_value = str(arguments.get("ref_image") or arguments.get("reference_image") or "").strip()
        labels = self._string_list(arguments.get("labels") or arguments.get("class_names"))
        training = dict(arguments.get("training") or {}) if isinstance(arguments.get("training"), dict) else {}
        runtime = dict(arguments.get("runtime") or {}) if isinstance(arguments.get("runtime"), dict) else {}
        split = dict(arguments.get("split") or {}) if isinstance(arguments.get("split"), dict) else {}

        missing: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []
        normalized: dict[str, Any] = {
            "labels": labels,
            "training": training,
            "runtime": runtime,
            "split": split,
        }

        dataset_path: Path | None = None
        if not dataset_value:
            missing.append("dataset_root")
        else:
            dataset_path = self._thread_scoped_path(
                paths,
                dataset_value,
                arguments,
                scopes={"uploads", "workspace", "outputs"},
            )
            normalized["dataset_root"] = self._display_path(paths, dataset_path, arguments)
            if dataset_path.is_file() and self._is_supported_archive(dataset_path):
                errors.append("dataset_root points to an archive; call extract_archive first and pass the extracted directory.")
            elif not dataset_path.exists():
                errors.append(f"dataset_root does not exist: {normalized['dataset_root']}")
            elif not dataset_path.is_dir():
                errors.append(f"dataset_root must be a directory: {normalized['dataset_root']}")
            else:
                image_count = sum(1 for path in dataset_path.rglob("*") if path.is_file() and self._is_image_file(path))
                normalized["image_count"] = image_count
                if image_count <= 0:
                    errors.append(f"dataset_root contains no image files: {normalized['dataset_root']}")
                elif image_count < 10:
                    warnings.append("dataset_root has fewer than 10 images; evaluation metrics may be unstable.")

        if ref_value:
            ref_path = self._thread_scoped_path(
                paths,
                ref_value,
                arguments,
                scopes={"uploads", "workspace", "outputs"},
            )
            normalized["ref_image"] = self._display_path(paths, ref_path, arguments)
            if not ref_path.is_file():
                errors.append(f"ref_image does not exist: {normalized['ref_image']}")
            elif not self._is_image_file(ref_path):
                errors.append(f"ref_image is not a supported image file: {normalized['ref_image']}")
        if not labels:
            missing.append("labels")

        for key in ("model", "epochs", "imgsz", "batch", "device"):
            if training.get(key) in {None, ""}:
                missing.append(f"training.{key}")
        for key in ("epochs", "imgsz", "batch", "workers", "patience"):
            if key not in training or training.get(key) in {None, ""}:
                continue
            try:
                if int(training[key]) <= 0:
                    errors.append(f"training.{key} must be greater than 0")
            except (TypeError, ValueError):
                errors.append(f"training.{key} must be an integer")
        if "amp" in training and not isinstance(training["amp"], bool):
            raw_amp = str(training["amp"]).strip().lower()
            if raw_amp not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                errors.append("training.amp must be a boolean")
        if not runtime.get("conda_env_name"):
            missing.append("runtime.conda_env_name")

        split_values: list[Any] = [
            split.get(key) for key in ("train", "val", "test") if split.get(key) not in {None, ""}
        ]
        if split_values:
            try:
                total = sum(float(value) for value in split_values)
                if abs(total - 1.0) > 0.05:
                    warnings.append(f"dataset split ratios sum to {total:.3f}; expected approximately 1.0.")
            except (TypeError, ValueError):
                errors.append("split ratios must be numeric")

        ready = not missing and not errors
        next_questions = self._yolo_next_questions(missing)
        text = "YOLO training inputs are ready." if ready else "YOLO training inputs are not ready."
        if missing:
            text += "\nMissing: " + ", ".join(missing)
        if errors:
            text += "\nErrors: " + "; ".join(errors)
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "ready": ready,
                "missing": missing,
                "errors": errors,
                "warnings": warnings,
                "next_questions": next_questions,
                "normalized": normalized,
            },
            is_error=False,
        )

    def _file_entries(
        self,
        paths: ThreadPaths,
        root: Path,
        scope: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        if max_results <= 0:
            return []
        entries: list[dict[str, Any]] = []
        for file_path in sorted(root.rglob("*")):
            if len(entries) >= max_results:
                break
            if not file_path.is_file():
                continue
            try:
                relative = file_path.resolve().relative_to(root.resolve()).as_posix()
                size = file_path.stat().st_size
            except OSError:
                continue
            virtual_path = f"/mnt/user-data/{scope}/{relative}"
            entry: dict[str, Any] = {
                "scope": scope,
                "path": relative,
                "virtual_path": virtual_path,
                "size": size,
                "kind": self._file_kind(file_path),
            }
            if scope == "workspace":
                entry["workspace_path"] = self._display_path(paths, file_path)
            entries.append(entry)
        return entries

    @staticmethod
    def _read_todos(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_items = data.get("todos") if isinstance(data, dict) else []
        if not isinstance(raw_items, list):
            return []
        todos: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, dict) and "id" in item and "text" in item:
                todos.append({"id": int(item["id"]), "text": str(item["text"]), "done": bool(item.get("done"))})
        return todos

    @staticmethod
    def _write_todos(path: Path, todos: list[dict[str, Any]]) -> None:
        path.write_text(json.dumps({"todos": todos}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _skip_file(path: Path) -> bool:
        try:
            return path.stat().st_size > 1_000_000
        except OSError:
            return True

    @staticmethod
    def _is_supported_archive(path: Path) -> bool:
        name = path.name.lower()
        return name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))

    @staticmethod
    def _is_image_file(path: Path) -> bool:
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".svg"}

    @staticmethod
    def _is_video_file(path: Path) -> bool:
        return path.suffix.lower() in {
            ".mp4",
            ".mov",
            ".m4v",
            ".avi",
            ".mkv",
            ".webm",
            ".flv",
            ".wmv",
            ".mpeg",
            ".mpg",
        }

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        if path.suffix.lower() == ".svg":
            return "image/svg+xml;charset=UTF-8"
        guessed, _encoding = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    @classmethod
    def _file_kind(cls, path: Path) -> str:
        if cls._is_supported_archive(path):
            return "archive"
        if cls._is_image_file(path):
            return "image"
        if cls._is_video_file(path):
            return "video"
        if path.suffix.lower() in {".yaml", ".yml", ".json", ".txt", ".md", ".csv"}:
            return "text"
        return "file"

    @staticmethod
    def _content_length(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value.strip())
        except ValueError:
            return None

    @staticmethod
    def _header_mime_type(value: str | None) -> str:
        if not value:
            return ""
        return value.split(";", 1)[0].strip().lower()

    @classmethod
    def _download_mime_type(cls, header_mime: str, guessed_mime: str) -> str:
        clean_guess = cls._header_mime_type(guessed_mime)
        if header_mime and header_mime != "application/octet-stream":
            return header_mime
        return clean_guess or header_mime or "application/octet-stream"

    @staticmethod
    def _mime_prefixes(value: object, *, default: list[str]) -> list[str]:
        raw_items: list[str]
        if isinstance(value, str):
            raw_items = value.replace("，", ",").split(",")
        elif isinstance(value, list):
            raw_items = [str(item) for item in value]
        else:
            raw_items = default
        prefixes: list[str] = []
        for item in raw_items:
            clean = item.strip().lower()
            if clean and clean not in prefixes:
                prefixes.append(clean)
        return prefixes or default

    @staticmethod
    def _mime_allowed(mime_type: str, allowed_prefixes: list[str]) -> bool:
        clean = mime_type.strip().lower()
        return any(clean.startswith(prefix) for prefix in allowed_prefixes)

    @classmethod
    def _download_filename(cls, url: str, requested: str, mime_type: str) -> str:
        raw_name = requested.strip() or Path(unquote(urlparse(url).path)).name
        if not raw_name:
            raw_name = "downloaded"
        name = Path(raw_name.replace("\\", "/")).name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        if not name:
            name = "downloaded"
        if "." not in name:
            extension = mimetypes.guess_extension(mime_type) if mime_type else None
            if extension:
                name += extension
        return name[:180]

    @staticmethod
    def _upload_target(paths: ThreadPaths, filename: str, *, overwrite: bool) -> Path:
        uploads = paths.uploads.resolve()
        target = (uploads / filename).resolve()
        try:
            target.relative_to(uploads)
        except ValueError as exc:
            raise ValueError("Upload path traversal blocked") from exc
        if overwrite or not target.exists():
            return target
        stem = target.stem or "downloaded"
        suffix = target.suffix
        for index in range(1, 10_000):
            candidate = (uploads / f"{stem}-{index}{suffix}").resolve()
            try:
                candidate.relative_to(uploads)
            except ValueError as exc:
                raise ValueError("Upload path traversal blocked") from exc
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not choose a unique upload filename for {filename}")

    @classmethod
    def _extract_archive_to(cls, archive: Path, output_dir: Path, *, max_files: int, max_bytes: int) -> list[Path]:
        extracted: list[Path] = []
        total_bytes = 0
        if archive.name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive) as handle:
                infos = [info for info in handle.infolist() if not info.is_dir()]
                if len(infos) > max_files:
                    raise ValueError(f"Archive contains too many files: {len(infos)} > {max_files}")
                for info in infos:
                    cls._reject_unsafe_archive_name(info.filename)
                    total_bytes += max(0, int(info.file_size))
                    if total_bytes > max_bytes:
                        raise ValueError(f"Archive expands beyond max_bytes: {total_bytes} > {max_bytes}")
                    target = cls._safe_extract_target(output_dir, info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(info) as source, target.open("wb") as dest:
                        dest.write(source.read())
                    extracted.append(target)
        else:
            with tarfile.open(archive) as handle:
                members = [member for member in handle.getmembers() if member.isfile()]
                if len(members) > max_files:
                    raise ValueError(f"Archive contains too many files: {len(members)} > {max_files}")
                for member in members:
                    cls._reject_unsafe_archive_name(member.name)
                    if member.issym() or member.islnk():
                        raise ValueError(f"Archive links are not allowed: {member.name}")
                    total_bytes += max(0, int(member.size))
                    if total_bytes > max_bytes:
                        raise ValueError(f"Archive expands beyond max_bytes: {total_bytes} > {max_bytes}")
                    target = cls._safe_extract_target(output_dir, member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tar_source = handle.extractfile(member)
                    if tar_source is None:
                        continue
                    with tar_source, target.open("wb") as dest:
                        dest.write(tar_source.read())
                    extracted.append(target)
        return extracted

    @staticmethod
    def _reject_unsafe_archive_name(name: str) -> None:
        normalized = name.replace("\\", "/")
        parts = Path(normalized).parts
        if normalized.startswith("/") or any(part in {"..", ""} for part in parts):
            raise ValueError(f"Unsafe archive path blocked: {name}")

    @staticmethod
    def _safe_extract_target(output_dir: Path, name: str) -> Path:
        target = (output_dir / name.replace("\\", "/")).resolve()
        root = output_dir.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Archive path traversal blocked: {name}") from exc
        return target

    @classmethod
    def _best_dataset_root(cls, output_dir: Path, image_files: list[Path]) -> Path:
        if not image_files:
            return output_dir
        image_dirs = {path.parent for path in image_files}
        candidates = [output_dir, output_dir / "images", output_dir / "dataset", output_dir / "data"]
        for candidate in candidates:
            if candidate.is_dir() and any(path.is_relative_to(candidate) for path in image_files):
                if candidate != output_dir or len(image_dirs) != 1:
                    return candidate
        common = Path(os.path.commonpath([str(path.parent) for path in image_files]))
        return common if common.is_dir() else output_dir

    @staticmethod
    def _top_level_entries(output_dir: Path) -> list[str]:
        try:
            return sorted(path.name for path in output_dir.iterdir())[:50]
        except OSError:
            return []

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if isinstance(value, str):
            raw_items = value.replace("，", ",").replace("、", ",").split(",")
        elif isinstance(value, list):
            raw_items = [str(item) for item in value]
        else:
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            clean = item.strip().strip("'\"`")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            result.append(clean)
        return result

    @staticmethod
    def _yolo_next_questions(missing: list[str]) -> list[str]:
        questions: list[str] = []
        if "dataset_root" in missing:
            questions.append("请上传数据集压缩包，或提供解压后的 dataset_root。")
        if "labels" in missing:
            questions.append("请补充 labels=person,cigarette 这样的检测类别。")
        if any(item.startswith("training.") for item in missing):
            questions.append("请补充 model/epochs/imgsz/batch/device 等训练参数。")
        if "runtime.conda_env_name" in missing:
            questions.append("请补充 conda_env_name，例如 conda_env_name=cv_train。")
        return questions

    @staticmethod
    def _display_path(paths: ThreadPaths, path: Path, arguments: dict[str, Any] | None = None) -> str:
        raw_session_cwd = str((arguments or {}).get("_session_cwd") or "").strip()
        if raw_session_cwd:
            try:
                return path.resolve().relative_to(Path(raw_session_cwd).expanduser().resolve()).as_posix()
            except ValueError:
                pass
        for root in LocalToolProvider._allowed_skill_roots(arguments):
            try:
                return f"/mnt/user-data/skills/{root.name}/{path.resolve().relative_to(root).as_posix()}"
            except ValueError:
                pass
        try:
            return f"/mnt/user-data/uploads/{path.resolve().relative_to(paths.uploads.resolve()).as_posix()}"
        except ValueError:
            pass
        try:
            return f"/mnt/user-data/outputs/{path.resolve().relative_to(paths.outputs.resolve()).as_posix()}"
        except ValueError:
            pass
        try:
            return path.resolve().relative_to(paths.workspace.resolve()).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _thread_scoped_path(
        paths: ThreadPaths,
        raw_path: str,
        arguments: dict[str, Any] | None = None,
        *,
        scopes: set[str],
    ) -> Path:
        if not raw_path.strip():
            raise ValueError("path is required")
        arguments = arguments or {}
        normalized_raw = raw_path.replace("\\", "/")
        virtual_roots = {
            "uploads": ("/mnt/user-data/uploads", paths.uploads),
            "workspace": ("/mnt/user-data/workspace", paths.workspace),
            "outputs": ("/mnt/user-data/outputs", paths.outputs),
        }
        for scope, (prefix, root) in virtual_roots.items():
            if scope not in scopes:
                continue
            if normalized_raw == prefix or normalized_raw.startswith(prefix + "/"):
                suffix = normalized_raw[len(prefix):].lstrip("/")
                candidate = (root / suffix).resolve()
                try:
                    candidate.relative_to(root.resolve())
                except ValueError as exc:
                    raise ValueError(f"{scope} path traversal blocked") from exc
                return candidate
        raw_session_cwd = str(arguments.get("_session_cwd") or "").strip()
        if raw_session_cwd:
            session_cwd = Path(raw_session_cwd).expanduser().resolve()
            candidate = (
                Path(normalized_raw).expanduser().resolve()
                if Path(normalized_raw).is_absolute()
                else (session_cwd / normalized_raw.lstrip("/")).resolve()
            )
            try:
                candidate.relative_to(session_cwd)
            except ValueError as exc:
                raise ValueError("Session path traversal blocked") from exc
            return candidate
        default_root = paths.outputs if "outputs" in scopes and normalized_raw.startswith("outputs/") else paths.workspace
        normalized = normalized_raw.removeprefix("outputs/").lstrip("/") if default_root == paths.outputs else normalized_raw.lstrip("/")
        candidate = (default_root / normalized).resolve()
        try:
            candidate.relative_to(default_root.resolve())
        except ValueError as exc:
            raise ValueError("Thread path traversal blocked") from exc
        return candidate

    @staticmethod
    def _workspace_path(paths: ThreadPaths, raw_path: str, arguments: dict[str, Any] | None = None) -> Path:
        if not raw_path.strip():
            raise ValueError("path is required")
        arguments = arguments or {}
        normalized_raw = raw_path.replace("\\", "/")
        uploads_prefix = "/mnt/user-data/uploads"
        if normalized_raw == uploads_prefix or normalized_raw.startswith(uploads_prefix + "/"):
            suffix = normalized_raw[len(uploads_prefix):].lstrip("/")
            candidate = (paths.uploads / suffix).resolve()
            uploads = paths.uploads.resolve()
            try:
                candidate.relative_to(uploads)
            except ValueError as exc:
                raise ValueError("Upload path traversal blocked") from exc
            return candidate
        workspace_prefix = "/mnt/user-data/workspace"
        if normalized_raw == workspace_prefix or normalized_raw.startswith(workspace_prefix + "/"):
            suffix = normalized_raw[len(workspace_prefix):].lstrip("/")
            candidate = (paths.workspace / suffix).resolve()
            workspace = paths.workspace.resolve()
            try:
                candidate.relative_to(workspace)
            except ValueError as exc:
                raise ValueError("Workspace path traversal blocked") from exc
            return candidate
        raw_session_cwd = str(arguments.get("_session_cwd") or "").strip()
        if raw_session_cwd:
            session_cwd = Path(raw_session_cwd).expanduser().resolve()
            if not session_cwd.is_dir():
                raise NotADirectoryError(f"session cwd is not a directory: {session_cwd}")
            candidate = (
                Path(normalized_raw).expanduser().resolve()
                if Path(normalized_raw).is_absolute()
                else (session_cwd / normalized_raw.lstrip("/")).resolve()
            )
            try:
                candidate.relative_to(session_cwd)
            except ValueError as exc:
                raise ValueError("Workspace path traversal blocked") from exc
            return candidate
        normalized = normalized_raw.lstrip("/")
        candidate = (paths.workspace / normalized).resolve()
        workspace = paths.workspace.resolve()
        try:
            candidate.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("Workspace path traversal blocked") from exc
        return candidate

    @staticmethod
    def _readable_path(paths: ThreadPaths, raw_path: str, arguments: dict[str, Any] | None = None) -> Path:
        skill_path = LocalToolProvider._skill_read_path(raw_path, arguments)
        if skill_path is not None:
            return skill_path
        return LocalToolProvider._workspace_path(paths, raw_path, arguments)

    @staticmethod
    def _skill_read_path(raw_path: str, arguments: dict[str, Any] | None = None) -> Path | None:
        if not raw_path.strip():
            return None
        normalized_raw = raw_path.replace("\\", "/").strip()
        raw_candidate = Path(normalized_raw).expanduser()
        if raw_candidate.is_absolute():
            absolute_candidates = [raw_candidate]
        elif normalized_raw:
            absolute_candidates = [Path("/" + normalized_raw)]
        else:
            absolute_candidates = []
        for root in LocalToolProvider._allowed_skill_roots(arguments):
            candidate_paths = list(absolute_candidates)
            if not raw_candidate.is_absolute():
                candidate_paths.append(root / normalized_raw.lstrip("/"))
            for candidate in candidate_paths:
                resolved = candidate.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                if resolved.is_file():
                    return resolved
        return None

    @staticmethod
    def _allowed_skill_roots(arguments: dict[str, Any] | None = None) -> list[Path]:
        raw_roots = (arguments or {}).get("_skill_roots")
        if not isinstance(raw_roots, list):
            return []
        roots: list[Path] = []
        seen: set[Path] = set()
        for item in raw_roots:
            raw = str(item or "").strip()
            if not raw:
                continue
            root = Path(raw).expanduser().resolve()
            if root in seen or not root.is_dir():
                continue
            seen.add(root)
            roots.append(root)
        return roots


def local_tool_definitions() -> list[ToolDefinition]:
    shell_enabled = os.getenv("LOCAL_SHELL_TOOL_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    return [
        ToolDefinition(
            name="local_read_file",
            title="Read Thread File",
            description="Read a UTF-8 text file from the current thread workspace or uploaded files.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000},
                },
                "required": ["path"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "read_file"},
            editable=False,
        ),
        ToolDefinition(
            name="local_write_file",
            title="Write Workspace File",
            description="Write UTF-8 text into the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "write_file"},
            editable=False,
        ),
        ToolDefinition(
            name="local_download_url",
            title="Download URL To Uploads",
            description=(
                "Download an http/https URL into the current thread uploads directory. "
                "Use this when a user-provided resource_link points to an image or video URL that tools need "
                "as a local /mnt/user-data/uploads file."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http:// or https:// URL to download."},
                    "filename": {
                        "type": "string",
                        "description": "Optional output filename. Path components are stripped for safety.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 524288000,
                        "default": 52428800,
                    },
                    "allowed_mime_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["image/", "video/"],
                        "description": "Allowed MIME prefixes. Defaults to images and videos.",
                    },
                    "overwrite": {"type": "boolean", "default": True},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
                },
                "required": ["url"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "path": {"type": "string"},
                    "filename": {"type": "string"},
                    "scope": {"type": "string"},
                    "bytes": {"type": "integer"},
                    "mime_type": {"type": "string"},
                    "kind": {"type": "string"},
                },
            },
            source={"type": LocalToolProvider.source_type, "operation": "download_url"},
            editable=False,
        ),
        ToolDefinition(
            name="local_file_to_base64",
            title="Encode Thread File As Base64",
            description=(
                "Read a file from uploads, workspace, or outputs and return Base64 plus an optional data URI. "
                "Use this before UploadFile-style platform commands that require Base64 file content."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path under /mnt/user-data/uploads, /mnt/user-data/workspace, or /mnt/user-data/outputs.",
                    },
                    "mime_type": {
                        "type": "string",
                        "description": "Optional MIME type override. SVG defaults to image/svg+xml;charset=UTF-8.",
                    },
                    "include_data_uri": {"type": "boolean", "default": True},
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10485760,
                        "default": 2097152,
                    },
                },
                "required": ["path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "bytes": {"type": "integer"},
                    "mime_type": {"type": "string"},
                    "encoding": {"type": "string"},
                    "base64": {"type": "string"},
                    "data_uri": {"type": ["string", "null"]},
                },
            },
            source={"type": LocalToolProvider.source_type, "operation": "file_to_base64"},
            editable=False,
        ),
        ToolDefinition(
            name="local_patch_file",
            title="Patch Workspace File",
            description=(
                "Patch a UTF-8 text file by replacing one exact old_text occurrence with new_text. "
                "Use this for localized edits, especially in large files. The patch is rejected if "
                "old_text is missing or matches more than once."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "patch_file"},
            editable=False,
        ),
        ToolDefinition(
            name="local_search_text",
            title="Search Workspace Text",
            description="Search text files in the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "case_sensitive": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["pattern"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "search_text"},
            editable=False,
        ),
        ToolDefinition(
            name="local_todo",
            title="Thread Todo",
            description="Manage a simple todo list stored in the current thread workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "add", "complete", "clear"]},
                    "text": {"type": "string"},
                    "id": {"type": "integer"},
                },
                "required": ["action"],
            },
            source={"type": LocalToolProvider.source_type, "operation": "todo"},
            editable=False,
        ),
        ToolDefinition(
            name="local_shell_command",
            title="Run Workspace Shell Command",
            description="Run a shell command inside the current thread workspace. Disabled unless LOCAL_SHELL_TOOL_ENABLED=true.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 60},
                    "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000},
                },
                "required": ["command"],
            },
            enabled=shell_enabled,
            source={"type": LocalToolProvider.source_type, "operation": "shell_command"},
            editable=False,
        ),
        ToolDefinition(
            name="extract_archive",
            title="Extract Thread Archive",
            description=(
                "Safely extract a .zip, .tar, .tar.gz, or .tgz file from uploads/workspace/outputs into "
                "the current thread workspace or outputs, then return the extracted dataset_root and image count."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Archive path, usually /mnt/user-data/uploads/name.zip."},
                    "output_dir": {
                        "type": "string",
                        "description": "Destination under /mnt/user-data/workspace or /mnt/user-data/outputs.",
                    },
                    "overwrite": {"type": "boolean", "default": True},
                    "max_files": {"type": "integer", "minimum": 1, "maximum": 100000},
                    "max_bytes": {"type": "integer", "minimum": 1024},
                },
                "required": ["path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "dataset_root": {"type": "string"},
                    "image_count": {"type": "integer"},
                    "file_count": {"type": "integer"},
                },
            },
            source={"type": LocalToolProvider.source_type, "operation": "extract_archive"},
            editable=False,
        ),
        ToolDefinition(
            name="validate_yolo_training_inputs",
            title="Validate YOLO Training Inputs",
            description=(
                "Check whether dataset_root, reference image, labels, runtime, and training parameters are ready "
                "before calling gpu-training-orchestrator. This tool does not train."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_root": {"type": "string"},
                    "ref_image": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "training": {"type": "object"},
                    "runtime": {"type": "object"},
                    "split": {"type": "object"},
                },
                "required": ["dataset_root", "labels", "training", "runtime"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean"},
                    "missing": {"type": "array"},
                    "errors": {"type": "array"},
                    "warnings": {"type": "array"},
                    "normalized": {"type": "object"},
                },
            },
            source={"type": LocalToolProvider.source_type, "operation": "validate_yolo_training_inputs"},
            editable=False,
        ),
        ToolDefinition(
            name="present_files",
            title="Present Thread Files",
            description="List files generated in outputs, uploaded by the user, and optionally workspace files.",
            input_schema={
                "type": "object",
                "properties": {
                    "include_workspace": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
                },
            },
            source={"type": LocalToolProvider.source_type, "operation": "present_files"},
            editable=False,
        ),
    ]
