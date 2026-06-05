from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import AgentConfig
from app.core.events import EventRecorder
from app.core.llm.openai_compatible import LlmToolCall
from app.core.skills import SkillRegistry
from app.core.tools.invocation import ToolInvocationService
from app.core.tools.schemas import ToolInvocationResult
from app.schemas import RuntimeOptions


_UPLOAD_FILE_MAX_BYTES = 10 * 1024 * 1024
_UPLOAD_FILE_NAME_KEYS = ("fileName", "filename", "file_name")
_UPLOAD_FILE_CONTENT_KEYS = ("content", "fileContent", "file_content", "base64")
_UPLOAD_FILE_CONTENT_TYPE_KEYS = ("contentType", "content_type", "mimeType", "mime_type")
_UPLOAD_FILE_PATH_KEYS = ("path", "filePath", "file_path")
_UPLOAD_FILE_VIRTUAL_ROOTS = (
    ("/mnt/user-data/uploads", "uploads"),
    ("/mnt/user-data/workspace", "workspace"),
    ("/mnt/user-data/outputs", "outputs"),
)


@dataclass(frozen=True)
class ToolExecutionOutcome:
    result: ToolInvocationResult
    arguments: dict[str, Any]
    duration_ms: float


class ToolOrchestrator:
    """Runtime-level wrapper around tool invocation.

    This keeps the public event contract identical to the previous direct
    `ToolInvocationService.call_tool` path while centralizing argument parsing,
    scoped runtime arguments, error mapping, and audit payload shaping.
    """

    def __init__(self, tool_service: ToolInvocationService) -> None:
        self.tool_service = tool_service

    def execute(
        self,
        tool_call: LlmToolCall,
        *,
        agent_config: AgentConfig,
        thread_id: str,
        recorder: EventRecorder,
        runtime_options: RuntimeOptions | None = None,
    ) -> ToolExecutionOutcome:
        started_at = time.perf_counter()
        recorder.emit("tool.started", {"tool_name": tool_call.name, "tool_call_id": tool_call.id})
        arguments: dict[str, Any] = {}
        try:
            arguments = self.tool_arguments(tool_call.arguments)
            self._apply_scoped_arguments(arguments, agent_config, thread_id, runtime_options)
            self._prepare_upload_file_arguments(tool_call.name, arguments)
            result = self.tool_service.call_tool(tool_call.name, arguments)
        except Exception as exc:
            error_code = self.tool_error_code(exc)
            result = ToolInvocationResult(
                content=[{"type": "text", "text": f"Tool {tool_call.name} failed: {exc}"}],
                structured_content={
                    "tool_name": tool_call.name,
                    "error": str(exc),
                    "error_code": error_code,
                    "recoverable": error_code in {"TOOL_ARGUMENTS_INVALID", "TOOL_NOT_FOUND", "TOOL_EXECUTION_FAILED"},
                },
                is_error=True,
            )

        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
        recorder.emit(
            "tool.completed" if not result.is_error else "tool.failed",
            {
                "tool_name": tool_call.name,
                "tool_call_id": tool_call.id,
                "is_error": result.is_error,
                "duration_ms": duration_ms,
                "arguments": self.observable_arguments(arguments),
                "error_code": result.structured_content.get("error_code") if result.is_error else None,
                "structured_content": result.structured_content,
            },
        )
        return ToolExecutionOutcome(result=result, arguments=arguments, duration_ms=duration_ms)

    def _apply_scoped_arguments(
        self,
        arguments: dict[str, Any],
        agent_config: AgentConfig,
        thread_id: str,
        runtime_options: RuntimeOptions | None,
    ) -> None:
        arguments["_thread_id"] = thread_id
        arguments["_agent_name"] = agent_config.name
        arguments["_memory_scope"] = agent_config.memory.scope
        arguments["_markdown_writable_scopes"] = agent_config.memory.markdown_writable_scopes
        arguments["_markdown_max_chars"] = agent_config.memory.markdown_max_chars
        arguments["_llm_model"] = (
            (runtime_options.model_name if runtime_options is not None else None)
            or agent_config.model.model
            or agent_config.model.default_model
            or ""
        )
        arguments["_llm_base_url"] = (
            (runtime_options.base_url if runtime_options is not None else None)
            or agent_config.model.base_url
            or ""
        )
        arguments["_llm_api_key"] = (
            (runtime_options.api_key if runtime_options is not None else None)
            or agent_config.model.api_key
            or ""
        )
        arguments["_llm_temperature"] = (
            runtime_options.temperature
            if runtime_options is not None and runtime_options.temperature is not None
            else agent_config.model.temperature
        )
        arguments["_llm_top_p"] = (
            runtime_options.top_p
            if runtime_options is not None and runtime_options.top_p is not None
            else agent_config.model.top_p
        )
        arguments["_llm_max_tokens"] = (
            runtime_options.max_tokens
            if runtime_options is not None and runtime_options.max_tokens is not None
            else agent_config.model.max_tokens
        )
        arguments["_llm_request_timeout_seconds"] = (
            runtime_options.request_timeout_seconds
            if runtime_options is not None and runtime_options.request_timeout_seconds is not None
            else agent_config.model.request_timeout_seconds
        )
        if runtime_options is not None:
            arguments["_user_id"] = runtime_options.user_id
            arguments["_project_id"] = runtime_options.project_id
            session_cwd = runtime_options.config_options.get("session_cwd")
            if isinstance(session_cwd, str) and session_cwd.strip():
                arguments["_session_cwd"] = session_cwd
            runtime_mcp_tools = runtime_options.config_options.get("runtime_mcp_tools")
            if isinstance(runtime_mcp_tools, list):
                arguments["_runtime_mcp_tools"] = runtime_mcp_tools
            skill_roots = self._selected_skill_roots(runtime_options)
            if skill_roots:
                arguments["_skill_roots"] = skill_roots

    @staticmethod
    def tool_arguments(raw_arguments: str) -> dict[str, Any]:
        if not raw_arguments.strip():
            return {}
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool arguments must be a JSON object")
        return parsed

    @staticmethod
    def observable_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        upload_file_content_autofilled = bool(arguments.get("_upload_file_content_autofilled"))
        visible = {
            key: ToolOrchestrator._observable_argument_value(key, value, upload_file_content_autofilled)
            for key, value in arguments.items()
            if not key.startswith("_")
        }
        rendered = json.dumps(visible, ensure_ascii=False, default=str)
        if len(rendered) <= 4000:
            return visible
        return {
            "_truncated": True,
            "json_chars": len(rendered),
            "preview": rendered[:3800],
        }

    @staticmethod
    def _observable_argument_value(key: str, value: Any, upload_file_content_autofilled: bool) -> Any:
        if key not in _UPLOAD_FILE_CONTENT_KEYS or not isinstance(value, str) or not value:
            return value
        if upload_file_content_autofilled or _looks_like_large_base64(value):
            return f"<base64 omitted chars={len(value)}>"
        return value

    @staticmethod
    def tool_error_code(exc: Exception) -> str:
        if isinstance(exc, json.JSONDecodeError):
            return "TOOL_ARGUMENTS_INVALID"
        if isinstance(exc, KeyError):
            return "TOOL_NOT_FOUND"
        message = str(exc).lower()
        if "disabled" in message:
            return "TOOL_DISABLED"
        if "arguments" in message and "json" in message:
            return "TOOL_ARGUMENTS_INVALID"
        return "TOOL_EXECUTION_FAILED"

    def _selected_skill_roots(self, runtime_options: RuntimeOptions) -> list[str]:
        roots: list[str] = []
        seen: set[str] = set()
        registry = SkillRegistry(self.tool_service.root_dir)
        for raw_name in runtime_options.selected_skills:
            name = str(raw_name or "").strip()
            if not name:
                continue
            try:
                skill = registry.get(name)
            except KeyError:
                continue
            root = skill.plugin_root
            if root is None and skill.manifest_path is not None:
                root = skill.manifest_path.parent
            if root is None:
                continue
            resolved = str(root.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(resolved)
        return roots

    def _prepare_upload_file_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if not _is_upload_file_tool(tool_name):
            return
        if _has_upload_file_content(arguments):
            return

        file_name = _first_string_argument(arguments, _UPLOAD_FILE_NAME_KEYS)
        file_path = _first_string_argument(arguments, _UPLOAD_FILE_PATH_KEYS)
        if not file_name and not file_path:
            return

        thread_id = str(arguments.get("_thread_id") or "").strip()
        if not thread_id:
            raise ValueError("UploadFile content is empty and _thread_id is missing.")
        paths = self.tool_service.artifact_store.prepare_thread(thread_id)
        target = (
            _resolve_upload_file_path(paths.uploads, paths.workspace, paths.outputs, file_path)
            if file_path
            else _find_upload_file_by_name(paths.uploads, paths.workspace, paths.outputs, file_name)
        )
        if target is None:
            lookup = file_path or file_name
            raise FileNotFoundError(f"UploadFile content is empty and no matching file was found: {lookup}")
        if not target.is_file():
            raise FileNotFoundError(f"UploadFile content source is not a file: {target}")

        file_size = target.stat().st_size
        if file_size > _UPLOAD_FILE_MAX_BYTES:
            raise ValueError(
                f"UploadFile content source is too large: {file_size} bytes > {_UPLOAD_FILE_MAX_BYTES} bytes"
            )

        arguments["content"] = base64.b64encode(target.read_bytes()).decode("ascii")
        arguments["_upload_file_content_autofilled"] = True
        if not file_name:
            arguments["fileName"] = target.name
        if not _first_string_argument(arguments, _UPLOAD_FILE_CONTENT_TYPE_KEYS):
            arguments["contentType"] = _guess_upload_file_content_type(target)


def _is_upload_file_tool(tool_name: str) -> bool:
    normalized = tool_name.rsplit("__", 1)[-1].rsplit("_", 1)[-1]
    return normalized == "UploadFile"


def _has_upload_file_content(arguments: dict[str, Any]) -> bool:
    for key in _UPLOAD_FILE_CONTENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _looks_like_large_base64(value: str) -> bool:
    stripped = value.strip()
    if stripped.startswith("data:") and ";base64," in stripped[:128]:
        return True
    if len(stripped) < 256 or len(stripped) % 4 != 0:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    return all(char in allowed for char in stripped)


def _first_string_argument(arguments: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _resolve_upload_file_path(uploads: Path, workspace: Path, outputs: Path, raw_path: str) -> Path | None:
    normalized = raw_path.replace("\\", "/").strip()
    for prefix, attr in _UPLOAD_FILE_VIRTUAL_ROOTS:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            root = {"uploads": uploads, "workspace": workspace, "outputs": outputs}[attr]
            suffix = normalized[len(prefix) :].lstrip("/")
            return _contained_path(root, suffix, traversal_message=f"{attr} path traversal blocked")
    return None


def _find_upload_file_by_name(uploads: Path, workspace: Path, outputs: Path, file_name: str) -> Path | None:
    normalized = file_name.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        return None
    matches: list[Path] = []
    for root in (uploads, workspace, outputs):
        candidate = _contained_path(root, normalized, traversal_message="UploadFile path traversal blocked")
        if candidate.is_file():
            matches.append(candidate)
        if "/" not in normalized:
            matches.extend(path for path in root.rglob(normalized) if path.is_file() and path != candidate)
    unique_matches = sorted({path.resolve() for path in matches})
    if not unique_matches:
        return None
    if len(unique_matches) > 1:
        rendered = ", ".join(str(path) for path in unique_matches[:5])
        raise ValueError(f"Multiple matching files found for UploadFile {file_name}: {rendered}")
    return unique_matches[0]


def _contained_path(root: Path, relative_path: str, *, traversal_message: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(traversal_message) from exc
    return candidate


def _guess_upload_file_content_type(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        return "image/svg+xml;charset=UTF-8"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
