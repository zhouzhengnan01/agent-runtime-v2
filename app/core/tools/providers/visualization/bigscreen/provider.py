from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from app.core.artifacts import ArtifactStore, ThreadPaths
from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


BIGSCREEN_TOOL_NAMES = {
    "visual_bigscreen_upload_file",
    "visualization_bigscreen_upload_file",
    "visualization_bigscreen_save_resource",
}


class BigscreenToolset:
    source_type = "visualization_bigscreen"

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def call(self, tool: ToolDefinition, arguments: dict[str, Any]) -> ToolInvocationResult:
        operation = str(tool.source.get("operation") or "")
        try:
            if operation == "upload_file":
                thread_id = str(arguments.get("_thread_id") or "mcp-local")
                paths = self.artifact_store.prepare_thread(thread_id)
                return self._upload_file(paths, arguments)
            if operation == "save_resource":
                return self._save_resource(arguments)
        except Exception as exc:
            return ToolInvocationResult(
                content=[{"type": "text", "text": f"{tool.name} failed: {exc}"}],
                structured_content={"tool_name": tool.name, "error": str(exc)},
                is_error=True,
            )
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Unsupported bigscreen operation: {operation}"}],
            structured_content={"tool_name": tool.name, "operation": operation},
            is_error=True,
        )

    def _upload_file(self, paths: ThreadPaths, arguments: dict[str, Any]) -> ToolInvocationResult:
        target = self._thread_scoped_path(
            paths,
            str(arguments.get("file_path") or arguments.get("path") or ""),
            arguments,
            scopes={"uploads", "workspace", "outputs"},
        )
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self._display_path(paths, target, arguments)}")
        max_bytes = self._bounded_int(
            arguments.get("max_bytes"),
            default=10 * 1024 * 1024,
            minimum=1,
            maximum=50 * 1024 * 1024,
        )
        file_size = target.stat().st_size
        if file_size > max_bytes:
            raise ValueError(f"File is too large for platform upload: {file_size} bytes > {max_bytes} bytes")

        content_type = str(arguments.get("content_type") or arguments.get("contentType") or "").strip()
        if not content_type:
            content_type = self._guess_mime_type(target)
        file_name = self._download_filename(
            "",
            str(arguments.get("file_name") or arguments.get("fileName") or target.name),
            content_type,
        )
        content = base64.b64encode(target.read_bytes()).decode("ascii")
        mcp_sources = self._platform_mcp_sources(arguments)
        if mcp_sources:
            return self._call_upload_file_via_mcp(
                paths,
                arguments,
                target=target,
                file_name=file_name,
                content_type=content_type,
                content=content,
                file_size=file_size,
                mcp_sources=mcp_sources,
            )

        upload_urls = self._platform_upload_urls(arguments)
        if not upload_urls:
            raise ValueError(
                "platform upload MCP server is missing. Pass mcpServers in runtime config, or provide "
                "visualBigscreenUploadUrl/fileUploadUrl for the legacy REST upload fallback."
            )
        headers = self._platform_upload_headers(arguments)
        timeout_seconds = self._bounded_int(arguments.get("timeout_seconds"), default=30, minimum=1, maximum=180)
        errors: list[str] = []
        for upload_url in upload_urls:
            try:
                with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                    with target.open("rb") as handle:
                        response = client.post(
                            upload_url,
                            headers=headers,
                            files={"file": (file_name, handle, content_type)},
                        )
                if response.status_code >= 400:
                    errors.append(f"{self._redacted_url(upload_url)} returned HTTP {response.status_code}: {response.text[:300]}")
                    continue
                payload = self._response_payload(response)
                file_id = self._find_id(payload)
                if not file_id:
                    errors.append(f"{self._redacted_url(upload_url)} returned no id/fileId")
                    continue
                display_path = self._display_path(paths, target, arguments)
                return ToolInvocationResult(
                    content=[{"type": "text", "text": f"Uploaded {display_path} to platform as {file_name}; file_id={file_id}."}],
                    structured_content={
                        "tool_name": "visualization_bigscreen_upload_file",
                        "path": display_path,
                        "file_name": file_name,
                        "content_type": content_type,
                        "bytes": file_size,
                        "file_id": file_id,
                        "id": file_id,
                        "upload_url": self._redacted_url(upload_url),
                        "response": payload,
                    },
                    is_error=False,
                )
            except Exception as exc:
                errors.append(f"{self._redacted_url(upload_url)} failed: {exc}")
        raise RuntimeError("platform upload failed: " + "; ".join(errors[:5]))

    def _call_upload_file_via_mcp(
        self,
        paths: ThreadPaths,
        arguments: dict[str, Any],
        *,
        target: Path,
        file_name: str,
        content_type: str,
        content: str,
        file_size: int,
        mcp_sources: list[dict[str, Any]],
    ) -> ToolInvocationResult:
        server_tool = self._configured_mcp_tool_name(
            arguments,
            "mcp_tool",
            "mcpTool",
            "_visual_bigscreen_upload_mcp_tool",
            env_key="VISUAL_BIGSCREEN_UPLOAD_MCP_TOOL",
            default="visual-bigscreen_UploadFile",
        )
        tool_arguments = {"fileName": file_name, "contentType": content_type, "content": content}
        result = self._call_mcp_tool(arguments, mcp_sources, server_tool, tool_arguments)
        file_id = self._find_id(result.structured_content) or self._find_id(result.content)
        if not file_id:
            file_id = self._find_id_from_tool_content(result.content)
        if not file_id:
            raise RuntimeError(f"{server_tool} returned no id/fileId")
        display_path = self._display_path(paths, target, arguments)
        return ToolInvocationResult(
            content=[{"type": "text", "text": f"Uploaded {display_path} to platform via MCP {server_tool} as {file_name}; file_id={file_id}."}],
            structured_content={
                "tool_name": "visualization_bigscreen_upload_file",
                "path": display_path,
                "file_name": file_name,
                "content_type": content_type,
                "bytes": file_size,
                "file_id": file_id,
                "id": file_id,
                "mcp_tool": server_tool,
                "mcp_response": result.structured_content,
                "mcp_content": result.content,
            },
            is_error=False,
        )

    def _save_resource(self, arguments: dict[str, Any]) -> ToolInvocationResult:
        data = arguments.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError("data must be a non-empty array")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("data must contain resource entity objects")
        mcp_sources = self._platform_mcp_sources(arguments)
        if not mcp_sources:
            raise ValueError("platform MCP server is missing. Pass mcpServers in runtime config.")
        server_tool = self._configured_mcp_tool_name(
            arguments,
            "mcp_tool",
            "mcpTool",
            "_visual_bigscreen_resource_mcp_tool",
            env_key="VISUAL_BIGSCREEN_RESOURCE_MCP_TOOL",
            default="visual-bigscreen_Add",
        )
        result = self._call_mcp_tool(arguments, mcp_sources, server_tool, {"data": data})
        entities = self._saved_entities(result.structured_content, result.content)
        text = f"Saved {len(data)} visualization resource entr{'y' if len(data) == 1 else 'ies'} via MCP {server_tool}."
        return ToolInvocationResult(
            content=[{"type": "text", "text": text}],
            structured_content={
                "tool_name": "visualization_bigscreen_save_resource",
                "mcp_tool": server_tool,
                "data": entities if entities else data,
                "saved": entities if entities else data,
                "mcp_response": result.structured_content,
                "mcp_content": result.content,
            },
            is_error=False,
        )

    def _call_mcp_tool(
        self,
        arguments: dict[str, Any],
        mcp_sources: list[dict[str, Any]],
        server_tool: str,
        tool_arguments: dict[str, Any],
    ) -> ToolInvocationResult:
        from app.core.tools.providers.mcp_streamable_http import McpStreamableHttpToolProvider

        timeout_seconds = self._bounded_int(arguments.get("timeout_seconds"), default=30, minimum=1, maximum=300)
        provider = McpStreamableHttpToolProvider()
        errors: list[str] = []
        for source in mcp_sources:
            url = self._first_string(source, "url")
            if not url:
                continue
            tool_source = dict(source)
            tool_source["type"] = McpStreamableHttpToolProvider.source_type
            tool_source["operation"] = "call"
            tool_source["server_tool"] = server_tool
            tool_source.setdefault("timeout_seconds", timeout_seconds)
            result = provider.call(
                ToolDefinition(
                    name=f"{self.source_type}_{server_tool}_bridge",
                    title=f"{server_tool} MCP Bridge",
                    description=f"Bridge to Java MCP {server_tool}.",
                    input_schema={"type": "object", "additionalProperties": True},
                    source=tool_source,
                    editable=False,
                ),
                tool_arguments,
            )
            if result.is_error:
                errors.append(f"{self._redacted_url(url)} {server_tool} failed: {self._tool_result_error_text(result)}")
                continue
            return result
        raise RuntimeError("platform MCP call failed: " + "; ".join(errors[:5]))

    @classmethod
    def _platform_upload_urls(cls, arguments: dict[str, Any]) -> list[str]:
        configured = (
            cls._first_string(arguments, "upload_url", "uploadUrl", "_visual_bigscreen_upload_url", "_file_upload_url")
            or os.getenv("VISUAL_BIGSCREEN_UPLOAD_URL", "").strip()
            or os.getenv("JETLINKS_FILE_UPLOAD_URL", "").strip()
        )
        if configured:
            return [configured]
        return []

    @classmethod
    def _platform_mcp_sources(cls, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        raw_tools = arguments.get("_runtime_mcp_tools")
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                if isinstance(source, dict):
                    sources.append(cls._mcp_source_from_runtime_source(source))
        raw_servers = arguments.get("_runtime_mcp_servers")
        if isinstance(raw_servers, list):
            for item in raw_servers:
                if isinstance(item, dict):
                    sources.append(cls._mcp_source_from_runtime_server(item))
        return cls._unique_mcp_sources(sources)

    @classmethod
    def _mcp_source_from_runtime_source(cls, source: dict[str, Any]) -> dict[str, Any]:
        result = {"url": cls._first_string(source, "url"), "headers": cls._headers_dict(source.get("headers"))}
        protocol_version = cls._first_string(source, "protocol_version", "protocolVersion")
        if protocol_version:
            result["protocol_version"] = protocol_version
        timeout = source.get("timeout_seconds") or source.get("timeoutSeconds")
        if isinstance(timeout, int | float):
            result["timeout_seconds"] = timeout
        return result

    @classmethod
    def _mcp_source_from_runtime_server(cls, server: dict[str, Any]) -> dict[str, Any]:
        result = {"url": cls._first_string(server, "url"), "headers": cls._headers_dict(server.get("headers"))}
        meta = server.get("_meta")
        if isinstance(meta, dict):
            protocol_version = cls._first_string(meta, "protocolVersion", "protocol_version")
            if protocol_version:
                result["protocol_version"] = protocol_version
            timeout = meta.get("timeoutSeconds") or meta.get("timeout_seconds")
            if isinstance(timeout, int | float):
                result["timeout_seconds"] = timeout
        return result

    @classmethod
    def _unique_mcp_sources(cls, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source in sources:
            url = cls._first_string(source, "url")
            if not url or url in seen:
                continue
            seen.add(url)
            result.append(source)
        return result

    @classmethod
    def _configured_mcp_tool_name(
        cls,
        arguments: dict[str, Any],
        *keys: str,
        env_key: str,
        default: str,
    ) -> str:
        return cls._first_string(arguments, *keys) or os.getenv(env_key, "").strip() or default

    @classmethod
    def _platform_upload_headers(cls, arguments: dict[str, Any]) -> dict[str, str]:
        headers: dict[str, str] = {}
        raw_tools = arguments.get("_runtime_mcp_tools")
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if not isinstance(item, dict):
                    continue
                source = item.get("source")
                if isinstance(source, dict):
                    cls._merge_headers(headers, source.get("headers"))
        raw_servers = arguments.get("_runtime_mcp_servers")
        if isinstance(raw_servers, list):
            for item in raw_servers:
                if isinstance(item, dict):
                    cls._merge_headers(headers, item.get("headers"))
        authorization = os.getenv("VISUAL_BIGSCREEN_UPLOAD_AUTHORIZATION", "").strip()
        if authorization:
            headers["Authorization"] = authorization
        return headers

    @classmethod
    def _headers_dict(cls, raw_headers: object) -> dict[str, str]:
        headers: dict[str, str] = {}
        cls._merge_headers(headers, raw_headers)
        return headers

    @staticmethod
    def _merge_headers(target: dict[str, str], raw_headers: object) -> None:
        if isinstance(raw_headers, dict):
            items = raw_headers.items()
        elif isinstance(raw_headers, list):
            items = (
                (str(item.get("name") or ""), str(item.get("value") or ""))
                for item in raw_headers
                if isinstance(item, dict)
            )
        else:
            return
        for key, value in items:
            clean_key = str(key or "").strip()
            clean_value = str(value or "").strip()
            if not clean_key or not clean_value:
                continue
            if clean_key.lower() in {"content-type", "content-length"}:
                continue
            target[clean_key] = clean_value

    @staticmethod
    def _response_payload(response: httpx.Response) -> dict[str, Any]:
        text = response.text.strip()
        if not text:
            return {}
        try:
            data = response.json()
        except ValueError:
            return {"text": text[:2000]}
        return data if isinstance(data, dict) else {"value": data}

    @classmethod
    def _find_id(cls, payload: object) -> str:
        if isinstance(payload, dict):
            for key in ("id", "fileId", "file_id", "resourceId", "resource_id"):
                value = payload.get(key)
                if isinstance(value, (str, int, float)) and str(value).strip():
                    return str(value).strip()
            for key in ("result", "data", "body", "content", "file", "fileInfo", "resource"):
                found = cls._find_id(payload.get(key))
                if found:
                    return found
            for value in payload.values():
                if isinstance(value, (dict, list)):
                    found = cls._find_id(value)
                    if found:
                        return found
        if isinstance(payload, list):
            for item in payload:
                found = cls._find_id(item)
                if found:
                    return found
        return ""

    @classmethod
    def _find_id_from_tool_content(cls, content: object) -> str:
        if not isinstance(content, list):
            return ""
        for item in content:
            if not isinstance(item, dict):
                continue
            found = cls._find_id(item)
            if found:
                return found
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            found = cls._find_id(cls._parse_possible_json(text))
            if found:
                return found
            match = re.search(r"\b(?:fileId|file_id|resourceId|resource_id|id)\s*[=:]\s*([A-Za-z0-9._:-]+)", text)
            if match:
                return match.group(1)
        return ""

    @classmethod
    def _saved_entities(cls, structured: object, content: object) -> list[dict[str, Any]]:
        candidates = cls._entity_arrays(structured)
        if candidates:
            return candidates[0]
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                candidates = cls._entity_arrays(item)
                if candidates:
                    return candidates[0]
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    candidates = cls._entity_arrays(cls._parse_possible_json(text))
                    if candidates:
                        return candidates[0]
        return []

    @classmethod
    def _entity_arrays(cls, payload: object) -> list[list[dict[str, Any]]]:
        arrays: list[list[dict[str, Any]]] = []
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            arrays.append([dict(item) for item in payload])
        elif isinstance(payload, dict):
            for key in ("data", "result", "value", "rows", "list", "items"):
                arrays.extend(cls._entity_arrays(payload.get(key)))
            for value in payload.values():
                if isinstance(value, (dict, list)):
                    arrays.extend(cls._entity_arrays(value))
        return arrays

    @staticmethod
    def _parse_possible_json(text: str) -> object:
        try:
            return json.loads(text)
        except ValueError:
            return {}

    @staticmethod
    def _tool_result_error_text(result: ToolInvocationResult) -> str:
        error = result.structured_content.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        texts = [
            str(item.get("text")).strip()
            for item in result.content
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        return "; ".join(texts)[:1000] if texts else "unknown error"

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value) if isinstance(value, (str, int, float)) else default
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _guess_mime_type(path: Path) -> str:
        if path.suffix.lower() == ".svg":
            return "image/svg+xml;charset=UTF-8"
        guessed, _encoding = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"

    @staticmethod
    def _display_path(paths: ThreadPaths, path: Path, arguments: dict[str, Any] | None = None) -> str:
        raw_session_cwd = str((arguments or {}).get("_session_cwd") or "").strip()
        if raw_session_cwd:
            try:
                return path.resolve().relative_to(Path(raw_session_cwd).expanduser().resolve()).as_posix()
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
                if scope == "workspace" and suffix and not candidate.exists():
                    session_candidate = BigscreenToolset._session_cwd_path(arguments, suffix)
                    if session_candidate is not None and session_candidate.exists():
                        return session_candidate
                return candidate
        raw_session_cwd = str(arguments.get("_session_cwd") or "").strip()
        if raw_session_cwd:
            session_candidate = BigscreenToolset._session_cwd_path(arguments, normalized_raw)
            if session_candidate is not None:
                return session_candidate
        default_root = paths.outputs if "outputs" in scopes and normalized_raw.startswith("outputs/") else paths.workspace
        normalized = normalized_raw.removeprefix("outputs/").lstrip("/") if default_root == paths.outputs else normalized_raw.lstrip("/")
        candidate = (default_root / normalized).resolve()
        try:
            candidate.relative_to(default_root.resolve())
        except ValueError as exc:
            raise ValueError("Thread path traversal blocked") from exc
        return candidate

    @staticmethod
    def _session_cwd_path(arguments: dict[str, Any], raw_path: str) -> Path | None:
        raw_session_cwd = str(arguments.get("_session_cwd") or "").strip()
        if not raw_session_cwd:
            return None
        session_cwd = Path(raw_session_cwd).expanduser().resolve()
        normalized_raw = raw_path.replace("\\", "/")
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

    @staticmethod
    def _download_filename(url: str, requested: str, mime_type: str) -> str:
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
    def _redacted_url(url: str) -> str:
        parsed = urlparse(url)
        if not parsed.query:
            return url
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?***"

    @staticmethod
    def _first_string(source: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


def bigscreen_tool_definitions() -> list[ToolDefinition]:
    upload_input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path under /mnt/user-data/uploads, /mnt/user-data/workspace, or /mnt/user-data/outputs.",
            },
            "file_name": {
                "type": "string",
                "default": "background.svg",
                "description": "Filename sent to the UploadFile command as fileName.",
            },
            "content_type": {
                "type": "string",
                "default": "image/svg+xml;charset=UTF-8",
                "description": "MIME type sent to the UploadFile command as contentType.",
            },
        },
        "required": ["file_path"],
    }
    upload_output_schema = {
        "type": "object",
        "properties": {
            "file_id": {"type": "string"},
            "id": {"type": "string"},
            "path": {"type": "string"},
            "file_name": {"type": "string"},
            "content_type": {"type": "string"},
            "bytes": {"type": "integer"},
        },
    }
    return [
        ToolDefinition(
            name="visualization_bigscreen_upload_file",
            title="Upload Bigscreen File To Platform",
            description=(
                "Upload a generated bigscreen asset file through the Java MCP visual-bigscreen_UploadFile command, "
                "then return the platform file id. Use this before writing canvas.backgroundImage.fileId."
            ),
            input_schema=upload_input_schema,
            output_schema=upload_output_schema,
            source={"type": BigscreenToolset.source_type, "operation": "upload_file"},
            editable=False,
        ),
        ToolDefinition(
            name="visual_bigscreen_upload_file",
            title="Upload Bigscreen File To Platform",
            description=(
                "Compatibility alias for visualization_bigscreen_upload_file. Upload a generated bigscreen asset "
                "through the Java MCP visual-bigscreen_UploadFile command."
            ),
            input_schema=upload_input_schema,
            output_schema=upload_output_schema,
            source={"type": BigscreenToolset.source_type, "operation": "upload_file"},
            editable=False,
        ),
        ToolDefinition(
            name="visualization_bigscreen_save_resource",
            title="Save Bigscreen Resource",
            description=(
                "Save generated visualization resource entities through Java MCP visual-bigscreen_Add. "
                "Pass the already assembled command parameters exactly as {\"data\": [resourceEntity, ...]}; "
                "this tool forwards entities as-is and returns the Java saved entity array."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Resource entity array for visualizationService:resource / Add.",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                },
                "required": ["data"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "object"}},
                    "saved": {"type": "array", "items": {"type": "object"}},
                    "mcp_tool": {"type": "string"},
                },
            },
            source={"type": BigscreenToolset.source_type, "operation": "save_resource"},
            editable=False,
        ),
    ]
