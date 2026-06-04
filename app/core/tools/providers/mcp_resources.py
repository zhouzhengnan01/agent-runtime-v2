from __future__ import annotations

import json
from typing import Any

from app.core.tools.schemas import ToolDefinition, ToolInvocationResult


MAX_RESOURCE_ENUM_VALUES = 50
MAX_RESOURCE_SUMMARY_CHARS = 1500


def should_list_mcp_tools(capabilities: object) -> bool:
    if not isinstance(capabilities, dict) or not capabilities:
        return True
    return "tools" in capabilities


def should_list_mcp_resources(capabilities: object, server: dict[str, Any]) -> bool:
    if _resource_discovery_forced(server):
        return True
    if not isinstance(capabilities, dict):
        return False
    resources = capabilities.get("resources")
    return resources is True or isinstance(resources, dict)


def should_list_mcp_resource_templates(
    capabilities: object, server: dict[str, Any]
) -> bool:
    return should_list_mcp_resources(capabilities, server)


def resource_list_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cursor": {
                "type": "string",
                "description": "Opaque pagination cursor returned by a previous resources/list call.",
            }
        },
        "additionalProperties": False,
    }


def resource_template_list_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "cursor": {
                "type": "string",
                "description": "Opaque pagination cursor returned by a previous resources/templates/list call.",
            }
        },
        "additionalProperties": False,
    }


def resource_read_input_schema(
    resources: list[dict[str, Any]],
    resource_templates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    uri_schema: dict[str, Any] = {
        "type": "string",
        "description": _resource_uri_description(resource_templates or []),
    }
    known_uris = _known_resource_uris(resources)
    if 0 < len(known_uris) <= MAX_RESOURCE_ENUM_VALUES:
        uri_schema["enum"] = known_uris
    return {
        "type": "object",
        "properties": {"uri": uri_schema},
        "required": ["uri"],
        "additionalProperties": False,
    }


def resource_list_description(server: dict[str, Any]) -> str:
    server_name = _server_display_name(server)
    return (
        f"List MCP resources advertised by the remote server {server_name}. "
        "Use this when the user asks what context resources are available or when you need a URI to read."
    )


def resource_template_list_description(server: dict[str, Any]) -> str:
    server_name = _server_display_name(server)
    return (
        f"List MCP resource templates advertised by the remote server {server_name}. "
        "Use this when resources/list is empty but the server exposes templated resource URIs."
    )


def resource_read_description(
    server: dict[str, Any],
    resources: list[dict[str, Any]],
    resource_templates: list[dict[str, Any]] | None = None,
) -> str:
    server_name = _server_display_name(server)
    description = (
        f"Read an MCP resource by URI from the remote server {server_name}. "
        "Use an exact URI from the known resource list, an expanded resource template URI, "
        "a URI from the user, or a URI from a prior resources/list result."
    )
    summary = resource_summary(resources)
    template_summary = resource_template_summary(resource_templates or [])
    suffix = " ".join(item for item in (summary, template_summary) if item)
    return f"{description} {suffix}" if suffix else description


def resource_summary(resources: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        uri = _string(resource.get("uri"))
        if not uri:
            continue
        name = _string(resource.get("title") or resource.get("name")) or uri
        mime_type = _string(resource.get("mimeType") or resource.get("mime_type"))
        item = f"{name} <{uri}>"
        if mime_type:
            item = f"{item} ({mime_type})"
        lines.append(item)
        if len("; ".join(lines)) >= MAX_RESOURCE_SUMMARY_CHARS:
            break
    if not lines:
        return ""
    summary = "; ".join(lines)
    if len(summary) > MAX_RESOURCE_SUMMARY_CHARS:
        summary = f"{summary[:MAX_RESOURCE_SUMMARY_CHARS].rstrip()}..."
    return f"Known resources: {summary}."


def resource_template_summary(resource_templates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for template in resource_templates:
        if not isinstance(template, dict):
            continue
        uri_template = _string(template.get("uriTemplate") or template.get("uri_template"))
        if not uri_template:
            continue
        name = _string(template.get("title") or template.get("name")) or uri_template
        item = f"{name} <{uri_template}>"
        description = _string(template.get("description"))
        if description:
            item = f"{item} - {description}"
        lines.append(item)
        if len("; ".join(lines)) >= MAX_RESOURCE_SUMMARY_CHARS:
            break
    if not lines:
        return ""
    summary = "; ".join(lines)
    if len(summary) > MAX_RESOURCE_SUMMARY_CHARS:
        summary = f"{summary[:MAX_RESOURCE_SUMMARY_CHARS].rstrip()}..."
    return f"Known resource templates: {summary}."


def resource_cursor_params(arguments: dict[str, Any]) -> dict[str, str]:
    cursor = _string(
        arguments.get("cursor")
        or arguments.get("nextCursor")
        or arguments.get("next_cursor")
    )
    return {"cursor": cursor} if cursor else {}


def resource_uri(tool: ToolDefinition, arguments: dict[str, Any]) -> str:
    configured = _string(tool.source.get("resource_uri"))
    if configured:
        return configured
    for key in ("uri", "resource_uri", "resourceUri"):
        value = _string(arguments.get(key))
        if value:
            return value
    raise ValueError("MCP resource URI is required")


def list_resources_result(
    tool: ToolDefinition, payload: dict[str, Any]
) -> ToolInvocationResult:
    raw_resources = payload.get("resources")
    resources = raw_resources if isinstance(raw_resources, list) else []
    next_cursor = _string(payload.get("nextCursor") or payload.get("next_cursor"))
    lines = _resource_lines(resources)
    if not lines:
        text = "Remote MCP returned no resources."
    else:
        text = "Remote MCP resources:\n" + "\n".join(lines)
    if next_cursor:
        text = f"{text}\nNext cursor: {next_cursor}"
    structured: dict[str, Any] = {"tool_name": tool.name, "resources": resources}
    if next_cursor:
        structured["nextCursor"] = next_cursor
    return ToolInvocationResult(
        content=[{"type": "text", "text": text}],
        structured_content=structured,
        is_error=False,
    )


def list_resource_templates_result(
    tool: ToolDefinition, payload: dict[str, Any]
) -> ToolInvocationResult:
    raw_templates = payload.get("resourceTemplates") or payload.get("resource_templates")
    resource_templates = raw_templates if isinstance(raw_templates, list) else []
    next_cursor = _string(payload.get("nextCursor") or payload.get("next_cursor"))
    lines = _resource_template_lines(resource_templates)
    if not lines:
        text = "Remote MCP returned no resource templates."
    else:
        text = "Remote MCP resource templates:\n" + "\n".join(lines)
    if next_cursor:
        text = f"{text}\nNext cursor: {next_cursor}"
    structured: dict[str, Any] = {
        "tool_name": tool.name,
        "resourceTemplates": resource_templates,
    }
    if next_cursor:
        structured["nextCursor"] = next_cursor
    return ToolInvocationResult(
        content=[{"type": "text", "text": text}],
        structured_content=structured,
        is_error=False,
    )


def read_resource_result(
    tool: ToolDefinition, payload: dict[str, Any]
) -> ToolInvocationResult:
    raw_contents = payload.get("contents")
    contents = raw_contents if isinstance(raw_contents, list) else []
    rendered = [_resource_content_text(item) for item in contents]
    rendered = [item for item in rendered if item]
    if not rendered:
        rendered = ["Remote MCP returned no resource contents."]
    return ToolInvocationResult(
        content=[{"type": "text", "text": text} for text in rendered],
        structured_content={"tool_name": tool.name, "contents": contents},
        is_error=False,
    )


def _resource_lines(resources: list[object]) -> list[str]:
    lines: list[str] = []
    for item in resources:
        if not isinstance(item, dict):
            continue
        uri = _string(item.get("uri"))
        name = _string(item.get("title") or item.get("name")) or uri
        mime_type = _string(item.get("mimeType") or item.get("mime_type"))
        description = _string(item.get("description"))
        line = f"- {name}: {uri}" if uri else f"- {name}"
        if mime_type:
            line = f"{line} ({mime_type})"
        if description:
            line = f"{line} - {description}"
        lines.append(line)
    return lines


def _resource_template_lines(resource_templates: list[object]) -> list[str]:
    lines: list[str] = []
    for item in resource_templates:
        if not isinstance(item, dict):
            continue
        uri_template = _string(item.get("uriTemplate") or item.get("uri_template"))
        name = _string(item.get("title") or item.get("name")) or uri_template
        description = _string(item.get("description"))
        line = f"- {name}: {uri_template}" if uri_template else f"- {name}"
        if description:
            line = f"{line} - {description}"
        lines.append(line)
    return lines


def _resource_content_text(item: object) -> str:
    if not isinstance(item, dict):
        return str(item)
    uri = _string(item.get("uri"))
    mime_type = _string(item.get("mimeType") or item.get("mime_type"))
    label = "Resource"
    if uri:
        label = f"{label} {uri}"
    if mime_type:
        label = f"{label} ({mime_type})"
    if "text" in item:
        return f"{label}:\n{item.get('text')}"
    if "blob" in item:
        blob = _string(item.get("blob"))
        return f"{label}: binary content returned as base64 ({len(blob)} characters)."
    return f"{label}:\n{json.dumps(item, ensure_ascii=False)}"


def _known_resource_uris(resources: list[dict[str, Any]]) -> list[str]:
    uris: list[str] = []
    seen: set[str] = set()
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        uri = _string(resource.get("uri"))
        if not uri or uri in seen:
            continue
        seen.add(uri)
        uris.append(uri)
    return uris


def _resource_uri_description(resource_templates: list[dict[str, Any]]) -> str:
    description = "Exact MCP resource URI to read."
    summary = resource_template_summary(resource_templates)
    if not summary:
        return description
    return (
        f"{description} If the server exposes URI templates, expand a template with concrete "
        f"parameter values before calling resources/read. {summary}"
    )


def _resource_discovery_forced(server: dict[str, Any]) -> bool:
    for source in (server, server.get("_meta")):
        if not isinstance(source, dict):
            continue
        for key in ("discoverResources", "discover_resources", "resources"):
            value = source.get(key)
            if value is True:
                return True
            if isinstance(value, str) and value.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return True
    return False


def _server_display_name(server: dict[str, Any]) -> str:
    return (
        _string(server.get("name"))
        or _string(server.get("url"))
        or _string(server.get("command"))
        or "remote MCP"
    )


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""
