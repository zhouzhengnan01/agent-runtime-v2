from __future__ import annotations

from typing import Any

from app.core.apps.models import AppTemplate
from app.core.skills.aliases import expand_skill_aliases
from app.schemas import RuntimeOptions


def merge_runtime_options_with_template(
    runtime_options: RuntimeOptions,
    template: AppTemplate,
) -> RuntimeOptions:
    """Apply an app template onto request runtime options with request fields taking precedence."""

    template_options = template.runtime_options if isinstance(template.runtime_options, dict) else {}
    merged: dict[str, Any] = dict(template_options)
    explicit_fields = set(runtime_options.model_fields_set)
    current = runtime_options.model_dump(mode="python")
    for key, value in current.items():
        if key not in explicit_fields and key not in {"app_template_name"}:
            continue
        if key == "skill_parameters":
            merged[key] = _merge_skill_parameters(template_options.get(key), value, request_explicit=True)
            continue
        if key == "config_options":
            merged[key] = _merge_config_options(template_options.get(key), value)
            continue
        if key in {"selected_skills", "selected_mcp_tools"}:
            merged[key] = _merge_string_lists(template_options.get(key), value, request_explicit=True)
            continue
        if key == "workflow":
            merged[key] = value or template.workflow
            continue
        if key == "app_template_name":
            merged[key] = template.name
            continue
        if value is None:
            merged.setdefault(key, None)
            continue
        merged[key] = value

    merged["app_template_name"] = template.name
    merged["workflow"] = merged.get("workflow") or template.workflow
    merged["selected_skills"] = expand_skill_aliases(_merge_string_lists(template.selected_skills, merged.get("selected_skills")))
    merged["selected_mcp_tools"] = _merge_string_lists(template.selected_mcp_tools, merged.get("selected_mcp_tools"))
    merged["skill_parameters"] = _merge_skill_parameters(template_options.get("skill_parameters"), merged.get("skill_parameters"))
    merged["config_options"] = _merge_config_options(template_options.get("config_options"), merged.get("config_options"))
    return RuntimeOptions.model_validate(merged)


def _merge_string_lists(template_value: Any, request_value: Any, *, request_explicit: bool = False) -> list[str]:
    if isinstance(request_value, list):
        cleaned = [str(item).strip() for item in request_value if str(item).strip()]
        if cleaned or request_explicit:
            return cleaned
    if isinstance(template_value, list):
        return [str(item).strip() for item in template_value if str(item).strip()]
    return []


def _merge_skill_parameters(
    template_value: Any,
    request_value: Any,
    *,
    request_explicit: bool = False,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if isinstance(template_value, dict):
        for key, value in template_value.items():
            if isinstance(value, dict):
                merged[str(key)] = dict(value)
    if isinstance(request_value, dict):
        if request_explicit and not request_value:
            return {}
        for key, value in request_value.items():
            if not isinstance(value, dict):
                continue
            normalized_key = str(key)
            merged[normalized_key] = {**merged.get(normalized_key, {}), **value}
    return merged


def _merge_config_options(template_value: Any, request_value: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(template_value, dict):
        merged.update(template_value)
    if isinstance(request_value, dict):
        merged.update(request_value)
    return merged
