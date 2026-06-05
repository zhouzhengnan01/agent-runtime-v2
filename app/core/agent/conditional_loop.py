from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Literal

from app.schemas import RuntimeOptions


ContinueWhen = Literal["contains", "not_contains"]
ConditionOperator = Literal[
    "exists",
    "not_exists",
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "not_contains",
    "is_empty",
    "is_not_empty",
]


@dataclass(frozen=True)
class StructuredCondition:
    json_path: str
    operator: ConditionOperator = "eq"
    expected: Any = None

    def matches(self, latest_tool_text: str) -> bool:
        payload = _json_payload(latest_tool_text)
        if payload is _MISSING:
            return False
        value = _resolve_json_path(payload, self.json_path)
        return _compare(value, self.operator, self.expected)

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "json_path": self.json_path,
            "operator": self.operator,
        }
        if self.operator not in {"exists", "not_exists", "is_empty", "is_not_empty"}:
            payload["expected"] = self.expected
        return payload


@dataclass(frozen=True)
class ConditionalToolLoopPolicy:
    """Runtime policy for poll-until style tool loops.

    `contains` means continue while the latest tool result contains `marker`.
    `not_contains` means continue while the latest tool result does not contain
    `marker`.
    """

    marker: str
    continue_when: ContinueWhen = "contains"
    delay_seconds: float = 0.0
    source: str = "natural_language"
    condition: StructuredCondition | None = None
    continue_on_condition: bool = True
    auto_repeat_tool_call: bool = False
    repeat_tool_name: str | None = None

    def should_continue(self, latest_tool_text: str) -> bool:
        if self.condition is not None:
            matched = self.condition.matches(latest_tool_text)
            return matched if self.continue_on_condition else not matched
        if not self.marker:
            return False
        contains = self.marker in latest_tool_text
        if self.continue_when == "contains":
            return contains
        return not contains

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "marker": self.marker,
            "continue_when": self.continue_when,
            "delay_seconds": self.delay_seconds,
            "source": self.source,
            "auto_repeat_tool_call": self.auto_repeat_tool_call,
        }
        if self.repeat_tool_name:
            payload["repeat_tool_name"] = self.repeat_tool_name
        if self.condition is not None:
            payload["condition"] = self.condition.to_metadata()
            payload["continue_on_condition"] = self.continue_on_condition
        return payload


def conditional_loop_policy(
    messages: list[dict[str, Any]],
    runtime_options: RuntimeOptions | None = None,
) -> ConditionalToolLoopPolicy | None:
    configured = _policy_from_runtime_options(runtime_options)
    if configured is not None:
        return configured
    return _policy_from_user_text(latest_user_message_text(messages))


def latest_tool_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") != "tool":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return _flatten_tool_content(content)
        return str(content or "")
    return ""


def latest_user_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == "user":
            return str(message.get("content") or "")
    return ""


def _policy_from_runtime_options(runtime_options: RuntimeOptions | None) -> ConditionalToolLoopPolicy | None:
    if runtime_options is None:
        return None
    raw_policy = runtime_options.config_options.get("conditional_tool_loop")
    if not isinstance(raw_policy, dict):
        raw_policy = runtime_options.config_options.get("conditionalToolLoop")
    if not isinstance(raw_policy, dict):
        return None
    structured = _structured_policy_from_runtime_options(raw_policy)
    if structured is not None:
        return structured
    marker = _string(
        raw_policy.get("marker")
        or raw_policy.get("target")
        or raw_policy.get("contains")
        or raw_policy.get("text")
    )
    if marker is None:
        return None
    continue_when = str(raw_policy.get("continue_when") or raw_policy.get("continueWhen") or "contains")
    if continue_when not in {"contains", "not_contains"}:
        continue_when = "contains"
    delay_seconds = _delay_from_value(
        raw_policy.get("delay_seconds")
        or raw_policy.get("delaySeconds")
        or raw_policy.get("poll_interval_seconds")
        or raw_policy.get("pollIntervalSeconds")
    )
    return ConditionalToolLoopPolicy(
        marker=marker,
        continue_when=continue_when,  # type: ignore[arg-type]
        delay_seconds=delay_seconds,
        source="runtime_options",
        auto_repeat_tool_call=_auto_repeat_from_runtime_policy(raw_policy),
        repeat_tool_name=_string(raw_policy.get("repeat_tool_name") or raw_policy.get("repeatToolName")),
    )


def _structured_policy_from_runtime_options(raw_policy: dict[str, Any]) -> ConditionalToolLoopPolicy | None:
    condition_payload = _dict_value(
        raw_policy.get("retry_when")
        or raw_policy.get("retryWhen")
        or raw_policy.get("continue_while")
        or raw_policy.get("continueWhile")
    )
    continue_on_condition = True
    if condition_payload is None:
        condition_payload = _dict_value(raw_policy.get("stop_when") or raw_policy.get("stopWhen"))
        continue_on_condition = False
    if condition_payload is None and any(key in raw_policy for key in ("json_path", "jsonPath", "path")):
        condition_payload = raw_policy
        continue_on_condition = True
    if condition_payload is None:
        return None

    json_path = _string(
        condition_payload.get("json_path")
        or condition_payload.get("jsonPath")
        or condition_payload.get("path")
    )
    if json_path is None:
        return None
    condition = StructuredCondition(
        json_path=json_path,
        operator=_condition_operator(condition_payload.get("operator") or condition_payload.get("op")),
        expected=condition_payload.get("expected", condition_payload.get("value")),
    )
    delay_seconds = _delay_from_value(
        raw_policy.get("delay_seconds")
        or raw_policy.get("delaySeconds")
        or raw_policy.get("poll_interval_seconds")
        or raw_policy.get("pollIntervalSeconds")
    )
    marker = _string(raw_policy.get("marker") or raw_policy.get("target")) or json_path
    return ConditionalToolLoopPolicy(
        marker=marker,
        delay_seconds=delay_seconds,
        source="runtime_options",
        condition=condition,
        continue_on_condition=continue_on_condition,
        auto_repeat_tool_call=_auto_repeat_from_runtime_policy(raw_policy),
        repeat_tool_name=_string(raw_policy.get("repeat_tool_name") or raw_policy.get("repeatToolName")),
    )


def _policy_from_user_text(text: str) -> ConditionalToolLoopPolicy | None:
    if not text:
        return None
    absent_marker = _absent_stop_marker(text)
    if absent_marker:
        return ConditionalToolLoopPolicy(
            marker=absent_marker,
            continue_when="contains",
            delay_seconds=_delay_from_text(text),
            auto_repeat_tool_call=_auto_repeat_from_text(text),
        )
    present_marker = _present_stop_marker(text)
    if present_marker:
        return ConditionalToolLoopPolicy(
            marker=present_marker,
            continue_when="not_contains",
            delay_seconds=_delay_from_text(text),
            auto_repeat_tool_call=_auto_repeat_from_text(text),
        )
    return None


def _absent_stop_marker(text: str) -> str:
    patterns = (
        r"(?:直到|直至|等到)[^。；;\n]*?没有(?P<target>[^，。；;\n]+?)(?:则|就)?(?:停止|为止)",
        r"(?:直到|直至|等到)[^。；;\n]*?不再(?:出现|包含|返回)?(?P<target>[^，。；;\n]+?)(?:则|就)?(?:停止|为止)",
        r"until[^.\n]*?\bno\s+(?P<target>[^,.;\n]+?)(?:\s+then)?\s+(?:stop|finish)",
        r"repeat[^.\n]*?\buntil\s+(?P<target>[^,.;\n]+?)\s+(?:is|are)?\s*(?:gone|absent|missing)",
    )
    marker = _first_clean_marker(patterns, text)
    if marker:
        return marker
    if _has_absent_stop_without_marker(text):
        return _query_target_marker(text)
    return ""


def _present_stop_marker(text: str) -> str:
    patterns = (
        r"直到[^。；;\n]*?(?:出现|包含|返回|有)(?P<target>[^，。；;\n]+?)(?:则|就)?停止",
        r"直到[^。；;\n]*?(?:状态|结果|字段|值)(?:为|是|等于)(?P<target>[^，。；;\n]+?)(?:则|就)?停止",
        r"until[^.\n]*?\b(?:contains|returns|has|status is|status equals)\s+(?P<target>[^,.;\n]+?)(?:\s+then)?\s+stop",
    )
    return _first_clean_marker(patterns, text)


def _first_clean_marker(patterns: tuple[str, ...], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        marker = _clean_marker(match.group("target"))
        if marker:
            return marker
    return ""


def _has_absent_stop_without_marker(text: str) -> bool:
    return bool(
        re.search(
            r"(?:直到|直至|等到)[^。；;\n]*(?:没有|不存在|为空|清空|查不到)[^。；;\n]*(?:停止|为止|结束)?",
            text,
            flags=re.IGNORECASE,
        )
    )


def _query_target_marker(text: str) -> str:
    patterns = (
        r"(?:查|查询|搜索|获取|检查)\s*(?P<target>[^，。；;\n]+?)(?=用户|信息|记录|数据|列表|，|,|。|；|;|如果|若|有|还有|就|并|然后|$)",
        r"(?:query|search|find|get|check)\s+(?P<target>[^,.;\n]+?)(?=\s+(?:until|if|when|then|and)|[,.;\n]|$)",
    )
    return _first_clean_marker(patterns, text)


def _delay_from_text(text: str) -> float:
    patterns = (
        r"(?:等|等待|间隔)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>毫秒|ms|秒|s|分钟|分|min|m)",
        r"(?:sleep|wait)\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|m)?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        return _delay_from_value(match.group("value"), unit=match.group("unit") or "秒")
    return 0.0


def _delay_from_value(value: object, *, unit: object = "秒") -> float:
    try:
        seconds = float(str(value))
    except (TypeError, ValueError):
        return 0.0
    normalized_unit = str(unit or "秒").lower()
    if normalized_unit in {"毫秒", "ms", "millisecond", "milliseconds"}:
        seconds = seconds / 1000
    elif normalized_unit in {"分钟", "分", "min", "mins", "minute", "minutes", "m"}:
        seconds = seconds * 60
    return max(0.0, min(seconds, 60.0))


def _auto_repeat_from_text(text: str) -> bool:
    return _auto_repeat_enabled_by_env() and bool(
        re.search(
            r"(?:循环|轮询|自动|反复|重复|一直|每隔|每次|继续|直到|直至|等到|不要确认|不用确认|无需确认|until|repeat|poll)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _auto_repeat_from_runtime_policy(raw_policy: dict[str, Any]) -> bool:
    del raw_policy
    return _auto_repeat_enabled_by_env()


def _auto_repeat_enabled_by_env() -> bool:
    return _env_bool("CONDITIONAL_TOOL_LOOP_AUTO_REPEAT_DEFAULT", default=True)


def _flatten_tool_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    return json.dumps(parsed, ensure_ascii=False, default=str)


_MISSING = object()


def _json_payload(content: str) -> object:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return _MISSING


def _resolve_json_path(payload: object, json_path: str) -> object:
    normalized = json_path.strip()
    if not normalized:
        return _MISSING
    if normalized.startswith("$."):
        normalized = normalized[2:]
    elif normalized == "$":
        return payload
    candidates = [payload]
    if isinstance(payload, dict):
        for key in ("structuredContent", "structured_content", "result", "data"):
            if key in payload:
                candidates.append(payload[key])
    for candidate in candidates:
        value = _resolve_path_from_root(candidate, normalized)
        if value is not _MISSING:
            return value
    return _MISSING


def _resolve_path_from_root(root: object, path: str) -> object:
    current = root
    for part in _path_parts(path):
        if part == "length":
            if isinstance(current, (list, tuple, dict, str)):
                current = len(current)
                continue
            return _MISSING
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING
            continue
        return _MISSING
    return current


def _path_parts(path: str) -> list[str]:
    parts: list[str] = []
    for item in path.split("."):
        item = item.strip()
        if not item:
            continue
        while "[" in item and item.endswith("]"):
            prefix, _, suffix = item.partition("[")
            if prefix:
                parts.append(prefix)
            parts.append(suffix[:-1])
            break
        else:
            parts.append(item)
    return parts


def _compare(value: object, op: ConditionOperator, expected: object) -> bool:
    if op == "exists":
        return value is not _MISSING
    if op == "not_exists":
        return value is _MISSING
    if value is _MISSING:
        return False
    if op == "eq":
        return value == _coerce_expected(expected, value)
    if op == "ne":
        return value != _coerce_expected(expected, value)
    if op in {"gt", "gte", "lt", "lte"}:
        left = _number(value)
        right = _number(expected)
        if left is None or right is None:
            return False
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right
    if op == "contains":
        return _contains(value, expected)
    if op == "not_contains":
        return not _contains(value, expected)
    if op == "is_empty":
        return _is_empty(value)
    if op == "is_not_empty":
        return not _is_empty(value)
    return False


def _contains(value: object, expected: object) -> bool:
    if isinstance(value, dict):
        return expected in value or str(expected) in value
    if isinstance(value, list | tuple | set):
        return expected in value or str(expected) in {str(item) for item in value}
    return str(expected) in str(value)


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, str)):
        return len(value) == 0
    return False


def _coerce_expected(expected: object, actual: object) -> object:
    if isinstance(actual, bool) and isinstance(expected, str):
        lowered = expected.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(actual, int) and not isinstance(actual, bool):
        try:
            return int(str(expected))
        except (TypeError, ValueError):
            return expected
    if isinstance(actual, float):
        try:
            return float(str(expected))
        except (TypeError, ValueError):
            return expected
    return expected


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clean_marker(value: str) -> str:
    marker = value.strip(" 的：:，。,.；; \t\r\n")
    marker = re.sub(r"(查询)?返回结果(里面|中)?(的)?", "", marker)
    marker = re.sub(r"(结果|列表|记录|数据)(里面|中)?(的)?", "", marker)
    marker = re.sub(r"(用户|信息|记录|数据|项)$", "", marker)
    marker = marker.strip(" 的：:，。,.；; \t\r\n")
    if len(marker) > 80:
        return ""
    return marker


def _string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _dict_value(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto", "enabled"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = os.getenv(f"JETLINKS_{name}")
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on", "auto", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _condition_operator(value: object) -> ConditionOperator:
    raw = str(value or "eq").strip().lower()
    aliases = {
        "==": "eq",
        "=": "eq",
        "equals": "eq",
        "!=": "ne",
        "<>": "ne",
        "not_equals": "ne",
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
        "empty": "is_empty",
        "not_empty": "is_not_empty",
        "present": "exists",
        "missing": "not_exists",
    }
    normalized = aliases.get(raw, raw)
    allowed = {
        "exists",
        "not_exists",
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "not_contains",
        "is_empty",
        "is_not_empty",
    }
    if normalized in allowed:
        return normalized  # type: ignore[return-value]
    return "eq"
