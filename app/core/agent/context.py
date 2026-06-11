from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# Context compaction is intentionally deterministic. It keeps the earliest
# anchor messages plus the newest working set, and replaces the middle with a
# machine-readable handoff summary so the tool loop can continue in-thread.
SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION - REFERENCE ONLY]\n"
    "Earlier middle conversation turns were compacted into the handoff summary below. "
    "Treat it as background state, not as a new user request. Continue from the latest "
    "uncompacted user message that appears after this summary."
)


@dataclass(frozen=True)
class ContextCompactionResult:
    messages: list[dict[str, Any]]
    compacted: bool
    before_chars: int
    after_chars: int
    original_message_count: int
    compacted_message_count: int
    summarized_message_count: int

    def event_payload(self) -> dict[str, Any]:
        return {
            "before_chars": self.before_chars,
            "after_chars": self.after_chars,
            "original_message_count": self.original_message_count,
            "compacted_message_count": self.compacted_message_count,
            "summarized_message_count": self.summarized_message_count,
        }


class ConversationContextManager:
    """Deterministic Hermes-like compaction for over-large chat histories."""

    def __init__(
        self,
        *,
        max_chars: int,
        keep_first_messages: int,
        keep_last_messages: int,
    ) -> None:
        self.max_chars = max_chars
        self.keep_first_messages = keep_first_messages
        self.keep_last_messages = keep_last_messages

    def compact(self, messages: list[dict[str, Any]]) -> ContextCompactionResult:
        normalized = self.normalize_for_model_prompt(messages)
        before_chars = self._serialized_chars(normalized)
        if before_chars <= self.max_chars:
            return self._unchanged(normalized, before_chars)

        if len(normalized) <= self.keep_first_messages + self.keep_last_messages + 1:
            return self._compact_short_oversized_history(normalized, before_chars)

        head_end = min(max(0, self.keep_first_messages), len(normalized))
        tail_start = max(head_end, len(normalized) - max(1, self.keep_last_messages))
        head_end = self._align_head_end(normalized, head_end)
        tail_start = self._align_tail_start(normalized, tail_start, minimum=head_end)

        if tail_start <= head_end:
            return self._unchanged(normalized, before_chars)

        # Keep the stable head and recent tail intact, then summarize only the
        # middle range so later runs can resume with enough context.
        head = normalized[:head_end]
        middle = normalized[head_end:tail_start]
        tail = normalized[tail_start:]
        summary_budget = self._summary_budget(head, tail)
        summary = self._summary_message(middle, max_chars=summary_budget)
        compacted = [*head, summary, *tail]
        after_chars = self._serialized_chars(compacted)
        return ContextCompactionResult(
            messages=compacted,
            compacted=True,
            before_chars=before_chars,
            after_chars=after_chars,
            original_message_count=len(normalized),
            compacted_message_count=len(compacted),
            summarized_message_count=len(middle),
        )

    def _compact_short_oversized_history(
        self,
        normalized: list[dict[str, Any]],
        before_chars: int,
    ) -> ContextCompactionResult:
        if len(normalized) <= 1:
            compacted = [self._summary_message(normalized, max_chars=max(800, self.max_chars // 2))]
            after_chars = self._serialized_chars(compacted)
            return ContextCompactionResult(
                messages=compacted,
                compacted=True,
                before_chars=before_chars,
                after_chars=after_chars,
                original_message_count=len(normalized),
                compacted_message_count=len(compacted),
                summarized_message_count=len(normalized),
            )

        latest_index = self._latest_user_message_index(normalized)
        if latest_index <= 0:
            latest_index = len(normalized) - 1
        summarized = normalized[:latest_index]
        tail = normalized[latest_index:]
        summary_budget = self._summary_budget([], tail)
        summary = self._summary_message(summarized, max_chars=summary_budget)
        compacted = [summary, *tail]
        after_chars = self._serialized_chars(compacted)
        return ContextCompactionResult(
            messages=compacted,
            compacted=True,
            before_chars=before_chars,
            after_chars=after_chars,
            original_message_count=len(normalized),
            compacted_message_count=len(compacted),
            summarized_message_count=len(summarized),
        )

    @staticmethod
    def _latest_user_message_index(messages: list[dict[str, Any]]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                return index
        return len(messages) - 1

    def _summary_budget(self, head: list[dict[str, Any]], tail: list[dict[str, Any]]) -> int:
        preserved_chars = self._serialized_chars([*head, *tail])
        remaining = self.max_chars - preserved_chars
        if remaining > 1000:
            return min(12000, remaining)
        return max(800, min(6000, self.max_chars // 3))

    @classmethod
    def _summary_message(cls, messages: list[dict[str, Any]], *, max_chars: int) -> dict[str, Any]:
        lines = [
            SUMMARY_PREFIX,
            "",
            "## Compacted Range",
            f"- Messages summarized: {len(messages)}",
            "",
            "## Handoff Summary",
        ]
        per_message_budget = max(160, min(900, max_chars // max(1, len(messages))))
        for index, message in enumerate(messages, start=1):
            lines.append(cls._message_summary(index, message, max_chars=per_message_budget))
        content = cls._clip_text("\n".join(lines), max_chars=max_chars)
        return {"role": "assistant", "content": content}

    @classmethod
    def _message_summary(cls, index: int, message: dict[str, Any], *, max_chars: int) -> str:
        role = str(message.get("role") or "message")
        content = cls._content_text(message.get("content"))
        tool_calls = cls._tool_calls_summary(message)
        parts = [f"{index}. {role}"]
        if tool_calls:
            parts.append(tool_calls)
        if content:
            parts.append(cls._clip_text(content.replace("\n", " "), max_chars=max_chars))
        return ": ".join(parts)

    @staticmethod
    def _tool_calls_summary(message: dict[str, Any]) -> str:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            return ""
        names: list[str] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and name:
                    names.append(name)
        if not names:
            return "tool_calls"
        return "tool_calls=" + ",".join(names)

    @classmethod
    def _content_text(cls, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return json.dumps(content, ensure_ascii=False, default=str)

    @staticmethod
    def _clip_text(text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        marker = f"...[truncated {len(text)} chars]"
        if len(marker) >= max_chars:
            return marker[:max_chars]
        keep = max(0, max_chars - len(marker))
        omitted = len(text) - keep
        marker = f"...[truncated {omitted} chars]"
        keep = max(0, max_chars - len(marker))
        return text[:keep] + marker

    @staticmethod
    def _align_head_end(messages: list[dict[str, Any]], head_end: int) -> int:
        while head_end > 0 and head_end < len(messages) and _has_tool_calls(messages[head_end - 1]):
            head_end += 1
            while head_end < len(messages) and messages[head_end].get("role") == "tool":
                head_end += 1
        return head_end

    @staticmethod
    def _align_tail_start(messages: list[dict[str, Any]], tail_start: int, *, minimum: int) -> int:
        while tail_start > minimum and messages[tail_start].get("role") == "tool":
            tail_start -= 1
        return tail_start

    @staticmethod
    def _serialized_chars(messages: list[dict[str, Any]]) -> int:
        return len(json.dumps(_messages_without_internal_fields(messages), ensure_ascii=False, default=str))

    @staticmethod
    def _unchanged(messages: list[dict[str, Any]], before_chars: int) -> ContextCompactionResult:
        return ContextCompactionResult(
            messages=messages,
            compacted=False,
            before_chars=before_chars,
            after_chars=before_chars,
            original_message_count=len(messages),
            compacted_message_count=len(messages),
            summarized_message_count=0,
        )

    @classmethod
    def normalize_for_model_prompt(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return model-safe chat history without changing valid transcripts.

        OpenAI-compatible chat completions require tool outputs to correspond to
        prior assistant tool calls. Stored history can become invalid after
        manual edits, legacy imports, failed runs, or deterministic compaction.
        This normalizer keeps valid call/output runs intact, drops orphan tool
        outputs, and inserts a short synthetic output only when a prior
        assistant tool call has no matching tool message before the next
        non-tool message.
        """

        normalized: list[dict[str, Any]] = []
        pending_tool_call_ids: list[str] = []
        for raw_message in messages:
            message = cls._normalized_message(raw_message)
            if message is None:
                continue

            role = message.get("role")
            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id in pending_tool_call_ids:
                    normalized.append(message)
                    pending_tool_call_ids.remove(tool_call_id)
                continue

            if pending_tool_call_ids:
                normalized.extend(cls._missing_tool_outputs(pending_tool_call_ids))
                pending_tool_call_ids = []

            normalized.append(message)
            if role == "assistant":
                pending_tool_call_ids = cls._tool_call_ids(message)

        if pending_tool_call_ids:
            normalized.extend(cls._missing_tool_outputs(pending_tool_call_ids))
        return normalized

    @staticmethod
    def _normalized_message(message: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return None
        normalized = dict(message)
        content = normalized.get("content")
        if content is None:
            normalized["content"] = ""
        elif not isinstance(content, str):
            normalized["content"] = ConversationContextManager._content_text(content)
        return normalized

    @staticmethod
    def _tool_call_ids(message: dict[str, Any]) -> list[str]:
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            return []
        ids: list[str] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            raw_id = raw_call.get("id")
            call_id = raw_id if isinstance(raw_id, str) and raw_id else f"call_{index}"
            ids.append(call_id)
        return ids

    @staticmethod
    def _missing_tool_outputs(tool_call_ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": "Tool call output was not available in stored conversation history.",
                            }
                        ],
                        "structuredContent": {"missing_tool_output": True},
                        "isError": True,
                    },
                    ensure_ascii=False,
                ),
            }
            for call_id in tool_call_ids
        ]


def _has_tool_calls(message: dict[str, Any]) -> bool:
    raw_calls = message.get("tool_calls")
    return isinstance(raw_calls, list) and bool(raw_calls)


def _messages_without_internal_fields(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        clean = dict(message)
        clean.pop("_attachments", None)
        sanitized.append(clean)
    return sanitized
