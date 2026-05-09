from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


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
        normalized = [dict(message) for message in messages]
        before_chars = self._serialized_chars(normalized)
        if before_chars <= self.max_chars:
            return self._unchanged(normalized, before_chars)

        if len(normalized) <= self.keep_first_messages + self.keep_last_messages + 1:
            return self._unchanged(normalized, before_chars)

        head_end = min(max(0, self.keep_first_messages), len(normalized))
        tail_start = max(head_end, len(normalized) - max(1, self.keep_last_messages))
        head_end = self._align_head_end(normalized, head_end)
        tail_start = self._align_tail_start(normalized, tail_start, minimum=head_end)

        if tail_start <= head_end:
            return self._unchanged(normalized, before_chars)

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
        return len(json.dumps(messages, ensure_ascii=False, default=str))

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


def _has_tool_calls(message: dict[str, Any]) -> bool:
    raw_calls = message.get("tool_calls")
    return isinstance(raw_calls, list) and bool(raw_calls)
