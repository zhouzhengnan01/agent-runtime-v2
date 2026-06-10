from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.artifacts.store import ThreadPaths
from app.schemas import Message


ChatHistoryMessage = Message | dict[str, Any]
_LOCAL_PLACEHOLDER_PREFIX = "v2 无状态 Agent 已收到请求。当前"
_LOCAL_PLACEHOLDER_SUFFIX = "因此返回本地占位响应。"


@dataclass(frozen=True)
class SessionConversationStore:
    """Persist recoverable conversation history inside a thread workspace.

    The runtime treats the thread workspace as the durable source of truth for
    multi-turn execution. Incoming messages are merged with stored history so a
    resumed run can continue without replaying duplicated turns.
    """

    max_history_messages: int = 200

    def history_path(self, paths: ThreadPaths) -> Path:
        return paths.memory / "conversation.jsonl"

    def transcript_path(self, paths: ThreadPaths) -> Path:
        return paths.memory / "conversation.md"

    def load(self, paths: ThreadPaths) -> list[dict[str, Any]]:
        history_file = paths.memory / "conversation.jsonl"
        if not history_file.is_file():
            return []

        messages: list[dict[str, Any]] = []
        for line in history_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            message = item.get("message")
            if isinstance(message, dict) and self._valid_message(message):
                messages.append(self._normalize_message(message))
        sanitized = self._sanitize_history(messages)
        return sanitized[-self.max_history_messages :]

    def merge(
        self,
        stored_messages: list[dict[str, Any]],
        incoming_messages: Sequence[ChatHistoryMessage],
    ) -> list[dict[str, Any]]:
        stored = self._sanitize_history(stored_messages)
        incoming = self._sanitize_history([self._normalize_message(message) for message in incoming_messages])
        if not stored:
            return incoming
        if not incoming:
            return stored[-self.max_history_messages :]

        # Request payloads often resend a suffix of the thread history. Detect
        # that overlap so we extend the conversation without duplicating turns.
        overlap = self._suffix_prefix_overlap(stored, incoming)
        merged = [*stored, *incoming[overlap:]]
        return merged[-self.max_history_messages :]

    def save(self, paths: ThreadPaths, messages: list[dict[str, Any]], *, run_id: str = "") -> None:
        normalized = [self._normalize_message(message) for message in messages if self._valid_message(message)]
        normalized = self._sanitize_history(normalized)
        normalized = normalized[-self.max_history_messages :]
        paths.memory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        history_file = paths.memory / "conversation.jsonl"
        lines = [
            json.dumps(
                {
                    "timestamp": timestamp,
                    "run_id": run_id,
                    "message": message,
                },
                ensure_ascii=False,
            )
            for message in normalized
        ]
        self._write_text_file(history_file, "\n".join(lines) + ("\n" if lines else ""))

        transcript_file = paths.memory / "conversation.md"
        self._write_text_file(transcript_file, self._transcript(normalized))

    @staticmethod
    def _write_text_file(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        tmp_file.write_text(content, encoding="utf-8")
        last_error: PermissionError | None = None
        for attempt in range(5):
            try:
                tmp_file.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        try:
            path.write_text(content, encoding="utf-8")
        except PermissionError:
            if last_error is not None:
                raise last_error
            raise
        finally:
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass

    @classmethod
    def _normalize_message(cls, message: ChatHistoryMessage) -> dict[str, Any]:
        raw = message.model_dump(mode="python") if isinstance(message, Message) else dict(message)
        normalized: dict[str, Any] = {}
        role = raw.get("role")
        normalized["role"] = role if isinstance(role, str) else "user"
        content = raw.get("content")
        normalized["content"] = content if isinstance(content, str) else ""
        if isinstance(raw.get("tool_call_id"), str):
            normalized["tool_call_id"] = raw["tool_call_id"]
        if isinstance(raw.get("tool_calls"), list):
            normalized["tool_calls"] = raw["tool_calls"]
        return normalized

    @staticmethod
    def _valid_message(message: dict[str, Any]) -> bool:
        return message.get("role") in {"user", "assistant", "tool", "system"}

    @classmethod
    def _suffix_prefix_overlap(cls, stored_messages: list[dict[str, Any]], incoming_messages: list[dict[str, Any]]) -> int:
        max_overlap = min(len(stored_messages), len(incoming_messages))
        for size in range(max_overlap, 0, -1):
            if stored_messages[-size:] == incoming_messages[:size]:
                return size
        return 0

    @classmethod
    def _sanitize_history(cls, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for message in messages:
            if cls._is_local_placeholder_message(message):
                while sanitized and sanitized[-1].get("role") == "user":
                    sanitized.pop()
                continue
            sanitized.append(message)
        return sanitized

    @staticmethod
    def _is_local_placeholder_message(message: dict[str, Any]) -> bool:
        if message.get("role") != "assistant":
            return False
        content = message.get("content")
        if not isinstance(content, str):
            return False
        return content.startswith(_LOCAL_PLACEHOLDER_PREFIX) and _LOCAL_PLACEHOLDER_SUFFIX in content

    @staticmethod
    def _transcript(messages: list[dict[str, Any]]) -> str:
        parts = ["# Conversation", ""]
        for message in messages:
            role = str(message.get("role") or "unknown")
            content = str(message.get("content") or "")
            parts.extend([f"## {role}", "", content, ""])
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                parts.extend(["```json", json.dumps(tool_calls, ensure_ascii=False, indent=2), "```", ""])
        return "\n".join(parts).rstrip() + "\n"
