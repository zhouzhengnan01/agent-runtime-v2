from __future__ import annotations

import json

import pytest

from app import cli
from app.schemas import AgentRunResult, ChatEvent


def test_cli_run_stream_prints_jsonl_events(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    class FakeRuntime:
        async def run_with_events(self, agent_config, request):
            del agent_config
            assert request.messages[0].content == "hello"
            result = AgentRunResult(agent="default", thread_id=request.runtime_options.thread_id or "cli-thread", reply="ok")
            events = [
                ChatEvent(type="run.started", data={"run_id": "run-cli"}),
                ChatEvent(type="agent.message.delta", data={"text": "ok"}),
                ChatEvent(type="run.completed", data={"result": result.model_dump()}),
            ]
            return result, events

    monkeypatch.setattr(cli, "AgentRuntime", FakeRuntime)

    import asyncio

    assert asyncio.run(cli.main_async(["run", "--agent", "default", "--message", "hello", "--stream"])) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["type"] for line in lines] == ["run.started", "agent.message.delta", "run.completed"]
    assert lines[-1]["data"]["result"]["reply"] == "ok"
