from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, cast


def test_acp_stdio_initialize_new_session_and_prompt() -> None:
    asyncio.run(_run_acp_stdio_flow())


async def _run_acp_stdio_flow() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "app.cli",
        "acp-stdio",
        "--agent",
        "behavior-detector",
        cwd=project_root,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": 1},
            },
        )
        initialized = await _read(process)
        assert initialized["id"] == 1
        assert initialized["result"]["agentInfo"]["name"] == "jetlinks-agent-runtime-v2"

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": str(project_root), "mcpServers": []},
            },
        )
        created = await _read(process)
        session_id = created["result"]["sessionId"]
        assert session_id.startswith("acp-session-")

        await _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "人员翻越围栏进入禁区"}],
                    "_meta": {"workflow": "evidence_first_detection"},
                },
            },
        )

        notifications: list[dict[str, Any]] = []
        final: dict[str, Any] | None = None
        for _ in range(30):
            packet = await _read(process)
            if packet.get("method") == "session/update":
                notifications.append(packet)
                continue
            if packet.get("id") == 3:
                final = packet
                break

        assert final is not None
        assert final["result"]["stopReason"] == "end_turn"
        assert any(
            "待上传" in notification["params"]["update"]["content"]["text"]
            for notification in notifications
            if notification["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
        )
    finally:
        if process.stdin is not None:
            process.stdin.close()
            await process.stdin.wait_closed()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.terminate()
            await process.wait()


async def _send(process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
    await process.stdin.drain()


async def _read(process: asyncio.subprocess.Process) -> dict[str, Any]:
    assert process.stdout is not None
    line = await asyncio.wait_for(process.stdout.readline(), timeout=5)
    if not line:
        stderr = b""
        if process.stderr is not None:
            stderr = await process.stderr.read()
        raise AssertionError(f"ACP stdio process closed before response. stderr={stderr.decode(errors='replace')}")
    return cast(dict[str, Any], json.loads(line))
