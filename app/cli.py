from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.core.agent import AgentRuntime
from app.core.config import AgentConfigLoader
from app.schemas import Attachment, ChatRequest, Message, RuntimeOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jetlinks-agent-v2")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-agents")

    show = sub.add_parser("show-agent")
    show.add_argument("agent")

    acp_stdio = sub.add_parser("acp-stdio", help="Run a standard ACP stdio agent server.")
    acp_stdio.add_argument("--agent", default="default")

    run = sub.add_parser("run")
    run.add_argument("--agent", default="default")
    run.add_argument("--message", required=True)
    run.add_argument("--thread-id")
    run.add_argument("--file", action="append", default=[])
    run.add_argument("--json", action="store_true")
    run.add_argument("--stream", action="store_true", help="Print run events as JSONL.")
    return parser


async def main_async(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    loader = AgentConfigLoader()

    if args.command == "list-agents":
        for agent in loader.list_agents():
            print(f"{agent.name}\t{agent.display_name}\t{agent.description}")
        return 0

    if args.command == "show-agent":
        print(json.dumps(loader.public_payload(loader.load(args.agent)), ensure_ascii=False, indent=2))
        return 0

    if args.command == "acp-stdio":
        from app.protocols.acp import run_acp_stdio_agent

        await run_acp_stdio_agent(agent_name=args.agent)
        return 0

    if args.command == "run":
        agent = loader.load(args.agent)
        attachments = [_attachment_from_path(Path(path)) for path in args.file]
        request = ChatRequest(
            messages=[Message(role="user", content=args.message)],
            attachments=attachments,
            runtime_options=RuntimeOptions(thread_id=args.thread_id),
        )
        runtime = AgentRuntime()
        if args.stream:
            result, events = await runtime.run_with_events(agent, request)
            for event in events:
                print(json.dumps(event.model_dump(), ensure_ascii=False))
            if not events or events[-1].type not in {"run.completed", "run.failed"}:
                print(json.dumps({"type": "run.completed", "data": {"result": result.model_dump()}}, ensure_ascii=False))
            return 0 if result.status == "completed" else 1

        result = await runtime.run(agent, request)
        if args.json:
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        else:
            print(result.reply)
            for artifact in result.artifacts:
                print(f"- {artifact.path} ({artifact.mime_type})")
        return 0

    return 2


def _attachment_from_path(path: Path) -> Attachment:
    return Attachment(name=path.name, path=str(path.resolve()), mime_type=None)


def main() -> None:
    raise SystemExit(asyncio.run(main_async(sys.argv[1:])))


if __name__ == "__main__":
    main()
