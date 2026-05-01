from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.agent import AgentRuntime
from app.core.config import AgentConfigLoader
from app.schemas import AgentRunResult, ChatRequest


router = APIRouter(prefix="/api/agents", tags=["agents"])
loader = AgentConfigLoader()
runtime = AgentRuntime()


@router.get("")
async def list_agents() -> dict[str, list[dict[str, str]]]:
    return {
        "agents": [
            {
                "name": agent.name,
                "display_name": agent.display_name,
                "description": agent.description,
            }
            for agent in loader.list_agents()
        ]
    }


@router.get("/{agent_name}")
async def get_agent(agent_name: str) -> dict[str, object]:
    try:
        agent = loader.load(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return loader.public_payload(agent)


@router.post("/{agent_name}/runs")
async def run_agent(agent_name: str, request: ChatRequest) -> AgentRunResult:
    try:
        agent = loader.load(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return await runtime.run(agent, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{agent_name}/runs/stream")
async def stream_agent(agent_name: str, request: ChatRequest) -> StreamingResponse:
    try:
        agent = loader.load(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(runtime.stream(agent, request), media_type="text/event-stream")
