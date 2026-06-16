from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.auth import require_runtime_token
from app.core.runtime import default_container
from app.core.runtime.health_state import ReviewSlot
from app.schemas import AgentRunResult, ChatRequest


router = APIRouter(prefix="/api/agents", tags=["agents"])
loader = default_container.loader
runtime = default_container.runtime


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


@router.get("/runs/recent", dependencies=[Depends(require_runtime_token)])
async def list_recent_runs(limit: int = 50) -> dict[str, object]:
    return {"runs": runtime.run_event_store.list(limit=limit)}


@router.get("/runs/{run_id}", dependencies=[Depends(require_runtime_token)])
async def get_run(run_id: str) -> dict[str, object]:
    try:
        return runtime.run_event_store.get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/runs/{run_id}/events", dependencies=[Depends(require_runtime_token)])
async def get_run_events(run_id: str) -> dict[str, object]:
    try:
        payload = runtime.run_event_store.get(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "run_id": payload.get("run_id", run_id),
        "agent": payload.get("agent", ""),
        "thread_id": payload.get("thread_id", ""),
        "events": payload.get("events", []),
    }


@router.get("/runs/{run_id}/debug-bundle", dependencies=[Depends(require_runtime_token)])
async def get_run_debug_bundle(run_id: str) -> dict[str, object]:
    try:
        return runtime.run_event_store.debug_bundle(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{agent_name}/runs", dependencies=[Depends(require_runtime_token)])
async def run_agent(agent_name: str, request: ChatRequest) -> AgentRunResult:
    try:
        agent = loader.load(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        async with ReviewSlot():
            return await runtime.run(agent, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _stream_with_review_slot(agent, request: ChatRequest):
    async with ReviewSlot():
        async for event in runtime.stream(agent, request):
            yield event


@router.post("/{agent_name}/runs/stream", dependencies=[Depends(require_runtime_token)])
async def stream_agent(agent_name: str, request: ChatRequest) -> StreamingResponse:
    try:
        agent = loader.load(agent_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(_stream_with_review_slot(agent, request), media_type="text/event-stream")
