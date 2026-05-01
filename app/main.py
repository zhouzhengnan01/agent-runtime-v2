from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import acp, agents, artifacts, health, mcp, sandbox, skills


def create_app() -> FastAPI:
    app = FastAPI(title="JetLinks Agent Runtime v2", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(acp.router)
    app.include_router(agents.router)
    app.include_router(artifacts.router)
    app.include_router(sandbox.router)
    app.include_router(skills.router)
    app.include_router(mcp.router)

    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
