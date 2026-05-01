from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import acp, agents, artifacts, health, mcp, sandbox, skills, workflows


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
    app.include_router(workflows.router)
    app.include_router(mcp.router)

    static_dir = Path(__file__).resolve().parents[1] / "static"

    @app.get("/", include_in_schema=False)
    async def workbench_index() -> HTMLResponse:
        workbench_path = static_dir / "workbench.html"
        return HTMLResponse(
            workbench_path.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
