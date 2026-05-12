from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import acp, agents, apps, artifacts, cron, health, mcp, sandbox, skills, uploads, workflows
from app.core.runtime import RuntimeBootstrapConfig, default_container


RUNTIME_BOOTSTRAP_ENV = "JETLINKS_RUNTIME_BOOTSTRAP"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    await cron.scheduler.start()
    try:
        yield
    finally:
        await cron.scheduler.stop()


def create_app(bootstrap: RuntimeBootstrapConfig | dict[str, Any] | None = None) -> FastAPI:
    bootstrap_config = bootstrap if bootstrap is not None else _runtime_bootstrap_from_env()
    if bootstrap_config is not None:
        default_container.configure(bootstrap_config)
    app = FastAPI(title="JetLinks Agent Runtime v2", version="0.1.0", lifespan=lifespan)
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
    app.include_router(apps.router)
    app.include_router(artifacts.router)
    app.include_router(sandbox.router)
    app.include_router(cron.router)
    app.include_router(skills.router)
    app.include_router(uploads.router)
    app.include_router(workflows.router)
    app.include_router(mcp.router)

    static_dir = Path(__file__).resolve().parents[1] / "static"
    workbench_path = static_dir / "workbench.html"
    if workbench_path.is_file():
        @app.get("/static/workbench.html", include_in_schema=False)
        async def workbench_html() -> FileResponse:
            response = FileResponse(workbench_path)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

    api_docs_path = static_dir / "api-docs.html"
    if api_docs_path.is_file():
        @app.get("/static/api-docs.html", include_in_schema=False)
        async def api_docs_html() -> FileResponse:
            response = FileResponse(api_docs_path)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

    workbench_dist = Path(__file__).resolve().parents[1] / "frontend" / "workbench" / "dist"
    workbench_index = workbench_dist / "index.html"
    if workbench_index.is_file():
        @app.get("/workbench", include_in_schema=False)
        async def workbench_app() -> FileResponse:
            response = FileResponse(workbench_index)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

        app.mount("/workbench", StaticFiles(directory=workbench_dist, html=True), name="workbench")

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


def _runtime_bootstrap_from_env() -> RuntimeBootstrapConfig | None:
    raw_path = os.getenv(RUNTIME_BOOTSTRAP_ENV, "").strip()
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{RUNTIME_BOOTSTRAP_ENV} must point to a JSON object: {path}")
    return RuntimeBootstrapConfig.model_validate(data)


app = create_app()
