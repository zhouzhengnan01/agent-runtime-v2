from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import acp, acp_http_stream, agents, apps, artifacts, cron, health, mcp, sandbox, skills, training, uploads, workflows
from app.core.runtime import RuntimeBootstrapConfig, default_container


logger = logging.getLogger("uvicorn.error")


def _configure_logging() -> None:
    """Use one timestamped formatter for uvicorn and application logs."""
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(logger_name)
        for handler in target.handlers:
            handler.setFormatter(formatter)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "runtime startup begin title=%s version=%s pid=%s python=%s executable=%s cwd=%s",
        app.title,
        app.version,
        os.getpid(),
        sys.version.split()[0],
        sys.executable,
        Path.cwd(),
    )
    await cron.scheduler.start()
    logger.info("cron scheduler started")
    try:
        yield
    finally:
        logger.info("runtime shutdown begin")
        await cron.scheduler.stop()
        logger.info("cron scheduler stopped")


def _bootstrap_from_env() -> dict[str, Any] | None:
    path = os.getenv("JETLINKS_RUNTIME_BOOTSTRAP")
    if not path:
        logger.info("runtime bootstrap skipped reason=env_not_set")
        return None
    bootstrap_path = Path(path)
    if not bootstrap_path.is_file():
        logger.warning("runtime bootstrap skipped reason=file_missing path=%s", bootstrap_path)
        return None
    logger.info("runtime bootstrap loading path=%s", bootstrap_path)
    data = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def create_app(bootstrap: RuntimeBootstrapConfig | dict[str, Any] | None = None) -> FastAPI:
    runtime_bootstrap = bootstrap if bootstrap is not None else _bootstrap_from_env()
    if runtime_bootstrap is not None:
        default_container.configure(runtime_bootstrap)
        logger.info("runtime bootstrap applied source=%s", "argument" if bootstrap is not None else "environment")
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
    app.include_router(acp_http_stream.router)
    app.include_router(agents.router)
    app.include_router(apps.router)
    app.include_router(artifacts.router)
    app.include_router(sandbox.router)
    app.include_router(cron.router)
    app.include_router(skills.router)
    app.include_router(training.router)
    app.include_router(uploads.router)
    app.include_router(workflows.router)
    app.include_router(mcp.router)
    logger.info("runtime routers registered count=%s", len(app.routes))

    static_dir = Path(__file__).resolve().parents[1] / "static"
    static_workbench_path = static_dir / "workbench.html"
    if static_workbench_path.is_file():
        @app.get("/static/workbench.html", include_in_schema=False)
        async def workbench_html() -> FileResponse:
            response = FileResponse(static_workbench_path)
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
    elif static_workbench_path.is_file():
        @app.get("/workbench", include_in_schema=False)
        async def legacy_workbench_app() -> FileResponse:
            response = FileResponse(static_workbench_path)
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info("static files mounted path=%s", static_dir)
    logger.info("runtime app created routes=%s", len(app.routes))
    return app


_configure_logging()
app = create_app()
