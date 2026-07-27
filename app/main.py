"""FastAPI application factory and process entry point.

The application loads the similarity engine and embedding model exactly once, in
the lifespan handler, and stores the ready service on ``app.state``. If loading
fails (for example, a missing index), the app still starts so that ``/`` and
``/health`` remain reachable and can report the problem — index-dependent
endpoints then return ``503`` until the index is available.

Run locally with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from engine.logging_utils import configure_logging

from .config import Settings, get_settings
from .models import ErrorResponse
from .routes import router
from .services import SimilarityService

logger = logging.getLogger("app.main")


def _load_service(app: FastAPI, settings: Settings) -> None:
    """Build the similarity service, recording any failure on ``app.state``."""
    app.state.service = None
    app.state.startup_error = None
    try:
        app.state.service = SimilarityService.from_settings(settings)
    except FileNotFoundError as exc:
        app.state.startup_error = str(exc)
        logger.error("Startup: index unavailable - %s", exc)
    except Exception as exc:  # noqa: BLE001 - never let startup crash the process
        app.state.startup_error = f"Failed to initialise engine: {exc}"
        logger.exception("Startup: unexpected engine initialisation failure")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings override (used by tests). Defaults to the
            environment-derived singleton.

    Returns:
        A configured :class:`FastAPI` instance.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
#1
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting %s v%s", settings.api_title, settings.api_version)
        _load_service(app, settings)
        yield
        logger.info("Shutting down %s", settings.api_title)

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="REST API for visual image similarity search over a WikiArt index.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(detail=str(exc.errors())).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(detail="Internal server error.").model_dump(),
        )

    app.include_router(router)
    return app


app = create_app()
