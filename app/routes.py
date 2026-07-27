"""HTTP routes.

Thin, declarative handlers: each validates input, delegates the real work to
:mod:`app.utils` (validation/storage) and :class:`app.services.SimilarityService`
(encode + search), and maps typed failures onto meaningful HTTP status codes.
The loaded service is provided by dependency injection so handlers hold no
global state and are trivial to test with an overridden dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .config import Settings
from .models import (
    EndpointInfo,
    HealthResponse,
    Match,
    RootResponse,
    SearchResponse,
    UploadResponse,
)
from .services import InvalidQueryImageError, SimilarityService
from .utils import (
    CorruptImageError,
    EmptyUploadError,
    UnsupportedMediaTypeError,
    UploadError,
    UploadTooLargeError,
    check_media_type,
    find_stored_upload,
    inspect_image,
    new_upload_id,
    read_capped,
    store_upload,
    temporary_image,
)

logger = logging.getLogger("app.routes")

router = APIRouter()


# --------------------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------------------
def get_settings(request: Request) -> Settings:
    """Return the settings bound to this application instance.

    Reading from ``app.state`` (rather than the global singleton) keeps each
    app — including those built with test-specific settings — self-contained.
    """
    return request.app.state.settings


def get_service(request: Request) -> SimilarityService:
    """Return the loaded similarity service, or 503 if it is unavailable.

    The service is built once at startup and stored on ``app.state``. Endpoints
    that require the index depend on this; ``/`` and ``/health`` deliberately do
    not, so they keep working even when the engine failed to load.
    """
    service: SimilarityService | None = getattr(request.app.state, "service", None)
    if service is None or not service.index_loaded:
        detail = getattr(request.app.state, "startup_error", None) or (
            "Similarity engine is not available."
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return service


# --------------------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------------------
def _http_error_for(exc: UploadError) -> HTTPException:
    """Translate an upload validation error into a meaningful HTTP status."""
    if isinstance(exc, UploadTooLargeError):
        code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    elif isinstance(exc, UnsupportedMediaTypeError):
        code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    elif isinstance(exc, EmptyUploadError):
        code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, CorruptImageError):
        code = status.HTTP_422_UNPROCESSABLE_ENTITY
    else:  # pragma: no cover - defensive default
        code = status.HTTP_400_BAD_REQUEST
    return HTTPException(code, detail=str(exc))


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------
@router.get("/", response_model=RootResponse, tags=["meta"], summary="Service metadata")
def root(settings: Settings = Depends(get_settings)) -> RootResponse:
    """Return the API name, version and the list of available endpoints."""
    return RootResponse(
        name=settings.api_title,
        version=settings.api_version,
        description="Visual image similarity search over a WikiArt embedding index.",
        endpoints=[
            EndpointInfo(method="GET", path="/", description="Service metadata."),
            EndpointInfo(method="GET", path="/health", description="Readiness and health."),
            EndpointInfo(method="POST", path="/upload", description="Validate and store an image."),
            EndpointInfo(
                method="POST",
                path="/search",
                description="Find the most similar indexed images for an upload.",
            ),
        ],
    )


@router.get("/health", response_model=HealthResponse, tags=["meta"], summary="Health check")
def health(
    request: Request, settings: Settings = Depends(get_settings)
) -> HealthResponse | JSONResponse:
    """Report detailed readiness: model loaded, index loaded, size and backend.

    Returns ``200`` when the service is fully ready and ``503`` otherwise, so the
    endpoint doubles as a load-balancer / orchestrator health probe. The response
    body has the same shape in both cases, so monitors can parse it uniformly.
    """
    service: SimilarityService | None = getattr(request.app.state, "service", None)
    startup_error: str | None = getattr(request.app.state, "startup_error", None)

    if service is not None and service.index_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=service.model_loaded,
            index_loaded=True,
            index_size=service.index_size,
            embedding_dim=int(service.meta.get("dim", 0)),
            search_backend=service.backend,
            model_name=str(service.meta.get("model_name", "")),
            detail=None,
        )

    payload = HealthResponse(
        status="unavailable",
        model_loaded=bool(service and service.model_loaded),
        index_loaded=False,
        index_size=0,
        embedding_dim=0,
        search_backend="none",
        model_name="",
        detail=startup_error or "Index not loaded.",
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload.model_dump()
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["images"],
    summary="Upload and validate an image",
)
async def upload(
    file: UploadFile = File(..., description="Image file (jpg, png, webp, bmp)."),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """Validate an uploaded image and store it, returning its metadata.

    Validation covers content-type/extension, size cap and a real decode pass
    that rejects corrupt or truncated files before anything is persisted.
    """
    try:
        suffix = check_media_type(file.filename, file.content_type, settings)
        data = await read_capped(file, settings.max_upload_bytes)
        info = await run_in_threadpool(inspect_image, data, suffix)
    except UploadError as exc:
        raise _http_error_for(exc) from exc

    upload_id = new_upload_id()
    await run_in_threadpool(store_upload, data, settings.upload_dir, upload_id, suffix)
    return UploadResponse(
        upload_id=upload_id,
        filename=file.filename or f"{upload_id}{suffix}",
        content_type=file.content_type or "",
        size_bytes=len(data),
        width=info.width,
        height=info.height,
        image_format=info.image_format,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["images"],
    summary="Search for visually similar images",
)
async def search(
    file: UploadFile | None = File(default=None, description="Image to search with."),
    upload_id: str | None = Form(default=None, description="Id from a prior /upload."),
    service: SimilarityService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    """Return the top matches for a query supplied inline or by upload id.

    Exactly one of ``file`` (a fresh multipart image) or ``upload_id`` (an image
    already stored via ``/upload``) must be provided.
    """
    if (file is None) == (upload_id is None):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Provide exactly one of 'file' or 'upload_id'.",
        )

    if upload_id is not None:
        # Query a previously stored upload by id.
        query_path = find_stored_upload(settings.upload_dir, upload_id)
        if query_path is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail=f"No stored upload with id {upload_id!r}."
            )
        return await _run_search(service, query_path, query_name=upload_id, k=settings.top_k)

    # Query an inline upload; the file is validated then held only transiently.
    assert file is not None  # narrowed by the exclusivity check above
    try:
        suffix = check_media_type(file.filename, file.content_type, settings)
        data = await read_capped(file, settings.max_upload_bytes)
        await run_in_threadpool(inspect_image, data, suffix)
    except UploadError as exc:
        raise _http_error_for(exc) from exc

    query_name = file.filename or "upload"
    with temporary_image(data, settings.upload_dir, suffix) as query_path:
        return await _run_search(service, query_path, query_name=query_name, k=settings.top_k)

#s
async def _run_search(
    service: SimilarityService, query_path: Path, query_name: str, k: int
) -> SearchResponse:
    """Encode + search a query image path and shape the response.

    The engine call is CPU-bound (image encode + vector search) and is run in a
    worker thread so it never blocks the event loop.
    """
    try:
        results = await run_in_threadpool(service.search, query_path, k)
    except InvalidQueryImageError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except FileNotFoundError as exc:  # pragma: no cover - path was just written
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    matches = [
        Match(image=r.path, score=round(r.score, 6), style=r.style, artist=r.artist, title=r.title)
        for r in results
    ]
    return SearchResponse(query=query_name, count=len(matches), matches=matches)
