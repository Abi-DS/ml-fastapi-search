"""Pydantic request and response schemas.

These models define the public API contract and are the single source of truth
for request validation and OpenAPI documentation. Keeping them in one module
lets the routes stay thin and declarative.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EndpointInfo(BaseModel):
    """A single advertised endpoint."""

    method: str = Field(..., examples=["POST"])
    path: str = Field(..., examples=["/search"])
    description: str


class RootResponse(BaseModel):
    """Payload for ``GET /`` — service identity and available endpoints."""

    name: str
    version: str
    description: str
    endpoints: list[EndpointInfo]


class HealthResponse(BaseModel):
    """Payload for ``GET /health`` — detailed readiness information."""

    status: str = Field(..., examples=["ok"], description="'ok' or 'unavailable'.")
    model_loaded: bool
    index_loaded: bool
    index_size: int = Field(..., description="Number of vectors in the index.")
    embedding_dim: int = Field(..., description="Vector dimensionality (0 if not loaded).")
    search_backend: str = Field(..., description="Active search backend name.")
    model_name: str
    detail: str | None = Field(default=None, description="Failure detail when unavailable.")


class UploadResponse(BaseModel):
    """Payload for ``POST /upload`` — metadata for a stored upload."""

    upload_id: str = Field(..., description="Server-generated identifier for the stored file.")
    filename: str = Field(..., description="Original client-supplied filename.")
    content_type: str
    size_bytes: int
    width: int
    height: int
    image_format: str = Field(..., description="Decoded image format, e.g. 'JPEG'.")


class Match(BaseModel):
    """A single ranked search match."""

    image: str = Field(..., description="Dataset-relative path of the matched image.")
    score: float = Field(..., description="Cosine similarity in [-1, 1].")
    style: str = Field(default="", description="Style label from the index manifest.")
    artist: str = Field(default="", description="Artist name from the index manifest.")
    title: str = Field(default="", description="Title from the index manifest.")


class SearchResponse(BaseModel):
    """Payload for ``POST /search`` — the query reference and ranked matches."""

    query: str = Field(..., description="Identifier of the query image (filename or upload id).")
    count: int = Field(..., description="Number of matches returned.")
    matches: list[Match]


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for non-2xx responses."""

    detail: str
