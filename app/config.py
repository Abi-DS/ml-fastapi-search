"""Application configuration.

All service tunables live in a single :class:`Settings` object, populated from
environment variables (prefix ``APP_``) with sensible defaults, so the same image
runs unchanged on a laptop, in CI or in production. Engine-level configuration
(model name, index location) is derived from these settings and handed to the
vendored :class:`engine.config.Config`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (…/ml-fastapi-search), derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Immutable runtime settings, resolved from ``APP_*`` environment variables.

    Attributes:
        api_title: Human-readable service name.
        api_version: Semantic version reported by ``/`` and ``/health``.
        embeddings_dir: Directory holding the three-file index
            (``embeddings.npy``, ``manifest.csv``, ``index_meta.json``).
        dataset_root: Optional corpus root; only used to resolve absolute paths
            of matched images. The service works without it.
        upload_dir: Directory where validated uploads are stored.
        max_upload_bytes: Hard cap on accepted upload size.
        allowed_content_types: Accepted ``Content-Type`` values for uploads.
        allowed_extensions: Accepted file extensions (lower-case, with dot).
        top_k: Number of matches returned by ``/search``.
        device: Engine device preference (``auto`` / ``cpu`` / ``cuda``).
        prefer_faiss: Use the FAISS backend when available (else NumPy).
        mmap: Memory-map the embedding matrix instead of loading it into RAM.
        eager_load: Load the index and model at startup (vs. lazily).
        log_level: Root logging level.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_title: str = "ML FastAPI Image Similarity Search"
    api_version: str = "1.0.0"

    embeddings_dir: Path = REPO_ROOT / "embeddings"
    dataset_root: Path | None = None

    upload_dir: Path = REPO_ROOT / "uploads"
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    )
    allowed_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    )

    top_k: int = Field(default=5, ge=1, le=50)

    device: str = "auto"
    prefer_faiss: bool = True
    mmap: bool = False
    eager_load: bool = True

    log_level: str = "INFO"

    @field_validator("allowed_content_types", "allowed_extensions", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated strings for the tuple-valued env overrides."""
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
