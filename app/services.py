"""Similarity service — the bridge between the REST layer and the engine.

:class:`SimilarityService` owns the two heavyweight objects the API needs and
loads each exactly once:

* a :class:`engine.search.SimilarityEngine` (the persisted index + FAISS/NumPy
  searcher), and
* an :class:`engine.model.EmbeddingModel` (the OpenCLIP encoder).

A single model instance is shared into the engine via the ``_model`` seam that
the reference engine exposes for injection, so an uploaded image is encoded and
searched through the engine's own, already-tested code path. The service is
constructed once at application startup and injected into routes as a FastAPI
dependency, keeping per-request latency to just encode + search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from engine.config import Config
from engine.model import EmbeddingModel
from engine.search import InvalidQueryImageError, SearchResult, SimilarityEngine

from .config import Settings

logger = logging.getLogger("app.services")

# Re-exported so routes can catch the engine's decode error without importing
# from the vendored package directly.
__all__ = ["SimilarityService", "InvalidQueryImageError"]


class SimilarityService:
    """Owns the loaded index and model and answers similarity queries.

    Args:
        engine: A ready :class:`SimilarityEngine`.
        model: The encoder to share into ``engine`` (``None`` for a lazy engine
            that will load its own model on first query).
        settings: Active application settings.
    """

    def __init__(
        self,
        engine: SimilarityEngine,
        model: EmbeddingModel | None,
        settings: Settings,
    ) -> None:
        self._engine = engine
        self._model = model
        self._settings = settings
        if model is not None:
            # Share the single, eagerly-loaded model with the engine so queries
            # never trigger a second (lazy) model load.
            engine._model = model  # noqa: SLF001 - documented injection seam

    @classmethod
    def from_settings(cls, settings: Settings) -> "SimilarityService":
        """Build a fully-loaded service from settings.

        Loads the persisted index and (when ``eager_load``) the embedding model.

        Raises:
            FileNotFoundError: If no complete index exists in ``embeddings_dir``.
        """
        # The engine's Config needs a dataset root only to resolve absolute
        # paths of matches; the service never requires the corpus on disk, so we
        # fall back to the embeddings dir when no dataset root is configured.
        config = Config(
            dataset_root=settings.dataset_root or settings.embeddings_dir,
            embeddings_dir=settings.embeddings_dir,
            device=settings.device,
        )
        engine = SimilarityEngine(
            config, prefer_faiss=settings.prefer_faiss, mmap=settings.mmap
        )
        model: EmbeddingModel | None = None
        if settings.eager_load:
            model = EmbeddingModel(
                engine.meta.get("model_name", config.model_name),
                engine.meta.get("pretrained", config.pretrained),
                config.device,
            )
        logger.info(
            "Service ready: %d vectors, backend=%s, model_loaded=%s",
            engine.size,
            engine.backend,
            model is not None,
        )
        return cls(engine, model, settings)

    # --- Introspection ------------------------------------------------------------
    @property
    def index_loaded(self) -> bool:
        """Whether a non-empty index is loaded and queryable."""
        return self._engine is not None and self._engine.size > 0

    @property
    def model_loaded(self) -> bool:
        """Whether the embedding model is loaded and ready to encode."""
        return self._model is not None

    @property
    def index_size(self) -> int:
        """Number of indexed vectors."""
        return self._engine.size

    @property
    def backend(self) -> str:
        """Active search backend name."""
        return self._engine.backend

    @property
    def meta(self) -> dict[str, Any]:
        """Index metadata sidecar (model, dim, counts)."""
        return self._engine.meta

    # --- Querying -----------------------------------------------------------------
    def search(self, image_path: str | Path, k: int) -> list[SearchResult]:
        """Return the top-``k`` matches for a query image on disk.

        The query is an externally uploaded image, so it is never part of the
        indexed corpus; self-exclusion is therefore disabled.

        Args:
            image_path: Path to the (already validated) query image.
            k: Number of matches to return.

        Returns:
            Ranked results, most similar first.

        Raises:
            FileNotFoundError: If the query path does not exist.
            InvalidQueryImageError: If the file cannot be decoded.
        """
        return self._engine.search(image_path, k=k, exclude_self=False)
