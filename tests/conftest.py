"""Shared test fixtures.

The suite is fast, hermetic and free of any model download or dataset access:

* A **tiny synthetic index** is written to disk with the real engine store, so
  the storage/search path exercised is the production one.
* A **stub encoder** maps any query image to a known indexed vector, so the
  FAISS/NumPy search runs for real while image decoding is bypassed. This mirrors
  the injection seam the reference engine exposes.

Only the image → vector step is faked; everything else is the real code path.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.services import SimilarityService
from engine.search import SimilarityEngine
from engine.store import MANIFEST_COLUMNS, EmbeddingStore

INDEX_SIZE = 10
INDEX_DIM = 16


def _unit_rows(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, d)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def _manifest(n: int) -> pd.DataFrame:
    rows = [
        {
            "path": f"Style_{i % 3}/artist-{i}_title-{i}.jpg",
            "style": f"Style_{i % 3}",
            "artist": f"artist {i}",
            "title": f"title {i}",
            "size": 1000 + i,
            "mtime": 1_700_000_000 + i,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


class _StubModel:
    """Encoder stub: returns a fixed indexed vector, bypassing image decoding."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector.astype(np.float32)

    def encode_file(self, path: str) -> np.ndarray:  # noqa: D401 - matches engine API
        return self._vector


def png_bytes(color: tuple[int, int, int] = (120, 30, 200), size: int = 32) -> bytes:
    """Return the bytes of a small valid PNG image."""
    buf = BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(color: tuple[int, int, int] = (10, 80, 160), size: int = 32) -> bytes:
    """Return the bytes of a small valid JPEG image."""
    buf = BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    """Write a tiny but real three-file index and return its directory."""
    embeddings = _unit_rows(INDEX_SIZE, INDEX_DIM)
    store = EmbeddingStore(tmp_path / "embeddings")
    store.save(embeddings, _manifest(INDEX_SIZE), {"model_name": "ViT-B-32", "pretrained": "tag"})
    return store.dir


@pytest.fixture
def settings(tmp_path: Path, index_dir: Path) -> Settings:
    """Settings pointed at the tiny index, with a small upload cap for tests."""
    return Settings(
        embeddings_dir=index_dir,
        dataset_root=tmp_path / "dataset",
        upload_dir=tmp_path / "uploads",
        max_upload_bytes=1 * 1024 * 1024,
        eager_load=False,  # avoid a real torch model load; a stub is injected
        log_level="WARNING",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    """A TestClient whose loaded service uses the real index and a stub encoder."""
    app = create_app(settings)
    with TestClient(app) as test_client:
        # The lifespan built a real engine (index loaded, model=None); inject a
        # stub encoder so search runs for real without loading OpenCLIP.
        engine: SimilarityEngine = app.state.service._engine  # noqa: SLF001
        indexed_vector = engine._embeddings[0]  # noqa: SLF001
        stub = _StubModel(np.asarray(indexed_vector))
        service: SimilarityService = app.state.service
        service._model = stub  # noqa: SLF001
        engine._model = stub  # noqa: SLF001
        yield test_client
