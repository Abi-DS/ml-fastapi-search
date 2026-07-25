"""Service construction and degraded-startup behaviour.

Covers the paths where the index is missing or incomplete: the app must still
start, ``/`` and ``/health`` must remain reachable, and index-dependent
endpoints must fail cleanly with ``503``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services import SimilarityService
from tests.conftest import png_bytes


def _settings_for(embeddings_dir: Path, tmp_path: Path) -> Settings:
    return Settings(
        embeddings_dir=embeddings_dir,
        upload_dir=tmp_path / "uploads",
        eager_load=False,
        log_level="WARNING",
    )


def test_from_settings_missing_index_raises(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path / "empty", tmp_path)
    with pytest.raises(FileNotFoundError):
        SimilarityService.from_settings(settings)


def test_missing_index_degrades_gracefully(tmp_path: Path) -> None:
    settings = _settings_for(tmp_path / "empty", tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        # Metadata endpoints stay up.
        assert client.get("/").status_code == 200
        # Health reports unavailable with a 503.
        health = client.get("/health")
        assert health.status_code == 503
        # Search depends on the index and returns 503.
        search = client.post("/search", files={"file": ("q.png", png_bytes(), "image/png")})
        assert search.status_code == 503
        # Upload does not need the index and still works.
        upload = client.post("/upload", files={"file": ("q.png", png_bytes(), "image/png")})
        assert upload.status_code == 201


def test_incomplete_index_is_treated_as_missing(tmp_path: Path) -> None:
    # Only the embeddings matrix exists; manifest + meta are absent.
    partial = tmp_path / "partial"
    partial.mkdir()
    np.save(partial / "embeddings.npy", np.zeros((3, 16), dtype=np.float32))

    settings = _settings_for(partial, tmp_path)
    with pytest.raises(FileNotFoundError):
        SimilarityService.from_settings(settings)

    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
