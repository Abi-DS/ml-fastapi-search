"""Endpoint tests: routing, validation, error mapping and response schemas.

These run against a :class:`TestClient` whose service uses a real (tiny) index
and a stub encoder — see :mod:`tests.conftest`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import INDEX_SIZE, jpeg_bytes, png_bytes


# --- GET / -----------------------------------------------------------------------------
def test_root_returns_metadata(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] and body["version"]
    paths = {e["path"] for e in body["endpoints"]}
    assert {"/", "/health", "/upload", "/search"} <= paths


# --- GET /health -----------------------------------------------------------------------
def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True
    assert body["model_loaded"] is True  # stub encoder injected
    assert body["index_size"] == INDEX_SIZE
    assert body["embedding_dim"] > 0
    assert body["search_backend"]


# --- POST /upload ----------------------------------------------------------------------
def test_upload_valid_png(client: TestClient) -> None:
    resp = client.post("/upload", files={"file": ("query.png", png_bytes(), "image/png")})
    assert resp.status_code == 201
    body = resp.json()
    assert body["upload_id"]
    assert body["image_format"] == "PNG"
    assert body["width"] == 32 and body["height"] == 32
    assert body["size_bytes"] > 0


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    resp = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 415


def test_upload_rejects_corrupt_image(client: TestClient) -> None:
    resp = client.post(
        "/upload", files={"file": ("broken.jpg", b"not a real jpeg", "image/jpeg")}
    )
    assert resp.status_code == 422


def test_upload_rejects_extension_content_mismatch(client: TestClient) -> None:
    # Real PNG bytes but a .jpg name + jpeg content-type: the decode reveals the
    # true format and the mismatch is rejected.
    resp = client.post("/upload", files={"file": ("liar.jpg", png_bytes(), "image/jpeg")})
    assert resp.status_code == 422


def test_upload_rejects_empty_file(client: TestClient) -> None:
    resp = client.post("/upload", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400


def test_upload_rejects_too_large(client: TestClient, settings) -> None:
    oversized = jpeg_bytes(size=8) + b"\x00" * (settings.max_upload_bytes + 1)
    resp = client.post("/upload", files={"file": ("big.jpg", oversized, "image/jpeg")})
    assert resp.status_code == 413


# --- POST /search ----------------------------------------------------------------------
def test_search_with_uploaded_file(client: TestClient) -> None:
    resp = client.post("/search", files={"file": ("q.jpg", jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "q.jpg"
    assert body["count"] == 5  # default top_k
    assert len(body["matches"]) == 5
    top = body["matches"][0]
    assert set(top) >= {"image", "score"}
    # The stub maps the query to indexed vector 0, so the top score is ~1.0.
    assert top["score"] == max(m["score"] for m in body["matches"])
    assert top["score"] > 0.99


def test_inline_search_leaves_no_residual_file(client: TestClient, settings) -> None:
    resp = client.post("/search", files={"file": ("q.jpg", jpeg_bytes(), "image/jpeg")})
    assert resp.status_code == 200
    # The transient query file is removed; only persisted uploads (none here) remain.
    residual = list(settings.upload_dir.glob("query-*"))
    assert residual == []


def test_upload_persists_file(client: TestClient, settings) -> None:
    resp = client.post("/upload", files={"file": ("q.png", png_bytes(), "image/png")})
    upload_id = resp.json()["upload_id"]
    assert list(settings.upload_dir.glob(f"{upload_id}.*"))


def test_search_by_upload_id(client: TestClient) -> None:
    up = client.post("/upload", files={"file": ("q.png", png_bytes(), "image/png")})
    upload_id = up.json()["upload_id"]

    resp = client.post("/search", data={"upload_id": upload_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == upload_id
    assert len(body["matches"]) == 5


def test_search_requires_exactly_one_input(client: TestClient) -> None:
    # Neither provided -> 400.
    neither = client.post("/search")
    assert neither.status_code == 400

    # Both provided -> 400.
    both = client.post(
        "/search",
        files={"file": ("q.jpg", jpeg_bytes(), "image/jpeg")},
        data={"upload_id": "deadbeef"},
    )
    assert both.status_code == 400


def test_search_unknown_upload_id_returns_404(client: TestClient) -> None:
    resp = client.post("/search", data={"upload_id": "abc123def456"})
    assert resp.status_code == 404


def test_search_corrupt_file_returns_422(client: TestClient) -> None:
    resp = client.post("/search", files={"file": ("q.jpg", b"garbage", "image/jpeg")})
    assert resp.status_code == 422


def test_search_response_matches_schema(client: TestClient) -> None:
    resp = client.post("/search", files={"file": ("q.jpg", jpeg_bytes(), "image/jpeg")})
    body = resp.json()
    assert set(body) == {"query", "count", "matches"}
    for match in body["matches"]:
        assert set(match) >= {"image", "score"}
        assert isinstance(match["image"], str)
        assert -1.0 <= match["score"] <= 1.0
