# ML FastAPI Image Similarity Search

A production-grade **FastAPI** service that exposes a visual image-similarity
engine over REST. Upload a painting and the service returns the most visually
similar works from an ~81,000-image WikiArt corpus, each with a cosine
similarity score.

The service is a thin, well-tested API layer over a proven similarity engine:
a pretrained **OpenCLIP ViT-B/32** encoder and an exact **FAISS** inner-product
index (with an automatic NumPy fallback). The engine, the embeddings and the
index format are **reused unchanged** from the reference project rather than
reimplemented — see [Reuse & architecture](#reuse--architecture).

```
Upload ─▶ validate ─▶ OpenCLIP encode ─▶ 512-d unit vector ─▶ FAISS top-5 ─▶ JSON matches + scores
```

---

## Table of contents

- [Overview](#overview)
- [Reuse & architecture](#reuse--architecture)
- [Repository layout](#repository-layout)
- [API documentation](#api-documentation)
- [Request / response examples](#request--response-examples)
- [Setup](#setup)
- [Running the service](#running-the-service)
- [Docker](#docker)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Assumptions](#assumptions)
- [Limitations](#limitations)

---

## Overview

The service has four responsibilities, each isolated in its own module:

1. **Advertise** itself (`GET /`) and **report health** (`GET /health`).
2. **Accept and validate** image uploads (`POST /upload`) — format, size and a
   real decode pass that rejects corrupt files — then store them safely.
3. **Search** the index (`POST /search`) for the top-5 most similar images,
   accepting either an inline image or a previously uploaded one.
4. **Load the model and index once** at startup and keep per-request latency to
   just *encode + search*.

Similarity is **cosine similarity**. Because the indexed embeddings are
L2-normalised at generation time, an inner product *is* the cosine, so both the
FAISS and NumPy backends return identical scores in `[-1, 1]`.

## Reuse & architecture

This repository is standalone but does **not** reimplement the ML core. The
embedding model, similarity search, index format and design decisions are the
reference project's; that engine is **vendored verbatim** into `engine/` (only
the modules needed to *serve* an existing index — ingestion is out of scope).
The FastAPI layer in `app/` builds on top of it and contains no embedding or
search logic of its own.

```
┌─────────────────────────── app/ (new: REST layer) ───────────────────────────┐
│  main.py    FastAPI factory + lifespan (load once) + exception handlers       │
│  routes.py  4 endpoints, dependency injection, HTTP status mapping            │
│  services.py SimilarityService — owns the loaded engine + model               │
│  models.py  Pydantic request/response schemas (the API contract)              │
│  utils.py   upload validation (size, type, corruption) + safe storage         │
│  config.py  environment-driven Settings                                       │
└───────────────────────────────────┬───────────────────────────────────────────┘
                                     │ depends on (never modifies)
┌────────────────────── engine/ (vendored reference core) ──────────────────────┐
│  model.py  OpenCLIP encoder → 512-d L2-normalised vectors                     │
│  search.py SimilarityEngine + FAISS/NumPy backends                            │
│  store.py  three-file index loader                                            │
│  config.py engine Config dataclass                                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Load-once lifecycle.** On startup the FastAPI lifespan builds one
`SimilarityService`, which loads the FAISS index and the OpenCLIP model a single
time and shares the model into the engine. The service is stored on
`app.state` and injected into routes via `Depends(get_service)`. If the index
cannot be loaded, the app still starts: `/` and `/health` keep working (health
reports `503`), and index-dependent endpoints return `503` until it is present.

**The persisted index** is three transparent files under `embeddings/`
(produced by the reference project, reused as-is):

| File | Contents |
| --- | --- |
| `embeddings.npy` | `float32 [81444, 512]` matrix of unit-norm vectors (159 MB) |
| `manifest.csv` | one row per image: `path, style, artist, title, size, mtime` |
| `index_meta.json` | model identity, dimensionality and counts |

Row *i* of the matrix corresponds to row *i* of the manifest.

## Repository layout

```
ml-fastapi-search/
├── app/                  # FastAPI application (this project)
│   ├── __init__.py
│   ├── main.py           # app factory, lifespan, exception handlers
│   ├── routes.py         # GET / , GET /health , POST /upload , POST /search
│   ├── services.py       # SimilarityService (loads engine + model once)
│   ├── models.py         # Pydantic schemas
│   ├── config.py         # Settings (APP_* environment variables)
│   └── utils.py          # upload validation + safe storage
├── engine/               # Vendored reference engine (embedding + search)
│   ├── config.py  model.py  store.py  search.py  logging_utils.py
├── tests/                # Automated tests (pytest + FastAPI TestClient)
│   ├── conftest.py  test_api.py  test_service.py
├── embeddings/           # Index files mounted/placed here (git-ignored)
├── uploads/              # Runtime upload storage (git-ignored)
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── pyproject.toml
└── README.md
```

## API documentation

Interactive OpenAPI docs are served at **`/docs`** (Swagger UI) and **`/redoc`**
when the service is running.

### `GET /`
Service metadata. Returns the API name, version and the list of endpoints.
Always `200`.

### `GET /health`
Detailed readiness for orchestrators and load balancers.

| Field | Meaning |
| --- | --- |
| `status` | `"ok"` when ready, `"unavailable"` otherwise |
| `model_loaded` | encoder is loaded |
| `index_loaded` | index is loaded and non-empty |
| `index_size` | number of indexed vectors |
| `embedding_dim` | vector dimensionality |
| `search_backend` | active backend (`faiss.IndexFlatIP` or `numpy.dot`) |
| `model_name` | encoder architecture |

Returns `200` when healthy, **`503`** when the engine is not ready.

### `POST /upload`
Validate and store an image. `multipart/form-data`, field **`file`**.

Validation, in order:

| Failure | Status |
| --- | --- |
| Extension / content-type not allowed | `415 Unsupported Media Type` |
| Larger than the size cap (default 10 MiB) | `413 Payload Too Large` |
| Empty file | `400 Bad Request` |
| Not decodable / truncated / format-mismatch | `422 Unprocessable Entity` |
| Valid | `201 Created` + metadata (incl. a server-generated `upload_id`) |

### `POST /search`
Return the top-5 matches. `multipart/form-data`; provide **exactly one** of:

- **`file`** — an inline image (validated, held only transiently), or
- **`upload_id`** — the id of an image previously stored via `/upload`.

| Failure | Status |
| --- | --- |
| Neither or both of `file` / `upload_id` | `400 Bad Request` |
| `upload_id` not found | `404 Not Found` |
| Inline file fails validation | `415` / `413` / `422` (as above) |
| Engine not loaded | `503 Service Unavailable` |
| Valid | `200 OK` + ranked matches |

Response shape:

```json
{
  "query": "william-turner_venedig.jpg",
  "count": 5,
  "matches": [
    { "image": "Romanticism/...jpg", "score": 0.94, "style": "...", "artist": "...", "title": "..." }
  ]
}
```

Each match always carries `image` (dataset-relative path) and `score` (cosine in
`[-1, 1]`); `style`, `artist` and `title` are added from the index manifest.

## Request / response examples

The examples below are **real output** from the running service querying the
full 81,444-image index with an OpenCLIP encoder.

**Health:**

```bash
curl -s http://localhost:8000/health
```
```json
{
  "status": "ok", "model_loaded": true, "index_loaded": true,
  "index_size": 81444, "embedding_dim": 512,
  "search_backend": "faiss.IndexFlatIP", "model_name": "ViT-B-32", "detail": null
}
```

**Search with an inline image:**

```bash
curl -s -X POST http://localhost:8000/search \
  -F "file=@william-turner_venedig.jpg;type=image/jpeg"
```
```json
{
  "query": "william-turner_venedig.jpg",
  "count": 5,
  "matches": [
    { "image": "Romanticism/william-turner_venedig.jpg", "score": 1.0,
      "style": "Romanticism", "artist": "william turner", "title": "venedig" },
    { "image": "Romanticism/william-turner_the-burning-of-the-houses-of-parliament-1.jpg",
      "score": 0.813763, "style": "Romanticism", "artist": "william turner",
      "title": "the burning of the houses of parliament 1" },
    { "image": "Romanticism/william-turner_approach-to-venice.jpg", "score": 0.808066,
      "style": "Romanticism", "artist": "william turner", "title": "approach to venice" },
    { "image": "Romanticism/william-turner_regulus-1837.jpg", "score": 0.803927,
      "style": "Romanticism", "artist": "william turner", "title": "regulus 1837" },
    { "image": "Romanticism/william-turner_departure-of-the-fleet.jpg", "score": 0.797471,
      "style": "Romanticism", "artist": "william turner", "title": "departure of the fleet" }
  ]
}
```

**Upload, then search by id:**

```bash
# 1) upload -> returns {"upload_id": "...", ...}
curl -s -X POST http://localhost:8000/upload \
  -F "file=@william-turner_venedig.jpg;type=image/jpeg"

# 2) search using the returned id
curl -s -X POST http://localhost:8000/search -F "upload_id=<UPLOAD_ID>"
```

## Setup

Requires **Python 3.11+**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
```

**Provide the index.** The service needs the three index files in
`embeddings/`. They are generated by the reference project and are *not*
committed (they are large, and are git-ignored here). Copy or mount them:

```bash
# copy the generated index into place …
cp /path/to/reference/embeddings/{embeddings.npy,manifest.csv,index_meta.json} embeddings/
# … or point the service elsewhere without copying:
export APP_EMBEDDINGS_DIR=/path/to/embeddings
```

On first search the OpenCLIP weights (~600 MB) are downloaded once and cached
under the Hugging Face cache. The dataset itself is **not** required to run the
service — only the index files are.

### Configuration

All settings are environment variables with the `APP_` prefix (defaults shown):

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_EMBEDDINGS_DIR` | `./embeddings` | Index location |
| `APP_UPLOAD_DIR` | `./uploads` | Where uploads are stored |
| `APP_MAX_UPLOAD_BYTES` | `10485760` (10 MiB) | Upload size cap |
| `APP_TOP_K` | `5` | Matches returned by `/search` |
| `APP_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `APP_PREFER_FAISS` | `true` | Use FAISS when available |
| `APP_MMAP` | `false` | Memory-map the index instead of RAM |
| `APP_EAGER_LOAD` | `true` | Load model + index at startup |
| `APP_LOG_LEVEL` | `INFO` | Logging level |

## Running the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/docs>.

## Docker

The image contains only code and dependencies; the index is mounted at runtime,
so the image stays small and the container is stateless. It installs CPU-only
PyTorch wheels and **runs as a non-root user**.

```bash
docker build -t ml-fastapi-search .

docker run --rm -p 8000:8000 \
  -v "$(pwd)/embeddings:/app/embeddings:ro" \
  ml-fastapi-search
```

A container `HEALTHCHECK` polls `/health`. To persist the Hugging Face model
cache across runs, also mount a volume at `/app/.cache/huggingface`.

## Testing

The suite is fast, hermetic and requires **no model download and no dataset**.
A tiny synthetic index is written with the real engine store, and a stub encoder
maps queries to known vectors — so the FAISS/NumPy search runs for real while
only the image→vector step is faked. It covers every endpoint, the full
validation matrix (unsupported type, oversize, empty, corrupt,
extension/content mismatch), both `/search` input modes, the response schema,
and degraded startup (missing / incomplete index → `503`).

```bash
pip install -r requirements-dev.txt
pytest
```

Result on the reference machine:

```
19 passed
```

Beyond the automated suite, the service was **run end-to-end against the full
81,444-vector index and the real OpenCLIP model**; the [examples above](#request--response-examples)
are captured from that run. Measured end-to-end `/search` latency (HTTP
round-trip + decode of a 1849×1382 JPEG + encode + FAISS search) was
**~150–210 ms** on CPU, dominated by query encoding — consistent with the
reference engine's benchmark (exact FAISS search over 81k vectors is a few
milliseconds; the encoder is the bottleneck).

## Design decisions

- **Reuse over reimplementation.** The embedding and search code is vendored
  from the reference project unchanged. This guarantees the API returns exactly
  the same results as the validated engine and avoids a second, drift-prone copy
  of the ML logic. Only the module namespace (`src` → `engine`) was adapted.
- **Load once, inject everywhere.** Model and index are expensive; they are
  loaded a single time at startup and provided to handlers by dependency
  injection. Handlers hold no global state and are trivial to test with an
  overridden dependency.
- **Non-blocking request handling.** The CPU-bound work (image decode + encode +
  search) is dispatched to a worker thread via `run_in_threadpool`, so a slow
  inference never blocks the event loop. Measured: with four concurrent
  `/search` requests in flight, `/health` still responds in ~16 ms (versus
  ~835 ms if the inference ran inline on the loop).
- **Graceful degradation.** A missing index does not crash the process. `/` and
  `/health` stay up so an orchestrator can observe the failure and restart or
  wait, rather than facing a boot loop.
- **Validate before you encode.** Uploads are checked for type, size and real
  decodability *before* reaching the model, so bad input is rejected cheaply and
  with precise status codes rather than surfacing as a 500 deep in inference.
- **Server-controlled storage paths.** Stored filenames are derived from a
  server-generated hex id plus the validated extension, never from the client
  filename — no path traversal, no overwrites. Inline `/search` queries use a
  temporary file that is deleted after the request; only `/upload` persists.
- **Meaningful status codes.** `415` (type), `413` (size), `422` (corrupt/
  undecodable), `400` (bad request), `404` (unknown id), `503` (engine down).
- **Cosine via inner product.** Indexed vectors are unit-norm, so search is a
  single matrix multiply and FAISS/NumPy scores are directly comparable.

## Assumptions

- The three index files in `embeddings/` were produced by the reference project
  with **OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`)**, 512-d, L2-normalised — the
  service reads `index_meta.json` to load a matching encoder.
- The corpus itself is **not** needed to serve queries; the returned `image`
  field is a dataset-relative path, and rendering the actual thumbnail (which
  needs the dataset) is a client concern, out of scope here.
- Uploaded query images are external to the corpus, so query self-exclusion is a
  no-op; an image that *is* in the corpus will legitimately match itself at 1.0.
- Supported upload formats are JPEG, PNG, WEBP and BMP (configurable).

## Limitations

- **CPU-bound query latency.** End-to-end `/search` is dominated by encoding the
  query image on CPU (~150–210 ms measured); the exact search itself is a few
  milliseconds. The same code path runs an order of magnitude faster on a GPU
  (`APP_DEVICE=cuda`).
- **Single-process, in-memory index.** The FAISS `IndexFlatIP` holds the full
  159 MB matrix in RAM. This is the right choice at 81k vectors; a much larger
  corpus would call for approximate indexing (IVF/HNSW/PQ) or the engine's
  memory-mapped serving path (`APP_MMAP=true`).
- **Uploads are stored, not reaped.** `/upload` persists files under
  `uploads/`; a retention/eviction policy (TTL, size cap, object storage) is
  left to the deployment. Inline `/search` queries are already transient.
- **No authentication or rate limiting.** The service is intended to run behind
  a gateway that provides auth, TLS termination and throttling.
- **Hard request-body limit belongs at the edge.** The app enforces
  `APP_MAX_UPLOAD_BYTES` and returns `413`, which bounds the in-process copy of
  the upload, but the ASGI server still buffers the request body first. A hard
  guarantee against oversized bodies should be set on the fronting proxy (e.g.
  nginx `client_max_body_size`).
- **First run needs network access** to download the ~600 MB model weights
  (cached thereafter).
