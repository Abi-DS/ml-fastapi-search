# Production image for the FastAPI image-similarity search service.
#
# The service is stateless: the embedding index is mounted at runtime so the
# image stays small and portable — it contains only code and dependencies.
#
# Build:
#   docker build -t ml-fastapi-search .
#
# Run (index mounted read-only, service on http://localhost:8000):
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/embeddings:/app/embeddings:ro" \
#     ml-fastapi-search

FROM python:3.11-slim AS runtime

# libgomp1 provides the OpenMP runtime required by FAISS and PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    APP_EMBEDDINGS_DIR=/app/embeddings \
    APP_UPLOAD_DIR=/app/uploads

WORKDIR /app

# Install CPU-only PyTorch wheels first to avoid pulling large CUDA packages,
# then the remaining dependencies. This layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt

# Application code (engine is the vendored, reused reference core).
COPY app ./app
COPY engine ./engine

# Run as an unprivileged user; pre-create writable mount points.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/embeddings /app/uploads /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Container-level health probe hitting the readiness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
