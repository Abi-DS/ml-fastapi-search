"""Visual image similarity search engine (vendored, unchanged reference core).

This package is reused verbatim from the ``ml-image-similarity`` reference
project — it is the authoritative implementation of the embedding model,
similarity search and on-disk index format. Only the modules required to
*serve* an existing index are vendored here (ingestion is out of scope):

- :mod:`engine.config`  - runtime configuration (env-var and CLI overridable).
- :mod:`engine.model`   - model loading, preprocessing and embedding generation.
- :mod:`engine.store`   - three-file embedding index persistence and loading.
- :mod:`engine.search`  - similarity search behind a backend-agnostic API.

The FastAPI application in :mod:`app` builds a thin REST layer on top of this
package; it does not reimplement any embedding or search logic.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
