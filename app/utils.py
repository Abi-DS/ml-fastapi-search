"""Upload validation and safe storage helpers.

The functions here enforce the upload contract *before* an image ever reaches
the (expensive) embedding model: size caps, an extension/content-type allowlist,
and a real decode pass that rejects corrupt or truncated files. Validation
failures raise the typed exceptions below, which the routes translate into
meaningful HTTP status codes.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from .config import Settings

logger = logging.getLogger("app.utils")

# Decode-time failures that indicate the bytes are not a usable image.
_DECODE_ERRORS: tuple[type[Exception], ...] = (
    UnidentifiedImageError,
    Image.DecompressionBombError,
    OSError,
    ValueError,
)

# Pillow format names accepted, keyed off the settings allowlist of extensions.
_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
}


class UploadError(Exception):
    """Base class for upload validation failures."""


class EmptyUploadError(UploadError):
    """Raised when the request carries no file or an empty file."""


class UploadTooLargeError(UploadError):
    """Raised when the upload exceeds the configured size cap."""


class UnsupportedMediaTypeError(UploadError):
    """Raised when the extension or content-type is not in the allowlist."""


class CorruptImageError(UploadError):
    """Raised when the bytes cannot be decoded as a supported image."""


@dataclass(frozen=True)
class ImageInfo:
    """Metadata extracted from a validated image."""

    image_format: str
    width: int
    height: int


async def read_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload fully but reject anything larger than ``max_bytes``.

    Reading is capped so a hostile or accidental huge upload cannot exhaust
    memory: at most ``max_bytes + 1`` bytes are ever held.

    Args:
        upload: The incoming multipart file.
        max_bytes: Inclusive maximum number of bytes to accept.

    Returns:
        The full file contents.

    Raises:
        EmptyUploadError: If the file is empty.
        UploadTooLargeError: If the file exceeds ``max_bytes``.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(
                f"Upload exceeds the maximum allowed size of {max_bytes} bytes."
            )
        chunks.append(chunk)
    if total == 0:
        raise EmptyUploadError("Uploaded file is empty.")
    return b"".join(chunks)


def check_media_type(filename: str | None, content_type: str | None, settings: Settings) -> str:
    """Validate the declared extension and content-type against the allowlist.

    Args:
        filename: Client-supplied filename (may be ``None``).
        content_type: Declared MIME type (may be ``None``).
        settings: Active settings carrying the allowlists.

    Returns:
        The lower-cased file extension (including the leading dot).

    Raises:
        UnsupportedMediaTypeError: If either the extension or content-type is
            not allowed.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise UnsupportedMediaTypeError(
            f"Unsupported file extension {suffix or '(none)'!r}. "
            f"Allowed: {', '.join(settings.allowed_extensions)}."
        )
    if content_type and content_type not in settings.allowed_content_types:
        raise UnsupportedMediaTypeError(
            f"Unsupported content-type {content_type!r}. "
            f"Allowed: {', '.join(settings.allowed_content_types)}."
        )
    return suffix


def inspect_image(data: bytes, expected_suffix: str) -> ImageInfo:
    """Decode the bytes to prove they are a genuine, supported image.

    A two-pass check is used: :meth:`PIL.Image.Image.verify` detects truncated or
    corrupt files, then a fresh decode reads the true format and dimensions.
    The decoded format must match the declared extension, which prevents a
    mislabelled file (e.g. a GIF renamed to ``.jpg``) from slipping through.

    Args:
        data: Raw image bytes.
        expected_suffix: The validated file extension.

    Returns:
        The decoded image's format and dimensions.

    Raises:
        CorruptImageError: If the bytes cannot be decoded or the true format is
            not the one implied by the extension.
    """
    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()  # cheap integrity check; consumes the object
        with Image.open(BytesIO(data)) as image:
            image.load()  # force a full decode to catch truncation
            fmt = (image.format or "").upper()
            width, height = image.size
    except _DECODE_ERRORS as exc:
        logger.warning("Rejected corrupt/unsupported upload: %s", exc)
        raise CorruptImageError(
            "The uploaded file could not be read as a valid image. It may be "
            "corrupt, truncated, or in an unsupported format."
        ) from exc

    expected_format = _EXTENSION_TO_FORMAT.get(expected_suffix)
    if expected_format and fmt != expected_format:
        raise CorruptImageError(
            f"File content is {fmt or 'unknown'} but the extension implies "
            f"{expected_format}. The extension and content must match."
        )
    return ImageInfo(image_format=fmt, width=width, height=height)


def new_upload_id() -> str:
    """Return a collision-resistant identifier for a stored upload."""
    return uuid.uuid4().hex


def store_upload(data: bytes, upload_dir: Path, upload_id: str, suffix: str) -> Path:
    """Persist upload bytes under a server-controlled, sanitised filename.

    The on-disk name is derived solely from a server-generated id and the
    validated extension, so a malicious client filename can never influence the
    storage path (no traversal, no overwrite of unrelated files).

    Args:
        data: Validated image bytes.
        upload_dir: Directory to write into (created if absent).
        upload_id: Server-generated identifier used as the base name.
        suffix: Validated file extension including the leading dot.

    Returns:
        The path the bytes were written to.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{upload_id}{suffix}"
    destination.write_bytes(data)
    logger.info("Stored upload %s (%d bytes) at %s", upload_id, len(data), destination)
    return destination


@contextmanager
def temporary_image(data: bytes, upload_dir: Path, suffix: str) -> Iterator[Path]:
    """Write bytes to a transient file for the duration of the ``with`` block.

    Used for ``/search`` queries supplied inline: the engine needs a path to
    decode, but the query image itself is not worth persisting, so the file is
    removed on exit regardless of success or failure.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"query-{new_upload_id()}{suffix}"
    path.write_bytes(data)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def find_stored_upload(upload_dir: Path, upload_id: str) -> Path | None:
    """Return the stored file for an upload id, or ``None`` if absent.

    The id is validated as a bare hex token so it can never escape ``upload_dir``.
    """
    if not upload_id or not all(c in "0123456789abcdef" for c in upload_id):
        return None
    matches = sorted(upload_dir.glob(f"{upload_id}.*"))
    return matches[0] if matches else None
