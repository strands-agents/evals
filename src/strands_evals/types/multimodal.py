"""Multimodal evaluation types and data structures."""

import base64
import binascii
import logging
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, field_validator, model_serializer

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Patterns for distinguishing source types
_HTTP_URL_PATTERN = re.compile(r"^https?://")
_DATA_URL_PATTERN = re.compile(r"^data:[^;]+;base64,")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_IMAGE_FORMAT_FROM_EXT = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".gif": "gif"}
_IMAGE_FORMATS = {"jpeg", "png", "gif", "webp"}


def _looks_like_file_path(value: str) -> bool:
    """Heuristic to determine if a string looks like a file path rather than base64."""
    ext = Path(value).suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        return True
    if value.startswith(("/", "./", "../", "~")):
        return True
    if "\\" in value:
        return True
    return False


def _detect_format(source: Any) -> Literal["jpeg", "png", "gif", "webp"] | None:
    """Detect image format from the source value."""
    if not isinstance(source, str):
        return None
    if _DATA_URL_PATTERN.match(source):
        media_part = source.split(";")[0]
        fmt = media_part.split("/")[-1].lower()
        return cast(Literal["jpeg", "png", "gif", "webp"], fmt) if fmt in _IMAGE_FORMATS else None
    is_local = _looks_like_file_path(source) and not _HTTP_URL_PATTERN.match(source)
    if is_local:
        result = _IMAGE_FORMAT_FROM_EXT.get(Path(source).suffix.lower())
        return cast(Literal["jpeg", "png", "gif", "webp"] | None, result)
    return None


def _download_http(url: str) -> bytes:
    """Download media from an HTTP/HTTPS URL.

    Uses ``urllib.request`` from the standard library (no extra dependencies).

    Raises:
        ValueError: If the download fails.
    """
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return response.read()
    except Exception as e:
        raise ValueError(f"Failed to download from {url}: {e}") from e


class ImageData(BaseModel):
    """Represents image data that can be passed to multimodal evaluators.

    Supports multiple input formats:
    - File path (str or Path): local file system path to an image
    - Base64 encoded string: raw base64-encoded image bytes
    - Data URL: ``data:image/<format>;base64,<data>``
    - HTTP/HTTPS URL: auto-fetched via ``urllib.request`` (stdlib)
    - PIL Image object (if PIL is available)
    - Raw bytes
    """

    model_config = {"arbitrary_types_allowed": True}

    source: str | bytes | Any
    format: Literal["jpeg", "png", "gif", "webp"] | None = None
    media_type: str | None = None

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v: Any) -> Any:
        """Validate and normalize image source data."""
        if isinstance(v, Path):
            v = str(v)

        if isinstance(v, str):
            if _DATA_URL_PATTERN.match(v):
                return v
            if _HTTP_URL_PATTERN.match(v):
                return v
            if _looks_like_file_path(v):
                path = Path(v)
                if not path.exists():
                    raise ValueError(
                        f"Image file not found: {path}. "
                        f"Supported sources: local file paths, base64, data URLs, HTTP URLs."
                    )
                if not path.is_file():
                    raise ValueError(f"Path is not a file: {path}")
                return str(path)
            # Assume remaining strings are base64-encoded data
            return v
        elif isinstance(v, bytes):
            return v
        elif PIL_AVAILABLE and isinstance(v, Image.Image):
            return v
        else:
            raise ValueError(
                f"Unsupported source type: {type(v)}. "
                f"Supported: file path, base64 string, data URL, HTTP URL, bytes, PIL Image."
            )

    def model_post_init(self, __context: Any) -> None:
        """Auto-detect format and media_type after initialization."""
        if self.format is None:
            self.format = _detect_format(self.source)
        if self.media_type is None and self.format:
            self.media_type = f"image/{self.format}"

    @model_serializer
    def _serialize(self) -> dict:
        """Custom serializer to ensure JSON-compatible output.

        Converts bytes sources to base64 strings and PIL Images to base64 strings
        so that model_dump() always returns JSON-serializable data.
        """
        source_val = self.source
        if isinstance(source_val, bytes):
            source_val = base64.b64encode(source_val).decode("utf-8")
        elif PIL_AVAILABLE and isinstance(source_val, Image.Image):
            buffer = BytesIO()
            fmt = (self.format or "png").upper()
            if fmt == "JPG":
                fmt = "JPEG"
            source_val.save(buffer, format=fmt)
            source_val = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"source": source_val, "format": self.format, "media_type": self.media_type}

    def to_bytes(self) -> bytes:
        """Convert image source to raw bytes for strands Agent content blocks."""
        return resolve_image_bytes(self)

    def to_base64(self) -> str:
        """Convert image to base64 string."""
        return base64.b64encode(self.to_bytes()).decode("utf-8")

    def to_data_url(self) -> str:
        """Convert image to data URL format."""
        base64_data = self.to_base64()
        media_type = self.media_type or "image/png"
        return f"data:{media_type};base64,{base64_data}"


def resolve_image_bytes(image_data: "ImageData") -> bytes:
    """Resolve an ImageData instance to raw bytes.

    Handles all supported source types: raw bytes, data URLs, HTTP/HTTPS URLs,
    local file paths, base64 strings, and PIL Images.

    This function is the single point of I/O for image resolution, making it
    easy to test and mock network/filesystem calls independently of the
    ImageData data model.

    Args:
        image_data: The ImageData instance to resolve.

    Returns:
        Raw image bytes.

    Raises:
        ValueError: If the source cannot be converted to bytes.
    """
    source = image_data.source
    if isinstance(source, bytes):
        return source
    elif isinstance(source, str):
        if _DATA_URL_PATTERN.match(source):
            parts = source.split(",", 1)
            if len(parts) < 2 or not parts[1]:
                raise ValueError(f"Malformed data URL: missing base64 data after comma in {source[:60]}...")
            try:
                return base64.b64decode(parts[1])
            except binascii.Error as e:
                raise ValueError(f"Invalid base64 data in data URL: {e}") from e
        if _HTTP_URL_PATTERN.match(source):
            return _download_http(source)
        if _looks_like_file_path(source):
            with open(source, "rb") as f:
                return f.read()
        # Assume raw base64 string
        try:
            return base64.b64decode(source)
        except binascii.Error as e:
            raise ValueError(f"Invalid base64 string: {e}") from e
    elif PIL_AVAILABLE and isinstance(source, Image.Image):
        buffer = BytesIO()
        fmt = (image_data.format or "png").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        source.save(buffer, format=fmt)
        return buffer.getvalue()
    else:
        raise ValueError(f"Cannot convert to bytes: {type(source)}")


# =============================================================================
# Type alias for any media (currently only ImageData)
# =============================================================================

# NOTE: Currently only ImageData is implemented. Future media types
# (VideoData, AudioData, PDFData, DocumentData) can be added to this type alias.
AnyMediaData = ImageData  # Future: ImageData | VideoData | AudioData | ...


# =============================================================================
# MultimodalInput
# =============================================================================


class MultimodalInput(BaseModel):
    """Input structure for multimodal evaluations.

    Attributes:
        media: Media data (image, video, audio, pdf, or document). Can be a single
               MediaData object, a list of MediaData objects, or a file path string.
               Currently only ImageData is supported.
        instruction: Text instruction or query about the media.
        context: Additional context or system prompts (optional).
    """

    model_config = {"arbitrary_types_allowed": True}

    media: AnyMediaData | list[AnyMediaData] | str
    instruction: str
    context: str | None = None
