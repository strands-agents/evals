"""Multimodal evaluation types and data structures."""

import base64
import binascii
import logging
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, field_validator, model_serializer
from typing_extensions import NotRequired, TypedDict

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)

# Patterns for distinguishing source types
_S3_URI_PATTERN = re.compile(r"^s3://")
_HTTP_URL_PATTERN = re.compile(r"^https?://")
_DATA_URL_PATTERN = re.compile(r"^data:[^;]+;base64,")

_S3_URI_PARSE_PATTERN = re.compile(r"^s3://([^/]+)/(.+)$")

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


def _detect_format(source: Any) -> Optional[str]:
    """Detect image format from the source value."""
    if not isinstance(source, str):
        return None
    if _DATA_URL_PATTERN.match(source):
        media_part = source.split(";")[0]
        fmt = media_part.split("/")[-1].lower()
        return fmt if fmt in _IMAGE_FORMATS else None
    is_local = _looks_like_file_path(source) and not _S3_URI_PATTERN.match(source)
    is_local = is_local and not _HTTP_URL_PATTERN.match(source)
    if is_local:
        return _IMAGE_FORMAT_FROM_EXT.get(Path(source).suffix.lower())
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


def _download_s3(uri: str) -> bytes:
    """Download media from an S3 URI.

    Requires ``boto3`` to be installed.

    Raises:
        ImportError: If ``boto3`` is not installed.
        ValueError: If the URI is malformed or the download fails.
    """
    if not BOTO3_AVAILABLE:
        raise ImportError(f"boto3 is required to download from S3 URIs ({uri}). Install it with: pip install boto3")
    match = _S3_URI_PARSE_PATTERN.match(uri)
    if not match:
        raise ValueError(f"Malformed S3 URI: {uri}. Expected format: s3://bucket/key")
    bucket, key = match.group(1), match.group(2)
    try:
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except Exception as e:
        raise ValueError(f"Failed to download from {uri}: {e}") from e


class ImageData(BaseModel):
    """Represents image data that can be passed to multimodal evaluators.

    Supports multiple input formats:
    - File path (str or Path): local file system path to an image
    - Base64 encoded string: raw base64-encoded image bytes
    - Data URL: ``data:image/<format>;base64,<data>``
    - HTTP/HTTPS URL: auto-fetched via ``urllib.request`` (stdlib)
    - S3 URI: auto-fetched via ``boto3`` (optional dependency)
    - PIL Image object (if PIL is available)
    - Raw bytes
    """

    model_config = {"arbitrary_types_allowed": True}

    source: Union[str, bytes, Any]
    format: Optional[Literal["jpeg", "png", "gif", "webp"]] = None
    media_type: Optional[str] = None

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v: Any) -> Any:
        """Validate and normalize image source data."""
        if isinstance(v, Path):
            v = str(v)

        if isinstance(v, str):
            if _DATA_URL_PATTERN.match(v):
                return v
            if _S3_URI_PATTERN.match(v):
                return v
            if _HTTP_URL_PATTERN.match(v):
                return v
            if _looks_like_file_path(v):
                path = Path(v)
                if not path.exists():
                    raise ValueError(
                        f"Image file not found: {path}. "
                        f"Supported sources: local file paths, base64, data URLs, S3 URIs, HTTP URLs."
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
                f"Supported: file path, base64 string, data URL, S3 URI, HTTP URL, bytes, PIL Image."
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
        if isinstance(self.source, bytes):
            return self.source
        elif isinstance(self.source, str):
            if _DATA_URL_PATTERN.match(self.source):
                parts = self.source.split(",", 1)
                if len(parts) < 2 or not parts[1]:
                    raise ValueError(f"Malformed data URL: missing base64 data after comma in {self.source[:60]}...")
                try:
                    return base64.b64decode(parts[1])
                except binascii.Error as e:
                    raise ValueError(f"Invalid base64 data in data URL: {e}") from e
            if _HTTP_URL_PATTERN.match(self.source):
                return _download_http(self.source)
            if _S3_URI_PATTERN.match(self.source):
                return _download_s3(self.source)
            if _looks_like_file_path(self.source):
                with open(self.source, "rb") as f:
                    return f.read()
            # Assume raw base64 string
            try:
                return base64.b64decode(self.source)
            except binascii.Error as e:
                raise ValueError(f"Invalid base64 string: {e}") from e
        elif PIL_AVAILABLE and isinstance(self.source, Image.Image):
            buffer = BytesIO()
            fmt = (self.format or "png").upper()
            if fmt == "JPG":
                fmt = "JPEG"
            self.source.save(buffer, format=fmt)
            return buffer.getvalue()
        else:
            raise ValueError(f"Cannot convert to bytes: {type(self.source)}")

    def to_base64(self) -> str:
        """Convert image to base64 string."""
        return base64.b64encode(self.to_bytes()).decode("utf-8")

    def to_data_url(self) -> str:
        """Convert image to data URL format."""
        base64_data = self.to_base64()
        media_type = self.media_type or "image/png"
        return f"data:{media_type};base64,{base64_data}"


# =============================================================================
# Union type for any media (currently only ImageData)
# =============================================================================

# NOTE: Currently only ImageData is implemented. Future media types
# (VideoData, AudioData, PDFData, DocumentData) can be added to this union.
AnyMediaData = Union[ImageData]


# =============================================================================
# MultimodalInput TypedDict
# =============================================================================


class MultimodalInput(TypedDict):
    """Input structure for multimodal evaluations.

    Attributes:
        media: Media data (image, video, audio, pdf, or document). Can be a single
               MediaData object, a list of MediaData objects, or a file path string.
               Currently only ImageData is supported.
        instruction: Text instruction or query about the media.
        context: Additional context or system prompts (optional).
    """

    media: Union[AnyMediaData, List[AnyMediaData], str]
    instruction: str
    context: NotRequired[Optional[str]]
