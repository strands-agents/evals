"""Tests for multimodal types and data structures."""

import base64
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from strands_evals.types.multimodal import (
    ImageData,
    MultimodalInput,
    _detect_format,
    _looks_like_file_path,
    resolve_image_bytes,
)

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def sample_png_bytes():
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def sample_png_base64(sample_png_bytes):
    """Base64-encoded PNG."""
    return base64.b64encode(sample_png_bytes).decode("utf-8")


@pytest.fixture
def sample_data_url(sample_png_base64):
    """Data URL with PNG."""
    return f"data:image/png;base64,{sample_png_base64}"


@pytest.fixture
def temp_image_file(sample_png_bytes):
    """Temporary PNG file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(sample_png_bytes)
        f.flush()
        yield f.name
    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# Tests for _looks_like_file_path
# =============================================================================


class TestLooksLikeFilePath:
    def test_absolute_path(self):
        assert _looks_like_file_path("/path/to/image.png") is True

    def test_relative_path_dot(self):
        assert _looks_like_file_path("./image.png") is True

    def test_relative_path_dotdot(self):
        assert _looks_like_file_path("../image.png") is True

    def test_home_path(self):
        assert _looks_like_file_path("~/images/photo.jpg") is True

    def test_windows_path(self):
        assert _looks_like_file_path("C:\\Users\\image.png") is True

    def test_image_extension_only(self):
        assert _looks_like_file_path("image.png") is True
        assert _looks_like_file_path("photo.jpg") is True
        assert _looks_like_file_path("picture.jpeg") is True
        assert _looks_like_file_path("animation.gif") is True
        assert _looks_like_file_path("modern.webp") is True

    def test_base64_string(self):
        # Base64 strings without file extensions should not look like file paths
        assert _looks_like_file_path("aGVsbG8gd29ybGQ=") is False

    def test_random_string(self):
        assert _looks_like_file_path("not_a_path") is False


# =============================================================================
# Tests for _detect_format
# =============================================================================


class TestDetectFormat:
    def test_data_url_png(self):
        assert _detect_format("data:image/png;base64,abc123") == "png"

    def test_data_url_jpeg(self):
        assert _detect_format("data:image/jpeg;base64,abc123") == "jpeg"

    def test_data_url_gif(self):
        assert _detect_format("data:image/gif;base64,abc123") == "gif"

    def test_data_url_webp(self):
        assert _detect_format("data:image/webp;base64,abc123") == "webp"

    def test_file_path_png(self):
        assert _detect_format("/path/to/image.png") == "png"

    def test_file_path_jpg(self):
        assert _detect_format("/path/to/image.jpg") == "jpeg"

    def test_file_path_jpeg(self):
        assert _detect_format("/path/to/image.jpeg") == "jpeg"

    def test_http_url_returns_none(self):
        # HTTP URLs don't have format detection from URL alone
        assert _detect_format("https://example.com/image.png") is None

    def test_bytes_returns_none(self):
        assert _detect_format(b"not a string") is None

    def test_unknown_format_returns_none(self):
        assert _detect_format("data:image/bmp;base64,abc123") is None


# =============================================================================
# Tests for ImageData
# =============================================================================


class TestImageData:
    def test_init_with_bytes(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes)
        assert image.source == sample_png_bytes
        assert image.format is None  # Cannot detect from bytes without parsing

    def test_init_with_base64(self, sample_png_base64):
        image = ImageData(source=sample_png_base64)
        assert image.source == sample_png_base64

    def test_init_with_data_url(self, sample_data_url):
        image = ImageData(source=sample_data_url)
        assert image.source == sample_data_url
        assert image.format == "png"
        assert image.media_type == "image/png"

    def test_init_with_file_path(self, temp_image_file):
        image = ImageData(source=temp_image_file)
        assert image.source == temp_image_file
        assert image.format == "png"
        assert image.media_type == "image/png"

    def test_init_with_path_object(self, temp_image_file):
        image = ImageData(source=Path(temp_image_file))
        assert image.source == temp_image_file

    def test_init_with_http_url(self):
        url = "https://example.com/image.png"
        image = ImageData(source=url)
        assert image.source == url

    def test_init_with_explicit_format(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes, format="jpeg")
        assert image.format == "jpeg"
        assert image.media_type == "image/jpeg"

    def test_init_file_not_found_raises(self):
        with pytest.raises(ValueError, match="Image file not found"):
            ImageData(source="/nonexistent/path/image.png")

    def test_init_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported source type"):
            ImageData(source=12345)

    def test_to_bytes_from_bytes(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes)
        assert image.to_bytes() == sample_png_bytes

    def test_to_bytes_from_base64(self, sample_png_bytes, sample_png_base64):
        image = ImageData(source=sample_png_base64)
        assert image.to_bytes() == sample_png_bytes

    def test_to_bytes_from_data_url(self, sample_png_bytes, sample_data_url):
        image = ImageData(source=sample_data_url)
        assert image.to_bytes() == sample_png_bytes

    def test_to_bytes_from_file(self, sample_png_bytes, temp_image_file):
        image = ImageData(source=temp_image_file)
        assert image.to_bytes() == sample_png_bytes

    def test_to_base64(self, sample_png_bytes, sample_png_base64):
        image = ImageData(source=sample_png_bytes)
        assert image.to_base64() == sample_png_base64

    def test_to_data_url(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes, format="png")
        data_url = image.to_data_url()
        assert data_url.startswith("data:image/png;base64,")

    def test_model_dump_with_bytes(self, sample_png_bytes, sample_png_base64):
        """Verify that model_dump converts bytes to base64."""
        image = ImageData(source=sample_png_bytes, format="png")
        dumped = image.model_dump()
        assert dumped["source"] == sample_png_base64
        assert dumped["format"] == "png"



# =============================================================================
# Tests for resolve_image_bytes
# =============================================================================


class TestResolveImageBytes:
    def test_resolve_from_bytes(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes)
        assert resolve_image_bytes(image) == sample_png_bytes

    def test_resolve_from_base64(self, sample_png_bytes, sample_png_base64):
        image = ImageData(source=sample_png_base64)
        assert resolve_image_bytes(image) == sample_png_bytes

    def test_resolve_from_data_url(self, sample_png_bytes, sample_data_url):
        image = ImageData(source=sample_data_url)
        assert resolve_image_bytes(image) == sample_png_bytes

    def test_resolve_from_file(self, sample_png_bytes, temp_image_file):
        image = ImageData(source=temp_image_file)
        assert resolve_image_bytes(image) == sample_png_bytes

    def test_resolve_malformed_data_url_raises(self):
        image = ImageData.model_construct(
            source="data:image/png;base64,",
            format="png",
            media_type="image/png",
        )
        with pytest.raises(ValueError, match="Malformed data URL"):
            resolve_image_bytes(image)

    def test_resolve_invalid_base64_raises(self):
        image = ImageData.model_construct(
            source="not_valid_base64!!!",
            format=None,
            media_type=None,
        )
        with pytest.raises(ValueError, match="Invalid base64 string"):
            resolve_image_bytes(image)

    @patch("strands_evals.types.multimodal._download_http")
    def test_resolve_from_http_url(self, mock_download, sample_png_bytes):
        mock_download.return_value = sample_png_bytes
        image = ImageData(source="https://example.com/image.png")
        result = resolve_image_bytes(image)
        assert result == sample_png_bytes
        mock_download.assert_called_once_with("https://example.com/image.png")


# =============================================================================
# Tests for MultimodalInput
# =============================================================================


class TestMultimodalInput:
    def test_init_with_image_data(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes)
        input_data = MultimodalInput(media=image, instruction="Describe this image")
        assert input_data.media == image
        assert input_data.instruction == "Describe this image"
        assert input_data.context is None

    def test_init_with_file_path_string(self, temp_image_file):
        input_data = MultimodalInput(media=temp_image_file, instruction="What is in this image?")
        assert input_data.media == temp_image_file
        assert input_data.instruction == "What is in this image?"

    def test_init_with_context(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes)
        input_data = MultimodalInput(
            media=image,
            instruction="Describe this image",
            context="You are an image analyst.",
        )
        assert input_data.context == "You are an image analyst."

    def test_init_with_multiple_images(self, sample_png_bytes):
        image1 = ImageData(source=sample_png_bytes)
        image2 = ImageData(source=sample_png_bytes)
        input_data = MultimodalInput(media=[image1, image2], instruction="Compare these images")
        assert isinstance(input_data.media, list)
        assert len(input_data.media) == 2
