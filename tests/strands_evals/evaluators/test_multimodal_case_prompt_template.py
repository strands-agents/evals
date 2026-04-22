"""Tests for multimodal_case_prompt_template.py."""

import pytest

from strands_evals.evaluators.prompt_templates.multimodal_case_prompt_template import (
    _build_media_content_block,
    _resolve_media_data,
    compose_multimodal_test_prompt,
)
from strands_evals.types import EvaluationData
from strands_evals.types.multimodal import ImageData, MultimodalInput


@pytest.fixture
def sample_png_bytes():
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def image_data(sample_png_bytes):
    return ImageData(source=sample_png_bytes, format="png")


# =============================================================================
# Tests for _resolve_media_data
# =============================================================================


class TestResolveMediaData:
    def test_resolve_image_data_passthrough(self, image_data):
        result = _resolve_media_data(image_data)
        assert result is image_data

    def test_resolve_string_creates_image_data(self, tmp_path, sample_png_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_png_bytes)

        result = _resolve_media_data(str(img_path))
        assert isinstance(result, ImageData)
        assert result.to_bytes() == sample_png_bytes

    def test_resolve_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Expected ImageData or str"):
            _resolve_media_data(12345)

    def test_resolve_list_raises(self):
        with pytest.raises(ValueError, match="Expected ImageData or str"):
            _resolve_media_data([1, 2, 3])


# =============================================================================
# Tests for _build_media_content_block
# =============================================================================


class TestBuildMediaContentBlock:
    def test_image_content_block_format(self, image_data):
        block = _build_media_content_block(image_data)

        assert "image" in block
        assert block["image"]["format"] == "png"
        assert "bytes" in block["image"]["source"]
        assert isinstance(block["image"]["source"]["bytes"], bytes)

    def test_image_defaults_to_png_when_no_format(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes)
        block = _build_media_content_block(image)

        assert block["image"]["format"] == "png"

    def test_image_uses_specified_format(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes, format="jpeg")
        block = _build_media_content_block(image)

        assert block["image"]["format"] == "jpeg"


# =============================================================================
# Tests for compose_multimodal_test_prompt
# =============================================================================


class TestComposeMultimodalTestPrompt:
    def test_text_only_returns_string(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Describe this"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Test rubric", include_media=False)

        assert isinstance(result, str)
        assert "<Output>A pixel</Output>" in result
        assert "<Rubric>Test rubric</Rubric>" in result

    def test_with_media_returns_list(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Describe this"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Test rubric", include_media=True)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "image" in result[0]
        assert "text" in result[-1]

    def test_multiple_media_items(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(
                media=[image_data, image_data],
                instruction="Compare these",
            ),
            actual_output="They are the same",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Test rubric", include_media=True)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "image" in result[0]
        assert "image" in result[1]
        assert "text" in result[2]

    def test_includes_instruction_when_enabled(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Tell me about this"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_inputs=True, include_media=False)

        assert "<Input>Tell me about this</Input>" in result

    def test_excludes_instruction_when_disabled(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Tell me about this"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_inputs=False, include_media=False)

        assert "<Input>" not in result

    def test_includes_context_in_input(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(
                media=image_data,
                instruction="Analyze this",
                context="You are a doctor",
            ),
            actual_output="Medical image",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_inputs=True, include_media=False)

        assert "Context: You are a doctor" in result
        assert "Instruction: Analyze this" in result

    def test_includes_expected_output(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Describe"),
            actual_output="A pixel",
            expected_output="A red pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_media=False)

        assert "<ExpectedOutput>A red pixel</ExpectedOutput>" in result

    def test_excludes_expected_output_when_none(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Describe"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_media=False)

        assert "<ExpectedOutput>" not in result

    def test_raises_when_actual_output_missing(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Describe"),
            name="test",
        )
        with pytest.raises(ValueError, match="task function return the output"):
            compose_multimodal_test_prompt(eval_case, rubric="Rubric")

    def test_non_multimodal_input_fallback(self):
        eval_case = EvaluationData(
            input="plain text input",
            actual_output="Some output",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_inputs=True, include_media=True)

        assert isinstance(result, str)
        assert "<Input>plain text input</Input>" in result

    def test_empty_media_list_warns_and_falls_back(self):
        eval_case = EvaluationData(
            input=MultimodalInput(media=[], instruction="Describe"),
            actual_output="Some output",
            name="test",
        )

        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_media=True)

        assert isinstance(result, str)

    def test_string_media_resolved_to_image_data(self, tmp_path, sample_png_bytes):
        img_path = tmp_path / "test.png"
        img_path.write_bytes(sample_png_bytes)

        eval_case = EvaluationData(
            input=MultimodalInput(media=str(img_path), instruction="Describe"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_media=True)

        assert isinstance(result, list)
        assert "image" in result[0]

    def test_score_instruction_in_text(self, image_data):
        eval_case = EvaluationData(
            input=MultimodalInput(media=image_data, instruction="Describe"),
            actual_output="A pixel",
            name="test",
        )
        result = compose_multimodal_test_prompt(eval_case, rubric="Rubric", include_media=False)

        assert "DECIMAL BETWEEN 0.0 AND 1.0" in result
