"""Tests for MultimodalOutputEvaluator."""

import warnings
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import MultimodalOutputEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput
from strands_evals.types.multimodal import ImageData, MultimodalInput

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
def image_data(sample_png_bytes):
    """Sample ImageData fixture."""
    return ImageData(source=sample_png_bytes, format="png")


@pytest.fixture
def multimodal_input(image_data):
    """Sample MultimodalInput fixture."""
    return MultimodalInput(media=image_data, instruction="Describe this image")


@pytest.fixture
def evaluation_data_with_reference(multimodal_input):
    """EvaluationData with expected_output (reference-based)."""
    return EvaluationData(
        input=multimodal_input,
        actual_output="A small blue square",
        expected_output="A blue square",
        name="test_case",
    )


@pytest.fixture
def evaluation_data_without_reference(multimodal_input):
    """EvaluationData without expected_output (reference-free)."""
    return EvaluationData(
        input=multimodal_input,
        actual_output="A small blue square",
        name="test_case",
    )


@pytest.fixture
def mock_agent():
    """Mock Agent for testing."""
    agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = EvaluationOutput(
        score=0.85, test_pass=True, reason="Good description"
    )
    agent.return_value = mock_result
    return agent


@pytest.fixture
def mock_async_agent():
    """Mock Agent for async testing."""
    agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(
            score=0.85, test_pass=True, reason="Good async description"
        )
        return mock_result

    agent.invoke_async = mock_invoke_async
    return agent


# =============================================================================
# Tests for initialization
# =============================================================================


class TestMultimodalOutputEvaluatorInit:
    def test_init_with_defaults(self):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric")

        assert evaluator.rubric == "Test rubric"
        assert evaluator.model is None
        assert evaluator.include_media is True
        assert evaluator.include_inputs is True
        assert evaluator.system_prompt is not None
        assert evaluator.reference_suffix == MultimodalOutputEvaluator.DEFAULT_REFERENCE_SUFFIX

    def test_init_with_custom_values(self):
        custom_prompt = "Custom system prompt"
        custom_suffix = "\nCustom reference suffix"

        evaluator = MultimodalOutputEvaluator(
            rubric="Custom rubric",
            model="claude-3-sonnet",
            include_media=False,
            include_inputs=False,
            system_prompt=custom_prompt,
            reference_suffix=custom_suffix,
        )

        assert evaluator.rubric == "Custom rubric"
        assert evaluator.model == "claude-3-sonnet"
        assert evaluator.include_media is False
        assert evaluator.include_inputs is False
        assert evaluator.system_prompt == custom_prompt
        assert evaluator.reference_suffix == custom_suffix

    def test_init_llm_mode(self):
        """Test LLM-as-a-Judge mode (no media)."""
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=False)
        assert evaluator.include_media is False


# =============================================================================
# Tests for _select_rubric
# =============================================================================


class TestSelectRubric:
    def test_select_rubric_with_reference(self, evaluation_data_with_reference):
        evaluator = MultimodalOutputEvaluator(rubric="Base rubric")
        result = evaluator._select_rubric(evaluation_data_with_reference)

        assert result == "Base rubric" + evaluator.reference_suffix
        assert "REFERENCE COMPARISON" in result

    def test_select_rubric_without_reference(self, evaluation_data_without_reference):
        evaluator = MultimodalOutputEvaluator(rubric="Base rubric")
        result = evaluator._select_rubric(evaluation_data_without_reference)

        assert result == "Base rubric"
        assert "REFERENCE COMPARISON" not in result

    def test_select_rubric_custom_suffix(self, evaluation_data_with_reference):
        custom_suffix = "\nCustom comparison instructions"
        evaluator = MultimodalOutputEvaluator(rubric="Base rubric", reference_suffix=custom_suffix)
        result = evaluator._select_rubric(evaluation_data_with_reference)

        assert result == "Base rubric" + custom_suffix


# =============================================================================
# Tests for _build_prompt
# =============================================================================


class TestBuildPrompt:
    def test_build_prompt_with_media_returns_list(self, evaluation_data_with_reference):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=True)
        result = evaluator._build_prompt(evaluation_data_with_reference)

        assert isinstance(result, list)
        # Should have image block(s) + text block
        assert len(result) >= 2
        # Last block should be text
        assert "text" in result[-1]
        # First block should be image
        assert "image" in result[0]

    def test_build_prompt_without_media_returns_string(self, evaluation_data_with_reference):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=False)
        result = evaluator._build_prompt(evaluation_data_with_reference)

        assert isinstance(result, str)
        assert "<Output>" in result
        assert "<ExpectedOutput>" in result
        assert "<Rubric>" in result

    def test_build_prompt_includes_input_when_enabled(self, evaluation_data_with_reference):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_inputs=True, include_media=False)
        result = evaluator._build_prompt(evaluation_data_with_reference)

        assert "<Input>" in result
        assert "Describe this image" in result

    def test_build_prompt_excludes_input_when_disabled(self, evaluation_data_with_reference):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_inputs=False, include_media=False)
        result = evaluator._build_prompt(evaluation_data_with_reference)

        assert "<Input>" not in result

    def test_build_prompt_includes_reference_suffix_when_expected_output_present(
        self, evaluation_data_with_reference
    ):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=False)
        result = evaluator._build_prompt(evaluation_data_with_reference)

        assert "REFERENCE COMPARISON" in result

    def test_build_prompt_excludes_reference_suffix_when_no_expected_output(
        self, evaluation_data_without_reference
    ):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=False)
        result = evaluator._build_prompt(evaluation_data_without_reference)

        assert "REFERENCE COMPARISON" not in result

    def test_build_prompt_warns_when_media_enabled_but_empty(self):
        """Test warning when include_media=True but no media in input."""
        empty_media_input = MultimodalInput(media=[], instruction="Test")
        evaluation_data = EvaluationData(
            input=empty_media_input,
            actual_output="Test output",
        )
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evaluator._build_prompt(evaluation_data)
            # Check if warning was raised
            assert any("no media found" in str(warning.message).lower() for warning in w)


# =============================================================================
# Tests for evaluate
# =============================================================================


class TestEvaluate:
    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate_with_multimodal_input(
        self, mock_agent_class, evaluation_data_with_reference, mock_agent
    ):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric")

        result = evaluator.evaluate(evaluation_data_with_reference)

        mock_agent_class.assert_called_once()
        mock_agent.assert_called_once()
        assert len(result) == 1
        assert result[0].score == 0.85
        assert result[0].test_pass is True

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate_passes_content_blocks_to_agent(
        self, mock_agent_class, evaluation_data_with_reference, mock_agent
    ):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=True)

        evaluator.evaluate(evaluation_data_with_reference)

        call_args = mock_agent.call_args
        prompt = call_args[0][0]
        # When include_media=True, prompt should be a list of content blocks
        assert isinstance(prompt, list)

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate_llm_mode_passes_string(
        self, mock_agent_class, evaluation_data_with_reference, mock_agent
    ):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric", include_media=False)

        evaluator.evaluate(evaluation_data_with_reference)

        call_args = mock_agent.call_args
        prompt = call_args[0][0]
        # When include_media=False, prompt should be a string
        assert isinstance(prompt, str)


# =============================================================================
# Tests for evaluate_async
# =============================================================================


class TestEvaluateAsync:
    @pytest.mark.asyncio
    @patch("strands_evals.evaluators.output_evaluator.Agent")
    async def test_evaluate_async_with_multimodal_input(
        self, mock_agent_class, evaluation_data_with_reference, mock_async_agent
    ):
        mock_agent_class.return_value = mock_async_agent
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric")

        result = await evaluator.evaluate_async(evaluation_data_with_reference)

        assert len(result) == 1
        assert result[0].score == 0.85
        assert result[0].test_pass is True


# =============================================================================
# Tests for error handling
# =============================================================================


class TestErrorHandling:
    def test_evaluate_missing_actual_output_raises(self, multimodal_input):
        evaluator = MultimodalOutputEvaluator(rubric="Test rubric")
        evaluation_data = EvaluationData(
            input=multimodal_input,
            expected_output="Expected",
        )

        with pytest.raises(Exception, match="Please make sure the task function return the output"):
            evaluator.evaluate(evaluation_data)
