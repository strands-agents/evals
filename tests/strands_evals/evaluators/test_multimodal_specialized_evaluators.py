"""Tests for the four specialized multimodal evaluator subclasses."""

from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import (
    MultimodalCorrectnessEvaluator,
    MultimodalFaithfulnessEvaluator,
    MultimodalInstructionFollowingEvaluator,
    MultimodalOutputEvaluator,
    MultimodalOverallQualityEvaluator,
)
from strands_evals.evaluators.prompt_templates.multimodal import (
    CORRECTNESS_RUBRIC_V0,
    FAITHFULNESS_RUBRIC_V0,
    INSTRUCTION_FOLLOWING_RUBRIC_V0,
    OVERALL_QUALITY_RUBRIC_V0,
)
from strands_evals.types import EvaluationData, EvaluationOutput
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
def evaluation_data(sample_png_bytes):
    image = ImageData(source=sample_png_bytes, format="png")
    return EvaluationData(
        input=MultimodalInput(media=image, instruction="Describe this image"),
        actual_output="A small pixel",
        expected_output="A red pixel",
        name="test_case",
    )


@pytest.fixture
def mock_agent():
    agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = EvaluationOutput(score=0.9, test_pass=True, reason="Accurate")
    agent.return_value = mock_result
    return agent


# =============================================================================
# MultimodalCorrectnessEvaluator
# =============================================================================


class TestMultimodalCorrectnessEvaluator:
    def test_init_defaults(self):
        evaluator = MultimodalCorrectnessEvaluator()

        assert evaluator.rubric == CORRECTNESS_RUBRIC_V0
        assert evaluator.model is None
        assert evaluator.include_media is True
        assert evaluator.include_inputs is True
        assert isinstance(evaluator, MultimodalOutputEvaluator)

    def test_init_custom_rubric(self):
        evaluator = MultimodalCorrectnessEvaluator(rubric="Custom correctness rubric")
        assert evaluator.rubric == "Custom correctness rubric"

    def test_init_custom_model(self):
        evaluator = MultimodalCorrectnessEvaluator(model="claude-3-sonnet")
        assert evaluator.model == "claude-3-sonnet"

    def test_init_media_disabled(self):
        evaluator = MultimodalCorrectnessEvaluator(include_media=False)
        assert evaluator.include_media is False

    def test_init_inputs_disabled(self):
        evaluator = MultimodalCorrectnessEvaluator(include_inputs=False)
        assert evaluator.include_inputs is False

    def test_init_custom_system_prompt(self):
        evaluator = MultimodalCorrectnessEvaluator(system_prompt="Custom prompt")
        assert evaluator.system_prompt == "Custom prompt"

    def test_uses_default_reference_suffix(self):
        evaluator = MultimodalCorrectnessEvaluator()
        assert evaluator.reference_suffix == MultimodalOutputEvaluator.DEFAULT_REFERENCE_SUFFIX

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate(self, mock_agent_class, evaluation_data, mock_agent):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalCorrectnessEvaluator()

        result = evaluator.evaluate(evaluation_data)

        assert len(result) == 1
        assert result[0].score == 0.9
        assert result[0].test_pass is True


# =============================================================================
# MultimodalFaithfulnessEvaluator
# =============================================================================


class TestMultimodalFaithfulnessEvaluator:
    def test_init_defaults(self):
        evaluator = MultimodalFaithfulnessEvaluator()

        assert evaluator.rubric == FAITHFULNESS_RUBRIC_V0
        assert evaluator.model is None
        assert evaluator.include_media is True
        assert evaluator.include_inputs is True
        assert isinstance(evaluator, MultimodalOutputEvaluator)

    def test_init_custom_rubric(self):
        evaluator = MultimodalFaithfulnessEvaluator(rubric="Custom faithfulness rubric")
        assert evaluator.rubric == "Custom faithfulness rubric"

    def test_init_custom_model(self):
        evaluator = MultimodalFaithfulnessEvaluator(model="claude-3-sonnet")
        assert evaluator.model == "claude-3-sonnet"

    def test_uses_default_reference_suffix(self):
        evaluator = MultimodalFaithfulnessEvaluator()
        assert evaluator.reference_suffix == MultimodalOutputEvaluator.DEFAULT_REFERENCE_SUFFIX

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate(self, mock_agent_class, evaluation_data, mock_agent):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalFaithfulnessEvaluator()

        result = evaluator.evaluate(evaluation_data)

        assert len(result) == 1
        assert result[0].score == 0.9


# =============================================================================
# MultimodalInstructionFollowingEvaluator
# =============================================================================


class TestMultimodalInstructionFollowingEvaluator:
    def test_init_defaults(self):
        evaluator = MultimodalInstructionFollowingEvaluator()

        assert evaluator.rubric == INSTRUCTION_FOLLOWING_RUBRIC_V0
        assert evaluator.model is None
        assert evaluator.include_media is True
        assert evaluator.include_inputs is True
        assert isinstance(evaluator, MultimodalOutputEvaluator)

    def test_init_custom_rubric(self):
        evaluator = MultimodalInstructionFollowingEvaluator(rubric="Custom IF rubric")
        assert evaluator.rubric == "Custom IF rubric"

    def test_uses_default_reference_suffix(self):
        evaluator = MultimodalInstructionFollowingEvaluator()
        assert evaluator.reference_suffix == MultimodalOutputEvaluator.DEFAULT_REFERENCE_SUFFIX

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate(self, mock_agent_class, evaluation_data, mock_agent):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalInstructionFollowingEvaluator()

        result = evaluator.evaluate(evaluation_data)

        assert len(result) == 1
        assert result[0].score == 0.9


# =============================================================================
# MultimodalOverallQualityEvaluator
# =============================================================================


class TestMultimodalOverallQualityEvaluator:
    def test_init_defaults(self):
        evaluator = MultimodalOverallQualityEvaluator()

        assert evaluator.rubric == OVERALL_QUALITY_RUBRIC_V0
        assert evaluator.model is None
        assert evaluator.include_media is True
        assert evaluator.include_inputs is True
        assert isinstance(evaluator, MultimodalOutputEvaluator)

    def test_init_custom_rubric(self):
        evaluator = MultimodalOverallQualityEvaluator(rubric="Custom quality rubric")
        assert evaluator.rubric == "Custom quality rubric"

    def test_uses_custom_reference_suffix(self):
        """OverallQuality has its own reference suffix, not the default."""
        evaluator = MultimodalOverallQualityEvaluator()
        assert evaluator.reference_suffix != MultimodalOutputEvaluator.DEFAULT_REFERENCE_SUFFIX
        assert "gold standard" in evaluator.reference_suffix
        assert "key points" in evaluator.reference_suffix

    def test_reference_suffix_appended_when_expected_output_present(self, evaluation_data):
        evaluator = MultimodalOverallQualityEvaluator()
        rubric = evaluator._select_rubric(evaluation_data)

        assert rubric.startswith(OVERALL_QUALITY_RUBRIC_V0)
        assert "REFERENCE COMPARISON" in rubric
        assert "gold standard" in rubric

    def test_reference_suffix_not_appended_without_expected_output(self, sample_png_bytes):
        image = ImageData(source=sample_png_bytes, format="png")
        eval_data = EvaluationData(
            input=MultimodalInput(media=image, instruction="Describe"),
            actual_output="A pixel",
            name="test",
        )
        evaluator = MultimodalOverallQualityEvaluator()
        rubric = evaluator._select_rubric(eval_data)

        assert rubric == OVERALL_QUALITY_RUBRIC_V0
        assert "REFERENCE COMPARISON" not in rubric

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_evaluate(self, mock_agent_class, evaluation_data, mock_agent):
        mock_agent_class.return_value = mock_agent
        evaluator = MultimodalOverallQualityEvaluator()

        result = evaluator.evaluate(evaluation_data)

        assert len(result) == 1
        assert result[0].score == 0.9
