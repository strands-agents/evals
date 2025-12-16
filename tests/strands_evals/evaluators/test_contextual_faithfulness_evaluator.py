from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import ContextualFaithfulnessEvaluator
from strands_evals.evaluators.contextual_faithfulness_evaluator import (
    ContextualFaithfulnessRating,
    ContextualFaithfulnessScore,
)
from strands_evals.types import EvaluationData


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="What is the company's refund policy?",
        actual_output="You can get a full refund within 30 days if the item is unopened.",
        retrieval_context=[
            "Our refund policy allows returns within 30 days of purchase.",
            "Items must be unopened and in original packaging for a full refund.",
            "Opened items may be eligible for store credit only.",
        ],
        name="refund_policy_test",
    )


@pytest.fixture
def evaluation_data_no_context():
    return EvaluationData(
        input="What is the company's refund policy?",
        actual_output="You can get a full refund within 30 days.",
        name="no_context_test",
    )


@pytest.fixture
def evaluation_data_no_output():
    return EvaluationData(
        input="What is the company's refund policy?",
        retrieval_context=["Returns allowed within 30 days."],
        name="no_output_test",
    )


def test_init_with_defaults():
    evaluator = ContextualFaithfulnessEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.include_input is True


def test_init_with_custom_values():
    evaluator = ContextualFaithfulnessEvaluator(
        version="v0",
        model="custom-model",
        system_prompt="Custom prompt",
        include_input=False,
    )

    assert evaluator.version == "v0"
    assert evaluator.model == "custom-model"
    assert evaluator.system_prompt == "Custom prompt"
    assert evaluator.include_input is False


@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_evaluate_fully_faithful(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(
        reasoning="All claims about 30-day refund and unopened items are supported by the context.",
        score=ContextualFaithfulnessScore.FULLY_FAITHFUL,
    )
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].label == ContextualFaithfulnessScore.FULLY_FAITHFUL
    assert "30-day refund" in result[0].reason


@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_evaluate_not_faithful(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(
        reasoning="The response contains fabricated information not in the context.",
        score=ContextualFaithfulnessScore.NOT_FAITHFUL,
    )
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.0
    assert result[0].test_pass is False
    assert result[0].label == ContextualFaithfulnessScore.NOT_FAITHFUL


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (ContextualFaithfulnessScore.NOT_FAITHFUL, 0.0, False),
        (ContextualFaithfulnessScore.PARTIALLY_FAITHFUL, 0.33, False),
        (ContextualFaithfulnessScore.MOSTLY_FAITHFUL, 0.67, True),
        (ContextualFaithfulnessScore.FULLY_FAITHFUL, 1.0, True),
    ],
)
@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(reasoning="Test", score=score)
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


def test_evaluate_missing_retrieval_context(evaluation_data_no_context):
    evaluator = ContextualFaithfulnessEvaluator()

    with pytest.raises(ValueError, match="retrieval_context is required"):
        evaluator.evaluate(evaluation_data_no_context)


def test_evaluate_missing_actual_output(evaluation_data_no_output):
    evaluator = ContextualFaithfulnessEvaluator()

    with pytest.raises(ValueError, match="actual_output is required"):
        evaluator.evaluate(evaluation_data_no_output)


@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_prompt_includes_input_by_default(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(
        reasoning="Test", score=ContextualFaithfulnessScore.FULLY_FAITHFUL
    )
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "# User Query:" in prompt
    assert "What is the company's refund policy?" in prompt


@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_prompt_excludes_input_when_disabled(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(
        reasoning="Test", score=ContextualFaithfulnessScore.FULLY_FAITHFUL
    )
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator(include_input=False)

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "# User Query:" not in prompt


@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_prompt_formats_context_documents(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(
        reasoning="Test", score=ContextualFaithfulnessScore.FULLY_FAITHFUL
    )
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "# Retrieval Context:" in prompt
    assert "[Document 1]" in prompt
    assert "[Document 2]" in prompt
    assert "[Document 3]" in prompt
    assert "30 days of purchase" in prompt


@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
def test_prompt_includes_response(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_agent.structured_output.return_value = ContextualFaithfulnessRating(
        reasoning="Test", score=ContextualFaithfulnessScore.FULLY_FAITHFUL
    )
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "# Assistant's Response:" in prompt
    assert "full refund within 30 days" in prompt


@pytest.mark.asyncio
@patch("strands_evals.evaluators.contextual_faithfulness_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_structured_output_async(*args, **kwargs):
        return ContextualFaithfulnessRating(
            reasoning="All claims are supported by context.",
            score=ContextualFaithfulnessScore.FULLY_FAITHFUL,
        )

    mock_agent.structured_output_async = mock_structured_output_async
    mock_agent_class.return_value = mock_agent
    evaluator = ContextualFaithfulnessEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].label == ContextualFaithfulnessScore.FULLY_FAITHFUL


@pytest.mark.asyncio
async def test_evaluate_async_missing_retrieval_context(evaluation_data_no_context):
    evaluator = ContextualFaithfulnessEvaluator()

    with pytest.raises(ValueError, match="retrieval_context is required"):
        await evaluator.evaluate_async(evaluation_data_no_context)


@pytest.mark.asyncio
async def test_evaluate_async_missing_actual_output(evaluation_data_no_output):
    evaluator = ContextualFaithfulnessEvaluator()

    with pytest.raises(ValueError, match="actual_output is required"):
        await evaluator.evaluate_async(evaluation_data_no_output)


def test_to_dict():
    evaluator = ContextualFaithfulnessEvaluator(version="v0", include_input=False)

    result = evaluator.to_dict()

    assert result["evaluator_type"] == "ContextualFaithfulnessEvaluator"
    assert result["include_input"] is False
    assert "model_id" in result


def test_to_dict_with_custom_model():
    evaluator = ContextualFaithfulnessEvaluator(model="custom-model-id")

    result = evaluator.to_dict()

    assert result["model"] == "custom-model-id"
