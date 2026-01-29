from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import OutputEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = EvaluationOutput(score=0.8, test_pass=True, reason="Mock evaluation result")
    agent.return_value = mock_result
    return agent


@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()

    # Create a mock coroutine function
    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(
            score=0.8, test_pass=True, reason="Mock async evaluation result"
        )
        return mock_result

    agent.invoke_async = mock_invoke_async
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(input="What is 2+2?", actual_output="4", expected_output="4", name="math_test")


def test_output_evaluator__init__with_defaults():
    """Test OutputEvaluator initialization with default values"""
    evaluator = OutputEvaluator(rubric="Test rubric")

    assert evaluator.rubric == "Test rubric"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None  # Uses default template


def test_output_evaluator__init__with_custom_values():
    """Test OutputEvaluator initialization with custom values"""
    custom_prompt = "Custom system prompt"
    evaluator = OutputEvaluator(
        rubric="Custom rubric", model="gpt-4", system_prompt=custom_prompt, include_inputs=False
    )

    assert evaluator.rubric == "Custom rubric"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == custom_prompt


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_with_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation with inputs included (the default behavior) and trajectory should not be included"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric")

    result = evaluator.evaluate(evaluation_data)

    # Verify Agent was created with correct parameters
    mock_agent_class.assert_called_once_with(model=None, system_prompt=evaluator.system_prompt, callback_handler=None)

    # Verify agent was called
    mock_agent.assert_called_once()
    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert call_args[1]["structured_output_model"] == EvaluationOutput
    assert "<Input>What is 2+2?</Input>" in prompt
    assert "<Trajectory>" not in prompt
    assert "<ExpectedTrajectory>" not in prompt
    assert "<Output>4</Output>" in prompt
    assert "<ExpectedOutput>4</ExpectedOutput>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_without_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation without inputs included and trajectory should not be included"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric", include_inputs=False)

    result = evaluator.evaluate(evaluation_data)

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert "<Input>" not in prompt
    assert "<Trajectory>" not in prompt
    assert "<ExpectedTrajectory>" not in prompt
    assert "<Output>4</Output>" in prompt
    assert "<ExpectedOutput>4</ExpectedOutput>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_output_evaluator_evaluate_without_expected_output(mock_agent_class, mock_agent):
    """Test evaluation without expected output"""
    mock_agent_class.return_value = mock_agent
    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_output="result",
    )

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert "<ExpectedOutput>" not in prompt
    assert "<Output>result</Output>" in prompt


def test_output_evaluator_evaluate_missing_actual_output():
    """Test evaluation raises exception when actual_output is missing"""
    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", expected_output="expected")

    with pytest.raises(Exception, match="Please make sure the task function return the output"):
        evaluator.evaluate(evaluation_data)


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_with_inputs(mock_agent_class, evaluation_data, mock_async_agent):
    """Test async evaluation with inputs included"""
    mock_agent_class.return_value = mock_async_agent
    evaluator = OutputEvaluator(rubric="Test rubric")

    result = await evaluator.evaluate_async(evaluation_data)

    # Verify Agent was created with correct parameters
    mock_agent_class.assert_called_once_with(model=None, system_prompt=evaluator.system_prompt, callback_handler=None)

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True
    assert result[0].reason == "Mock async evaluation result"


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_output_evaluator_evaluate_async_without_inputs(mock_agent_class, evaluation_data, mock_async_agent):
    """Test async evaluation without inputs included"""
    mock_agent_class.return_value = mock_async_agent
    evaluator = OutputEvaluator(rubric="Test rubric", include_inputs=False)

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 0.8
    assert result[0].test_pass is True


@pytest.mark.asyncio
async def test_output_evaluator_evaluate_async_missing_actual_output():
    """Test async evaluation raises exception when actual_output is missing"""
    evaluator = OutputEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", expected_output="expected")

    with pytest.raises(Exception, match="Please make sure the task function return the output"):
        await evaluator.evaluate_async(evaluation_data)
