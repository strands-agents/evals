from unittest.mock import Mock, patch

import pytest
from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(
        score=0.9, test_pass=True, reason="Mock trajectory evaluation"
    )
    return agent


@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()

    # Create a mock coroutine function
    async def mock_structured_output_async(*args, **kwargs):
        return EvaluationOutput(score=0.9, test_pass=True, reason="Mock async trajectory evaluation")

    agent.structured_output_async = mock_structured_output_async
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="What's 2x2?",
        actual_output="2x2 is 4.",
        expected_output="2x2 is 4.",
        actual_trajectory=["calculator"],
        expected_trajectory=["calculator"],
        name="Simple math test",
    )


def test_trajectory_evaluator_init_with_defaults():
    """Test TrajectoryEvaluator initialization with default values"""
    evaluator = TrajectoryEvaluator(rubric="Test trajectory rubric")

    assert evaluator.rubric == "Test trajectory rubric"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None  # Should have a default


def test_trajectory_evaluator_init_with_custom_values():
    """Test TrajectoryEvaluator initialization with custom values"""
    custom_prompt = "Custom trajectory prompt"
    evaluator = TrajectoryEvaluator(
        rubric="Custom rubric",
        trajectory_description={"tool1": "description"},
        model="gpt-4",
        system_prompt=custom_prompt,
        include_inputs=False,
    )

    assert evaluator.rubric == "Custom rubric"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == custom_prompt


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_with_full_data(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation with complete trajectory data"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric")

    result = evaluator.evaluate(evaluation_data)

    # Verify Agent creation
    mock_agent_class.assert_called_once_with(
        model=None, system_prompt=evaluator.system_prompt, tools=evaluator._tools, callback_handler=None
    )

    # Verify structured_output call
    mock_agent.structured_output.assert_called_once()
    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]

    assert "<Input>What's 2x2?</Input>" in prompt
    assert "<Output>2x2 is 4." in prompt
    assert "<ExpectedOutput>2x2 is 4.</ExpectedOutput>" in prompt
    assert "<Trajectory>['calculator']</Trajectory>" in prompt
    assert "<ExpectedTrajectory>['calculator']</ExpectedTrajectory>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt
    assert result.score == 0.9
    assert result.test_pass is True


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_without_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation without inputs included"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric", include_inputs=False)

    result = evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "<Input>" not in prompt
    assert "<Output>2x2 is 4." in prompt
    assert "<ExpectedOutput>2x2 is 4.</ExpectedOutput>" in prompt
    assert "<Trajectory>['calculator']</Trajectory>" in prompt
    assert "<ExpectedTrajectory>['calculator']</ExpectedTrajectory>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt
    assert result.score == 0.9
    assert result.test_pass is True


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_without_expected_output(mock_agent_class, mock_agent):
    """Test evaluation without expected output"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_output="result",
        actual_trajectory=["step1", "step2"],
        expected_trajectory=["step1", "step2"],
    )

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "<ExpectedOutput>" not in prompt
    assert "<Output>result</Output>" in prompt
    assert "<Trajectory>['step1', 'step2']</Trajectory>" in prompt
    assert "<ExpectedTrajectory>['step1', 'step2']</ExpectedTrajectory>" in prompt


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_without_expected_trajectory(mock_agent_class, mock_agent):
    """Test evaluation without expected trajectory"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_output="result",
        expected_output="result",
        actual_trajectory=["step1", "step2"],
    )

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "<ExpectedTrajectory>" not in prompt
    assert "<Trajectory>['step1', 'step2']</Trajectory>" in prompt


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_missing_actual_output(mock_agent_class, mock_agent):
    """Test evaluation raises exception when actual_output is missing"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", actual_trajectory=["step1"])

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.structured_output.call_args
    prompt = call_args[0][1]
    assert "<Output>" not in prompt


def test_trajectory_evaluator_evaluate_missing_actual_trajectory():
    """Test evaluation raises exception when actual_trajectory is missing"""
    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_output="result",
    )

    with pytest.raises(
        Exception, match="Please make sure the task function return a dictionary with the key 'trajectory'"
    ):
        evaluator.evaluate(evaluation_data)


@pytest.mark.asyncio
@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
async def test_trajectory_evaluator_evaluate_async_with_full_data(mock_agent_class, evaluation_data, mock_async_agent):
    """Test async evaluation with complete trajectory data"""
    mock_agent_class.return_value = mock_async_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric")

    result = await evaluator.evaluate_async(evaluation_data)

    # Verify Agent creation
    mock_agent_class.assert_called_once_with(
        model=None, system_prompt=evaluator.system_prompt, tools=evaluator._tools, callback_handler=None
    )

    assert result.score == 0.9
    assert result.test_pass is True
    assert result.reason == "Mock async trajectory evaluation"


@pytest.mark.asyncio
@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
async def test_trajectory_evaluator_evaluate_async_without_inputs(mock_agent_class, evaluation_data, mock_async_agent):
    """Test async evaluation without inputs included"""
    mock_agent_class.return_value = mock_async_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric", include_inputs=False)

    result = await evaluator.evaluate_async(evaluation_data)

    # Verify Agent creation
    mock_agent_class.assert_called_once_with(
        model=None, system_prompt=evaluator.system_prompt, tools=evaluator._tools, callback_handler=None
    )

    assert result.score == 0.9
    assert result.test_pass is True
    assert result.reason == "Mock async trajectory evaluation"


@pytest.mark.asyncio
async def test_trajectory_evaluator_evaluate_async_missing_actual_trajectory():
    """Test async evaluation raises exception when actual_trajectory is missing"""
    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(
        input="test",
        actual_output="result",
    )

    with pytest.raises(
        Exception, match="Please make sure the task function return a dictionary with the key 'trajectory'"
    ):
        await evaluator.evaluate_async(evaluation_data)


def test_trajectory_evaluator_update_trajectory_description():
    """Test updating trajectory description"""
    initial_description = {"tool1": "Initial description"}
    evaluator = TrajectoryEvaluator(rubric="Test rubric", trajectory_description=initial_description)

    assert evaluator.trajectory_description == initial_description

    new_description = {"tool1": "Updated description", "tool2": "New tool"}
    evaluator.update_trajectory_description(new_description)

    assert evaluator.trajectory_description == new_description


def test_trajectory_evaluator_to_dict():
    """Test that TrajectoryEvaluator to_dict works properly"""
    evaluator = TrajectoryEvaluator(rubric="custom rubric", model="custom_model", system_prompt="custom system prompt")
    evaluator_dict = evaluator.to_dict()
    assert evaluator_dict["evaluator_type"] == "TrajectoryEvaluator"
    assert evaluator_dict["rubric"] == "custom rubric"
    assert evaluator_dict["model"] == "custom_model"
    assert evaluator_dict["system_prompt"] == "custom system prompt"
    assert evaluator_dict.get("include_inputs") is None  # shouldn't include default
    assert evaluator_dict.get("trajectory_description") is None
