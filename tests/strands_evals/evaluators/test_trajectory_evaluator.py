import json
import logging
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    """Mock Agent for testing"""
    agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = EvaluationOutput(score=0.9, test_pass=True, reason="Mock trajectory evaluation")
    agent.return_value = mock_result
    return agent


@pytest.fixture
def mock_async_agent():
    """Mock Agent for testing with async"""
    agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(
            score=0.9, test_pass=True, reason="Mock async trajectory evaluation"
        )
        return mock_result

    agent.invoke_async = mock_invoke_async
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

    # Verify agent call
    mock_agent.assert_called_once()
    call_args = mock_agent.call_args
    prompt = call_args[0][0]

    assert "<Input>What's 2x2?</Input>" in prompt
    assert "<Output>2x2 is 4." in prompt
    assert "<ExpectedOutput>2x2 is 4.</ExpectedOutput>" in prompt
    assert "<Trajectory>['calculator']</Trajectory>" in prompt
    assert "<ExpectedTrajectory>['calculator']</ExpectedTrajectory>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt
    assert result[0].score == 0.9
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_without_inputs(mock_agent_class, evaluation_data, mock_agent):
    """Test evaluation without inputs included"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric", include_inputs=False)

    result = evaluator.evaluate(evaluation_data)

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert "<Input>" not in prompt
    assert "<Output>2x2 is 4." in prompt
    assert "<ExpectedOutput>2x2 is 4.</ExpectedOutput>" in prompt
    assert "<Trajectory>['calculator']</Trajectory>" in prompt
    assert "<ExpectedTrajectory>['calculator']</ExpectedTrajectory>" in prompt
    assert "<Rubric>Test rubric</Rubric>" in prompt
    assert result[0].score == 0.9
    assert result[0].test_pass is True


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

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
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

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert "<ExpectedTrajectory>" not in prompt
    assert "<Trajectory>['step1', 'step2']</Trajectory>" in prompt


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_missing_actual_output(mock_agent_class, mock_agent):
    """Test evaluation raises exception when actual_output is missing"""
    mock_agent_class.return_value = mock_agent
    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluation_data = EvaluationData(input="test", actual_trajectory=["step1"])

    evaluator.evaluate(evaluation_data)

    call_args = mock_agent.call_args
    prompt = call_args[0][0]
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

    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].test_pass is True
    assert result[0].reason == "Mock async trajectory evaluation"


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

    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].test_pass is True
    assert result[0].reason == "Mock async trajectory evaluation"


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


def test_trajectory_evaluator_init_with_tools_merges_with_defaults():
    """Test that custom tools are appended after the default scoring tools"""
    from strands_evals.tools.evaluation_tools import (
        any_order_match_scorer,
        exact_match_scorer,
        in_order_match_scorer,
    )

    def verify_step(step: str) -> str:
        return "valid"

    evaluator = TrajectoryEvaluator(rubric="Test rubric", tools=[verify_step])

    assert evaluator._tools == [exact_match_scorer, in_order_match_scorer, any_order_match_scorer, verify_step]


def test_trajectory_evaluator_init_without_tools_keeps_default_scorers():
    """Test that default scoring tools are unchanged when no custom tools are provided"""
    from strands_evals.tools.evaluation_tools import (
        any_order_match_scorer,
        exact_match_scorer,
        in_order_match_scorer,
    )

    evaluator = TrajectoryEvaluator(rubric="Test rubric")

    assert evaluator._tools == [exact_match_scorer, in_order_match_scorer, any_order_match_scorer]


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_trajectory_evaluator_evaluate_passes_custom_tools_to_agent(mock_agent_class, evaluation_data, mock_agent):
    """Test that merged tools (defaults + custom) reach the evaluator agent"""
    from strands_evals.tools.evaluation_tools import (
        any_order_match_scorer,
        exact_match_scorer,
        in_order_match_scorer,
    )

    mock_agent_class.return_value = mock_agent

    def verify_step(step: str) -> str:
        return "valid"

    evaluator = TrajectoryEvaluator(rubric="Test rubric", tools=[verify_step])

    result = evaluator.evaluate(evaluation_data)

    mock_agent_class.assert_called_once_with(
        model=None,
        system_prompt=evaluator.system_prompt,
        tools=[exact_match_scorer, in_order_match_scorer, any_order_match_scorer, verify_step],
        callback_handler=None,
    )
    assert result[0].score == 0.9


def test_trajectory_evaluator_tools_property_excludes_default_scorers():
    """Test that the public tools attribute exposes only user-supplied tools, not the
    built-in exact/in-order/any-order scorers that _tools always carries (issue #381)"""

    def verify_step(step: str) -> str:
        return "valid"

    evaluator = TrajectoryEvaluator(rubric="Test rubric", tools=[verify_step])

    assert evaluator.tools == [verify_step]


def test_trajectory_evaluator_init_without_tools_defaults_to_none():
    """Test TrajectoryEvaluator has no user tools by default"""
    evaluator = TrajectoryEvaluator(rubric="Test rubric")

    assert evaluator.tools is None
    assert "tools" not in evaluator.to_dict()


def test_trajectory_evaluator_to_dict_skips_non_serializable_tools(caplog):
    """Test that to_dict() output is JSON-serializable when callable tools are set,
    instead of silently dropping them with no public attribute to notice by (issue #381)"""

    def verify_step(step: str) -> str:
        return "valid"

    evaluator = TrajectoryEvaluator(rubric="Test rubric", tools=[verify_step])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert "tools" not in evaluator_dict
    json.dumps(evaluator_dict)
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text


def test_trajectory_evaluator_to_dict_skips_circular_reference_tools_with_no_name(caplog):
    """Test that a nameless tool raising ValueError (circular reference) is skipped by
    a truncated ascii() preview rather than crashing on a missing tool_name/__name__"""
    circular: dict = {"name": "x" * 100}
    circular["self"] = circular

    evaluator = TrajectoryEvaluator(rubric="Test rubric", tools=[circular])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert "tools" not in evaluator_dict
    json.dumps(evaluator_dict)
    assert "skipping tool that cannot be written as valid utf-8 JSON" in caplog.text


def test_trajectory_evaluator_to_dict_keeps_serializable_tools(caplog):
    """Test that a serializable user tool round-trips through to_dict() while the
    built-in scorers are never included (issue #381)"""

    def verify_step(step: str) -> str:
        return "valid"

    evaluator = TrajectoryEvaluator(rubric="Test rubric", tools=["my_pkg.calculator", verify_step])
    with caplog.at_level(logging.WARNING):
        evaluator_dict = evaluator.to_dict()

    assert evaluator_dict["tools"] == ["my_pkg.calculator"]
    json.dumps(evaluator_dict)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1  # the serializable tool and the default scorers must not warn
    assert "verify_step" in warnings[0].getMessage()


def test_trajectory_evaluator_tools_setter_stays_in_sync_with_tools():
    """Test that reassigning tools updates both the public attribute and the merged
    list the evaluator agent actually runs with"""
    from strands_evals.tools.evaluation_tools import (
        any_order_match_scorer,
        exact_match_scorer,
        in_order_match_scorer,
    )

    def verify_step(step: str) -> str:
        return "valid"

    evaluator = TrajectoryEvaluator(rubric="Test rubric")
    evaluator.tools = [verify_step]

    assert evaluator.tools == [verify_step]
    assert evaluator._tools == [exact_match_scorer, in_order_match_scorer, any_order_match_scorer, verify_step]
