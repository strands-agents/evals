from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.environment_state_evaluator import EnvironmentStateEvaluator
from strands_evals.types import EnvironmentState, EvaluationData, EvaluationOutput


@pytest.fixture
def mock_agent():
    agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = EvaluationOutput(score=0.9, test_pass=True, reason="State matches expectations")
    agent.return_value = mock_result
    return agent


@pytest.fixture
def mock_async_agent():
    agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(
            score=0.9, test_pass=True, reason="Async state matches expectations"
        )
        return mock_result

    agent.invoke_async = mock_invoke_async
    return agent


@pytest.fixture
def evaluation_data():
    return EvaluationData(
        input="Fix the failing test",
        actual_output="I fixed the bug",
        actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0, "passed": 5})],
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        name="swebench_test",
    )


def test_init_with_defaults():
    evaluator = EnvironmentStateEvaluator(rubric="Test rubric")
    assert evaluator.rubric == "Test rubric"
    assert evaluator.model is None
    assert evaluator.include_inputs is True
    assert evaluator.system_prompt is not None


def test_init_with_custom_values():
    evaluator = EnvironmentStateEvaluator(
        rubric="Custom rubric", model="gpt-4", system_prompt="Custom prompt", include_inputs=False
    )
    assert evaluator.rubric == "Custom rubric"
    assert evaluator.model == "gpt-4"
    assert evaluator.include_inputs is False
    assert evaluator.system_prompt == "Custom prompt"


@patch("strands_evals.evaluators.environment_state_evaluator.Agent")
def test_evaluate_includes_environment_state_in_prompt(mock_agent_class, evaluation_data, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = EnvironmentStateEvaluator(rubric="Check if tests pass")

    result = evaluator.evaluate(evaluation_data)

    mock_agent_class.assert_called_once_with(model=None, system_prompt=evaluator.system_prompt, callback_handler=None)
    mock_agent.assert_called_once()
    call_args = mock_agent.call_args
    prompt = call_args[0][0]
    assert call_args[1]["structured_output_model"] == EvaluationOutput
    assert "<ActualEnvironmentState>" in prompt
    assert "<ExpectedEnvironmentState>" in prompt
    assert "<Rubric>Check if tests pass</Rubric>" in prompt
    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].test_pass is True


@patch("strands_evals.evaluators.environment_state_evaluator.Agent")
def test_evaluate_includes_input_when_enabled(mock_agent_class, evaluation_data, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = EnvironmentStateEvaluator(rubric="Test rubric")

    evaluator.evaluate(evaluation_data)

    prompt = mock_agent.call_args[0][0]
    assert "<Input>Fix the failing test</Input>" in prompt


@patch("strands_evals.evaluators.environment_state_evaluator.Agent")
def test_evaluate_excludes_input_when_disabled(mock_agent_class, evaluation_data, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = EnvironmentStateEvaluator(rubric="Test rubric", include_inputs=False)

    evaluator.evaluate(evaluation_data)

    prompt = mock_agent.call_args[0][0]
    assert "<Input>" not in prompt


@patch("strands_evals.evaluators.environment_state_evaluator.Agent")
def test_evaluate_without_expected_state(mock_agent_class, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = EnvironmentStateEvaluator(rubric="Test rubric")
    data = EvaluationData(
        input="test",
        actual_output="result",
        actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
    )

    evaluator.evaluate(data)

    prompt = mock_agent.call_args[0][0]
    assert "<ActualEnvironmentState>" in prompt
    assert "<ExpectedEnvironmentState>" not in prompt


@patch("strands_evals.evaluators.environment_state_evaluator.Agent")
def test_evaluate_without_actual_output(mock_agent_class, mock_agent):
    mock_agent_class.return_value = mock_agent
    evaluator = EnvironmentStateEvaluator(rubric="Check side effects")
    data = EvaluationData(
        input="create a file",
        actual_environment_state=[EnvironmentState(name="file_system", state={"created": ["out.txt"]})],
    )

    result = evaluator.evaluate(data)

    prompt = mock_agent.call_args[0][0]
    assert "<Output>" not in prompt
    assert "<ActualEnvironmentState>" in prompt
    assert len(result) == 1


def test_evaluate_missing_actual_environment_state():
    evaluator = EnvironmentStateEvaluator(rubric="Test rubric")
    data = EvaluationData(input="test", actual_output="result")

    with pytest.raises(Exception, match="environment_state"):
        evaluator.evaluate(data)


@pytest.mark.asyncio
@patch("strands_evals.evaluators.environment_state_evaluator.Agent")
async def test_evaluate_async(mock_agent_class, evaluation_data, mock_async_agent):
    mock_agent_class.return_value = mock_async_agent
    evaluator = EnvironmentStateEvaluator(rubric="Test rubric")

    result = await evaluator.evaluate_async(evaluation_data)

    mock_agent_class.assert_called_once_with(model=None, system_prompt=evaluator.system_prompt, callback_handler=None)
    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].test_pass is True
