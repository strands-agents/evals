import asyncio
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from strands.models.model import Model
from strands.types.exceptions import EventLoopException, ModelThrottledException

from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    Contains,
    Equals,
    Evaluator,
    InteractionsEvaluator,
    OutputEvaluator,
    StartsWith,
    ToolCalled,
    TrajectoryEvaluator,
)
from strands_evals.evaluators.evaluator import DEFAULT_BEDROCK_MODEL_ID
from strands_evals.experiment import is_throttling_error
from strands_evals.providers.trace_provider import TraceProvider
from strands_evals.types import EvaluationData, EvaluationOutput


class MockEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        # Simple mock: pass if actual equals expected
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return [EvaluationOutput(score=score, test_pass=score > 0.5, reason="Mock evaluation")]


class MockEvaluator2(Evaluator[str, str]):
    """Second mock evaluator that always returns 0.5 for distinguishable results"""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        return [EvaluationOutput(score=0.5, test_pass=True, reason="Mock evaluation 2")]


class ThrowingEvaluator(Evaluator[str, str]):
    """Evaluator that always throws an exception - used to test error isolation"""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
        raise RuntimeError("Evaluator exploded")


@pytest.fixture
def mock_evaluator():
    return MockEvaluator()


@pytest.fixture
def mock_span():
    """Fixture that creates a mock span for tracing tests"""
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


@pytest.fixture
def simple_task():
    """Fixture that provides a simple echo task function"""

    def task(case):
        return case.input

    return task


def test_experiment__init__full(mock_evaluator):
    """Test creating an Experiment with test cases and evaluators"""
    cases = [
        Case(name="test1", input="hello", expected_output="world"),
        Case(name="test2", input="foo", expected_output="bar"),
    ]

    experiment = Experiment(cases=cases, evaluators=[mock_evaluator])

    assert len(experiment.cases) == 2
    assert experiment.evaluators == [mock_evaluator]

    # Test with multiple evaluators
    eval2 = MockEvaluator2()
    experiment2 = Experiment(cases=cases, evaluators=[mock_evaluator, eval2])
    assert len(experiment2.evaluators) == 2


def test_experiment__init__partial_cases():
    """Test creating an Experiment with test cases only"""
    cases = [
        Case(name="test1", input="hello", expected_output="world"),
        Case(name="test2", input="foo", expected_output="bar"),
    ]

    experiment = Experiment(cases=cases)

    assert len(experiment.cases) == 2
    # Should have default evaluator list
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], Evaluator)


def test_experiment__init__partial_evaluators():
    """Test creating an Experiment with evaluators only"""
    evaluator = Evaluator()
    experiment = Experiment(evaluators=[evaluator])

    assert len(experiment.cases) == 0
    assert experiment.evaluators == [evaluator]


def test_experiment_cases_getter_deep_copy():
    """Test cases getter should return deep copies"""
    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    retrieved = experiment.cases
    retrieved[0].name = "modified"

    assert experiment.cases == [case]


def test_experiment_cases_setter():
    """Test cases setter updates experiment"""
    case1 = Case(name="test1", input="hello", expected_output="world")
    case2 = Case(name="test2", input="hi", expected_output="there")
    experiment = Experiment(cases=[case1], evaluators=[MockEvaluator()])

    experiment.cases = [case2]
    assert experiment.cases == [case2]


def test_experiment_evaluators_getter():
    """Test evaluators getter returns evaluators"""
    evaluator = MockEvaluator()
    experiment = Experiment(cases=[], evaluators=[evaluator])

    retrieved = experiment.evaluators
    assert retrieved == [evaluator]


def test_experiment_evaluators_setter():
    """Test evaluators setter updates experiment"""
    eval1 = Evaluator()
    eval2 = MockEvaluator()
    experiment = Experiment(cases=[], evaluators=[eval1])

    experiment.evaluators = [eval2]
    assert experiment.evaluators == [eval2]

    # Test setting multiple evaluators
    eval3 = MockEvaluator2()
    experiment.evaluators = [eval2, eval3]
    assert experiment.evaluators == [eval2, eval3]


@pytest.mark.asyncio
async def test_experiment__run_task_async_with_input_update():
    """Test _run_task_async with dictionary output containing updated input"""
    case = Case(name="test", input="original_input", expected_output="world")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    def task_with_input_update(c):
        return {"output": f"response to {c.input}", "input": "async_updated_input"}

    result = await experiment._run_task_async(task_with_input_update, case)

    assert result.input == "async_updated_input"
    assert result.actual_output == "response to original_input"


def test_experiment_run_evaluations_async_function_raises_error(mock_evaluator):
    """Test run_evaluations raises ValueError when async task is passed"""
    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluators=[mock_evaluator])

    async def async_task(c):
        return f"response to {c.input}"

    with pytest.raises(ValueError, match="Async task is not supported. Please use run_evaluations_async instead."):
        experiment.run_evaluations(async_task)


@pytest.mark.asyncio
async def test_experiment__run_task_async_with_sync_task():
    """Test _run_task_async with a synchronous task function"""

    def sync_task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])
    evaluation_context = await experiment._run_task_async(sync_task, case)
    assert evaluation_context.input == "hello"
    assert evaluation_context.actual_output == "hello"
    assert evaluation_context.expected_output == "world"


@pytest.mark.asyncio
async def test_experiment__run_task_async_with_async_task():
    """Test _run_task_async with an asynchronous task function"""

    async def async_task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])
    evaluation_context = await experiment._run_task_async(async_task, case)
    assert evaluation_context.input == "hello"
    assert evaluation_context.actual_output == "hello"
    assert evaluation_context.expected_output == "world"


def test_experiment_run_evaluations(mock_evaluator):
    """Test complete evaluation run"""
    cases = [
        Case(name="match", input="hello", expected_output="hello"),
        Case(name="no_match", input="foo", expected_output="bar"),
    ]
    experiment = Experiment(cases=cases, evaluators=[mock_evaluator])

    def echo_task(c):
        return c.input

    reports = experiment.run_evaluations(echo_task)

    # Returns list of reports, one per evaluator
    assert len(reports) == 1
    report = reports[0]
    assert len(report.scores) == 2
    assert report.scores[0] == 1.0  # match
    assert report.scores[1] == 0.0  # no match
    assert report.test_passes[0] is True
    assert report.test_passes[1] is False
    assert report.overall_score == 0.5
    assert len(report.cases) == 2

    # Test with multiple evaluators - each gets its own report
    experiment2 = Experiment(cases=cases, evaluators=[mock_evaluator, MockEvaluator2()])
    reports2 = experiment2.run_evaluations(echo_task)
    assert len(reports2) == 2
    assert reports2[0].scores[0] == 1.0  # MockEvaluator on match
    assert reports2[1].scores[0] == 0.5  # MockEvaluator2 always returns 0.5


def test_experiment_run_evaluations_task_executed_once():
    """Test that task is executed only once per case even with multiple evaluators"""
    task_call_count = 0

    def counting_task(c):
        nonlocal task_call_count
        task_call_count += 1
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator(), MockEvaluator2()])

    experiment.run_evaluations(counting_task)

    # Task should be called once per case, not once per evaluator
    assert task_call_count == 1


def test_experiment_to_dict_empty(mock_evaluator):
    """Test converting empty experiment to dictionary"""
    experiment = Experiment(cases=[], evaluators=[mock_evaluator])
    assert experiment.to_dict() == {"cases": [], "evaluators": [{"evaluator_type": "MockEvaluator"}]}


def test_experiment_to_dict_non_empty(mock_evaluator):
    """Test converting non-empty experiment to dictionary"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    experiment = Experiment(cases=cases, evaluators=[mock_evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [{"evaluator_type": "MockEvaluator"}],
    }

    # Test with multiple evaluators
    eval2 = OutputEvaluator(rubric="test rubric")
    experiment2 = Experiment(cases=cases, evaluators=[mock_evaluator, eval2])
    result = experiment2.to_dict()
    assert len(result["evaluators"]) == 2
    assert result["evaluators"][0] == {"evaluator_type": "MockEvaluator"}
    assert result["evaluators"][1]["evaluator_type"] == "OutputEvaluator"


def test_experiment_to_dict_OutputEvaluator_full():
    """Test converting experiment with OutputEvaluator to dictionary with no defaults."""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
    experiment = Experiment(cases=cases, evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [
            {
                "evaluator_type": "OutputEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt",
            }
        ],
    }


def test_experiment_to_dict_OutputEvaluator_default():
    """Test converting experiment with OutputEvaluator to dictionary with defaults.
    The evaluator's data should not include default information."""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="rubric")
    experiment = Experiment(cases=cases, evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    result = experiment.to_dict()
    assert result == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [{"evaluator_type": "OutputEvaluator", "rubric": "rubric", "model_id": DEFAULT_BEDROCK_MODEL_ID}],
    }


def test_experiment_to_dict_TrajectoryEvaluator_default():
    """Test converting experiment with TrajectoryEvaluator to dictionary with defaults.
    The evaluator's data should not include default information."""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    evaluator = TrajectoryEvaluator(rubric="rubric")
    experiment = Experiment(cases=cases, evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": ["step1", "step2"],
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [
            {
                "evaluator_type": "TrajectoryEvaluator",
                "rubric": "rubric",
                "model_id": DEFAULT_BEDROCK_MODEL_ID,
            }
        ],
    }


def test_experiment_to_dict_TrajectoryEvaluator_full():
    """Test converting experiment with TrajectoryEvaluator to dictionary with no defaults."""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    evaluator = TrajectoryEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
    experiment = Experiment(cases=cases, evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": ["step1", "step2"],
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [
            {
                "evaluator_type": "TrajectoryEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt",
            }
        ],
    }


def test_experiment_to_dict_InteractionsEvaluator_default():
    """Test converting experiment with InteractionsEvaluator to dictionary with defaults."""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    evaluator = InteractionsEvaluator(rubric="rubric")
    experiment = Experiment(cases=cases, evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": interactions,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [
            {
                "evaluator_type": "InteractionsEvaluator",
                "rubric": "rubric",
                "model_id": DEFAULT_BEDROCK_MODEL_ID,
            }
        ],
    }


def test_experiment_to_dict_InteractionsEvaluator_full():
    """Test converting experiment with InteractionsEvaluator to dictionary with no defaults."""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    evaluator = InteractionsEvaluator(
        rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt"
    )
    experiment = Experiment(cases=cases, evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": interactions,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [
            {
                "evaluator_type": "InteractionsEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt",
            }
        ],
    }


def test_experiment_to_dict_case_dict():
    """Test converting experiment with Case with dictionaries as types."""
    case = Case(name="test", input={"field1": "hello"}, expected_output={"field2": "world"}, metadata={})
    evaluator = MockEvaluator()
    experiment = Experiment(cases=[case], evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": {"field1": "hello"},
                "expected_output": {"field2": "world"},
                "expected_trajectory": None,
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": {},
            }
        ],
        "evaluators": [{"evaluator_type": "MockEvaluator"}],
    }


def test_experiment_to_dict_case_function():
    """Test converting experiment with Case with function as types."""

    def simple_echo(query):
        return query

    case = Case(name="test", input=simple_echo)
    evaluator = MockEvaluator()
    experiment = Experiment(cases=[case], evaluators=[evaluator])
    session_id = experiment.cases[0].session_id
    assert experiment.to_dict() == {
        "cases": [
            {
                "name": "test",
                "session_id": session_id,
                "input": simple_echo,
                "expected_output": None,
                "expected_trajectory": None,
                "expected_interactions": None,
                "expected_environment_state": None,
                "metadata": None,
            }
        ],
        "evaluators": [{"evaluator_type": "MockEvaluator"}],
    }


def test_experiment_from_dict_custom():
    """Test creating an Experiment with a custom evaluator and empty cases"""
    dict_experiment = {"cases": [], "evaluators": [{"evaluator_type": "MockEvaluator"}]}
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[MockEvaluator])
    assert experiment.cases == []
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], MockEvaluator)

    # Test with multiple evaluators
    dict_experiment2 = {
        "cases": [],
        "evaluators": [
            {"evaluator_type": "MockEvaluator"},
            {"evaluator_type": "OutputEvaluator", "rubric": "test"},
        ],
    }
    experiment2 = Experiment.from_dict(dict_experiment2, custom_evaluators=[MockEvaluator])
    assert len(experiment2.evaluators) == 2
    assert isinstance(experiment2.evaluators[0], MockEvaluator)
    assert isinstance(experiment2.evaluators[1], OutputEvaluator)


def test_experiment_from_dict_OutputEvaluator():
    """Test creating an Experiment with a OutputEvaluator"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {
        "cases": cases,
        "evaluators": [
            {
                "evaluator_type": "OutputEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt",
            }
        ],
    }
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[OutputEvaluator])
    assert experiment.cases == cases
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], OutputEvaluator)
    assert experiment.evaluators[0].rubric == "rubric"
    assert experiment.evaluators[0].model == "model"
    assert experiment.evaluators[0].include_inputs is False
    assert experiment.evaluators[0].system_prompt == "system prompt"


def test_experiment_from_dict_OutputEvaluator_defaults():
    """Test creating an Experiment with a OutputEvaluator with defaults"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {"cases": cases, "evaluators": [{"evaluator_type": "OutputEvaluator", "rubric": "rubric"}]}
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[OutputEvaluator])
    assert experiment.cases == cases
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], OutputEvaluator)
    assert experiment.evaluators[0].rubric == "rubric"
    assert experiment.evaluators[0].model is None
    assert experiment.evaluators[0].include_inputs is True


def test_experiment_from_dict_with_model_id():
    """Test creating an Experiment from dict with model_id (should convert to model parameter)"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    dict_experiment = {
        "cases": cases,
        "evaluators": [
            {
                "evaluator_type": "OutputEvaluator",
                "rubric": "test rubric",
                "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            }
        ],
    }
    experiment = Experiment.from_dict(dict_experiment)

    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], OutputEvaluator)
    assert experiment.evaluators[0].rubric == "test rubric"
    assert experiment.evaluators[0].model == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_experiment_to_dict_from_dict_roundtrip_with_model():
    """Test that to_dict and from_dict work correctly for roundtrip with model"""

    # Create experiment with Model instance
    mock_model = MagicMock(spec=Model)
    mock_model.config = {"model_id": "test-model-roundtrip"}

    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="test rubric", model=mock_model)
    experiment = Experiment(cases=cases, evaluators=[evaluator])

    # Serialize to dict
    experiment_dict = experiment.to_dict()
    assert experiment_dict["evaluators"][0]["model_id"] == "test-model-roundtrip"
    assert "model" not in experiment_dict["evaluators"][0]

    # Deserialize from dict
    restored_experiment = Experiment.from_dict(experiment_dict)
    assert len(restored_experiment.evaluators) == 1
    assert isinstance(restored_experiment.evaluators[0], OutputEvaluator)
    assert restored_experiment.evaluators[0].model == "test-model-roundtrip"


def test_experiment_to_dict_from_dict_roundtrip_with_string_model():
    """Test that to_dict and from_dict work correctly for roundtrip with string model"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="test rubric", model="bedrock-model-id")
    experiment = Experiment(cases=cases, evaluators=[evaluator])

    # Serialize to dict
    experiment_dict = experiment.to_dict()
    assert experiment_dict["evaluators"][0]["model"] == "bedrock-model-id"

    # Deserialize from dict
    restored_experiment = Experiment.from_dict(experiment_dict)
    assert len(restored_experiment.evaluators) == 1
    assert isinstance(restored_experiment.evaluators[0], OutputEvaluator)
    assert restored_experiment.evaluators[0].model == "bedrock-model-id"


def test_experiment_from_dict_TrajectoryEvaluator():
    """Test creating an Experiment with a TrajectoryEvaluator"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {
        "cases": cases,
        "evaluators": [
            {
                "evaluator_type": "TrajectoryEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt",
            }
        ],
    }
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[TrajectoryEvaluator])
    assert experiment.cases == cases
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], TrajectoryEvaluator)
    assert experiment.evaluators[0].rubric == "rubric"
    assert experiment.evaluators[0].model == "model"
    assert experiment.evaluators[0].include_inputs is False
    assert experiment.evaluators[0].system_prompt == "system prompt"


def test_experiment_from_dict_TrajectoryEvaluator_defaults():
    """Test creating an Experiment with a Trajectory evaluator with defaults"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {"cases": cases, "evaluators": [{"evaluator_type": "TrajectoryEvaluator", "rubric": "rubric"}]}
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[TrajectoryEvaluator])
    assert experiment.cases == cases
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], TrajectoryEvaluator)
    assert experiment.evaluators[0].rubric == "rubric"
    assert experiment.evaluators[0].model is None
    assert experiment.evaluators[0].include_inputs is True


def test_experiment_from_dict_InteractionsEvaluator():
    """Test creating an Experiment with an InteractionsEvaluator"""
    interactions = [{"node_name": "agent1", "dependencies": [], "message": "hello"}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    dict_experiment = {
        "cases": cases,
        "evaluators": [
            {
                "evaluator_type": "InteractionsEvaluator",
                "rubric": "rubric",
                "model": "model",
                "include_inputs": False,
                "system_prompt": "system prompt",
            }
        ],
    }
    experiment = Experiment.from_dict(dict_experiment)
    assert experiment.cases == cases
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], InteractionsEvaluator)
    assert experiment.evaluators[0].rubric == "rubric"
    assert experiment.evaluators[0].model == "model"
    assert experiment.evaluators[0].include_inputs is False
    assert experiment.evaluators[0].system_prompt == "system prompt"


def test_experiment_from_dict_InteractionsEvaluator_defaults():
    """Test creating an Experiment with an Interactions evaluator with defaults"""
    interactions = [{"node_name": "agent1", "dependencies": [], "message": "hello"}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    dict_experiment = {"cases": cases, "evaluators": [{"evaluator_type": "InteractionsEvaluator", "rubric": "rubric"}]}
    experiment = Experiment.from_dict(dict_experiment)
    assert experiment.cases == cases
    assert len(experiment.evaluators) == 1
    assert isinstance(experiment.evaluators[0], InteractionsEvaluator)
    assert experiment.evaluators[0].rubric == "rubric"
    assert experiment.evaluators[0].model is None
    assert experiment.evaluators[0].include_inputs is True


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async():
    """Test run_evaluations_async with a simple task"""

    def task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    experiment = Experiment(cases=[case, case1], evaluators=[MockEvaluator()])

    reports = await experiment.run_evaluations_async(task)

    assert len(reports) == 1
    report = reports[0]
    assert len(report.scores) == 2
    assert all(score == 1.0 for score in report.scores)
    assert all(test_pass for test_pass in report.test_passes)
    assert report.overall_score == 1.0

    # Test with multiple evaluators
    experiment2 = Experiment(cases=[case], evaluators=[MockEvaluator(), MockEvaluator2()])
    reports2 = await experiment2.run_evaluations_async(task)
    assert len(reports2) == 2
    assert reports2[0].scores[0] == 1.0  # MockEvaluator
    assert reports2[1].scores[0] == 0.5  # MockEvaluator2


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_with_async_task():
    """Test run_evaluations_async with an async task"""

    async def async_task(c):
        await asyncio.sleep(0.01)
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    experiment = Experiment(cases=[case, case1], evaluators=[MockEvaluator()])
    reports = await experiment.run_evaluations_async(async_task)

    assert len(reports) == 1
    report = reports[0]
    assert len(report.scores) == 2
    assert all(score == 1.0 for score in report.scores)
    assert all(test_pass for test_pass in report.test_passes)
    assert report.overall_score == 1.0


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_with_errors():
    """Test run_evaluations_async handles errors gracefully"""

    def failing_task(c):
        if c.input == "hello":
            raise ValueError("Test error")
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    experiment = Experiment(cases=[case, case1], evaluators=[MockEvaluator()])
    reports = await experiment.run_evaluations_async(failing_task)

    assert len(reports) == 1
    report = reports[0]
    assert len(report.scores) == 2
    # One of the cases should have failed (score 0) and one passed (score 1)
    assert 0.0 in report.scores
    assert 1.0 in report.scores
    # Check that error message is in reasons
    error_reasons = [r for r in report.reasons if "Test error" in r]
    assert len(error_reasons) == 1


def test_experiment_run_evaluations_with_interactions():
    """Test evaluation run with interactions data"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["test message"]}]
    case = Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    def task_with_interactions(c):
        return {"output": c.input, "interactions": interactions}

    reports = experiment.run_evaluations(task_with_interactions)

    assert len(reports) == 1
    report = reports[0]
    assert len(report.cases) == 1
    assert report.cases[0]["actual_interactions"] == interactions
    assert report.cases[0]["expected_interactions"] == interactions


def test_experiment_init_always_initializes_tracer():
    """Test that Experiment always initializes tracer in __init__"""
    with patch("strands_evals.experiment.get_tracer") as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        experiment = Experiment(cases=[], evaluators=[MockEvaluator()])

        mock_get_tracer.assert_called_once()
        assert experiment._tracer == mock_tracer


def test_experiment_run_evaluations_creates_spans(mock_span, simple_task):
    """Test that run_evaluations creates spans with correct attributes"""
    case = Case(name="test_case", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        experiment.run_evaluations(simple_task)

        # Verify spans were created
        assert mock_start_span.called
        # Verify set_attributes was called with evaluation results
        mock_span.set_attributes.assert_called()


def test_experiment_run_evaluations_with_trajectory_in_span(mock_span):
    """Test that run_evaluations includes trajectory in span attributes"""
    case = Case(name="test_case", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

        def task_with_trajectory(c):
            return {"output": c.input, "trajectory": ["step1", "step2"]}

        experiment.run_evaluations(task_with_trajectory)

        # Check that set_attributes was called
        mock_span.set_attributes.assert_called()


def test_experiment_run_evaluations_with_interactions_in_span(mock_span):
    """Test that run_evaluations includes interactions in span attributes"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    case = Case(name="test_case", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

        def task_with_interactions(c):
            return {"output": c.input, "interactions": interactions}

        experiment.run_evaluations(task_with_interactions)

        # Check that set_attributes was called
        mock_span.set_attributes.assert_called()


def test_experiment_run_evaluations_records_exception_in_span(mock_span):
    """Test that run_evaluations handles exceptions gracefully"""
    case = Case(name="test_case", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    def failing_task(c):
        raise ValueError("Test error")

    reports = experiment.run_evaluations(failing_task)

    # Verify error was handled and report contains error info
    assert len(reports) == 1
    report = reports[0]
    assert len(report.scores) == 1
    assert report.scores[0] == 0
    assert report.test_passes[0] is False
    assert "Test error" in report.reasons[0]


def test_experiment_run_evaluations_with_unnamed_case(mock_span, simple_task):
    """Test that run_evaluations handles unnamed cases correctly"""
    case = Case(input="hello", expected_output="hello")  # No name
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):
            reports = experiment.run_evaluations(simple_task)

            # Should complete successfully
            assert len(reports) == 1
            assert reports[0].scores[0] == 1.0


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_creates_spans(mock_span):
    """Test that run_evaluations_async creates spans"""
    case = Case(name="async_test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):

            async def async_task(c):
                return c.input

            await experiment.run_evaluations_async(async_task)

            # Verify execute_case, task_execution, and evaluator spans were created
            calls = mock_start_span.call_args_list
            assert len(calls) == 3

            # execute_case span has case.name and case.input attributes
            assert calls[0][0][0] == "execute_case async_test"
            assert calls[0][1]["attributes"]["gen_ai.evaluation.case.name"] == "async_test"
            assert calls[0][1]["attributes"]["gen_ai.evaluation.case.input"] == '"hello"'

            # task_execution span has task.type and case.name attributes
            assert calls[1][0][0] == "task_execution async_test"
            assert calls[1][1]["attributes"]["gen_ai.evaluation.task.type"] == "agent_task"
            assert calls[1][1]["attributes"]["gen_ai.evaluation.case.name"] == "async_test"

            # evaluator span has evaluation.name and case.name attributes
            assert calls[2][0][0] == "evaluator MockEvaluator"
            assert calls[2][1]["attributes"]["gen_ai.evaluation.name"] == "MockEvaluator"
            assert calls[2][1]["attributes"]["gen_ai.evaluation.case.name"] == "async_test"


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_data_attrs_on_task_span():
    """Test that data attributes (input, expected_output, etc.) are set on the task_execution span, not execute_case."""
    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    # Create distinct mock spans so we can tell which span gets which set_attributes call
    case_span = MagicMock()
    case_span.__enter__ = MagicMock(return_value=case_span)
    case_span.__exit__ = MagicMock(return_value=False)
    task_span = MagicMock()
    task_span.__enter__ = MagicMock(return_value=task_span)
    task_span.__exit__ = MagicMock(return_value=False)
    eval_span = MagicMock()
    eval_span.__enter__ = MagicMock(return_value=eval_span)
    eval_span.__exit__ = MagicMock(return_value=False)

    spans = [case_span, task_span, eval_span]
    span_index = 0

    def fake_start_span(name, **kwargs):
        nonlocal span_index
        span = spans[span_index]
        span_index += 1
        return span

    with patch.object(experiment._tracer, "start_as_current_span", side_effect=fake_start_span):
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):

            async def async_task(c):
                return c.input

            await experiment.run_evaluations_async(async_task)

            # data attributes should be on task_span, NOT case_span
            task_span.set_attributes.assert_called_once()
            data_attrs = task_span.set_attributes.call_args[0][0]
            assert "gen_ai.evaluation.data.input" in data_attrs
            assert "gen_ai.evaluation.data.expected_output" in data_attrs
            assert "gen_ai.evaluation.data.actual_output" in data_attrs
            assert "gen_ai.evaluation.data.has_trajectory" in data_attrs
            assert "gen_ai.evaluation.data.has_interactions" in data_attrs

            # case_span should NOT have data attributes set via set_attributes
            for call in case_span.set_attributes.call_args_list:
                attrs = call[0][0]
                assert "gen_ai.evaluation.data.input" not in attrs


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_records_exception(mock_span):
    """Test that run_evaluations_async records exceptions"""
    case = Case(name="async_test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

        async def failing_async_task(c):
            raise ValueError("Async test error")

        reports = await experiment.run_evaluations_async(failing_async_task)

        # Verify the error was handled gracefully
        assert len(reports) == 1
        report = reports[0]
        assert len(report.scores) == 1
        assert report.scores[0] == 0
        assert "Async test error" in report.reasons[0]


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_with_dict_output(mock_span):
    """Test that run_evaluations_async handles dict output with trajectory/interactions"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    case = Case(name="async_test", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):

            async def async_task_with_dict(c):
                return {"output": c.input, "trajectory": ["step1"], "interactions": interactions}

            await experiment.run_evaluations_async(async_task_with_dict)

            # Check that set_attributes was called (trajectory/interactions are set via set_attributes)
            mock_span.set_attributes.assert_called()
            # Verify has_trajectory and has_interactions flags are set
            set_attrs_calls = mock_span.set_attributes.call_args_list
            has_trajectory_set = any(
                "gen_ai.evaluation.data.has_trajectory" in call[0][0] for call in set_attrs_calls if call[0]
            )
            has_interactions_set = any(
                "gen_ai.evaluation.data.has_interactions" in call[0][0] for call in set_attrs_calls if call[0]
            )
            assert has_trajectory_set
            assert has_interactions_set


def test_experiment_run_evaluations_multiple_cases(mock_span, simple_task):
    """Test that each case is evaluated correctly"""
    cases = [
        Case(name="case1", input="hello", expected_output="hello"),
        Case(name="case2", input="world", expected_output="world"),
    ]
    experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):
        with patch("strands_evals.experiment.format_trace_id", return_value="mock_trace_id"):
            reports = experiment.run_evaluations(simple_task)

            assert len(reports) == 1
            assert len(reports[0].scores) == 2
            assert all(score == 1.0 for score in reports[0].scores)


def test_experiment_run_evaluations_evaluator_error_isolated():
    """Test that one evaluator failing doesn't affect other evaluators."""
    case = Case(name="test", input="hello", expected_output="hello")

    # MockEvaluator succeeds, ThrowingEvaluator fails
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator(), ThrowingEvaluator()])

    def echo_task(c):
        return c.input

    reports = experiment.run_evaluations(echo_task)

    assert len(reports) == 2

    # First evaluator (MockEvaluator) should succeed
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True
    assert reports[0].reasons[0] == "Mock evaluation"

    # Second evaluator (ThrowingEvaluator) should fail with error message
    assert reports[1].scores[0] == 0
    assert reports[1].test_passes[0] is False
    assert "Evaluator error" in reports[1].reasons[0]
    assert "Evaluator exploded" in reports[1].reasons[0]


def testis_throttling_error_detects_model_throttled_exception():
    """Test that ModelThrottledException is detected as throttling error"""
    exc = ModelThrottledException("Too many tokens")
    assert is_throttling_error(exc) is True


def testis_throttling_error_detects_event_loop_exception():
    """Test that EventLoopException is detected as throttling error"""
    exc = EventLoopException(Exception("Throttling"), {})
    assert is_throttling_error(exc) is True


def testis_throttling_error_detects_botocore_throttling():
    """Test that botocore ThrottlingException is detected"""

    # Create a mock exception with ThrottlingException class name
    class ThrottlingException(Exception):
        pass

    exc = ThrottlingException("Rate limit exceeded")
    assert is_throttling_error(exc) is True


def testis_throttling_error_detects_client_error_with_codes():
    """Test that ClientError with throttling codes is detected"""
    # Test various throttling error codes
    throttling_codes = [
        "ThrottlingException",
        "TooManyRequestsException",
        "RequestLimitExceeded",
        "ServiceUnavailable",
        "ProvisionedThroughputExceededException",
    ]

    for code in throttling_codes:
        exc = ClientError({"Error": {"Code": code, "Message": "Test"}}, "TestOperation")
        assert is_throttling_error(exc) is True, f"Failed to detect {code}"


def testis_throttling_error_ignores_regular_exceptions():
    """Test that regular exceptions are not detected as throttling"""
    regular_exceptions = [
        ValueError("Invalid input"),
        RuntimeError("Runtime error"),
        TypeError("Type error"),
    ]

    for exc in regular_exceptions:
        assert is_throttling_error(exc) is False


def testis_throttling_error_ignores_client_error_non_throttling():
    """Test that ClientError with non-throttling codes is not detected"""
    exc = ClientError({"Error": {"Code": "ValidationException", "Message": "Invalid"}}, "TestOperation")
    assert is_throttling_error(exc) is False


def test_experiment_run_evaluations_retries_on_throttling():
    """Test that run_evaluations retries task execution on throttling errors"""
    call_count = 0

    def throttling_task(c):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ModelThrottledException("Too many tokens")
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01):
        with patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02):
            reports = experiment.run_evaluations(throttling_task)

    # Task should have been retried
    assert call_count == 3
    assert len(reports) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


def test_experiment_run_evaluations_fails_after_max_retries():
    """Test that run_evaluations fails after max retries on throttling"""
    call_count = 0

    def always_throttling_task(c):
        nonlocal call_count
        call_count += 1
        raise ModelThrottledException("Too many tokens")

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch("strands_evals.experiment._MAX_RETRY_ATTEMPTS", 3):
        with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01):
            with patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02):
                reports = experiment.run_evaluations(always_throttling_task)

    # Should have retried max times
    assert call_count == 3
    assert len(reports) == 1
    assert reports[0].scores[0] == 0
    assert reports[0].test_passes[0] is False
    assert "An error occurred" in reports[0].reasons[0]


def test_experiment_run_evaluations_no_retry_on_non_throttling():
    """Test that run_evaluations doesn't retry on non-throttling errors"""
    call_count = 0

    def non_throttling_error_task(c):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid input")

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    reports = experiment.run_evaluations(non_throttling_error_task)

    # Should NOT have retried
    assert call_count == 1
    assert len(reports) == 1
    assert reports[0].scores[0] == 0
    assert "Invalid input" in reports[0].reasons[0]


def test_experiment_run_evaluations_exponential_backoff():
    """Test that run_evaluations uses exponential backoff for retries"""
    sleep_delays = []

    async def mock_async_sleep(delay):
        sleep_delays.append(delay)

    call_count = 0

    def throttling_task(c):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ModelThrottledException("Too many tokens")
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch("asyncio.sleep", mock_async_sleep):
        with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 1):
            with patch("strands_evals.experiment._MAX_RETRY_DELAY", 10):
                experiment.run_evaluations(throttling_task)

    # Verify exponential backoff: 1, 2, 4
    assert len(sleep_delays) == 3
    assert sleep_delays[0] == 1
    assert sleep_delays[1] == 2
    assert sleep_delays[2] == 4


def test_experiment_run_evaluations_evaluator_retries_on_throttling():
    """Test that evaluator execution retries on throttling errors"""

    class ThrottlingEvaluator(Evaluator[str, str]):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        def evaluate(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
            self.call_count += 1
            if self.call_count <= 2:
                raise ModelThrottledException("Too many tokens")
            return [EvaluationOutput(score=1.0, test_pass=True, reason="Success after retry")]

    def simple_task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    evaluator = ThrottlingEvaluator()
    experiment = Experiment(cases=[case], evaluators=[evaluator])

    with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01):
        with patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02):
            reports = experiment.run_evaluations(simple_task)

    # Evaluator should have been retried
    assert evaluator.call_count == 3
    assert len(reports) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_retries_on_throttling():
    """Test that run_evaluations_async retries task execution on throttling"""
    call_count = 0

    async def throttling_task(c):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ModelThrottledException("Too many tokens")
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01):
        with patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02):
            reports = await experiment.run_evaluations_async(throttling_task, max_workers=1)

    # Task should have been retried
    assert call_count == 3
    assert len(reports) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_fails_after_max_retries():
    """Test that run_evaluations_async fails after max retries"""
    call_count = 0

    async def always_throttling_task(c):
        nonlocal call_count
        call_count += 1
        raise ModelThrottledException("Too many tokens")

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch("strands_evals.experiment._MAX_RETRY_ATTEMPTS", 3):
        with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01):
            with patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02):
                reports = await experiment.run_evaluations_async(always_throttling_task, max_workers=1)

    # Should have retried max times
    assert call_count == 3
    assert len(reports) == 1
    assert reports[0].scores[0] == 0
    assert reports[0].test_passes[0] is False
    assert "An error occurred" in reports[0].reasons[0]


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_no_retry_on_non_throttling():
    """Test that run_evaluations_async doesn't retry on non-throttling errors"""
    call_count = 0

    async def non_throttling_error_task(c):
        nonlocal call_count
        call_count += 1
        raise ValueError("Invalid input")

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    reports = await experiment.run_evaluations_async(non_throttling_error_task, max_workers=1)

    # Should NOT have retried
    assert call_count == 1
    assert len(reports) == 1
    assert reports[0].scores[0] == 0
    assert "Invalid input" in reports[0].reasons[0]


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_exponential_backoff():
    """Test that run_evaluations_async uses exponential backoff"""
    sleep_delays = []

    async def mock_async_sleep(delay):
        sleep_delays.append(delay)

    call_count = 0

    async def throttling_task(c):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ModelThrottledException("Too many tokens")
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator()])

    with patch("asyncio.sleep", mock_async_sleep):
        with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 1):
            with patch("strands_evals.experiment._MAX_RETRY_DELAY", 10):
                await experiment.run_evaluations_async(throttling_task, max_workers=1)

    # Verify exponential backoff: 1, 2, 4
    retry_delays = [d for d in sleep_delays if d >= 1]
    assert len(retry_delays) == 3
    assert retry_delays[0] == 1
    assert retry_delays[1] == 2
    assert retry_delays[2] == 4


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_evaluator_retries():
    """Test that async evaluator execution retries on throttling"""

    class AsyncThrottlingEvaluator(Evaluator[str, str]):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> list[EvaluationOutput]:
            self.call_count += 1
            if self.call_count <= 2:
                raise ModelThrottledException("Too many tokens")
            return [EvaluationOutput(score=1.0, test_pass=True, reason="Success after retry")]

    async def simple_task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    evaluator = AsyncThrottlingEvaluator()
    experiment = Experiment(cases=[case], evaluators=[evaluator])

    with patch("strands_evals.experiment._INITIAL_RETRY_DELAY", 0.01):
        with patch("strands_evals.experiment._MAX_RETRY_DELAY", 0.02):
            reports = await experiment.run_evaluations_async(simple_task, max_workers=1)

    # Evaluator should have been retried
    assert evaluator.call_count == 3
    assert len(reports) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_task_executed_once_with_retry():
    """Test that task is executed once per case even with evaluator retries"""
    task_call_count = 0

    async def counting_task(c):
        nonlocal task_call_count
        task_call_count += 1
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluators=[MockEvaluator(), MockEvaluator2()])

    await experiment.run_evaluations_async(counting_task, max_workers=1)

    # Task should be called once per case, not once per evaluator
    assert task_call_count == 1


def _simulate_agent(case):
    """Simulate an agent that processes queries and uses tools.

    Mimics the real pattern where a task function invokes a strands Agent,
    extracts tool usage from messages, and returns structured output.
    """
    knowledge = {
        "What is the capital of France?": {
            "response": "The capital of France is Paris.",
            "tools_used": ["knowledge_base", "formatter"],
        },
        "What is 2+2?": {
            "response": "2+2 equals 4.",
            "tools_used": ["calculator"],
        },
    }
    result = knowledge.get(case.input, {"response": f"I don't know about: {case.input}", "tools_used": []})
    return {
        "output": result["response"],
        "trajectory": result["tools_used"],
    }


def test_deterministic_evaluator_alongside_mock_evaluator():
    """Test deterministic evaluators alongside LLM-style evaluators in a realistic agent scenario."""
    cases = [
        Case(
            name="geography", input="What is the capital of France?", expected_output="The capital of France is Paris."
        ),
        Case(name="math", input="What is 2+2?", expected_output="2+2 equals 4."),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[Equals(), MockEvaluator()],
    )
    reports = experiment.run_evaluations(_simulate_agent)

    assert len(reports) == 2

    # Equals: both cases match expected_output exactly
    equals_report = reports[0]
    assert equals_report.scores == [1.0, 1.0]
    assert equals_report.test_passes == [True, True]
    assert equals_report.overall_score == 1.0

    # MockEvaluator: also matches (actual==expected)
    mock_report = reports[1]
    assert mock_report.scores == [1.0, 1.0]
    assert mock_report.test_passes == [True, True]


@pytest.mark.asyncio
async def test_deterministic_evaluator_alongside_mock_evaluator_async():
    """Test deterministic evaluators in async experiment with realistic agent task."""
    cases = [
        Case(
            name="geography",
            input="What is the capital of France?",
            expected_output="The capital of France is Paris.",
        ),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[Equals(), MockEvaluator()],
    )
    reports = await experiment.run_evaluations_async(_simulate_agent)

    assert len(reports) == 2
    assert reports[0].scores == [1.0]
    assert reports[0].test_passes == [True]
    assert reports[1].scores == [1.0]
    assert reports[1].test_passes == [True]


def test_multiple_deterministic_evaluators_in_experiment():
    """Test multiple deterministic evaluators validating different aspects of agent output."""
    cases = [
        Case(
            name="geography", input="What is the capital of France?", expected_output="The capital of France is Paris."
        ),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[
            Equals(),
            Contains(value="Paris"),
            StartsWith(value="The capital"),
        ],
    )
    reports = experiment.run_evaluations(_simulate_agent)

    assert len(reports) == 3
    for report in reports:
        assert report.scores == [1.0]
        assert report.test_passes == [True]


def test_tool_called_evaluator_with_trajectory_task():
    """Test ToolCalled evaluator with agent task that returns tool usage trajectory."""
    cases = [
        Case(
            name="math_with_tools",
            input="What is 2+2?",
            expected_output="2+2 equals 4.",
            expected_trajectory=["calculator"],
        ),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[
            ToolCalled(tool_name="calculator"),
            Contains(value="4"),
        ],
    )
    reports = experiment.run_evaluations(_simulate_agent)

    assert len(reports) == 2

    # calculator was called in trajectory
    assert reports[0].scores == [1.0]
    assert reports[0].test_passes == [True]

    # output contains "4"
    assert reports[1].scores == [1.0]
    assert reports[1].test_passes == [True]


def test_tool_called_evaluator_tool_not_found():
    """Test ToolCalled evaluator when agent doesn't use expected tool."""
    cases = [
        Case(
            name="geography_no_calc",
            input="What is the capital of France?",
            expected_trajectory=["calculator"],
        ),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[ToolCalled(tool_name="calculator")],
    )
    reports = experiment.run_evaluations(_simulate_agent)

    assert len(reports) == 1
    # Agent used knowledge_base and formatter, not calculator
    assert reports[0].scores == [0.0]
    assert reports[0].test_passes == [False]


def test_deterministic_evaluator_from_dict_round_trip():
    """Test that Experiment with deterministic evaluators survives to_dict/from_dict."""
    cases = [
        Case(
            name="geography", input="What is the capital of France?", expected_output="The capital of France is Paris."
        ),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[
            Equals(),
            Contains(value="Paris", case_sensitive=False),
            StartsWith(value="The capital"),
            ToolCalled(tool_name="knowledge_base"),
        ],
    )

    data = experiment.to_dict()
    restored = Experiment.from_dict(data)

    assert len(restored.evaluators) == 4
    assert restored.evaluators[0].get_type_name() == "Equals"
    assert restored.evaluators[1].get_type_name() == "Contains"
    assert restored.evaluators[2].get_type_name() == "StartsWith"
    assert restored.evaluators[3].get_type_name() == "ToolCalled"

    # Verify restored evaluators have correct parameters
    assert restored.evaluators[1].value == "Paris"
    assert restored.evaluators[1].case_sensitive is False
    assert restored.evaluators[3].tool_name == "knowledge_base"

    # Run restored experiment against the same agent — results should be identical
    original_reports = experiment.run_evaluations(_simulate_agent)
    restored_reports = restored.run_evaluations(_simulate_agent)

    for orig, rest in zip(original_reports, restored_reports, strict=True):
        assert orig.scores == rest.scores
        assert orig.test_passes == rest.test_passes


def test_deterministic_evaluator_error_isolation():
    """Test that a failing deterministic evaluator doesn't crash other evaluators."""
    cases = [
        Case(
            name="geography", input="What is the capital of France?", expected_output="The capital of France is Paris."
        ),
    ]

    experiment = Experiment(
        cases=cases,
        evaluators=[
            ThrowingEvaluator(),
            Equals(),
        ],
    )
    reports = experiment.run_evaluations(_simulate_agent)

    assert len(reports) == 2

    # ThrowingEvaluator failed with error isolation
    assert reports[0].scores == [0]
    assert reports[0].test_passes == [False]
    assert "Evaluator exploded" in reports[0].reasons[0]

    # Equals still ran successfully despite the ThrowingEvaluator failure
    assert reports[1].scores == [1.0]
    assert reports[1].test_passes == [True]


class DictEvaluationDataStore:
    """Simple in-memory store for testing."""

    def __init__(self):
        self._data: dict[str, EvaluationData] = {}

    def load(self, case_name: str) -> EvaluationData | None:
        return self._data.get(case_name)

    def save(self, case_name: str, result: EvaluationData) -> None:
        self._data[case_name] = result


class TestEvaluationDataStore:
    def test_run_evaluations_with_store_saves_results(self):
        """First run with empty store should execute task and save results."""
        store = DictEvaluationDataStore()
        task_call_count = 0

        def counting_task(c):
            nonlocal task_call_count
            task_call_count += 1
            return c.input

        cases = [Case(name="case1", input="hello", expected_output="hello")]
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        experiment.run_evaluations(counting_task, evaluation_data_store=store)

        assert task_call_count == 1
        # Verify result was saved
        loaded = store.load("case1")
        assert loaded is not None
        assert loaded.actual_output == "hello"

    def test_run_evaluations_with_store_loads_cached_results(self):
        """Second run with populated store should skip task and use cached results."""
        store = DictEvaluationDataStore()

        # Pre-populate the store
        cached_data = EvaluationData(
            input="hello",
            actual_output="hello",
            name="case1",
            expected_output="hello",
        )
        store.save("case1", cached_data)

        task_call_count = 0

        def counting_task(c):
            nonlocal task_call_count
            task_call_count += 1
            return c.input

        cases = [Case(name="case1", input="hello", expected_output="hello")]
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        reports = experiment.run_evaluations(counting_task, evaluation_data_store=store)

        # Task should NOT have been called
        assert task_call_count == 0
        # Evaluators should still run on cached data
        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0

    def test_run_evaluations_with_store_requires_case_names(self):
        """Should raise ValueError if any case lacks a name when store is provided."""
        store = DictEvaluationDataStore()
        cases = [Case(input="hello", expected_output="hello")]  # no name
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        with pytest.raises(ValueError, match="name"):
            experiment.run_evaluations(lambda c: c.input, evaluation_data_store=store)

    def test_run_evaluations_with_store_requires_unique_names(self):
        """Should raise ValueError if case names are not unique when store is provided."""
        store = DictEvaluationDataStore()
        cases = [
            Case(name="dupe", input="a", expected_output="a"),
            Case(name="dupe", input="b", expected_output="b"),
        ]
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        with pytest.raises(ValueError, match="unique"):
            experiment.run_evaluations(lambda c: c.input, evaluation_data_store=store)

    def test_run_evaluations_without_store_unchanged(self):
        """Default behavior without store should be unaffected."""
        cases = [Case(name="case1", input="hello", expected_output="hello")]
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        reports = experiment.run_evaluations(lambda c: c.input)

        assert len(reports) == 1
        assert reports[0].scores[0] == 1.0


class MockTraceProvider(TraceProvider):
    """Simple mock provider for testing."""

    def __init__(self, data: dict[str, dict]):
        self._data = data
        self.call_count = 0
        self.called_session_ids: list[str] = []

    def get_evaluation_data(self, session_id: str) -> dict:
        self.call_count += 1
        self.called_session_ids.append(session_id)
        return self._data[session_id]


class TestProviderIntegration:
    def test_run_evaluations_with_provider(self):
        """provider.as_task() should be called with each case's session_id."""
        cases = [
            Case(name="c1", session_id="sess-1", input="hello", expected_output="hello"),
            Case(name="c2", session_id="sess-2", input="foo", expected_output="foo"),
        ]
        provider = MockTraceProvider(
            {
                "sess-1": {"output": "hello"},
                "sess-2": {"output": "foo"},
            }
        )
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        reports = experiment.run_evaluations(provider.as_task())

        assert provider.call_count == 2
        assert set(provider.called_session_ids) == {"sess-1", "sess-2"}
        assert len(reports) == 1
        assert reports[0].scores == [1.0, 1.0]

    @pytest.mark.asyncio
    async def test_run_evaluations_async_with_provider(self):
        """Async variant should also accept a provider as the task arg."""
        cases = [
            Case(name="c1", session_id="sess-1", input="hello", expected_output="hello"),
        ]
        provider = MockTraceProvider(
            {
                "sess-1": {"output": "hello"},
            }
        )
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        reports = await experiment.run_evaluations_async(provider.as_task())

        assert provider.call_count == 1
        assert provider.called_session_ids == ["sess-1"]
        assert len(reports) == 1
        assert reports[0].scores == [1.0]

    def test_run_evaluations_with_provider_and_data_store_caches(self):
        """When data store has cached data, provider should not be called for that case."""
        store = DictEvaluationDataStore()
        cached_data = EvaluationData(
            input="hello",
            actual_output="hello",
            name="c1",
            expected_output="hello",
        )
        store.save("c1", cached_data)

        provider = MockTraceProvider(
            {
                "sess-1": {"output": "hello"},
            }
        )
        cases = [Case(name="c1", session_id="sess-1", input="hello", expected_output="hello")]
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        reports = experiment.run_evaluations(provider.as_task(), evaluation_data_store=store)

        # Provider should NOT have been called - data was cached
        assert provider.call_count == 0
        assert len(reports) == 1
        assert reports[0].scores == [1.0]

    def test_run_evaluations_with_task_positional_arg_unchanged(self):
        """Existing positional task argument should continue to work."""
        cases = [Case(name="c1", input="hello", expected_output="hello")]
        experiment = Experiment(cases=cases, evaluators=[MockEvaluator()])

        # Positional arg - existing behavior
        reports = experiment.run_evaluations(lambda c: c.input)

        assert len(reports) == 1
        assert reports[0].scores == [1.0]
