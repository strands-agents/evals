import asyncio
from unittest.mock import MagicMock, patch

import pytest
from strands.models.model import Model

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator, InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator
from strands_evals.evaluators.evaluator import DEFAULT_BEDROCK_MODEL_ID
from strands_evals.types import EvaluationData, EvaluationOutput


class MockEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Simple mock: pass if actual equals expected
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return [EvaluationOutput(score=score, test_pass=score > 0.5, reason="Mock evaluation")]

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Add a small delay to simulate async processing
        await asyncio.sleep(0.01)
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return [EvaluationOutput(score=score, test_pass=score > 0.5, reason="Async test evaluation")]


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
    """Test creating an Experiment with test cases and evaluator"""
    cases = [
        Case(name="test1", input="hello", expected_output="world"),
        Case(name="test2", input="foo", expected_output="bar"),
    ]

    experiment = Experiment(cases=cases, evaluator=mock_evaluator)

    assert len(experiment.cases) == 2
    assert experiment.evaluator == mock_evaluator


def test_experiment__init__partial_cases():
    """Test creating an Experiment with test cases only"""
    cases = [
        Case(name="test1", input="hello", expected_output="world"),
        Case(name="test2", input="foo", expected_output="bar"),
    ]

    experiment = Experiment(cases=cases)

    assert len(experiment.cases) == 2


def test_experiment__init__partial_evaluator():
    """Test creating an Experiment with evaluator only"""
    evaluator = Evaluator()
    experiment = Experiment(evaluator=evaluator)

    assert len(experiment.cases) == 0
    assert experiment.evaluator == evaluator


def test_experiment_cases_getter_deep_copy():
    """Test cases getter should return deep copies"""
    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    retrieved = experiment.cases
    retrieved[0].name = "modified"

    assert experiment.cases == [case]


def test_experiment_cases_setter():
    """Test cases setter updates experiment"""
    case1 = Case(name="test1", input="hello", expected_output="world")
    case2 = Case(name="test2", input="hi", expected_output="there")
    experiment = Experiment(cases=[case1], evaluator=MockEvaluator())

    experiment.cases = [case2]
    assert experiment.cases == [case2]


def test_experiment_evaluator_getter():
    """Test evaluator getter returns evaluator"""
    evaluator = MockEvaluator()
    experiment = Experiment(cases=[], evaluator=evaluator)

    retrieved = experiment.evaluator
    assert retrieved == evaluator


def test_experiment_evaluator_setter():
    """Test evaluator setter updates experiment"""
    eval1 = Evaluator()
    eval2 = MockEvaluator()
    experiment = Experiment(cases=[], evaluator=eval1)

    experiment.evaluator = eval2
    assert experiment.evaluator == eval2


def test_experiment__run_task_simple_output(mock_evaluator):
    """Test _run_task with simple output"""
    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=mock_evaluator)

    def simple_task(c):
        return f"response to {c.input}"

    result = experiment._run_task(simple_task, case)

    assert result.input == "hello"
    assert result.actual_output == "response to hello"
    assert result.expected_output == "world"
    assert result.name == "test"
    assert result.expected_trajectory is None
    assert result.actual_trajectory is None
    assert result.metadata is None
    assert result.actual_interactions is None
    assert result.expected_interactions is None


def test_experiment__run_task_dict_output(mock_evaluator):
    """Test _run_task with dictionary output containing trajectory"""
    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=mock_evaluator)

    def dict_task(c):
        return {"output": f"response to {c.input}", "trajectory": ["step1", "step2"]}

    result = experiment._run_task(dict_task, case)

    assert result.actual_output == "response to hello"
    assert result.actual_trajectory == ["step1", "step2"]


def test_experiment_run_task_dict_output_with_interactions(mock_evaluator):
    """Test _run_task with dictionary output containing interactions"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    case = Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluator=mock_evaluator)

    def dict_task(c):
        return {
            "output": f"response to {c.input}",
            "trajectory": ["step1", "step2"],
            "interactions": interactions,
        }

    result = experiment._run_task(dict_task, case)

    assert result.actual_output == "response to hello"
    assert result.actual_trajectory == ["step1", "step2"]
    assert result.actual_interactions == interactions
    assert result.expected_output == "world"
    assert result.expected_trajectory is None
    assert result.expected_interactions == interactions


def test_experiment__run_task_dict_output_with_input_update(mock_evaluator):
    """Test _run_task with dictionary output containing updated input"""
    case = Case(name="test", input="original_input", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=mock_evaluator)

    def task_with_input_update(c):
        return {"output": f"response to {c.input}", "input": "updated_input", "trajectory": ["step1"]}

    result = experiment._run_task(task_with_input_update, case)

    assert result.input == "updated_input"
    assert result.actual_output == "response to original_input"
    assert result.actual_trajectory == ["step1"]


@pytest.mark.asyncio
async def test_experiment__run_task_async_with_input_update():
    """Test _run_task_async with dictionary output containing updated input"""
    case = Case(name="test", input="original_input", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    def task_with_input_update(c):
        return {"output": f"response to {c.input}", "input": "async_updated_input"}

    result = await experiment._run_task_async(task_with_input_update, case)

    assert result.input == "async_updated_input"
    assert result.actual_output == "response to original_input"


def test_experiment__run_task_async_function_raises_error(mock_evaluator):
    """Test _run_task raises ValueError when async task is passed"""
    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=mock_evaluator)

    async def async_task(c):
        return f"response to {c.input}"

    with pytest.raises(ValueError, match="Async task is not supported. Please use run_evaluations_async instead."):
        experiment._run_task(async_task, case)


@pytest.mark.asyncio
async def test_experiment__run_task_async_with_sync_task():
    """Test _run_task_async with a synchronous task function"""

    def sync_task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())
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
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())
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
    experiment = Experiment(cases=cases, evaluator=mock_evaluator)

    def echo_task(c):
        return c.input

    report = experiment.run_evaluations(echo_task)

    assert len(report.scores) == 2
    assert report.scores[0] == 1.0  # match
    assert report.scores[1] == 0.0  # no match
    assert report.test_passes[0] is True
    assert report.test_passes[1] is False
    assert report.overall_score == 0.5
    assert len(report.cases) == 2


def test_experiment_to_dict_empty(mock_evaluator):
    """Test converting empty experiment to dictionary"""
    experiment = Experiment(cases=[], evaluator=mock_evaluator)
    assert experiment.to_dict() == {"cases": [], "evaluator": {"evaluator_type": "MockEvaluator"}}


def test_experiment_to_dict_non_empty(mock_evaluator):
    """Test converting non-empty experiment to dictionary"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    experiment = Experiment(cases=cases, evaluator=mock_evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "MockEvaluator"},
    }


def test_experiment_to_dict_OutputEvaluator_full():
    """Test converting experiment with OutputEvaluator to dictionary with no defaults."""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
    experiment = Experiment(cases=cases, evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {
            "evaluator_type": "OutputEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }


def test_experiment_to_dict_OutputEvaluator_default():
    """Test converting experiment with OutputEvaluator to dictionary with defaults.
    The evaluator's data should not include default information."""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="rubric")
    experiment = Experiment(cases=cases, evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "OutputEvaluator", "rubric": "rubric", "model_id": DEFAULT_BEDROCK_MODEL_ID},
    }


def test_experiment_to_dict_TrajectoryEvaluator_default():
    """Test converting experiment with TrajectoryEvaluator to dictionary with defaults.
    The evaluator's data should not include default information."""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    evaluator = TrajectoryEvaluator(rubric="rubric")
    experiment = Experiment(cases=cases, evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {
            "evaluator_type": "TrajectoryEvaluator",
            "rubric": "rubric",
            "model_id": DEFAULT_BEDROCK_MODEL_ID,
        },
    }


def test_experiment_to_dict_TrajectoryEvaluator_full():
    """Test converting experiment with TrajectoryEvaluator to dictionary with no defaults."""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    evaluator = TrajectoryEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
    experiment = Experiment(cases=cases, evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {
            "evaluator_type": "TrajectoryEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }


def test_experiment_to_dict_InteractionsEvaluator_default():
    """Test converting experiment with InteractionsEvaluator to dictionary with defaults."""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    evaluator = InteractionsEvaluator(rubric="rubric")
    experiment = Experiment(cases=cases, evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {
            "evaluator_type": "InteractionsEvaluator",
            "rubric": "rubric",
            "model_id": DEFAULT_BEDROCK_MODEL_ID,
        },
    }


def test_experiment_to_dict_InteractionsEvaluator_full():
    """Test converting experiment with InteractionsEvaluator to dictionary with no defaults."""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    evaluator = InteractionsEvaluator(
        rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt"
    )
    experiment = Experiment(cases=cases, evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {
            "evaluator_type": "InteractionsEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }


def test_experiment_to_dict_case_dict():
    """Test converting experiment with Case with dictionaries as types."""
    case = Case(name="test", input={"field1": "hello"}, expected_output={"field2": "world"}, metadata={})
    evaluator = MockEvaluator()
    experiment = Experiment(cases=[case], evaluator=evaluator)
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
                "metadata": {},
            }
        ],
        "evaluator": {"evaluator_type": "MockEvaluator"},
    }


def test_experiment_to_dict_case_function():
    """Test converting experiment with Case with function as types."""

    def simple_echo(query):
        return query

    case = Case(name="test", input=simple_echo)
    evaluator = MockEvaluator()
    experiment = Experiment(cases=[case], evaluator=evaluator)
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
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "MockEvaluator"},
    }


def test_experiment_from_dict_custom():
    """Test creating an Experiment with a custom evaluator and empty cases"""
    dict_experiment = {"cases": [], "evaluator": {"evaluator_type": "MockEvaluator"}}
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[MockEvaluator])
    assert experiment.cases == []
    assert isinstance(experiment.evaluator, MockEvaluator)


def test_experiment_from_dict_OutputEvaluator():
    """Test creating an Experiment with a OutputEvaluator"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "OutputEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[OutputEvaluator])
    assert experiment.cases == cases
    assert isinstance(experiment.evaluator, OutputEvaluator)
    assert experiment.evaluator.rubric == "rubric"
    assert experiment.evaluator.model == "model"
    assert experiment.evaluator.include_inputs is False
    assert experiment.evaluator.system_prompt == "system prompt"


def test_experiment_from_dict_OutputEvaluator_defaults():
    """Test creating an Experiment with a OutputEvaluator with defaults"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {"cases": cases, "evaluator": {"evaluator_type": "OutputEvaluator", "rubric": "rubric"}}
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[OutputEvaluator])
    assert experiment.cases == cases
    assert isinstance(experiment.evaluator, OutputEvaluator)
    assert experiment.evaluator.rubric == "rubric"
    assert experiment.evaluator.model is None
    assert experiment.evaluator.include_inputs is True


def test_experiment_from_dict_with_model_id():
    """Test creating an Experiment from dict with model_id (should convert to model parameter)"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    dict_experiment = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "OutputEvaluator",
            "rubric": "test rubric",
            "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        },
    }
    experiment = Experiment.from_dict(dict_experiment)

    assert isinstance(experiment.evaluator, OutputEvaluator)
    assert experiment.evaluator.rubric == "test rubric"
    assert experiment.evaluator.model == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_experiment_to_dict_from_dict_roundtrip_with_model():
    """Test that to_dict and from_dict work correctly for roundtrip with model"""

    # Create experiment with Model instance
    mock_model = MagicMock(spec=Model)
    mock_model.config = {"model_id": "test-model-roundtrip"}

    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="test rubric", model=mock_model)
    experiment = Experiment(cases=cases, evaluator=evaluator)

    # Serialize to dict
    experiment_dict = experiment.to_dict()
    assert experiment_dict["evaluator"]["model_id"] == "test-model-roundtrip"
    assert "model" not in experiment_dict["evaluator"]

    # Deserialize from dict
    restored_experiment = Experiment.from_dict(experiment_dict)
    assert isinstance(restored_experiment.evaluator, OutputEvaluator)
    assert restored_experiment.evaluator.model == "test-model-roundtrip"


def test_experiment_to_dict_from_dict_roundtrip_with_string_model():
    """Test that to_dict and from_dict work correctly for roundtrip with string model"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="test rubric", model="bedrock-model-id")
    experiment = Experiment(cases=cases, evaluator=evaluator)

    # Serialize to dict
    experiment_dict = experiment.to_dict()
    assert experiment_dict["evaluator"]["model"] == "bedrock-model-id"

    # Deserialize from dict
    restored_experiment = Experiment.from_dict(experiment_dict)
    assert isinstance(restored_experiment.evaluator, OutputEvaluator)
    assert restored_experiment.evaluator.model == "bedrock-model-id"


def test_experiment_from_dict_TrajectoryEvaluator():
    """Test creating an Experiment with a TrajectoryEvaluator"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "TrajectoryEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[TrajectoryEvaluator])
    assert experiment.cases == cases
    assert isinstance(experiment.evaluator, TrajectoryEvaluator)
    assert experiment.evaluator.rubric == "rubric"
    assert experiment.evaluator.model == "model"
    assert experiment.evaluator.include_inputs is False
    assert experiment.evaluator.system_prompt == "system prompt"


def test_experiment_from_dict_TrajectoryEvaluator_defaults():
    """Test creating an Experiment with a Trajectory evaluator with defaults"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_experiment = {"cases": cases, "evaluator": {"evaluator_type": "TrajectoryEvaluator", "rubric": "rubric"}}
    experiment = Experiment.from_dict(dict_experiment, custom_evaluators=[TrajectoryEvaluator])
    assert experiment.cases == cases
    assert isinstance(experiment.evaluator, TrajectoryEvaluator)
    assert experiment.evaluator.rubric == "rubric"
    assert experiment.evaluator.model is None
    assert experiment.evaluator.include_inputs is True


def test_experiment_from_dict_InteractionsEvaluator():
    """Test creating an Experiment with an InteractionsEvaluator"""
    interactions = [{"node_name": "agent1", "dependencies": [], "message": "hello"}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    dict_experiment = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "InteractionsEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }
    experiment = Experiment.from_dict(dict_experiment)
    assert experiment.cases == cases
    assert isinstance(experiment.evaluator, InteractionsEvaluator)
    assert experiment.evaluator.rubric == "rubric"
    assert experiment.evaluator.model == "model"
    assert experiment.evaluator.include_inputs is False
    assert experiment.evaluator.system_prompt == "system prompt"


def test_experiment_from_dict_InteractionsEvaluator_defaults():
    """Test creating an Experiment with an Interactions evaluator with defaults"""
    interactions = [{"node_name": "agent1", "dependencies": [], "message": "hello"}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    dict_experiment = {"cases": cases, "evaluator": {"evaluator_type": "InteractionsEvaluator", "rubric": "rubric"}}
    experiment = Experiment.from_dict(dict_experiment)
    assert experiment.cases == cases
    assert isinstance(experiment.evaluator, InteractionsEvaluator)
    assert experiment.evaluator.rubric == "rubric"
    assert experiment.evaluator.model is None
    assert experiment.evaluator.include_inputs is True


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async():
    """Test run_evaluations_async with a simple task"""

    def task(c):
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    experiment = Experiment(cases=[case, case1], evaluator=MockEvaluator())

    report = await experiment.run_evaluations_async(task)

    assert len(report.scores) == 2
    assert all(score == 1.0 for score in report.scores)
    assert all(test_pass for test_pass in report.test_passes)
    assert report.overall_score == 1.0


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_with_async_task():
    """Test run_evaluations_async with an async task"""

    async def async_task(c):
        await asyncio.sleep(0.01)
        return c.input

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    experiment = Experiment(cases=[case, case1], evaluator=MockEvaluator())
    report = await experiment.run_evaluations_async(async_task)

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
    experiment = Experiment(cases=[case, case1], evaluator=MockEvaluator())
    report = await experiment.run_evaluations_async(failing_task)

    assert len(report.scores) == 2
    assert report.scores[0] == 0.0  # case1 fails
    assert report.scores[1] == 1.0  # case2 passes
    assert "Test error" in report.reasons[0]


def test_experiment_run_evaluations_with_interactions():
    """Test evaluation run with interactions data"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["test message"]}]
    case = Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    def task_with_interactions(c):
        return {"output": c.input, "interactions": interactions}

    report = experiment.run_evaluations(task_with_interactions)

    assert len(report.cases) == 1
    assert report.cases[0]["actual_interactions"] == interactions
    assert report.cases[0]["expected_interactions"] == interactions


def test_experiment_init_always_initializes_tracer():
    """Test that Experiment always initializes tracer in __init__"""
    with patch("strands_evals.experiment.get_tracer") as mock_get_tracer:
        mock_tracer = MagicMock()
        mock_get_tracer.return_value = mock_tracer

        experiment = Experiment(cases=[], evaluator=MockEvaluator())

        mock_get_tracer.assert_called_once()
        assert experiment._tracer == mock_tracer


def test_experiment_run_evaluations_creates_case_span(mock_span, simple_task):
    """Test that run_evaluations creates a span for each case with correct attributes"""
    case = Case(name="test_case", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        experiment.run_evaluations(simple_task)

        # Verify evaluator span was created with case information
        calls = mock_start_span.call_args_list
        assert len(calls) == 1
        evaluator_span_call = calls[0]
        assert evaluator_span_call[0][0] == "evaluator MockEvaluator"
        assert "gen_ai.evaluation.case.name" in evaluator_span_call[1]["attributes"]
        assert evaluator_span_call[1]["attributes"]["gen_ai.evaluation.case.name"] == "test_case"


def test_experiment_run_evaluations_creates_task_span(mock_span, simple_task):
    """Test that run_evaluations creates a task span with correct attributes"""
    case = Case(name="test_case", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        experiment.run_evaluations(simple_task)

        # Current implementation only creates evaluator span
        calls = mock_start_span.call_args_list
        assert len(calls) == 1
        evaluator_span_call = calls[0]
        assert evaluator_span_call[0][0] == "evaluator MockEvaluator"
        # Verify set_attributes was called with evaluation results
        mock_span.set_attributes.assert_called()


def test_experiment_run_evaluations_creates_evaluator_span(mock_span, simple_task):
    """Test that run_evaluations creates an evaluator span with correct attributes"""
    case = Case(name="test_case", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        experiment.run_evaluations(simple_task)

        # Verify evaluator span was created
        calls = mock_start_span.call_args_list
        assert len(calls) == 1
        evaluator_span_call = calls[0]
        assert evaluator_span_call[0][0] == "evaluator MockEvaluator"
        assert "gen_ai.evaluation.name" in evaluator_span_call[1]["attributes"]
        assert "gen_ai.evaluation.case.name" in evaluator_span_call[1]["attributes"]
        # Verify set_attributes was called with output attributes
        mock_span.set_attributes.assert_called()


def test_experiment_run_evaluations_with_trajectory_in_span(mock_span):
    """Test that run_evaluations includes trajectory in task span attributes"""
    case = Case(name="test_case", input="hello", expected_output="world")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

        def task_with_trajectory(c):
            return {"output": c.input, "trajectory": ["step1", "step2"]}

        experiment.run_evaluations(task_with_trajectory)

        # Check that set_attributes was called
        mock_span.set_attributes.assert_called()
        # Verify has_trajectory flag is set
        set_attrs_calls = mock_span.set_attributes.call_args_list
        # The attributes dict is the first positional argument in the call
        has_trajectory_set = any(
            "gen_ai.evaluation.data.has_trajectory" in call[0][0] if call[0] and isinstance(call[0][0], dict) else False
            for call in set_attrs_calls
        )
        assert has_trajectory_set, f"has_trajectory not found in set_attributes calls: {set_attrs_calls}"


def test_experiment_run_evaluations_with_interactions_in_span(mock_span):
    """Test that run_evaluations includes interactions in task span attributes"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    case = Case(name="test_case", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

        def task_with_interactions(c):
            return {"output": c.input, "interactions": interactions}

        experiment.run_evaluations(task_with_interactions)

        # Check that set_attributes was called
        mock_span.set_attributes.assert_called()
        # Verify has_interactions flag is set
        set_attrs_calls = mock_span.set_attributes.call_args_list
        # The attributes dict is the first positional argument in the call
        has_interactions_set = any(
            "gen_ai.evaluation.data.has_interactions" in call[0][0]
            if call[0] and isinstance(call[0][0], dict)
            else False
            for call in set_attrs_calls
        )
        assert has_interactions_set


def test_experiment_run_evaluations_records_exception_in_span(mock_span):
    """Test that run_evaluations handles exceptions gracefully"""
    case = Case(name="test_case", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    def failing_task(c):
        raise ValueError("Test error")

    report = experiment.run_evaluations(failing_task)

    # Verify error was handled and report contains error info
    assert len(report.scores) == 1
    assert report.scores[0] == 0
    assert report.test_passes[0] is False
    assert "Test error" in report.reasons[0]


def test_experiment_run_evaluations_with_unnamed_case(mock_span, simple_task):
    """Test that run_evaluations handles unnamed cases correctly"""
    case = Case(input="hello", expected_output="hello")  # No name
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        experiment.run_evaluations(simple_task)

        # Verify evaluator span was created with auto-generated case name
        calls = mock_start_span.call_args_list
        assert len(calls) == 1
        evaluator_span_call = calls[0]
        assert evaluator_span_call[0][0] == "evaluator MockEvaluator"
        # Check that case name was auto-generated
        assert "gen_ai.evaluation.case.name" in evaluator_span_call[1]["attributes"]
        assert evaluator_span_call[1]["attributes"]["gen_ai.evaluation.case.name"] == "case_0"


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_creates_spans(mock_span):
    """Test that run_evaluations_async creates spans with correct attributes"""
    case = Case(name="async_test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:

        async def async_task(c):
            return c.input

        await experiment.run_evaluations_async(async_task)

        # Verify evaluator span was created
        calls = mock_start_span.call_args_list
        assert len(calls) == 1
        evaluator_span_call = calls[0]
        assert evaluator_span_call[0][0] == "evaluator MockEvaluator"


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_records_exception(mock_span):
    """Test that run_evaluations_async records exceptions in spans"""
    case = Case(name="async_test", input="hello", expected_output="hello")
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

        async def failing_async_task(c):
            raise ValueError("Async test error")

        report = await experiment.run_evaluations_async(failing_async_task)

        # Verify the error was handled gracefully
        assert len(report.scores) == 1
        assert report.scores[0] == 0
        assert "Async test error" in report.reasons[0]


@pytest.mark.asyncio
async def test_experiment_run_evaluations_async_with_dict_output(mock_span):
    """Test that run_evaluations_async handles dict output with trajectory/interactions"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    case = Case(name="async_test", input="hello", expected_output="world", expected_interactions=interactions)
    experiment = Experiment(cases=[case], evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span):

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


def test_experiment_run_evaluations_multiple_cases_separate_traces(mock_span, simple_task):
    """Test that each case gets its own separate trace (case span)"""
    cases = [
        Case(name="case1", input="hello", expected_output="hello"),
        Case(name="case2", input="world", expected_output="world"),
    ]
    experiment = Experiment(cases=cases, evaluator=MockEvaluator())

    with patch.object(experiment._tracer, "start_as_current_span", return_value=mock_span) as mock_start_span:
        experiment.run_evaluations(simple_task)

        # Verify we have separate evaluator spans for each case
        calls = mock_start_span.call_args_list
        assert len(calls) == 2
        # Both should be evaluator spans
        assert calls[0][0][0] == "evaluator MockEvaluator"
        assert calls[1][0][0] == "evaluator MockEvaluator"
        # But with different case names
        assert calls[0][1]["attributes"]["gen_ai.evaluation.case.name"] == "case1"
        assert calls[1][1]["attributes"]["gen_ai.evaluation.case.name"] == "case2"
