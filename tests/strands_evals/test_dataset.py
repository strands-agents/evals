import asyncio

import pytest

from strands_evals import Case, Dataset
from strands_evals.evaluators import Evaluator, InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput


class MockEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Simple mock: pass if actual equals expected
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Mock evaluation")

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Add a small delay to simulate async processing
        await asyncio.sleep(0.01)
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Async test evaluation")


@pytest.fixture
def mock_evaluator():
    return MockEvaluator()


def test_dataset__init__full(mock_evaluator):
    """Test creating a Dataset with test cases and evaluator"""
    cases = [
        Case(name="test1", input="hello", expected_output="world"),
        Case(name="test2", input="foo", expected_output="bar"),
    ]

    dataset = Dataset(cases=cases, evaluator=mock_evaluator)

    assert len(dataset.cases) == 2
    assert dataset.evaluator == mock_evaluator


def test_dataset__init__partial_cases():
    """Test creating a Dataset with test cases only"""
    cases = [
        Case(name="test1", input="hello", expected_output="world"),
        Case(name="test2", input="foo", expected_output="bar"),
    ]

    dataset = Dataset(cases=cases)

    assert len(dataset.cases) == 2


def test_dataset__init__partial_evaluator():
    """Test creating a Dataset with evaluator only"""
    evaluator = Evaluator()
    dataset = Dataset(evaluator=evaluator)

    assert len(dataset.cases) == 0
    assert dataset.evaluator == evaluator


def test_dataset_cases_getter_deep_copy():
    """Test cases getter should return deep copies"""
    case = Case(name="test", input="hello", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=MockEvaluator())

    retrieved = dataset.cases
    retrieved[0].name = "modified"

    assert dataset.cases == [case]


def test_dataset_cases_setter():
    """Test cases setter updates dataset"""
    case1 = Case(name="test1", input="hello", expected_output="world")
    case2 = Case(name="test2", input="hi", expected_output="there")
    dataset = Dataset(cases=[case1], evaluator=MockEvaluator())

    dataset.cases = [case2]
    assert dataset.cases == [case2]


def test_dataset_evaluator_getter():
    """Test evaluator getter returns evaluator"""
    evaluator = MockEvaluator()
    dataset = Dataset(cases=[], evaluator=evaluator)

    retrieved = dataset.evaluator
    assert retrieved == evaluator


def test_dataset_evaluator_setter():
    """Test evaluator setter updates dataset"""
    eval1 = Evaluator()
    eval2 = MockEvaluator()
    dataset = Dataset(cases=[], evaluator=eval1)

    dataset.evaluator = eval2
    assert dataset.evaluator == eval2


def test_dataset__run_task_simple_output(mock_evaluator):
    """Test _run_task with simple output"""
    case = Case(name="test", input="hello", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=mock_evaluator)

    def simple_task(input_val):
        return f"response to {input_val}"

    result = dataset._run_task(simple_task, case)

    assert result.input == "hello"
    assert result.actual_output == "response to hello"
    assert result.expected_output == "world"
    assert result.name == "test"
    assert result.expected_trajectory is None
    assert result.actual_trajectory is None
    assert result.metadata is None
    assert result.actual_interactions is None
    assert result.expected_interactions is None


def test_dataset__run_task_dict_output(mock_evaluator):
    """Test _run_task with dictionary output containing trajectory"""
    case = Case(name="test", input="hello", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=mock_evaluator)

    def dict_task(input_val):
        return {"output": f"response to {input_val}", "trajectory": ["step1", "step2"]}

    result = dataset._run_task(dict_task, case)

    assert result.actual_output == "response to hello"
    assert result.actual_trajectory == ["step1", "step2"]


def test_dataset_run_task_dict_output_with_interactions(mock_evaluator):
    """Test _run_task with dictionary output containing interactions"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    case = Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)
    dataset = Dataset(cases=[case], evaluator=mock_evaluator)

    def dict_task(input_val):
        return {
            "output": f"response to {input_val}",
            "trajectory": ["step1", "step2"],
            "interactions": interactions,
        }

    result = dataset._run_task(dict_task, case)

    assert result.actual_output == "response to hello"
    assert result.actual_trajectory == ["step1", "step2"]
    assert result.actual_interactions == interactions
    assert result.expected_output == "world"
    assert result.expected_trajectory is None
    assert result.expected_interactions == interactions


def test_dataset__run_task_dict_output_with_input_update(mock_evaluator):
    """Test _run_task with dictionary output containing updated input"""
    case = Case(name="test", input="original_input", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=mock_evaluator)

    def task_with_input_update(input_val):
        return {"output": f"response to {input_val}", "input": "updated_input", "trajectory": ["step1"]}

    result = dataset._run_task(task_with_input_update, case)

    assert result.input == "updated_input"
    assert result.actual_output == "response to original_input"
    assert result.actual_trajectory == ["step1"]


@pytest.mark.asyncio
async def test_dataset__run_task_async_with_input_update():
    """Test _run_task_async with dictionary output containing updated input"""
    case = Case(name="test", input="original_input", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=MockEvaluator())

    def task_with_input_update(input_val):
        return {"output": f"response to {input_val}", "input": "async_updated_input"}

    result = await dataset._run_task_async(task_with_input_update, case)

    assert result.input == "async_updated_input"
    assert result.actual_output == "response to original_input"


def test_dataset__run_task_async_function_raises_error(mock_evaluator):
    """Test _run_task raises ValueError when async task is passed"""
    case = Case(name="test", input="hello", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=mock_evaluator)

    async def async_task(input_val):
        return f"response to {input_val}"

    with pytest.raises(ValueError, match="Async task is not supported. Please use run_evaluations_async instead."):
        dataset._run_task(async_task, case)


@pytest.mark.asyncio
async def test_dataset__run_task_async_with_sync_task():
    """Test _run_task_async with a synchronous task function"""

    def sync_task(input_val):
        return input_val

    case = Case(name="test", input="hello", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=MockEvaluator())
    evaluation_context = await dataset._run_task_async(sync_task, case)
    assert evaluation_context.input == "hello"
    assert evaluation_context.actual_output == "hello"
    assert evaluation_context.expected_output == "world"


@pytest.mark.asyncio
async def test_dataset__run_task_async_with_async_task():
    """Test _run_task_async with an asynchronous task function"""

    async def async_task(input_val):
        return input_val

    case = Case(name="test", input="hello", expected_output="world")
    dataset = Dataset(cases=[case], evaluator=MockEvaluator())
    evaluation_context = await dataset._run_task_async(async_task, case)
    assert evaluation_context.input == "hello"
    assert evaluation_context.actual_output == "hello"
    assert evaluation_context.expected_output == "world"


def test_dataset_run_evaluations(mock_evaluator):
    """Test complete evaluation run"""
    cases = [
        Case(name="match", input="hello", expected_output="hello"),
        Case(name="no_match", input="foo", expected_output="bar"),
    ]
    dataset = Dataset(cases=cases, evaluator=mock_evaluator)

    def echo_task(input_val):
        return input_val

    report = dataset.run_evaluations(echo_task)

    assert len(report.scores) == 2
    assert report.scores[0] == 1.0  # match
    assert report.scores[1] == 0.0  # no match
    assert report.test_passes[0] is True
    assert report.test_passes[1] is False
    assert report.overall_score == 0.5
    assert len(report.cases) == 2


def test_dataset_to_dict_empty(mock_evaluator):
    """Test converting empty dataset to dictionary"""
    dataset = Dataset(cases=[], evaluator=mock_evaluator)
    assert dataset.to_dict() == {"cases": [], "evaluator": {"evaluator_type": "MockEvaluator"}}


def test_dataset_to_dict_non_empty(mock_evaluator):
    """Test converting non-empty dataset to dictionary"""
    cases = [Case(name="test", input="hello", expected_output="world")]
    dataset = Dataset(cases=cases, evaluator=mock_evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": None,
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "MockEvaluator"},
    }


def test_dataset_to_dict_OutputEvaluator_full():
    """Test converting dataset with OutputEvaluator to dictionary with no defaults."""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
    dataset = Dataset(cases=cases, evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
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


def test_dataset_to_dict_OutputEvaluator_default():
    """Test converting dataset with OutputEvaluator to dictionary with defaults.
    The evaluator's data should not include default information."""
    cases = [Case(name="test", input="hello", expected_output="world")]
    evaluator = OutputEvaluator(rubric="rubric")
    dataset = Dataset(cases=cases, evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": None,
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "OutputEvaluator", "rubric": "rubric"},
    }


def test_dataset_to_dict_TrajectoryEvaluator_default():
    """Test converting dataset with TrajectoryEvaluator to dictionary with defaults.
    The evaluator's data should not include default information."""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    evaluator = TrajectoryEvaluator(rubric="rubric")
    dataset = Dataset(cases=cases, evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": ["step1", "step2"],
                "expected_interactions": None,
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "TrajectoryEvaluator", "rubric": "rubric"},
    }


def test_dataset_to_dict_TrajectoryEvaluator_full():
    """Test converting dataset with TrajectoryEvaluator to dictionary with no defaults."""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    evaluator = TrajectoryEvaluator(rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt")
    dataset = Dataset(cases=cases, evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
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


def test_dataset_to_dict_InteractionsEvaluator_default():
    """Test converting dataset with InteractionsEvaluator to dictionary with defaults."""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    evaluator = InteractionsEvaluator(rubric="rubric")
    dataset = Dataset(cases=cases, evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
                "input": "hello",
                "expected_output": "world",
                "expected_trajectory": None,
                "expected_interactions": interactions,
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "InteractionsEvaluator", "rubric": "rubric"},
    }


def test_dataset_to_dict_InteractionsEvaluator_full():
    """Test converting dataset with InteractionsEvaluator to dictionary with no defaults."""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["hello"]}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    evaluator = InteractionsEvaluator(
        rubric="rubric", model="model", include_inputs=False, system_prompt="system prompt"
    )
    dataset = Dataset(cases=cases, evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
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


def test_dataset_to_dict_case_dict():
    """Test converting dataset with Case with dictionaries as types."""
    case = Case(name="test", input={"field1": "hello"}, expected_output={"field2": "world"}, metadata={})
    evaluator = MockEvaluator()
    dataset = Dataset(cases=[case], evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
                "input": {"field1": "hello"},
                "expected_output": {"field2": "world"},
                "expected_trajectory": None,
                "expected_interactions": None,
                "metadata": {},
            }
        ],
        "evaluator": {"evaluator_type": "MockEvaluator"},
    }


def test_dataset_to_dict_case_function():
    """Test converting dataset with Case with function as types."""

    def simple_echo(query):
        return query

    case = Case(name="test", input=simple_echo)
    evaluator = MockEvaluator()
    dataset = Dataset(cases=[case], evaluator=evaluator)
    assert dataset.to_dict() == {
        "cases": [
            {
                "name": "test",
                "input": simple_echo,
                "expected_output": None,
                "expected_trajectory": None,
                "expected_interactions": None,
                "metadata": None,
            }
        ],
        "evaluator": {"evaluator_type": "MockEvaluator"},
    }


def test_dataset_from_dict_custom():
    """Test creating a Dataset with a custom evaluator and empty cases"""
    dict_dataset = {"cases": [], "evaluator": {"evaluator_type": "MockEvaluator"}}
    dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[MockEvaluator])
    assert dataset.cases == []
    assert isinstance(dataset.evaluator, MockEvaluator)


def test_dataset_from_dict_OutputEvaluator():
    """Test creating a Dataset with a OutputEvaluator"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_dataset = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "OutputEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }
    dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[OutputEvaluator])
    assert dataset.cases == cases
    assert isinstance(dataset.evaluator, OutputEvaluator)
    assert dataset.evaluator.rubric == "rubric"
    assert dataset.evaluator.model == "model"
    assert dataset.evaluator.include_inputs is False
    assert dataset.evaluator.system_prompt == "system prompt"


def test_dataset_from_dict_OutputEvaluator_defaults():
    """Test creating a Dataset with a OutputEvaluator with defaults"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_dataset = {"cases": cases, "evaluator": {"evaluator_type": "OutputEvaluator", "rubric": "rubric"}}
    dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[OutputEvaluator])
    assert dataset.cases == cases
    assert isinstance(dataset.evaluator, OutputEvaluator)
    assert dataset.evaluator.rubric == "rubric"
    assert dataset.evaluator.model is None
    assert dataset.evaluator.include_inputs is True


def test_dataset_from_dict_TrajectoryEvaluator():
    """Test creating a Dataset with a TrajectoryEvaluator"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_dataset = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "TrajectoryEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }
    dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[TrajectoryEvaluator])
    assert dataset.cases == cases
    assert isinstance(dataset.evaluator, TrajectoryEvaluator)
    assert dataset.evaluator.rubric == "rubric"
    assert dataset.evaluator.model == "model"
    assert dataset.evaluator.include_inputs is False
    assert dataset.evaluator.system_prompt == "system prompt"


def test_dataset_from_dict_TrajectoryEvaluator_defaults():
    """Test creating a Dataset with a Trajectory evaluator with defaults"""
    cases = [Case(name="test", input="hello", expected_output="world", expected_trajectory=["step1", "step2"])]
    dict_dataset = {"cases": cases, "evaluator": {"evaluator_type": "TrajectoryEvaluator", "rubric": "rubric"}}
    dataset = Dataset.from_dict(dict_dataset, custom_evaluators=[TrajectoryEvaluator])
    assert dataset.cases == cases
    assert isinstance(dataset.evaluator, TrajectoryEvaluator)
    assert dataset.evaluator.rubric == "rubric"
    assert dataset.evaluator.model is None
    assert dataset.evaluator.include_inputs is True


def test_dataset_from_dict_InteractionsEvaluator():
    """Test creating a Dataset with an InteractionsEvaluator"""
    interactions = [{"node_name": "agent1", "dependencies": [], "message": "hello"}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    dict_dataset = {
        "cases": cases,
        "evaluator": {
            "evaluator_type": "InteractionsEvaluator",
            "rubric": "rubric",
            "model": "model",
            "include_inputs": False,
            "system_prompt": "system prompt",
        },
    }
    dataset = Dataset.from_dict(dict_dataset)
    assert dataset.cases == cases
    assert isinstance(dataset.evaluator, InteractionsEvaluator)
    assert dataset.evaluator.rubric == "rubric"
    assert dataset.evaluator.model == "model"
    assert dataset.evaluator.include_inputs is False
    assert dataset.evaluator.system_prompt == "system prompt"


def test_dataset_from_dict_InteractionsEvaluator_defaults():
    """Test creating a Dataset with an Interactions evaluator with defaults"""
    interactions = [{"node_name": "agent1", "dependencies": [], "message": "hello"}]
    cases = [Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)]
    dict_dataset = {"cases": cases, "evaluator": {"evaluator_type": "InteractionsEvaluator", "rubric": "rubric"}}
    dataset = Dataset.from_dict(dict_dataset)
    assert dataset.cases == cases
    assert isinstance(dataset.evaluator, InteractionsEvaluator)
    assert dataset.evaluator.rubric == "rubric"
    assert dataset.evaluator.model is None
    assert dataset.evaluator.include_inputs is True


@pytest.mark.asyncio
async def test_dataset_run_evaluations_async():
    """Test run_evaluations_async with a simple task"""

    def task(input_str):
        return input_str

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    dataset = Dataset(cases=[case, case1], evaluator=MockEvaluator())

    report = await dataset.run_evaluations_async(task)

    assert len(report.scores) == 2
    assert all(score == 1.0 for score in report.scores)
    assert all(test_pass for test_pass in report.test_passes)
    assert report.overall_score == 1.0


@pytest.mark.asyncio
async def test_dataset_run_evaluations_async_with_async_task():
    """Test run_evaluations_async with an async task"""

    async def async_task(input_str):
        await asyncio.sleep(0.01)
        return input_str

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    dataset = Dataset(cases=[case, case1], evaluator=MockEvaluator())
    report = await dataset.run_evaluations_async(async_task)

    assert len(report.scores) == 2
    assert all(score == 1.0 for score in report.scores)
    assert all(test_pass for test_pass in report.test_passes)
    assert report.overall_score == 1.0


@pytest.mark.asyncio
async def test_datset_run_evaluations_async_with_errors():
    """Test run_evaluations_async handles errors gracefully"""

    def failing_task(input_str):
        if input_str == "hello":
            raise ValueError("Test error")
        return input_str

    case = Case(name="test", input="hello", expected_output="hello")
    case1 = Case(name="test1", input="world", expected_output="world")
    dataset = Dataset(cases=[case, case1], evaluator=MockEvaluator())
    report = await dataset.run_evaluations_async(failing_task)

    assert len(report.scores) == 2
    assert report.scores[0] == 0.0  # case1 fails
    assert report.scores[1] == 1.0  # case2 passes
    assert "Test error" in report.reasons[0]


def test_dataset_run_evaluations_with_interactions():
    """Test evaluation run with interactions data"""
    interactions = [{"node_name": "agent1", "dependencies": [], "messages": ["test message"]}]
    case = Case(name="test", input="hello", expected_output="world", expected_interactions=interactions)
    dataset = Dataset(cases=[case], evaluator=MockEvaluator())

    def task_with_interactions(input_val):
        return {"output": input_val, "interactions": interactions}

    report = dataset.run_evaluations(task_with_interactions)

    assert len(report.cases) == 1
    assert report.cases[0]["actual_interactions"] == interactions
    assert report.cases[0]["expected_interactions"] == interactions
