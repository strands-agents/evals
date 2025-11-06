import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from strands_evals import Case, Dataset
from strands_evals.evaluators import Evaluator, InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator
from strands_evals.types import EvaluationData, EvaluationOutput, Interaction, TaskOutput


class SimpleEvaluator(Evaluator[str, str]):
    """Simple evaluator for integration testing"""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return [EvaluationOutput(score=score, test_pass=score > 0.5, reason="Integration test")]

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        """Async version of evaluate"""
        # Add a small delay to simulate async processing
        await asyncio.sleep(0.01)
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return [EvaluationOutput(score=score, test_pass=score > 0.5, reason="Async integration test")]


@pytest.fixture
def cases():
    return [
        Case(name="exact_match", input="hello", expected_output="hello"),
        Case(name="no_match", input="foo", expected_output="bar"),
        Case(name="partial", input="test", expected_output="test_result"),
    ]


@pytest.fixture
def interaction_case():
    return [
        Case(
            name="interaction_test",
            input="hello",
            expected_output="world",
            expected_interactions=[
                {"node_name": "agent1", "dependencies": [], "messages": ["processing hello"]},
                {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["final result"]},
            ],
        )
    ]


mock_score = 0.8


@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.structured_output.return_value = EvaluationOutput(score=mock_score, test_pass=True, reason="LLM evaluation")
    return agent


@pytest.fixture
def mock_async_agent():
    agent = Mock()
    agent.structured_output_async = AsyncMock(
        return_value=EvaluationOutput(score=mock_score, test_pass=True, reason="Async LLM evaluation")
    )
    return agent


def test_integration_dataset_with_simple_evaluator(cases):
    """Test complete workflow: Dataset + Cases + SimpleEvaluator + EvaluationReport"""
    dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

    def echo_task(case):
        return case.input

    report = dataset.run_evaluations(echo_task)

    # Verify complete workflow
    assert len(report.scores) == 3
    assert report.scores[0] == 1.0  # exact match
    assert report.scores[1] == 0.0  # no match
    assert report.scores[2] == 0.0  # partial no match
    assert report.overall_score == 1.0 / 3
    assert report.test_passes == [True, False, False]
    assert len(report.cases) == 3


def test_integration_dataset_with_dict_output_task(cases):
    """Test Dataset with task returning dictionary output"""
    dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

    def dict_task(case):
        return TaskOutput(
            output=case.input,
            trajectory=["step1", "step2"],
            interactions=[Interaction(node_name="agent1", dependencies=[], messages=["processing hello"])],
        )

    report = dataset.run_evaluations(dict_task)

    assert len(report.scores) == 3
    assert report.scores[0] == 1.0  # exact match
    assert report.scores[1] == 0.0  # no match
    assert report.scores[2] == 0.0  # partial no match
    assert report.overall_score == 1.0 / 3
    assert report.test_passes == [True, False, False]
    assert len(report.cases) == 3


@patch("strands_evals.evaluators.output_evaluator.Agent")
def test_integration_dataset_with_output_evaluator(mock_agent_class, cases, mock_agent):
    """Test Dataset with OutputEvaluator integration"""
    mock_agent_class.return_value = mock_agent

    output_evaluator = OutputEvaluator(rubric="Test if outputs match exactly")
    dataset = Dataset(cases=cases, evaluator=output_evaluator)

    def simple_task(case):
        return f"processed_{case.input}"

    report = dataset.run_evaluations(simple_task)

    # Verify LLM evaluator was called for each test case
    assert mock_agent.structured_output.call_count == 3
    assert len(report.scores) == 3
    assert all(abs(score - mock_score) <= 0.00001 for score in report.scores)
    assert abs(report.overall_score - mock_score) <= 0.00001


def test_integration_evaluation_report_display(cases):
    """Test that EvaluationReport display works with real data"""
    dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

    def mixed_task(case):
        if case.input == "hello":
            return "hello"
        return "different"

    report = dataset.run_evaluations(mixed_task)

    # Test that display method doesn't crash
    try:
        report.display()
        display_success = True
    except Exception:
        display_success = False

    assert display_success
    assert len(report.cases) == 3


@patch("strands_evals.evaluators.trajectory_evaluator.Agent")
def test_integration_dataset_with_trajectory_evaluator(mock_agent_class, cases, mock_agent):
    """Test Dataset with TrajectoryEvaluator integration"""
    mock_agent_class.return_value = mock_agent
    trajectory_evaluator = TrajectoryEvaluator(rubric="Test if trajectories match exactly")
    dataset = Dataset(cases=cases, evaluator=trajectory_evaluator)

    def simple_task(case):
        return {"output": f"processed_{case.input}", "trajectory": ["step1", "step2"]}

    report = dataset.run_evaluations(simple_task)

    # Verify the evaluator was called for each test case
    assert len(report.scores) == 3
    assert all(abs(score - mock_score) <= 0.00001 for score in report.scores)
    assert abs(report.overall_score - mock_score) <= 0.00001


def test_integration_dataset_with_list_inputs():
    """Test Dataset with list inputs"""
    cases = [
        Case(input=["hello", "world"], expected_output=["hello", "world"]),
        Case(input=["foo", "bar"], expected_output=["foo", "bar"]),
    ]
    dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

    def list_task(case):
        return case.input

    report = dataset.run_evaluations(list_task)

    # no error in display
    report.display()
    assert len(report.scores) == 2
    assert report.scores[0] == 1.0  # exact match
    assert report.scores[1] == 1.0  # exact match
    assert report.overall_score == 1.0
    assert report.test_passes == [True, True]
    assert len(report.cases) == 2


@pytest.mark.asyncio
async def test_integration_async_dataset_with_simple_evaluator(cases):
    """Test async workflow: Dataset + Cases + SimpleEvaluator"""
    dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

    def echo_task(case):
        return case.input

    report = await dataset.run_evaluations_async(echo_task)

    # Verify complete workflow
    assert len(report.scores) == 3
    assert report.scores[0] == 1.0  # exact match
    assert report.scores[1] == 0.0  # no match
    assert report.scores[2] == 0.0  # partial no match
    assert report.overall_score == 1.0 / 3
    assert report.test_passes == [True, False, False]
    assert len(report.cases) == 3


@pytest.mark.asyncio
async def test_integration_async_dataset_with_async_task(cases):
    """Test async workflow with async task function"""
    dataset = Dataset(cases=cases, evaluator=SimpleEvaluator())

    async def async_echo_task(case):
        await asyncio.sleep(0.01)  # Simulate async work
        return case.input

    report = await dataset.run_evaluations_async(async_echo_task)

    # Verify complete workflow
    assert len(report.scores) == 3
    assert report.scores[0] == 1.0  # exact match
    assert report.scores[1] == 0.0  # no match
    assert report.scores[2] == 0.0  # partial no match
    assert report.overall_score == 1.0 / 3
    assert report.test_passes == [True, False, False]
    assert len(report.cases) == 3


@pytest.mark.asyncio
@patch("strands_evals.evaluators.output_evaluator.Agent")
async def test_integration_async_dataset_with_output_evaluator(mock_agent_class, cases, mock_async_agent):
    """Test async Dataset with OutputEvaluator integration"""
    mock_agent_class.return_value = mock_async_agent

    output_evaluator = OutputEvaluator(rubric="Test if outputs match exactly")
    dataset = Dataset(cases=cases, evaluator=output_evaluator)

    def simple_task(case):
        return f"processed_{case.input}"

    report = await dataset.run_evaluations_async(simple_task)

    # Verify results
    assert len(report.scores) == 3
    assert all(abs(score - mock_score) <= 0.00001 for score in report.scores)
    assert abs(report.overall_score - mock_score) <= 0.00001


@pytest.mark.asyncio
async def test_integration_async_dataset_concurrency():
    """Test that async evaluations run concurrently"""
    # Create a dataset with more test cases
    many_cases = [Case(name=f"case{i}", input=f"input{i}", expected_output=f"input{i}") for i in range(10)]
    dataset = Dataset(cases=many_cases, evaluator=SimpleEvaluator())

    # Create a task with noticeable delay
    async def slow_task(case):
        await asyncio.sleep(0.1)  # Each task takes 0.1s
        return case.input

    # Time the execution
    start_time = asyncio.get_event_loop().time()
    report = await dataset.run_evaluations_async(slow_task, max_workers=5)
    end_time = asyncio.get_event_loop().time()

    # With 10 tasks taking 0.1s each and 5 workers, should take ~0.2s
    # (two batches of 5 tasks), not 1.0s (if sequential)
    assert end_time - start_time < 0.5  # Allow some overhead

    # Verify results
    assert len(report.scores) == 10
    assert all(score == 1.0 for score in report.scores)
    assert report.overall_score == 1.0


@patch("strands_evals.evaluators.interactions_evaluator.Agent")
def test_dataset_with_interactions_evaluator(mock_agent_class, interaction_case, mock_agent):
    """Test Dataset with InteractionsEvaluator integration"""
    mock_agent_class.return_value = mock_agent
    interactions_evaluator = InteractionsEvaluator(rubric="Test if interactions match expected sequence")
    dataset = Dataset(cases=interaction_case, evaluator=interactions_evaluator)

    def task_with_interactions(case):
        return {
            "output": "world",
            "interactions": [
                {"node_name": "agent1", "dependencies": [], "messages": ["processing hello"]},
                {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["final result"]},
            ],
        }

    report = dataset.run_evaluations(task_with_interactions)

    # Verify the evaluator was called (once per interaction, so 2 times)
    assert mock_agent.structured_output.call_count == 2
    assert len(report.scores) == 1
    assert abs(report.scores[0] - mock_score) <= 0.00001
    assert abs(report.overall_score - mock_score) <= 0.00001


@pytest.mark.asyncio
async def test_async_dataset_with_interactions(interaction_case):
    """Test async Dataset with interactions data"""
    dataset = Dataset(cases=interaction_case, evaluator=SimpleEvaluator())

    async def async_interactions_task(case):
        await asyncio.sleep(0.01)
        return {
            "output": "world",
            "interactions": [
                {"node_name": "agent1", "dependencies": [], "messages": ["processing hello"]},
                {"node_name": "agent2", "dependencies": ["agent1"], "messages": ["final result"]},
            ],
        }

    report = await dataset.run_evaluations_async(async_interactions_task)

    assert len(report.scores) == 1
    assert len(report.cases) == 1
    assert report.cases[0].get("actual_interactions") is not None
    assert len(report.cases[0].get("actual_interactions")) == 2
