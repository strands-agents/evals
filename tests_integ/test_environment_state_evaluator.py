"""
Integration tests for EnvironmentStateEvaluator.

These tests make actual API calls to test the full evaluation workflow.
They are separate from unit tests to avoid unnecessary API costs during regular testing.
"""

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import EnvironmentStateEvaluator, StateEquals
from strands_evals.types import EnvironmentState


@pytest.mark.asyncio
async def test_environment_state_evaluator_matching_state():
    """Test EnvironmentStateEvaluator with environment state that matches expectations."""

    def task_with_state(_case: Case) -> dict:
        return {
            "output": "Tests passed",
            "environment_state": [EnvironmentState(name="test_results", state={"exit_code": 0, "passed": 3})],
        }

    test_case = Case(
        name="passing_tests",
        input="Fix the failing test in auth.py",
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
    )

    evaluator = EnvironmentStateEvaluator(
        rubric="Score 1.0 if all tests pass (exit_code is 0). Score 0.0 if any tests fail.",
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(task_with_state)

    assert len(reports[0].scores) == 1
    assert reports[0].test_passes[0] is True
    assert reports[0].scores[0] >= 0.8


@pytest.mark.asyncio
async def test_environment_state_evaluator_mismatched_state():
    """Test EnvironmentStateEvaluator with environment state that does not match expectations."""

    def task_with_failing_state(_case: Case) -> dict:
        return {
            "output": "Attempted fix",
            "environment_state": [EnvironmentState(name="test_results", state={"exit_code": 1, "failed": 2})],
        }

    test_case = Case(
        name="failing_tests",
        input="Fix the failing test",
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
    )

    evaluator = EnvironmentStateEvaluator(
        rubric="Score 1.0 if all tests pass (exit_code is 0). Score 0.0 if any tests fail.",
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(task_with_failing_state)

    assert len(reports[0].scores) == 1
    assert reports[0].test_passes[0] is False
    assert reports[0].scores[0] <= 0.3


def test_environment_state_evaluator_sync():
    """Test EnvironmentStateEvaluator with synchronous evaluation."""

    def task_with_state(_case: Case) -> dict:
        return {
            "output": "File created",
            "environment_state": [EnvironmentState(name="file_system", state={"created": ["output.txt"]})],
        }

    test_case = Case(
        name="file_creation",
        input="Create output.txt",
        expected_environment_state=[EnvironmentState(name="file_system", state={"created": ["output.txt"]})],
    )

    evaluator = EnvironmentStateEvaluator(
        rubric="Score 1.0 if the expected file was created. Score 0.0 otherwise.",
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = experiment.run_evaluations(task_with_state)

    assert len(reports[0].scores) == 1
    assert reports[0].test_passes[0] is True
    assert reports[0].scores[0] >= 0.8


@pytest.mark.asyncio
async def test_environment_state_evaluator_without_output():
    """Test EnvironmentStateEvaluator when task returns only environment state, no output."""

    def side_effect_only_task(_case: Case) -> dict:
        return {
            "environment_state": [EnvironmentState(name="db", state={"rows_inserted": 1})],
        }

    test_case = Case(
        name="side_effect_only",
        input="Insert a row into the database",
        expected_environment_state=[EnvironmentState(name="db", state={"rows_inserted": 1})],
    )

    evaluator = EnvironmentStateEvaluator(
        rubric="Score 1.0 if the expected database row was inserted. Score 0.0 otherwise.",
        include_inputs=True,
    )

    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(side_effect_only_task)

    assert len(reports[0].scores) == 1
    assert reports[0].test_passes[0] is True
    assert reports[0].scores[0] >= 0.8


def test_state_equals_matching_via_expected():
    """Test StateEquals through the full Experiment pipeline using expected_environment_state."""

    def task_with_state(_case: Case) -> dict:
        return {
            "environment_state": [EnvironmentState(name="test_results", state={"exit_code": 0, "passed": 5})],
        }

    test_case = Case(
        name="state_equals_expected",
        input="Run tests",
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0, "passed": 5})],
    )

    evaluator = StateEquals(name="test_results")
    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = experiment.run_evaluations(task_with_state)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


def test_state_equals_matching_via_explicit_value():
    """Test StateEquals with an explicit value parameter."""

    def task_with_state(_case: Case) -> dict:
        return {
            "environment_state": [EnvironmentState(name="exit_code", state=0)],
        }

    test_case = Case(
        name="state_equals_explicit",
        input="Run tests",
    )

    evaluator = StateEquals(name="exit_code", value=0)
    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = experiment.run_evaluations(task_with_state)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


def test_state_equals_mismatch():
    """Test StateEquals when actual state doesn't match expected."""

    def task_with_state(_case: Case) -> dict:
        return {
            "environment_state": [EnvironmentState(name="test_results", state={"exit_code": 1})],
        }

    test_case = Case(
        name="state_equals_mismatch",
        input="Run tests",
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
    )

    evaluator = StateEquals(name="test_results")
    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = experiment.run_evaluations(task_with_state)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] == 0.0
    assert reports[0].test_passes[0] is False


@pytest.mark.asyncio
async def test_state_equals_async_with_multiple_states():
    """Test StateEquals async with multiple environment states, selecting the correct one."""

    def task_with_multiple_states(_case: Case) -> dict:
        return {
            "environment_state": [
                EnvironmentState(name="file_system", state={"created": ["a.txt"]}),
                EnvironmentState(name="test_results", state={"exit_code": 0}),
                EnvironmentState(name="db", state={"rows": 3}),
            ],
        }

    test_case = Case(
        name="multi_state",
        input="Do work",
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
    )

    evaluator = StateEquals(name="test_results")
    experiment = Experiment(cases=[test_case], evaluators=[evaluator])
    reports = await experiment.run_evaluations_async(task_with_multiple_states)

    assert len(reports[0].scores) == 1
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True


def test_state_equals_combined_with_environment_state_evaluator():
    """Test StateEquals alongside EnvironmentStateEvaluator in the same experiment."""

    def task_with_state(_case: Case) -> dict:
        return {
            "output": "Fixed the bug",
            "environment_state": [EnvironmentState(name="test_results", state={"exit_code": 0, "passed": 3})],
        }

    test_case = Case(
        name="combined_evaluators",
        input="Fix the test",
        expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0, "passed": 3})],
    )

    deterministic = StateEquals(name="test_results")
    llm_judge = EnvironmentStateEvaluator(
        rubric="Score 1.0 if all tests pass (exit_code is 0). Score 0.0 if any tests fail.",
    )

    experiment = Experiment(cases=[test_case], evaluators=[deterministic, llm_judge])
    reports = experiment.run_evaluations(task_with_state)

    # deterministic evaluator
    assert reports[0].scores[0] == 1.0
    assert reports[0].test_passes[0] is True
    # LLM judge evaluator
    assert reports[1].scores[0] >= 0.8
    assert reports[1].test_passes[0] is True
