"""
Integration tests for InstructionFollowingEvaluator.

These tests make actual API calls to test the full evaluation workflow.
They are separate from unit tests to avoid unnecessary API costs during regular testing.
"""

import pytest

from strands_evals import Case, Experiment
from strands_evals.evaluators import InstructionFollowingEvaluator

PROMPT = "List exactly three primary colors as a comma-separated list, in lowercase, with no other text."

RESPONSES = {
    "followed": "red, yellow, blue",
    "not_followed": (
        "Sure! The primary colors are:\n1. Red\n2. Yellow\n3. Blue\n4. Green\n"
        "These colors can be mixed to create many other colors."
    ),
}


@pytest.mark.asyncio
async def test_instruction_following_evaluator_async(make_single_turn_session):
    """Responses that meet every explicit constraint pass; responses that break them fail."""

    def task(case: Case) -> dict:
        response = RESPONSES[case.name]
        return {"output": response, "trajectory": make_single_turn_session(case.input, response, case.name)}

    test_cases = [
        Case(name="followed", input=PROMPT),
        Case(name="not_followed", input=PROMPT),
    ]

    experiment = Experiment(cases=test_cases, evaluators=[InstructionFollowingEvaluator()])
    report = await experiment.run_evaluations_async(task)

    assert len(report.scores) == 2
    results = {
        case["name"]: (score, passed)
        for case, score, passed in zip(report.cases, report.scores, report.test_passes, strict=True)
    }
    assert results["followed"] == (1.0, True)
    assert results["not_followed"] == (0.0, False)


def test_instruction_following_evaluator_sync(make_single_turn_session):
    """The sync entry point runs the judge end to end."""

    def task(case: Case) -> dict:
        response = RESPONSES["followed"]
        return {"output": response, "trajectory": make_single_turn_session(case.input, response)}

    test_case = Case(name="sync_test", input=PROMPT)

    experiment = Experiment(cases=[test_case], evaluators=[InstructionFollowingEvaluator()])
    report = experiment.run_evaluations(task)

    assert len(report.scores) == 1
    assert report.test_passes[0] is True
    assert report.detailed_results[0][0].label == "Yes"
