import pytest

from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput


class SimpleEvaluator(Evaluator[str, str]):
    """Simple implementation for testing"""

    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        score = 1.0 if evaluation_case.actual_output == evaluation_case.expected_output else 0.0
        return EvaluationOutput(score=score, test_pass=score > 0.5, reason="Test evaluation")

    async def evaluate_async(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        return self.evaluate(evaluation_case)


@pytest.fixture
def evaluation_data():
    return EvaluationData(input="test input", actual_output="actual", expected_output="expected", name="case")


def test_evaluator_not_implemented_evaluate(evaluation_data):
    """Test that base Evaluator raises NotImplementedError evaluate"""
    evaluator = Evaluator[str, str]()

    with pytest.raises(NotImplementedError, match="This method should be implemented in subclasses"):
        evaluator.evaluate(evaluation_data)


@pytest.mark.asyncio
async def test_evaluator_not_implemented_evaluate_async(evaluation_data):
    """Test that base Evaluator raises NotImplementedError for evaluate_async"""
    evaluator = Evaluator[str, str]()

    with pytest.raises(
        NotImplementedError,
        match="This method should be implemented in subclasses,"
        " especially if you want to run evaluations asynchronously.",
    ):
        await evaluator.evaluate_async(evaluation_data)


def test_evaluator_custom_implementation(evaluation_data):
    """Test that simple implementation works"""
    evaluator = SimpleEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert isinstance(result, EvaluationOutput)
    assert result.score == 0.0  # evaluation data has conflicting outputs
    assert result.test_pass is False
    assert result.reason == "Test evaluation"


@pytest.mark.asyncio
async def test_evaluator_custom_evaluate_async(evaluation_data):
    """Test that simple implementation works for evaluate_async"""
    evaluator = SimpleEvaluator()
    result = await evaluator.evaluate_async(evaluation_data)

    assert isinstance(result, EvaluationOutput)
    assert result.score == 0.0  # evaluation data has conflicting outputs
    assert result.test_pass is False
    assert result.reason == "Test evaluation"


def test_evaluator_to_dict():
    """Test that evaluator to_dict works properly"""
    evaluator = Evaluator()
    evaluator_dict = evaluator.to_dict()
    assert evaluator_dict["evaluator_type"] == "Evaluator"
    assert len(evaluator_dict) == 1
