"""Custom Evaluator subclass for --custom-evaluator tests."""

from __future__ import annotations

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput


class AlwaysPasses(Evaluator):
    """Evaluator that always returns a passing score. Pure stub for CLI tests."""

    def __init__(self, label: str = "ok"):
        super().__init__()
        self.label = label

    async def evaluate_async(self, data: EvaluationData) -> list[EvaluationOutput]:
        return [EvaluationOutput(score=1.0, test_pass=True, reason=self.label)]

    def evaluate(self, data: EvaluationData) -> list[EvaluationOutput]:
        return [EvaluationOutput(score=1.0, test_pass=True, reason=self.label)]
