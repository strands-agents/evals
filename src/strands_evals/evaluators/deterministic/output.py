from typing_extensions import Any

from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator


class Equals(Evaluator[InputT, OutputT]):
    """Checks if actual_output equals an expected value."""

    def __init__(self, value: Any | None = None):
        super().__init__()
        self.value = value

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        expected = self.value if self.value is not None else evaluation_case.expected_output
        match = evaluation_case.actual_output == expected
        return [
            EvaluationOutput(
                score=1.0 if match else 0.0,
                test_pass=match,
                reason=f"actual_output {'matches' if match else 'does not match'} expected value",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)


class Contains(Evaluator[InputT, OutputT]):
    """Checks if actual_output contains a substring."""

    def __init__(self, value: str, case_sensitive: bool = True):
        super().__init__()
        self.value = value
        self.case_sensitive = case_sensitive

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        actual = str(evaluation_case.actual_output)
        target = self.value
        if not self.case_sensitive:
            actual = actual.lower()
            target = target.lower()
        found = target in actual
        return [
            EvaluationOutput(
                score=1.0 if found else 0.0,
                test_pass=found,
                reason=f"actual_output {'contains' if found else 'does not contain'} '{self.value}'",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)


class StartsWith(Evaluator[InputT, OutputT]):
    """Checks if actual_output starts with a prefix."""

    def __init__(self, value: str, case_sensitive: bool = True):
        super().__init__()
        self.value = value
        self.case_sensitive = case_sensitive

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        actual = str(evaluation_case.actual_output)
        target = self.value
        if not self.case_sensitive:
            actual = actual.lower()
            target = target.lower()
        found = actual.startswith(target)
        return [
            EvaluationOutput(
                score=1.0 if found else 0.0,
                test_pass=found,
                reason=f"actual_output {'starts with' if found else 'does not start with'} '{self.value}'",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)


