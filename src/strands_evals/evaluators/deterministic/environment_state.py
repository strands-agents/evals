from typing_extensions import Any

from ...types.evaluation import EnvironmentState, EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator


def _find_state_by_name(states: list[EnvironmentState], name: str) -> EnvironmentState | None:
    """Find an EnvironmentState by name in a list of states."""
    for state in states:
        if state.name == name:
            return state
    return None


class StateEquals(Evaluator[InputT, OutputT]):
    """Checks if a named environment state matches an expected value."""

    def __init__(self, name: str, value: Any | None = None):
        super().__init__()
        self.name = name
        self.value = value

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if not evaluation_case.actual_environment_state:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"state '{self.name}' not found: actual_environment_state is empty or None",
                )
            ]

        actual_state = _find_state_by_name(evaluation_case.actual_environment_state, self.name)
        if actual_state is None:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"state '{self.name}' not found in actual_environment_state",
                )
            ]

        if self.value is not None:
            expected = self.value
        elif evaluation_case.expected_environment_state:
            expected_state = _find_state_by_name(evaluation_case.expected_environment_state, self.name)
            if expected_state is None:
                raise ValueError(
                    f"state '{self.name}' not found in expected_environment_state and no explicit value provided"
                )
            expected = expected_state.state
        else:
            raise ValueError(
                f"no expected value for state '{self.name}': provide value param or expected_environment_state"
            )

        match = actual_state.state == expected
        return [
            EvaluationOutput(
                score=1.0 if match else 0.0,
                test_pass=match,
                reason=f"state '{self.name}' {'matches' if match else 'does not match'} expected value",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)
