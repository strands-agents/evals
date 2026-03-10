from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ...types.trace import Session, ToolExecutionSpan
from ..evaluator import Evaluator


class ToolCalled(Evaluator[InputT, OutputT]):
    """Checks if a specific tool was called in the trajectory."""

    def __init__(self, tool_name: str):
        super().__init__()
        self.tool_name = tool_name

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        trajectory = evaluation_case.actual_trajectory
        if trajectory is None:
            return [EvaluationOutput(score=0.0, test_pass=False, reason="no trajectory provided")]

        if isinstance(trajectory, Session):
            found = self._check_session(trajectory)
        elif isinstance(trajectory, list):
            found = self.tool_name in trajectory
        else:
            return [
                EvaluationOutput(
                    score=0.0,
                    test_pass=False,
                    reason=f"unsupported trajectory type: {type(trajectory).__name__}",
                )
            ]

        return [
            EvaluationOutput(
                score=1.0 if found else 0.0,
                test_pass=found,
                reason=f"tool '{self.tool_name}' {'was called' if found else 'was not called'}",
            )
        ]

    def _check_session(self, session: Session) -> bool:
        for trace in session.traces:
            for span in trace.spans:
                if isinstance(span, ToolExecutionSpan) and span.tool_call.name == self.tool_name:
                    return True
        return False

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)
