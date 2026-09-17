from typing import cast

from strands import Agent
from strands.models.model import Model
from typing_extensions import Any

from ..tools.evaluation_tools import any_order_match_scorer, exact_match_scorer, in_order_match_scorer
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ._trace_index import TraceIndex
from .evaluator import DisclosureMode, Evaluator
from .prompt_templates.case_prompt_template import compose_test_prompt
from .prompt_templates.prompt_templates import judge_trajectory_template_tools as SYSTEM_PROMPT


class TrajectoryEvaluator(Evaluator[InputT, OutputT]):
    """
    An evaluator that is trajectory-based.

    Attributes:
        rubric: The user-specified criteria for evaluating a collection of test cases.
        trajectory_description: A description of the available trajectory types. eg. tool descriptions
        model: A string representing the model-id for Bedrock to use, or a Model instance.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use one of the default template.
        include_inputs: Whether to include inputs to the task in the evaluation or not.
        tools: Optional additional tools for the evaluator agent. Merged with the
                    default trajectory scoring tools (exact/in-order/any-order match).
    """

    def __init__(
        self,
        rubric: str,
        trajectory_description: dict | None = None,
        model: Model | str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        include_inputs: bool = True,
        name: str | None = None,
        tools: list[Any] | None = None,
        disclosure: DisclosureMode = "auto",
    ):
        super().__init__(name=name)
        self.rubric = rubric
        self.trajectory_description = trajectory_description
        self.model = model
        self.include_inputs = include_inputs
        self._tools: list[str | dict[str, str] | Any] = [
            exact_match_scorer,
            in_order_match_scorer,
            any_order_match_scorer,
            *(tools or []),
        ]
        self.system_prompt = system_prompt
        self.disclosure = self._validate_disclosure(disclosure)

    def update_trajectory_description(self, new_description: dict) -> None:
        """
        Update the description of the available trajectories.

        Args:
            new_description: The new description of the available trajectories.
        """
        self.trajectory_description = new_description

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        evaluation_prompt, disclosure_tools = self._render_with_disclosure(
            evaluation_case, lambda idx: self._compose(evaluation_case, idx)
        )
        evaluator_agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=[*self._tools, *disclosure_tools],
            callback_handler=None,
        )
        result = evaluator_agent(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases asynchronously.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        evaluation_prompt, disclosure_tools = self._render_with_disclosure(
            evaluation_case, lambda idx: self._compose(evaluation_case, idx)
        )
        evaluator_agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=[*self._tools, *disclosure_tools],
            callback_handler=None,
        )
        result = await evaluator_agent.invoke_async(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]

    def _compose(self, evaluation_case: EvaluationData[InputT, OutputT], trace_index: TraceIndex | None) -> str:
        """Compose the trajectory judge prompt, substituting the trace overview on overflow."""
        override = self._disclosed_trace_section(trace_index) if trace_index is not None else None
        return compose_test_prompt(
            evaluation_case=evaluation_case,
            rubric=self.rubric,
            include_inputs=self.include_inputs,
            uses_trajectory=True,
            trajectory_override=override,
        )
