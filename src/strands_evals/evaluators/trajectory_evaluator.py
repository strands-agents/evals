import json
import logging
from typing import cast

from strands import Agent
from strands.models.model import Model
from typing_extensions import Any

from ..tools.evaluation_tools import any_order_match_scorer, exact_match_scorer, in_order_match_scorer
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.case_prompt_template import compose_test_prompt
from .prompt_templates.prompt_templates import judge_trajectory_template_tools as SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
    ):
        super().__init__(name=name)
        self.rubric = rubric
        self.trajectory_description = trajectory_description
        self.model = model
        self.include_inputs = include_inputs
        self._user_tools = tools
        self._tools: list[str | dict[str, str] | Any] = [
            exact_match_scorer,
            in_order_match_scorer,
            any_order_match_scorer,
            *(tools or []),
        ]
        self.system_prompt = system_prompt

    @property
    def tools(self) -> list[Any] | None:
        """User-supplied additional tools for the evaluator agent.

        Excludes the built-in exact/in-order/any-order match scorers in `_tools`: those
        are always active and are not part of what the caller passed in, so `to_dict()`
        below only serializes tools from this property. Only tools that can be written as
        valid JSON in a utf-8 file survive `to_dict()`; the rest are skipped with a warning
        and must be re-attached after `from_dict()`.
        """
        return self._user_tools

    @tools.setter
    def tools(self, value: list[Any] | None) -> None:
        self._user_tools = value
        self._tools = [exact_match_scorer, in_order_match_scorer, any_order_match_scorer, *(value or [])]

    def to_dict(self) -> dict:
        """
        Convert the evaluator into a dictionary.

        Returns:
            dict: A dictionary containing the evaluator's information. Includes only
            user-supplied tools that can be written as valid JSON in a utf-8 file. Tools
            that cannot (decorated functions, NaN or Infinity floats, strings with
            unpaired surrogates) are skipped with a warning and must be re-attached after
            `from_dict()`.
        """
        _dict = super().to_dict()
        if self._user_tools:
            serializable_tools = []
            for tool in self._user_tools:
                try:
                    json.dumps(tool, ensure_ascii=False, allow_nan=False).encode("utf-8")
                except (TypeError, ValueError):
                    tool_name = getattr(tool, "tool_name", None) or getattr(tool, "__name__", None)
                    if not isinstance(tool_name, str):
                        tool_name = ascii(tool)
                        if len(tool_name) > 80:
                            tool_name = tool_name[:77] + "..."
                    logger.warning(
                        "tool_name=<%s> | skipping tool that cannot be written as valid utf-8 JSON, "
                        "re-attach it via the `tools` attribute after loading",
                        tool_name,
                    )
                else:
                    serializable_tools.append(tool)
            if serializable_tools:
                _dict["tools"] = serializable_tools
        return _dict

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
        evaluator_agent = Agent(
            model=self.model, system_prompt=self.system_prompt, tools=self._tools, callback_handler=None
        )
        evaluation_prompt = compose_test_prompt(
            evaluation_case=evaluation_case,
            rubric=self.rubric,
            include_inputs=self.include_inputs,
            uses_trajectory=True,
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
        evaluator_agent = Agent(
            model=self.model, system_prompt=self.system_prompt, tools=self._tools, callback_handler=None
        )
        evaluation_prompt = compose_test_prompt(
            evaluation_case=evaluation_case,
            rubric=self.rubric,
            include_inputs=self.include_inputs,
            uses_trajectory=True,
        )
        result = await evaluator_agent.invoke_async(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]
