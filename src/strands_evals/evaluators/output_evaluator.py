from typing import cast

from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.case_prompt_template import compose_test_prompt
from .prompt_templates.prompt_templates import judge_output_template as SYSTEM_PROMPT


class OutputEvaluator(Evaluator[InputT, OutputT]):
    """
    An evaluator that is LLM-based.

    Attributes:
        rubric: The user-specified criteria for evaluating a collection of test cases.
        model: A string representing the model-id for Bedrock to use, or a Model instance.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use one of the default template.
        include_inputs: Whether to include inputs to the task in the evaluation or not.
    """

    def __init__(
        self,
        rubric: str,
        model: Union[Model, str, None] = None,
        system_prompt: str = SYSTEM_PROMPT,
        include_inputs: bool = True,
    ):
        super().__init__()
        self.rubric = rubric
        self.model = model
        self.include_inputs = include_inputs
        self.system_prompt = system_prompt

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """
        Evaluate the performance of the task on the given test cases.

        Args:
            evaluation_case: The test case with all of the neccessary context to be evaluated.

        Returns:
            The results of the evaluation as EvaluationOutput.
        """
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        evaluation_prompt = compose_test_prompt(
            evaluation_case=evaluation_case, rubric=self.rubric, include_inputs=self.include_inputs
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
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        evaluation_prompt = compose_test_prompt(
            evaluation_case=evaluation_case, rubric=self.rubric, include_inputs=self.include_inputs
        )
        result = await evaluator_agent.invoke_async(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]
