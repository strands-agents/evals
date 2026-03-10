from typing import cast

from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.case_prompt_template import compose_test_prompt
from .prompt_templates.prompt_templates import judge_environment_state_template as SYSTEM_PROMPT


class EnvironmentStateEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates environment state produced by a task using an LLM judge.

    Attributes:
        rubric: The user-specified criteria for evaluating environment state.
        model: A string representing the model-id for Bedrock to use, or a Model instance.
                    Defaults to strands.models.BedrockModel if None.
        system_prompt: System prompt to guide model behavior.
                    If None, the evaluator will use the default environment state template.
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
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        evaluation_prompt = compose_test_prompt(
            evaluation_case=evaluation_case,
            rubric=self.rubric,
            include_inputs=self.include_inputs,
            uses_environment_state=True,
        )
        result = evaluator_agent(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        evaluation_prompt = compose_test_prompt(
            evaluation_case=evaluation_case,
            rubric=self.rubric,
            include_inputs=self.include_inputs,
            uses_environment_state=True,
        )
        result = await evaluator_agent.invoke_async(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]
