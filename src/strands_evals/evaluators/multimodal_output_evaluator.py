"""Multimodal output evaluator using MLLM-as-a-Judge."""

import logging
import warnings
from typing import List, Union, cast

from strands import Agent
from strands.models.model import Model

from ..types.evaluation import EvaluationData, EvaluationOutput
from ..types.multimodal import MultimodalInput
from .output_evaluator import OutputEvaluator
from .prompt_templates.multimodal_case_prompt_template import compose_multimodal_test_prompt
from .prompt_templates.multimodal_judge_system_prompt import MLLM_JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class MultimodalOutputEvaluator(OutputEvaluator[MultimodalInput, str]):
    """MLLM-as-a-Judge evaluator for multimodal tasks.

    Extends OutputEvaluator to handle multimodal inputs containing media (images,
    documents, etc.) and text. Supports both MLLM-as-a-Judge (with media) and
    LLM-as-a-Judge (text-only) modes, as well as reference-based and reference-free
    evaluation.

    Attributes:
        rubric: Evaluation criteria (e.g., correctness, faithfulness rubric)
        model: Model instance or model ID string for the MLLM judge
        include_media: Whether to include the media in the judge prompt (MLLM vs LLM mode)
        include_inputs: Whether to include the original instruction in evaluation
        system_prompt: System prompt for the MLLM judge
        ref_rubric: Reference-based rubric variant, auto-selected when expected_output is provided
    """

    def __init__(
        self,
        rubric: str,
        model: Union[Model, str, None] = None,
        include_media: bool = True,
        include_inputs: bool = True,
        system_prompt: Union[str, None] = None,
        ref_rubric: Union[str, None] = None,
    ):
        super().__init__(
            rubric=rubric,
            model=model,
            system_prompt=system_prompt if system_prompt is not None else MLLM_JUDGE_SYSTEM_PROMPT,
            include_inputs=include_inputs,
        )
        self.include_media = include_media
        self.ref_rubric = ref_rubric

    def _select_rubric(self, evaluation_case: EvaluationData[MultimodalInput, str]) -> str:
        """Select the appropriate rubric based on whether a reference output is available."""
        if self.ref_rubric and evaluation_case.expected_output is not None:
            return self.ref_rubric
        return self.rubric

    def evaluate(self, evaluation_case: EvaluationData[MultimodalInput, str]) -> List[EvaluationOutput]:
        """Evaluate a multimodal test case.

        Automatically selects the reference-based rubric when ``expected_output`` is
        provided and a ``ref_rubric`` was configured, otherwise uses the default rubric.

        Args:
            evaluation_case: Test case with multimodal input and expected/actual outputs.

        Returns:
            List containing a single EvaluationOutput with score, pass/fail, and reasoning.
        """
        if self.include_media and isinstance(evaluation_case.input, dict) and "media" not in evaluation_case.input:
            warnings.warn(
                "include_media=True but no 'media' key found in input. Falling back to text-only evaluation.",
                UserWarning,
                stacklevel=2,
            )

        effective_rubric = self._select_rubric(evaluation_case)

        evaluation_prompt = compose_multimodal_test_prompt(
            evaluation_case=evaluation_case,
            rubric=effective_rubric,
            include_inputs=self.include_inputs,
            include_media=self.include_media,
        )

        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]

    async def evaluate_async(self, evaluation_case: EvaluationData[MultimodalInput, str]) -> List[EvaluationOutput]:
        """Evaluate a multimodal test case asynchronously.

        Automatically selects the reference-based rubric when ``expected_output`` is
        provided and a ``ref_rubric`` was configured, otherwise uses the default rubric.

        Args:
            evaluation_case: Test case with multimodal input and expected/actual outputs.

        Returns:
            List containing a single EvaluationOutput with score, pass/fail, and reasoning.
        """
        if self.include_media and isinstance(evaluation_case.input, dict) and "media" not in evaluation_case.input:
            warnings.warn(
                "include_media=True but no 'media' key found in input. Falling back to text-only evaluation.",
                UserWarning,
                stacklevel=2,
            )

        effective_rubric = self._select_rubric(evaluation_case)

        evaluation_prompt = compose_multimodal_test_prompt(
            evaluation_case=evaluation_case,
            rubric=effective_rubric,
            include_inputs=self.include_inputs,
            include_media=self.include_media,
        )

        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(evaluation_prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]
