"""Instruction Following evaluator for multimodal tasks."""

from strands.models.model import Model

from .multimodal_output_evaluator import MultimodalOutputEvaluator
from .prompt_templates.multimodal import INSTRUCTION_FOLLOWING_RUBRIC_V0


class MultimodalInstructionFollowingEvaluator(MultimodalOutputEvaluator):
    """Evaluates instruction following in multimodal responses (P1).

    Assesses whether the response addresses the query's requirements and constraints.
    Ships with an image-to-text rubric by default; pass a custom rubric for other media types.

    Automatically appends a reference comparison suffix when ``expected_output`` is provided.
    """

    def __init__(
        self,
        model: Model | str | None = None,
        rubric: str | None = None,
        include_inputs: bool = True,
        system_prompt: str | None = None,
        reference_suffix: str | None = None,
        uses_environment_state: bool = False,
    ):
        super().__init__(
            rubric=rubric if rubric is not None else INSTRUCTION_FOLLOWING_RUBRIC_V0,
            model=model,
            include_inputs=include_inputs,
            system_prompt=system_prompt,
            reference_suffix=reference_suffix,
            uses_environment_state=uses_environment_state,
        )
