"""Faithfulness evaluator for multimodal tasks."""

from strands.models.model import Model

from .multimodal_output_evaluator import MultimodalOutputEvaluator
from .prompt_templates.multimodal import FAITHFULNESS_RUBRIC_V0


class MultimodalFaithfulnessEvaluator(MultimodalOutputEvaluator):
    """Evaluates faithfulness of multimodal responses (P1).

    Assesses whether the response is grounded in the media content without hallucinations.
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
            rubric=rubric if rubric is not None else FAITHFULNESS_RUBRIC_V0,
            model=model,
            include_inputs=include_inputs,
            system_prompt=system_prompt,
            reference_suffix=reference_suffix,
            uses_environment_state=uses_environment_state,
        )
