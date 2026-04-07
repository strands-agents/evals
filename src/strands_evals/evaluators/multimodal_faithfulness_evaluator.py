"""Faithfulness evaluator for multimodal tasks."""

from strands.models.model import Model
from typing_extensions import Union

from .multimodal_output_evaluator import MultimodalOutputEvaluator
from .prompt_templates.multimodal import FAITHFULNESS_RUBRIC_V0, FAITHFULNESS_RUBRIC_V0_REF


class MultimodalFaithfulnessEvaluator(MultimodalOutputEvaluator):
    """Evaluates faithfulness of multimodal responses (P1).

    Assesses whether the response is grounded in the media content without hallucinations.
    Ships with an image-to-text rubric by default; pass a custom rubric for other media types.

    Automatically uses the reference-based rubric when ``expected_output`` is provided.
    """

    def __init__(
        self,
        model: Union[Model, str, None] = None,
        rubric: Union[str, None] = None,
        include_media: bool = True,
        include_inputs: bool = True,
        system_prompt: Union[str, None] = None,
    ):
        super().__init__(
            rubric=rubric if rubric is not None else FAITHFULNESS_RUBRIC_V0,
            ref_rubric=None if rubric is not None else FAITHFULNESS_RUBRIC_V0_REF,
            model=model,
            include_media=include_media,
            include_inputs=include_inputs,
            system_prompt=system_prompt,
        )
