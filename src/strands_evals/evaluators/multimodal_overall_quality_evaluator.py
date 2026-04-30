"""Overall Quality evaluator for multimodal tasks."""

from strands.models.model import Model

from .multimodal_output_evaluator import MultimodalOutputEvaluator
from .prompt_templates.multimodal import OVERALL_QUALITY_RUBRIC_V0

_OVERALL_QUALITY_REFERENCE_SUFFIX = """

REFERENCE COMPARISON:
- Compare the response against the Oracle reference answer above.
- The reference should be treated as the gold standard.
- Reward responses that cover the same key points as the reference.
- Penalize responses that miss important information present in the reference.
- Penalize responses that contain claims contradicting the reference.
- The response does NOT need to match the reference word-for-word — only the factual content matters."""


class MultimodalOverallQualityEvaluator(MultimodalOutputEvaluator):
    """Evaluates overall quality of multimodal responses (P0).

    Assesses the response across four dimensions: visual accuracy, instruction
    adherence, completeness, and coherence/helpfulness. Ships with an image-to-text
    rubric by default; pass a custom rubric for other media types.

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
            rubric=rubric if rubric is not None else OVERALL_QUALITY_RUBRIC_V0,
            model=model,
            include_inputs=include_inputs,
            system_prompt=system_prompt,
            reference_suffix=reference_suffix if reference_suffix is not None else _OVERALL_QUALITY_REFERENCE_SUFFIX,
            uses_environment_state=uses_environment_state,
        )
