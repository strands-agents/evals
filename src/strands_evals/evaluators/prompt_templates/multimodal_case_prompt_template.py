"""Multimodal-specific prompt template functions."""

import logging
import warnings
from typing import Any

from pydantic import ValidationError

from ...types.evaluation import EvaluationData, InputT, OutputT
from ...types.multimodal import ImageData, MultimodalInput

logger = logging.getLogger(__name__)


def _coerce_multimodal_input(value: Any) -> Any:
    """Coerce a raw dict back to MultimodalInput when possible.

    When an experiment is saved via ``Experiment.to_dict`` and reloaded via
    ``Experiment.from_dict``, ``Case.model_validate`` runs without the generic
    ``[MultimodalInput, str]`` parameterization, so ``case.input`` comes back as
    a raw ``dict``. This coercion happens at the single prompt-composer boundary
    so the downstream ``isinstance(input_, MultimodalInput)`` dispatch is correct.
    """
    if not isinstance(value, dict):
        return value
    # Only coerce dicts that look like a MultimodalInput payload.
    if "instruction" not in value:
        return value
    try:
        return MultimodalInput.model_validate(value)
    except ValidationError:
        return value


def _resolve_media_data(media_source: Any) -> ImageData:
    """Resolve a media source to a concrete media data instance.

    Currently supports ImageData objects and strings (file paths, base64, data URLs).
    To add a new modality (e.g., VideoData), add an isinstance check here and a
    corresponding content block builder in ``_build_media_content_block``.
    """
    if isinstance(media_source, ImageData):
        return media_source
    if isinstance(media_source, str):
        return ImageData(source=media_source)
    raise ValueError(
        f"Expected ImageData or str for media source, got {type(media_source)}. "
        f"Supported: ImageData, file path, base64, data URL."
    )


def _build_media_content_block(media_data: ImageData) -> dict[str, Any]:
    """Build a strands SDK content block from a media data instance.

    Returns the appropriate content block dict for the media type. Currently
    only ImageData is supported; to add new modalities, add an isinstance
    branch that returns the correct content block key (e.g., "video", "document").
    """
    if isinstance(media_data, ImageData):
        media_format = media_data.format or "png"
        media_bytes = media_data.to_bytes()
        return {"image": {"format": media_format, "source": {"bytes": media_bytes}}}

    raise ValueError(f"Unsupported media type for content block: {type(media_data)}")


def compose_multimodal_test_prompt(
    evaluation_case: EvaluationData[InputT, OutputT],
    rubric: str,
    include_inputs: bool = True,
) -> str | list[dict[str, Any]]:
    """Compose the evaluation prompt for a multimodal test case.

    Dispatch is data-driven: when the input is a ``MultimodalInput`` carrying
    media, a list of strands ContentBlock dicts (media + text) is returned;
    otherwise a plain text prompt is returned for text-only (LLM) evaluation.

    The media block uses the strands SDK ContentBlock format, e.g. for images::

        {"image": {"format": "jpeg", "source": {"bytes": b"..."}}}

    Args:
        evaluation_case: Evaluation data containing multimodal input and outputs.
        rubric: Evaluation criteria to apply.
        include_inputs: Whether to include the instruction in the prompt.

    Returns:
        Either a text prompt string or a list of strands ContentBlock dicts.

    Raises:
        ValueError: If ``actual_output`` is missing or media data is invalid.
    """
    # Coerce raw dicts back to MultimodalInput (save/load round-trip safety).
    input_ = _coerce_multimodal_input(evaluation_case.input)

    # Build text portion of the prompt
    text_parts = [
        "Evaluate this singular test case. THE FINAL SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 "
        "(NOT 0 to 10 OR 0 to 100). \n"
    ]

    if include_inputs and isinstance(input_, MultimodalInput):
        instruction = input_.instruction
        context = input_.context
        input_text = f"Context: {context}\nInstruction: {instruction}" if context else instruction
        text_parts.append(f"<Input>{input_text}</Input>\n")
    elif include_inputs:
        text_parts.append(f"<Input>{input_}</Input>\n")

    if evaluation_case.actual_output is None:
        raise ValueError("Please make sure the task function return the output or a dictionary with the key 'output'.")
    text_parts.append(f"<Output>{evaluation_case.actual_output}</Output>\n")

    if evaluation_case.expected_output is not None:
        text_parts.append(f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>\n")

    text_parts.append(f"<Rubric>{rubric}</Rubric>")

    evaluation_text = "".join(text_parts)

    # Data-driven dispatch: include media only when the input carries it.
    if isinstance(input_, MultimodalInput):
        media = input_.media

        if not media:
            warnings.warn(
                "MultimodalInput has empty media; evaluating as text-only.",
                UserWarning,
                stacklevel=2,
            )
            return evaluation_text

        # Normalize to list
        if isinstance(media, list):
            media_list = media
        elif isinstance(media, ImageData):
            media_list = [media]
        else:
            media_list = [_resolve_media_data(media)]

        # Build content blocks: all media first, then text
        messages: list[dict[str, Any]] = []
        for media_item in media_list:
            media_data = _resolve_media_data(media_item)
            content_block = _build_media_content_block(media_data)
            messages.append(content_block)

        messages.append({"text": evaluation_text})
        return messages

    return evaluation_text
