"""Multimodal-specific prompt template functions."""

import logging
import warnings
from typing import Any, Dict, List, Union

from ...types.evaluation import EvaluationData
from ...types.multimodal import ImageData, MultimodalInput

logger = logging.getLogger(__name__)


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


def _build_media_content_block(media_data: ImageData) -> Dict[str, Any]:
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
    evaluation_case: EvaluationData[MultimodalInput, str],
    rubric: str,
    include_inputs: bool = True,
    include_media: bool = True,
) -> Union[str, List[Dict[str, Any]]]:
    """Compose the evaluation prompt for a multimodal test case.

    When ``include_media=True`` and media data is available, returns a list of
    strands ContentBlock dicts (media + text) suitable for the strands Agent.
    Otherwise returns a plain text prompt for text-only (LLM) evaluation.

    The media block uses the strands SDK ContentBlock format, e.g. for images::

        {"image": {"format": "jpeg", "source": {"bytes": b"..."}}}

    Args:
        evaluation_case: Evaluation data containing multimodal input and outputs.
        rubric: Evaluation criteria to apply.
        include_inputs: Whether to include the instruction in the prompt.
        include_media: Whether to include the media (True for MLLM, False for LLM).

    Returns:
        Either a text prompt string or a list of strands ContentBlock dicts.

    Raises:
        Exception: If actual_output is missing.
        ValueError: If media data is invalid when include_media=True.
    """
    # Build text portion of the prompt
    text_parts = [
        "Evaluate this singular test case. THE FINAL SCORE MUST BE A DECIMAL BETWEEN 0.0 AND 1.0 "
        "(NOT 0 to 10 OR 0 to 100). \n"
    ]

    if include_inputs and isinstance(evaluation_case.input, dict):
        instruction = evaluation_case.input.get("instruction", "")
        context = evaluation_case.input.get("context", "")
        input_text = f"Context: {context}\nInstruction: {instruction}" if context else instruction
        text_parts.append(f"<Input>{input_text}</Input>\n")
    elif include_inputs:
        text_parts.append(f"<Input>{evaluation_case.input}</Input>\n")

    if evaluation_case.actual_output is None:
        raise ValueError("Please make sure the task function return the output or a dictionary with the key 'output'.")
    text_parts.append(f"<Output>{evaluation_case.actual_output}</Output>\n")

    if evaluation_case.expected_output is not None:
        text_parts.append(f"<ExpectedOutput>{evaluation_case.expected_output}</ExpectedOutput>\n")

    text_parts.append(f"<Rubric>{rubric}</Rubric>")

    evaluation_text = "".join(text_parts)

    # Build multimodal content blocks when media is requested and available
    if include_media and isinstance(evaluation_case.input, dict) and "media" in evaluation_case.input:
        media = evaluation_case.input["media"]

        # Normalize to list
        if isinstance(media, list):
            media_list = media
        else:
            media_list = [media]

        if not media_list:
            warnings.warn(
                "include_media=True but media list is empty. Falling back to text-only evaluation.",
                UserWarning,
                stacklevel=2,
            )
            return evaluation_text

        # Build content blocks: all media first, then text
        messages: List[Dict[str, Any]] = []
        for media_item in media_list:
            media_data = _resolve_media_data(media_item)
            content_block = _build_media_content_block(media_data)
            messages.append(content_block)

        messages.append({"text": evaluation_text})
        return messages

    return evaluation_text
