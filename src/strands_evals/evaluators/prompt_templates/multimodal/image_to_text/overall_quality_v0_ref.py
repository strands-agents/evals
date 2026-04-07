"""Overall Quality evaluation rubric with reference comparison for multimodal image-to-text tasks.

Reference comparison section is injected when expected_output is provided.
"""

from .overall_quality_v0 import OVERALL_QUALITY_RUBRIC_V0

OVERALL_QUALITY_RUBRIC_V0_REF = OVERALL_QUALITY_RUBRIC_V0 + """

REFERENCE COMPARISON:
- Compare the response against the Oracle reference answer above.
- The reference should be treated as the gold standard.
- Reward responses that cover the same key points as the reference.
- Penalize responses that miss important information present in the reference.
- Penalize responses that contain claims contradicting the reference.
- The response does NOT need to match the reference word-for-word — only the factual content matters."""
