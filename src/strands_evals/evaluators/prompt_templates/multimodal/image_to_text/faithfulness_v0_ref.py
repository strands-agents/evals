"""Faithfulness evaluation rubric with reference comparison for multimodal image-to-text tasks (binary).

Reference comparison section is injected when expected_output is provided.
"""

from .faithfulness_v0 import FAITHFULNESS_RUBRIC_V0

FAITHFULNESS_RUBRIC_V0_REF = (
    FAITHFULNESS_RUBRIC_V0
    + """

REFERENCE COMPARISON:
- Compare the response against the Oracle reference answer above.
- The reference is the gold standard. Use discrepancies as evidence for your judgment."""
)
