"""Correctness evaluation rubric with reference comparison for multimodal image-to-text tasks (binary).

Reference comparison section is injected when expected_output is provided.
"""

from .correctness_v0 import CORRECTNESS_RUBRIC_V0

CORRECTNESS_RUBRIC_V0_REF = CORRECTNESS_RUBRIC_V0 + """

REFERENCE COMPARISON:
- Compare the response against the Oracle reference answer above.
- The reference is the gold standard. Use discrepancies as evidence for your judgment."""
