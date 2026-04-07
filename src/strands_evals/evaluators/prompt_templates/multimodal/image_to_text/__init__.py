"""Image-to-text evaluation prompt templates."""

from .correctness_v0 import CORRECTNESS_RUBRIC_V0
from .correctness_v0_ref import CORRECTNESS_RUBRIC_V0_REF
from .faithfulness_v0 import FAITHFULNESS_RUBRIC_V0
from .faithfulness_v0_ref import FAITHFULNESS_RUBRIC_V0_REF
from .instruction_following_v0 import INSTRUCTION_FOLLOWING_RUBRIC_V0
from .instruction_following_v0_ref import INSTRUCTION_FOLLOWING_RUBRIC_V0_REF
from .overall_quality_v0 import OVERALL_QUALITY_RUBRIC_V0
from .overall_quality_v0_ref import OVERALL_QUALITY_RUBRIC_V0_REF

__all__ = [
    "CORRECTNESS_RUBRIC_V0",
    "CORRECTNESS_RUBRIC_V0_REF",
    "FAITHFULNESS_RUBRIC_V0",
    "FAITHFULNESS_RUBRIC_V0_REF",
    "INSTRUCTION_FOLLOWING_RUBRIC_V0",
    "INSTRUCTION_FOLLOWING_RUBRIC_V0_REF",
    "OVERALL_QUALITY_RUBRIC_V0",
    "OVERALL_QUALITY_RUBRIC_V0_REF",
]
