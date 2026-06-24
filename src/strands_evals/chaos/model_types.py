"""Model output corruption types and configuration.

Defines the enums and Pydantic config model used to parameterize model
output corruption effects.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelOutputCorruptionType(str, Enum):
    """Output corruption types for model response format mutation."""

    MALFORMED_JSON = "malformed_json"
    TRUNCATED_RESPONSE = "truncated_response"
    SCHEMA_VIOLATION = "schema_violation"
    EMPTY_RESPONSE = "empty_response"
    GARBAGE_OUTPUT = "garbage_output"
    FULL_REFUSAL = "full_refusal"
    TOXIC_CONTENT = "toxic_content"


class ModelOutputHallucinationType(str, Enum):
    """Hallucination types for model response semantic mutation.

    Separated from ModelOutputCorruptionType to allow future LLM-based
    hallucination generation strategies.
    """

    FACTUAL_ERROR = "factual_error"
    CONFABULATION = "confabulation"
    CONTEXT_UNFAITHFULNESS = "context_unfaithfulness"
    TOOL_CLAIM_FABRICATION = "tool_claim_fabrication"


class ModelOutputCorruptionConfig(BaseModel):
    """Configuration for model output corruption.

    Mutates model responses after generation to test agent resilience
    against malformed, truncated, or otherwise corrupted LLM output.

    Attributes:
        apply_rate: Probability (0.0 to 1.0) that corruption is applied per call.
            0.0 disables output corruption.
        corruption_type: Type of output corruption to apply.
        truncate_ratio: Fraction of content to corrupt for TRUNCATED_RESPONSE.
        target_entity_types: Entity types for FACTUAL_ERROR filtering.
        add_success_framing: Prepend a confident success prefix after corruption.
    """

    apply_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of corruption per model call")
    corruption_type: ModelOutputCorruptionType | ModelOutputHallucinationType = Field(
        default=ModelOutputCorruptionType.MALFORMED_JSON,
        description="Type of output corruption or hallucination to apply",
    )
    truncate_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of content to corrupt for TRUNCATED_RESPONSE",
    )
    target_entity_types: Optional[list[str]] = Field(
        default=None,
        description="Entity types for FACTUAL_ERROR: 'date', 'number', 'location', 'price'. None = all.",
    )
    add_success_framing: bool = Field(
        default=False,
        description="Prepend a confident success prefix before the corrupted content. "
        "Composable with any corruption_type or hallucination_type.",
    )
