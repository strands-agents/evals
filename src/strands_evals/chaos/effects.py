"""Chaos effect definitions.

Provides a flat enum of all chaos effects and an optional config class
for advanced parameterization.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ToolChaosEffect(str, Enum):
    """All chaos effects a tool can experience.

    Error effects cause the tool call to be cancelled before execution.
    Corruption effects mutate the tool response after execution.
    """

    # Error effects (tool call is cancelled with a simulated error)
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"

    # Response corruption effects (tool executes, response is mangled)
    TRUNCATE_FIELDS = "truncate_fields"
    REMOVE_FIELDS = "remove_fields"
    CORRUPT_VALUES = "corrupt_values"





# Sets for classification
TOOL_ERROR_EFFECTS = {
    ToolChaosEffect.TIMEOUT,
    ToolChaosEffect.NETWORK_ERROR,
    ToolChaosEffect.EXECUTION_ERROR,
    ToolChaosEffect.VALIDATION_ERROR,
}

TOOL_CORRUPTION_EFFECTS = {
    ToolChaosEffect.TRUNCATE_FIELDS,
    ToolChaosEffect.REMOVE_FIELDS,
    ToolChaosEffect.CORRUPT_VALUES,
}


class ChaosEffectConfig(BaseModel):
    """Advanced chaos effect configuration.

    Use when the bare ToolChaosEffect enum needs tuning. Most users only need
    the enum value directly in ChaosScenario.tool_effects.

    Example::

        # Simple (90% of cases): just use the enum
        tool_effects={"search_tool": ToolChaosEffect.TIMEOUT}

        # Advanced: custom error message
        tool_effects={"search_tool": ChaosEffectConfig(
            effect=ToolChaosEffect.TIMEOUT,
            error_message="Request timed out after 30s",
        )}

        # Advanced: tune corruption ratio
        tool_effects={"db_tool": ChaosEffectConfig(
            effect=ToolChaosEffect.REMOVE_FIELDS,
            remove_ratio=0.5,
        )}
    """

    effect: ToolChaosEffect = Field(..., description="The chaos effect to apply")
    apply_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability that this effect fires (1.0 = always, deterministic)",
    )

    # Error-specific parameters
    error_message: Optional[str] = Field(
        default=None,
        description="Custom error message (for TIMEOUT, NETWORK_ERROR, EXECUTION_ERROR, VALIDATION_ERROR)",
    )

    # Corruption-specific parameters
    remove_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of fields to remove (for REMOVE_FIELDS, default 0.33)",
    )
    corrupt_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of values to corrupt (for CORRUPT_VALUES, default 0.4)",
    )

    @model_validator(mode="after")
    def validate_params_match_effect(self) -> "ChaosEffectConfig":
        """Warn if irrelevant parameters are set for the given effect."""
        if self.effect in TOOL_ERROR_EFFECTS:
            if self.remove_ratio is not None or self.corrupt_rate is not None:
                raise ValueError(
                    f"remove_ratio and corrupt_rate are not applicable to error effect "
                    f"'{self.effect.value}'. These params only apply to corruption effects."
                )
        if self.effect in TOOL_CORRUPTION_EFFECTS:
            if self.error_message is not None:
                raise ValueError(
                    f"error_message is not applicable to corruption effect "
                    f"'{self.effect.value}'. This param only applies to error effects."
                )
        return self
