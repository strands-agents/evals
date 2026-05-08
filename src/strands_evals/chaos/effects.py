"""Chaos effect definitions.

Effects are first-class parameterized classes organized in a hierarchy:
    ChaosEffect → ToolEffect → concrete effects (Timeout, NetworkError, etc.)
                → ModelEffect → (reserved for v2)

Each concrete effect carries only the parameters meaningful to it.
The `hook` class variable indicates whether the effect fires pre-tool-call
(error effects) or post-tool-call (corruption effects).

Pre-hook effects provide `error_message` (the plugin cancels the tool call).
Post-hook effects implement `apply(response)` (the plugin passes the response through).
"""

import math
import random
from abc import abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class ChaosEffect(BaseModel):
    """Base for all chaos effects.

    Attributes:
        apply_rate: Probability that this effect fires.
            In v1 this field is accepted but ignored (always fires).
        hook: Whether this effect fires pre-call ("pre") or post-call ("post").
    """

    hook: ClassVar[Literal["pre", "post"]]

    apply_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability that this effect fires (1.0 = always).",
    )

    @abstractmethod
    def apply(self, context: Any = None) -> Any:
        """Apply the chaos effect.

        Pre-hook effects return an error message string.
        Post-hook effects accept a response dict and return the corrupted dict.
        """
        ...


class ToolEffect(ChaosEffect):
    """Effect valid at the tool invocation boundary.

    - "pre": effect fires before tool execution (cancels the call with an error)
    - "post": effect fires after tool execution (corrupts the response)
    """


# ---------------------------------------------------------------------------
# Pre-hook effect — cancels the tool call before execution
# ---------------------------------------------------------------------------

# All supported failure types
ToolCallFailureType = Literal["timeout", "network_error", "execution_error", "validation_error"]

# Default error messages per failure type
_DEFAULT_ERROR_MESSAGES: dict[str, str] = {
    "timeout": "Tool call timed out",
    "network_error": "Network unreachable",
    "execution_error": "Tool execution failed",
    "validation_error": "Tool input validation failed",
}


class ToolCallFailure(ToolEffect):
    """Simulates a tool call failure that prevents the tool from executing.

    The tool call is cancelled before execution with a simulated error message.

    Example::

        ChaosScenario(
            name="search_timeout",
            effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
        )

        ChaosScenario(
            name="db_network_error",
            effects={"database_tool": [ToolCallFailure(
                error_type="network_error",
                error_message="Connection refused on port 5432",
            )]},
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "pre"
    error_type: ToolCallFailureType = Field(
        default="execution_error",
        description="Type of failure to simulate.",
    )
    error_message: str | None = Field(
        default=None,
        description="Custom error message. If None, uses a default for the error_type.",
    )

    def apply(self, context: Any = None) -> str:
        """Return the error message to cancel the tool call with."""
        if self.error_message is not None:
            return self.error_message
        return _DEFAULT_ERROR_MESSAGES[self.error_type]


# ---------------------------------------------------------------------------
# Concrete tool corruption effects (post-hook — mutate the response)
#
# Post-hook effects implement apply(response) -> response.
# The plugin calls effect.apply(response_dict) and uses the return value.
# ---------------------------------------------------------------------------


class TruncateFields(ToolEffect):
    """Truncates string values in the tool response.

    The tool executes normally, but string fields in the response are
    truncated to at most `max_length` characters.

    Example::

        ChaosScenario(
            name="search_truncated",
            effects={
                "search_tool": [TruncateFields(max_length=5)],
            },
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    max_length: int = Field(default=10, ge=0, description="Maximum length to truncate string values to")

    def apply(self, response: dict[str, Any]) -> dict[str, Any]:
        """Truncate string values to max_length.

        Args:
            response: The tool response dict to corrupt.

        Returns:
            Response with string values truncated.
        """
        result: dict[str, Any] = {}
        for key, value in response.items():
            if isinstance(value, str) and len(value) > self.max_length:
                result[key] = value[: self.max_length]
            elif isinstance(value, dict):
                result[key] = self._truncate(value)
            else:
                result[key] = value
        return result


class RemoveFields(ToolEffect):
    """Removes a fraction of fields from the tool response.

    The tool executes normally, but a portion of the response fields
    are deleted.

    Example::

        ChaosScenario(
            name="db_remove_fields",
            effects={
                "database_tool": [RemoveFields(remove_ratio=0.5)],
            },
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    remove_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of fields to remove from the response",
    )

    def apply(self, response: dict[str, Any]) -> dict[str, Any]:
        """Remove a fraction of fields from the response.

        Always removes at least 1 field when called.

        Args:
            response: The tool response dict to corrupt.

        Returns:
            Response with fields removed.
        """
        keys = list(response.keys())
        if not keys:
            return response

        num_to_remove = max(1, math.ceil(len(keys) * self.remove_ratio))
        keys_to_remove = set(random.sample(keys, min(num_to_remove, len(keys))))
        return {k: v for k, v in response.items() if k not in keys_to_remove}


class CorruptValues(ToolEffect):
    """Replaces a fraction of values with garbage data.

    The tool executes normally, but a portion of the response values
    are replaced with wrong types or nonsense data.

    Example::

        ChaosScenario(
            name="db_corrupt",
            effects={
                "database_tool": [CorruptValues(corrupt_ratio=0.8)],
            },
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    corrupt_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of values to corrupt in the response",
    )

    _CORRUPTIONS: ClassVar[list[Any]] = [None, 99999, "", True, [], "CORRUPTED_DATA"]

    def apply(self, response: dict[str, Any]) -> dict[str, Any]:
        """Replace a fraction of values with wrong types or garbage data.

        Always corrupts at least 1 field when called.

        Args:
            response: The tool response dict to corrupt.

        Returns:
            Response with corrupted values.
        """
        keys = list(response.keys())
        if not keys:
            return response

        num_to_corrupt = max(1, math.ceil(len(keys) * self.corrupt_ratio))
        keys_to_corrupt = set(random.sample(keys, min(num_to_corrupt, len(keys))))

        result: dict[str, Any] = {}
        for key, value in response.items():
            if key in keys_to_corrupt:
                candidates = [c for c in self._CORRUPTIONS if c != value]
                result[key] = random.choice(candidates) if candidates else "CORRUPTED_DATA"
            elif isinstance(value, dict):
                result[key] = self.apply(value)
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Convenience sets for classification (derived from hierarchy, not maintained manually)
# ---------------------------------------------------------------------------

# All concrete pre-hook (error) effect classes
TOOL_ERROR_EFFECTS: set[type[ToolEffect]] = {ToolCallFailure}

# All concrete post-hook (corruption) effect classes
TOOL_CORRUPTION_EFFECTS: set[type[ToolEffect]] = {TruncateFields, RemoveFields, CorruptValues}
