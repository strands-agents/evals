"""Chaos effect definitions.

Effects are first-class parameterized classes organized in a hierarchy:
    ChaosEffect → ToolEffect → concrete effects (Timeout, NetworkError, etc.)

Each concrete effect carries only the parameters meaningful to it.
The `hook` class variable indicates whether the effect fires pre-tool-call
(error effects) or post-tool-call (corruption effects).

Each concrete effect has an `effect_type` discriminator field for Pydantic
discriminated-union serialization, ensuring full round-trip fidelity.
"""

import math
import random
from abc import abstractmethod
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag


class ChaosEffect(BaseModel):
    """Base for all chaos effects.

    Attributes:
        apply_rate: Probability that this effect fires, defaults to 1 (always fire).
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
        """Apply the chaos effect to the given context and return the result."""
        ...


class ToolEffect(ChaosEffect):
    """Effect that operates at the tool invocation boundary.

    This intermediate class enables type-based dispatch so the plugin can
    distinguish tool-level effects from other planned effect categories
    (e.g., upcoming ``ModelEffect`` for LLM input and output chaos injection).
    """


# ---------------------------------------------------------------------------
# Pre-hook effects: cancel the tool call before execution
# ---------------------------------------------------------------------------


class Timeout(ToolEffect):
    """Simulates a tool call timeout.

    The tool call is cancelled before execution with a timeout error message.

    Example::

        ChaosCase(
            name="search_timeout",
            input="Find flights",
            effects={"tool_effects": {"search_tool": [Timeout()]}},
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "pre"
    effect_type: Literal["timeout"] = "timeout"
    error_message: str = Field(
        default="Tool call timed out",
        description="Error message returned to the agent.",
    )

    def apply(self, context: Any = None) -> str:
        """Return the error message to cancel the tool call with."""
        return self.error_message


class NetworkError(ToolEffect):
    """Simulates a network error during tool invocation.

    The tool call is cancelled before execution with a network error message.

    Example::

        ChaosCase(
            name="db_network_error",
            input="Query database",
            effects={"tool_effects": {"database_tool": [NetworkError()]}},
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "pre"
    effect_type: Literal["network_error"] = "network_error"
    error_message: str = Field(
        default="Network unreachable",
        description="Error message returned to the agent.",
    )

    def apply(self, context: Any = None) -> str:
        """Return the error message to cancel the tool call with."""
        return self.error_message


class ExecutionError(ToolEffect):
    """Simulates a tool execution failure.

    The tool call is cancelled before execution with an execution error message.

    Example::

        ChaosCase(
            name="tool_exec_error",
            input="Run analysis",
            effects={"tool_effects": {"analysis_tool": [ExecutionError()]}},
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "pre"
    effect_type: Literal["execution_error"] = "execution_error"
    error_message: str = Field(
        default="Tool execution failed",
        description="Error message returned to the agent.",
    )

    def apply(self, context: Any = None) -> str:
        """Return the error message to cancel the tool call with."""
        return self.error_message


class ValidationError(ToolEffect):
    """Simulates a tool input validation failure.

    The tool call is cancelled before execution with a validation error message.

    Example::

        ChaosCase(
            name="bad_input",
            input="Search with invalid params",
            effects={"tool_effects": {"search_tool": [ValidationError()]}},
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "pre"
    effect_type: Literal["validation_error"] = "validation_error"
    error_message: str = Field(
        default="Tool input validation failed",
        description="Error message returned to the agent.",
    )

    def apply(self, context: Any = None) -> str:
        """Return the error message to cancel the tool call with."""
        return self.error_message


# ---------------------------------------------------------------------------
# Post-hook effects: corrupt the tool response after execution
# ---------------------------------------------------------------------------


class TruncateFields(ToolEffect):
    """Truncates string values in the tool response.

    The tool executes normally, but string fields in the response are
    truncated to at most `max_length` characters.

    Example::

        ChaosCase(
            name="search_truncated",
            input="Find flights",
            effects={
                "tool_effects": {"search_tool": [TruncateFields(max_length=5)]},
            },
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["truncate_fields"] = "truncate_fields"
    max_length: int = Field(default=10, ge=0, description="Maximum length to truncate string values to")

    def apply(self, response: Any = None) -> Any:
        """Truncate string values to max_length.

        Args:
            response: The tool response dict to corrupt.

        Returns:
            Response with string values truncated.
        """
        if not isinstance(response, dict):
            return response
        result: dict[str, Any] = {}
        for key, value in response.items():
            if isinstance(value, str) and len(value) > self.max_length:
                result[key] = value[: self.max_length]
            elif isinstance(value, dict):
                result[key] = self.apply(value)
            else:
                result[key] = value
        return result


class RemoveFields(ToolEffect):
    """Removes a fraction of fields from the tool response.

    The tool executes normally, but a portion of the response fields
    are deleted.

    Example::

        ChaosCase(
            name="db_remove_fields",
            input="Query database",
            effects={
                "tool_effects": {"database_tool": [RemoveFields(remove_ratio=0.5)]},
            },
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["remove_fields"] = "remove_fields"
    remove_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of fields to remove from the response",
    )

    def apply(self, response: Any = None) -> Any:
        """Remove a fraction of fields from the response.

        Always removes at least 1 field when called.

        Args:
            response: The tool response dict to corrupt.

        Returns:
            Response with fields removed.
        """
        if not isinstance(response, dict):
            return response
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

        ChaosCase(
            name="db_corrupt",
            input="Query database",
            effects={
                "tool_effects": {"database_tool": [CorruptValues(corrupt_ratio=0.8)]},
            },
        )
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["corrupt_values"] = "corrupt_values"
    corrupt_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of values to corrupt in the response",
    )

    _CORRUPTIONS: ClassVar[list[Any]] = [None, 99999, "", True, [], "CORRUPTED_DATA"]

    def apply(self, response: Any = None) -> Any:
        """Replace a fraction of values with wrong types or garbage data.

        Always corrupts at least 1 field when called.

        Args:
            response: The tool response dict to corrupt.

        Returns:
            Response with corrupted values.
        """
        if not isinstance(response, dict):
            return response
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
# Discriminated union type for Pydantic serialization
# ---------------------------------------------------------------------------

ToolEffectUnion = Annotated[
    Union[
        Annotated[Timeout, Tag("timeout")],
        Annotated[NetworkError, Tag("network_error")],
        Annotated[ExecutionError, Tag("execution_error")],
        Annotated[ValidationError, Tag("validation_error")],
        Annotated[TruncateFields, Tag("truncate_fields")],
        Annotated[RemoveFields, Tag("remove_fields")],
        Annotated[CorruptValues, Tag("corrupt_values")],
    ],
    Discriminator("effect_type"),
]
"""Discriminated union of all concrete tool effects.

Used in ChaosCase.effects to ensure full round-trip serialization fidelity
with Pydantic's model_dump() / model_validate().
"""
