"""Chaos effect definitions.

Effects are first-class parameterized classes organized in a hierarchy:
    ChaosEffect → ToolEffect → concrete effects (Timeout, NetworkError, etc.)

Each concrete effect carries only the parameters meaningful to it.
The `hook` class variable indicates whether the effect fires pre-tool-call
(error effects) or post-tool-call (corruption effects).

Each concrete effect has an `effect_type` discriminator field for Pydantic
discriminated-union serialization, ensuring full round-trip fidelity.
"""

import json
import math
import random
import re
from abc import abstractmethod
from typing import Annotated, Any, Callable, ClassVar, Literal, Union

from pydantic import BaseModel, Discriminator, Field, Tag


class ChaosEffect(BaseModel):
    """Base for all chaos effects.

    Attributes:
        hook: Whether this effect fires pre-call ("pre") or post-call ("post").
    """

    hook: ClassVar[Literal["pre", "post"]]

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


class ModelEffect(ChaosEffect):
    """Effect that operates on model output content.

    Intermediate class parallel to ToolEffect. Enables type-based dispatch
    so the plugin can distinguish model-output effects from tool-level effects.
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"


# ---------------------------------------------------------------------------
# Helper — module-level (used by text-transforming model effects)
# ---------------------------------------------------------------------------


def _apply_text_to_blocks(blocks: list, fn: Callable[[str], str]) -> list:
    """Apply *fn* to every ``"text"`` value; leave other blocks untouched."""
    result = []
    for block in blocks:
        block = dict(block)
        if "text" in block and isinstance(block["text"], str):
            block["text"] = fn(block["text"])
        result.append(block)
    return result


# ---------------------------------------------------------------------------
# a) MalformedJson
# ---------------------------------------------------------------------------


class MalformedJson(ModelEffect):
    """Corrupts JSON structures in model output."""

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["malformed_json"] = "malformed_json"

    def apply(self, content: Any = None) -> Any:
        if content is None:
            raise ValueError("MalformedJson.apply() requires content")
        if isinstance(content, str):
            return self._malform_text(content)
        elif isinstance(content, list):
            return self._malform_blocks(content)
        raise ValueError(f"MalformedJson.apply() received unsupported type {type(content).__name__}")

    @staticmethod
    def _malform_text(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped[: len(stripped) // 2]
        return text

    @staticmethod
    def _malform_blocks(blocks: list) -> list:
        result = []
        for block in blocks:
            block = dict(block)
            if "toolUse" in block:
                tool_use = dict(block["toolUse"])
                raw = json.dumps(tool_use.get("input", {}))
                tool_use["input"] = raw[:-1] if raw.endswith("}") else raw + "{{{"
                block["toolUse"] = tool_use
            elif "text" in block and isinstance(block["text"], str):
                block["text"] = MalformedJson._malform_text(block["text"])
            result.append(block)
        return result


# ---------------------------------------------------------------------------
# b) EmptyResponse
# ---------------------------------------------------------------------------


class EmptyResponse(ModelEffect):
    """Returns empty content."""

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["empty_response"] = "empty_response"

    def apply(self, content: Any = None) -> Any:
        if content is None:
            raise ValueError("EmptyResponse.apply() requires content")
        if isinstance(content, str):
            return ""
        elif isinstance(content, list):
            return []
        raise ValueError(f"EmptyResponse.apply() received unsupported type {type(content).__name__}")


# ---------------------------------------------------------------------------
# c) Confabulation
# ---------------------------------------------------------------------------


class Confabulation(ModelEffect):
    """Injects fabricated citations into model output text."""

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["confabulation"] = "confabulation"

    _CONFABULATION_TEMPLATES: ClassVar[list[str]] = [
        "According to a 2019 study published in Nature, ",
        "Research conducted by MIT in 2021 demonstrated that ",
        "A comprehensive report by the World Health Organization found that ",
        "Based on data from the U.S. Census Bureau (2020), ",
        "As documented in the IEEE Transactions on Software Engineering, ",
        "A peer-reviewed analysis by Stanford researchers confirmed that ",
        "The European Commission's 2022 technical report states that ",
        "Historical records from the Smithsonian Institution indicate that ",
    ]

    def apply(self, content: Any = None) -> Any:
        if content is None:
            raise ValueError("Confabulation.apply() requires content")
        if isinstance(content, str):
            return self._confabulate(content)
        elif isinstance(content, list):
            return _apply_text_to_blocks(content, self._confabulate)
        raise ValueError(f"Confabulation.apply() received unsupported type {type(content).__name__}")

    def _confabulate(self, text: str) -> str:
        if not text:
            return text
        template = random.choice(self._CONFABULATION_TEMPLATES)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return template + text
        insert_pos = random.randint(1, len(sentences) - 1)
        sentences.insert(insert_pos, template)
        return " ".join(sentences)


# ---------------------------------------------------------------------------
# d) FullRefusal
# ---------------------------------------------------------------------------


class FullRefusal(ModelEffect):
    """Replaces model output with a refusal message."""

    hook: ClassVar[Literal["pre", "post"]] = "pre"
    effect_type: Literal["full_refusal"] = "full_refusal"

    _REFUSAL_TEMPLATES: ClassVar[list[str]] = [
        "I'm sorry, but I can't help with that request.",
        "I'm not able to help with this particular request. Could you try rephrasing?",
        "I can't assist with that. Please try a different approach.",
        "I'm unable to process this request at this time.",
        "This is beyond what I'm able to do. Please consider an alternative approach.",
        "I can't assist with the request as described. Could you provide more context?",
    ]

    def cancel_message(self) -> str:
        """Return a random refusal template string for use with event.cancel."""
        return random.choice(self._REFUSAL_TEMPLATES)

    def apply(self, content: Any = None) -> Any:
        if content is None:
            raise ValueError("FullRefusal.apply() requires content")
        template = random.choice(self._REFUSAL_TEMPLATES)
        if isinstance(content, str):
            return template
        elif isinstance(content, list):
            return [{"text": template}]
        raise ValueError(f"FullRefusal.apply() received unsupported type {type(content).__name__}")


# ---------------------------------------------------------------------------
# e) SuccessFraming (composable post-step)
# ---------------------------------------------------------------------------


class SuccessFraming(ModelEffect):
    """Prepends a confident success prefix to content.

    This is composable — applied AFTER another effect to disguise corruption.
    """

    hook: ClassVar[Literal["pre", "post"]] = "post"
    effect_type: Literal["success_framing"] = "success_framing"

    _SUCCESS_PREFIXES: ClassVar[list[str]] = [
        "Successfully completed the requested operation.",
        "Done! Here are the results you asked for.",
        "Great news \u2014 everything worked as expected.",
        "Operation finished successfully. Here's what I found:",
        "All done! The task has been completed without issues.",
        "I've successfully processed your request. Here's the output:",
        "Task completed. Below are the verified results:",
    ]

    def apply(self, content: Any = None) -> Any:
        if content is None:
            raise ValueError("SuccessFraming.apply() requires content")
        prefix = random.choice(self._SUCCESS_PREFIXES)
        if isinstance(content, str):
            return prefix + " " + content
        elif isinstance(content, list):
            # Prepend into first text block if one exists
            for block in content:
                if isinstance(block, dict) and "text" in block and isinstance(block["text"], str):
                    block["text"] = prefix + " " + block["text"]
                    return content
            # No text block — prepend a new one
            return [{"text": prefix}] + content
        raise ValueError(f"SuccessFraming.apply() received unsupported type {type(content).__name__}")


# ---------------------------------------------------------------------------
# ModelEffectUnion — discriminated union for Pydantic deserialization
# ---------------------------------------------------------------------------

ModelEffectUnion = Annotated[
    Union[
        Annotated[MalformedJson, Tag("malformed_json")],
        Annotated[EmptyResponse, Tag("empty_response")],
        Annotated[Confabulation, Tag("confabulation")],
        Annotated[FullRefusal, Tag("full_refusal")],
        Annotated[SuccessFraming, Tag("success_framing")],
    ],
    Discriminator("effect_type"),
]
