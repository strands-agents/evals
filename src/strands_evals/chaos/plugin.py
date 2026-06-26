"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system. Handles BOTH tool-level and model-output chaos:

- BeforeToolCallEvent: cancels tool calls for pre-hook effects (Timeout, etc.)
- AfterToolCallEvent: corrupts tool responses for post-hook effects (TruncateFields, etc.)
- MessageAddedEvent: corrupts model output for the final assistant response

Model output corruption uses dict mutation on event.message["content"]. This is
necessary because AfterModelCallEvent.stop_response is read-only (only `retry`
is writeable). Dict mutation bypasses _can_write (which only intercepts __setattr__).

The model-chaos callback is GUARDED to only corrupt the FINAL agent response:
- role == "assistant"
- message content contains NO toolUse blocks

This guard prevents destructive effects (EMPTY_RESPONSE, FULL_REFUSAL) from
deleting toolUse blocks on mid-turn tool_use messages, which would break the
agent loop. MessageAddedEvent fires BEFORE tool dispatch (event_loop.py L427-428
fires the event; L187 branches on stop_reason; L476 extracts toolUse from the
same message object).
"""

import json
import logging

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, MessageAddedEvent
from strands.plugins import Plugin, hook

from ._context import _current_chaos_case
from .effects import (
    ChaosEffect,
    ModelEffect,
    ModelEffectUnion,
    SuccessFraming,
    TruncateFields,
)

logger = logging.getLogger(__name__)


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on configuration.

    Handles both tool-level chaos (P0) and model-output chaos (P1):

    Tool chaos (P0):
        - BeforeToolCallEvent: cancels tool calls for pre-hook effects
        - AfterToolCallEvent: corrupts tool responses for post-hook effects

    Model output chaos (P1):
        - MessageAddedEvent: corrupts the final assistant response content

    The active ChaosCase is managed via a ContextVar (set by ChaosExperiment).
    When no ChaosCase is active or the case has no effects, all tools behave normally.

    Model output corruption is configured via `model_effects` on the ChaosPlugin
    instance. Effects are applied sequentially. SuccessFraming is always applied
    LAST (composable post-step).

    Example::

        from strands import Agent
        from strands_evals.chaos import ChaosPlugin
        from strands_evals.chaos.effects import EmptyResponse, SuccessFraming

        chaos = ChaosPlugin(
            model_effects=[EmptyResponse(), SuccessFraming()],
        )

        agent = Agent(
            model=my_model,
            tools=[search_tool, database_tool],
            plugins=[chaos],
        )

    NOTE on structured_output: When the agent uses structured_output_model, the
    final response is a toolUse block (containing the structured output tool call).
    The toolUse guard will skip these messages, so model chaos does NOT affect
    structured_output responses. This is intentional — corrupting the structured
    output tool call would break parsing. Future work may add a dedicated
    structured_output chaos effect that corrupts the tool input fields specifically.
    """

    name = "chaos-testing"

    def __init__(self, model_effects: list[ModelEffectUnion] | None = None) -> None:
        """Initialize the ChaosPlugin.

        Args:
            model_effects: Optional list of model-output effects to apply sequentially.
                When provided, enables model-output chaos on final assistant responses.
                SuccessFraming (if included) is always applied last regardless of list order.
                When None, only tool-level chaos (from ChaosCase effects) is active.
        """
        super().__init__()
        self._model_effects = model_effects

    @property
    def model_effects(self) -> list[ModelEffectUnion] | None:
        """The active model output effects list."""
        return self._model_effects

    @model_effects.setter
    def model_effects(self, value: list[ModelEffectUnion] | None) -> None:
        """Update the model output effects list."""
        self._model_effects = value

    # -----------------------------------------------------------------------
    # Tool chaos hooks (P0) — from PR #224
    # -----------------------------------------------------------------------

    @hook  # type: ignore[call-overload]
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Intercept tool calls to inject pre-hook (error) effects.

        For pre-hook effects (Timeout, NetworkError, ExecutionError,
        ValidationError), cancels the tool call with the effect's error_message
        before the tool executes.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None or not chaos_case.tool_effects:
            return

        tool_name = event.tool_use.get("name", "")
        effects = chaos_case.tool_effects.get(tool_name, [])
        if not effects:
            return

        # First pre-hook effect wins (tool is cancelled once)
        for effect in effects:
            if effect.hook == "pre":
                event.cancel_tool = effect.apply()
                logger.info("effect=<%s>, tool=<%s> | injected chaos pre-hook", type(effect).__name__, tool_name)
                return

    @hook  # type: ignore[call-overload]
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Intercept tool results to inject post-hook (corruption) effects.

        For corruption effects (TruncateFields, RemoveFields, CorruptValues),
        applies effect.apply() to JSON content blocks in the tool response.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None or not chaos_case.tool_effects:
            return

        tool_name = event.tool_use.get("name", "")
        effects = chaos_case.tool_effects.get(tool_name, [])
        if not effects:
            return

        # Apply all post-hook effects sequentially (they compose)
        for effect in effects:
            if effect.hook != "post":
                continue

            if event.result is None:
                continue

            result = event.result
            content = result.get("content")

            if isinstance(content, list):
                result["content"] = self._apply_to_blocks(effect, content)  # type: ignore[assignment]

            logger.info("effect=<%s>, tool=<%s> | applied chaos post-hook", type(effect).__name__, tool_name)

    # -----------------------------------------------------------------------
    # Model output chaos hook (P1)
    # -----------------------------------------------------------------------

    @hook  # type: ignore[call-overload]
    def message_added(self, event: MessageAddedEvent) -> None:
        """Intercept messages to corrupt the final assistant response.

        GUARD: corruption is applied ONLY when ALL conditions hold:
            1. message role == "assistant"
            2. message content contains NO toolUse blocks

        This prevents destructive effects from breaking mid-turn tool dispatch.
        MessageAddedEvent fires BEFORE the agent extracts toolUse blocks for
        execution, so corrupting a tool_use message would break the agent loop.

        NOTE: stop_reason is NOT available on MessageAddedEvent (the event only
        carries `message: Message`). We use the toolUse-presence check as a proxy:
        messages with toolUse blocks are mid-turn tool_use messages; messages
        without are final end_turn responses. This is reliable because end_turn
        messages never contain toolUse blocks.
        """
        if self._model_effects is None:
            return

        message = event.message

        # Guard 1: only assistant messages
        if message.get("role") != "assistant":
            return

        # Guard 2: skip messages with toolUse blocks (mid-turn tool dispatch)
        content = message.get("content")
        if content is None:
            return

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "toolUse" in block:
                    return

        # Separate SuccessFraming from primary effects (applied last)
        primary_effects: list[ModelEffect] = [e for e in self._model_effects if not isinstance(e, SuccessFraming)]
        framing_effects: list[ModelEffect] = [e for e in self._model_effects if isinstance(e, SuccessFraming)]

        # Apply primary effects sequentially
        corrupted = content
        for effect in primary_effects:
            corrupted = effect.apply(corrupted)

        # Apply success framing last (composable post-step)
        for effect in framing_effects:
            corrupted = effect.apply(corrupted)

        # Mutate via dict assignment (NOT attribute assignment on the event)
        message["content"] = corrupted

        effect_names = ", ".join(type(e).__name__ for e in self._model_effects)
        logger.info(
            "effects=<%s> | applied model output chaos to assistant message",
            effect_names,
        )

    # -----------------------------------------------------------------------
    # Tool corruption helpers (P0)
    # -----------------------------------------------------------------------

    def _apply_to_blocks(self, effect: ChaosEffect, blocks: list) -> list:
        """Apply effect to text blocks in a content list."""
        corrupted_blocks = []
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                text_data = block["text"]
                if isinstance(text_data, str):
                    try:
                        parsed = json.loads(text_data)
                        if isinstance(parsed, dict):
                            corrupted = effect.apply(parsed)
                            block = {**block, "text": json.dumps(corrupted)}
                    except (json.JSONDecodeError, ValueError):
                        # Plain text — apply truncation if effect is TruncateFields
                        if isinstance(effect, TruncateFields):
                            block = {**block, "text": text_data[: effect.max_length]}
            corrupted_blocks.append(block)
        return corrupted_blocks
