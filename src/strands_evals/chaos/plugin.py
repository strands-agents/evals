"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system. Handles BOTH tool-level and model-output chaos:

- BeforeToolCallEvent: cancels tool calls for pre-hook effects (Timeout, etc.)
- AfterToolCallEvent: corrupts tool responses for post-hook effects (TruncateFields, etc.)
- BeforeModelCallEvent: cancels model call for pre-hook effects (FullRefusal)
- MessageAddedEvent: corrupts model output for post-hook effects (EmptyResponse, etc.)

Model output corruption uses dict mutation on event.message["content"]. This is
necessary because AfterModelCallEvent.stop_response is read-only (only `retry`
is writeable). Dict mutation bypasses _can_write (which only intercepts __setattr__).

The post-model-chaos callback is GUARDED to only corrupt the FINAL agent response:
- role == "assistant"
- message content contains NO toolUse blocks

This guard prevents destructive effects (EMPTY_RESPONSE, FULL_REFUSAL) from
deleting toolUse blocks on mid-turn tool_use messages, which would break the
agent loop. MessageAddedEvent fires BEFORE tool dispatch.

The pre-model-chaos callback uses BeforeModelCallEvent.cancel to inject a refusal
message before the model is called. The cancel path builds an assistant message,
sets stop_reason="end_turn", and ends the agent cycle — no guard needed since
there is no existing message at pre time.
"""

import json
import logging

from strands.hooks import (
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.plugins import Plugin, hook

from ._context import _current_chaos_case
from .effects import (
    ChaosEffect,
    SuccessFraming,
    TruncateFields,
)

logger = logging.getLogger(__name__)


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on configuration.

    Handles both tool-level chaos and model-output chaos:

    Tool chaos:
        - BeforeToolCallEvent: cancels tool calls for pre-hook effects
        - AfterToolCallEvent: corrupts tool responses for post-hook effects

    Model output chaos:
        - BeforeModelCallEvent: cancels model call for pre-hook effects (FullRefusal)
        - MessageAddedEvent: corrupts the final assistant response content (post effects)

    The active ChaosCase is managed via a ContextVar (set by ChaosExperiment).
    When no ChaosCase is active or the case has no model_effects, all hooks
    pass through without modification.

    Model output effects are configured via `model_effects` on the ChaosCase.
    Effects are applied sequentially. SuccessFraming is always applied LAST
    (composable post-step).

    Example::

        from strands import Agent
        from strands_evals.chaos import ChaosCase, ChaosPlugin
        from strands_evals.chaos.effects import FullRefusal, EmptyResponse

        chaos_case = ChaosCase(
            name="refusal_test",
            input="Tell me about quantum physics",
            model_effects=[FullRefusal()],
        )
        chaos = ChaosPlugin()
        agent = Agent(model=my_model, tools=[...], plugins=[chaos])

    NOTE on structured_output: When the agent uses structured_output_model, the
    final response is a toolUse block (containing the structured output tool call).
    The toolUse guard will skip these messages, so model chaos does NOT affect
    structured_output responses. This is intentional — corrupting the structured
    output tool call would break parsing. Future work may add a dedicated
    structured_output chaos effect that corrupts the tool input fields specifically.
    """

    name = "chaos-testing"

    # -----------------------------------------------------------------------
    # Tool chaos hooks
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
    # Model output chaos hooks
    # -----------------------------------------------------------------------

    @hook  # type: ignore[call-overload]
    def before_model_invocation(self, event: BeforeModelCallEvent) -> None:
        """Intercept model calls to inject pre-hook effects (FullRefusal).

        For pre-hook effects, cancels the model call by setting event.cancel
        to the effect's cancel_message. The SDK builds an assistant message
        from the cancel text, sets stop_reason="end_turn", and ends the cycle.

        No role/toolUse guard is needed here — there is no message yet at pre time.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None or not chaos_case.model_effects:
            return

        pre = [e for e in chaos_case.model_effects if e.hook == "pre"]
        if not pre:
            return

        # First pre effect wins (cancel short-circuits, only one can win)
        first_pre = pre[0]
        if hasattr(first_pre, "cancel_message"):
            event.cancel = first_pre.cancel_message()
        logger.info("effect=<%s> | injected model pre-hook (cancel)", type(first_pre).__name__)

    @hook  # type: ignore[call-overload]
    def after_model_invocation(self, event: MessageAddedEvent) -> None:
        """Intercept messages to corrupt the final assistant response.

        GUARD: corruption is applied ONLY when ALL conditions hold:
            1. message role == "assistant"
            2. message content contains NO toolUse blocks

        This prevents destructive effects from breaking mid-turn tool dispatch.
        MessageAddedEvent fires BEFORE the agent extracts toolUse blocks for
        execution, so corrupting a tool_use message would break the agent loop.

        Only post-hook effects are applied here. If no post effects exist (e.g.
        only FullRefusal in model_effects), this returns early to prevent
        double-corruption in mixed pre+post cases.

        NOTE: stop_reason is NOT available on MessageAddedEvent (the event only
        carries `message: Message`). We use the toolUse-presence check as a proxy:
        messages with toolUse blocks are mid-turn tool_use messages; messages
        without are final end_turn responses. This is reliable because end_turn
        messages never contain toolUse blocks.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None or not chaos_case.model_effects:
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

        # If any pre effects exist, they already produced the turn — skip post
        # to prevent double-corruption in mixed pre+post cases.
        pre_effects = [e for e in chaos_case.model_effects if e.hook == "pre"]
        if pre_effects:
            return

        # Filter to post effects only
        post_effects = [e for e in chaos_case.model_effects if e.hook == "post"]
        if not post_effects:
            return

        # Separate SuccessFraming from primary effects (applied last)
        primary_effects: list = [e for e in post_effects if not isinstance(e, SuccessFraming)]
        framing_effects: list = [e for e in post_effects if isinstance(e, SuccessFraming)]

        # Apply primary effects sequentially
        corrupted = content
        for effect in primary_effects:
            corrupted = effect.apply(corrupted)

        # Apply success framing last (composable post-step)
        for effect in framing_effects:
            corrupted = effect.apply(corrupted)

        # Mutate via dict assignment (NOT attribute assignment on the event)
        message["content"] = corrupted

        effect_names = ", ".join(type(e).__name__ for e in post_effects)
        logger.info(
            "effects=<%s> | applied model output chaos to assistant message",
            effect_names,
        )

    # -----------------------------------------------------------------------
    # Tool corruption helpers
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
