"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system. Handles BOTH tool-level and model-output chaos:

- BeforeToolCallEvent: cancels tool calls for pre-hook effects (Timeout, etc.), and
  injects one structured-output parse failure per invocation for MalformedJson
- AfterToolCallEvent: corrupts tool responses for post-hook effects (TruncateFields, etc.)
- BeforeModelCallEvent: cancels model call for pre-hook effects (FullRefusal, EmptyResponse)
- MessageAddedEvent: corrupts final text model output for post-hook effects
  (Confabulation, MalformedJson, SuccessFraming)

MalformedJson does not corrupt structured-output payloads in the message history.
Instead it injects a single structured-output parse failure per agent invocation via
BeforeToolCallEvent.cancel_tool, which tests whether the agent recovers; the SDK's
corrected attempt passes through unchanged, so a typed caller still receives validated
structured output. after_model_invocation never touches messages carrying toolUse blocks.
"""

import json
import logging
from typing import Protocol, cast

from strands.hooks import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.plugins import Plugin, hook

from ._context import _current_chaos_case
from .case import ChaosCase
from .effects import (
    ChaosEffect,
    MalformedJson,
    SuccessFraming,
    TruncateFields,
)

logger = logging.getLogger(__name__)

_CHAOS_STATE_KEY = "strands_evals.chaos"
_MALFORMED_OUTPUT_APPLIED = "malformed_structured_output_applied"


class PreModelEffect(Protocol):
    """A model effect that cancels the model call with a message."""

    def cancel_message(self) -> str: ...


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on configuration.

    Handles both tool-level chaos and model-output chaos:

    Tool chaos:
        - BeforeToolCallEvent: cancels tool calls for pre-hook effects
        - AfterToolCallEvent: corrupts tool responses for post-hook effects

    Model output chaos:
        - BeforeModelCallEvent: cancels model call for pre-hook effects (FullRefusal, EmptyResponse)
        - MessageAddedEvent: corrupts the final assistant response content (post effects)

    The active ChaosCase is managed via a ContextVar (set by ChaosExperiment).
    When no ChaosCase is active or the case has no model_effects, all hooks
    pass through without modification.

    Model output effects are configured via `model_effects` on the ChaosCase.
    Post effects apply to final text responses only and are applied sequentially, with
    SuccessFraming always LAST (composable post-step). Messages carrying toolUse blocks
    are never corrupted; MalformedJson instead injects one structured-output parse
    failure per invocation at the tool boundary.

    Example::

        from strands import Agent
        from strands_evals.chaos import ChaosCase, ChaosPlugin
        from strands_evals.chaos.effects import FullRefusal, EmptyResponse

        chaos_case = ChaosCase(
            name="refusal_test",
            input="Tell me about quantum physics",
            effects={
                "model_effects": {"*": [FullRefusal()]},
            },
        )
        chaos = ChaosPlugin()
        agent = Agent(model=my_model, tools=[...], plugins=[chaos])
    """

    name = "chaos-testing"

    @hook  # type: ignore[call-overload]
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Intercept tool calls to inject pre-hook (error) effects.

        Cancels the tool call with the effect's error_message before execution.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None:
            return

        if self._inject_structured_output_failure(event, chaos_case):
            return

        if not chaos_case.tool_effects:
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

    def _inject_structured_output_failure(self, event: BeforeToolCallEvent, chaos_case: ChaosCase) -> bool:
        """Fail the first structured-output attempt so the agent must recover.

        Cancelling the tool produces an error toolResult, which drives the SDK's
        structured-output correction loop. The invocation_state marker makes this fire
        exactly once per agent invocation, so the corrected attempt passes through and
        the caller still receives validated structured output.
        """
        effect = next((e for e in chaos_case.model_effects if isinstance(e, MalformedJson)), None)
        if effect is None:
            return False
        if event.selected_tool is None:
            return False
        if event.selected_tool.tool_type != "structured_output":
            return False

        state = event.invocation_state.setdefault(_CHAOS_STATE_KEY, {})
        if state.get(_MALFORMED_OUTPUT_APPLIED):
            return False
        state[_MALFORMED_OUTPUT_APPLIED] = True

        event.cancel_tool = (
            "Structured output was malformed and could not be parsed. Please produce a corrected response."
        )
        logger.info("effect=<%s> | injected structured output parse failure", type(effect).__name__)
        return True

    @hook  # type: ignore[call-overload]
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Intercept tool results to inject post-hook (corruption) effects.

        Applies corruption effects to JSON content blocks in the tool response.
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
                result["content"] = self._apply_to_tool_blocks(effect, content)  # type: ignore[assignment]

            logger.info("effect=<%s>, tool=<%s> | applied chaos post-hook", type(effect).__name__, tool_name)

    @hook  # type: ignore[call-overload]
    def after_invocation(self, event: AfterInvocationEvent) -> None:
        """Clear per-invocation chaos state.

        The SDK uses the caller-supplied invocation_state by reference and does not strip
        plugin keys when the invocation ends, so a reused dict would suppress injection on
        every subsequent invocation.
        """
        chaos_state = event.invocation_state.get(_CHAOS_STATE_KEY)
        if not isinstance(chaos_state, dict):
            return
        chaos_state.pop(_MALFORMED_OUTPUT_APPLIED, None)
        if not chaos_state:
            event.invocation_state.pop(_CHAOS_STATE_KEY, None)

    @hook  # type: ignore[call-overload]
    def before_model_invocation(self, event: BeforeModelCallEvent) -> None:
        """Cancel the model call when a pre-hook model effect is configured."""
        effect = self._select_pre_model_effect()
        if effect is None:
            return
        event.cancel = effect.cancel_message()
        logger.info("effect=<%s> | injected model pre-hook cancel", type(effect).__name__)

    @hook  # type: ignore[call-overload]
    def after_model_invocation(self, event: MessageAddedEvent) -> None:
        """Corrupt eligible model output with the configured post-hook model effects."""
        effects = self._get_post_model_effects()
        if not effects:
            return
        content = self._classify_model_output(event)
        if content is None:
            return
        event.message["content"] = self._apply_to_model_blocks(effects, content)
        logger.info(
            "effects=<%s> | applied model output chaos",
            ", ".join(type(e).__name__ for e in effects),
        )

    def _select_pre_model_effect(self) -> PreModelEffect | None:
        """Return the single configured pre-hook model effect, or None.

        ChaosCase validation guarantees at most one pre effect, so no ordering policy is needed.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None:
            return None
        for effect in chaos_case.model_effects:
            if effect.hook == "pre":
                return cast(PreModelEffect, effect)
        return None

    def _get_post_model_effects(self) -> list:
        """Return the configured post-hook model effects.

        Empty when a pre effect is configured: the pre effect already produced the turn,
        so applying post effects would corrupt it twice.
        """
        chaos_case = _current_chaos_case.get()
        if chaos_case is None:
            return []
        if any(e.hook == "pre" for e in chaos_case.model_effects):
            return []
        return [e for e in chaos_case.model_effects if e.hook == "post"]

    def _classify_model_output(self, event: MessageAddedEvent) -> list | None:
        """Return the final-text content eligible for corruption, or None to leave the message alone.

        Any message carrying a toolUse block is left alone: MessageAddedEvent fires before
        dispatch, and structured-output failures are injected at the tool boundary instead.
        """
        message = event.message
        if message.get("role") != "assistant":
            return None
        content = message.get("content")
        if content is None:
            return None
        if isinstance(content, list) and any(isinstance(b, dict) and "toolUse" in b for b in content):
            return None
        return content

    def _apply_to_model_blocks(self, post_effects: list, content: list) -> list:
        """Apply model post effects to content blocks sequentially.

        SuccessFraming runs last so it frames whatever the other effects produced.
        """
        primary = [e for e in post_effects if not isinstance(e, SuccessFraming)]
        framing = [e for e in post_effects if isinstance(e, SuccessFraming)]

        corrupted = content
        for effect in primary:
            corrupted = effect.apply(corrupted)
        for effect in framing:
            corrupted = effect.apply(corrupted)
        return corrupted

    def _apply_to_tool_blocks(self, effect: ChaosEffect, blocks: list) -> list:
        """Apply effect to text blocks in a tool content list."""
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
