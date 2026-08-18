"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system. Handles BOTH tool-level and model-output chaos:

- BeforeToolCallEvent: cancels tool calls for pre-hook effects (Timeout, etc.)
- AfterToolCallEvent: corrupts tool responses for post-hook effects (TruncateFields, etc.)
- BeforeModelCallEvent: cancels model call for pre-hook effects (FullRefusal, EmptyResponse)
- MessageAddedEvent: corrupts model output for post-hook effects (MalformedJson, Confabulation, etc.)
"""

import json
import logging
from enum import Enum, auto
from typing import NamedTuple, Protocol, cast

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
    MalformedJson,
    SuccessFraming,
    TruncateFields,
)

logger = logging.getLogger(__name__)


class MessageKind(Enum):
    """Kind of corruptible model output."""

    STRUCTURED_OUTPUT = auto()
    FINAL_TEXT = auto()


class ModelOutputTarget(NamedTuple):
    """A model output eligible for corruption."""

    kind: MessageKind
    content: list
    structured_output_tool_names: set[str]


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
    Effects are applied sequentially. SuccessFraming is always applied LAST
    (composable post-step). MalformedJson can reach structured-output toolUse
    blocks; other post effects skip toolUse messages.

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

    # Tool chaos hooks

    @hook  # type: ignore[call-overload]
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Intercept tool calls to inject pre-hook (error) effects.

        Cancels the tool call with the effect's error_message before execution.
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

    # Model output chaos hooks

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
        target = self._classify_model_output(event)
        if target is None:
            return
        applicable = self._applicable_model_effects(effects, target)
        if not applicable:
            return
        event.message["content"] = self._apply_to_model_blocks(
            applicable, target.content, target.structured_output_tool_names
        )
        logger.info(
            "effects=<%s> | applied model output chaos",
            ", ".join(type(e).__name__ for e in applicable),
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

    def _classify_model_output(self, event: MessageAddedEvent) -> ModelOutputTarget | None:
        """Return the corruptible target for this message, or None if it must be left alone.

        Ordinary tool dispatch is excluded: MessageAddedEvent fires before dispatch, so
        corrupting those blocks breaks the agent loop.
        """
        message = event.message
        if message.get("role") != "assistant":
            return None
        content = message.get("content")
        if content is None:
            return None
        if not isinstance(content, list):
            return ModelOutputTarget(MessageKind.FINAL_TEXT, content, set())

        if not any(isinstance(block, dict) and "toolUse" in block for block in content):
            return ModelOutputTarget(MessageKind.FINAL_TEXT, content, set())

        structured_output_tool_names = self._get_structured_output_tool_names(event.agent)
        targets_structured_output = any(
            isinstance(block, dict)
            and "toolUse" in block
            and block["toolUse"].get("name", "") in structured_output_tool_names
            for block in content
        )
        if targets_structured_output:
            return ModelOutputTarget(MessageKind.STRUCTURED_OUTPUT, content, structured_output_tool_names)
        return None

    def _get_structured_output_tool_names(self, agent) -> set[str]:  # type: ignore[type-arg]
        """Identify structured-output tools by their declared tool_type."""
        return {
            name for name, tool in agent.tool_registry.dynamic_tools.items() if tool.tool_type == "structured_output"
        }

    def _applicable_model_effects(self, effects: list, target: ModelOutputTarget) -> list:
        """Narrow the configured effects to those allowed against this target.

        Structured-output toolUse is reachable only by MalformedJson; any other effect
        would break the structured-output contract.
        """
        if target.kind is MessageKind.STRUCTURED_OUTPUT:
            return [e for e in effects if isinstance(e, MalformedJson)]
        return effects

    def _apply_to_model_blocks(
        self, post_effects: list, content: list, structured_output_tool_names: set[str] | None = None
    ) -> list:
        """Apply model post effects to content blocks sequentially.

        SuccessFraming runs last so it frames whatever the other effects produced.
        """
        primary = [e for e in post_effects if not isinstance(e, SuccessFraming)]
        framing = [e for e in post_effects if isinstance(e, SuccessFraming)]

        corrupted = content
        for effect in primary:
            if isinstance(effect, MalformedJson) and structured_output_tool_names:
                corrupted = self._apply_malformed_json_selective(effect, corrupted, structured_output_tool_names)
            else:
                corrupted = effect.apply(corrupted)
        for effect in framing:
            corrupted = effect.apply(corrupted)
        return corrupted

    def _apply_malformed_json_selective(
        self, effect: MalformedJson, blocks: list, structured_output_tool_names: set[str]
    ) -> list:
        """Apply MalformedJson: text blocks get malformed; only SO toolUse blocks get corrupted."""
        result = []
        for block in blocks:
            if isinstance(block, dict) and "toolUse" in block:
                tool_name = block["toolUse"].get("name", "")
                if tool_name in structured_output_tool_names:
                    block = effect._malform_tool_use_block(block)
                # else: ordinary toolUse — leave untouched
            elif isinstance(block, dict) and "text" in block and isinstance(block["text"], str):
                block = dict(block)
                block["text"] = effect._malform_text(block["text"])
            result.append(block)
        return result

    # Tool corruption helpers

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
