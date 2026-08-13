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
    """Classification of messages for model output chaos routing."""

    IRRELEVANT = auto()
    ORDINARY_TOOL_USE = auto()
    STRUCTURED_OUTPUT = auto()
    FINAL_TEXT = auto()


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
        """Intercept model calls to inject pre-hook effects (FullRefusal, EmptyResponse).

        Cancels the model call by setting event.cancel to the effect's cancel_message.
        No role/toolUse guard is needed — there is no message yet at pre time.
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
        """Apply post-hook model effects based on message classification."""
        chaos_case = _current_chaos_case.get()
        if chaos_case is None or not chaos_case.model_effects:
            return

        # Pre effects already produced the turn — skip post
        if any(e.hook == "pre" for e in chaos_case.model_effects):
            return

        post_effects = [e for e in chaos_case.model_effects if e.hook == "post"]
        if not post_effects:
            return

        kind, content, so_tool_names = self._classify_model_message(event)

        if kind == MessageKind.IRRELEVANT:
            return
        elif kind == MessageKind.ORDINARY_TOOL_USE:
            return
        elif kind == MessageKind.STRUCTURED_OUTPUT:
            # Only MalformedJson reaches structured-output toolUse
            if not any(isinstance(e, MalformedJson) for e in post_effects):
                return
            assert content is not None  # guaranteed by classifier for STRUCTURED_OUTPUT
            corrupted = self._apply_to_model_blocks(post_effects, content, so_tool_names)
        else:  # FINAL_TEXT
            assert content is not None  # guaranteed by classifier for FINAL_TEXT
            corrupted = self._apply_to_model_blocks(post_effects, content, set())

        event.message["content"] = corrupted
        effect_names = ", ".join(type(e).__name__ for e in post_effects)
        logger.info("effects=<%s> | applied model output chaos", effect_names)

    # Message classification

    def _classify_model_message(self, event: MessageAddedEvent) -> tuple[MessageKind, list | None, set[str]]:
        """Classify a message for chaos routing. Computed once per hook invocation."""
        message = event.message
        if message.get("role") != "assistant":
            return MessageKind.IRRELEVANT, None, set()
        content = message.get("content")
        if content is None:
            return MessageKind.IRRELEVANT, None, set()
        if not isinstance(content, list):
            return MessageKind.FINAL_TEXT, content, set()

        has_tool_use = any(isinstance(block, dict) and "toolUse" in block for block in content)
        if not has_tool_use:
            return MessageKind.FINAL_TEXT, content, set()

        # Has toolUse — determine if it's structured-output
        structured_output_tool_names = self._get_structured_output_tool_names(event.agent)
        has_so = any(
            isinstance(block, dict)
            and "toolUse" in block
            and block["toolUse"].get("name", "") in structured_output_tool_names
            for block in content
        )
        if has_so:
            return MessageKind.STRUCTURED_OUTPUT, content, structured_output_tool_names
        return MessageKind.ORDINARY_TOOL_USE, content, set()

    def _get_structured_output_tool_names(self, agent) -> set[str]:  # type: ignore[type-arg]
        """Identify structured-output tools via isinstance(tool, StructuredOutputTool)."""
        from strands.tools.structured_output.structured_output_tool import StructuredOutputTool

        return {
            name for name, tool in agent.tool_registry.dynamic_tools.items() if isinstance(tool, StructuredOutputTool)
        }

    # Model corruption helpers

    def _apply_to_model_blocks(
        self, post_effects: list, content: list, structured_output_tool_names: set[str] | None = None
    ) -> list:
        """Apply model post effects to content blocks sequentially.

        Handles text blocks (Confabulation, SuccessFraming, MalformedJson on text)
        and toolUse blocks (MalformedJson on structured-output tool input only).
        Ordinary mid-turn toolUse blocks are left untouched.
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
                    block = MalformedJson.malform_tool_use_block(block)
                # else: ordinary toolUse — leave untouched
            elif isinstance(block, dict) and "text" in block and isinstance(block["text"], str):
                block = dict(block)
                block["text"] = MalformedJson.malform_text(block["text"])
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
