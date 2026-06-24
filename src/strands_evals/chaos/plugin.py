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
import random
from typing import Any

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, MessageAddedEvent
from strands.plugins import Plugin, hook

from ._context import _current_chaos_case
from .effects import ChaosEffect, TruncateFields
from .model_effects import (
    HALLUCINATION_TYPES,
    FormatCorruptionEffect,
    HallucinationEffect,
    RefusalEffect,
)
from .model_types import ModelOutputCorruptionConfig, ModelOutputCorruptionType
from .model_utils import _SUCCESS_PREFIXES

logger = logging.getLogger(__name__)


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on the active ChaosCase.

    Handles both tool-level chaos (P0) and model-output chaos (P1):

    Tool chaos (P0):
        - BeforeToolCallEvent: cancels tool calls for pre-hook effects
        - AfterToolCallEvent: corrupts tool responses for post-hook effects

    Model output chaos (P1):
        - MessageAddedEvent: corrupts the final assistant response content

    The active ChaosCase is managed via a ContextVar (set by ChaosExperiment).
    When no ChaosCase is active or the case has no effects, all tools and model
    output behave normally.

    Model output corruption is configured via `model_output_config` on the
    ChaosPlugin instance or via the ChaosCase effects dict (key: "model_effects").

    Example::

        from strands import Agent
        from strands_evals.chaos import ChaosPlugin
        from strands_evals.chaos.model_types import (
            ModelOutputCorruptionConfig,
            ModelOutputHallucinationType,
        )

        chaos = ChaosPlugin(
            model_output_config=ModelOutputCorruptionConfig(
                apply_rate=1.0,
                corruption_type=ModelOutputHallucinationType.CONFABULATION,
                add_success_framing=True,
            )
        )

        agent = Agent(
            model=my_model,
            tools=[search_tool, database_tool],
            plugins=[chaos],
        )
    """

    name = "chaos-testing"

    def __init__(self, model_output_config: ModelOutputCorruptionConfig | None = None) -> None:
        """Initialize the ChaosPlugin.

        Args:
            model_output_config: Optional configuration for model output corruption.
                When provided, enables model-output chaos on final assistant responses.
                When None, only tool-level chaos (from ChaosCase effects) is active.
        """
        super().__init__()
        self._model_output_config = model_output_config

    @property
    def model_output_config(self) -> ModelOutputCorruptionConfig | None:
        """The active model output corruption configuration."""
        return self._model_output_config

    @model_output_config.setter
    def model_output_config(self, value: ModelOutputCorruptionConfig | None) -> None:
        """Update the model output corruption configuration."""
        self._model_output_config = value

    # ------------------------------------------------------------------
    # Tool chaos hooks (P0) — from PR #224
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Model output chaos hook (P1)
    # ------------------------------------------------------------------

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
        if self._model_output_config is None:
            return

        if self._model_output_config.apply_rate <= 0:
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

        # Guard 3: apply_rate probabilistic check
        if random.random() >= self._model_output_config.apply_rate:
            return

        # Dispatch to effect
        corrupted = self._apply_model_corruption(content)

        # Apply success framing if content was mutated
        if self._model_output_config.add_success_framing and corrupted != content:
            corrupted = self._apply_success_framing(corrupted)

        # Mutate via dict assignment (NOT attribute assignment on the event)
        message["content"] = corrupted

        logger.info(
            "corruption_type=<%s> | applied model output chaos to assistant message",
            self._model_output_config.corruption_type.value,
        )

    # ------------------------------------------------------------------
    # Model corruption helpers
    # ------------------------------------------------------------------

    def _apply_model_corruption(self, content: Any) -> Any:
        """Dispatch to the appropriate model effect and apply corruption."""
        config = self._model_output_config
        assert config is not None
        ct = config.corruption_type
        
        effect: ChaosEffect
        if ct in HALLUCINATION_TYPES:
            effect = HallucinationEffect(config)
        elif ct == ModelOutputCorruptionType.FULL_REFUSAL:
            effect = RefusalEffect(config)
        else:
            effect = FormatCorruptionEffect(config)

        return effect.apply(content)

    @staticmethod
    def _apply_success_framing(content: Any) -> Any:
        """Prepend a success prefix to corrupted content."""
        prefix = random.choice(_SUCCESS_PREFIXES)

        if isinstance(content, str):
            return prefix + " " + content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block and isinstance(block["text"], str):
                    block["text"] = prefix + " " + block["text"]
                    return content
            return [{"text": prefix}] + content
        return content

    # ------------------------------------------------------------------
    # Tool corruption helpers (P0)
    # ------------------------------------------------------------------

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
