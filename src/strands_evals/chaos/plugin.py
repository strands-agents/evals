"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system (BeforeToolCallEvent / AfterToolCallEvent).

The plugin reads the active scenario from a module-level ContextVar at hook
time. The ChaosExperiment manages the ContextVar lifecycle.
"""

import json
import logging

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook

from ._context import _current_scenario
from .effects import ChaosEffect

logger = logging.getLogger(__name__)


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on the active scenario.

    The plugin intercepts tool calls via Strands' native hook system:
    - BeforeToolCallEvent: cancels tool calls for pre-hook effects (ToolCallFailure)
    - AfterToolCallEvent: corrupts tool responses for post-hook effects (TruncateFields, etc.)

    The active scenario is managed via a ContextVar (set by ChaosExperiment).
    When no scenario is active, all tools behave normally.

    Example::

        from strands import Agent
        from strands_evals.chaos import ChaosPlugin

        chaos = ChaosPlugin()
        agent = Agent(
            model=my_model,
            tools=[search_tool, database_tool],
            plugins=[chaos],
        )

        # The ChaosExperiment handles scenario activation via ContextVar.
        # The user's task body contains zero chaos concepts.
    """

    name = "chaos-testing"

    def __init__(self) -> None:
        super().__init__()

    @hook  # type: ignore[call-overload]
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Intercept tool calls to inject pre-hook (error) effects.

        For error effects (Timeout, NetworkError, etc.), cancels the tool call
        with the effect's error_message before the tool executes.
        """
        scenario = _current_scenario.get()
        if scenario is None:
            return

        tool_name = event.tool_use.get("name", "")
        effects = scenario.effects.get(tool_name, [])
        if not effects:
            return

        # First pre-hook effect wins (tool is cancelled once)
        for effect in effects:
            if effect.hook == "pre":
                event.cancel_tool = effect.apply()
                logger.info(f"[Chaos] Injected {type(effect).__name__} on tool '{tool_name}'")
                return

    @hook  # type: ignore[call-overload]
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Intercept tool results to inject post-hook (corruption) effects.

        For corruption effects (TruncateFields, RemoveFields, CorruptValues),
        applies effect.apply() to JSON content blocks in the tool response.
        """
        scenario = _current_scenario.get()
        if scenario is None:
            return

        tool_name = event.tool_use.get("name", "")
        effects = scenario.effects.get(tool_name, [])
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

            logger.info(f"[Chaos] Applied {type(effect).__name__} on tool '{tool_name}'")

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
                        # Plain text — apply truncation via effect if applicable
                        if hasattr(effect, "max_length"):
                            block = {**block, "text": text_data[: effect.max_length]}
            corrupted_blocks.append(block)
        return corrupted_blocks
