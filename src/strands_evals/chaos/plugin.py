"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system (BeforeToolCallEvent / AfterToolCallEvent).

The plugin is stateless — it reads the active scenario from a module-level
ContextVar at hook time. The ChaosExperiment manages the ContextVar lifecycle.

The plugin is a thin router:
- Pre-hook effects: reads effect.error_message, cancels the tool call.
- Post-hook effects: calls effect.apply(response), uses the return value.
"""

import json
import logging
from typing import Any

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook

from ._context import _current_scenario
from .effects import ChaosEffect

logger = logging.getLogger(__name__)


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on the active scenario.

    The plugin intercepts tool calls via Strands' native hook system:
    - BeforeToolCallEvent: cancels tool calls for pre-hook effects (Timeout, NetworkError, etc.)
    - AfterToolCallEvent: corrupts tool responses for post-hook effects (TruncateFields, etc.)

    The active scenario is managed via a ContextVar (set by ChaosExperiment).
    When no scenario is active, all tools behave normally.

    The plugin is stateless — no set_active_scenario method, no instance state
    for the current scenario. This makes it safe under concurrent execution.

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

    @hook
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
                logger.info(
                    f"[Chaos] Injected {type(effect).__name__} on tool '{tool_name}'"
                )
                return

    @hook
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Intercept tool results to inject post-hook (corruption) effects.

        For corruption effects (TruncateFields, RemoveFields, CorruptValues),
        calls effect.apply(response) to mutate the tool response.

        Handles Strands ToolResult content shapes:
        - dict content: pass directly to effect.apply()
        - list of blocks: extract text dicts, parse JSON, apply effect
        - plain dict result: pass directly to effect.apply()

        Envelope fields (status, toolUseId) are preserved around the corruption.
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

            if not hasattr(event, "result") or event.result is None:
                continue

            result = event.result

            if hasattr(result, "content"):
                if isinstance(result.content, dict):
                    result.content = self._apply_with_envelope(effect, result.content)
                elif isinstance(result.content, list):
                    result.content = self._apply_to_blocks(effect, result.content)
            elif isinstance(result, dict):
                event.result = self._apply_with_envelope(effect, result)

            logger.info(f"[Chaos] Applied {type(effect).__name__} on tool '{tool_name}'")

    def _apply_with_envelope(self, effect: ChaosEffect, response: dict[str, Any]) -> dict[str, Any]:
        """Apply effect while preserving envelope fields."""
        envelope_fields = {"status", "toolUseId"}
        saved = {k: response[k] for k in envelope_fields if k in response}

        # Strip envelope before passing to effect
        payload = {k: v for k, v in response.items() if k not in envelope_fields}
        corrupted = effect.apply(payload)

        # Restore envelope
        corrupted.update(saved)
        return corrupted

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
