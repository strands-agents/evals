"""Chaos Plugin for Strands Agents.

Implements chaos injection as a standard Strands Plugin using the SDK's
native hook system (BeforeToolCallEvent / AfterToolCallEvent).
"""

import logging
import math
import random
from typing import Any, Optional

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook

from .effects import (
    TOOL_CORRUPTION_EFFECTS,
    TOOL_ERROR_EFFECTS,
    ToolChaosEffect,
    ChaosEffectConfig,
)
from .scenario import ChaosScenario

logger = logging.getLogger(__name__)


class ChaosPlugin(Plugin):
    """Strands Plugin that injects deterministic chaos based on the active scenario.

    The plugin intercepts tool calls via Strands' native hook system:
    - BeforeToolCallEvent: cancels tool calls for error effects (TIMEOUT, NETWORK_ERROR, etc.)
    - AfterToolCallEvent: corrupts tool responses for corruption effects (TRUNCATE_FIELDS, etc.)

    The active scenario is set externally (typically by ChaosExperiment) before
    each evaluation run. When no scenario is active, all tools behave normally.

    Example::

        from strands import Agent
        from strands_evals.chaos import ChaosPlugin, ChaosScenario, ToolChaosEffect

        chaos = ChaosPlugin()
        agent = Agent(
            model=my_model,
            tools=[search_tool, database_tool],
            plugins=[chaos],
        )

        # Activate a scenario
        chaos.set_active_scenario(ChaosScenario(
            name="search_timeout",
            tool_effects={"search_tool": ToolChaosEffect.TIMEOUT},
        ))

        result = agent("Find flights to Tokyo")
        # search_tool will be cancelled with a timeout error
    """

    name = "chaos-testing"

    def __init__(self) -> None:
        super().__init__()
        self._active_scenario: Optional[ChaosScenario] = None

    @property
    def active_scenario(self) -> Optional[ChaosScenario]:
        """The currently active chaos scenario, or None for baseline (no chaos)."""
        return self._active_scenario

    def set_active_scenario(self, scenario: Optional[ChaosScenario]) -> None:
        """Set the scenario that drives chaos injection for subsequent tool calls.

        Args:
            scenario: The scenario to activate, or None to disable chaos (baseline).
        """
        self._active_scenario = scenario
        if scenario:
            logger.info(f"Chaos scenario activated: {scenario.name}")
        else:
            logger.info("Chaos scenario cleared (baseline mode)")

    def _should_apply(self, config: ChaosEffectConfig) -> bool:
        """Check if the effect should fire based on apply_rate."""
        if config.apply_rate >= 1.0:
            return True
        if config.apply_rate <= 0.0:
            return False
        return random.random() < config.apply_rate

    @hook
    def before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Intercept tool calls to inject error effects.

        For error effects (TIMEOUT, NETWORK_ERROR, etc.), cancels the tool call
        with a simulated error message before the tool executes.
        """
        if not self._active_scenario:
            return

        tool_name = event.tool_use.get("name", "")
        chaos_effect = self._active_scenario.tool_effects.get(tool_name)
        if chaos_effect is None:
            return

        if isinstance(chaos_effect, ToolChaosEffect):
            chaos_config = ChaosEffectConfig(effect=chaos_effect)
        elif isinstance(chaos_effect, ChaosEffectConfig):
            chaos_config = chaos_effect
        else:
            raise TypeError(
                f"Unexpected effect type for tool '{tool_name}': {type(chaos_effect).__name__}. "
                f"Expected ToolChaosEffect or ChaosEffectConfig."
            )

        # Only handle error effects in the before hook
        if chaos_config.effect not in TOOL_ERROR_EFFECTS:
            return

        if not self._should_apply(chaos_config):
            return

        # Cancel the tool call with a simulated error
        error_message = chaos_config.error_message or f"Simulated {chaos_config.effect.value}"
        event.cancel_tool = error_message
        logger.info(
            f"[Chaos] Injected {chaos_config.effect.value} on tool '{tool_name}': {error_message}"
        )

    @hook
    def after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Intercept tool results to inject corruption effects.

        For corruption effects (TRUNCATE_FIELDS, REMOVE_FIELDS, CORRUPT_VALUES),
        mutates the tool response after successful execution.
        """
        if not self._active_scenario:
            return

        tool_name = event.tool_use.get("name", "")
        chaos_effect = self._active_scenario.tool_effects.get(tool_name)
        if chaos_effect is None:
            return

        if isinstance(chaos_effect, ToolChaosEffect):
            chaos_config = ChaosEffectConfig(effect=chaos_effect)
        elif isinstance(chaos_effect, ChaosEffectConfig):
            chaos_config = chaos_effect
        else:
            raise TypeError(
                f"Unexpected effect type for tool '{tool_name}': {type(chaos_effect).__name__}. "
                f"Expected ToolChaosEffect or ChaosEffectConfig."
            )

        # Only handle corruption effects in the after hook
        if chaos_config.effect not in TOOL_CORRUPTION_EFFECTS:
            return

        if not self._should_apply(chaos_config):
            return

        # Corrupt the tool result
        if hasattr(event, "result") and event.result is not None:
            result = event.result
            # Handle ToolResult-like objects with content
            if hasattr(result, "content") and isinstance(result.content, dict):
                result.content = self._apply_corruption(chaos_config, result.content)
            elif isinstance(result, dict):
                event.result = self._apply_corruption(chaos_config, result)

            logger.info(
                f"[Chaos] Applied {chaos_config.effect.value} corruption on tool '{tool_name}'"
            )

    # ------------------------------------------------------------------
    # Corruption helpers (private)
    # ------------------------------------------------------------------

    def _apply_corruption(self, effect_config: ChaosEffectConfig, response: Any) -> Any:
        """Apply a corruption effect to a tool response.

        Args:
            effect_config: The normalized effect configuration.
            response: The tool result to corrupt. Expected to be a dict.

        Returns:
            The corrupted response.
        """
        if not isinstance(response, dict):
            return response

        effect = effect_config.effect

        if effect == ToolChaosEffect.TRUNCATE_FIELDS:
            return self._truncate_fields(response)
        elif effect == ToolChaosEffect.REMOVE_FIELDS:
            ratio = effect_config.remove_ratio if effect_config.remove_ratio is not None else 0.33
            return self._remove_fields(response, ratio)
        elif effect == ToolChaosEffect.CORRUPT_VALUES:
            rate = effect_config.corrupt_rate if effect_config.corrupt_rate is not None else 0.4
            return self._corrupt_values(response, rate)

        return response

    @staticmethod
    def _truncate_fields(response: dict[str, Any]) -> dict[str, Any]:
        """Truncate string values to partial content."""
        result: dict[str, Any] = {}
        for key, value in response.items():
            if isinstance(value, str) and len(value) > 0:
                result[key] = value[: random.randint(0, max(0, len(value) - 1))]
            elif isinstance(value, dict):
                result[key] = ChaosPlugin._truncate_fields(value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _remove_fields(response: dict[str, Any], remove_ratio: float) -> dict[str, Any]:
        """Remove a fraction of fields from the response."""
        keys = list(response.keys())
        if not keys:
            return response

        num_to_remove = max(1, math.ceil(len(keys) * remove_ratio))
        keys_to_remove = set(random.sample(keys, min(num_to_remove, len(keys))))
        return {k: v for k, v in response.items() if k not in keys_to_remove}

    @staticmethod
    def _corrupt_values(response: dict[str, Any], corrupt_rate: float) -> dict[str, Any]:
        """Replace a fraction of values with wrong types or garbage data."""
        corruptions: list[Any] = [None, 99999, "", True, [], "CORRUPTED_DATA"]

        keys = list(response.keys())
        if not keys:
            return response

        num_to_corrupt = max(1, math.ceil(len(keys) * corrupt_rate))
        keys_to_corrupt = set(random.sample(keys, min(num_to_corrupt, len(keys))))

        result: dict[str, Any] = {}
        for key, value in response.items():
            if key in keys_to_corrupt:
                candidates = [c for c in corruptions if c != value]
                result[key] = random.choice(candidates) if candidates else "CORRUPTED_DATA"
            elif isinstance(value, dict):
                result[key] = ChaosPlugin._corrupt_values(value, corrupt_rate)
            else:
                result[key] = value
        return result
