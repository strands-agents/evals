"""Chaos scenario definition.

A ChaosScenario is a named, deterministic mapping of tool names to the
chaos effects that will fire when those tools are invoked.
"""

from typing import Union

from pydantic import BaseModel, Field

from .effects import ToolChaosEffect, ChaosEffectConfig


# Type alias for what a tool_effects value can be
EffectSpec = Union[ToolChaosEffect, ChaosEffectConfig]


class ChaosScenario(BaseModel):
    """A single, deterministic chaos injection scenario.

    Each scenario maps tool names to the exact effect that fires when that
    tool is invoked. Tools not listed in tool_effects behave normally (no chaos).

    Example::

        # Simple: one tool fails with timeout
        ChaosScenario(
            name="search_timeout",
            tool_effects={"search_tool": ToolChaosEffect.TIMEOUT},
        )

        # Multiple tools affected
        ChaosScenario(
            name="both_tools_down",
            tool_effects={
                "search_tool": ToolChaosEffect.TIMEOUT,
                "database_tool": ToolChaosEffect.NETWORK_ERROR,
            },
        )

        # Advanced: custom parameters
        ChaosScenario(
            name="partial_corruption",
            tool_effects={
                "database_tool": ChaosEffectConfig(
                    effect=ToolChaosEffect.REMOVE_FIELDS,
                    remove_ratio=0.5,
                ),
            },
        )
    """

    name: str = Field(..., description="Human-readable name for this scenario")
    tool_effects: dict[str, EffectSpec] = Field(
        default_factory=dict,
        description="Mapping of tool_name -> effect to inject. "
        "Tools not listed here behave normally.",
    )

    def __repr__(self) -> str:
        effects_str = ", ".join(
            f"{tool}: {eff.value if isinstance(eff, ToolChaosEffect) else eff.effect.value}"
            for tool, eff in self.tool_effects.items()
        )
        return f"ChaosScenario(name='{self.name}', tool_effects={{{effects_str}}})"
