"""Chaos scenario definition.

A ChaosScenario is a named, deterministic configuration of chaos effects
that will fire simultaneously when the scenario is active.
"""

from typing import Optional

from pydantic import BaseModel, Field

from .effects import ChaosEffect


class ChaosScenario(BaseModel):
    """A single, deterministic chaos injection scenario.

    Each scenario defines a set of tool effects that fire simultaneously when
    the scenario is active.

    Tools not listed in tool_effects behave normally (no chaos).

    Example::

        from strands_evals.chaos import ChaosScenario
        from strands_evals.chaos.effects import Timeout, NetworkError, CorruptValues

        # Baseline — no chaos
        ChaosScenario(name="baseline")

        # Single-fault: one tool fails
        ChaosScenario(
            name="search_timeout",
            effects={"search_tool": [Timeout()]},
        )

        # Compound: multiple tools/models fail simultaneously
        ChaosScenario(
            name="search_times_out_while_book_corrupts",
            description=(
                "Worst-case compound: primary path fails hard while the "
                "recovery path silently returns bad data."
            ),
            effects={
                "search_tool": [Timeout()],
                "book_tool": [CorruptValues(corrupt_ratio=0.8)],
            },
        )
    """

    name: str = Field(..., description="Human-readable name for this scenario")
    description: Optional[str] = Field(
        default=None,
        description="Optional description of what this scenario tests.",
    )
    effects: dict[str, list[ChaosEffect]] = Field(
        default_factory=dict,
        description="Mapping of target_name -> list of effects to inject simultaneously. "
        "Targets not listed here behave normally.",
    )

    def __repr__(self) -> str:
        effects_str = ", ".join(
            f"{target}: [{', '.join(type(e).__name__ for e in effs)}]" for target, effs in self.effects.items()
        )
        return f"ChaosScenario(name='{self.name}', effects={{{effects_str}}})"
