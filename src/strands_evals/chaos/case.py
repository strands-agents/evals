"""Chaos case definition.

A ChaosCase extends Case with chaos-specific fields, providing a stable
extension point for failure injection configuration without modifying the
base Case class.
"""

import uuid

from pydantic import Field, model_validator
from typing_extensions import Generic

from ..case import Case
from ..types.evaluation import InputT, OutputT
from .effects import ToolEffectUnion


class ChaosCase(Case, Generic[InputT, OutputT]):
    """A test case with associated chaos effects.

    Extends Case to carry the effects mapping that the ChaosPlugin reads
    at hook time. A ChaosCase with empty effects is a baseline run.

    The ``expand`` class method provides the Cartesian product of cases ×
    effect maps, producing a flat list of ChaosCase objects ready for
    ChaosExperiment.

    Attributes:
        effects: A dict keyed by effect category. Currently supports
            ``"tool_effects"`` mapping tool_name -> list of effects.

    Example::

        from strands_evals import Case
        from strands_evals.chaos import ChaosCase
        from strands_evals.chaos.effects import Timeout, TruncateFields

        # Direct construction
        chaos_case = ChaosCase(
            name="search_timeout",
            input="Find flights to Tokyo",
            effects={"tool_effects": {"search_tool": [Timeout()]}},
        )

        # Expansion from base cases × named effect maps
        cases = [
            Case(name="flight_search", input="Find flights to Tokyo"),
            Case(name="hotel_search", input="Find hotels in Tokyo"),
        ]
        effect_maps = {
            "search_timeout": {"tool_effects": {"search_tool": [Timeout()]}},
            "search_truncated": {"tool_effects": {"search_tool": [TruncateFields(max_length=5)]}},
        }
        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        # Produces 6 ChaosCase objects: 2 cases × (2 effect maps + 1 baseline)
    """

    effects: dict[str, dict[str, list[ToolEffectUnion]]] = Field(
        default_factory=dict,
        description="Effect categories. Currently supports 'tool_effects' mapping "
        "tool_name -> list of effects. Empty dict means baseline (no chaos).",
    )

    @model_validator(mode="after")
    def _validate_tool_effects(self) -> "ChaosCase":
        """Validate tool effects configuration."""
        allowed_categories = {"tool_effects"}
        unknown = set(self.effects.keys()) - allowed_categories
        if unknown:
            raise ValueError(
                f"Unknown effect categories: {sorted(unknown)}. Allowed categories: {sorted(allowed_categories)}."
            )

        for tool_name, effects_list in self.tool_effects.items():
            if len(effects_list) > 1:
                raise ValueError(
                    f"Tool '{tool_name}' has {len(effects_list)} effects — only 1 is allowed per "
                    f"ChaosCase. Use separate ChaosCase instances to test effects independently."
                )
        return self

    @classmethod
    def expand(
        cls,
        cases: list[Case],
        effect_maps: dict[str, dict[str, dict[str, list[ToolEffectUnion]]]],
        include_no_effect_baseline: bool = False,
    ) -> list["ChaosCase"]:
        """Generate the Cartesian product of cases × named effect maps.

        Produces a flat list of ChaosCase objects, one for each (case, effect_map)
        combination. Each ChaosCase gets a fresh session_id and a composite name
        built from the case name and the effect map key.

        Args:
            cases: Base test cases to expand.
            effect_maps: Named effect configurations. Keys are short human-readable
                names (used in the composite case name); values are dicts keyed by
                effect category (e.g. ``"tool_effects"``) mapping tool_name -> list
                of effect instances.
                Example::

                    {
                        "search_timeout": {
                            "tool_effects": {"search_tool": [Timeout()]}
                        },
                    }
            include_no_effect_baseline: If True, includes a baseline (no chaos)
                variant for each case. Defaults to False.

        Returns:
            Flat list of ChaosCase objects with composite names like
            "flight_search|baseline" or "flight_search|search_timeout".
        """
        all_entries: list[tuple[str, dict[str, dict[str, list[ToolEffectUnion]]]]] = []

        if include_no_effect_baseline:
            all_entries.append(("baseline", {}))

        for name, effects_config in effect_maps.items():
            all_entries.append((name, effects_config))

        expanded: list[ChaosCase] = []
        for case in cases:
            for condition_name, effects_config in all_entries:
                session_id = str(uuid.uuid4())
                expanded_name = f"{case.name}|{condition_name}" if case.name else condition_name

                expanded.append(
                    cls(
                        name=expanded_name,
                        session_id=session_id,
                        input=case.input,
                        expected_output=case.expected_output,
                        expected_assertion=case.expected_assertion,
                        expected_trajectory=case.expected_trajectory,
                        expected_interactions=case.expected_interactions,
                        expected_environment_state=case.expected_environment_state,
                        metadata=case.metadata,
                        effects=effects_config,
                    )
                )

        return expanded

    @property
    def tool_effects(self) -> dict[str, list[ToolEffectUnion]]:
        """Convenience accessor for effects['tool_effects']."""
        return self.effects.get("tool_effects", {})

    def __repr__(self) -> str:
        effects_str = ", ".join(
            f"{target}: [{', '.join(type(e).__name__ for e in effs)}]" for target, effs in self.tool_effects.items()
        )
        return f"ChaosCase(name='{self.name}', effects={{{effects_str}}})"
