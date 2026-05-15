"""Chaos case definition.

A ChaosCase extends Case with chaos-specific fields, providing a stable
extension point for failure injection configuration without modifying the
base Case class.
"""

import uuid

from pydantic import Field
from typing_extensions import Generic

from ..case import Case
from ..types.evaluation import InputT, OutputT
from .effects import ChaosEffect


class ChaosCase(Case, Generic[InputT, OutputT]):
    """A test case with associated chaos effects.

    Extends Case to carry the effects mapping that the ChaosPlugin reads
    at hook time. A ChaosCase with empty effects is a baseline run.

    The ``expand`` class method provides the Cartesian product of cases ×
    effect maps, producing a flat list of ChaosCase objects ready for
    ChaosExperiment.

    Attributes:
        effects: Mapping of tool_name -> list of effects to inject for this case.
            Tools not listed behave normally. Empty dict means baseline (no chaos).

    Example::

        from strands_evals import Case
        from strands_evals.chaos import ChaosCase
        from strands_evals.chaos.effects import ToolCallFailure, TruncateFields

        # Direct construction
        chaos_case = ChaosCase(
            name="search_timeout",
            input="Find flights to Tokyo",
            effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
        )

        # Expansion from base cases × named effect maps
        cases = [
            Case(name="flight_search", input="Find flights to Tokyo"),
            Case(name="hotel_search", input="Find hotels in Tokyo"),
        ]
        effect_maps = {
            "search_timeout": {"search_tool": [ToolCallFailure(error_type="timeout")]},
            "search_truncated": {"search_tool": [TruncateFields(max_length=5)]},
        }
        chaos_cases = ChaosCase.expand(cases, effect_maps, include_no_effect_baseline=True)
        # Produces 6 ChaosCase objects: 2 cases × (2 effect maps + 1 baseline)
    """

    effects: dict[str, list[ChaosEffect]] = Field(
        default_factory=dict,
        description="Mapping of tool_name -> list of effects to inject for this case. "
        "Empty dict means baseline (no chaos).",
    )

    @classmethod
    def expand(
        cls,
        cases: list[Case],
        effect_maps: dict[str, dict[str, list[ChaosEffect]]],
        include_no_effect_baseline: bool = False,
    ) -> list["ChaosCase"]:
        """Generate the Cartesian product of cases × named effect maps.

        Produces a flat list of ChaosCase objects, one for each (case, effect_map)
        combination. Each ChaosCase gets a fresh session_id and a composite name
        built from the case name and the effect map key.

        Args:
            cases: Base test cases to expand.
            effect_maps: Named effect configurations. Keys are short human-readable
                names (used in the composite case name); values are mappings of
                tool_name -> list of ChaosEffect instances.
            include_no_effect_baseline: If True, includes a baseline (no chaos)
                variant for each case. Defaults to False.

        Returns:
            Flat list of ChaosCase objects with composite names like
            "flight_search|baseline" or "flight_search|search_timeout".
        """
        all_entries: list[tuple[str, dict[str, list[ChaosEffect]]]] = []

        if include_no_effect_baseline:
            all_entries.append(("baseline", {}))

        for name, effects in effect_maps.items():
            all_entries.append((name, effects))

        expanded: list[ChaosCase] = []
        for case in cases:
            for condition_name, effects in all_entries:
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
                        effects=effects,
                    )
                )

        return expanded

    def __repr__(self) -> str:
        effects_str = ", ".join(
            f"{target}: [{', '.join(type(e).__name__ for e in effs)}]" for target, effs in self.effects.items()
        )
        return f"ChaosCase(name='{self.name}', effects={{{effects_str}}})"
