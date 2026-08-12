"""Chaos case definition.

A ChaosCase extends Case with chaos-specific fields, providing a stable
extension point for failure injection configuration without modifying the
base Case class.
"""

import uuid
from typing import cast

from pydantic import Field, model_validator
from typing_extensions import Generic

from ..case import Case
from ..types.evaluation import InputT, OutputT
from .effects import ModelEffect, ModelEffectUnion, ToolEffect, ToolEffectUnion

# Type alias for the effects dict structure
EffectsDict = dict[str, dict[str, list[ToolEffectUnion | ModelEffectUnion]]]


class ChaosCase(Case, Generic[InputT, OutputT]):
    """A test case with associated chaos effects.

    Extends Case to carry the effects mapping that the ChaosPlugin reads
    at hook time. A ChaosCase with empty effects is a baseline run.

    The ``expand`` class method provides the Cartesian product of cases ×
    effect maps, producing a flat list of ChaosCase objects ready for
    ChaosExperiment.

    Attributes:
        effects: A dict keyed by effect category. Supports ``"tool_effects"``
            mapping tool_name -> list of effects, and ``"model_effects"``
            mapping model_name (or ``"*"`` wildcard) -> list of effects.

    Example::

        from strands_evals import Case
        from strands_evals.chaos import ChaosCase
        from strands_evals.chaos.effects import FullRefusal, Timeout, TruncateFields

        # Direct construction with model effects
        chaos_case = ChaosCase(
            name="refusal_test",
            input="Tell me something",
            effects={
                "model_effects": {"*": [FullRefusal()]},
            },
        )

        # Direct construction with tool effects
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

    effects: EffectsDict = Field(
        default_factory=dict,
        description="Effect categories. Supports 'tool_effects' mapping "
        "tool_name -> list of effects, and 'model_effects' mapping "
        "model_name (or '*' wildcard) -> list of effects. "
        "Empty dict means baseline (no chaos).",
    )

    @model_validator(mode="after")
    def _validate_effects(self) -> "ChaosCase":
        """Validate effects configuration structure."""
        allowed_categories = {"tool_effects", "model_effects"}
        unknown = set(self.effects.keys()) - allowed_categories
        if unknown:
            raise ValueError(
                f"Unknown effect categories: {sorted(unknown)}. Allowed categories: {sorted(allowed_categories)}."
            )

        # Validate tool_effects: dict[str, list[ToolEffectUnion]]
        for tool_name, effects_list in self.tool_effects.items():
            if len(effects_list) > 1:
                raise ValueError(
                    f"Tool '{tool_name}' has {len(effects_list)} effects — only 1 is allowed per "
                    f"ChaosCase. Use separate ChaosCase instances to test effects independently."
                )
            # Fix A: enforce effect-family membership
            for effect in effects_list:
                if not isinstance(effect, ToolEffect):
                    raise ValueError(
                        f"Effect {type(effect).__name__} in tool_effects['{tool_name}'] is not a {ToolEffect.__name__}"
                    )

        # Validate model_effects: dict[str, list[ModelEffectUnion]]
        model_effects_map = self.effects.get("model_effects", {})
        if model_effects_map:
            if not isinstance(model_effects_map, dict):
                raise ValueError("'model_effects' must be a dict keyed by model name (or '*' wildcard).")
            # Fix B: reject non-"*" keys
            for model_name in model_effects_map:
                if model_name != "*":
                    raise ValueError(
                        f"model_effects key '{model_name}' is not supported; "
                        f"model targeting not yet implemented. Use '*' for all models."
                    )
            for model_name, effects_list in model_effects_map.items():  # type: ignore[assignment]
                if not isinstance(model_name, str):
                    raise ValueError(f"model_effects keys must be strings, got {type(model_name).__name__}.")
                if not isinstance(effects_list, list):
                    raise ValueError(
                        f"model_effects['{model_name}'] must be a list of model effects, "
                        f"got {type(effects_list).__name__}."
                    )
                # Fix A: enforce effect-family membership
                for effect in effects_list:
                    if not isinstance(effect, ModelEffect):
                        raise ValueError(
                            f"Effect {type(effect).__name__} in model_effects['{model_name}'] "
                            f"is not a {ModelEffect.__name__}"
                        )

        return self

    @classmethod
    def expand(
        cls,
        cases: list[Case],
        effect_maps: dict[str, EffectsDict],
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
                effect category (e.g. ``"tool_effects"``, ``"model_effects"``)
                mapping target -> list of effect instances.
                Example::

                    {
                        "search_timeout": {
                            "tool_effects": {"search_tool": [Timeout()]}
                        },
                        "refusal": {
                            "model_effects": {"*": [FullRefusal()]}
                        },
                    }
            include_no_effect_baseline: If True, includes a baseline (no chaos)
                variant for each case. Defaults to False.

        Returns:
            Flat list of ChaosCase objects with composite names like
            "flight_search|baseline" or "flight_search|search_timeout".
        """
        all_entries: list[tuple[str, EffectsDict]] = []

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
        return cast(dict[str, list[ToolEffectUnion]], self.effects.get("tool_effects", {}))

    @property
    def model_effects(self) -> list[ModelEffectUnion]:
        """Resolve model effects. '*' wildcard applies to all models."""
        model_effects_map = self.effects.get("model_effects", {})
        if not model_effects_map:
            return []
        # For now, resolve "*" (wildcard = applies to all models)
        return cast(list[ModelEffectUnion], model_effects_map.get("*", []))

    def __repr__(self) -> str:
        effects_str = ", ".join(
            f"{target}: [{', '.join(type(e).__name__ for e in effs)}]" for target, effs in self.tool_effects.items()
        )
        parts = [f"name='{self.name}'", f"effects={{{effects_str}}}"]
        if self.model_effects:
            model_str = ", ".join(type(e).__name__ for e in self.model_effects)
            parts.append(f"model_effects=[{model_str}]")
        return f"ChaosCase({', '.join(parts)})"
