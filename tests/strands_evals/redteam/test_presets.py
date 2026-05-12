"""Tests for red team attack presets."""

from strands_evals.redteam.presets import ATTACK_REGISTRY
from strands_evals.redteam.types import AttackPreset


def test_registry_exposes_expected_presets():
    """Guard against accidental renames/removals of built-in presets."""
    assert set(ATTACK_REGISTRY.keys()) == {"jailbreak", "prompt_extraction", "harmful_content"}


def test_registry_values_are_validated_presets():
    """All registry entries are AttackPreset instances (validated at import time)."""
    for name, preset in ATTACK_REGISTRY.items():
        assert isinstance(preset, AttackPreset), f"{name} is not an AttackPreset"
        assert preset.seed_inputs, f"{name} has no seed_inputs"
        assert preset.actor_goal, f"{name} has no actor_goal"
