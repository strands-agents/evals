"""Tests for red team attack presets."""

import pytest

from strands_evals.redteam.presets import (
    ATTACK_REGISTRY,
    HARMFUL_CONTENT,
    JAILBREAK,
    PROMPT_EXTRACTION,
)

REQUIRED_FIELDS = ["traits", "context", "actor_goal", "seed_inputs", "severity", "evaluation_metrics"]


class TestAttackRegistry:
    """Tests for ATTACK_REGISTRY structure and completeness."""

    def test_registry_contains_all_presets(self):
        assert "jailbreak" in ATTACK_REGISTRY
        assert "prompt_extraction" in ATTACK_REGISTRY
        assert "harmful_content" in ATTACK_REGISTRY

    def test_registry_maps_to_correct_presets(self):
        assert ATTACK_REGISTRY["jailbreak"] is JAILBREAK
        assert ATTACK_REGISTRY["prompt_extraction"] is PROMPT_EXTRACTION
        assert ATTACK_REGISTRY["harmful_content"] is HARMFUL_CONTENT


class TestPresetStructure:
    """Tests that each preset has all required fields with valid values."""

    @pytest.mark.parametrize("preset_name", list(ATTACK_REGISTRY.keys()))
    def test_preset_has_required_fields(self, preset_name):
        preset = ATTACK_REGISTRY[preset_name]
        for field in REQUIRED_FIELDS:
            assert field in preset, f"'{preset_name}' missing field: {field}"

    @pytest.mark.parametrize("preset_name", list(ATTACK_REGISTRY.keys()))
    def test_preset_has_nonempty_seed_inputs(self, preset_name):
        preset = ATTACK_REGISTRY[preset_name]
        assert len(preset["seed_inputs"]) > 0, f"'{preset_name}' has no seed inputs"

    @pytest.mark.parametrize("preset_name", list(ATTACK_REGISTRY.keys()))
    def test_seed_inputs_are_strings(self, preset_name):
        preset = ATTACK_REGISTRY[preset_name]
        for seed in preset["seed_inputs"]:
            assert isinstance(seed, str), f"'{preset_name}' has non-string seed input"

    @pytest.mark.parametrize("preset_name", list(ATTACK_REGISTRY.keys()))
    def test_traits_has_attack_type(self, preset_name):
        preset = ATTACK_REGISTRY[preset_name]
        assert "attack_type" in preset["traits"]

    @pytest.mark.parametrize("preset_name", list(ATTACK_REGISTRY.keys()))
    def test_severity_is_valid(self, preset_name):
        preset = ATTACK_REGISTRY[preset_name]
        assert preset["severity"] in ("low", "medium", "high", "critical")
