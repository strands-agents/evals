"""Tests for red team attack presets."""

from strands_evals.redteam.presets import (
    ATTACK_REGISTRY,
    HARMFUL_CONTENT,
    JAILBREAK,
    PROMPT_EXTRACTION,
)


def test_registry_exposes_expected_presets():
    """Guard against accidental renames/removals of built-in presets."""
    assert ATTACK_REGISTRY == {
        "jailbreak": JAILBREAK,
        "prompt_extraction": PROMPT_EXTRACTION,
        "harmful_content": HARMFUL_CONTENT,
    }
