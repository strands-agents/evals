"""Tests for shared prompt-block assembly in strategies/_common.py."""

from strands_evals.experimental.redteam.strategies._common import (
    EXIT_BLOCK,
    FORMAT_BLOCK,
    PROFILE_BLOCK,
    ROLE_BLOCK,
    RULES_BLOCK,
    _build_system_prompt,
)


def test_build_system_prompt_includes_all_common_blocks():
    prompt = _build_system_prompt("PLAYBOOK CONTENT")

    assert ROLE_BLOCK in prompt
    assert PROFILE_BLOCK in prompt
    assert "PLAYBOOK CONTENT" in prompt
    assert RULES_BLOCK in prompt
    assert EXIT_BLOCK in prompt
    assert FORMAT_BLOCK in prompt


def test_build_system_prompt_orders_blocks_role_profile_playbook_rules_exit_format():
    """Downstream .format() calls fill {actor_goal}/{max_turns} placeholders positionally
    within this fixed layout, so the block order is load-bearing, not incidental."""
    prompt = _build_system_prompt("PLAYBOOK")

    positions = [
        prompt.index(ROLE_BLOCK),
        prompt.index(PROFILE_BLOCK),
        prompt.index("PLAYBOOK"),
        prompt.index(RULES_BLOCK),
        prompt.index(EXIT_BLOCK),
        prompt.index(FORMAT_BLOCK),
    ]

    assert positions == sorted(positions)


def test_profile_and_exit_blocks_retain_format_placeholders():
    """These blocks are filled in later via str.format(actor_profile=..., max_turns=...);
    _build_system_prompt itself must not consume those placeholders."""
    prompt = _build_system_prompt("PLAYBOOK")

    assert "{actor_profile}" in prompt
    assert "{max_turns}" in prompt
