"""Unit tests for ChaosPlugin."""

import json
from unittest.mock import MagicMock

import pytest

from strands_evals.chaos import ChaosPlugin, ChaosScenario
from strands_evals.chaos._context import _current_scenario
from strands_evals.chaos.effects import (
    ToolCallFailure,
    TruncateFields,
)


@pytest.fixture
def chaos_plugin():
    return ChaosPlugin()


@pytest.fixture
def before_event():
    """Create a mock BeforeToolCallEvent."""
    event = MagicMock()
    event.tool_use = {"name": "search_tool"}
    event.cancel_tool = None
    return event


@pytest.fixture
def after_event():
    """Create a mock AfterToolCallEvent with list content."""
    event = MagicMock()
    event.tool_use = {"name": "search_tool"}
    event.result = {
        "content": [{"text": json.dumps({"title": "Long Title Here", "count": 42})}],
        "status": "success",
        "toolUseId": "tool-123",
    }
    return event


class TestChaosPluginBeforeToolCall:
    """Tests for the before_tool_call hook."""

    def test_no_scenario_active_does_nothing(self, chaos_plugin, before_event):
        token = _current_scenario.set(None)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool is None
        finally:
            _current_scenario.reset(token)

    def test_scenario_without_matching_tool_does_nothing(self, chaos_plugin, before_event):
        scenario = ChaosScenario(
            name="other_tool_fails",
            effects={"other_tool": [ToolCallFailure(error_type="timeout")]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool is None
        finally:
            _current_scenario.reset(token)

    def test_pre_hook_effect_cancels_tool(self, chaos_plugin, before_event):
        scenario = ChaosScenario(
            name="search_timeout",
            effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool == "Tool call timed out"
        finally:
            _current_scenario.reset(token)

    def test_post_hook_effect_does_not_cancel_tool(self, chaos_plugin, before_event):
        scenario = ChaosScenario(
            name="search_truncated",
            effects={"search_tool": [TruncateFields(max_length=5)]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool is None
        finally:
            _current_scenario.reset(token)

    def test_first_pre_hook_effect_wins(self, chaos_plugin, before_event):
        scenario = ChaosScenario(
            name="multi_pre",
            effects={
                "search_tool": [
                    ToolCallFailure(error_type="timeout"),
                    ToolCallFailure(error_type="network_error"),
                ]
            },
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool == "Tool call timed out"
        finally:
            _current_scenario.reset(token)


class TestChaosPluginAfterToolCall:
    """Tests for the after_tool_call hook."""

    def test_no_scenario_active_does_nothing(self, chaos_plugin, after_event):
        token = _current_scenario.set(None)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_scenario.reset(token)

    def test_scenario_without_matching_tool_does_nothing(self, chaos_plugin, after_event):
        scenario = ChaosScenario(
            name="other_tool",
            effects={"other_tool": [TruncateFields(max_length=3)]},
        )
        token = _current_scenario.set(scenario)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_scenario.reset(token)

    def test_post_hook_corrupts_json_text_blocks(self, chaos_plugin, after_event):
        scenario = ChaosScenario(
            name="truncate",
            effects={"search_tool": [TruncateFields(max_length=3)]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.after_tool_call(after_event)
            corrupted = json.loads(after_event.result["content"][0]["text"])
            assert corrupted["title"] == "Lon"
            assert corrupted["count"] == 42  # non-string preserved
        finally:
            _current_scenario.reset(token)

    def test_pre_hook_effect_ignored_in_after_hook(self, chaos_plugin, after_event):
        scenario = ChaosScenario(
            name="pre_only",
            effects={"search_tool": [ToolCallFailure(error_type="timeout")]},
        )
        token = _current_scenario.set(scenario)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_scenario.reset(token)

    def test_none_result_is_skipped(self, chaos_plugin):
        event = MagicMock()
        event.tool_use = {"name": "search_tool"}
        event.result = None

        scenario = ChaosScenario(
            name="truncate",
            effects={"search_tool": [TruncateFields(max_length=3)]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.after_tool_call(event)  # Should not raise
        finally:
            _current_scenario.reset(token)

    def test_plain_text_truncation(self, chaos_plugin):
        """Test that plain (non-JSON) text blocks get truncated if effect has max_length."""
        event = MagicMock()
        event.tool_use = {"name": "search_tool"}
        event.result = {
            "content": [{"text": "This is plain text, not JSON"}],
            "status": "success",
            "toolUseId": "tool-456",
        }

        scenario = ChaosScenario(
            name="truncate",
            effects={"search_tool": [TruncateFields(max_length=4)]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.after_tool_call(event)
            assert event.result["content"][0]["text"] == "This"
        finally:
            _current_scenario.reset(token)


class TestApplyRate:
    """Tests for the apply_rate probability check in ChaosPlugin."""

    def test_apply_rate_zero_skips_pre_hook_effect(self, chaos_plugin, before_event):
        """Effect with apply_rate=0.0 should never fire."""
        scenario = ChaosScenario(
            name="never_fires",
            effects={"search_tool": [ToolCallFailure(error_type="timeout", apply_rate=0.0)]},
        )
        token = _current_scenario.set(scenario)
        try:
            # Run multiple times to confirm it never fires
            for _ in range(20):
                before_event.cancel_tool = None
                chaos_plugin.before_tool_call(before_event)
                assert before_event.cancel_tool is None
        finally:
            _current_scenario.reset(token)

    def test_apply_rate_one_always_fires_pre_hook(self, chaos_plugin, before_event):
        """Effect with apply_rate=1.0 should always fire."""
        scenario = ChaosScenario(
            name="always_fires",
            effects={"search_tool": [ToolCallFailure(error_type="timeout", apply_rate=1.0)]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.before_tool_call(before_event)
            assert before_event.cancel_tool == "Tool call timed out"
        finally:
            _current_scenario.reset(token)

    def test_apply_rate_zero_skips_post_hook_effect(self, chaos_plugin, after_event):
        """Post-hook effect with apply_rate=0.0 should never fire."""
        scenario = ChaosScenario(
            name="never_truncates",
            effects={"search_tool": [TruncateFields(max_length=3, apply_rate=0.0)]},
        )
        token = _current_scenario.set(scenario)
        try:
            original_content = after_event.result["content"][0]["text"]
            chaos_plugin.after_tool_call(after_event)
            assert after_event.result["content"][0]["text"] == original_content
        finally:
            _current_scenario.reset(token)

    def test_apply_rate_one_always_fires_post_hook(self, chaos_plugin, after_event):
        """Post-hook effect with apply_rate=1.0 should always fire."""
        scenario = ChaosScenario(
            name="always_truncates",
            effects={"search_tool": [TruncateFields(max_length=3, apply_rate=1.0)]},
        )
        token = _current_scenario.set(scenario)
        try:
            chaos_plugin.after_tool_call(after_event)
            corrupted = json.loads(after_event.result["content"][0]["text"])
            assert corrupted["title"] == "Lon"
        finally:
            _current_scenario.reset(token)
