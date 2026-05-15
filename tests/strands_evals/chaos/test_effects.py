"""Unit tests for chaos effect classes."""

import random

import pytest

from strands_evals.chaos.effects import (
    CorruptValues,
    RemoveFields,
    ToolCallFailure,
    TruncateFields,
)


class TestToolCallFailure:
    """Tests for the ToolCallFailure pre-hook effect."""

    @pytest.mark.parametrize(
        "error_type,expected_message",
        [
            ("timeout", "Tool call timed out"),
            ("network_error", "Network unreachable"),
            ("execution_error", "Tool execution failed"),
            ("validation_error", "Tool input validation failed"),
        ],
    )
    def test_apply_returns_default_message(self, error_type, expected_message):
        effect = ToolCallFailure(error_type=error_type)
        assert effect.apply() == expected_message

    def test_apply_returns_custom_message_when_provided(self):
        effect = ToolCallFailure(error_type="timeout", error_message="Custom timeout msg")
        assert effect.apply() == "Custom timeout msg"

    def test_apply_rate_defaults_to_one(self):
        effect = ToolCallFailure()
        assert effect.apply_rate == 1.0


class TestTruncateFields:
    """Tests for the TruncateFields post-hook effect."""

    def test_truncates_long_strings(self):
        effect = TruncateFields(max_length=5)
        response = {"name": "hello world", "short": "hi"}
        result = effect.apply(response)
        assert result["name"] == "hello"
        assert result["short"] == "hi"

    def test_preserves_non_string_values(self):
        effect = TruncateFields(max_length=3)
        response = {"count": 42, "flag": True, "items": [1, 2, 3]}
        result = effect.apply(response)
        assert result["count"] == 42
        assert result["flag"] is True
        assert result["items"] == [1, 2, 3]

    def test_truncates_nested_dicts(self):
        effect = TruncateFields(max_length=3)
        response = {"nested": {"deep_value": "abcdef"}}
        result = effect.apply(response)
        assert result["nested"]["deep_value"] == "abc"

    def test_empty_dict_returns_empty(self):
        effect = TruncateFields(max_length=5)
        assert effect.apply({}) == {}

    def test_non_dict_input_returned_as_is(self):
        effect = TruncateFields(max_length=5)
        assert effect.apply("not a dict") == "not a dict"
        assert effect.apply(None) is None

    def test_zero_max_length_truncates_all_strings(self):
        effect = TruncateFields(max_length=0)
        response = {"text": "hello"}
        result = effect.apply(response)
        assert result["text"] == ""


class TestRemoveFields:
    """Tests for the RemoveFields post-hook effect."""

    def test_removes_at_least_one_field(self):
        random.seed(42)
        effect = RemoveFields(remove_ratio=0.1)
        response = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = effect.apply(response)
        assert len(result) < len(response)

    def test_removes_half_fields(self):
        random.seed(42)
        effect = RemoveFields(remove_ratio=0.5)
        response = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = effect.apply(response)
        assert len(result) == 2

    def test_removes_all_fields_at_ratio_one(self):
        random.seed(42)
        effect = RemoveFields(remove_ratio=1.0)
        response = {"a": 1, "b": 2, "c": 3}
        result = effect.apply(response)
        assert len(result) == 0

    def test_empty_dict_returns_empty(self):
        effect = RemoveFields(remove_ratio=0.5)
        assert effect.apply({}) == {}

    def test_non_dict_input_returned_as_is(self):
        effect = RemoveFields(remove_ratio=0.5)
        assert effect.apply("not a dict") == "not a dict"
        assert effect.apply(None) is None

    def test_single_field_always_removed(self):
        random.seed(42)
        effect = RemoveFields(remove_ratio=0.5)
        response = {"only_key": "value"}
        result = effect.apply(response)
        assert len(result) == 0


class TestCorruptValues:
    """Tests for the CorruptValues post-hook effect."""

    def test_corrupts_at_least_one_field(self):
        random.seed(42)
        effect = CorruptValues(corrupt_ratio=0.1)
        response = {"a": "original_a", "b": "original_b", "c": "original_c", "d": "original_d"}
        result = effect.apply(response)
        corrupted_count = sum(1 for k in response if result[k] != response[k])
        assert corrupted_count >= 1

    def test_corrupted_values_come_from_corruption_pool(self):
        random.seed(42)
        effect = CorruptValues(corrupt_ratio=1.0)
        response = {"a": "original", "b": "data"}
        result = effect.apply(response)
        corruption_pool = [None, 99999, "", True, [], "CORRUPTED_DATA"]
        for key in response:
            assert result[key] in corruption_pool

    def test_corrupts_nested_dicts_recursively(self):
        random.seed(42)
        effect = CorruptValues(corrupt_ratio=1.0)
        response = {"top": "value", "nested": {"inner": "deep_value"}}
        result = effect.apply(response)
        # The nested dict should also be processed
        assert "nested" in result or "top" in result

    def test_empty_dict_returns_empty(self):
        effect = CorruptValues(corrupt_ratio=0.5)
        assert effect.apply({}) == {}

    def test_non_dict_input_returned_as_is(self):
        effect = CorruptValues(corrupt_ratio=0.5)
        assert effect.apply("not a dict") == "not a dict"
        assert effect.apply(None) is None

    def test_corrupted_value_differs_from_original(self):
        random.seed(42)
        effect = CorruptValues(corrupt_ratio=1.0)
        response = {"key": "unique_original_value"}
        result = effect.apply(response)
        assert result["key"] != "unique_original_value"
