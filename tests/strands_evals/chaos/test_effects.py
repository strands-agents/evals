"""Unit tests for chaos effect classes."""

import random

from strands_evals.chaos.effects import (
    CorruptValues,
    ExecutionError,
    NetworkError,
    RemoveFields,
    Timeout,
    TruncateFields,
    ValidationError,
)


class TestTimeout:
    """Tests for the Timeout pre-hook effect."""

    def test_apply_returns_default_message(self):
        effect = Timeout()
        assert effect.apply() == "Tool call timed out"

    def test_apply_returns_custom_message(self):
        effect = Timeout(error_message="Custom timeout msg")
        assert effect.apply() == "Custom timeout msg"

    def test_hook_is_pre(self):
        assert Timeout.hook == "pre"

    def test_effect_type(self):
        effect = Timeout()
        assert effect.effect_type == "timeout"


class TestNetworkError:
    """Tests for the NetworkError pre-hook effect."""

    def test_apply_returns_default_message(self):
        effect = NetworkError()
        assert effect.apply() == "Network unreachable"

    def test_apply_returns_custom_message(self):
        effect = NetworkError(error_message="Connection refused on port 5432")
        assert effect.apply() == "Connection refused on port 5432"

    def test_hook_is_pre(self):
        assert NetworkError.hook == "pre"

    def test_effect_type(self):
        effect = NetworkError()
        assert effect.effect_type == "network_error"


class TestExecutionError:
    """Tests for the ExecutionError pre-hook effect."""

    def test_apply_returns_default_message(self):
        effect = ExecutionError()
        assert effect.apply() == "Tool execution failed"

    def test_apply_returns_custom_message(self):
        effect = ExecutionError(error_message="Segfault in native code")
        assert effect.apply() == "Segfault in native code"

    def test_hook_is_pre(self):
        assert ExecutionError.hook == "pre"

    def test_effect_type(self):
        effect = ExecutionError()
        assert effect.effect_type == "execution_error"


class TestValidationError:
    """Tests for the ValidationError pre-hook effect."""

    def test_apply_returns_default_message(self):
        effect = ValidationError()
        assert effect.apply() == "Tool input validation failed"

    def test_apply_returns_custom_message(self):
        effect = ValidationError(error_message="Missing required field: origin")
        assert effect.apply() == "Missing required field: origin"

    def test_hook_is_pre(self):
        assert ValidationError.hook == "pre"

    def test_effect_type(self):
        effect = ValidationError()
        assert effect.effect_type == "validation_error"


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

    def test_effect_type(self):
        effect = TruncateFields()
        assert effect.effect_type == "truncate_fields"


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

    def test_effect_type(self):
        effect = RemoveFields()
        assert effect.effect_type == "remove_fields"


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

    def test_effect_type(self):
        effect = CorruptValues()
        assert effect.effect_type == "corrupt_values"
