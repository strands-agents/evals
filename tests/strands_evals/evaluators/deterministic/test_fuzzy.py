import pytest

from strands_evals.evaluators.deterministic.fuzzy import FuzzyEquals
from strands_evals.types import EvaluationData


class TestFuzzyEqualsExact:
    def test_matches_expected_output(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output="Paris", expected_output="Paris")
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_fails_when_actual_differs(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output="London", expected_output="Paris")
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_explicit_value_takes_precedence_over_expected_output(self):
        evaluator = FuzzyEquals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris", expected_output="London")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_explicit_none_value_asserts_against_none(self):
        evaluator = FuzzyEquals(value=None)
        data = EvaluationData(input="q", actual_output=None, expected_output="ignored")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_explicit_none_value_fails_non_none_actual(self):
        evaluator = FuzzyEquals(value=None)
        data = EvaluationData(input="q", actual_output="something")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False


class TestIgnoreOrder:
    def test_list_order_ignored(self):
        evaluator = FuzzyEquals(ignore_order=True)
        data = EvaluationData(input="q", actual_output=[3, 1, 2], expected_output=[1, 2, 3])
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_list_order_matters_by_default(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output=[3, 1, 2], expected_output=[1, 2, 3])
        assert evaluator.evaluate(data)[0].test_pass is False

    def test_unordered_reports_extra_elements(self):
        evaluator = FuzzyEquals(ignore_order=True)
        data = EvaluationData(input="q", actual_output=[1, 2, 3], expected_output=[1, 2])
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "unexpected" in result.reason

    def test_unordered_nested_dicts(self):
        evaluator = FuzzyEquals(ignore_order=True)
        expected = [{"id": 1}, {"id": 2}]
        actual = [{"id": 2}, {"id": 1}]
        data = EvaluationData(input="q", actual_output=actual, expected_output=expected)
        assert evaluator.evaluate(data)[0].test_pass is True


class TestIgnoreKeyOrder:
    def test_key_order_ignored_by_default(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output={"b": 2, "a": 1}, expected_output={"a": 1, "b": 2})
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_key_order_enforced_when_disabled(self):
        evaluator = FuzzyEquals(ignore_key_order=False)
        data = EvaluationData(input="q", actual_output={"b": 2, "a": 1}, expected_output={"a": 1, "b": 2})
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "key order" in result.reason

    def test_key_order_match_when_disabled_and_ordered(self):
        evaluator = FuzzyEquals(ignore_key_order=False)
        data = EvaluationData(input="q", actual_output={"a": 1, "b": 2}, expected_output={"a": 1, "b": 2})
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_missing_key_reported(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output={"a": 1}, expected_output={"a": 1, "b": 2})
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "key set differs" in result.reason


class TestNumericTolerance:
    def test_within_tolerance_passes(self):
        evaluator = FuzzyEquals(numeric_tolerance=0.01)
        data = EvaluationData(input="q", actual_output=3.14, expected_output=3.14159)
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_outside_tolerance_fails(self):
        evaluator = FuzzyEquals(numeric_tolerance=0.01)
        data = EvaluationData(input="q", actual_output=3.0, expected_output=3.14159)
        assert evaluator.evaluate(data)[0].test_pass is False

    def test_exact_required_by_default(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output=3.14, expected_output=3.14159)
        assert evaluator.evaluate(data)[0].test_pass is False

    def test_tolerance_applies_nested(self):
        evaluator = FuzzyEquals(numeric_tolerance=0.05)
        data = EvaluationData(
            input="q",
            actual_output={"total": 100.0, "avg": 33.0},
            expected_output={"total": 102.0, "avg": 33.5},
        )
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_zero_values_exact_within_tolerance(self):
        evaluator = FuzzyEquals(numeric_tolerance=0.1)
        data = EvaluationData(input="q", actual_output=0.0, expected_output=0.0)
        assert evaluator.evaluate(data)[0].test_pass is True


class TestCaseSensitivity:
    def test_case_insensitive_passes(self):
        evaluator = FuzzyEquals(case_sensitive=False)
        data = EvaluationData(input="q", actual_output="PARIS", expected_output="paris")
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_case_sensitive_by_default(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output="PARIS", expected_output="paris")
        assert evaluator.evaluate(data)[0].test_pass is False


class TestIgnoreWhitespace:
    def test_leading_trailing_whitespace_ignored(self):
        evaluator = FuzzyEquals(ignore_whitespace=True)
        data = EvaluationData(input="q", actual_output="  hello  ", expected_output="hello")
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_internal_whitespace_collapsed(self):
        evaluator = FuzzyEquals(ignore_whitespace=True)
        data = EvaluationData(input="q", actual_output="a\t b\n c", expected_output="a b c")
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_whitespace_matters_by_default(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output="  hello  ", expected_output="hello")
        assert evaluator.evaluate(data)[0].test_pass is False


class TestSubsetMode:
    def test_dict_subset_passes(self):
        evaluator = FuzzyEquals(subset_mode=True)
        data = EvaluationData(
            input="q",
            actual_output={"a": 1, "b": 2, "c": 3},
            expected_output={"a": 1, "b": 2},
        )
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_dict_subset_fails_on_missing_key(self):
        evaluator = FuzzyEquals(subset_mode=True)
        data = EvaluationData(input="q", actual_output={"a": 1}, expected_output={"a": 1, "z": 9})
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "missing expected key" in result.reason

    def test_dict_subset_fails_on_wrong_value(self):
        evaluator = FuzzyEquals(subset_mode=True)
        data = EvaluationData(input="q", actual_output={"a": 2}, expected_output={"a": 1})
        assert evaluator.evaluate(data)[0].test_pass is False

    def test_ordered_list_subsequence_passes(self):
        evaluator = FuzzyEquals(subset_mode=True)
        data = EvaluationData(input="q", actual_output=[1, 2, 3, 4], expected_output=[2, 4])
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_ordered_list_subsequence_fails_when_out_of_order(self):
        evaluator = FuzzyEquals(subset_mode=True)
        data = EvaluationData(input="q", actual_output=[1, 2, 3, 4], expected_output=[4, 2])
        assert evaluator.evaluate(data)[0].test_pass is False

    def test_unordered_subset_list_passes(self):
        evaluator = FuzzyEquals(subset_mode=True, ignore_order=True)
        data = EvaluationData(input="q", actual_output=[3, 1, 2], expected_output=[2, 3])
        assert evaluator.evaluate(data)[0].test_pass is True


class TestTypeCoercion:
    def test_numeric_string_matches_number(self):
        evaluator = FuzzyEquals(type_coercion=True)
        data = EvaluationData(input="q", actual_output="42", expected_output=42)
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_float_string_matches_float(self):
        evaluator = FuzzyEquals(type_coercion=True)
        data = EvaluationData(input="q", actual_output="3.14", expected_output=3.14)
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_bool_string_matches_bool(self):
        evaluator = FuzzyEquals(type_coercion=True)
        data = EvaluationData(input="q", actual_output="true", expected_output=True)
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_no_coercion_by_default(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output="42", expected_output=42)
        assert evaluator.evaluate(data)[0].test_pass is False

    def test_coercion_with_tolerance(self):
        evaluator = FuzzyEquals(type_coercion=True, numeric_tolerance=0.01)
        data = EvaluationData(input="q", actual_output="3.14", expected_output=3.14159)
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_bool_not_matched_numerically(self):
        # bool is an int subclass; True must not equal 1 under coercion.
        evaluator = FuzzyEquals(type_coercion=True)
        data = EvaluationData(input="q", actual_output=True, expected_output=1)
        assert evaluator.evaluate(data)[0].test_pass is False


class TestTypeMismatch:
    def test_dict_vs_list_fails(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output=[1, 2], expected_output={"a": 1})
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "expected an object" in result.reason

    def test_list_vs_scalar_fails(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(input="q", actual_output="not-a-list", expected_output=[1, 2])
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "expected a list" in result.reason


class TestStructuredScenarios:
    def test_sql_result_rows_unordered(self):
        evaluator = FuzzyEquals(ignore_order=True, ignore_key_order=True)
        expected = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        actual = [{"name": "Bob", "id": 2}, {"name": "Alice", "id": 1}]
        data = EvaluationData(input="q", actual_output=actual, expected_output=expected)
        assert evaluator.evaluate(data)[0].test_pass is True

    def test_nested_json_all_knobs(self):
        evaluator = FuzzyEquals(
            ignore_order=True,
            numeric_tolerance=0.01,
            case_sensitive=False,
            ignore_whitespace=True,
        )
        expected = {"users": [{"name": "Alice", "score": 90.0}], "total": 100.0}
        actual = {"total": 100.5, "users": [{"score": 90.5, "name": " ALICE "}]}
        data = EvaluationData(input="q", actual_output=actual, expected_output=expected)
        assert evaluator.evaluate(data)[0].test_pass is True


class TestReasonAndSerialization:
    def test_reason_on_match(self):
        evaluator = FuzzyEquals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris")
        assert "matches" in evaluator.evaluate(data)[0].reason

    def test_reason_includes_path(self):
        evaluator = FuzzyEquals()
        data = EvaluationData(
            input="q",
            actual_output={"a": {"b": 1}},
            expected_output={"a": {"b": 2}},
        )
        result = evaluator.evaluate(data)[0]
        assert result.test_pass is False
        assert "$.a.b" in result.reason

    def test_to_dict_records_non_default_knobs(self):
        evaluator = FuzzyEquals(value=[1, 2], ignore_order=True, numeric_tolerance=0.01)
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "FuzzyEquals"
        assert d["value"] == [1, 2]
        assert d["ignore_order"] is True
        assert d["numeric_tolerance"] == 0.01

    def test_to_dict_omits_unset_value(self):
        evaluator = FuzzyEquals()
        d = evaluator.to_dict()
        assert "value" not in d
        assert d["evaluator_type"] == "FuzzyEquals"

    @pytest.mark.asyncio
    async def test_evaluate_async_delegates(self):
        evaluator = FuzzyEquals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris")
        results = await evaluator.evaluate_async(data)
        assert results[0].test_pass is True
