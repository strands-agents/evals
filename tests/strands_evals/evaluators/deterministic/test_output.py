import pytest

from strands_evals.evaluators.deterministic.output import Contains, Equals, StartsWith
from strands_evals.types import EvaluationData


class TestEquals:
    def test_matches_expected_output(self):
        evaluator = Equals()
        data = EvaluationData(input="q", actual_output="Paris", expected_output="Paris")
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_fails_when_actual_differs_from_expected(self):
        evaluator = Equals()
        data = EvaluationData(input="q", actual_output="London", expected_output="Paris")
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_matches_explicit_value(self):
        evaluator = Equals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris")
        results = evaluator.evaluate(data)
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_explicit_value_takes_precedence_over_expected_output(self):
        evaluator = Equals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris", expected_output="London")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_none_actual_output_does_not_match_string(self):
        evaluator = Equals(value="Paris")
        data = EvaluationData(input="q", actual_output=None)
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_none_matches_none(self):
        evaluator = Equals()
        data = EvaluationData(input="q", actual_output=None, expected_output=None)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_non_string_types(self):
        evaluator = Equals(value=42)
        data = EvaluationData(input="q", actual_output=42)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_reason_on_match(self):
        evaluator = Equals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris")
        results = evaluator.evaluate(data)
        assert "matches" in results[0].reason

    def test_reason_on_mismatch(self):
        evaluator = Equals(value="Paris")
        data = EvaluationData(input="q", actual_output="London")
        results = evaluator.evaluate(data)
        assert "does not match" in results[0].reason

    @pytest.mark.asyncio
    async def test_evaluate_async_delegates_to_evaluate(self):
        evaluator = Equals(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris")
        results = await evaluator.evaluate_async(data)
        assert results[0].test_pass is True

    def test_to_dict(self):
        evaluator = Equals(value="Paris")
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "Equals"
        assert d["value"] == "Paris"

    def test_to_dict_no_value(self):
        evaluator = Equals()
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "Equals"
        assert "value" not in d


class TestContains:
    def test_substring_found(self):
        evaluator = Contains(value="Paris")
        data = EvaluationData(input="q", actual_output="The capital is Paris")
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_substring_not_found(self):
        evaluator = Contains(value="Paris")
        data = EvaluationData(input="q", actual_output="The capital is London")
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_case_sensitive_by_default(self):
        evaluator = Contains(value="paris")
        data = EvaluationData(input="q", actual_output="Paris is the capital")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_case_insensitive(self):
        evaluator = Contains(value="paris", case_sensitive=False)
        data = EvaluationData(input="q", actual_output="Paris is the capital")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_empty_string_always_found(self):
        evaluator = Contains(value="")
        data = EvaluationData(input="q", actual_output="anything")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_non_string_actual_output_coerced(self):
        evaluator = Contains(value="42")
        data = EvaluationData(input="q", actual_output=42)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_none_actual_output_coerced(self):
        evaluator = Contains(value="None")
        data = EvaluationData(input="q", actual_output=None)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_reason_on_found(self):
        evaluator = Contains(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris")
        results = evaluator.evaluate(data)
        assert "contains" in results[0].reason

    def test_reason_on_not_found(self):
        evaluator = Contains(value="Paris")
        data = EvaluationData(input="q", actual_output="London")
        results = evaluator.evaluate(data)
        assert "does not contain" in results[0].reason

    @pytest.mark.asyncio
    async def test_evaluate_async_delegates_to_evaluate(self):
        evaluator = Contains(value="Paris")
        data = EvaluationData(input="q", actual_output="Paris is great")
        results = await evaluator.evaluate_async(data)
        assert results[0].test_pass is True

    def test_to_dict(self):
        evaluator = Contains(value="Paris", case_sensitive=False)
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "Contains"
        assert d["value"] == "Paris"
        assert d["case_sensitive"] is False

    def test_to_dict_default_case_sensitive(self):
        evaluator = Contains(value="Paris")
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "Contains"
        assert "case_sensitive" not in d


class TestStartsWith:
    def test_prefix_matches(self):
        evaluator = StartsWith(value="The capital")
        data = EvaluationData(input="q", actual_output="The capital is Paris")
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_prefix_does_not_match(self):
        evaluator = StartsWith(value="The capital")
        data = EvaluationData(input="q", actual_output="Paris is the capital")
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_case_sensitive_by_default(self):
        evaluator = StartsWith(value="the capital")
        data = EvaluationData(input="q", actual_output="The capital is Paris")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is False

    def test_case_insensitive(self):
        evaluator = StartsWith(value="the capital", case_sensitive=False)
        data = EvaluationData(input="q", actual_output="The Capital Is Paris")
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_non_string_actual_output_coerced(self):
        evaluator = StartsWith(value="42")
        data = EvaluationData(input="q", actual_output=4200)
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_reason_on_match(self):
        evaluator = StartsWith(value="Hello")
        data = EvaluationData(input="q", actual_output="Hello world")
        results = evaluator.evaluate(data)
        assert "starts with" in results[0].reason

    def test_reason_on_mismatch(self):
        evaluator = StartsWith(value="Hello")
        data = EvaluationData(input="q", actual_output="Goodbye world")
        results = evaluator.evaluate(data)
        assert "does not start with" in results[0].reason

    @pytest.mark.asyncio
    async def test_evaluate_async_delegates_to_evaluate(self):
        evaluator = StartsWith(value="Hello")
        data = EvaluationData(input="q", actual_output="Hello world")
        results = await evaluator.evaluate_async(data)
        assert results[0].test_pass is True

    def test_to_dict(self):
        evaluator = StartsWith(value="Hello", case_sensitive=False)
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "StartsWith"
        assert d["value"] == "Hello"
        assert d["case_sensitive"] is False
