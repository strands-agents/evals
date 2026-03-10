import pytest

from strands_evals.evaluators.deterministic.environment_state import StateEquals
from strands_evals.types import EnvironmentState, EvaluationData


class TestStateEquals:
    def test_matches_expected_state_by_name(self):
        evaluator = StateEquals(name="test_results")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
            expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 1.0
        assert results[0].test_pass is True

    def test_fails_when_state_differs(self):
        evaluator = StateEquals(name="test_results")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 1})],
            expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        results = evaluator.evaluate(data)
        assert len(results) == 1
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_fails_when_named_state_not_found(self):
        evaluator = StateEquals(name="test_results")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="other_state", state="something")],
            expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_none_actual_environment_state(self):
        evaluator = StateEquals(name="test_results")
        data = EvaluationData(
            input="q",
            actual_environment_state=None,
            expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        results = evaluator.evaluate(data)
        assert results[0].score == 0.0
        assert results[0].test_pass is False

    def test_explicit_value_takes_precedence(self):
        evaluator = StateEquals(name="test_results", value={"exit_code": 0})
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
            expected_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 99})],
        )
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_explicit_value_without_expected_state(self):
        evaluator = StateEquals(name="test_results", value=42)
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state=42)],
        )
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_non_string_state_values(self):
        evaluator = StateEquals(name="count")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="count", state=42)],
            expected_environment_state=[EnvironmentState(name="count", state=42)],
        )
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_list_state_values(self):
        evaluator = StateEquals(name="files")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="files", state=["a.py", "b.py"])],
            expected_environment_state=[EnvironmentState(name="files", state=["a.py", "b.py"])],
        )
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_multiple_states_finds_correct_one(self):
        evaluator = StateEquals(name="db_state")
        data = EvaluationData(
            input="q",
            actual_environment_state=[
                EnvironmentState(name="test_results", state={"exit_code": 0}),
                EnvironmentState(name="db_state", state={"rows": 5}),
            ],
            expected_environment_state=[
                EnvironmentState(name="db_state", state={"rows": 5}),
            ],
        )
        results = evaluator.evaluate(data)
        assert results[0].test_pass is True

    def test_reason_on_match(self):
        evaluator = StateEquals(name="test_results", value={"exit_code": 0})
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        results = evaluator.evaluate(data)
        assert "matches" in results[0].reason

    def test_reason_on_mismatch(self):
        evaluator = StateEquals(name="test_results", value={"exit_code": 0})
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 1})],
        )
        results = evaluator.evaluate(data)
        assert "does not match" in results[0].reason

    def test_reason_on_state_not_found(self):
        evaluator = StateEquals(name="missing")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="other", state="x")],
            expected_environment_state=[EnvironmentState(name="missing", state="x")],
        )
        results = evaluator.evaluate(data)
        assert "not found" in results[0].reason

    @pytest.mark.asyncio
    async def test_evaluate_async_delegates_to_evaluate(self):
        evaluator = StateEquals(name="test_results", value={"exit_code": 0})
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        results = await evaluator.evaluate_async(data)
        assert results[0].test_pass is True

    def test_raises_when_no_expected_environment_state(self):
        evaluator = StateEquals(name="test_results")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="test_results", state={"exit_code": 0})],
        )
        with pytest.raises(ValueError, match="no expected value"):
            evaluator.evaluate(data)

    def test_raises_when_name_not_in_expected_environment_state(self):
        evaluator = StateEquals(name="missing_state")
        data = EvaluationData(
            input="q",
            actual_environment_state=[EnvironmentState(name="missing_state", state="x")],
            expected_environment_state=[EnvironmentState(name="other_state", state="y")],
        )
        with pytest.raises(ValueError, match="not found in expected_environment_state"):
            evaluator.evaluate(data)

    def test_to_dict(self):
        evaluator = StateEquals(name="test_results", value={"exit_code": 0})
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "StateEquals"
        assert d["name"] == "test_results"
        assert d["value"] == {"exit_code": 0}

    def test_to_dict_no_value(self):
        evaluator = StateEquals(name="test_results")
        d = evaluator.to_dict()
        assert d["evaluator_type"] == "StateEquals"
        assert d["name"] == "test_results"
        assert "value" not in d
