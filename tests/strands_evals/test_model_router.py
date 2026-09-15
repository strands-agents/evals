from unittest.mock import Mock, patch

import pytest

from strands_evals import Case, Experiment, ModelRouter, RoutingRule
from strands_evals.evaluators import OutputEvaluator, TrajectoryEvaluator
from strands_evals.evaluators.deterministic import Contains
from strands_evals.types import EvaluationData, EvaluationOutput

FAST_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
STRONG_MODEL = "us.anthropic.claude-opus-4-1-20250805-v1:0"


@pytest.fixture
def evaluation_data():
    return EvaluationData(input="What is 2+2?", actual_output="4", expected_output="4", name="math_test")


class TestRoutingRule:
    def test_matches_by_class_name_string(self, evaluation_data):
        rule = RoutingRule(evaluator_types=["OutputEvaluator"], model=FAST_MODEL)
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert rule.matches(evaluator, evaluation_data) is True

    def test_matches_by_class_object(self, evaluation_data):
        rule = RoutingRule(evaluator_types=[OutputEvaluator], model=FAST_MODEL)
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert rule.matches(evaluator, evaluation_data) is True

    def test_does_not_match_other_evaluator_types(self, evaluation_data):
        rule = RoutingRule(evaluator_types=[TrajectoryEvaluator], model=FAST_MODEL)
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert rule.matches(evaluator, evaluation_data) is False

    def test_condition_gates_matching(self, evaluation_data):
        rule = RoutingRule(
            evaluator_types=[OutputEvaluator],
            model=STRONG_MODEL,
            condition=lambda case: len(str(case.actual_output)) > 100,
        )
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert rule.matches(evaluator, evaluation_data) is False

        long_output = EvaluationData(input="q", actual_output="x" * 200)
        assert rule.matches(evaluator, long_output) is True

    def test_condition_exception_skips_rule(self, evaluation_data):
        def broken_condition(case):
            raise RuntimeError("boom")

        rule = RoutingRule(evaluator_types=[OutputEvaluator], model=FAST_MODEL, condition=broken_condition)
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert rule.matches(evaluator, evaluation_data) is False


class TestModelRouter:
    def test_first_matching_rule_wins(self, evaluation_data):
        router = ModelRouter(
            rules=[
                RoutingRule(evaluator_types=[OutputEvaluator], model=FAST_MODEL),
                RoutingRule(evaluator_types=[OutputEvaluator], model=STRONG_MODEL),
            ]
        )
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert router.select_model(evaluator, evaluation_data) == FAST_MODEL

    def test_default_model_when_no_rule_matches(self, evaluation_data):
        router = ModelRouter(
            rules=[RoutingRule(evaluator_types=[TrajectoryEvaluator], model=FAST_MODEL)],
            default_model=STRONG_MODEL,
        )
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert router.select_model(evaluator, evaluation_data) == STRONG_MODEL

    def test_no_rule_and_no_default_returns_none(self, evaluation_data):
        router = ModelRouter(rules=[RoutingRule(evaluator_types=[TrajectoryEvaluator], model=FAST_MODEL)])
        evaluator = OutputEvaluator(rubric="Test rubric")

        assert router.select_model(evaluator, evaluation_data) is None

    def test_route_returns_copy_with_routed_model(self, evaluation_data):
        router = ModelRouter(rules=[RoutingRule(evaluator_types=[OutputEvaluator], model=FAST_MODEL)])
        evaluator = OutputEvaluator(rubric="Test rubric")

        routed = router.route(evaluator, evaluation_data)

        assert routed is not evaluator
        assert routed.model == FAST_MODEL
        # Original instance is never mutated
        assert evaluator.model is None

    def test_route_preserves_explicit_model(self, evaluation_data):
        """An evaluator constructed with an explicit model must never be re-routed."""
        router = ModelRouter(rules=[RoutingRule(evaluator_types=[OutputEvaluator], model=FAST_MODEL)])
        evaluator = OutputEvaluator(rubric="Test rubric", model=STRONG_MODEL)

        routed = router.route(evaluator, evaluation_data)

        assert routed is evaluator
        assert routed.model == STRONG_MODEL

    def test_route_leaves_evaluator_without_model_attribute_unchanged(self, evaluation_data):
        """Deterministic evaluators have no model attribute and must pass through."""
        router = ModelRouter(
            rules=[RoutingRule(evaluator_types=["Contains"], model=FAST_MODEL)], default_model=FAST_MODEL
        )
        evaluator = Contains(value="4")

        routed = router.route(evaluator, evaluation_data)

        assert routed is evaluator

    def test_route_without_match_returns_original(self, evaluation_data):
        router = ModelRouter(rules=[RoutingRule(evaluator_types=[TrajectoryEvaluator], model=FAST_MODEL)])
        evaluator = OutputEvaluator(rubric="Test rubric")

        routed = router.route(evaluator, evaluation_data)

        assert routed is evaluator
        assert routed.model is None


class TestExperimentIntegration:
    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_experiment_routes_evaluator_model(self, mock_agent_class):
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(score=1.0, test_pass=True, reason="ok")
        mock_agent.return_value = mock_result

        async def mock_invoke_async(*args, **kwargs):
            return mock_result

        mock_agent.invoke_async = mock_invoke_async
        mock_agent_class.return_value = mock_agent

        router = ModelRouter(rules=[RoutingRule(evaluator_types=[OutputEvaluator], model=FAST_MODEL)])
        evaluator = OutputEvaluator(rubric="Output must be correct.")
        experiment = Experiment(
            cases=[Case(name="math", input="What is 2+2?", expected_output="4")],
            evaluators=[evaluator],
            model_router=router,
        )

        report = experiment.run_evaluations(lambda case: "4")

        # The judge agent was constructed with the routed model
        assert mock_agent_class.call_args[1]["model"] == FAST_MODEL
        assert report.test_passes == [True]
        # The user's evaluator instance is untouched
        assert evaluator.model is None

    @patch("strands_evals.evaluators.output_evaluator.Agent")
    def test_experiment_without_router_keeps_default_model(self, mock_agent_class):
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.structured_output = EvaluationOutput(score=1.0, test_pass=True, reason="ok")
        mock_agent.return_value = mock_result

        async def mock_invoke_async(*args, **kwargs):
            return mock_result

        mock_agent.invoke_async = mock_invoke_async
        mock_agent_class.return_value = mock_agent

        experiment = Experiment(
            cases=[Case(name="math", input="What is 2+2?", expected_output="4")],
            evaluators=[OutputEvaluator(rubric="Output must be correct.")],
        )

        experiment.run_evaluations(lambda case: "4")

        assert mock_agent_class.call_args[1]["model"] is None
