"""Model routing for evaluators.

Lets an Experiment match model tier to evaluation complexity: cheap/fast models
for structural or rubric-style checks, stronger models for nuanced judgment.

Routing is opt-in and advisory:
- An evaluator constructed with an explicit ``model=`` always keeps that model.
- Evaluators whose ``model`` is None (the default) are routed through the
  router's rules; the first matching rule wins.
- If no rule matches, the router's ``default_model`` (if any) is used;
  otherwise the evaluator keeps its default model resolution.

Example::

    from strands_evals import Experiment, ModelRouter, RoutingRule

    router = ModelRouter(rules=[
        # Structural checks -> fast model
        RoutingRule(
            evaluator_types=["ToolSelectionAccuracyEvaluator", "ToolParameterAccuracyEvaluator"],
            model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
        # Complex judgment on long traces -> strongest model
        RoutingRule(
            evaluator_types=["GoalSuccessRateEvaluator"],
            model="us.anthropic.claude-opus-4-1-20250805-v1:0",
            condition=lambda case: len(str(case.actual_trajectory or "")) > 10_000,
        ),
    ])

    experiment = Experiment(cases=cases, evaluators=evaluators, model_router=router)
"""

import copy
import logging
from typing import TYPE_CHECKING, Callable

from strands.models.model import Model

if TYPE_CHECKING:
    from .evaluators.evaluator import Evaluator
    from .types.evaluation import EvaluationData

logger = logging.getLogger(__name__)


class RoutingRule:
    """A rule mapping evaluator types (and optionally case properties) to a model.

    Attributes:
        evaluator_types: Evaluator classes or class names this rule applies to.
        model: The model (Model instance or Bedrock model-id string) to use when
            this rule matches.
        condition: Optional predicate over the evaluation data. When provided,
            the rule only matches cases for which it returns True. This enables
            complexity-based routing (e.g., trace length, span count).
    """

    def __init__(
        self,
        evaluator_types: list[type | str],
        model: Model | str,
        condition: Callable[["EvaluationData"], bool] | None = None,
    ):
        self._type_names = {t if isinstance(t, str) else t.__name__ for t in evaluator_types}
        self.model = model
        self.condition = condition

    def matches(self, evaluator: "Evaluator", evaluation_data: "EvaluationData") -> bool:
        """Check whether this rule applies to the given evaluator and case.

        Args:
            evaluator: The evaluator about to run.
            evaluation_data: The evaluation context for the current case.

        Returns:
            True if the evaluator's type is covered by this rule and the
            condition (if any) holds for the case.
        """
        if evaluator.get_type_name() not in self._type_names:
            return False
        if self.condition is not None:
            try:
                return bool(self.condition(evaluation_data))
            except Exception as e:
                logger.warning(
                    "rule_types=<%s>, error=<%s> | routing condition raised, skipping rule", self._type_names, e
                )
                return False
        return True


class ModelRouter:
    """Selects a model for each (evaluator, case) pair using an ordered rule list.

    Attributes:
        rules: Ordered list of RoutingRule; the first matching rule wins.
        default_model: Model used when no rule matches. None means "leave the
            evaluator's own model resolution untouched".
    """

    def __init__(self, rules: list[RoutingRule] | None = None, default_model: Model | str | None = None):
        self.rules = rules or []
        self.default_model = default_model

    def select_model(self, evaluator: "Evaluator", evaluation_data: "EvaluationData") -> Model | str | None:
        """Select a model for an evaluator run, or None to keep the evaluator's default.

        Args:
            evaluator: The evaluator about to run.
            evaluation_data: The evaluation context for the current case.

        Returns:
            The routed model, the router default, or None when routing does not apply.
        """
        for rule in self.rules:
            if rule.matches(evaluator, evaluation_data):
                return rule.model
        return self.default_model

    def route(self, evaluator: "Evaluator", evaluation_data: "EvaluationData") -> "Evaluator":
        """Return the evaluator to run, applying model routing when appropriate.

        An evaluator with an explicit model (its ``model`` attribute is set) is
        returned unchanged — user configuration always wins. Evaluators without
        a ``model`` attribute (e.g., deterministic evaluators) are also returned
        unchanged. Otherwise a shallow copy with the routed model is returned so
        the shared evaluator instance is never mutated across concurrent workers.

        Args:
            evaluator: The evaluator about to run.
            evaluation_data: The evaluation context for the current case.

        Returns:
            Either the original evaluator or a shallow copy carrying the routed model.
        """
        if getattr(evaluator, "model", None) is not None or not hasattr(evaluator, "model"):
            return evaluator

        model = self.select_model(evaluator, evaluation_data)
        if model is None:
            return evaluator

        routed = copy.copy(evaluator)
        routed.model = model
        logger.debug(
            "evaluator=<%s>, model=<%s> | routed evaluator to model",
            evaluator.get_name(),
            model if isinstance(model, str) else type(model).__name__,
        )
        return routed
