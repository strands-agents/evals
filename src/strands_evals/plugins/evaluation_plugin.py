"""Plugin that evaluates agent invocations and retries with improvements on failure."""

import logging
from typing import Any, Union, cast

from pydantic import BaseModel
from strands import Agent
from strands.models import Model
from strands.plugins.plugin import Plugin

from strands_evals.evaluators.evaluator import Evaluator
from strands_evals.plugins.prompt_templates.improvement_suggestion import (
    IMPROVEMENT_SYSTEM_PROMPT,
    compose_improvement_prompt,
)
from strands_evals.types.evaluation import EvaluationData, EvaluationOutput

logger = logging.getLogger(__name__)


class ImprovementSuggestion(BaseModel):
    """Structured output from the improvement suggestion LLM."""

    reasoning: str
    system_prompt: str


class EvaluationPlugin(Plugin):
    """Evaluates agent output after each invocation and retries with improved system prompts on failure."""

    @property
    def name(self) -> str:
        return "strands-evals"

    def __init__(
        self,
        evaluators: list[Evaluator],
        max_retries: int = 1,
        expected_output: Any = None,
        expected_trajectory: list[Any] | None = None,
        model: Union[Model, str, None] = None,
    ):
        self._evaluators = evaluators
        self._max_retries = max_retries
        self._expected_output = expected_output
        self._expected_trajectory = expected_trajectory
        self._model = model
        self._agent: Any = None

    def init_agent(self, agent: Any) -> None:
        self._agent = agent
        original_call = agent.__class__.__call__
        plugin = self

        def wrapped_call(self_agent: Any, prompt: Any = None, **kwargs: Any) -> Any:
            return plugin._invoke_with_evaluation(self_agent, original_call, prompt, **kwargs)

        wrapped_class = type(
            agent.__class__.__name__,
            (agent.__class__,),
            {"__call__": wrapped_call},
        )
        agent.__class__ = wrapped_class

    def _invoke_with_evaluation(self, agent: Any, original_call: Any, prompt: Any, **kwargs: Any) -> Any:
        original_system_prompt = agent.system_prompt
        original_messages = list(agent.messages)
        invocation_state = kwargs.get("invocation_state") or {}

        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                agent.messages = list(original_messages)

            result = original_call(agent, prompt, **kwargs)

            evaluation_data = self._build_evaluation_data(prompt, result, invocation_state)
            outputs = self._run_evaluators(evaluation_data)
            all_pass = all(o.test_pass for o in outputs)

            if all_pass or attempt == self._max_retries:
                break

            logger.debug(
                "attempt=<%s>, evaluation_pass=<%s> | evaluation failed, generating improvements",
                attempt + 1,
                all_pass,
            )

            expected_output = evaluation_data.expected_output
            suggestion = self._suggest_improvements(prompt, str(result), outputs, agent.system_prompt, expected_output)
            agent.system_prompt = suggestion

        agent.system_prompt = original_system_prompt
        return result

    def _build_evaluation_data(self, prompt: Any, result: Any, invocation_state: dict) -> EvaluationData:
        expected_output = invocation_state.get("expected_output", self._expected_output)
        expected_trajectory = invocation_state.get("expected_trajectory", self._expected_trajectory)

        return EvaluationData(
            input=prompt,
            actual_output=str(result),
            expected_output=expected_output,
            expected_trajectory=expected_trajectory,
        )

    def _suggest_improvements(
        self,
        prompt: Any,
        actual_output: str,
        outputs: list[EvaluationOutput],
        current_system_prompt: str | None,
        expected_output: Any = None,
    ) -> str:
        failure_reasons = [o.reason for o in outputs if not o.test_pass and o.reason]
        improvement_prompt = compose_improvement_prompt(
            user_prompt=str(prompt),
            actual_output=actual_output,
            failure_reasons=failure_reasons,
            current_system_prompt=current_system_prompt,
            expected_output=str(expected_output) if expected_output is not None else None,
        )
        suggestion_agent = Agent(model=self._model, system_prompt=IMPROVEMENT_SYSTEM_PROMPT, callback_handler=None)
        result = suggestion_agent(improvement_prompt, structured_output_model=ImprovementSuggestion)
        suggestion = cast(ImprovementSuggestion, result.structured_output)
        return suggestion.system_prompt

    def _run_evaluators(self, evaluation_data: EvaluationData) -> list[EvaluationOutput]:
        all_outputs: list[EvaluationOutput] = []
        for evaluator in self._evaluators:
            try:
                outputs = evaluator.evaluate(evaluation_data)
                all_outputs.extend(outputs)
            except Exception:
                logger.exception("evaluator=<%s> | evaluator raised an exception", type(evaluator).__name__)
                all_outputs.append(EvaluationOutput(score=0.0, test_pass=False, reason="evaluator raised an exception"))
        return all_outputs
