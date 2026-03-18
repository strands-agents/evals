"""Plugin that evaluates agent invocations and retries with improvements on failure."""

import logging
from typing import Any, cast

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
        model: Model | str | None = None,
    ):
        """Initialize the evaluation plugin.

        Args:
            evaluators: Evaluators to run against agent output after each invocation.
            max_retries: Maximum number of retry attempts when evaluation fails.
            expected_output: Default expected output for evaluation. Can be overridden per-invocation
                via ``invocation_state``.
            expected_trajectory: Default expected trajectory for evaluation. Can be overridden
                per-invocation via ``invocation_state``.
            model: Model used by the improvement suggestion agent. Accepts a Model instance,
                a model ID string, or None to use the default.
        """
        self._evaluators = evaluators
        self._max_retries = max_retries
        self._expected_output = expected_output
        self._expected_trajectory = expected_trajectory
        self._model = model
        self._agent: Any = None

    def init_agent(self, agent: Any) -> None:
        """Wrap the agent's ``__call__`` to intercept invocations for evaluation and retry.

        Creates a dynamic subclass of the agent's class with a wrapped ``__call__`` that runs
        evaluators after each invocation and retries with an improved system prompt on failure.

        Args:
            agent: The agent instance whose invocations will be evaluated.
        """
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
        """Run the agent, evaluate output, and retry with an improved system prompt on failure.

        Restores the original system prompt and messages after all attempts complete.

        Args:
            agent: The agent instance being invoked.
            original_call: The unwrapped ``__call__`` method.
            prompt: The user prompt passed to the agent.
            **kwargs: Additional keyword arguments forwarded to the agent call.

        Returns:
            The result from the last agent invocation attempt.
        """
        original_system_prompt = agent.system_prompt
        original_messages = list(agent.messages)
        invocation_state = kwargs.get("invocation_state", {})

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
            logger.debug(
                "attempt=<%s>, reasoning=<%s> | applying improved system prompt", attempt + 1, suggestion.reasoning
            )
            agent.system_prompt = suggestion.system_prompt

        agent.system_prompt = original_system_prompt
        return result

    def _build_evaluation_data(self, prompt: Any, result: Any, invocation_state: dict) -> EvaluationData:
        """Assemble evaluation data from the invocation context.

        Args:
            prompt: The user prompt.
            result: The agent's output.
            invocation_state: Per-invocation overrides for expected values.

        Returns:
            An EvaluationData instance ready for evaluator consumption.
        """
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
    ) -> ImprovementSuggestion:
        """Ask an LLM to suggest an improved system prompt based on evaluation failures.

        Args:
            prompt: The original user prompt.
            actual_output: The agent's output as a string.
            outputs: Evaluation outputs from the failed attempt.
            current_system_prompt: The agent's current system prompt.
            expected_output: The expected output, if available.

        Returns:
            An ImprovementSuggestion containing the reasoning and a revised system prompt.
        """
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
        return cast(ImprovementSuggestion, result.structured_output)

    def _run_evaluators(self, evaluation_data: EvaluationData) -> list[EvaluationOutput]:
        """Run all evaluators against the given evaluation data.

        Exceptions raised by individual evaluators are caught, logged, and recorded as failures
        so that a single broken evaluator does not prevent the others from running.

        Args:
            evaluation_data: The data to evaluate.

        Returns:
            A list of evaluation outputs from all evaluators.
        """
        all_outputs: list[EvaluationOutput] = []
        for evaluator in self._evaluators:
            try:
                outputs = evaluator.evaluate(evaluation_data)
                all_outputs.extend(outputs)
            except Exception:
                logger.exception("evaluator=<%s> | evaluator raised an exception", type(evaluator).__name__)
                all_outputs.append(EvaluationOutput(score=0.0, test_pass=False, reason="evaluator raised an exception"))
        return all_outputs
