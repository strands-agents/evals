"""Red team experiment."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast

from strands import Agent
from strands.models.model import Model

from ...case import Case
from ...evaluation_data_store import EvaluationDataStore
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...types import InputT, OutputT
from .evaluators import AttackSuccessEvaluator
from .report import RedTeamReport
from .task import _build_attacker_task


class RedTeamExperiment(Experiment[InputT, OutputT]):
    """Experiment specialized for red team evaluation.

    When ``target`` is supplied, ``run_evaluations()`` builds a default
    multi-turn attacker task internally; pass an explicit ``task`` to
    customize. Returns a :class:`RedTeamReport`.

    Example:
        ```python
        from strands_evals.experimental.redteam import (
            AdversarialCaseGenerator, RedTeamExperiment,
        )

        cases = AdversarialCaseGenerator(model=model).generate_cases(target=agent)
        experiment = RedTeamExperiment(cases=cases, target=agent, max_turns=10)
        report = experiment.run_evaluations()
        report.display()
        ```
    """

    def __init__(
        self,
        cases: list[Case[InputT, OutputT]] | None = None,
        *,
        target: Agent | Callable[[str], Any] | None = None,
        evaluators: list[Evaluator[InputT, OutputT]] | None = None,
        max_turns: int = 10,
        model: Model | str | None = None,
    ):
        super().__init__(
            cases=cases,
            evaluators=evaluators or [AttackSuccessEvaluator(model=model)],
        )
        self._target = target
        self._max_turns = max_turns
        self._model = model

    def run_evaluations(  # type: ignore[override]
        self,
        task: Callable[[Case[InputT, OutputT]], Any] | None = None,
        evaluation_data_store: EvaluationDataStore | None = None,
    ) -> RedTeamReport:
        task = task or self._default_task()
        if inspect.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")
        return asyncio.run(self.run_evaluations_async(task, max_workers=1, evaluation_data_store=evaluation_data_store))

    async def run_evaluations_async(  # type: ignore[override]
        self,
        task: Callable | None = None,
        max_workers: int = 1,
        evaluation_data_store: EvaluationDataStore | None = None,
    ) -> RedTeamReport:
        # max_workers=1: parallel runs would interleave on the shared target Agent.
        task = task or self._default_task()
        report = await super().run_evaluations_async(
            task, max_workers=max_workers, evaluation_data_store=evaluation_data_store
        )
        return RedTeamReport.from_evaluation_report(report)

    def _default_task(self) -> Callable[[Case[InputT, OutputT]], Any]:
        if self._target is None:
            raise ValueError(
                "RedTeamExperiment requires either `target` at construction "
                "or an explicit `task` argument to run_evaluations()."
            )
        return cast(
            Callable[[Case[InputT, OutputT]], Any],
            _build_attacker_task(
                target=self._target,
                max_turns=self._max_turns,
                model=self._model,
            ),
        )
