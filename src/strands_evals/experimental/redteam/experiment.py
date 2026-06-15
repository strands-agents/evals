"""Red team experiment."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from strands import Agent
from strands.models.model import Model

from ...case import Case
from ...evaluation_data_store import EvaluationDataStore
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...types import InputT, OutputT
from .case import RedTeamCase
from .evaluators import AttackSuccessEvaluator
from .report import RedTeamReport
from .strategies import AttackStrategy
from .strategies.target_session import TargetSession
from .task import _build_attacker_task
from .utils import _put_model_field


class RedTeamExperiment(Experiment[InputT, OutputT]):
    """Experiment specialized for red team evaluation.

    Holds the attack strategies and runs the case × strategy cross-product:
    every case is attacked by every strategy. When ``agent`` is supplied,
    ``run_evaluations()`` builds a default multi-turn attacker task internally;
    pass an explicit ``task`` to customize. Returns a :class:`RedTeamReport`.

    Example:
        ```python
        from strands_evals.experimental.redteam import (
            AdversarialCaseGenerator, RedTeamExperiment, CrescendoStrategy,
        )

        cases = AdversarialCaseGenerator(model=model).generate_cases(agent=agent)
        experiment = RedTeamExperiment(
            cases=cases, agent=agent, attack_strategies=[CrescendoStrategy(max_turns=10)]
        )
        report = experiment.run_evaluations()
        report.display()
        ```
    """

    def __init__(
        self,
        cases: list[Case[InputT, OutputT]] | None = None,
        *,
        agent: Agent | TargetSession | None = None,
        attack_strategies: list[AttackStrategy] | None = None,
        evaluators: list[Evaluator[InputT, OutputT]] | None = None,
        model: Model | str | None = None,
    ):
        super().__init__(
            cases=cases,
            evaluators=evaluators or [AttackSuccessEvaluator(model=model)],
        )
        self._agent = agent
        self._attack_strategies = attack_strategies or []
        self._by_label = self._build_by_label(self._attack_strategies)
        self._model = model
        # case name -> strategy run metadata (turns_used, backtracks, ...); the
        # default task records into this and we join it onto the report, since the
        # base Experiment doesn't carry task-returned metadata into EvaluationData.
        self._run_meta: dict[str, dict[str, Any]] = {}

    @property
    def agent(self) -> Agent | TargetSession | None:
        """The target the default attacker task talks to.

        Persisted experiments do not include the agent (live `Agent` /
        `MultiAgentBase` / `TargetSession` instances are not JSON-serializable),
        so the canonical load path is `from_file` followed by `exp.agent = ...`
        before `run_evaluations`. Setting to `None` is allowed and restores the
        "must supply explicit task" error path.
        """
        return self._agent

    @agent.setter
    def agent(self, value: Agent | TargetSession | None) -> None:
        self._agent = value

    @property
    def attack_strategies(self) -> list[AttackStrategy]:
        """The configured attack strategies (read-only view).

        The experiment's `_by_label` index is built from this list at
        construction; mutating the returned list will not re-index. Reload
        with `from_dict` if you need to change strategies after construction.
        """
        return list(self._attack_strategies)

    @staticmethod
    def _build_by_label(strategies: list[AttackStrategy]) -> dict[str, AttackStrategy]:
        by_label: dict[str, AttackStrategy] = {}
        for strategy in strategies:
            if strategy.label in by_label:
                raise ValueError(
                    f"Duplicate strategy label {strategy.label!r}. "
                    "Pass distinct label= values to compare same-type strategies."
                )
            by_label[strategy.label] = strategy
        return by_label

    def _expand_cross_product(self) -> list[Case[InputT, OutputT]]:
        """Expand held cases into the case × strategy cross-product.

        Each work item is a copy of the case tagged with one strategy's label in
        ``metadata["strategy"]`` and a unique name ``"{case}__{label}"`` (so the
        evaluation_data_store cache keys stay unique). The base worker still sees
        a plain case queue.

        Pure: returns a new list and never mutates ``self._cases``, so reusing
        the experiment across runs does not re-expand an already-expanded list.

        Returns:
            The expanded case list, or the held cases unchanged when no
            strategies were supplied.
        """
        if not self._attack_strategies:
            return list(self._cases)
        expanded: list[Case[InputT, OutputT]] = []
        for case in self._cases:
            for strategy in self._attack_strategies:
                item = case.model_copy(deep=True)
                item.name = f"{case.name}__{strategy.label}"
                metadata = dict(item.metadata or {})
                metadata["strategy"] = strategy.label
                item.metadata = metadata
                expanded.append(item)
        return expanded

    def run_evaluations(  # type: ignore[override]
        self,
        task: Callable[[Case[InputT, OutputT]], Any] | None = None,
        evaluation_data_store: EvaluationDataStore | None = None,
    ) -> RedTeamReport:
        if inspect.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")
        return asyncio.run(self.run_evaluations_async(task, max_workers=1, evaluation_data_store=evaluation_data_store))

    async def run_evaluations_async(  # type: ignore[override]
        self,
        task: Callable | None = None,
        max_workers: int = 1,
        evaluation_data_store: EvaluationDataStore | None = None,
    ) -> RedTeamReport:
        # Parallel workers would interleave on the shared target Agent and on each
        # strategy instance's per-case state, so red team runs are strictly sequential.
        if max_workers != 1:
            raise ValueError("RedTeamExperiment requires max_workers=1 (shared target Agent and strategy state).")
        self._run_meta.clear()
        if task is None:
            task = self._default_task()
        # Swap self._cases to the expanded cross-product only for the duration of the
        # base run, then restore it, so the experiment can be re-run without re-expanding
        # (validated by test_run_evaluations_twice_is_idempotent). This temporary mutation
        # is safe ONLY because max_workers=1 is enforced above: no other worker or
        # event-loop task reads self._cases concurrently. Passing the expanded list down
        # without mutating self is cleaner and is part of the standalone (composition)
        # refactor tracked in the fast-follow plan; the base API doesn't accept an
        # explicit case list today.
        original_cases = self._cases
        self._cases = self._expand_cross_product()
        try:
            # base now returns a single flattened EvaluationReport (#241), each case row
            # tagged with its evaluator; RedTeamReport wraps that and joins run_meta.
            report = await super().run_evaluations_async(
                task, max_workers=max_workers, evaluation_data_store=evaluation_data_store
            )
        finally:
            self._cases = original_cases
        return RedTeamReport.from_evaluation_report(report, run_meta=self._run_meta)

    def _default_task(self) -> Callable[[Case[InputT, OutputT]], Any]:
        if self._agent is None:
            raise ValueError(
                "RedTeamExperiment requires either `agent` at construction "
                "or an explicit `task` argument to run_evaluations()."
            )
        return cast(
            Callable[[Case[InputT, OutputT]], Any],
            _build_attacker_task(
                self._agent,
                self._by_label,
                model=self._model,
                run_meta=self._run_meta,
            ),
        )

    def to_dict(self) -> dict:  # type: ignore[override]
        """Serialize the experiment without the live target agent.

        Adds `attack_strategies` and `model` on top of the base shape. The
        target `agent` is intentionally omitted — `Agent` / `MultiAgentBase` /
        `TargetSession` instances are not JSON-serializable, so callers must
        re-attach via the `agent` setter (or pass an explicit `task`) after
        loading. `_run_meta` is per-run state and is not persisted.
        """
        out = super().to_dict()
        out["attack_strategies"] = [strategy.to_dict() for strategy in self._attack_strategies]
        # Coerce via the strategy helper for consistency with how strategies serialize their own model field.
        _put_model_field(out, self._model)
        return out

    @classmethod
    def from_file(  # type: ignore[override]
        cls,
        path: str,
        custom_evaluators: list[type[Evaluator]] | None = None,
        custom_strategies: list[type[AttackStrategy]] | None = None,
    ):
        """Load a RedTeamExperiment from a JSON file.

        Same contract as :meth:`from_dict`: the loaded experiment has no
        `agent` attached, so set `exp.agent = ...` before `run_evaluations`.
        """
        file_path = Path(path)
        if file_path.suffix != ".json":
            raise ValueError(
                f"Only .json format is supported. Got file: {path}. Please provide a path with .json extension."
            )
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(
            data,
            custom_evaluators=custom_evaluators,
            custom_strategies=custom_strategies,
        )

    @classmethod
    def from_dict(  # type: ignore[override]
        cls,
        data: dict,
        custom_evaluators: list[type[Evaluator]] | None = None,
        custom_strategies: list[type[AttackStrategy]] | None = None,
    ):
        """Reconstruct a RedTeamExperiment from its serialized form.

        The loaded experiment has no `agent` attached: set `exp.agent = ...`
        before `run_evaluations`, or pass an explicit `task=`. Cases are
        validated as `RedTeamCase` so the typed `config` survives reload.
        `AttackSuccessEvaluator` is registered automatically; pass
        `custom_evaluators` for user-defined evaluator subclasses and
        `custom_strategies` for user-defined strategy subclasses.
        """
        merged_evaluators: list[type[Evaluator]] = [AttackSuccessEvaluator, *(custom_evaluators or [])]
        # Reuse the base evaluator-resolving path (registry + custom merge), but skip
        # its case parser: we want RedTeamCase, not Case.
        payload = dict(data)
        case_dicts = payload.pop("cases", [])
        strategy_dicts = payload.pop("attack_strategies", [])
        model = payload.pop("model", None)
        # Drive the base only for evaluator resolution by giving it an empty case list.
        base_for_evaluators = super().from_dict(
            {"cases": [], "evaluators": payload.get("evaluators", [])},
            custom_evaluators=merged_evaluators,
        )
        cases: list[Case[InputT, OutputT]] = [RedTeamCase.model_validate(case_data) for case_data in case_dicts]
        strategies = [
            AttackStrategy.from_dict(strategy_data, custom_strategies=custom_strategies)
            for strategy_data in strategy_dicts
        ]
        return cls(
            cases=cases,
            attack_strategies=strategies,
            evaluators=base_for_evaluators.evaluators,
            model=model,
        )
