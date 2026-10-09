"""Red team experiment."""

from __future__ import annotations

import asyncio
import inspect
import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from strands import Agent
from strands.models.model import Model
from strands.multiagent.base import MultiAgentBase

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

_AGENT_DEPRECATION = (
    "Attaching `{name}` to RedTeamExperiment is deprecated and will be removed in a future release. "
    "Pass `agent_factory=` to `run_evaluations()` / `run_evaluations_async()` instead."
)


class RedTeamExperiment(Experiment[InputT, OutputT, RedTeamReport]):
    """Experiment that runs the case x strategy cross-product and returns a `RedTeamReport`.

    Run it either with your own `task` or with an `agent_factory`, which the built-in attacker task
    calls once per case. The factory can return a single `strands.Agent`, a multi-agent system
    (`strands.multiagent.Graph`, `strands.multiagent.Swarm`, or any
    `strands.multiagent.base.MultiAgentBase`), or a custom `TargetSession` Protocol implementer. The
    experiment holds no live target, so it round-trips through `to_file` / `from_file` as is.

    Example:
        ```python
        def agent_factory() -> Agent:
            return Agent(system_prompt="You are a helpful customer-support assistant.")

        cases = AdversarialCaseGenerator(model=model).generate_cases(agent=agent_factory())
        experiment = RedTeamExperiment(cases=cases, attack_strategies=[CrescendoStrategy(max_turns=10)])
        report = asyncio.run(experiment.run_evaluations_async(agent_factory=agent_factory))
        report.display()
        ```
    """

    report_cls = RedTeamReport

    def __init__(
        self,
        cases: list[Case[InputT, OutputT]] | None = None,
        *,
        agent: Agent | MultiAgentBase | TargetSession | None = None,
        agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None = None,
        attack_strategies: list[AttackStrategy] | None = None,
        evaluators: list[Evaluator[InputT, OutputT]] | None = None,
        model: Model | str | None = None,
    ):
        """Initialize the experiment.

        Args:
            cases: The `RedTeamCase`s to attack. Plain `Case` inputs are deprecated.
            agent: Deprecated. A shared target for sequential runs; pass `agent_factory=` to the run
                methods instead.
            agent_factory: Deprecated. Pass it to the run methods instead.
            attack_strategies: Strategies to run against every case. Labels must be distinct.
            evaluators: Evaluators for each attack. Defaults to `[AttackSuccessEvaluator(model=model)]`.
            model: Model for the default judge and for strategy-internal LLM calls.
        """
        super().__init__(
            cases=cases,
            evaluators=evaluators or [AttackSuccessEvaluator(model=model)],
        )
        if any(not isinstance(case, RedTeamCase) for case in self._cases):
            warnings.warn(
                "Passing plain `Case` objects to RedTeamExperiment is deprecated and will be rejected in a "
                "future release. Use `RedTeamCase`.",
                DeprecationWarning,
                stacklevel=2,
            )
        for name, value in (("agent", agent), ("agent_factory", agent_factory)):
            if value is not None:
                warnings.warn(_AGENT_DEPRECATION.format(name=name), DeprecationWarning, stacklevel=2)
        self._agent = agent
        self._agent_factory = agent_factory
        self._attack_strategies = attack_strategies or []
        self._by_label = self._build_by_label(self._attack_strategies)
        self._model = model

    @property
    def agent(self) -> Agent | MultiAgentBase | TargetSession | None:
        """Deprecated. The shared target the built-in attacker task talks to in sequential runs.

        Pass `agent_factory=` to the run methods instead. Not persisted by `to_dict`.
        """
        return self._agent

    @agent.setter
    def agent(self, value: Agent | MultiAgentBase | TargetSession | None) -> None:
        warnings.warn(_AGENT_DEPRECATION.format(name="agent"), DeprecationWarning, stacklevel=2)
        self._agent = value

    @property
    def agent_factory(self) -> Callable[[], Agent | MultiAgentBase | TargetSession] | None:
        """Deprecated. The fallback factory used when the run methods get no `agent_factory`.

        Pass `agent_factory=` to the run methods instead. Not persisted by `to_dict`.
        """
        return self._agent_factory

    @agent_factory.setter
    def agent_factory(self, value: Callable[[], Agent | MultiAgentBase | TargetSession] | None) -> None:
        warnings.warn(_AGENT_DEPRECATION.format(name="agent_factory"), DeprecationWarning, stacklevel=2)
        self._agent_factory = value

    @property
    def attack_strategies(self) -> list[AttackStrategy]:
        """The configured attack strategies (read-only copy)."""
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
        """Return a new list of (case x strategy) work items; does not mutate `self._cases`.

        Each item is a copy of the case named `"{case}__{label}"` and tagged with `metadata["strategy"] = label`
        so cache keys stay unique.
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

    def run_evaluations(
        self,
        task: Callable[[Case[InputT, OutputT]], Any] | None = None,
        evaluation_data_store: EvaluationDataStore | None = None,
        *,
        agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None = None,
    ) -> RedTeamReport:
        """Run the case-strategy cross-product sequentially.

        Args:
            task: Your own task, called once per case x strategy variant. Mutually exclusive with
                `agent_factory`.
            evaluation_data_store: Optional store for loading/saving evaluation data.
            agent_factory: Zero-arg callable returning a fresh target; the built-in attacker task calls
                it once per case.
        """
        if inspect.iscoroutinefunction(task):
            raise ValueError("Async task is not supported. Please use run_evaluations_async instead.")
        return asyncio.run(
            self.run_evaluations_async(
                task, max_workers=1, evaluation_data_store=evaluation_data_store, agent_factory=agent_factory
            )
        )

    async def run_evaluations_async(
        self,
        task: Callable | None = None,
        max_workers: int = 5,
        evaluation_data_store: EvaluationDataStore | None = None,
        *,
        agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None = None,
    ) -> RedTeamReport:
        """Run the case-strategy cross-product, in parallel when `max_workers > 1`.

        Defaults to `max_workers=5`: callers reaching for the async entry point are opting into
        concurrency, and 5 is the largest value safe across most provider tiers without user-side
        rate-limit tuning. Bump higher for fast targets / generous TPM budgets; drop to 1 for
        deterministic ordering or to debug a single case.

        Without a `task`, the built-in attacker task calls `agent_factory` once per case, so
        concurrent `invoke()` calls never interleave on shared agent state. A user-supplied `task`
        callable is treated as already parallel-safe.

        Strategy instances are shared across all concurrent cases; per-case state must live in
        `run_attack` locals, not on `self`. See `AttackStrategy.run_attack` for the contract.

        Args:
            task: Your own task, called once per case x strategy variant. Mutually exclusive with
                `agent_factory`.
            max_workers: Maximum number of cases run concurrently.
            evaluation_data_store: Optional store for loading/saving evaluation data.
            agent_factory: Zero-arg callable returning a fresh target; the built-in attacker task calls
                it once per case.
        """
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        if task is not None and agent_factory is not None:
            raise ValueError("Pass either `task` or `agent_factory`, not both.")
        if task is None:
            task = self._default_task(agent_factory, parallel=max_workers > 1)
        # Swap _cases for the expanded cross-product. Each case has a unique name (case x strategy
        # label), so parallel workers never collide on the same key in the base runner's results
        # buffer.
        original_cases = self._cases
        self._cases = self._expand_cross_product()
        try:
            report = await super().run_evaluations_async(
                task, max_workers=max_workers, evaluation_data_store=evaluation_data_store
            )
        finally:
            self._cases = original_cases
        # Rebuild rather than return the base report: its overall_score counts errored and
        # NOT_APPLICABLE rows as 0.0 judgments, which the red team view excludes.
        return RedTeamReport.from_evaluation_report(report)

    def _default_task(
        self,
        agent_factory: Callable[[], Agent | MultiAgentBase | TargetSession] | None,
        *,
        parallel: bool = False,
    ) -> Callable[[Case[InputT, OutputT]], Any]:
        # The constructor's (deprecated) factory and agent are fallbacks for the run-time factory.
        factory = agent_factory or self._agent_factory
        if factory is None and self._agent is None:
            raise ValueError("RedTeamExperiment needs a `task` or an `agent_factory` passed to run_evaluations().")
        return cast(
            Callable[[Case[InputT, OutputT]], Any],
            _build_attacker_task(
                self._agent,
                self._by_label,
                agent_factory=factory,
                model=self._model,
                parallel=parallel,
            ),
        )

    def to_dict(self) -> dict:  # type: ignore[override]
        """Serialize the experiment, omitting any live target and per-run state."""
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
        """Load a RedTeamExperiment from JSON.

        The target is runtime-only and not serialized: pass `agent_factory=` (or a `task`) to the run
        methods of the loaded experiment.
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

        Cases are validated as `RedTeamCase`. Pass `custom_evaluators` / `custom_strategies` to register
        user-defined subclasses.
        """
        merged_evaluators: list[type[Evaluator]] = [AttackSuccessEvaluator, *(custom_evaluators or [])]
        # Reuse the base evaluator-resolving path but skip its case parser; we want RedTeamCase.
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
