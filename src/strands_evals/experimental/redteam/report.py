"""Red team report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console

from ...types.evaluation_report import EvaluationReport

_console = Console()


@dataclass
class AttackResult:
    """One attack case with scores from every evaluator that ran on it."""

    case_name: str
    risk_category: str
    strategy: str
    severity: str
    objective: str = ""
    turns_used: int | None = None
    backtracks: int | None = None
    conversation: list[dict] = field(default_factory=list)
    pruned_branches: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    passes: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def score(self) -> float:
        # Worst-case across evaluators: for red teaming the strongest attack signal
        # (the highest score = most successful breach) is what matters, so aggregate
        # with max(), not min().
        return max(self.scores.values()) if self.scores else 0.0

    @property
    def passed(self) -> bool:
        return all(self.passes.values()) if self.passes else True

    @property
    def reason(self) -> str:
        return " | ".join(f"[{k}] {v}" for k, v in self.reasons.items() if v)


@dataclass
class GroupedSummary:
    """Aggregated summary for a group of attack results."""

    group_name: str
    count: int
    avg_score: float
    pass_rate: float


class RedTeamReport(EvaluationReport):
    """Case-centric report for red team evaluation.

    Note:
        `trajectory` holds raw tool I/O — sanitize before sharing if
        target tools return sensitive data.
    """

    @classmethod
    def from_evaluation_report(cls, report: EvaluationReport, run_meta: dict[str, dict] | None = None) -> RedTeamReport:
        """Wrap a flattened evaluation report as a case-centric red team report.

        The base `Experiment.run_evaluations_async` already returns a single flattened
        report (#241) and tags each case row with its `evaluator` (regardless of evaluator
        count). We reuse that shape directly.

        Args:
            report: The single flattened EvaluationReport from the base experiment.
            run_meta: Optional per-case strategy run metadata (turns_used, backtracks, ...)
                keyed by case name, merged onto each case's metadata so it surfaces in the
                report. Supplied by ``RedTeamExperiment`` because the base ``Experiment``
                does not carry task-returned metadata into the ``EvaluationData``.
        """
        run_meta = run_meta or {}
        n = len(report.cases)
        if not (len(report.scores) == n and len(report.test_passes) == n and len(report.reasons) == n):
            raise ValueError("EvaluationReport: cases/scores/passes/reasons length mismatch")

        cases = []
        for case_data in report.cases:
            case_meta = case_data.get("metadata") or {}
            merged_metadata = {**case_meta, **run_meta.get(case_data.get("name", ""), {})}
            evaluator = case_data.get("evaluator", "evaluator")
            cases.append({**case_data, "evaluator": evaluator, "metadata": merged_metadata})

        return cls(
            overall_score=report.overall_score,
            scores=list(report.scores),
            cases=cases,
            test_passes=list(report.test_passes),
            reasons=list(report.reasons),
            detailed_results=[report.detailed_results[i] if i < len(report.detailed_results) else [] for i in range(n)],
        )

    def attack_results(self) -> list[AttackResult]:
        by_case: dict[str, AttackResult] = {}
        for i, case_data in enumerate(self.cases):
            name = case_data.get("name", f"case_{i}")
            evaluator = case_data.get("evaluator", "evaluator")
            metadata = case_data.get("metadata") or {}
            result = by_case.setdefault(
                name,
                AttackResult(
                    case_name=name,
                    risk_category=metadata.get("risk_category", "unknown"),
                    strategy=metadata.get("strategy", "unknown"),
                    severity=metadata.get("severity", "unknown"),
                    objective=metadata.get("actor_goal", ""),
                    turns_used=metadata.get("turns_used"),
                    backtracks=metadata.get("backtracks"),
                    conversation=case_data.get("actual_output") or [],
                    pruned_branches=metadata.get("pruned_branches") or [],
                ),
            )
            result.scores[evaluator] = self.scores[i]
            result.passes[evaluator] = self.test_passes[i]
            result.reasons[evaluator] = self.reasons[i]
        return list(by_case.values())

    def _group_by(self, key: str) -> dict[str, list[AttackResult]]:
        groups: dict[str, list[AttackResult]] = {}
        for r in self.attack_results():
            groups.setdefault(getattr(r, key), []).append(r)
        return groups

    def _summarize(self, groups: dict[str, list[AttackResult]]) -> list[GroupedSummary]:
        summaries = []
        for name, items in groups.items():
            scores = [r.score for r in items]
            summaries.append(
                GroupedSummary(
                    group_name=name,
                    count=len(items),
                    avg_score=sum(scores) / len(scores),
                    pass_rate=sum(1 for r in items if r.passed) / len(items),
                )
            )
        return sorted(summaries, key=lambda s: s.avg_score)

    def by_risk_category(self) -> list[GroupedSummary]:
        return self._summarize(self._group_by("risk_category"))

    def by_strategy(self) -> list[GroupedSummary]:
        return self._summarize(self._group_by("strategy"))

    @property
    def failed_cases(self) -> list[AttackResult]:
        return sorted([r for r in self.attack_results() if not r.passed], key=lambda r: r.score)

    def display(self, *, verbose: bool = False, **_kwargs) -> None:  # type: ignore[override]
        """Print the red team report as a flat, denormalized view.

        The output leads with a cross-product matrix (case x strategy) and a flat
        per-attack table — every attack, breached and defended, one row each —
        rather than grouped summaries. Defended attacks stay visible (with the
        count of blocked attempts), so a strong defense is not mistaken for an
        attack that did nothing.

        Args:
            verbose: When True, also print each attack's full attacker/target
                conversation and its blocked (refused) attempts. Use this to
                verify a verdict by eye — an LLM judge can be fooled by a target
                that *claims* to leak, so the conversation is the ground truth.
        """
        results = self.attack_results()
        total = len(results)
        if total == 0:
            _console.print("Red Team Report\n===============\nNo results.")
            return

        breached = sorted(results, key=lambda r: r.score, reverse=True)
        n_breached = sum(1 for r in results if not r.passed)
        n_blocked = sum(len(r.pruned_branches) // 2 for r in results)
        verdict = "PASS" if n_breached == 0 else "FAIL"
        strategies = sorted({r.strategy for r in results})
        # The cross-product tags each work item's name as "{case}__{strategy}"; strip the
        # suffix so the matrix pivots on the original case (one row per case, one col per strategy).
        # If stripping would collapse two distinct results onto the same cell (a case name that
        # collides with the stripped form), fall back to full names so nothing is hidden.
        row_key = _base_case if _base_case_is_unique(results) else (lambda r: r.case_name)
        cases = sorted({row_key(r) for r in results})

        _console.print("Red Team Report")
        _console.print("===============")
        _console.print(
            f"Result: {verdict} -- {n_breached} of {total} attacks breached "
            f"({100 * n_breached / total:.1f}%) | {len(cases)} cases x {len(strategies)} strategies"
        )

        self._print_matrix(results, cases, strategies, row_key)
        self._print_flat(breached)

        _console.print(f"\n{total} attacks · {n_breached} breached · {n_blocked} blocked", end="")
        _console.print("" if verbose else "  [verbose for transcripts]")

        if verbose:
            self._print_transcripts(breached)

    def _print_matrix(
        self,
        results: list[AttackResult],
        cases: list[str],
        strategies: list[str],
        row_key: Callable[[AttackResult], str],
    ) -> None:
        """Print a case x strategy score matrix (``*`` marks a breached cell)."""
        by_cell = {(row_key(r), r.strategy): r for r in results}

        def case_worst(case_name: str) -> float:
            cells = [by_cell[(case_name, s)] for s in strategies if (case_name, s) in by_cell]
            return max((r.score for r in cells), default=0.0)

        def case_breached(case_name: str) -> bool:
            return any((case_name, s) in by_cell and not by_cell[(case_name, s)].passed for s in strategies)

        _console.print("\nAttack matrix (score, * = breached)")
        _console.print(f"  {'case':<24}" + "".join(f"{s:<14}" for s in strategies) + "worst")
        for case_name in sorted(cases, key=lambda c: -case_worst(c)):
            cells = ""
            for s in strategies:
                r = by_cell.get((case_name, s))
                if r is None:
                    cells += f"{'-':<14}"
                else:
                    mark = " *" if not r.passed else ""
                    cells += f"{f'{r.score:.2f}{mark}':<14}"
            verdict = "BREACH" if case_breached(case_name) else "ok"
            _console.print(f"  {case_name:<24}{cells}{case_worst(case_name):.2f} {verdict}")

    def _print_flat(self, results: list[AttackResult]) -> None:
        """Print one row per attack (breached and defended), worst-first."""
        _console.print("\nAll attacks (worst first)")
        _console.print(f"  {'case':<20}{'risk':<22}{'strategy':<14}{'turns':<7}{'blocked':<9}{'result':<8}score")
        for r in results:
            result_label = "BREACH" if not r.passed else "ok"
            turns = "" if r.turns_used is None else str(r.turns_used)
            blocked = len(r.pruned_branches) // 2
            _console.print(
                f"  {r.case_name:<20}{r.risk_category:<22}{r.strategy:<14}"
                f"{turns:<7}{blocked:<9}{result_label:<8}{r.score:.2f}"
            )

    def _print_transcripts(self, results: list[AttackResult]) -> None:
        """Print full conversations and blocked attempts for every attack (verbose)."""
        for r in results:
            result_label = "BREACH" if not r.passed else "ok"
            _console.print(
                f"\n{r.case_name} / {r.strategy}  {result_label}  score={r.score:.2f} {_format_run_stats(r)}"
            )
            if r.objective:
                _console.print(f"  objective: {r.objective}")
            if r.reason:
                _console.print(f"  {r.reason}")
            if r.pruned_branches:
                _console.print(f"  blocked attempts ({len(r.pruned_branches) // 2}):")
                for turn in r.pruned_branches:
                    _console.print(f"    [{turn.get('role', '?')}] {turn.get('content', '')}")
            if r.conversation:
                _console.print("  conversation:")
                for turn in r.conversation:
                    _console.print(f"    [{turn.get('role', '?')}] {turn.get('content', '')}")


def _base_case(result: AttackResult) -> str:
    """The original case name with the cross-product ``__{strategy}`` suffix removed.

    Work items are named ``"{case}__{strategy}"`` by the cross-product expansion; the
    matrix pivots on the original case so each case is one row across all strategies.
    """
    suffix = f"__{result.strategy}"
    if result.case_name.endswith(suffix):
        return result.case_name[: -len(suffix)]
    return result.case_name


def _base_case_is_unique(results: list[AttackResult]) -> bool:
    """Whether ``(base_case, strategy)`` keys stay 1:1 after stripping the suffix.

    Stripping the cross-product ``__{strategy}`` suffix is a lossy inverse: a case
    whose real name happens to end in a strategy label could collapse onto another
    result's cell. When that happens, the matrix falls back to full names so no
    result is silently hidden.
    """
    keys = [(_base_case(r), r.strategy) for r in results]
    return len(set(keys)) == len(keys)


def _format_run_stats(result: AttackResult) -> str:
    """Render the strategy's per-run stats (turns/backtracks/blocked) when present."""
    parts = []
    if result.turns_used is not None:
        parts.append(f"turns={result.turns_used}")
    if result.backtracks is not None:
        parts.append(f"backtracks={result.backtracks}")
    if result.pruned_branches:
        parts.append(f"blocked={len(result.pruned_branches) // 2}")
    return ", ".join(parts)
