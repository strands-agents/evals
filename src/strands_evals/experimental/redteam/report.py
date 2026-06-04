"""Red team report."""

from __future__ import annotations

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
    scores: dict[str, float] = field(default_factory=dict)
    passes: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return min(self.scores.values()) if self.scores else 0.0

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
        ``trajectory`` holds raw tool I/O — sanitize before sharing if
        target tools return sensitive data.
    """

    @classmethod
    def from_evaluation_reports(
        cls, reports: list[EvaluationReport], run_meta: dict[str, dict] | None = None
    ) -> RedTeamReport:
        """Merge per-evaluator reports into a single case-centric report.

        Args:
            reports: One EvaluationReport per evaluator.
            run_meta: Optional per-case strategy run metadata (turns_used,
                backtracks, ...) keyed by case name, merged onto each case's
                metadata so it surfaces in the report. Supplied by
                ``RedTeamExperiment`` because the base ``Experiment`` does not
                carry task-returned metadata into the ``EvaluationData``.
        """
        run_meta = run_meta or {}
        scores: list[float] = []
        cases: list[dict] = []
        passes: list[bool] = []
        reasons: list[str] = []
        detailed: list = []

        for report in reports:
            evaluator = report.evaluator_name or "evaluator"
            n = len(report.cases)
            if not (len(report.scores) == n and len(report.test_passes) == n and len(report.reasons) == n):
                raise ValueError(f"EvaluationReport {evaluator!r}: cases/scores/passes/reasons length mismatch")
            # detailed_results is optional; pad with [] when shorter than cases.
            for i, case_data in enumerate(report.cases):
                merged_metadata = {**(case_data.get("metadata") or {}), **run_meta.get(case_data.get("name", ""), {})}
                cases.append({**case_data, "evaluator": evaluator, "metadata": merged_metadata})
                scores.append(report.scores[i])
                passes.append(report.test_passes[i])
                reasons.append(report.reasons[i])
                detailed.append(report.detailed_results[i] if i < len(report.detailed_results) else [])

        return cls(
            evaluator_name="RedTeam",
            overall_score=sum(scores) / len(scores) if scores else 0.0,
            scores=scores,
            cases=cases,
            test_passes=passes,
            reasons=reasons,
            detailed_results=detailed,
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

    def display(self, **_kwargs) -> None:  # type: ignore[override]
        results = self.attack_results()
        total = len(results)
        if total == 0:
            _console.print("Red Team Report\n===============\nNo results.")
            return

        failed = self.failed_cases
        verdict = "PASS" if not failed else "FAIL"
        _console.print("Red Team Report")
        _console.print("===============")
        _console.print(
            f"Result: {verdict} -- {len(failed)} of {total} attacks succeeded ({100 * len(failed) / total:.1f}%)"
        )

        _console.print("\nBy risk category:")
        for s in self.by_risk_category():
            succeeded = s.count - round(s.pass_rate * s.count)
            _console.print(f"  {s.group_name:<20} {succeeded}/{s.count} succeeded ({100 * (1 - s.pass_rate):.1f}%)")

        if len(self.by_strategy()) > 1:
            _console.print("\nBy strategy:")
            for s in self.by_strategy():
                succeeded = s.count - round(s.pass_rate * s.count)
                _console.print(f"  {s.group_name:<20} {succeeded}/{s.count} succeeded ({100 * (1 - s.pass_rate):.1f}%)")

        if failed:
            _console.print("\nFailures:")
            for r in failed:
                _console.print(f"  [FAIL] score={r.score:.2f} severity={r.severity} strategy={r.strategy}")
                if r.objective:
                    _console.print(f"      objective: {r.objective}")
                run_stats = _format_run_stats(r)
                if run_stats:
                    _console.print(f"      {run_stats}")
                if r.reason:
                    _console.print(f"      {r.reason}")


def _format_run_stats(result: AttackResult) -> str:
    """Render the strategy's per-run stats (turns/backtracks) when present."""
    parts = []
    if result.turns_used is not None:
        parts.append(f"turns={result.turns_used}")
    if result.backtracks is not None:
        parts.append(f"backtracks={result.backtracks}")
    return ", ".join(parts)
