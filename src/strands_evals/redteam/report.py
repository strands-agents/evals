"""Red team report.

Wraps per-evaluator EvaluationReport results into a single case-centric view
where each AttackResult carries multi-metric scores from all evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console

from strands_evals.types.evaluation_report import EvaluationReport

_console = Console()


@dataclass
class AttackResult:
    """One attack case with scores from every evaluator that ran on it."""

    case_name: str
    attack_type: str
    strategy: str
    severity: str
    scores: dict[str, float] = field(default_factory=dict)
    passes: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

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


class RedTeamReport:
    """Case-centric view over one or more EvaluationReports."""

    def __init__(self, results: list[AttackResult]):
        self.results = results

    @classmethod
    def from_evaluation_reports(cls, reports: list[EvaluationReport]) -> RedTeamReport:
        by_case: dict[str, AttackResult] = {}
        for report in reports:
            evaluator = report.evaluator_name or "evaluator"
            for i, case_data in enumerate(report.cases):
                name = case_data.get("name", f"case_{i}")
                meta = case_data.get("metadata") or {}
                result = by_case.setdefault(
                    name,
                    AttackResult(
                        case_name=name,
                        attack_type=meta.get("attack_type", "unknown"),
                        strategy=meta.get("strategy", "unknown"),
                        severity=meta.get("severity", "unknown"),
                    ),
                )
                result.scores[evaluator] = report.scores[i] if i < len(report.scores) else 0.0
                result.passes[evaluator] = report.test_passes[i] if i < len(report.test_passes) else True
                result.reasons[evaluator] = report.reasons[i] if i < len(report.reasons) else ""
        return cls(results=list(by_case.values()))

    def _group_by(self, key: str) -> dict[str, list[AttackResult]]:
        groups: dict[str, list[AttackResult]] = {}
        for r in self.results:
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

    def by_attack_type(self) -> list[GroupedSummary]:
        return self._summarize(self._group_by("attack_type"))

    def by_strategy(self) -> list[GroupedSummary]:
        return self._summarize(self._group_by("strategy"))

    @property
    def overall_score(self) -> float:
        return sum(r.score for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0.0

    @property
    def failed_cases(self) -> list[AttackResult]:
        return sorted([r for r in self.results if not r.passed], key=lambda r: r.score)

    def display(self) -> None:
        total = len(self.results)
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

        _console.print("\nBy attack type:")
        for s in self.by_attack_type():
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
                _console.print(f"  [FAIL] score={r.score:.2f} severity={r.severity} | {r.reason}")
