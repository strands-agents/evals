"""Red team report wrapper.

Provides grouped views over EvaluationReport results — by attack type,
strategy, and severity — without modifying the base report type.
"""

from __future__ import annotations

from dataclasses import dataclass

from strands_evals.types.evaluation_report import EvaluationReport


@dataclass
class AttackResult:
    """Single attack case result extracted from an EvaluationReport."""

    case_name: str
    attack_type: str
    strategy: str
    severity: str
    score: float
    passed: bool
    reason: str


@dataclass
class GroupedSummary:
    """Aggregated summary for a group of attack results."""

    group_name: str
    count: int
    avg_score: float
    pass_rate: float
    worst_case: AttackResult | None = None


class RedTeamReport:
    """Wraps EvaluationReport list with red-team-specific grouping views."""

    def __init__(self, results: list[AttackResult], raw_reports: list[EvaluationReport]):
        self.results = results
        self.raw_reports = raw_reports

    @classmethod
    def from_evaluation_reports(cls, reports: list[EvaluationReport]) -> RedTeamReport:
        """Build a RedTeamReport from standard EvaluationReport list."""
        results: list[AttackResult] = []
        for report in reports:
            for i, case_data in enumerate(report.cases):
                meta = case_data.get("metadata") or {}
                results.append(
                    AttackResult(
                        case_name=case_data.get("name", f"case_{i}"),
                        attack_type=meta.get("attack_type", "unknown"),
                        strategy=meta.get("strategy", "unknown"),
                        severity=meta.get("severity", "unknown"),
                        score=report.scores[i] if i < len(report.scores) else 0.0,
                        passed=report.test_passes[i] if i < len(report.test_passes) else True,
                        reason=report.reasons[i] if i < len(report.reasons) else "",
                    )
                )
        return cls(results=results, raw_reports=reports)

    def _group_by(self, key: str) -> dict[str, list[AttackResult]]:
        groups: dict[str, list[AttackResult]] = {}
        for r in self.results:
            k = getattr(r, key)
            groups.setdefault(k, []).append(r)
        return groups

    def _summarize(self, groups: dict[str, list[AttackResult]]) -> list[GroupedSummary]:
        summaries = []
        for name, items in groups.items():
            scores = [r.score for r in items]
            worst = min(items, key=lambda r: r.score)
            summaries.append(
                GroupedSummary(
                    group_name=name,
                    count=len(items),
                    avg_score=sum(scores) / len(scores),
                    pass_rate=sum(1 for r in items if r.passed) / len(items),
                    worst_case=worst,
                )
            )
        return sorted(summaries, key=lambda s: s.avg_score)

    def by_attack_type(self) -> list[GroupedSummary]:
        """Group results by attack type."""
        return self._summarize(self._group_by("attack_type"))

    def by_strategy(self) -> list[GroupedSummary]:
        """Group results by strategy."""
        return self._summarize(self._group_by("strategy"))

    def by_severity(self) -> list[GroupedSummary]:
        """Group results by severity level."""
        return self._summarize(self._group_by("severity"))

    @property
    def overall_score(self) -> float:
        scores = [r.score for r in self.results]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0.0

    @property
    def failed_cases(self) -> list[AttackResult]:
        """Cases where the target's defenses did not hold."""
        return sorted(
            [r for r in self.results if not r.passed],
            key=lambda r: r.score,
        )

    def to_evaluation_report(self) -> EvaluationReport:
        """Flatten back into a single EvaluationReport for compatibility."""
        return EvaluationReport.flatten(self.raw_reports)
