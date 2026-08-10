"""Group failed cases by evaluator for failure cohort analysis.

Given an EvaluationReport, this module buckets failed cases by the evaluator
that produced them, sorted largest-first. A large bucket indicates a systemic
problem worth investigating; scattered one-off failures are noise.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..types.evaluation_report import EvaluationReport


class FailureCohort(BaseModel):
    """A group of test cases that all failed the same evaluator."""

    evaluator_name: str
    failed_case_indices: list[int]
    failed_case_names: list[str]
    count: int

    @property
    def is_systemic(self) -> bool:
        """Two or more failures suggests a shared root cause."""
        return self.count >= 2


class CohortAnalysis(BaseModel):
    """Result of grouping failed cases by evaluator."""

    cohorts: list[FailureCohort]
    total_failures: int
    total_cases: int

    @property
    def systemic_cohorts(self) -> list[FailureCohort]:
        """Return only cohorts with two or more failures."""
        return [c for c in self.cohorts if c.is_systemic]

    @property
    def one_off_failures(self) -> list[FailureCohort]:
        """Return only cohorts with exactly one failure."""
        return [c for c in self.cohorts if c.count == 1]


def analyze_failure_cohorts(report: EvaluationReport) -> CohortAnalysis:
    """Group failed cases by evaluator, sorted largest-first.

    Each case in the report has an "evaluator" key identifying which evaluator
    produced that row. Failed cases get bucketed by that key.

    Args:
        report: An EvaluationReport (typically from Experiment.run_evaluations or
            EvaluationReport.from_file).

    Returns:
        A CohortAnalysis containing sorted failure cohorts and summary counts.
    """
    failures_by_evaluator: dict[str, list[tuple[int, str]]] = {}

    for i, (case, passed) in enumerate(zip(report.cases, report.test_passes, strict=False)):
        if passed:
            continue
        eval_name = case.get("evaluator", "unknown")
        case_name = case.get("name", f"case_{i}")
        failures_by_evaluator.setdefault(eval_name, []).append((i, case_name))

    cohorts = []
    for eval_name, members in failures_by_evaluator.items():
        indices = [m[0] for m in members]
        names = [m[1] for m in members]
        cohorts.append(
            FailureCohort(
                evaluator_name=eval_name,
                failed_case_indices=indices,
                failed_case_names=names,
                count=len(members),
            )
        )

    cohorts.sort(key=lambda c: (-c.count, c.evaluator_name))

    return CohortAnalysis(
        cohorts=cohorts,
        total_failures=sum(1 for p in report.test_passes if not p),
        total_cases=len(report.test_passes),
    )


def print_cohort_summary(analysis: CohortAnalysis) -> None:
    """Print a Rich table summarizing the failure cohorts.

    Args:
        analysis: A CohortAnalysis returned by analyze_failure_cohorts.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Failure Cohorts ({analysis.total_failures}/{analysis.total_cases} failed)")
    table.add_column("Evaluator", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Cases")

    for cohort in analysis.cohorts:
        names = ", ".join(cohort.failed_case_names[:5])
        if cohort.count > 5:
            names += f" (+{cohort.count - 5} more)"
        table.add_row(cohort.evaluator_name, str(cohort.count), names)

    console.print(table)
