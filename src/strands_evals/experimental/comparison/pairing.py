"""Deterministic pairing for baseline and candidate evaluation reports."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Collection, Sequence
from statistics import fmean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ...types.evaluation_report import EvaluationReport


class _ScorePair(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    evaluator: str


class PairedScore(_ScorePair):
    """One evaluator score matched between baseline and candidate reports.

    `baseline_index` and `candidate_index` identify source rows for a direct
    report pair. They are `None` after multiple trial rows have been averaged.

    Attributes:
        case_id: Stable identifier derived from the case name or input.
        evaluator: Name of the evaluator that produced both scores.
        baseline_score: Score produced by the baseline.
        candidate_score: Score produced by the candidate.
        baseline_index: Source row in the baseline report, when applicable.
        candidate_index: Source row in the candidate report, when applicable.
        baseline_trial_count: Applicable baseline trials included in an
            aggregate, or `None` for a direct report pair.
        candidate_trial_count: Applicable candidate trials included in an
            aggregate, or `None` for a direct report pair.
        delta: Candidate score minus baseline score.
    """

    baseline_score: float
    candidate_score: float
    baseline_index: int | None = Field(default=None, ge=0)
    candidate_index: int | None = Field(default=None, ge=0)
    baseline_trial_count: int | None = Field(default=None, ge=1)
    candidate_trial_count: int | None = Field(default=None, ge=1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delta(self) -> float:
        """Return the candidate-minus-baseline score change."""
        return self.candidate_score - self.baseline_score


class PairedTrialScore(_ScorePair):
    """One evaluator's baseline and candidate observations for a trial.

    Attributes:
        baseline_score: Baseline score, or `None` when not applicable.
        candidate_score: Candidate score, or `None` when not applicable.
        baseline_index: Source row in the baseline report for this trial.
        candidate_index: Source row in the candidate report for this trial.
        trial_index: Zero-based position of the trial in the report sequences.
        delta: Candidate-minus-baseline change when both sides are applicable.
    """

    baseline_score: float | None
    candidate_score: float | None
    baseline_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    trial_index: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def delta(self) -> float | None:
        """Return the change when both observations are applicable."""
        if self.baseline_score is None or self.candidate_score is None:
            return None
        return self.candidate_score - self.baseline_score


def pair_reports(*, baseline: EvaluationReport, candidate: EvaluationReport) -> list[PairedScore]:
    """Strictly match comparable evaluator scores from two reports.

    Cases are paired by their exact case name when one is present; inputs are
    not compared for named cases. An unnamed case is paired by a deterministic
    hash of its input. Evaluator names must also match. Row order, generated
    session IDs, and task outputs do not affect pairing. Results are sorted by
    evaluator and case identifier.

    Rows that are not applicable in both reports are omitted. A mismatch in
    applicability is rejected because the two scores do not represent the same
    measurement. This strict behavior is intended for controlled regression
    comparisons where both runs must use the same cases and evaluators.

    The returned source indices let later layers inspect report reasons or
    exclude infrastructure-corrupted trials without inferring corruption from
    a low score.

    Args:
        baseline: Evaluation report produced by the existing configuration.
        candidate: Evaluation report produced by the changed configuration.

    Returns:
        The matched score rows and their candidate-minus-baseline deltas.

    Raises:
        ValueError: If either report is malformed, contains ambiguous rows, or
            cannot be paired exactly with the other report.
    """

    baseline_rows = _index_report(baseline, "baseline")
    candidate_rows = _index_report(candidate, "candidate")

    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)
    if baseline_keys != candidate_keys:
        raise ValueError(
            "reports do not contain the same case/evaluator rows: "
            f"baseline_only={_summarize_keys(baseline_keys - candidate_keys)}, "
            f"candidate_only={_summarize_keys(candidate_keys - baseline_keys)}"
        )

    pairs: list[PairedScore] = []
    for key in sorted(baseline_keys):
        baseline_index = baseline_rows[key]
        candidate_index = candidate_rows[key]
        evaluator, case_id = key

        baseline_details = (
            baseline.detailed_results[baseline_index] if baseline_index < len(baseline.detailed_results) else []
        )
        candidate_details = (
            candidate.detailed_results[candidate_index] if candidate_index < len(candidate.detailed_results) else []
        )
        baseline_applicable = EvaluationReport.is_applicable(baseline_details)
        candidate_applicable = EvaluationReport.is_applicable(candidate_details)
        if baseline_applicable != candidate_applicable:
            raise ValueError(
                f"case applicability differs between reports: case_id={case_id!r}, evaluator={evaluator!r}"
            )
        if not baseline_applicable:
            continue

        baseline_score = baseline.scores[baseline_index]
        candidate_score = candidate.scores[candidate_index]
        pairs.append(
            PairedScore(
                case_id=case_id,
                evaluator=evaluator,
                baseline_index=baseline_index,
                candidate_index=candidate_index,
                baseline_score=baseline_score,
                candidate_score=candidate_score,
            )
        )

    return pairs


def pair_report_trials(
    *,
    baseline_reports: Sequence[EvaluationReport],
    candidate_reports: Sequence[EvaluationReport],
    corrupted_trial_indices: Collection[int] = (),
) -> list[PairedTrialScore]:
    """Align baseline and candidate report observations across trials.

    Every trial must contain the same case/evaluator rows. Applicability may
    vary independently between baseline and candidate trials because repeated
    agent runs are not paired by seed or state. Each side is averaged over its
    own applicable trials during aggregation. Explicitly corrupted trials are
    skipped before their reports are validated. Returned rows retain their
    original trial and source indices.

    Args:
        baseline_reports: Baseline reports ordered by trial.
        candidate_reports: Candidate reports ordered by trial.
        corrupted_trial_indices: Trial indices to remove from both report
            sequences before validation and pairing.

    Returns:
        Aligned evaluator observations from every trial.

    Raises:
        ValueError: If no trials are supplied, the trial counts differ, a
            corrupted index is unknown, no usable trials remain, or the
            remaining reports are not comparable across trials.
    """

    if not baseline_reports or not candidate_reports:
        raise ValueError("baseline and candidate report sequences must contain at least one trial")
    if len(baseline_reports) != len(candidate_reports):
        raise ValueError(
            "baseline and candidate report sequences must contain the same number of trials: "
            f"baseline={len(baseline_reports)}, candidate={len(candidate_reports)}"
        )

    trial_indices = set(range(len(baseline_reports)))
    corrupted = set(corrupted_trial_indices)
    unknown_corrupted = corrupted - trial_indices
    if unknown_corrupted:
        raise ValueError(
            "corrupted_trial_indices contain unknown trials: "
            f"unknown={sorted(unknown_corrupted)}, valid={sorted(trial_indices)}"
        )
    if corrupted == trial_indices:
        raise ValueError("no usable trials remain after removing corrupted trials")

    reference_row_keys: set[tuple[str, str]] | None = None
    trial_pairs: list[PairedTrialScore] = []
    for trial_index, (baseline, candidate) in enumerate(zip(baseline_reports, candidate_reports, strict=True)):
        if trial_index in corrupted:
            continue

        try:
            baseline_rows = _index_report(baseline, f"baseline trial {trial_index}")
            candidate_rows = _index_report(candidate, f"candidate trial {trial_index}")
        except ValueError as error:
            raise ValueError(f"trial {trial_index}: {error}") from error

        row_keys = set(baseline_rows)
        candidate_keys = set(candidate_rows)
        if row_keys != candidate_keys:
            raise ValueError(
                f"trial {trial_index}: reports do not contain the same case/evaluator rows: "
                f"baseline_only={_summarize_keys(row_keys - candidate_keys)}, "
                f"candidate_only={_summarize_keys(candidate_keys - row_keys)}"
            )
        if reference_row_keys is None:
            reference_row_keys = row_keys
        elif row_keys != reference_row_keys:
            raise ValueError(
                "trial reports do not contain the same case/evaluator rows: "
                f"trial_index={trial_index}, "
                f"missing={_summarize_keys(reference_row_keys - row_keys)}, "
                f"unexpected={_summarize_keys(row_keys - reference_row_keys)}"
            )

        for evaluator, case_id in sorted(row_keys):
            baseline_index = baseline_rows[(evaluator, case_id)]
            candidate_index = candidate_rows[(evaluator, case_id)]
            baseline_details = (
                baseline.detailed_results[baseline_index] if baseline_index < len(baseline.detailed_results) else []
            )
            candidate_details = (
                candidate.detailed_results[candidate_index] if candidate_index < len(candidate.detailed_results) else []
            )
            baseline_applicable = EvaluationReport.is_applicable(baseline_details)
            candidate_applicable = EvaluationReport.is_applicable(candidate_details)
            trial_pairs.append(
                PairedTrialScore(
                    case_id=case_id,
                    evaluator=evaluator,
                    baseline_index=baseline_index,
                    candidate_index=candidate_index,
                    baseline_score=baseline.scores[baseline_index] if baseline_applicable else None,
                    candidate_score=candidate.scores[candidate_index] if candidate_applicable else None,
                    trial_index=trial_index,
                )
            )

    return sorted(trial_pairs, key=lambda pair: (pair.evaluator, pair.case_id, pair.trial_index))


def aggregate_trial_pairs(pairs: Sequence[PairedTrialScore]) -> list[PairedScore]:
    """Average repeated trial scores within each case and evaluator.

    Baseline and candidate scores are averaged independently over the trials
    where each side is applicable. Rows that are never applicable on either
    side are omitted. A row that is applicable on only one side across all
    trials is rejected because no comparison can be formed. Aggregating before
    statistical comparison keeps repeated agent runs from being treated as
    independent cases.

    Args:
        pairs: Trial-indexed scores returned by `pair_report_trials`.

    Returns:
        One averaged paired score per case and evaluator, including the
        applicable trial count for each side.

    Raises:
        ValueError: If a trial row is duplicated or a case/evaluator row has
            no applicable trials on one side.
    """

    if not pairs:
        return []

    grouped: defaultdict[tuple[str, str], list[PairedTrialScore]] = defaultdict(list)
    seen: set[tuple[int, str, str]] = set()
    for pair in pairs:
        row_key = (pair.trial_index, pair.evaluator, pair.case_id)
        if row_key in seen:
            raise ValueError(
                "pairs contain duplicate trial/case/evaluator row: "
                f"trial_index={pair.trial_index}, case_id={pair.case_id!r}, evaluator={pair.evaluator!r}"
            )
        seen.add(row_key)
        grouped[(pair.evaluator, pair.case_id)].append(pair)

    aggregated: list[PairedScore] = []
    for evaluator, case_id in sorted(grouped):
        case_pairs = grouped[(evaluator, case_id)]
        baseline_scores = [pair.baseline_score for pair in case_pairs if pair.baseline_score is not None]
        candidate_scores = [pair.candidate_score for pair in case_pairs if pair.candidate_score is not None]
        if not baseline_scores and not candidate_scores:
            continue
        if not baseline_scores or not candidate_scores:
            missing_side = "baseline" if not baseline_scores else "candidate"
            raise ValueError(
                "case/evaluator row has no applicable trials on one side: "
                f"case_id={case_id!r}, evaluator={evaluator!r}, missing={missing_side!r}"
            )
        aggregated.append(
            PairedScore(
                case_id=case_id,
                evaluator=evaluator,
                baseline_score=fmean(baseline_scores),
                candidate_score=fmean(candidate_scores),
                baseline_trial_count=len(baseline_scores),
                candidate_trial_count=len(candidate_scores),
            )
        )

    return aggregated


def _index_report(report: EvaluationReport, report_name: str) -> dict[tuple[str, str], int]:
    if len(report.cases) != len(report.scores):
        raise ValueError(f"{report_name} report has {len(report.cases)} cases but {len(report.scores)} scores")

    rows: dict[tuple[str, str], int] = {}
    for index, case in enumerate(report.cases):
        evaluator = case.get("evaluator")
        if not isinstance(evaluator, str) or not evaluator.strip():
            raise ValueError(f"{report_name} report row {index} is missing a non-empty evaluator name")

        if "input" not in case:
            raise ValueError(f"{report_name} report row {index} is missing its case input")

        case_id = _case_id(case, report_name, index)
        key = (evaluator, case_id)
        if key in rows:
            raise ValueError(
                f"{report_name} report contains duplicate case/evaluator row: "
                f"case_id={case_id!r}, evaluator={evaluator!r}"
            )

        rows[key] = index

    return rows


def _case_id(case: dict, report_name: str, index: int) -> str:
    name = case.get("name")
    if name is not None and not isinstance(name, str):
        raise ValueError(f"{report_name} report row {index} has a non-string case name")
    if isinstance(name, str) and name.strip():
        return name

    input_signature = _canonical_json(case["input"])
    digest = hashlib.sha256(input_signature.encode("utf-8")).hexdigest()
    return f"input:{digest}"


def _canonical_json(value: Any) -> str:
    try:
        json_value = json.loads(json.dumps(value, default=str))
        return json.dumps(json_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise ValueError("case input must be JSON-serializable") from error


def _summarize_keys(keys: set[tuple[str, str]], limit: int = 10) -> str:
    ordered = sorted(keys)
    if len(ordered) <= limit:
        return str(ordered)
    return f"{ordered[:limit]} ... ({len(ordered) - limit} more)"
