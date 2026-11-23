"""Scores, splits, labels, and the end phase's records for the optimizer."""

from __future__ import annotations

import math
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from .estimates import Estimate

Range = tuple[float | None, float | None]  # (low, high); None = unbounded on that side


def near(value: float, end: float) -> bool:
    """Equal up to float noise: the tolerance every range check, constraint bound, and score range uses."""
    return math.isclose(value, end, rel_tol=1e-9, abs_tol=1e-12)


LabelOutcome = Literal["feasible", "infeasible", "ineligible"]  # also the first three ObservationOutcomes
LABEL_OUTCOMES: tuple[LabelOutcome, ...] = get_args(LabelOutcome)  # the same outcomes, to iterate over
PickOutcome = Literal["top_finalist", "baseline_best", "no_feasible_finalist", "no_finalists"]  # how the pick ended


class CaseSplit(BaseModel):
    """The run's one train / validation / test split, from `Split.assign`; test may be empty."""

    model_config = ConfigDict(frozen=True)

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


class CandidateScores(BaseModel):
    """One candidate's scores on one split.

    A run counts when it is completed, or an `agent_failure` scored by `agent_failure_score` (default "worst");
    infrastructure and evaluator failures are missing until a re-run succeeds.

    Attributes:
        objective_by_case: Case name -> mean of that case's measured per-run objectives. For Reflection's
            per-case scores and the test report, never for ranking.
        metrics_by_case: Case name -> metric name -> that case's mean, for every metric with a value on that
            case (each evaluator and every usage metric), so `measured_metrics()` can report tokens and seconds
            next to the scores. Usage means cover completed and `agent_failure` runs, so a crash's cost counts.
        unavailable: Case name -> run index -> metric -> why scoring took the value out (e.g. outside the
            evaluator's range).
        objective_by_run: Case name -> each run's objective (None = not measured); written to the `score`
            event.
        metrics_by_run: Case name -> metric name -> each run's value; what the score model fits (via
            `FitData.scores`).
        objective: The measured objective: `Criteria.objective_value` over each goal metric's mean across
            cases. Without a usable fit the ranking starts from the same means and fills in the cases another
            assessed candidate has a value on and this one has none (`Assessment.objective`).
        is_eligible: Every case has at least one run that counts for every goal metric (an evaluator N/A on a
            case doesn't count against it), and the objective (under `Pareto` every term) is measured. Only
            eligible candidates are fitted, ranked as parents, or chosen as finalists.
        failures: `agent_failure` trials on this split (each trial's last attempt); infrastructure and
            evaluator failures aren't counted.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    objective_by_case: dict[str, float]
    metrics_by_case: dict[str, dict[str, float]]
    unavailable: dict[str, dict[int, dict[str, str]]]
    objective_by_run: dict[str, list[float | None]]
    metrics_by_run: dict[str, dict[str, list[float | None]]]
    objective: float | None
    is_eligible: bool
    failures: int


class Measured(BaseModel):
    """One metric measured over a split's cases; for reporting only.

    Attributes:
        mean: Mean over the units: the case (its mean over runs), or that case's runs when only one case
            has a value of the metric.
        ci95: 95% t-interval over the units, clipped to the metric's range; None when `n < 2`.
        n: Units with a value: cases, or runs when only one case has a value.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    ci95: tuple[float, float] | None
    n: int


class SplitScores(BaseModel):
    """One `CandidateScores` per split, never mixed.

    The search (fit, selection, labels, observe, Reflection) reads `train`; the end phase's pick reads
    `validation`; the test report reads `test`. Every scored candidate has train scores; `validation` is
    None unless it was a finalist or the baseline, `test` None unless it was the champion, the baseline, or
    (under `Pareto`) a tested front member.
    """

    model_config = ConfigDict(frozen=True)

    train: CandidateScores
    validation: CandidateScores | None
    test: CandidateScores | None


class PairedComparison(BaseModel):
    """A paired t-interval on the cases both candidates measured (candidate - reference).

    For a ratio, `mean_diff` holds the ratio of means. `ci_low` / `ci_high` are None with one shared case
    (no interval from one difference). `candidate_mean` and `reference_mean` are each side's mean over the
    same `n_cases` shared cases, so `mean_diff` is their difference (or ratio); None in a comparison saved
    before they existed.
    """

    model_config = ConfigDict(frozen=True)

    mean_diff: float
    ci_low: float | None
    ci_high: float | None
    n_cases: int
    candidate_mean: float | None = None
    reference_mean: float | None = None


class CandidateLabel(BaseModel):
    """A candidate's label on train, recorded in the `label` event; a label, never a filter.

    `infeasible` means the ranking's feasibility says no (still a possible parent, ranked lower);
    `ineligible` means some train case, or the objective, is unmeasured (out of the fit and the parents until a re-run
    succeeds). `estimates`, `p_met`, `adjusted_means`, `is_from_fit`, and `measured` are the ranking's view at
    this step as a snapshot, never read on resume.

    Attributes:
        outcome: `feasible`, `infeasible`, or `ineligible`.
        reason: One sentence, e.g. "tokens <= 4000: P(met) 0.62 < 0.9".
        unmet_constraints: Each constraint not met, as the ranking assessed it, as a sentence (e.g. "tokens <= 4000:
            P(met) 0.62 < 0.9"); unlike the per-constraint margins elsewhere named `violations`. For a constraint
            judged by its mean (a flagged metric, or on measured means) the sentence's mean is the adjusted one
            the ranking judged (`adjusted_means`), which can differ from the mean in `measured` for a candidate
            with N/A cases that another assessed candidate was judged on.
        estimates: Per goal metric the fit judged (`Assessment.estimates`; a flagged metric is left out);
            empty when `is_from_fit` is False.
        p_met: Per constraint the fit judged (`Assessment.p_met`); empty when `is_from_fit` is False.
        adjusted_means: Per goal metric, the adjusted mean the ranking used (`Assessment.adjusted_means`): the
            plain mean in `measured`, or with the N/A cases another assessed candidate was judged on filled in;
            only the goal metrics the fit didn't judge (the flagged ones under a fit, every one on measured means),
            so empty when the fit judged every one, when the candidate is ineligible, and in a label saved before
            it existed.
        is_from_fit: False when assessed by measured means alone.
        measured: Every metric measured on train, usage included; a metric with no value is left out.
    """

    model_config = ConfigDict(frozen=True)

    outcome: LabelOutcome
    reason: str
    unmet_constraints: list[str]
    estimates: dict[str, Estimate]
    p_met: dict[str, float]
    adjusted_means: dict[str, float] = Field(default_factory=dict)
    is_from_fit: bool
    measured: dict[str, Measured]


class ValidationReport(BaseModel):
    """How the end phase picked the champion on validation, saved in the `picked` event.

    Attributes:
        champion_id: What ships; the baseline's id when nothing feasible beats it.
        front: Under a `Pareto` objective, the validated front (the baseline included when on it), in ranking
            order, so the champion is first unless the front is empty; `[]` otherwise.
        objectives: Candidate id -> its validation objective under the validation fit (a flagged metric's
            measured mean instead), or by measured validation means (`Assessment.objective`) when `problem` is
            set; None when it isn't eligible on validation or no validation trial of it completed. In the
            pick's order: the baseline, then the finalists in search ranking order.
        estimates: Candidate id -> goal metric -> the validation fit's estimate (a flagged metric is left out);
            empty when assessed by measured means.
        adjusted_means: Candidate id -> goal metric -> the adjusted validation mean the pick used for a metric
            the validation fit didn't judge (`Assessment.adjusted_means`): the plain mean in `measured`, or with
            the N/A cases another assessed candidate was judged on filled in; only the metrics the validation fit
            didn't judge (a flagged one, or every one on measured means). Every assessed candidate, as in
            `estimates`, so `{}` when the fit judged every metric (no candidate in a report saved before it
            existed).
        infeasible: Candidate id -> why it can't ship (a constraint not met on validation, not eligible
            there, or no validation trial of it completed).
        measured: Candidate id -> every metric measured on validation.
        p_met: Candidate id -> constraint (as `run.json` writes it, e.g. "tokens <= 4000") -> P(met) under the
            validation fit, for each constraint the fit judged (its metric not flagged); `{}` when assessed by
            measured means, and in a report saved before it existed.
        problem: Why the validation fit wasn't used for every goal metric: "fewer than 2 eligible candidates",
            "the model returned no fit", "the model raised: <Type>: <message>", "flagged: <metrics>" (every
            goal metric flagged), or "flagged: <metrics> (by adjusted means; the other metrics by the fit)";
            None when the fit judged every goal metric.
        rule: The pick rule's config (`FinalPick.rule.config()`), e.g. `{"kind": "Mean"}`; None in a report
            saved before it existed.
        reason: One sentence, e.g. "c3a1 highest on validation: 0.81 vs baseline 0.77".
    """

    model_config = ConfigDict(frozen=True)

    champion_id: str
    front: list[str]
    objectives: dict[str, float | None]
    estimates: dict[str, dict[str, Estimate]]
    adjusted_means: dict[str, dict[str, float]] = Field(default_factory=dict)
    infeasible: dict[str, str]
    measured: dict[str, dict[str, Measured]]
    p_met: dict[str, dict[str, float]] = Field(default_factory=dict)
    problem: str | None
    rule: dict[str, Any] | None = None
    reason: str

    @property
    def outcome(self) -> PickOutcome:
        """How the pick ended, from `champion_id`, `objectives` (the baseline first), and `infeasible`:
        `no_finalists` (the search gave none, so the baseline ships), `top_finalist` (a finalist ranks first
        among the feasible candidates and ships), `baseline_best` (the baseline ranks first and ships), or
        `no_feasible_finalist` (no finalist is feasible, so the baseline ships)."""
        baseline, *finalists = self.objectives
        if not finalists:
            return "no_finalists"
        if self.champion_id != baseline:
            return "top_finalist"
        return "baseline_best" if any(c not in self.infeasible for c in finalists) else "no_feasible_finalist"


class TestReport(BaseModel):
    """The champion against the baseline on test, saved in the `test` event; report only.

    When the baseline ships, `champion_id == baseline_id`: the baseline is measured once and compared with
    itself (differences 0, ratios 1), and `missing` and `means` hold one key. Under a `Pareto` objective
    each tested front member gets its own report against the baseline (`champion_id` = that member).

    Attributes:
        objective: `Criteria.objective_value` per case, champion - baseline; None when no test case was
            measured by both.
        difference: Per goal metric; a metric with no test case measured by both is left out.
        ratio: Per goal metric; None when the baseline mean is 0 or the metric can be < 0; a metric with no
            test case measured by both is left out, as in `difference`.
        missing: Candidate id -> number of test cases with no counted run (left out).
        means: Champion and baseline -> every metric measured on test.
    """

    __test__ = False  # not a pytest test class

    model_config = ConfigDict(frozen=True)

    champion_id: str
    baseline_id: str
    objective: PairedComparison | None
    difference: dict[str, PairedComparison]
    ratio: dict[str, PairedComparison | None]
    missing: dict[str, int]
    means: dict[str, dict[str, Measured]]
