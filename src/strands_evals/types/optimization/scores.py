"""Scores, splits, labels, and the end phase's records for the optimizer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .estimates import Estimate

Range = tuple[float | None, float | None]  # (low, high); None = unbounded on that side


class CaseSplit(BaseModel):
    """The run's one train / validation / test split, from `Split.assign`; test may be empty."""

    model_config = ConfigDict(frozen=True)

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


class CandidateScores(BaseModel):
    """One candidate's scores on one split.

    A run counts when it is completed, or an `agent_failure` scored by `crash_score` (default "worst");
    infrastructure and verifier failures are missing until a re-run succeeds.

    Attributes:
        objective_by_case: Case name -> mean of that case's measured per-run objectives. For Reflection's
            per-case scores and the test report, never for ranking.
        metrics_by_case: Case name -> metric name -> that case's mean, for every metric (each evaluator and
            every usage metric), so `measured()` can report tokens and seconds next to the scores. Usage
            means cover completed and `agent_failure` runs, so a crash's cost counts.
        unavailable: Case name -> run index -> metric -> why scoring took the value out (e.g. outside the
            evaluator's range).
        objective_by_run: Case name -> each run's objective (None = not measured); written to the `score`
            event.
        metrics_by_run: Case name -> metric name -> each run's value; what `FitData.observations` is
            built from.
        objective: The measured objective: `Criteria.objective_value` over each goal metric's mean across
            cases; what the ranking uses without a fit.
        eligible: Every case has at least one run that counts for every goal metric; an evaluator N/A on a
            case doesn't count against it. Only eligible candidates are fitted, ranked as parents, or chosen
            as finalists.
        failures: `agent_failure` trials on this split (each trial's last attempt); infrastructure and
            verifier failures aren't counted.
    """

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    objective_by_case: dict[str, float]
    metrics_by_case: dict[str, dict[str, float]]
    unavailable: dict[str, dict[int, dict[str, str]]]
    objective_by_run: dict[str, list[float | None]]
    metrics_by_run: dict[str, dict[str, list[float | None]]]
    objective: float | None
    eligible: bool
    failures: int


class Measured(BaseModel):
    """One metric measured over a split's cases; for reporting only.

    Attributes:
        mean: Mean of the per-case means (the case is the unit).
        ci90: 90% t-interval over the per-case means; None when `n < 2`.
        n: Cases with a value.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    ci90: tuple[float, float] | None
    n: int


class SplitScores(BaseModel):
    """One `CandidateScores` per split, never mixed.

    The search (fit, selection, labels, observe, Reflection) reads `train`; the end phase's pick reads
    `validation`; the test report reads `test`. Every scored candidate has train scores; `validation` is
    None unless it was a finalist or the baseline, `test` None unless it was the champion or the baseline.
    """

    model_config = ConfigDict(frozen=True)

    train: CandidateScores
    validation: CandidateScores | None
    test: CandidateScores | None


class PairedComparison(BaseModel):
    """A paired t-interval on the cases both candidates measured (candidate - reference)."""

    model_config = ConfigDict(frozen=True)

    mean_diff: float
    ci_low: float
    ci_high: float
    n_cases: int


class CandidateLabel(BaseModel):
    """A candidate's label on train, recorded in the `label` event; a label, never a filter.

    `infeasible` means the ranking's feasibility says no (still a possible parent, ranked lower);
    `ineligible` means some train case is unmeasured (out of the fit and the parents until a re-run
    succeeds). `estimates`, `p_met`, `by_fit`, and `measured` are the ranking's view at this step
    as a snapshot, never read on resume.

    Attributes:
        outcome: `feasible`, `infeasible`, or `ineligible`.
        reason: One sentence, e.g. "tokens <= 4000: P(met) 0.62 < 0.9".
        unmet: Each constraint not met, as the ranking assessed it, as a sentence (e.g. "tokens <= 4000:
            P(met) 0.62 < 0.9"); unlike the per-constraint margins elsewhere named `violations`.
        estimates: Per goal metric (`Assessment.estimates`); empty when `by_fit` is False.
        p_met: Per constraint (`Assessment.p_met`); empty when `by_fit` is False.
        by_fit: False when assessed by measured means.
        measured: Every metric measured on train, usage included; a metric with no value is left out.
    """

    model_config = ConfigDict(frozen=True)

    outcome: Literal["feasible", "infeasible", "ineligible"]
    reason: str
    unmet: list[str]
    estimates: dict[str, Estimate]
    p_met: dict[str, float]
    by_fit: bool
    measured: dict[str, Measured]


class ValidationReport(BaseModel):
    """How the end phase picked the champion on validation, saved in the `selected` event.

    Attributes:
        champion_id: What ships; the baseline's id when nothing feasible beats it.
        front: Under a `Pareto` objective, the validated front (the baseline included when on it), in ranking
            order, so the champion is first unless the front is empty; `[]` otherwise.
        objectives: Candidate id -> its validation objective under the validation fit, or by measured
            validation means when `problem` is set; None when it isn't eligible on validation.
        estimates: Candidate id -> goal metric -> the validation fit's estimate; empty when assessed by
            measured means.
        infeasible: Candidate id -> why it can't ship (a constraint not met on validation, or not
            eligible there).
        measured: Candidate id -> every metric measured on validation.
        problem: Why the validation fit wasn't used, e.g. "fewer than 2 eligible candidates" or
            "the model raised: <Type>: <message>"; None when it was used.
        reason: One sentence, e.g. "c3a1 highest on validation: 0.81 vs baseline 0.77".
    """

    model_config = ConfigDict(frozen=True)

    champion_id: str
    front: list[str]
    objectives: dict[str, float | None]
    estimates: dict[str, dict[str, Estimate]]
    infeasible: dict[str, str]
    measured: dict[str, dict[str, Measured]]
    problem: str | None
    reason: str


class TestReport(BaseModel):
    """The champion against the baseline on test, saved in the `test` event; report only.

    When the baseline ships, `champion_id == baseline_id`: the baseline is measured once and compared with
    itself (differences 0, ratios 1), and `missing` and `means` hold one key. Under a `Pareto` objective
    each tested front member gets its own report against the baseline (`champion_id` = that member).

    Attributes:
        objective: `Criteria.objective_value` per case, champion - baseline.
        difference: Per goal metric.
        ratio: Per goal metric; None when the baseline mean is 0 or the metric can be < 0.
        missing: Candidate id -> number of test cases with no counted run (left out).
        means: Champion and baseline -> every metric measured on test.
    """

    __test__ = False  # not a pytest test class

    model_config = ConfigDict(frozen=True)

    champion_id: str
    baseline_id: str
    objective: PairedComparison
    difference: dict[str, PairedComparison]
    ratio: dict[str, PairedComparison | None]
    missing: dict[str, int]
    means: dict[str, dict[str, Measured]]
