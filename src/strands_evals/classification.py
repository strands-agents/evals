"""Post-hoc risk classification for multi-run evaluation results.

When the same `(case, evaluator)` pair is evaluated several times, the raw pass/fail
counts do not say *what kind* of problem a failure is. A case that fails every run is a
deterministic bug; one that fails intermittently is flaky. The two need different responses,
so this module turns a sequence of repeated `EvaluationOutput` rows into a single
`RiskLabel`.

The utilities here are pure and stateless: they read `EvaluationOutput` rows and return a
label. They do not mutate `EvaluationOutput`, `EvaluationReport`, or `Experiment`, and they
do not require multi-run execution to be wired into the runner — callers invoke them only when
they already hold repeated results.
"""

from collections.abc import Sequence
from enum import Enum

from .types.evaluation import EvaluationOutput


class RiskLabel(str, Enum):
    """Stability of a `(case, evaluator)` pair across repeated runs.

    Attributes:
        BUG: Every gradable run failed. A deterministic problem worth fixing.
        FLAKY: Gradable runs were a mix of pass and fail. Non-determinism worth investigating.
        CNE: No run was gradable (all rows were `not_applicable`). Nothing to classify.
        PASS: Every gradable run passed. Stable.
    """

    BUG = "bug"
    FLAKY = "flaky"
    CNE = "cne"
    PASS = "pass"


# Worst-of ordering for rolling several evaluator verdicts up to one case verdict. A
# deterministic failure is the most actionable signal, so it dominates; a pair with nothing to
# judge (CNE) carries the least information.
_SEVERITY: dict[RiskLabel, int] = {
    RiskLabel.CNE: 0,
    RiskLabel.PASS: 1,
    RiskLabel.FLAKY: 2,
    RiskLabel.BUG: 3,
}


def classify_risk(results: Sequence[EvaluationOutput]) -> RiskLabel:
    """Classify one `(case, evaluator)` pair's stability from its repeated runs.

    Rows that had nothing to judge (`EvaluationOutput.not_applicable`) are dropped before
    classification, matching how `EvaluationReport` treats them: they are diagnoses, not
    verdicts, so they must not tip a case into pass or fail. Classification then reduces to
    whether the remaining gradable rows all passed, all failed, or were a mix.

    Args:
        results: `EvaluationOutput` rows from repeated runs of the same `(case, evaluator)`
            pair. Order does not matter.

    Returns:
        `RiskLabel.CNE` when there are no rows, or no gradable rows; `RiskLabel.PASS` when
        every gradable row passed; `RiskLabel.BUG` when every gradable row failed; otherwise
        `RiskLabel.FLAKY`.
    """
    gradable = [row for row in results if not row.not_applicable]

    if not gradable:
        return RiskLabel.CNE

    if all(row.test_pass for row in gradable):
        return RiskLabel.PASS

    if not any(row.test_pass for row in gradable):
        return RiskLabel.BUG

    return RiskLabel.FLAKY


def classify_task_risk(evaluator_risks: Sequence[RiskLabel]) -> RiskLabel:
    """Roll several evaluator verdicts on one case up to a single worst-of verdict.

    A case with one deterministic bug and two flaky evaluators is a bug: the deterministic
    failure is the actionable signal, so the most severe label wins.

    Args:
        evaluator_risks: One `RiskLabel` per evaluator that scored the case.

    Returns:
        The most severe label by `BUG > FLAKY > PASS > CNE`, or `RiskLabel.CNE` when no
        evaluator verdicts are given.
    """
    if not evaluator_risks:
        return RiskLabel.CNE

    return max(evaluator_risks, key=lambda risk: _SEVERITY[risk])
