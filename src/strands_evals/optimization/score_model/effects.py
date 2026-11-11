"""Per-option effects from a fit, and `effects.json`."""

from __future__ import annotations

from typing import Any

from ...types.optimization import Estimate
from ..bayesbench import linear_contrast
from ..scoring.objectives import AtMost, Criteria, Maximize, Minimize
from .bayesbench import BayesBenchFit
from .design import design
from .protocols import Fit


def effects(fit: Fit) -> list[dict[str, Any]]:
    """Per goal metric and option (choice option, number bin, text rewritten vs default).

    Each row: `metric`, `key`, `level`, `reference`, `mean`, `ci90`, `p_positive`, in the metric's units:
    the level's effect against the reference level, every other factor at its reference (a number bin's
    includes its trend between the two bins). A `BayesBenchModel` fit only (`Fit` has no effects method);
    another model's fit gives `[]`.

    Args:
        fit: The search fit.
    """
    if not isinstance(fit, BayesBenchFit):
        return []
    rows = []
    for metric, model in fit.metrics.items():
        for factor, level, contrast_row in fit.design.contrasts():
            contrast = linear_contrast(model.posterior, contrast_row)
            estimate = Estimate(mean=contrast.mean * model.unit, sd=contrast.sd * model.unit)
            rows.append(
                {
                    "metric": metric,
                    "key": factor.key,
                    "level": level,
                    "reference": factor.levels[0],
                    "mean": estimate.mean,
                    "ci90": list(estimate.ci90),
                    "p_positive": contrast.p_positive,
                }
            )
    return rows


def effects_json(fit: Fit | None, criteria: Criteria, *, error: str | None = None) -> dict[str, Any]:
    """What `<run>/effects.json` holds at the end.

    `{"effects": effects(fit), "factors", "omitted" (design(fit)), "goals", "metrics", "reason": None}`.
    `goals` lists each objective and constraint (`role`, `index`, `metric`, `better`: "higher" | "lower");
    `metrics` gives each fitted goal metric's scale (`unit`, `offset`), noise, observation count,
    held-out diagnostics, and prior scales with the weights of the likeliest grid points. With no fit, a last
    refit that raised (`error`), or any metric flagged, `effects`, `factors`, and `omitted` are empty and
    `reason` says why: "no fit: fewer than 2 eligible candidates",
    "the model raised: ...", "flagged: ...", or, for a fit from another score model than
    `BayesBenchModel`, "effects need a BayesBenchModel fit". `metrics` is filled for a flagged fit too, so
    the diagnostics show why.

    Args:
        fit: The search fit, or None.
        criteria: Gives each goal metric's direction.
        error: The last refit's error, if it raised.
    """
    objective = criteria.objective
    objectives = [objective] if isinstance(objective, (Maximize, Minimize)) else list(objective.objectives)
    goals = [
        {
            "role": "objective",
            "index": i,
            "metric": o.metric,
            "better": "higher" if isinstance(o, Maximize) else "lower",
        }
        for i, o in enumerate(objectives)
    ] + [
        {"role": "constraint", "index": i, "metric": c.metric, "better": "lower" if isinstance(c, AtMost) else "higher"}
        for i, c in enumerate(criteria.constraints)
    ]
    out: dict[str, Any] = {"effects": [], "factors": [], "omitted": [], "goals": goals, "metrics": {}, "reason": None}
    if error is not None:
        out["reason"] = f"the model raised: {error}"
        return out
    if fit is None:
        out["reason"] = "no fit: fewer than 2 eligible candidates"
        return out
    if not isinstance(fit, BayesBenchFit):
        out["reason"] = "effects need a BayesBenchModel fit"
        return out
    out["metrics"] = {
        metric: {
            "scale": d["scale"],
            "noise": d["noise"],
            "observations": d["observations"],
            "diagnostics": {k: d[k] for k in ("n_holdout", "z_mean", "z_sq_sum", "flagged", "messages") if k in d},
            "prior_scales": d["prior_scales"],
            "prior_scale_weights": d["prior_scale_weights"],
        }
        for metric, d in fit.diagnostics.items()
    }
    if fit.flagged:
        out["reason"] = f"flagged: {', '.join(fit.flagged)}"
        return out
    out["effects"] = effects(fit)
    out.update(design(fit))
    return out
