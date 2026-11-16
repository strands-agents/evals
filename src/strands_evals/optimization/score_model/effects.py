"""Per-option effects from a fit, and `effects.json`."""

from __future__ import annotations

from typing import Any

from ..scoring.objectives import Criteria
from .protocols import Fit


def effects(fit: Fit) -> list[dict[str, Any]]:
    """Per goal metric and option (choice option, number bin, text rewritten vs default).

    Each row: `metric`, `key`, `level`, `reference`, `mean`, `ci90`, `p_positive`. A `BayesBenchModel`
    fit only in v1 (`Fit` has no effects method); another model's fit gives `[]`.

    Args:
        fit: The search fit.
    """
    raise NotImplementedError


def effects_json(fit: Fit | None, criteria: Criteria, *, error: str | None = None) -> dict[str, Any]:
    """What `<run>/effects.json` holds at the end.

    `{"effects": effects(fit), "factors", "omitted" (design(fit)), "metrics": each goal metric's direction,
    scaling, and observations, "reason": None}`. With no fit, a last refit that raised (`error`), or any
    metric flagged, `effects`, `factors`, and `omitted` are empty and `reason` says why: "no fit: fewer
    than 2 eligible candidates", "the model raised: ...", "flagged: ...", or, for a fit from another
    score model than `BayesBenchModel`, "effects need a BayesBenchModel fit".

    Args:
        fit: The search fit, or None.
        criteria: Gives each goal metric's direction.
        error: The last refit's error, if it raised.
    """
    raise NotImplementedError
