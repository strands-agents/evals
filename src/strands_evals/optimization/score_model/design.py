"""Candidates to factors: what the effects are measured against."""

from __future__ import annotations

from typing import Any

from .protocols import Fit


def design(fit: Fit) -> dict[str, Any]:
    """The factors and omitted keys of a `BayesBenchModel` fit; both lists empty for another model's fit.

    `{"factors": [{key, kind ("choice" | "range" | "text"), levels, reference}], "omitted": [{key, why}]}`.
    Levels are a choice's options, a number's bins as "(low, high]", or "rewritten" / "default" for a text
    block, plus an "inactive" level for a key that exists only under some choices. Omitted keys are left
    out of the design, e.g. "one value" when every fitted candidate has the same value.

    Args:
        fit: The fit to describe.
    """
    raise NotImplementedError
