"""Candidates to factors: what the effects are measured against.

`_Design` lays out the columns of `BayesBenchModel`'s linear model at one fit, and gives each configuration's row:

1. the intercept, each choice (`OneOf`, `choice()`) by its option and each number binned at the quantiles of the
   values fitted (an "inactive" level where the space leaves the key out), with main effects and pairwise
   interactions (`bayesbench.Encoder`);
2. each text block (the texts one proposer rewrites together) by whether it is rewritten;
3. each number's linear trend over its values (their logarithms for a log-scale one), standardized, so a value
   anywhere in its bin is predicted along the trend rather than at the bin's;
4. each fitted candidate's own effect (the first, the baseline, is the reference), for what its factors leave
   unexplained (a rewrite's text, a number within its bin);
5. the case (the first is the reference), so case difficulty is shared by every candidate.

Groups 2-5 have main effects only. numpy is imported lazily: it is in the optional `optimization` extra.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ...types.optimization import Value, Values
from ..declaration.space import Space
from .protocols import Fit

INACTIVE = "inactive"


def design(fit: Fit) -> dict[str, Any]:
    """The factors and omitted keys of a `BayesBenchModel` fit; both lists empty for another model's fit.

    `{"factors": [{key, kind ("choice" | "range" | "text"), levels, reference}], "omitted": [{key, why}]}`.
    Levels are a choice's options, a number's bins as "(low, high]", or "default" / "rewritten" for a text
    block (key `rewritten:<block name>`), plus an "inactive" level for a key that exists only under some
    choices. Omitted keys are left out of the design: "one value" when every fitted candidate has the same
    value of a number, "never active" when no fitted candidate has the key.

    Args:
        fit: The fit to describe.
    """
    layout = getattr(fit, "design", None)
    if not isinstance(layout, _Design):
        return {"factors": [], "omitted": []}
    return {"factors": layout.describe(), "omitted": [dict(o) for o in layout.omitted]}


@dataclass
class _Factor:
    """One factor: a choice, a binned number, or whether a text block is rewritten; the reference level first."""

    key: str
    kind: str  # "choice" | "range" | "text"
    levels: list[str]
    text_keys: tuple[str, ...] = ()
    bins: list[float] = field(default_factory=list)  # a range's upper bin edges
    defaults: dict[str, Value] = field(default_factory=dict)  # a text block's baseline texts

    def level(self, values: Values) -> str:
        if self.kind == "text":
            return "default" if all(values.get(k) == self.defaults.get(k) for k in self.text_keys) else "rewritten"
        if self.key not in values:
            return INACTIVE
        value = values[self.key]
        if self.kind == "choice":
            return str(value)
        for i, edge in enumerate(self.bins):
            if float(cast(float, value)) <= edge:  # a range factor's key is an Int / Float
                return self.bin_name(i)
        return self.bin_name(len(self.bins) - 1)

    def bin_name(self, i: int) -> str:
        low = "-inf" if i == 0 else f"{self.bins[i - 1]:.6g}"
        return f"({low}, {self.bins[i]:.6g}]"


class _Design:
    """The factors at one fit, and each configuration's row of the design matrix.

    Attributes:
        factors: The choices and binned numbers (group 1), encoded with interactions.
        texts: The text blocks some fitted candidate rewrote (group 2).
        omitted: Keys left out, with why.
        trends: (key, log, center, spread) per number with a spread (group 3).
        bin_centers: (key, level) -> the mean trend value of the fitted values in that bin.
        cases: The case names; the first is the reference.
        fitted: Every fitted candidate's id.
        candidate_ids: The fitted candidates with an own-effect column (every one but the first).
        main_columns, pair_columns: Group 1's main-effect and interaction columns.
        n_design, n_text, p: Group 1's columns (the intercept included), group 2's, and the total.
    """

    def __init__(
        self,
        space: Space,
        text_blocks: Mapping[str, Sequence[str]],
        candidates: Mapping[str, Values],
        cases: Sequence[str],
    ) -> None:
        """Lay out the columns.

        Args:
            space: The declared space.
            text_blocks: Block name -> its text keys.
            candidates: The fitted candidates' values, in scoring order (the baseline first).
            cases: The split's case names.
        """
        from ..bayesbench import Encoder, FactorSpace  # optional extra: optimization

        fitted = list(candidates.values())
        in_texts = {k for keys in text_blocks.values() for k in keys}
        self.factors: list[_Factor] = []
        self.omitted: list[dict[str, str]] = []
        for info in space.params():
            key = info.key
            # A one-option choice (a bare component()) can never change, so it isn't a factor.
            if key in in_texts or info.kind == "text" or (info.kind == "choice" and len(info.options or ()) < 2):
                continue
            present = [v[key] for v in fitted if key in v]
            absent = [INACTIVE] if any(key not in v for v in fitted) else []
            if not present:
                self.omitted.append({"key": key, "why": "never active"})
                continue
            if info.kind in ("one_of", "choice"):
                # Every option the space offers, measured or not: one never measured stays at its prior.
                options = list(dict.fromkeys([*(str(o) for o in info.options or ()), *(str(v) for v in present)]))
                reference = str(info.default) if str(info.default) in options else options[0]
                factor = _Factor(key, "choice", [reference, *[o for o in options if o != reference], *absent])
            else:
                factor = _Factor(key, "range", [], bins=_bins([float(cast(float, v)) for v in present]))
                names = list(dict.fromkeys(factor.bin_name(i) for i in range(len(factor.bins))))
                reference = factor.level({key: info.default})
                factor.levels = [reference, *[n for n in names if n != reference], *absent]
            if len([lv for lv in factor.levels if lv != INACTIVE]) < 2 and not absent:
                self.omitted.append({"key": key, "why": "one value"})
                continue
            self.factors.append(factor)
        self.encoder = (
            Encoder(FactorSpace({f.key: f.levels for f in self.factors}), interactions=len(self.factors) >= 2)
            if self.factors
            else None
        )
        self.texts: list[_Factor] = []
        defaults = {info.key: info.default for info in space.params()}
        for name, keys in text_blocks.items():
            text = _Factor(
                f"rewritten:{name}",
                "text",
                ["default", "rewritten"],
                tuple(keys),
                defaults={k: defaults[k] for k in keys},
            )
            if any(text.level(v) == "rewritten" for v in fitted):
                self.texts.append(text)
        self.cases = list(cases)
        infos = {info.key: info for info in space.params()}
        self.trends: list[tuple[str, bool, float, float]] = []
        self.bin_centers: dict[tuple[str, str], float] = {}
        for f in self.factors:
            if f.kind != "range":
                continue
            log = infos[f.key].log
            trend = [_trend_value(v[f.key], log) for v in fitted if f.key in v]
            spread = statistics.pstdev(trend) if len(trend) >= 2 else 0.0
            if spread > 0:
                self.trends.append((f.key, log, statistics.fmean(trend), spread))
                for level in f.levels:
                    inside = [_trend_value(v[f.key], log) for v in fitted if f.key in v and f.level(v) == level]
                    if inside:
                        self.bin_centers[(f.key, level)] = statistics.fmean(inside)
        self.fitted = frozenset(candidates)
        self.candidate_ids = list(candidates)[1:]
        self.main_columns = [c for cols in self.encoder.main_columns.values() for c in cols] if self.encoder else []
        self.pair_columns = (
            [c for cols in self.encoder.interaction_columns.values() for c in cols] if self.encoder else []
        )
        self.n_design = self.encoder.p if self.encoder else 1
        self.n_text = len(self.texts)  # one column each: "rewritten"
        self.p = self.n_design + self.n_text + len(self.trends) + len(self.candidate_ids) + max(len(self.cases) - 1, 0)
        self._candidate_column = {cid: j for j, cid in enumerate(self.candidate_ids)}

    @property
    def trend_start(self) -> int:
        return self.n_design + self.n_text

    @property
    def candidate_start(self) -> int:
        return self.trend_start + len(self.trends)

    @property
    def case_start(self) -> int:
        return self.candidate_start + len(self.candidate_ids)

    @property
    def groups(self) -> list[str]:
        """The effect groups whose prior scales are fitted to the data."""
        out = []
        if self.main_columns:
            out.append("main")
        if self.pair_columns:
            out.append("pair")
        if self.candidate_ids:
            out.append("candidate")
        return out

    def row(self, values: Values, case: str | None = None, candidate: str | None = None) -> Any:
        """A configuration's row: at one case, or (`case=None`) averaged over the cases; with the candidate's own
        effect when `candidate` is a fitted one, else at what its factors predict."""
        import numpy as np  # optional extra: optimization

        x = np.zeros(self.p)
        if self.encoder:
            x[: self.n_design] = self.encoder.row(tuple(self._level(f, values) for f in self.factors))
        else:
            x[0] = 1.0
        for j, text in enumerate(self.texts):
            if text.level(values) == "rewritten":
                x[self.n_design + j] = 1.0
        for j, (key, log, center, spread) in enumerate(self.trends):
            if key in values:  # where the space leaves the number out, its bin factor's inactive level says so
                x[self.trend_start + j] = (_trend_value(values[key], log) - center) / spread
        if candidate is not None and candidate in self._candidate_column:
            x[self.candidate_start + self._candidate_column[candidate]] = 1.0
        if case is None:
            x[self.case_start :] = 1.0 / len(self.cases) if len(self.cases) > 1 else 0.0
        elif case in self.cases[1:]:
            x[self.case_start + self.cases.index(case) - 1] = 1.0
        return x

    def contrasts(self) -> Iterator[tuple[_Factor, str, Any]]:
        """Each factor level's contrast row against its reference, every other factor at its reference.

        A binned number's contrast includes its trend between the two bins' centers. The inactive level has none.
        """
        import numpy as np  # optional extra: optimization

        reference = tuple(f.levels[0] for f in self.factors)
        for factor in self.factors:
            assert self.encoder is not None  # there are factors
            for level in factor.levels[1:]:
                if level == INACTIVE:
                    continue
                a = np.zeros(self.p)
                key = tuple(level if f is factor else f.levels[0] for f in self.factors)
                a[: self.n_design] = self.encoder.row(key) - self.encoder.row(reference)
                here = self.bin_centers.get((factor.key, level))
                there = self.bin_centers.get((factor.key, factor.levels[0]))
                for j, (trend_key, _log, _center, spread) in enumerate(self.trends):
                    if trend_key == factor.key and here is not None and there is not None:
                        a[self.trend_start + j] = (here - there) / spread
                yield factor, level, a
        for j, text in enumerate(self.texts):
            a = np.zeros(self.p)
            a[self.n_design + j] = 1.0
            yield text, "rewritten", a

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"key": f.key, "kind": f.kind, "levels": list(f.levels), "reference": f.levels[0]}
            for f in [*self.factors, *self.texts]
        ]

    def _level(self, factor: _Factor, values: Values) -> str:
        level = factor.level(values)
        # A key every fitted candidate had, here inactive: at its reference.
        return level if level in factor.levels else factor.levels[0]


def _trend_value(value: Any, log: bool) -> float:
    number = float(value)
    return math.log(number) if log and number > 0 else number


def _bins(values: list[float]) -> list[float]:
    """Upper bin edges at the quantiles of the fitted values: 3 bins, 4 from 12 distinct values, each with at least
    2 values where the values allow."""
    distinct = sorted(set(values))
    if len(distinct) <= 3:
        return distinct
    n = 4 if len(distinct) >= 12 else 3
    edges = sorted({*statistics.quantiles(values, n=n, method="inclusive"), distinct[-1]})
    keep: list[float] = []
    last = -math.inf
    for edge in edges:
        count = sum(1 for v in values if last < v <= edge)
        if count >= 2 or edge == edges[-1]:
            keep.append(edge)
            last = edge
    return keep if len(keep) >= 2 else [distinct[-1]]
