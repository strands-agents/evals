"""The run's train / validation / test split; validation is required."""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ...case import Case
from ...types.optimization import CaseSplit

_TOLERANCE = 1e-9
_ROLES = ("train", "validation", "test")


class _DefaultTest:
    """The type of `_DEFAULT_TEST`; named in reprs, so the signature reads as what it resolves to."""

    def __repr__(self) -> str:
        return "<0.3 with shares, None with case lists>"


# `Split(test=)` not given: 0.3 with shares, no test cases with lists (None is an explicit choice).
_DEFAULT_TEST: Any = _DefaultTest()


def _is_number(value: object) -> bool:
    """Whether `value` is an int or a float (a share, not a list of cases); a bool is not a number here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True, kw_only=True)
class Split:
    """Shares or explicit case lists, not a mix; train searches, validation picks, test reports.

    Shares must sum to 1 (`test=None` counts as 0), else `ValueError` naming the sum. Case lists must be
    disjoint and give every case a role. An omitted `test` is 0.3 with shares and no test cases with lists, so
    `Split(train=train_cases, validation=validation_cases)` works. `seed=None` uses the run seed
    (`Optimizer(seed=)`), as `TPE` and `Thompson` do, so two runs with different seeds get different splits;
    `run.json` stores the seed used.
    """

    train: float | Sequence[Case] = 0.4
    validation: float | Sequence[Case] = 0.3
    test: float | Sequence[Case] | None = _DEFAULT_TEST
    seed: int | None = None

    def __post_init__(self) -> None:
        """Resolve an omitted `test`, then check the shares sum to 1 and lists aren't mixed with shares.

        Raises:
            ValueError: Shares mixed with lists, a share outside 0 to 1, a train or validation share of 0,
                shares not summing to 1, or a case listed more than once.
        """
        if self.test is _DEFAULT_TEST:
            object.__setattr__(self, "test", 0.3 if _is_number(self.train) else None)
        given = [v for v in (self.train, self.validation, self.test) if v is not None]
        shares = [_is_number(v) for v in given]
        if any(shares) and not all(shares):
            raise ValueError("Split: give every role a share or every role a list of cases, not a mix")
        if all(shares):
            train, validation, test = float(self.train), float(self.validation), float(self.test or 0.0)  # type: ignore[arg-type]
            if any(not 0 <= p <= 1 for p in (train, validation, test)):
                raise ValueError(
                    f"Split: shares are fractions of the cases and must lie in 0 to 1, got {self._describe()}; "
                    "for exact counts, pass case lists"
                )
            if train == 0 or validation == 0:
                raise ValueError(f"Split: train and validation need a share above 0, got {self._describe()}")
            total = train + validation + test
            if abs(total - 1.0) > _TOLERANCE:
                raise ValueError(
                    f"Split({self._describe()}) sums to {total:g}: pass shares that sum to 1 "
                    "(test=None means no test cases)"
                )
            return
        seen: set[str] = set()
        twice: set[str] = set()
        for role in _ROLES:
            for case in getattr(self, role) or ():
                if case.name in seen:
                    twice.add(case.name)
                seen.add(case.name)
        if twice:
            raise ValueError(f"Split: cases listed more than once: {sorted(twice)}")

    def assign(self, cases: Sequence[Case], *, run_seed: int) -> CaseSplit:
        """Assign every case a role.

        With shares: sorted by name, shuffled with `random.Random(seed)` (`run_seed` when `seed` is None),
        counts by largest remainder. With lists: as given, each list in its own order.
        0 test cases is allowed (no test report); 1 train case warns (the model's case factor drops
        out); 1-4 test cases warn (the test report is noisy).

        Args:
            cases: Every case; names are unique and not None.
            run_seed: The run's seed, used when `seed` is None.

        Returns:
            The three case-name lists; every case is in exactly one.

        Raises:
            ValueError: Fewer than 1 train or 1 validation case, naming the shortfall; with lists, a listed
                case not among `cases` or a case given no role; a missing or repeated case name.
        """
        names = [case.name for case in cases if case.name is not None]
        if len(names) != len(cases) or len(set(names)) != len(names):
            raise ValueError("Split: the cases need unique, non-None names")
        if _is_number(self.train):
            split = self._by_share(names, self.seed if self.seed is not None else run_seed)
        else:
            split = self._by_list(names)
        for role in ("train", "validation"):
            if not getattr(split, role):
                raise ValueError(
                    f"Split: {self._describe()} gives 0 {role} cases of {len(names)}; at least 1 is needed"
                )
        if len(split.train) == 1:
            warnings.warn(
                "Split: 1 train case; the score model can't tell cases apart, so effects are less certain",
                UserWarning,
                stacklevel=2,
            )
        if 1 <= len(split.test) <= 4:
            warnings.warn(
                f"Split: {len(split.test)} test case(s); the test report's interval will be wide",
                UserWarning,
                stacklevel=2,
            )
        return split

    def _by_share(self, names: list[str], seed: int) -> CaseSplit:
        """Deal the cases by share: sorted by name, shuffled with `seed`, counted by largest remainder.

        Sorting first makes the split independent of the order the cases came in. Each role gets the floor of
        share x n, and the cases left over go to the roles with the largest fractional parts, so the counts
        sum to n and each is within 1 of share x n.

        Args:
            names: Every case name, unique.
            seed: `Split.seed`, or the run seed when it is None.

        Returns:
            The cases by role, each in shuffled order; test may be empty.
        """
        order = sorted(names)
        random.Random(seed).shuffle(order)
        n = len(order)
        shares = [float(self.train), float(self.validation), float(self.test or 0.0)]  # type: ignore[arg-type]
        exact = [s * n for s in shares]
        counts = [math.floor(x) for x in exact]
        by_remainder = sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)
        for i in by_remainder[: n - sum(counts)]:
            counts[i] += 1
        n_train, n_validation = counts[0], counts[1]
        return CaseSplit(
            train=tuple(order[:n_train]),
            validation=tuple(order[n_train : n_train + n_validation]),
            test=tuple(order[n_train + n_validation :]),
        )

    def _by_list(self, names: list[str]) -> CaseSplit:
        """Take the roles from the case lists, each in the order given.

        `__post_init__` already checked that no case is listed twice.

        Args:
            names: Every case name in the experiment.

        Returns:
            The cases by role; test is empty when `test` is None.

        Raises:
            ValueError: A listed case not in the experiment, or an experiment case in no list.
        """
        roles = {role: tuple(case.name for case in getattr(self, role) or ()) for role in _ROLES}
        listed = {name for role_names in roles.values() for name in role_names}
        outside = sorted(listed - set(names))
        if outside:
            raise ValueError(f"Split: cases not in the experiment: {outside}")
        unassigned = sorted(set(names) - listed)
        if unassigned:
            raise ValueError(
                f"Split: cases {unassigned} have no role; give them one or leave them out of the experiment"
            )
        return CaseSplit(**roles)

    def _describe(self) -> str:
        """The roles for error messages: `train=0.4` for a share, `train=<5 cases>` for a list."""

        def role(value: object) -> str:
            return f"{value:g}" if _is_number(value) else f"<{len(value)} cases>"  # type: ignore[arg-type]

        parts = [f"train={role(self.train)}", f"validation={role(self.validation)}"]
        parts.append("test=None" if self.test is None else f"test={role(self.test)}")
        return ", ".join(parts)
