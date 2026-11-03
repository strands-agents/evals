"""The planned train / validation / test split: sorted by name, shuffled by seed, counts by largest remainder."""

import math
import random
from collections.abc import Sequence

from ...types.optimization import CaseSplit

DEFAULT_SHARES = (0.4, 0.3, 0.3)
_SUM_TOLERANCE = 1e-9


def parse_shares(text: str) -> tuple[float, float, float]:
    """Parse `TRAIN,VALIDATION[,TEST]`; the shares must sum to 1.

    Raises:
        ValueError: The text has the wrong form, a share is negative, or the sum is not 1.
    """
    try:
        shares = [float(part) for part in text.split(",")]
    except ValueError as exc:
        raise ValueError(f"--split must be TRAIN,VALIDATION[,TEST] numbers, got {text!r}") from exc
    if len(shares) not in (2, 3) or any(not math.isfinite(share) or share < 0 for share in shares):
        raise ValueError(f"--split must be two or three shares >= 0, got {text!r}")
    train, validation, test = (*shares, 0.0) if len(shares) == 2 else shares
    if abs(train + validation + test - 1.0) > _SUM_TOLERANCE:
        raise ValueError(f"--split shares must sum to 1, got {train + validation + test:g}")
    return train, validation, test


def assign_split(names: Sequence[str], shares: tuple[float, float, float], seed: int) -> CaseSplit:
    """Give every case one role; test may be empty.

    Raises:
        ValueError: The split leaves no train case or no validation case.
    """
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    train, validation, _ = _largest_remainder(len(ordered), shares)
    if train < 1 or validation < 1:
        raise ValueError(
            f"the split gives {train} train and {validation} validation case(s) from {len(ordered)} case(s); "
            "each role needs at least 1: add cases or change --split"
        )
    return CaseSplit(
        train=tuple(ordered[:train]),
        validation=tuple(ordered[train : train + validation]),
        test=tuple(ordered[train + validation :]),
    )


def _largest_remainder(total: int, shares: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [total * share for share in shares]
    counts = [math.floor(value) for value in raw]
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]
