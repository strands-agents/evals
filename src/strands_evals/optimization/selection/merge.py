"""Variable-wise merge of two frontier members."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ...types.optimization import MergePlan, Value


def find_merge(
    front: Mapping[str, frozenset[str]],
    values: Mapping[str, Mapping[str, Value]],
    parents: Mapping[str, tuple[str, ...]],
    tried: frozenset[frozenset[str]],
    name_of: Callable[[str], str],
) -> MergePlan | None:
    """The first untried pair that changed disjoint variables since their latest common ancestor.

    A choice and its sub-keys move as one unit. Pairs are tried by selection cases covered, then
    pool order.

    Args:
        front: Member id -> the cases it leads.
        values: Candidate id -> key -> value.
        parents: Candidate id -> parent ids.
        tried: Pairs already merged.
        name_of: Returns a key's declared name (the merge unit).
    """
    raise NotImplementedError
