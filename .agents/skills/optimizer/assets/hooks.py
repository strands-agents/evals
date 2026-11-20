"""A `--search` object for `strands-evals optimize run` that measures only the baseline (0 iterations).

`--max-iterations` must be at least 1, so pass `--search <this file>:baseline_only` instead.
"""

from typing import ClassVar

from strands_evals.optimization import Schedule, ScheduleContext, Search


class _BaselineOnly(Schedule):
    """Plans no step, so the run scores the baseline on train and goes straight to the end phase."""

    name: ClassVar[str] = "baseline_only"

    def next(self, ctx: ScheduleContext) -> None:
        return None


baseline_only = Search([], schedule=_BaselineOnly())
