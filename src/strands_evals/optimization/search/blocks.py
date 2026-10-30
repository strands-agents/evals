"""Blocks (a proposer owning some variables), schedules (what each iteration does), and the search that holds them.

One module, so `Schedule.next`'s hints resolve without an import cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from ...types.optimization import IterationRecord, Values
from ..declaration.space import Space
from .proposer import Proposer


class Block:
    """One proposer owns some variables; a top-level name expands to every key under it.

    Attributes:
        proposer: Proposes values for these variables.
        variables: The declared names or keys as given.
        name: Defaults to `type(proposer).__name__`. It keys `observe` events and the resume state,
            so it must be stable across resume.
    """

    proposer: Proposer
    variables: tuple[str, ...]
    name: str

    def __init__(self, proposer: Proposer, *variables: str, name: str | None = None):
        """Give `proposer` ownership of `variables`.

        Args:
            proposer: Proposes values for these variables.
            *variables: Declared names or flat keys.
            name: Stable block name; defaults to the proposer's class name.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Step:
    """What the next iteration does; planned by the schedule, run by the optimizer.

    Attributes:
        blocks: Who proposes.
        n: Proposals per block (`proposals_per_iteration`; a round's opener: 1).
        parents: The step's pool, the candidates selection may choose from; None = every eligible candidate.
        round: The round path, outermost `Nested` first: `(2,)` = round 2 of the only `Nested`, `(2, 0)` =
            round 0 of the inner `Nested` within the outer's round 2. None outside any `Nested`.
        role: `opener` (a round's first step) or `inner` (every other step, including steps outside any
            `Nested`).
        observe_at_round_end: Blocks whose `observe` waits for the round's end (the opener's outer block).
    """

    blocks: tuple[Block, ...]
    n: int
    parents: tuple[str, ...] | None
    round: tuple[int, ...] | None
    role: Literal["opener", "inner"]
    observe_at_round_end: tuple[str, ...]


@dataclass(frozen=True)
class ScheduleContext:
    """What a schedule plans from. A dataclass (it holds the `Space`), never saved.

    Attributes:
        past: Completed iterations, oldest first.
        blocks: The resolved blocks, the default ones included.
        keys: Block name -> its expanded keys (from `Search.resolve`), so `Nested` can narrow to blocks
            with an active key under the opener's values (`Block.variables` holds names as declared).
        space: Narrows blocks to those active under an opener's values.
        values: Candidate id -> values, every saved candidate.
        eligible: Ids eligible now (a re-scored candidate can become eligible later).
        ranking: Eligible ids in ranking order (the incumbent first).
        proposals_per_iteration: A step's proposals per block (an opener's: 1).
        redraws: `Search.redraws`; `Nested` stops after `redraws + 1` rounds in a row measure nothing new.
    """

    past: Sequence[IterationRecord]
    blocks: Sequence[Block]
    keys: Mapping[str, Sequence[str]]
    space: Space
    values: Mapping[str, Values]
    eligible: frozenset[str]
    ranking: Sequence[str]
    proposals_per_iteration: int
    redraws: int


class Schedule(ABC):
    """Plans each iteration from the run's history; deterministic, so resume replays it.

    Attributes:
        name: Stored in `run.json` with `config()`, compared on resume.
    """

    name: ClassVar[str]

    @abstractmethod
    def next(self, ctx: ScheduleContext) -> Step | None:
        """The next step; None means nothing is left to try (`stop_reason="exhausted"`)."""

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json`; default `{}`."""
        return {}


class Joint(Schedule):
    """Every block, every iteration; each patch is its own candidate changing only its block."""

    name: ClassVar[str] = "joint"

    def next(self, ctx: ScheduleContext) -> Step | None:
        raise NotImplementedError


class Alternate(Schedule):
    """One block per iteration, in block order, so each candidate's effect is attributable. No arguments."""

    name: ClassVar[str] = "alternate"

    def next(self, ctx: ScheduleContext) -> Step | None:
        raise NotImplementedError


class Nested(Schedule):
    """Rounds over a structural block, with an inner schedule inside each round.

    Each round opens with one candidate that changes only `outer`, from the incumbent (`ctx.ranking[0]`);
    then `inner_iterations` inner steps (the inner schedule's blocks, narrowed to those with an active key in
    `ctx.keys` under the opener's values) whose parents are the round's own eligible candidates (the incumbent if none
    is). An opener repeating a measured candidate opens the round on it at no cost. At the round's end the
    outer proposer observes the round's best candidate (`Observation.source_id`); an opener that isn't
    eligible is observed as None with outcome `ineligible`, and the round's inner steps are skipped. Rounds stop after
    `ctx.redraws + 1` in a row measure nothing new; then `next()` returns None. Nested inner schedules add
    one entry to `Step.round` per level.
    """

    name: ClassVar[str] = "nested"

    def __init__(self, outer: Block | str, inner: Schedule | None = None, *, inner_iterations: int = 4):
        """Configure the rounds.

        Args:
            outer: The structural block (or its name) each round opens on.
            inner: The schedule inside a round; None means `Joint()`.
            inner_iterations: Inner steps per round.
        """
        raise NotImplementedError

    def next(self, ctx: ScheduleContext) -> Step | None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        """The outer block's name, the inner schedule's name and config, and `inner_iterations`."""
        raise NotImplementedError


class Search:
    """Which proposer changes which variables, and the schedule the blocks propose in.

    Not to be confused with `Selection`, which picks the parent each iteration.

    Attributes:
        blocks: The user's blocks, in order.
        schedule: Plans each iteration; None means the default schedule (one `Nested` per structural block
            around `Alternate()`, or `Joint()` with no structural block).
        redraws: Caps re-asks per block per iteration after a duplicate (of an eligible candidate), an
            invalid patch, a patch touching keys outside its block, or no patches at all.
    """

    blocks: tuple[Block, ...]
    schedule: Schedule | None
    redraws: int

    def __init__(self, blocks: Sequence[Block] = (), *, schedule: Schedule | None = None, redraws: int = 3):
        """Two blocks with the same name (e.g. two unnamed TPE blocks) raise `ValueError`.

        Args:
            blocks: The user's blocks, in order.
            schedule: Plans each iteration; None means the default schedule.
            redraws: Re-asks per block per iteration after an unusable proposal.

        Raises:
            ValueError: Two blocks share a name.
        """
        raise NotImplementedError

    def resolve(self, space: Space) -> tuple[list[tuple[Block, list[str]]], Schedule]:
        """Each block with its expanded keys, and the schedule to run them with.

        Variables in no block go to their type's default block: text/file ->
        `Reflection()` ("text"); `choice()` -> `TPE(n_startup_trials=2)` ("structure"); `Int`, `Float`,
        `OneOf` -> `TPE()` ("numbers"); appended after the user's blocks, so `Search()` is the default
        search. `Thompson` is never a default block.

        The schedule is `self.schedule`, else the default built over every resolved block, defaults
        included: one `Nested` per structural block (one owning `choice()` variables), outermost first,
        around `Alternate()`; when every block is structural, the innermost isn't nested; `Joint()` when
        there is no structural block.

        Raises:
            ValueError: A variable in two blocks; a variable whose kind the block's proposer can't propose
                (not in `proposer.kinds`, e.g. text given to `TPE`), naming the key and the block; or a
                default block whose name a user block already has ("text", "structure", "numbers"),
                suggesting `Block(..., name=)`.
        """
        raise NotImplementedError
