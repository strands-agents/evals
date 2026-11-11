"""Blocks (a proposer owning some variables), schedules (what each iteration does), and the search that holds them.

One module, so `Schedule.next`'s hints resolve without an import cycle.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from ...types.optimization import IterationRecord, Values
from ..declaration.space import ParamInfo, Space
from .proposer import Proposer
from .reflection import Reflection
from .tpe import TPE

logger = logging.getLogger(__name__)


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

        Raises:
            TypeError: `proposer` is not a `Proposer`, or a variable or `name` is not a string.
            ValueError: No variables, or an empty `name`.
        """
        if not isinstance(proposer, Proposer):
            raise TypeError(
                f"Block(proposer, *variables): the proposer comes first, as in Block(TPE(), 'temperature'); "
                f"got {proposer!r}"
            )
        if not variables:
            raise ValueError("a block needs at least one variable: a declared name or a flat key")
        not_names = [v for v in variables if not isinstance(v, str)]
        if not_names:
            raise TypeError(f"Block variables are declared names or flat keys (strings), not {not_names!r}")
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Block(name=...) must be a string, not {name!r}")
        if name == "":
            raise ValueError("Block(name=...) must not be empty")
        self.proposer = proposer
        self.variables = tuple(variables)
        self.name = name if name is not None else type(proposer).__name__

    def __repr__(self) -> str:
        return f"Block({type(self.proposer).__name__}(), {', '.join(map(repr, self.variables))}, name={self.name!r})"


@dataclass(frozen=True)
class Step:
    """What the next iteration does; planned by the schedule, run by the optimizer.

    Attributes:
        blocks: Who proposes. With several blocks, candidate i merges patch i of every block.
        n: Candidates in the step (`proposals_per_iteration`; a round's opener: 1).
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
        past: Completed iterations, oldest first; one per planned step, a step that saved nothing included.
        blocks: The resolved blocks, the default ones included.
        keys: Block name -> its expanded keys (from `Search.resolve`), so `Nested` can narrow to blocks
            with an active key under the opener's values (`Block.variables` holds names as declared).
        space: Narrows blocks to those active under an opener's values.
        values: Candidate id -> values, every saved candidate.
        eligible: Ids eligible now (a re-scored candidate can become eligible later).
        ranking: Eligible ids in ranking order (the incumbent first).
        proposals_per_iteration: A step's candidates (an opener's: 1).
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
    """Plans each iteration from the run's history.

    `next` must be a pure function of its context (no state kept between calls), so a resumed run, given the
    same history, plans the same steps.

    Attributes:
        name: Stored in `run.json` with `config()`, compared on resume.
    """

    name: ClassVar[str]

    @abstractmethod
    def next(self, ctx: ScheduleContext) -> Step | None:
        """The next step; None means nothing is left to try (`stop_reason="exhausted"`), or, inside a
        `Nested` round, that the round is over."""

    def config(self) -> dict[str, Any]:
        """JSON-serializable constructor arguments, stored in `run.json`; default `{}`."""
        return {}


class Joint(Schedule):
    """Every block, every iteration: candidate i of a step merges patch i of every block.

    Ends after a step that saved no candidate.
    """

    name: ClassVar[str] = "joint"

    def next(self, ctx: ScheduleContext) -> Step | None:
        if not ctx.blocks or (ctx.past and not _saved(ctx.past[-1])):
            return None
        return Step(tuple(ctx.blocks), ctx.proposals_per_iteration, None, None, "inner", ())


class Alternate(Schedule):
    """One block per iteration, in block order, so each candidate changes one block. No arguments.

    Ends after a full cycle of steps (one per block) that saved no candidate.
    """

    name: ClassVar[str] = "alternate"

    def next(self, ctx: ScheduleContext) -> Step | None:
        if not ctx.blocks:
            return None
        misses = 0
        for record in reversed(ctx.past):
            if _saved(record):
                break
            misses += 1
        if misses >= len(ctx.blocks):
            return None
        block = ctx.blocks[len(ctx.past) % len(ctx.blocks)]
        return Step((block,), ctx.proposals_per_iteration, None, None, "inner", ())


class Nested(Schedule):
    """Rounds over a structural block, with an inner schedule inside each round.

    Each round opens with one candidate that changes only `outer`, from the incumbent (`ctx.ranking[0]`);
    then up to `inner_iterations` inner steps that save a candidate (the inner schedule's blocks, narrowed to
    those with an active key in `ctx.keys` under the opener's values) whose parents are the round's own
    eligible candidates (the incumbent if none is). An opener repeating a measured candidate opens the round
    on it at no cost. At the round's end the outer proposer observes the round's best candidate
    (`Observation.source_id`); an opener that isn't eligible is observed as None with outcome `ineligible`,
    and the round's inner steps are skipped, as they are when the opener saved nothing (every draw invalid).
    A round also ends when the inner schedule returns None. Rounds stop after `ctx.redraws + 1` in a row
    measure nothing new (an opener that saved nothing counts as one); then `next()` returns None. When
    `outer` is not among `ctx.blocks` (an enclosing round narrowed it out), the inner schedule runs alone.
    Nested inner schedules add one entry to `Step.round` per level.

    Attributes:
        outer: The outer block's name.
        inner: The schedule inside a round.
        inner_iterations: Inner steps per round.
    """

    name: ClassVar[str] = "nested"

    def __init__(self, outer: Block | str, inner: Schedule | None = None, *, inner_iterations: int = 4):
        """Configure the rounds.

        Args:
            outer: The structural block (or its name) each round opens on.
            inner: The schedule inside a round; None means `Joint()`.
            inner_iterations: Inner steps per round.

        Raises:
            TypeError: `outer` is not a `Block` or a block name, or `inner` is not a `Schedule`.
            ValueError: `inner_iterations` is negative.
        """
        if isinstance(outer, Block):
            outer = outer.name
        if not isinstance(outer, str) or not outer:
            raise TypeError(f"Nested(outer=...) takes a Block or a block name, not {outer!r}")
        if inner is not None and not isinstance(inner, Schedule):
            raise TypeError(f"Nested(inner=...) takes a Schedule, not {inner!r}")
        if not isinstance(inner_iterations, int) or isinstance(inner_iterations, bool) or inner_iterations < 0:
            raise ValueError(f"Nested(inner_iterations=...) must be an int >= 0, not {inner_iterations!r}")
        self.outer = outer
        self.inner = inner if inner is not None else Joint()
        self.inner_iterations = inner_iterations

    def next(self, ctx: ScheduleContext) -> Step | None:
        blocks = {block.name: block for block in ctx.blocks}
        if self.outer not in blocks:
            return self.inner.next(ctx)
        rounds: dict[int, list[IterationRecord]] = {}
        for record in ctx.past:
            if record.round:
                rounds.setdefault(record.round[0], []).append(record)
        if not rounds:
            return self._opener(ctx, 0, blocks[self.outer])
        current = max(rounds)
        records = rounds[current]
        opener = records[0]
        if opener.role != "opener" or opener.round != (current,):
            raise ValueError(f"round {current} of Nested({self.outer!r}) has no opener as its first iteration")
        opener_id = next((p.candidate_id for p in reversed(opener.proposals) if p.outcome != "invalid"), None)
        if opener_id is not None and opener_id in ctx.eligible:
            steps = sum(1 for record in records[1:] if record.role == "inner" and _saved(record))
            if steps < self.inner_iterations:
                sub = self._round_context(ctx, records, opener_id)
                step = self.inner.next(sub)
                if step is not None:
                    return dataclasses.replace(
                        step,
                        parents=step.parents if step.parents is not None else tuple(sub.ranking),
                        round=(current, *(step.round or ())),
                    )
        empty = 0
        for r in sorted(rounds, reverse=True):
            if any(_saved(record) for record in rounds[r]):
                break
            empty += 1
        if empty > ctx.redraws:
            return None
        return self._opener(ctx, current + 1, blocks[self.outer])

    def config(self) -> dict[str, Any]:
        """The outer block's name, the inner schedule's name and config, and `inner_iterations`."""
        return {
            "outer": self.outer,
            "inner": {"name": self.inner.name, "config": self.inner.config()},
            "inner_iterations": self.inner_iterations,
        }

    def _opener(self, ctx: ScheduleContext, round_index: int, outer: Block) -> Step:
        return Step((outer,), 1, (ctx.ranking[0],), (round_index,), "opener", (self.outer,))

    def _round_context(
        self, ctx: ScheduleContext, records: Sequence[IterationRecord], opener_id: str
    ) -> ScheduleContext:
        """The inner schedule's view of the round: its own steps, the blocks active under the opener, and
        the round's eligible candidates (else the enclosing incumbent)."""
        opened = ctx.values[opener_id]
        blocks = [
            block
            for block in ctx.blocks
            if block.name != self.outer and any(ctx.space.is_active(key, opened) for key in ctx.keys[block.name])
        ]
        ids = {opener_id} | {
            p.candidate_id
            for record in records[1:]
            for p in record.proposals
            if p.candidate_id is not None and p.outcome not in ("invalid", "duplicate")
        }
        ranking = [i for i in ctx.ranking if i in ids] or list(ctx.ranking[:1])
        past = [record.model_copy(update={"round": (record.round or ())[1:] or None}) for record in records[1:]]
        return dataclasses.replace(ctx, past=past, blocks=blocks, ranking=ranking)


class Search:
    """Which proposer changes which variables, and the schedule the blocks propose in.

    Not to be confused with `Selection`, which picks the parent each iteration.

    Attributes:
        blocks: The user's blocks, in order.
        schedule: Plans each iteration; None means the default schedule (one `Nested` per structural block
            around `within`, or `Joint()` with no structural block).
        within: The default schedule's inner schedule; None means `Alternate()`.
        redraws: Caps re-asks per block per iteration after a duplicate (of an eligible candidate), an
            invalid patch, a patch touching keys outside its block, or no patches at all.
    """

    blocks: tuple[Block, ...]
    schedule: Schedule | None
    within: Schedule | None
    redraws: int

    def __init__(
        self,
        blocks: Sequence[Block] = (),
        *,
        schedule: Schedule | None = None,
        within: Schedule | None = None,
        redraws: int = 3,
    ):
        """Two blocks with the same name (e.g. two unnamed TPE blocks) raise `ValueError`.

        Args:
            blocks: The user's blocks, in order.
            schedule: Plans each iteration; None means the default schedule.
            within: What the default schedule runs inside its rounds, e.g. `Joint()` to change every
                block on every inner step; None means `Alternate()`.
            redraws: Re-asks per block per iteration after an unusable proposal.

        Raises:
            TypeError: A block is not a `Block`, or `schedule` or `within` is not a `Schedule`.
            ValueError: Two blocks share a name, `redraws` is negative, or both `schedule` and `within` are
                given (`within` shapes only the default schedule).
        """
        blocks = tuple(blocks)
        not_blocks = [b for b in blocks if not isinstance(b, Block)]
        if not_blocks:
            raise TypeError(f"Search(blocks=...) takes Block objects, not {not_blocks!r}")
        seen: set[str] = set()
        for block in blocks:
            if block.name in seen:
                raise ValueError(f"two blocks are named {block.name!r}; give one of them Block(..., name=...)")
            seen.add(block.name)
        if schedule is not None and not isinstance(schedule, Schedule):
            raise TypeError(f"Search(schedule=...) takes a Schedule, not {schedule!r}")
        if within is not None and not isinstance(within, Schedule):
            raise TypeError(f"Search(within=...) takes a Schedule, not {within!r}")
        if schedule is not None and within is not None:
            raise ValueError(
                "Search(within=...) shapes only the default schedule; with schedule=... given, put the inner "
                "schedule inside it"
            )
        if not isinstance(redraws, int) or isinstance(redraws, bool) or redraws < 0:
            raise ValueError(f"Search(redraws=...) must be an int >= 0, not {redraws!r}")
        self.blocks = blocks
        self.schedule = schedule
        self.within = within
        self.redraws = redraws

    def resolve(self, space: Space) -> tuple[list[tuple[Block, list[str]]], Schedule]:
        """Each block with its expanded keys, and the schedule to run them with.

        A variable that is a declared name expands to every key under it; otherwise it must be a flat key.
        A key named directly (the declared name of a plain marker, or a flat key) goes to that block even
        when another block's declared name reaches it, so `Block(TPE(), "manager")` next to
        `Block(TPE(), "manager.window.window_size")` gives the first block the rest of `manager`.
        Keys in no block go to their kind's default block, key by key: text/file keys ->
        `Reflection()` ("text"); a `choice()`'s selector key -> `TPE(n_startup_trials=2)` ("structure");
        `int`, `float`, and `one_of` keys, those under a choice's alternatives included -> `TPE()`
        ("numbers"). So `manager` goes to "structure" and `manager.window.window_size` to "numbers". A
        choice key with one option (a bare `component()`) goes to no block, since nothing can change it.
        Default blocks are appended after the user's blocks, in the order their first key is declared, so
        `Search()` is the default search. `Thompson` is never a default block.

        The schedule is `self.schedule`, else the default built over every resolved block, defaults
        included: one `Nested` per structural block (one holding a choice key with 2+ options), outermost
        first (a block whose keys exist only under another block's choice comes after it), around
        `self.within` (default `Alternate()`); when every block is structural, the innermost isn't nested;
        `Joint()` when there is no structural block.

        Raises:
            ValueError: A variable that is neither a declared name nor a key; a key two blocks name the
                same way (both directly, or both through a declared name); a block left with no key; a
                variable whose kind the block's proposer can't propose (not in `proposer.kinds`, e.g. text
                given to `TPE`), naming the key and the block; a default block whose name a user block
                already has ("text", "structure", "numbers"), suggesting `Block(..., name=)`; or a given
                `Nested` whose outer block is outside its scope, or whose outer keys exist only under a
                choice in one of its inner blocks.
        """
        params = space.params()
        order = {info.key: i for i, info in enumerate(params)}
        by_name: dict[str, list[str]] = {}
        for info in params:
            by_name.setdefault(info.name, []).append(info.key)
        claims: dict[str, tuple[int, bool]] = {}  # key -> (block index, named directly)
        for i, block in enumerate(self.blocks):
            for variable in block.variables:
                expanded = by_name.get(variable) or ([variable] if variable in order else None)
                if expanded is None:
                    raise ValueError(
                        f"block {block.name!r} names {variable!r}, which is neither a declared name nor a key; "
                        f"declared names: {sorted(by_name)}"
                    )
                for key in expanded:
                    direct = key == variable
                    held = claims.get(key)
                    if held is None or (direct and not held[1]):
                        claims[key] = (i, direct)
                    elif held[0] != i and held[1] == direct:
                        raise ValueError(
                            f"{key} is in blocks {self.blocks[held[0]].name!r} and {block.name!r}; give it one block"
                        )
        owner = {key: self.blocks[i].name for key, (i, _) in claims.items()}
        resolved: list[tuple[Block, list[str]]] = []
        for i, block in enumerate(self.blocks):
            ordered = [info.key for info in params if claims.get(info.key, (None,))[0] == i]
            if not ordered:
                raise ValueError(f"block {block.name!r} holds no variable of its own (other blocks name all of them)")
            for key in ordered:
                kind = space.param(key).kind
                if kind not in block.proposer.kinds:
                    raise ValueError(
                        f"block {block.name!r} holds {key} ({kind}), which {type(block.proposer).__name__} can't "
                        f"propose; put it in another block"
                    )
            resolved.append((block, ordered))
        defaults: dict[str, list[str]] = {}
        for info in params:
            if info.key in owner:
                continue
            if info.kind == "text":
                defaults.setdefault("text", []).append(info.key)
            elif info.kind == "choice":
                if _is_structural(info):
                    defaults.setdefault("structure", []).append(info.key)
            else:
                defaults.setdefault("numbers", []).append(info.key)
        for group, keys in defaults.items():
            if any(block.name == group for block in self.blocks):
                raise ValueError(
                    f"{keys} need the default block {group!r}, but one of your blocks has that name; rename it "
                    f"with Block(..., name=...)"
                )
            proposer: Proposer = (
                Reflection() if group == "text" else TPE(n_startup_trials=2) if group == "structure" else TPE()
            )
            logger.info("default block %r (%s) for %s", group, type(proposer).__name__, keys)
            resolved.append((Block(proposer, *keys, name=group), keys))
        block_of = {key: block.name for block, keys in resolved for key in keys}
        keys_of = {block.name: keys for block, keys in resolved}
        if self.schedule is None:
            return resolved, _default_schedule(resolved, space, block_of, self.within)
        _check(self.schedule, [block.name for block, _ in resolved], keys_of, space, block_of)
        return resolved, self.schedule


def _saved(record: IterationRecord) -> bool:
    """Whether a step saved a new candidate (a duplicate or an invalid patch saves none)."""
    return any(p.outcome not in ("invalid", "duplicate") for p in record.proposals)


def _is_structural(info: ParamInfo) -> bool:
    return info.kind == "choice" and len(info.options or ()) >= 2


def _selectors(space: Space, key: str) -> list[str]:
    """The choice keys `key` exists under, innermost first."""
    out = []
    condition = space.param(key).active_when
    while condition is not None:
        out.append(condition[0])
        condition = space.param(condition[0]).active_when
    return out


def _default_schedule(
    resolved: Sequence[tuple[Block, list[str]]],
    space: Space,
    block_of: Mapping[str, str],
    within: Schedule | None,
) -> Schedule:
    structural = [(block, keys) for block, keys in resolved if any(_is_structural(space.param(key)) for key in keys)]
    after = {
        block.name: {block_of[s] for key in keys for s in _selectors(space, key) if s in block_of} - {block.name}
        for block, keys in structural
    }
    levels: list[Block] = []
    while len(levels) < len(structural):
        placed = {block.name for block in levels}
        ready = [block for block, _ in structural if block.name not in placed and after[block.name] <= placed]
        if not ready:
            raise ValueError(f"the structural blocks' choices depend on each other in a cycle: {after}")
        levels.append(ready[0])
    if len(structural) == len(resolved):
        levels = levels[:-1]
    schedule: Schedule = (within if within is not None else Alternate()) if levels else Joint()
    for block in reversed(levels):
        schedule = Nested(block, schedule)
    return schedule


def _check(
    schedule: Schedule,
    scope: Sequence[str],
    keys_of: Mapping[str, Sequence[str]],
    space: Space,
    block_of: Mapping[str, str],
) -> None:
    """Raise when a given `Nested` can't run over the blocks in its scope."""
    if not isinstance(schedule, Nested):
        return
    if schedule.outer not in scope:
        raise ValueError(f"Nested(outer={schedule.outer!r}): no such block in its scope; the blocks are {list(scope)}")
    inner = [name for name in scope if name != schedule.outer]
    for key in keys_of[schedule.outer]:
        for selector in _selectors(space, key):
            if block_of.get(selector) in inner:
                raise ValueError(
                    f"Nested(outer={schedule.outer!r}): {key} exists only for some values of {selector}, which is in "
                    f"the inner block {block_of[selector]!r}; search {block_of[selector]!r} in an enclosing round"
                )
    _check(schedule.inner, inner, keys_of, space, block_of)
