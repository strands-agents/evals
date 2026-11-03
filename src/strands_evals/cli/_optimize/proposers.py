"""Proposers for the interim engine: `Reflection` (a model), a proposal file, and `Random` (a stand-in for `TPE`)."""

import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from strands import Agent

from ...optimization.declaration.space import ParamInfo
from ...types.optimization import Patch, Usage, Value
from .reflection_v0 import SYSTEM_PROMPT
from .scoring import zero_usage
from .space import CliSpace

SIGNIFICANT_DIGITS = 4


@dataclass(frozen=True)
class ProposalContext:
    """What a proposer sees for one turn: the parent, the block's keys, and train feedback."""

    space: CliSpace
    keys: tuple[str, ...]
    parent: Mapping[str, Value]
    parent_number: int
    count: int
    iteration: int
    consumed: int
    feedback: dict[str, Any]


@dataclass(frozen=True)
class ProposalBatch:
    """The patches of one turn and what the proposer spent."""

    patches: list[Patch]
    usage: Usage


class Proposer(Protocol):
    """One search block: a name (the candidates' `origin`), its keys, and how it proposes."""

    @property
    def name(self) -> str:
        """The block name, recorded as each candidate's `origin`."""
        ...

    @property
    def kind(self) -> str:
        """The proposer type, for example `Reflection`."""
        ...

    @property
    def keys(self) -> tuple[str, ...]:
        """The keys that this block may change."""
        ...

    def config(self) -> dict[str, Any]:
        """What `run.json` stores and resume compares."""
        ...

    def propose(self, context: ProposalContext) -> ProposalBatch:
        """Return up to `context.count` patches; an empty list means nothing left to propose."""
        ...


@dataclass(frozen=True)
class ReflectionProposer:
    """Asks a model for new texts from the contrast between passed and failed train cases."""

    model: str
    keys: tuple[str, ...]
    name: str = "text"
    kind: str = "Reflection"

    def config(self) -> dict[str, Any]:
        """The proposer and its model."""
        return {"block": self.name, "proposer": self.kind, "keys": list(self.keys), "model": self.model}

    def propose(self, context: ProposalContext) -> ProposalBatch:
        """One model call that returns `count` patches.

        Raises:
            ValueError: The reply is not the requested JSON.
        """
        agent = Agent(model=self.model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
        started = time.monotonic()
        result = agent(json.dumps(_request(context), ensure_ascii=False))
        patches = parse_patches(str(result), context.count)
        return ProposalBatch(patches=patches, usage=_usage(result, self.model, time.monotonic() - started))


@dataclass(frozen=True)
class FileProposer:
    """Returns fixed patches from a file in order, `count` per turn, for offline runs."""

    patches: tuple[Patch, ...]
    keys: tuple[str, ...]
    source: str
    name: str = "text"
    kind: str = "ProposalFile"

    def config(self) -> dict[str, Any]:
        """The proposer, its file, and the patches."""
        patches = [patch.model_dump(mode="json") for patch in self.patches]
        return {
            "block": self.name,
            "proposer": self.kind,
            "keys": list(self.keys),
            "file": self.source,
            "patches": patches,
        }

    def propose(self, context: ProposalContext) -> ProposalBatch:
        """The next patches after the `consumed` ones."""
        end = context.consumed + context.count
        return ProposalBatch(patches=list(self.patches[context.consumed : end]), usage=zero_usage())


@dataclass(frozen=True)
class RandomProposer:
    """Draws numbers and choices at random, seeded per turn; replace with `TPE` when it lands."""

    keys: tuple[str, ...]
    seed: int
    name: str = "values"
    kind: str = "Random"

    def config(self) -> dict[str, Any]:
        """The proposer and its seed."""
        return {"block": self.name, "proposer": self.kind, "keys": list(self.keys), "seed": self.seed}

    def propose(self, context: ProposalContext) -> ProposalBatch:
        """`count` patches; each draws every key of the block."""
        generator = random.Random(f"{self.seed}:{context.iteration}:{self.name}")
        params = [context.space.param(key) for key in self.keys]
        patches = [
            Patch(updates={param.key: _draw(param, generator) for param in params}, rationale="random draw")
            for _ in range(context.count)
        ]
        return ProposalBatch(patches=patches, usage=zero_usage())


def load_proposal_file(path: Path, space: CliSpace) -> tuple[Patch, ...]:
    """Read patches: a JSON list of `{"updates": {...}, "rationale": "..."}`, or texts for a lone text key.

    A plain-text file, or a JSON list of strings, needs exactly one text key in the space.

    Raises:
        ValueError: The file is empty, has the wrong shape, or gives texts when the space has several text keys.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".json":
        return (_text_patch(text, space, 1),)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--proposal-file is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("--proposal-file JSON must be a nonempty list")
    return tuple(
        _text_patch(item, space, number) if isinstance(item, str) else _patch(item, space, number)
        for number, item in enumerate(data, start=1)
    )


def parse_patches(reply: str, count: int) -> list[Patch]:
    """Read `{"patches": [...]}` from a model reply, without code fences; keep at most `count`.

    Raises:
        ValueError: The reply is not that JSON shape.
    """
    body = reply.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ValueError(f"proposer reply is not JSON: {exc}") from exc
    items = data.get("patches") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("proposer reply must be an object with a 'patches' list")
    patches = []
    for item in items[:count]:
        updates = item.get("updates") if isinstance(item, dict) else None
        if not isinstance(updates, dict) or not all(isinstance(value, str) for value in updates.values()):
            raise ValueError("each proposed patch needs 'updates' with text values")
        patches.append(Patch(updates=updates, rationale=str(item.get("rationale", "")).strip()))
    return patches


def _request(context: ProposalContext) -> dict[str, Any]:
    editable = [
        {
            "key": key,
            "text": context.parent[key],
            "size_cap": context.space.param(key).size_cap,
            "edit_instructions": context.space.param(key).edit_instructions,
        }
        for key in context.keys
    ]
    return {
        "editable_keys": editable,
        "count": context.count,
        "current_version": context.parent_number,
        **context.feedback,
    }


def _usage(result: Any, model: str, seconds: float) -> Usage:
    metrics = getattr(result, "metrics", None)
    usage = getattr(metrics, "accumulated_usage", None) or {}
    total = int(usage.get("totalTokens", 0))
    output = int(usage.get("outputTokens", 0))
    return Usage(
        tokens=total,
        input_tokens=total - output,
        output_tokens=output,
        cache_read_tokens=usage.get("cacheReadInputTokens"),
        cache_write_tokens=usage.get("cacheWriteInputTokens"),
        model_calls=int(getattr(metrics, "cycle_count", 0) or 0),
        seconds=seconds,
        models=[model],
    )


def _draw(param: ParamInfo, generator: random.Random) -> Value:
    if param.kind == "one_of":
        return generator.choice(list(param.options or ()))
    low, high = float(param.min_value or 0), float(param.max_value or 0)
    if param.kind == "int":
        if param.log:
            return min(int(high), max(int(low), round(math.exp(generator.uniform(math.log(low), math.log(high))))))
        return generator.randint(int(low), int(high))
    value = math.exp(generator.uniform(math.log(low), math.log(high))) if param.log else generator.uniform(low, high)
    return min(high, max(low, float(f"{value:.{SIGNIFICANT_DIGITS}g}")))


def _text_patch(text: str, space: CliSpace, number: int) -> Patch:
    if len(space.text_keys) != 1:
        raise ValueError(f"--proposal-file item {number}: a bare text needs exactly one text key in the space")
    if not text.strip():
        raise ValueError(f"--proposal-file item {number}: the text is empty")
    return Patch(updates={space.text_keys[0]: text}, rationale=f"proposal file item {number}")


def _patch(item: Any, space: CliSpace, number: int) -> Patch:
    updates = item.get("updates") if isinstance(item, dict) else None
    if not isinstance(updates, dict) or not updates:
        raise ValueError(f"--proposal-file item {number}: needs a nonempty 'updates' object")
    normalized = {key: _normalize(space, key, value) for key, value in updates.items()}
    rationale = item.get("rationale", f"proposal file item {number}")
    return Patch(updates=normalized, rationale=str(rationale))


def _normalize(space: CliSpace, key: str, value: Any) -> Value:
    known = {param.key: param for param in space.params}
    if key in known and known[key].kind == "float" and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value  # type: ignore[no-any-return]
