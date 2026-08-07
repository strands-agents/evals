"""The types the skill extractors return."""

from __future__ import annotations

from typing import Literal, NamedTuple


class AvailableSkill(NamedTuple):
    """A skill exposed to the agent at runtime."""

    name: str
    description: str


class SkillLoadEvent(NamedTuple):
    """One load attempt, as it appeared in the trajectory.

    The harness-independent form every adapter converts its own messages into. One event per
    attempt, in trajectory order, before any judgment about what the attempts add up to: two loads
    of the same skill are two events, and a refusal followed by a retry is two events rather than
    one outcome. `InvokedSkill` is the per-skill summary built from these, so evaluators that need
    their own definition of selected, invoked, or followed read the events instead.
    """

    name: str
    # "attempted" means the call was made and the trajectory never carried its outcome, which is
    # not the same as a load that succeeded and whose body went uncaptured: one is a run we cannot
    # see the end of, the other is a run we saw succeed. "failed" is a refusal the harness reported.
    status: Literal["attempted", "loaded", "failed"]
    body: str | None = None  # SKILL.md text if captured from the trajectory, else None
    error: str | None = None  # the harness's message, on a refusal
    call_id: str | None = None  # the harness's tool-call identifier, when it keys its results
    position: int | None = None  # index of the message or span the attempt was found in
    agent_id: str | None = None  # which agent made the attempt, when the harness records it


class InvokedSkill(NamedTuple):
    """A skill the agent selected during the run, whether or not the load succeeded.

    One row per skill, folded from the `SkillLoadEvent`s for that skill: repeated loads collapse,
    the fullest body recovered wins, and one success anywhere makes the skill loaded. Read
    `extract_skill_load_events` instead where the individual attempts matter.
    """

    name: str
    body: str | None  # SKILL.md text if captured from the trajectory, else None
    # "failed" means the harness refused the load (unknown skill, sandbox error). Kept rather than
    # dropped because a refused load and no attempt at all are different runs: the agent that asked
    # for the right skill and was refused made a correct selection, and reporting it as an
    # abstention credits or blames the wrong decision.
    status: Literal["loaded", "failed"] = "loaded"
    # The harness's own refusal message, on a failed load. Which refusal it was decides what to
    # fix: "Skill 'pdf-procesing' not found. Available skills: pdf-processing" is a misspelled
    # name in the agent's call, while "Available skills: (none)" is a harness that mounted no
    # skills at all. Collapsing both into "the load failed" hides that difference from whoever
    # reads the result.
    error: str | None = None
