import copy
import os
import random
import warnings
from types import SimpleNamespace

import pytest
from strands import tool

from strands_evals.optimization import (
    TPE,
    Alternate,
    Block,
    Float,
    Int,
    Joint,
    Nested,
    OneOf,
    Proposer,
    Reflection,
    Search,
    Space,
    TextValue,
    Thompson,
    choice,
    component,
    tool_descriptions,
)
from strands_evals.optimization.search.blocks import ScheduleContext, Step
from strands_evals.types.optimization import IterationRecord, ProposalOutcome


class Window:
    def __init__(self, size: int = 40):
        self.size = size


class Summary:
    def __init__(self, ratio: float = 0.3):
        self.ratio = ratio


class Router:
    def __init__(self, k: int = 2):
        self.k = k


@tool
def lookup(query: str) -> str:
    """Look up a record."""
    return query


class Fake(Proposer):
    kinds = frozenset({"int", "float", "one_of", "choice", "text"})

    async def propose(self, request):
        return []

    def observe(self, observation):
        pass


def manager():
    return choice(component(Window, size=Int(1, 100)), component(Summary, ratio=Float(0.1, 0.9)))


def full_space():
    return Space(
        {
            "manager": manager(),
            "temperature": Float(0.0, 1.0, default=0.7),
            "effort": OneOf("low", "high"),
            "prompt": TextValue(default="You help.", size_cap=50),
        }
    )


def names(resolved):
    return [(block.name, type(block.proposer).__name__, keys) for block, keys in resolved]


def shape(schedule):
    if isinstance(schedule, Nested):
        return ("nested", schedule.outer, shape(schedule.inner))
    return schedule.name


# ---- Block and Search -------------------------------------------------------------------------------------------


def test_an_unnamed_block_is_named_after_its_kind_made_unique():
    a, b, mixed = Block(TPE(), "temperature"), Block(TPE(), "effort"), Block(Fake(), "manager")
    assert a.name is None

    resolved, _ = Search([a, b, mixed]).resolve(full_space())

    assert [block.name for block, _ in resolved] == ["numbers", "numbers 2", "block", "text"]
    assert a.name is None  # named on the returned copy; your Block is unchanged
    # a default block whose name is taken gets the next number; your named blocks keep theirs
    resolved, _ = Search([Block(Fake(), "temperature", name="numbers")]).resolve(full_space())
    assert [block.name for block, _ in resolved] == ["numbers", "structure", "numbers 2", "text"]
    assert Block(TPE(), "temperature", name="mine").name == "mine"


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (lambda: Block("temperature", TPE()), TypeError),
        (lambda: Block(TPE()), ValueError),
        (lambda: Block(TPE(), 3), TypeError),
        (lambda: Block(TPE(), "temperature", name=""), ValueError),
    ],
)
def test_block_rejects_bad_arguments(make, error):
    with pytest.raises(error):
        make()


def test_search_rejects_two_blocks_with_one_name():
    with pytest.raises(ValueError, match="two blocks are named 'TPE'"):
        Search([Block(TPE(), "temperature", name="TPE"), Block(TPE(), "effort", name="TPE")])


def test_search_rejects_two_blocks_sharing_one_proposer():
    tpe = TPE()
    with pytest.raises(ValueError, match="share one proposer"):
        Search([Block(tpe, "temperature"), Block(tpe, "effort")])
    block = Block(TPE(), "temperature")
    with pytest.raises(ValueError, match="share one proposer"):
        Search([block, block])
    Search([Block(TPE(), "temperature"), Block(TPE(), "effort")])  # one instance each


def test_search_rejects_negative_redraws_and_non_blocks():
    with pytest.raises(ValueError, match="redraws"):
        Search(redraws=-1)
    with pytest.raises(TypeError, match="Block objects"):
        Search([TPE()])


# ---- Search.resolve: blocks -------------------------------------------------------------------------------------


def test_default_search_assigns_every_key_by_kind():
    resolved, schedule = Search().resolve(full_space())

    assert names(resolved) == [
        ("structure", "TPE", ["manager"]),
        ("numbers", "TPE", ["manager.Window.size", "manager.Summary.ratio", "temperature", "effort"]),
        ("text", "Reflection", ["prompt"]),
    ]
    assert resolved[0][0].proposer.n_startup_trials == 2
    assert resolved[1][0].proposer.n_startup_trials == 4
    assert shape(schedule) == ("nested", "structure", "alternate")


def test_a_bare_component_goes_to_no_block():
    space = Space({"manager": component(Window, size=Int(1, 100)), "temperature": Float(0.0, 1.0, default=0.5)})

    resolved, schedule = Search().resolve(space)

    assert names(resolved) == [("numbers", "TPE", ["manager.Window.size", "temperature"])]
    assert shape(schedule) == "joint"


def test_a_declared_name_expands_to_every_key_under_it():
    resolved, _ = Search([Block(Fake(), "manager", name="mine")]).resolve(full_space())

    assert resolved[0][1] == ["manager", "manager.Window.size", "manager.Summary.ratio"]
    assert [b.name for b, _ in resolved] == ["mine", "numbers", "text"]


def test_a_flat_key_is_one_variable_and_keeps_declaration_order():
    resolved, _ = Search([Block(Fake(), "temperature", "manager.Summary.ratio", name="mine")]).resolve(full_space())

    assert resolved[0][1] == ["manager.Summary.ratio", "temperature"]
    assert ("numbers", "TPE", ["manager.Window.size", "effort"]) in names(resolved)


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ([Block(Fake(), "temperature", name="a"), Block(Fake(), "temperature", name="b")], "in blocks 'a' and 'b'"),
        ([Block(Fake(), "manager", name="a"), Block(Fake(), "manager", name="b")], "in blocks 'a' and 'b'"),
        (
            [Block(Fake(), "manager.Window.size", name="a"), Block(Fake(), "manager", "manager.Window.size", name="b")],
            "in blocks 'a' and 'b'",
        ),
        ([Block(Fake(), "nope", name="a")], "neither a declared name nor a key"),
        ([Block(TPE(), "prompt")], r"prompt \(text\), which TPE can't propose"),
    ],
)
def test_resolve_rejects_bad_blocks(blocks, message):
    with pytest.raises(ValueError, match=message):
        Search(blocks).resolve(full_space())


@pytest.mark.skipif(os.name == "nt", reason="Reflection needs Linux or macOS")
async def test_reflection_warns_once_when_it_first_proposes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reflection = Reflection()  # building one, as resolve does for the default text block, doesn't warn
        Search().resolve(full_space())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            assert await reflection.propose(SimpleNamespace(n=0)) == []  # a request asking for nothing
    assert len([w for w in caught if "not a sandbox" in str(w.message)]) == 1


def test_proposer_configs():
    assert TPE(seed=3, n_startup_trials=2).config() == {"seed": 3, "n_startup_trials": 2}
    config = Reflection("my-model", system_prompt="be brief", success_threshold=None).config()
    assert config["model"] == "my-model"
    assert config["success_threshold"] is None
    assert len(config["system_prompt_sha256"]) == 64
    assert Reflection().config()["system_prompt_sha256"] is None
    assert Reflection().config()["model"] == "global.anthropic.claude-sonnet-5-5"
    with pytest.raises(ValueError, match="success_threshold"):
        Reflection(success_threshold=1.5)
    with pytest.raises(ValueError, match="n_startup_trials"):
        TPE(n_startup_trials=-1)


# ---- Search.resolve: the default schedule and the check ----------------------------------------------------------


def two_structures(**extra):
    return Space(
        {
            "manager": manager(),
            "router": choice(router=component(Router, k=Int(1, 5)), off=None),
            **extra,
        }
    )


@pytest.mark.parametrize(
    ("space", "expected"),
    [
        (Space({"temperature": Float(0.0, 1.0, default=0.5)}), "joint"),
        (Space({"manager": manager()}), ("nested", "structure", "alternate")),
        (
            two_structures(temperature=Float(0.0, 1.0, default=0.5)),
            ("nested", "a", ("nested", "b", "alternate")),
        ),
        (two_structures(), ("nested", "a", "alternate")),
    ],
)
def test_default_schedule_nests_the_structural_blocks(space, expected):
    blocks = []
    if "router" in {p.name for p in space.params()}:
        blocks = [Block(TPE(), "manager", name="a"), Block(TPE(), "router", name="b")]
    _, schedule = Search(blocks).resolve(space)

    assert shape(schedule) == expected


def test_only_structural_blocks_and_one_of_them_is_joint():
    space = Space({"manager": choice(component(Window), component(Summary))})

    resolved, schedule = Search().resolve(space)

    assert names(resolved) == [("structure", "TPE", ["manager"])]
    assert shape(schedule) == "joint"


def test_a_declared_name_takes_only_the_keys_its_proposer_can_propose():
    resolved, _ = Search([Block(Thompson(), "manager")]).resolve(full_space())

    assert names(resolved) == [
        ("structure", "Thompson", ["manager"]),
        ("numbers", "TPE", ["manager.Window.size", "manager.Summary.ratio", "temperature", "effort"]),
        ("text", "Reflection", ["prompt"]),
    ]
    with pytest.raises(ValueError, match=r"'b' holds no variable TPE can propose \(tools.lookup \(text\)\)"):
        Search([Block(TPE(), "tools", name="b")]).resolve(Space({"tools": tool_descriptions(lookup)}))
    with pytest.raises(ValueError, match=r"manager.Window.size \(int\), which Thompson can't propose"):
        Search([Block(Thompson(), "manager", "manager.Window.size")]).resolve(full_space())


def test_a_key_named_directly_wins_over_a_declared_name():
    blocks = [Block(Fake(), "manager", name="structure"), Block(TPE(), "manager.Window.size", name="window")]

    resolved, _ = Search(blocks).resolve(full_space())

    assert names(resolved)[:2] == [
        ("structure", "Fake", ["manager", "manager.Summary.ratio"]),
        ("window", "TPE", ["manager.Window.size"]),
    ]
    with pytest.raises(ValueError, match="'all' holds no variable of its own"):
        Search([Block(Fake(), "tools", name="all"), Block(Fake(), "tools.lookup", name="one")]).resolve(
            Space({"tools": tool_descriptions(lookup)})
        )


def test_a_block_under_another_blocks_choice_comes_after_it():
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [Block(TPE(), "router", "manager.Summary.ratio", name="b"), Block(TPE(), "manager", name="a")]

    _, schedule = Search(blocks).resolve(space)

    assert shape(schedule) == ("nested", "a", ("nested", "b", "alternate"))


def test_a_key_under_a_one_option_choice_depends_on_no_block():
    # "router" is a bare component (one option), claimed by a non-structural block; the structural block holding
    # its k is neither ordered after that block nor rejected as nested around it
    space = Space({"manager": manager(), "router": component(Router, k=Int(1, 5))})
    blocks = [Block(TPE(), "router", name="r"), Block(TPE(), "manager", "router.Router.k", name="a")]

    _, schedule = Search(blocks).resolve(space)
    assert shape(schedule) == ("nested", "a", "alternate")
    Search(blocks, schedule=Nested("a")).resolve(space)


def test_a_given_schedule_is_checked_and_returned():
    space = full_space()
    nested = Nested("structure", Alternate())
    _, schedule = Search(schedule=nested).resolve(space)
    assert schedule.config() == nested.config()  # a copy: the Nested names its outer block

    with pytest.raises(ValueError, match="no such block in its scope"):
        Search(schedule=Nested("nope")).resolve(space)
    with pytest.raises(ValueError, match="exists only for some values of manager"):
        Search(
            [Block(Fake(), "manager", name="s"), Block(Fake(), "manager.Window.size", name="w")],
            schedule=Nested("w"),
        ).resolve(space)


def test_within_sets_the_default_rounds_inner_schedule():
    _, schedule = Search(within=Joint()).resolve(full_space())
    assert shape(schedule) == ("nested", "structure", "joint")
    # without a structural block the default is Joint() whatever within says
    _, schedule = Search(within=Alternate()).resolve(Space({"temperature": Float(0.0, 1.0, default=0.5)}))
    assert shape(schedule) == "joint"
    with pytest.raises(TypeError, match=r"within=\.\.\.\) takes a Schedule"):
        Search(within="alternate")
    with pytest.raises(ValueError, match="shapes only the default schedule"):
        Search(schedule=Joint(), within=Alternate())


def test_every_schedule_is_checked_the_default_and_within_included():
    with pytest.raises(ValueError, match="no such block in its scope"):
        Search(within=Nested("strucutre")).resolve(full_space())


def test_a_wrong_type_is_a_type_error_and_a_bad_value_a_value_error():
    with pytest.raises(TypeError, match="must be an int"):
        Nested("s", inner_iterations="4")
    with pytest.raises(TypeError, match="must be an int"):
        Search(redraws="3")
    with pytest.raises(ValueError, match="empty name"):
        Nested("")


def test_nested_takes_an_unnamed_block_named_by_resolve():
    structure = Block(TPE(n_startup_trials=2), "manager")
    nested = Nested(structure, Alternate())

    resolved, schedule = Search([structure], schedule=nested).resolve(full_space())

    assert schedule is not nested and schedule.outer == resolved[0][0].name == "block"


def test_resolve_leaves_the_users_blocks_and_schedule_unchanged():
    structure = Block(TPE(n_startup_trials=2), "manager")
    numbers = Block(TPE(), "temperature")
    nested = Nested(structure, Alternate())
    search = Search([structure, numbers], schedule=nested)

    first, _ = search.resolve(full_space())
    second, _ = search.resolve(full_space())

    assert structure.name is None and numbers.name is None and nested._outer is structure
    assert [b.name for b, _ in first] == [b.name for b, _ in second] == ["block", "numbers", "numbers 2", "text"]
    assert first[0][0] is not structure and first[0][0].proposer is structure.proposer
    with pytest.raises(ValueError, match="no name"):
        _ = nested.outer


def test_a_nested_on_an_unnamed_block_outside_the_search_raises():
    with pytest.raises(ValueError, match="not in this Search"):
        Search(schedule=Nested(Block(TPE(), "manager"))).resolve(full_space())


def test_nested_config():
    assert Nested(Block(TPE(), "manager", name="s"), Alternate(), inner_iterations=2).config() == {
        "outer": "s",
        "inner": {"name": "alternate", "config": {}},
        "inner_iterations": 2,
    }
    with pytest.raises(ValueError, match="inner_iterations"):
        Nested("s", inner_iterations=-1)
    with pytest.raises(TypeError, match="Block or a block name"):
        Nested(3)


# ---- schedules over a scripted history ---------------------------------------------------------------------------

BASE = {"manager": "Window", "manager.Window.size": 40, "temperature": 0.7}
SUMMARY = {"manager": "Summary", "manager.Summary.ratio": 0.3, "temperature": 0.7}


class History:
    """A scripted run: plans with the schedule, records what each step saved, and keeps every context."""

    def __init__(self, schedule, blocks, space=None, *, n=4, redraws=3):
        self.space = space or Space({"manager": manager(), "temperature": Float(0.0, 1.0, default=0.7)})
        resolved, self.schedule = Search(blocks, schedule=schedule, redraws=redraws).resolve(self.space)
        self.blocks = [block for block, _ in resolved]
        self.keys = {block.name: keys for block, keys in resolved}
        self.n, self.redraws = n, redraws
        self.past: list[IterationRecord] = []
        self.values = {"base": BASE}
        self.eligible = {"base"}
        self.ranking = ["base"]
        self.contexts: list[ScheduleContext] = []
        self.steps: list[Step | None] = []

    def plan(self):
        ctx = ScheduleContext(
            past=list(self.past),
            blocks=self.blocks,
            keys=self.keys,
            space=self.space,
            values=dict(self.values),
            eligible=frozenset(self.eligible),
            ranking=list(self.ranking),
            run_ranking=list(self.ranking),
            proposals_per_iteration=self.n,
            redraws=self.redraws,
        )
        step = self.schedule.next(ctx)
        self.contexts.append(ctx)
        self.steps.append(step)
        return step

    def save(self, step, *results):
        """Each result: "feasible" / "ineligible" / "invalid" for a new candidate, ("feasible", values), or
        ("duplicate", existing id)."""
        parent = step.parents[0] if step.parents else self.ranking[0]
        proposals = []
        for result in results:
            outcome, extra = (result, None) if isinstance(result, str) else result
            if outcome == "invalid":
                candidate_id = None
            elif outcome == "duplicate":
                candidate_id = extra
            else:
                candidate_id = f"c{len(self.values)}"
                self.values[candidate_id] = extra or BASE
                if outcome == "feasible":
                    self.eligible.add(candidate_id)
                    self.ranking.insert(0, candidate_id)
            proposals.append(
                ProposalOutcome(
                    candidate_id=candidate_id,
                    number=None,
                    parent_id=parent,
                    origin=step.blocks[0].name,
                    changed_keys=[],
                    rationale="",
                    outcome=outcome,
                    objective=None,
                )
            )
        self.past.append(
            IterationRecord(
                iteration=len(self.past) + 1,
                parent_id=parent,
                round=step.round,
                role=step.role,
                blocks=[b.name for b in step.blocks],
                n_candidates=step.n_candidates,
                proposals=proposals,
                rollouts_spent=0,
            )
        )

    def run(self, results):
        """Plan and save until the schedule stops; `results(step, index)` gives what a step saved."""
        while (step := self.plan()) is not None:
            self.save(step, *results(step, len(self.past)))
            assert len(self.past) < 200
        return self.steps

    def replays(self, make_schedule):
        """A fresh schedule, given each context again, plans the same steps (resume)."""
        fresh = make_schedule()
        return [fresh.next(copy.copy(ctx)) for ctx in self.contexts] == self.steps


def view(step):
    if step is None:
        return None
    return (
        [b.name for b in step.blocks],
        step.n_candidates,
        step.parents,
        step.round,
        step.role,
        step.observe_at_round_end,
    )


def blocks4():
    return [
        Block(Fake(), "manager", name="structure"),
        Block(Fake(), "manager.Window.size", name="window"),
        Block(Fake(), "manager.Summary.ratio", name="summary"),
        Block(Fake(), "temperature", name="temp"),
    ]


def test_joint_proposes_every_block_until_a_step_saves_nothing():
    history = History(Joint(), [Block(Fake(), "temperature", name="t"), Block(Fake(), "manager", name="m")])

    steps = history.run(lambda step, i: ["feasible"] if i < 2 else [("duplicate", "base")])

    assert [view(s) for s in steps] == [(["t", "m"], 4, None, None, "inner", ())] * 3 + [None]
    assert history.replays(Joint)


def test_alternate_rotates_and_ends_after_a_cycle_of_misses():
    history = History(Alternate(), [Block(Fake(), "temperature", name="t"), Block(Fake(), "manager", name="m")])
    script = {0: ["feasible"], 1: ["invalid"], 2: ["feasible"], 3: ["invalid"], 4: [("duplicate", "base")]}

    steps = history.run(lambda step, i: script[i])

    assert [view(s)[0] if s else None for s in steps] == [["t"], ["m"], ["t"], ["m"], ["t"], None]
    assert history.replays(Alternate)


def nested_history(**kwargs):
    return History(Nested("structure", Alternate(), inner_iterations=2), blocks4(), **kwargs)


def test_nested_opens_from_the_incumbent_then_runs_inner_steps_on_the_rounds_candidates():
    history = nested_history()

    opener = history.plan()
    assert view(opener) == (["structure"], 1, ("base",), (0,), "opener", ("structure",))
    history.save(opener, ("feasible", SUMMARY))  # c1 chooses Summary

    first = history.plan()
    # window is narrowed out under Summary; the parents are the round's own eligible candidates
    assert view(first) == (["summary"], 4, ("c1",), (0,), "inner", ())
    history.save(first, "feasible", "ineligible")  # c2 eligible, c3 not

    second = history.plan()
    assert view(second) == (["temp"], 4, ("c2", "c1"), (0,), "inner", ())
    history.save(second, "invalid")  # saves nothing: doesn't use up the round

    third = history.plan()
    assert view(third) == (["summary"], 4, ("c2", "c1"), (0,), "inner", ())
    history.save(third, "feasible")

    # two inner steps saved candidates: the round is over, the next opens from the global incumbent
    assert view(history.plan()) == (["structure"], 1, ("c4",), (1,), "opener", ("structure",))
    assert history.replays(lambda: Nested("structure", Alternate(), inner_iterations=2))


def test_an_ineligible_opener_skips_the_inner_steps():
    history = nested_history()
    history.save(history.plan(), "ineligible")

    assert view(history.plan()) == (["structure"], 1, ("base",), (1,), "opener", ("structure",))


def test_a_repeated_opener_opens_the_round_on_the_measured_candidate():
    history = nested_history()
    history.save(history.plan(), "feasible")  # c1, under Window
    history.save(history.plan(), "feasible")
    history.save(history.plan(), "feasible")  # round 0 done
    opener = history.plan()
    history.save(opener, ("duplicate", "c1"))

    step = history.plan()

    assert view(step) == (["window"], 4, ("c1",), (1,), "inner", ())


def test_an_opener_that_saved_nothing_ends_only_its_round():
    history = nested_history(redraws=1)
    history.save(history.plan(), "invalid", "invalid")  # round 0: every draw invalid, no inner steps

    opener = history.plan()
    assert view(opener) == (["structure"], 1, ("base",), (1,), "opener", ("structure",))
    history.save(opener, "invalid")  # round 1 saves nothing either: redraws + 1 empty rounds in a row

    assert history.plan() is None


def test_rounds_that_measure_nothing_new_end_after_redraws_plus_one():
    history = nested_history(redraws=1)
    history.save(history.plan(), "feasible")  # c1: round 0 saves something
    history.save(history.plan(), "feasible")
    history.save(history.plan(), "feasible")
    for _ in range(2):  # rounds 1 and 2 reopen on c1 and their inner steps repeat it
        history.save(history.plan(), ("duplicate", "c1"))
        history.save(history.plan(), ("duplicate", "c1"))
        history.save(history.plan(), ("duplicate", "c1"))  # two misses: Alternate's cycle ends the round

    assert history.plan() is None


def test_inner_iterations_zero_is_openers_only():
    history = History(Nested("structure", inner_iterations=0), blocks4())

    steps = history.run(lambda step, i: ["feasible"] if i < 3 else [("duplicate", "base")])

    assert [view(s)[3:5] if s else None for s in steps[:3]] == [((0,), "opener"), ((1,), "opener"), ((2,), "opener")]


def test_an_outer_block_narrowed_out_runs_the_inner_schedule_alone():
    # "summary" exists only under Summary; with the incumbent under Window, the inner Nested skips its rounds
    blocks = blocks4()
    schedule = Nested("structure", Nested("summary", Joint()), inner_iterations=1)
    history = History(schedule, blocks)
    history.save(history.plan(), "feasible")  # c1 keeps Window

    step = history.plan()

    assert view(step) == (["window", "temp"], 4, ("c1",), (0,), "inner", ())


def test_two_levels_place_the_inner_rounds_inside_the_outer_round():
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [
        Block(Fake(), "manager", name="a"),
        Block(Fake(), "router", name="b"),
        Block(Fake(), "temperature", name="t"),
    ]
    schedule = Nested("a", Nested("b", Alternate(), inner_iterations=2), inner_iterations=3)
    history = History(schedule, blocks, space)
    history.values["base"] = {
        "manager": "Window",
        "manager.Window.size": 40,
        "router": "router",
        "router.router.k": 2,
        "temperature": 0.5,
    }

    steps = history.run(lambda step, i: ["feasible"] if i < 12 else [("duplicate", "base")])

    assert [(view(s)[0], view(s)[3], view(s)[4]) for s in steps[:8]] == [
        (["a"], (0,), "opener"),
        (["b"], (0, 0), "opener"),
        (["t"], (0, 0), "inner"),
        (["t"], (0, 0), "inner"),  # Alternate over the inner round's one block
        (["b"], (0, 1), "opener"),  # the inner round's 2 steps are done
        (["t"], (0, 1), "inner"),  # the outer round's 3rd inner step
        (["a"], (1,), "opener"),
        (["b"], (1, 0), "opener"),
    ]
    # an inner opener opens from the enclosing round's incumbent
    assert steps[4].parents == ("c4",)
    # the outer round holds 1 + 3 * 4 = 13 candidates; the inner openers use 2, so the last step gets the 2 left
    assert [s.n_candidates for s in steps[:6]] == [1, 1, 4, 4, 1, 2]
    assert history.replays(lambda: Nested("a", Nested("b", Alternate(), inner_iterations=2), inner_iterations=3))


def test_inner_openers_use_up_the_outer_round():
    # every inner opener saves and every inner step repeats: the outer round still ends
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [
        Block(Fake(), "manager", name="a"),
        Block(Fake(), "router", name="b"),
        Block(Fake(), "temperature", name="t"),
    ]
    history = History(Nested("a", Nested("b", Joint(), inner_iterations=2), inner_iterations=3), blocks, space)
    for _ in range(60):
        step = history.plan()
        history.save(step, "feasible" if step.role == "opener" else ("duplicate", "base"))

    outer_rounds = {record.round[0] for record in history.past}
    inner_rounds = {record.round for record in history.past if len(record.round) == 2 and record.round[0] == 0}
    assert len(outer_rounds) > 1
    assert len(inner_rounds) == 12  # the outer round's 12 slots after its opener, one per inner opener
    assert history.replays(lambda: Nested("a", Nested("b", Joint(), inner_iterations=2), inner_iterations=3))


def test_at_width_one_a_round_holds_its_steps_and_the_inner_openers():
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [
        Block(Fake(), "manager", name="a"),
        Block(Fake(), "router", name="b"),
        Block(Fake(), "temperature", name="t"),
    ]
    history = History(Nested("a", Nested("b", Joint(), inner_iterations=2), inner_iterations=6), blocks, space, n=1)
    history.run(lambda step, i: ["feasible"] if i < 20 else [("duplicate", "base")])

    first = [record for record in history.past if record.round[0] == 0]
    assert len(first) == 7  # 1 + 6 slots: the opener, then 2 inner rounds of an opener and 2 steps
    assert {record.round for record in first if len(record.round) == 2} == {(0, 0), (0, 1)}
    assert history.replays(lambda: Nested("a", Nested("b", Joint(), inner_iterations=2), inner_iterations=6))


def test_an_outer_round_is_charged_the_n_a_deeper_round_gave():
    # three levels, 4 per step: the middle round's room (4, less one inner opener) cuts the innermost step to 3,
    # and the outer round is charged 3, not 4, so it still holds 1 + 2 * 4 = 9 candidates
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5), top_p=Float(0.0, 1.0, default=0.9))
    blocks = [
        Block(Fake(), "manager", name="a"),
        Block(Fake(), "router", name="b"),
        Block(Fake(), "temperature", name="c"),
        Block(Fake(), "top_p", name="p"),
    ]

    def make():
        return Nested(
            "a", Nested("b", Nested("c", Joint(), inner_iterations=1), inner_iterations=1), inner_iterations=2
        )

    history = History(make(), blocks, space)
    history.values["base"] = {**TWO_BASE, "top_p": 0.9}
    while (step := history.plan()) is not None and step.round[0] == 0:
        history.save(step, *["feasible"] * step.n_candidates)

    assert [(r.round, r.role, r.n_candidates) for r in history.past] == [
        ((0,), "opener", 1),
        ((0, 0), "opener", 1),
        ((0, 0, 0), "opener", 1),
        ((0, 0, 0), "inner", 3),
        ((0, 1), "opener", 1),
        ((0, 1, 0), "opener", 1),
        ((0, 1, 0), "inner", 1),
    ]
    assert step.round == (1,)
    assert history.replays(make)


def test_an_inner_round_ranks_an_opener_repeating_a_candidate_from_an_earlier_round():
    # outer round 1's b-opener lands on the baseline, which outer round 1 never made, so only the run's ranking
    # holds it; the inner round still builds on it, and keeps it ahead of a worse candidate the round makes
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [
        Block(Fake(), "manager", name="a"),
        Block(Fake(), "router", name="b"),
        Block(Fake(), "temperature", name="t"),
    ]

    def make():
        return Nested("a", Nested("b", Alternate(), inner_iterations=2), inner_iterations=4)

    history = History(make(), blocks, space, n=1)
    history.values["base"] = dict(TWO_BASE)
    while (step := history.plan()).round[0] == 0:
        history.save(step, "feasible")
    history.save(step, "feasible")  # outer round 1's a-opener: a new candidate, the round's only one
    inner_opener = history.plan()
    assert view(inner_opener)[3:5] == ((1, 0), "opener")
    history.save(inner_opener, ("duplicate", "base"))

    first = history.plan()
    assert view(first) == (["t"], 1, ("base",), (1, 0), "inner", ())
    history.save(first, "feasible")
    worse = history.ranking.pop(0)
    history.ranking.append(worse)  # the round's new candidate ranks below the baseline

    assert history.plan().parents == ("base", worse)
    assert history.replays(make)


def test_nested_raises_on_an_empty_ranking():
    history = nested_history()
    history.ranking = []

    with pytest.raises(ValueError, match="no eligible candidate"):
        history.plan()


def test_a_joint_step_that_saved_nothing_ends_its_round_early():
    history = History(Nested("structure", Joint(), inner_iterations=4), blocks4())
    history.save(history.plan(), "feasible")  # the opener; the round has room for 16 more
    history.save(history.plan(), ("duplicate", "base"))  # Joint's first step saves nothing

    assert view(history.plan())[3:5] == ((1,), "opener")


TWO_BASE = {
    "manager": "Window",
    "manager.Window.size": 40,
    "router": "router",
    "router.router.k": 2,
    "temperature": 0.5,
}


def shapes():
    """(name, schedule factory, blocks factory, space factory, base values) for the randomized test."""

    def two():
        return two_structures(temperature=Float(0.0, 1.0, default=0.5))

    def three():
        return [
            Block(Fake(), "manager", name="a"),
            Block(Fake(), "router", name="b"),
            Block(Fake(), "temperature", name="t"),
        ]

    return [
        ("one level", lambda: Nested("structure", Alternate(), inner_iterations=2), blocks4, None, BASE),
        (
            "two levels",
            lambda: Nested("a", Nested("b", Joint(), inner_iterations=2), inner_iterations=3),
            three,
            two,
            TWO_BASE,
        ),
        ("openers only", lambda: Nested("structure", inner_iterations=0), blocks4, None, BASE),
        ("default", lambda: None, lambda: [], two, TWO_BASE),
    ]


@pytest.mark.parametrize("shape", range(4), ids=lambda i: shapes()[i][0])
@pytest.mark.parametrize("seed", range(25))
def test_random_histories_keep_every_round_rule(shape, seed):
    name, make_schedule, make_blocks, make_space, base = shapes()[shape]
    rng = random.Random(f"{seed}:{name}")
    history = History(make_schedule(), make_blocks(), make_space() if make_space else None, redraws=1)
    history.values["base"] = base
    schedule = history.schedule

    def result(step, i):
        def one():
            # new candidates for 40 steps, then only repeats and invalid draws, so the rounds must end
            outcome = rng.choice(
                ["feasible", "feasible", "ineligible", "invalid", "duplicate"] if i < 40 else ["invalid", "duplicate"]
            )
            if outcome == "duplicate":
                return ("duplicate", rng.choice(sorted(history.eligible)))
            if outcome == "feasible" and make_space is None and rng.random() < 0.5:
                return ("feasible", SUMMARY)
            return outcome

        return [one() for _ in range(step.n_candidates)]

    steps = history.run(result)

    assert steps[-1] is None
    for step, record in zip(steps, history.past, strict=False):
        assert step is not None and step.blocks and 1 <= step.n_candidates <= history.n
        assert record.round is not None
    if isinstance(schedule, Nested):
        # outer rounds are numbered without gaps, and each holds at most 1 + inner_iterations * n candidates
        outer = sorted({record.round[0] for record in history.past})
        assert outer == list(range(len(outer)))
        for r in outer:
            # the round's own opener uses 1, an inner opener 1 unless it repeated, a step that saved its n
            used = sum(
                1 if record.role == "opener" else step.n_candidates
                for step, record in zip(steps, history.past, strict=False)
                if record.round[0] == r
                and (
                    (record.role == "opener" and record.round == (r,))
                    or (record.role == "opener" and record.proposals[-1].outcome != "duplicate")
                    or (
                        record.role == "inner"
                        and any(p.outcome not in ("invalid", "duplicate") for p in record.proposals)
                    )
                )
            )
            assert used <= 1 + schedule.inner_iterations * history.n
        # it ended after redraws + 1 outer rounds in a row that saved nothing new
        empty = 0
        for r in reversed(outer):
            records = [record for record in history.past if record.round[0] == r]
            if any(p.outcome not in ("invalid", "duplicate") for record in records for p in record.proposals):
                break
            empty += 1
        assert empty == history.redraws + 1
    assert history.replays(lambda: copy.deepcopy(schedule))  # a schedule keeps no state, so a copy is a fresh one


def test_a_repeated_inner_opener_costs_only_its_own_round():
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [
        Block(Fake(), "manager", name="a"),
        Block(Fake(), "router", name="b"),
        Block(Fake(), "temperature", name="t"),
    ]
    history = History(Nested("a", Nested("b", Joint(), inner_iterations=1), inner_iterations=4), blocks, space, n=1)
    inner_openers = 0
    while (step := history.plan()) is not None and step.round[0] == 0:
        repeat = step.role == "opener" and len(step.round) == 2 and (inner_openers := inner_openers + 1) == 2
        history.save(step, ("duplicate", history.ranking[0]) if repeat else "feasible")

    saved = [r for r in history.past if r.round[0] == 0 and r.proposals[-1].outcome != "duplicate"]
    assert len(saved) == 5  # 1 + 4 slots, the repeated inner opener not among them
    assert history.replays(lambda: Nested("a", Nested("b", Joint(), inner_iterations=1), inner_iterations=4))
