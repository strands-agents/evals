import copy
import warnings

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


@pytest.fixture(autouse=True)
def _quiet_reflection():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Reflection runs model-written", UserWarning)
        yield


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


def test_block_defaults_its_name_to_the_proposer_class():
    assert Block(TPE(), "temperature").name == "TPE"
    assert Block(TPE(), "temperature", name="numbers").name == "numbers"


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
        Search([Block(TPE(), "temperature"), Block(TPE(), "effort")])


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
        ([Block(Fake(), "temperature", name="numbers")], "need the default block 'numbers'"),
    ],
)
def test_resolve_rejects_bad_blocks(blocks, message):
    with pytest.raises(ValueError, match=message):
        Search(blocks).resolve(full_space())


def test_reflection_warns_that_its_shell_is_not_a_sandbox():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="not a sandbox"):
            Reflection()


def test_proposer_configs():
    assert TPE(seed=3, n_startup_trials=2).config() == {"seed": 3, "n_startup_trials": 2}
    config = Reflection("my-model", system_prompt="be brief", success_threshold=None).config()
    assert config["model"] == "my-model"
    assert config["success_threshold"] is None
    assert len(config["system_prompt_sha256"]) == 64
    assert Reflection().config()["system_prompt_sha256"] is None
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


def test_a_key_named_directly_wins_over_a_declared_name():
    blocks = [Block(Fake(), "manager", name="structure"), Block(TPE(), "manager.Window.size", name="window")]

    resolved, _ = Search(blocks).resolve(full_space())

    assert names(resolved)[:2] == [
        ("structure", "Fake", ["manager", "manager.Summary.ratio"]),
        ("window", "TPE", ["manager.Window.size"]),
    ]
    with pytest.raises(ValueError, match="block 'all' holds no variable of its own"):
        Search([Block(Fake(), "tools", name="all"), Block(Fake(), "tools.lookup", name="one")]).resolve(
            Space({"tools": tool_descriptions(lookup)})
        )


def test_a_block_under_another_blocks_choice_comes_after_it():
    space = two_structures(temperature=Float(0.0, 1.0, default=0.5))
    blocks = [Block(TPE(), "router", "manager.Summary.ratio", name="b"), Block(TPE(), "manager", name="a")]

    _, schedule = Search(blocks).resolve(space)

    assert shape(schedule) == ("nested", "a", ("nested", "b", "alternate"))


def test_a_given_schedule_is_checked_and_returned():
    space = full_space()
    nested = Nested("structure", Alternate())
    _, schedule = Search(schedule=nested).resolve(space)
    assert schedule is nested

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
    return ([b.name for b in step.blocks], step.n, step.parents, step.round, step.role, step.observe_at_round_end)


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
    assert history.replays(lambda: Nested("a", Nested("b", Alternate(), inner_iterations=2), inner_iterations=3))
