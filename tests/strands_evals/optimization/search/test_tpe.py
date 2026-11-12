import asyncio
import random
import statistics

import pytest

from strands_evals.optimization import (
    TPE,
    AtMost,
    Criteria,
    Float,
    Int,
    Maximize,
    MetricRegistry,
    Minimize,
    OneOf,
    Pareto,
    Space,
    choice,
    component,
)
from strands_evals.optimization.candidate import Candidate
from strands_evals.optimization.search.proposer import ProposalRequest
from strands_evals.types.optimization import CandidateScores, CandidateSummary, Evidence, Observation

REGISTRY = MetricRegistry({"Score": (0.0, 1.0), "Cost": (0.0, 1.0)})
CRITERIA = Criteria(Maximize("Score"), [], REGISTRY)


class Window:
    def __init__(self, size: int = 10):
        self.size = size


class Summary:
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio


def numbers_space():
    return Space({"x": Float(0.0, 1.0, default=0.5), "k": Int(1, 9, default=5), "mode": OneOf("a", "b", default="a")})


def structured_space():
    return Space(
        {
            "manager": choice(window=component(Window, size=Int(2, 40)), summary=component(Summary)),
            "temperature": Float(0.0, 1.0, default=0.5),
        }
    )


def request(space, keys, *, parent=None, n=1, iteration=1, redraw=0, criteria=CRITERIA, seed=0, candidates=()):
    scores = CandidateScores(
        candidate_id="p",
        objective_by_case={},
        metrics_by_case={},
        unavailable={},
        objective_by_run={},
        metrics_by_run={},
        objective=None,
        is_eligible=True,
        failures=0,
    )
    return ProposalRequest(
        space=space,
        keys=keys,
        parent=parent or Candidate.baseline(space),
        trials=[],
        scores=scores,
        inputs={},
        fit=None,
        candidates=list(candidates),
        n=n,
        iteration=iteration,
        seed=seed,
        redraw=redraw,
        criteria=criteria,
        log=lambda *args, **kwargs: None,
    )


def propose(tpe, req):
    return asyncio.run(tpe.propose(req))


def told(
    patch,
    values,
    objective,
    *,
    outcome="feasible",
    violations=None,
    term_values=None,
    cid="id",
    source=None,
    round_candidate_ids=None,
):
    return Observation(
        patch=patch,
        candidate_id=None if values is None else cid,
        values=values,
        source_id=source,
        round_candidate_ids=round_candidate_ids,
        objective=objective,
        term_values=term_values,
        outcome=outcome,
        reason="",
        violations=violations if violations is not None else ([] if objective is not None else None),
    )


def evidence(values, objective, *, violations=None, cid="other"):
    return Evidence(
        candidate_id=cid,
        values=values,
        origin="other",
        objective=objective,
        term_values=None,
        violations=[] if violations is None else violations,
        outcome="feasible",
    )


def summary(cid, objective, *, violations=(), term_values=None, is_eligible=True):
    """A listed candidate as the latest ranking assesses it."""
    return CandidateSummary(
        candidate_id=cid,
        parent_id=None,
        origin="test",
        values={},
        number=0,
        measured_objective=objective,
        measured={},
        trial_count=1,
        ranking_objective=objective if is_eligible else None,
        term_values=term_values if is_eligible else None,
        rank=1 if is_eligible else None,
        estimates={},
        violations=list(violations) if is_eligible else None,
        is_feasible=all(v <= 0 for v in violations) if is_eligible else None,
        is_eligible=is_eligible,
        changed_keys=[],
    )


def run(tpe, space, keys, objective, iterations, *, n=1):
    """Propose and tell `objective(values)` for `iterations` steps; returns every proposal's values."""
    seen = []
    parent = Candidate.baseline(space)
    for i in range(iterations):
        patches = propose(tpe, request(space, keys, parent=parent, n=n, iteration=i))
        for patch in patches:
            values = space.complete({**parent.values, **patch.updates})
            seen.append(values)
            tpe.observe(told(patch, values, objective(values)))
    return seen


# ---- what it proposes -----------------------------------------------------------------------------------------------
def test_proposals_are_typed_values_inside_the_block():
    space = numbers_space()
    patches = propose(TPE(), request(space, ["x", "k", "mode"], n=4))
    assert len(patches) == 4
    for patch in patches:
        assert set(patch.updates) == {"x", "k", "mode"}
        assert type(patch.updates["x"]) is float and 0.0 <= patch.updates["x"] <= 1.0
        assert type(patch.updates["k"]) is int and 1 <= patch.updates["k"] <= 9
        assert patch.updates["mode"] in ("a", "b")
        space.validate(space.complete({**Candidate.baseline(space).values, **patch.updates}))


def test_a_choice_is_drawn_before_its_sub_keys():
    space = structured_space()
    keys = ["manager", "manager.window.size"]
    patches = propose(TPE(), request(space, keys, n=12))
    managers = {p.updates["manager"] for p in patches}
    assert managers == {"window", "summary"}
    for patch in patches:
        assert ("manager.window.size" in patch.updates) == (patch.updates["manager"] == "window")


def test_a_key_inactive_under_the_parents_choices_is_not_proposed():
    space = structured_space()
    parent = Candidate(space.complete({"manager": "summary"}), None, "baseline", "")
    assert propose(TPE(), request(space, ["manager.window.size"], parent=parent)) == []


def test_options_python_compares_equal_stay_apart():
    space = Space({"flag": OneOf(1, True, default=1)})
    patches = [p for seed in range(4) for p in propose(TPE(), request(space, ["flag"], n=4, seed=seed))]
    assert {(type(p.updates["flag"]), p.updates["flag"]) for p in patches} == {(int, 1), (bool, True)}
    tpe = TPE()
    patches = propose(tpe, request(space, ["flag"], n=2))
    tpe.observe(told(patches[0], {"flag": True}, 0.9))
    study = tpe._study_for(tpe._pending[0].context, space, ["flag"], request(space, ["flag"]), 0)
    assert [t.params["flag"] for t in study.trials if t.state.name == "COMPLETE"] == [1]  # True is option 1


# ---- the step's proposals -------------------------------------------------------------------------------------------
def test_a_steps_proposals_differ_in_one_call_and_across_calls():
    space = numbers_space()
    tpe = TPE()
    in_one = propose(tpe, request(space, ["x"], n=4))
    across = [propose(tpe, request(space, ["x"]))[0] for _ in range(4)]  # one call per candidate, same step
    xs = [p.updates["x"] for p in in_one + across]
    assert len(set(xs)) == len(xs)


def test_two_blocks_asked_in_one_step_draw_differently():
    space = Space({"a": Float(0.0, 1.0, default=0.5), "b": Float(0.0, 1.0, default=0.5)})
    a, b = TPE(), TPE()
    for i in range(4):
        (pa,) = propose(a, request(space, ["a"], iteration=i))
        (pb,) = propose(b, request(space, ["b"], iteration=i))
        assert pa.updates["a"] != pb.updates["b"]
        a.observe(told(pa, space.complete(pa.updates), 0.5))
        b.observe(told(pb, space.complete(pb.updates), 0.5))


def test_pending_proposals_are_running_trials_until_observed():
    space = numbers_space()
    tpe = TPE()
    patches = propose(tpe, request(space, ["x"], n=3))
    study = tpe._study_for(tpe._pending[0].context, space, ["x"], request(space, ["x"]), 0)
    assert [t.state.name for t in study.trials] == ["RUNNING"] * 3
    for patch in patches:
        tpe.observe(told(patch, space.complete(patch.updates), 0.5))
    assert tpe._pending == []
    study = tpe._study_for(patches[0].state["context"], space, ["x"], request(space, ["x"]), 0)
    assert [t.state.name for t in study.trials] == ["COMPLETE"] * 3


# ---- what it learns -------------------------------------------------------------------------------------------------
def test_each_trial_is_valued_at_its_candidates_current_assessment():
    space = numbers_space()
    criteria = Criteria(Maximize("Score"), [AtMost("Cost", 0.5)], REGISTRY)
    tpe = TPE()
    a, b, c, d = propose(tpe, request(space, ["x"], n=4, criteria=criteria))
    tpe.observe(told(a, space.complete(a.updates), 0.3, violations=[-0.1], cid="a"))
    tpe.observe(told(b, space.complete(b.updates), 0.8, violations=[0.2], cid="b"))
    tpe.observe(told(c, None, None, outcome="invalid"))
    tpe.observe(told(d, space.complete(d.updates), None, outcome="ineligible", cid="d"))
    tpe.evidence(evidence(space.complete({"x": 0.9}), 0.6, violations=[-0.3], cid="e"))
    tpe.evidence(evidence(space.complete({"x": 0.1}), None, violations=None, cid="f"))  # never measured: no evidence
    context = a.state["context"]
    as_told = tpe._study_for(context, space, ["x"], request(space, ["x"], criteria=criteria), 0)
    assert [(t.state.name, t.values, t.constraints) for t in as_told.trials] == [
        ("COMPLETE", [0.3], {"0": -0.1}),
        ("COMPLETE", [0.8], {"0": 0.2}),
        ("FAIL", None, {}),
        ("COMPLETE", [0.6], {"0": -0.3}),
    ]  # nothing listed: as told
    refitted = [
        summary("a", 0.5, violations=[0.1]),  # re-estimated, now over the limit
        summary("b", 0.8, is_eligible=False),  # a re-score left a case unmeasured
        summary("d", 0.9),  # told as never measured: stays failed
        summary("f", 0.7),  # never measured when shared: still no trial
    ]
    now = tpe._study_for(context, space, ["x"], request(space, ["x"], criteria=criteria, candidates=refitted), 0)
    assert [(t.state.name, t.values, t.constraints) for t in now.trials] == [
        ("COMPLETE", [0.5], {"0": 0.1}),
        ("FAIL", None, {}),
        ("FAIL", None, {}),
        ("COMPLETE", [0.6], {"0": -0.3}),  # e isn't listed: as told
    ]


def test_a_round_opener_is_valued_at_its_rounds_best_by_the_latest_ranking():
    space = numbers_space()
    tpe = TPE()
    (patch,) = propose(tpe, request(space, ["x"]))
    tpe.observe(
        told(patch, space.complete(patch.updates), 0.7, cid="o", source="r1", round_candidate_ids=["o", "r1", "r2"])
    )
    context = patch.state["context"]
    ranked = [summary("r2", 0.75), summary("x", 0.9), summary("r1", 0.7), summary("o", 0.6)]
    study = tpe._study_for(context, space, ["x"], request(space, ["x"], candidates=ranked), 0)
    assert [t.values for t in study.trials] == [[0.75]]  # r2 now ranks first in the round; x isn't in it
    none_eligible = [summary(m, 0.7, is_eligible=False) for m in ("o", "r1", "r2")]
    study = tpe._study_for(context, space, ["x"], request(space, ["x"], candidates=none_eligible), 0)
    assert [t.state.name for t in study.trials] == ["FAIL"]


def test_an_opener_that_won_its_round_is_valued_at_the_rounds_best_later():
    space = numbers_space()
    tpe = TPE()
    (patch,) = propose(tpe, request(space, ["x"]))
    tpe.observe(told(patch, space.complete(patch.updates), 0.7, cid="o", round_candidate_ids=["o", "r1"]))
    ranked = [summary("r1", 0.72), summary("o", 0.68)]  # after a refit an inner candidate ranks first
    study = tpe._study_for(patch.state["context"], space, ["x"], request(space, ["x"], candidates=ranked), 0)
    assert [t.values for t in study.trials] == [[0.72]]


def test_a_duplicate_is_valued_at_the_existing_candidate():
    space = numbers_space()
    tpe = TPE()
    (patch,) = propose(tpe, request(space, ["x"]))
    existing = space.complete({"x": 0.25})
    tpe.observe(told(patch, existing, 0.7, outcome="duplicate", cid="existing"))
    context = patch.state["context"]
    study = tpe._study_for(context, space, ["x"], request(space, ["x"], candidates=[summary("existing", 0.65)]), 0)
    assert [(t.params["x"], t.values) for t in study.trials] == [(0.25, [0.65])]


def test_numbers_under_a_choice_are_learned_per_structure():
    space = structured_space()
    tpe = TPE()
    window = Candidate(space.complete({"manager": "window"}), None, "baseline", "")
    summary = Candidate(space.complete({"manager": "summary"}), None, "baseline", "")
    tpe.evidence(evidence({**window.values, "temperature": 0.1}, 0.9))
    tpe.evidence(evidence({**summary.values, "temperature": 0.9}, 0.2))
    (patch,) = propose(tpe, request(space, ["temperature"], parent=window))
    study = tpe._study_for(patch.state["context"], space, ["temperature"], request(space, ["temperature"]), 0)
    assert [t.params["temperature"] for t in study.trials if t.state.name == "COMPLETE"] == [0.1]


def test_a_structure_block_has_one_study_whatever_the_other_choices():
    space = Space(
        {
            "manager": choice(window=component(Window), summary=component(Summary)),
            "router": choice(fast=component(Window), slow=component(Summary)),
        }
    )
    tpe = TPE()
    tpe.evidence(evidence(space.complete({"router": "fast"}), 0.4))
    tpe.evidence(evidence(space.complete({"router": "slow"}), 0.6))
    (patch,) = propose(tpe, request(space, ["manager"]))
    study = tpe._study_for(patch.state["context"], space, ["manager"], request(space, ["manager"]), 0)
    assert len([t for t in study.trials if t.state.name == "COMPLETE"]) == 2


def test_under_pareto_the_study_is_told_each_terms_value():
    space = numbers_space()
    criteria = Criteria(Pareto(Maximize("Score"), Minimize("Cost")), [], REGISTRY)
    tpe = TPE()
    (patch,) = propose(tpe, request(space, ["x"], criteria=criteria))
    tpe.observe(told(patch, space.complete(patch.updates), 0.4, term_values=[0.7, 0.2]))
    study = tpe._study_for(patch.state["context"], space, ["x"], request(space, ["x"], criteria=criteria), 0)
    assert [d.name for d in study.directions] == ["MAXIMIZE", "MINIMIZE"]
    assert [t.values for t in study.trials] == [[0.7, 0.2]]
    refitted = request(space, ["x"], criteria=criteria, candidates=[summary("id", 0.5, term_values=[0.6, 0.3])])
    study = tpe._study_for(patch.state["context"], space, ["x"], refitted, 0)
    assert [t.values for t in study.trials] == [[0.6, 0.3]]
    assert len(propose(tpe, request(space, ["x"], criteria=criteria, iteration=2))) == 1


def test_it_moves_toward_a_better_region_after_its_startup_trials():
    space = Space({"x": Float(0.0, 1.0, default=0.5)})
    seen = run(TPE(seed=3), space, ["x"], lambda v: 1 - (v["x"] - 0.8) ** 2, 30)
    startup, modeled = seen[:4], seen[10:]
    assert statistics.fmean(abs(v["x"] - 0.8) for v in modeled) < 0.2  # uniform draws average 0.34
    assert statistics.fmean(abs(v["x"] - 0.8) for v in modeled) < statistics.fmean(abs(v["x"] - 0.8) for v in startup)


# ---- resume ---------------------------------------------------------------------------------------------------------
def test_replaying_the_same_history_proposes_the_same_patches():
    space = numbers_space()
    rng = random.Random(0)
    first = TPE()
    history = []
    parent = Candidate.baseline(space)
    for i in range(8):
        for j, patch in enumerate(propose(first, request(space, ["x", "k", "mode"], parent=parent, n=2, iteration=i))):
            event = told(patch, space.complete({**parent.values, **patch.updates}), rng.random(), cid=f"c{i}-{j}")
            first.observe(event)
            history.append(event)
        other = evidence(space.complete({"x": rng.random()}), rng.random(), cid=f"e{i}")
        first.evidence(other)
        history.append(other)
    resumed = TPE()
    for event in history:
        (resumed.observe if isinstance(event, Observation) else resumed.evidence)(event)
    refitted = [summary(f"c{i}-0", rng.random()) for i in range(8)]  # re-valued by a later fit
    for candidates in ((), refitted):
        req = request(space, ["x", "k", "mode"], parent=parent, n=3, iteration=8, candidates=candidates)
        assert propose(resumed, req) == propose(first, req)


def test_the_run_seed_is_used_unless_one_is_given():
    space = numbers_space()
    by_run_seed = [propose(TPE(), request(space, ["x"], seed=s))[0].updates for s in (1, 1, 2)]
    assert by_run_seed[0] == by_run_seed[1] != by_run_seed[2]
    fixed = [propose(TPE(seed=7), request(space, ["x"], seed=s))[0].updates for s in (1, 2)]
    assert fixed[0] == fixed[1]


def test_a_redraw_draws_again():
    space = numbers_space()
    assert propose(TPE(), request(space, ["x"], redraw=0)) != propose(TPE(), request(space, ["x"], redraw=1))


@pytest.mark.parametrize("seed, n_startup_trials", [(None, 4), (5, 2)])
def test_config_is_its_constructor_arguments(seed, n_startup_trials):
    assert TPE(seed=seed, n_startup_trials=n_startup_trials).config() == {
        "seed": seed,
        "n_startup_trials": n_startup_trials,
    }
