import asyncio
import random
from collections import Counter

import pytest

from strands_evals.optimization import (
    AtMost,
    Criteria,
    Maximize,
    MetricRegistry,
    OneOf,
    Space,
    Thompson,
    WeightedSum,
    choice,
    component,
)
from strands_evals.optimization.candidate import Candidate, content_id
from strands_evals.optimization.score_model.bayesbench import BayesBenchModel
from strands_evals.optimization.score_model.protocols import FitData
from strands_evals.optimization.search.proposer import ProposalRequest
from strands_evals.types.optimization import CandidateScores, CandidateSummary, Observation

REGISTRY = MetricRegistry({"A": (0.0, 1.0), "B": (0.0, 1.0)})
# `level` is outside the block: measuring each option at several levels lets the fit tell an option's effect from a
# candidate's own.
SPACE = Space(
    {
        "option": OneOf("p", "q", "r", default="p"),
        "other": OneOf("x", "y", default="x"),
        "level": OneOf(1, 2, 3, 4, default=1),
    }
)
KEYS = ["option", "other"]


class Window:
    def __init__(self, size: int = 10, pin: int = 0):
        self.size, self.pin = size, pin


class Summary:
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio


def world(space, truths, *, cases=12, runs=2, noise=0.03, seed=0):
    """Measured candidates from configuration -> metric -> true mean: their summaries and a fit on them."""
    rng = random.Random(seed)
    case_effects = {f"case{k}": rng.gauss(0, 0.05) for k in range(cases)}
    candidates, scores, summaries = {}, {}, []
    for partial, means in truths:
        values = space.complete(partial)
        cid = content_id(values)
        candidates[cid] = values
        runs_by_case = {
            case: {m: [mean + effect + rng.gauss(0, noise) for _ in range(runs)] for m, mean in means.items()}
            for case, effect in case_effects.items()
        }
        scores[cid] = CandidateScores(
            candidate_id=cid,
            objective_by_case={},
            metrics_by_case={},
            unavailable={},
            objective_by_run={},
            metrics_by_run=runs_by_case,
            objective=None,
            is_eligible=True,
            failures=0,
        )
        summaries.append(summary(cid, values))
    metrics = list(truths[0][1])
    data = FitData(
        space=space,
        text_blocks={},
        candidates=candidates,
        cases=list(case_effects),
        scores=scores,
        ranges={m: REGISTRY.range(m) for m in metrics},
    )
    return BayesBenchModel().fit(data), summaries


def summary(cid, values, *, is_eligible=True):
    return CandidateSummary(
        candidate_id=cid,
        parent_id=None,
        origin="test",
        values=dict(values),
        number=0,
        measured_objective=None,
        measured={},
        trial_count=1,
        ranking_objective=None,
        term_values=None,
        rank=None,
        estimates={},
        violations=None,
        is_feasible=None,
        is_eligible=is_eligible,
        changed_keys=[],
    )


def request(space, keys, *, fit=None, candidates=(), criteria=None, n=1, iteration=1, redraw=0, seed=0, parent=None):
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
        fit=fit,
        candidates=list(candidates),
        n=n,
        iteration=iteration,
        seed=seed,
        redraw=redraw,
        criteria=criteria or Criteria(Maximize("A"), [], REGISTRY),
        log=lambda *args, **kwargs: None,
    )


def propose(thompson, req):
    return asyncio.run(thompson.propose(req))


def observe_all(thompson, patches):
    for patch in patches:
        thompson.observe(
            Observation(
                patch=patch,
                candidate_id="id",
                values=dict(patch.updates),
                objective=0.5,
                term_values=[0.5],
                outcome="feasible",
                reason="",
                violations=[],
            )
        )


def picks(thompson, make_request, iterations):
    """The combination proposed at each of `iterations` iterations, each step observed before the next."""
    out = []
    for i in range(iterations):
        patches = propose(thompson, make_request(i))
        out.extend((p.updates["option"], p.updates["other"]) for p in patches)
        observe_all(thompson, patches)
    return Counter(out)


def grid(by_option, *, also=()):
    """Every option measured at other="x" and every level, with metric -> mean per option; plus `also`."""
    truths = [
        ({"option": o, "other": "x", "level": level}, {m: mean + 0.01 * level for m, mean in means.items()})
        for o, means in by_option.items()
        for level in (1, 2, 3, 4)
    ]
    return truths + list(also)


# Measured at other="x" only (and (p, y)), so (q, y) and (r, y) are unmeasured at the parent's level; q is best on A.
AT_X = grid({"p": {"A": 0.40}, "q": {"A": 0.75}, "r": {"A": 0.30}}, also=[({"option": "p", "other": "y"}, {"A": 0.42})])


def test_it_prefers_the_option_with_a_known_positive_effect():
    fit, summaries = world(SPACE, AT_X)
    counts = picks(Thompson(), lambda i: request(SPACE, KEYS, fit=fit, candidates=summaries, iteration=i), 20)
    assert counts[("q", "y")] >= 16


def test_it_never_proposes_a_measured_combination_while_one_is_unmeasured():
    fit, summaries = world(SPACE, AT_X)
    measured = {(s.values["option"], s.values["other"]) for s in summaries if s.values["level"] == 1}
    for i in range(10):
        patches = propose(Thompson(), request(SPACE, KEYS, fit=fit, candidates=summaries, iteration=i, n=2))
        assert len(patches) == 2
        assert not {(p.updates["option"], p.updates["other"]) for p in patches} & measured


def test_once_every_combination_is_measured_it_proposes_the_draws_best_one():
    truths = grid({"p": {"A": 0.35}, "q": {"A": 0.75}, "r": {"A": 0.35}})
    truths += [({"option": o, "other": "y"}, {"A": 0.75 if o == "q" else 0.35}) for o in "pqr"]
    fit, summaries = world(SPACE, truths)
    counts = picks(Thompson(), lambda i: request(SPACE, KEYS, fit=fit, candidates=summaries, iteration=i), 20)
    assert sum(counts.values()) == 20
    assert counts[("q", "x")] + counts[("q", "y")] >= 16


def test_a_steps_proposals_differ_in_one_call_and_across_calls():
    fit, summaries = world(SPACE, AT_X)
    thompson = Thompson()
    one_call = propose(thompson, request(SPACE, KEYS, fit=fit, candidates=summaries, n=2))
    next_call = propose(thompson, request(SPACE, KEYS, fit=fit, candidates=summaries))  # same step, not yet observed
    combinations = [tuple(p.updates.values()) for p in one_call + next_call]
    assert sorted(combinations[:2]) == [("q", "y"), ("r", "y")]  # the two unmeasured ones, each once
    assert len(combinations) == 3  # then, none fresh, any combination


def test_the_same_request_proposes_the_same_patches():
    fit, summaries = world(SPACE, AT_X)
    req = request(SPACE, KEYS, fit=fit, candidates=summaries, n=2, iteration=3)
    assert propose(Thompson(), req) == propose(Thompson(), req)
    assert propose(Thompson(seed=5), req) == propose(
        Thompson(seed=5), request(SPACE, KEYS, fit=fit, candidates=summaries, n=2, iteration=3, seed=9)
    )


def test_two_blocks_asked_in_one_step_draw_differently():
    space = Space({"a": OneOf("p", "q", "r", "s", default="p"), "b": OneOf("p", "q", "r", "s", default="p")})
    same = 0
    for i in range(20):
        (pa,) = propose(Thompson(), request(space, ["a"], iteration=i))
        (pb,) = propose(Thompson(), request(space, ["b"], iteration=i))
        same += pa.updates["a"] == pb.updates["b"]
    assert same < 12  # one shared stream would match on all 20 (expected about 5 of 20 when independent)


def test_without_a_usable_fit_it_picks_an_unmeasured_combination_at_random():
    fit, summaries = world(SPACE, AT_X)

    class Flagged:
        flagged = ["A"]

    measured = {(s.values["option"], s.values["other"]) for s in summaries if s.values["level"] == 1}
    for fit_given in (None, Flagged()):
        seen = Counter()
        for i in range(30):
            patches = propose(Thompson(), request(SPACE, KEYS, fit=fit_given, candidates=summaries, iteration=i))
            seen.update((p.updates["option"], p.updates["other"]) for p in patches)
            assert patches[0].rationale.startswith("Thompson: a combination at random")
        assert set(seen) == {("q", "y"), ("r", "y")} and not set(seen) & measured


def test_it_prefers_a_combination_that_meets_the_drawn_constraints():
    # q is best on A but far over the limit on B
    truths = grid(
        {"p": {"A": 0.40, "B": 0.20}, "q": {"A": 0.80, "B": 0.90}, "r": {"A": 0.60, "B": 0.20}},
        also=[({"option": "p", "other": "y"}, {"A": 0.42, "B": 0.22})],
    )
    fit, summaries = world(SPACE, truths)
    criteria = Criteria(Maximize("A"), [AtMost("B", 0.5)], REGISTRY)
    counts = picks(
        Thompson(), lambda i: request(SPACE, KEYS, fit=fit, candidates=summaries, criteria=criteria, iteration=i), 20
    )
    assert counts[("r", "y")] >= 16


def test_random_weights_aim_at_other_trade_offs_and_declared_weights_do_not():
    truths = grid(
        {"p": {"A": 0.80, "B": 0.20}, "q": {"A": 0.20, "B": 0.80}, "r": {"A": 0.40, "B": 0.40}},
        also=[({"option": "r", "other": "y"}, {"A": 0.41, "B": 0.41})],
    )
    fit, summaries = world(SPACE, truths)
    criteria = Criteria(WeightedSum(Maximize("A", weight=10), Maximize("B")), [], REGISTRY)

    def make(i):
        return request(SPACE, KEYS, fit=fit, candidates=summaries, criteria=criteria, iteration=i)

    declared = picks(Thompson(random_weights=False), make, 20)
    assert declared == Counter({("p", "y"): 20})
    random_weights = picks(Thompson(), make, 20)
    assert random_weights[("q", "y")] >= 3 and random_weights[("p", "y")] >= 3


def test_a_combination_counts_once_by_the_keys_active_under_it():
    space = Space({"manager": choice(window=component(Window, pin=OneOf(0, 1)), summary=component(Summary))})
    keys = ["manager", "manager.window.pin"]
    measured_summary = space.complete({"manager": "summary"})
    summaries = [summary(content_id(measured_summary), measured_summary)]
    patches = propose(Thompson(), request(space, keys, candidates=summaries, n=2))
    assert sorted(p.updates["manager.window.pin"] for p in patches) == [0, 1]
    assert all(p.updates["manager"] == "window" for p in patches)
    # Every combination fresh is taken, so the third falls back to all three: summary once, not once per pin.
    (third,) = propose(Thompson(), request(space, keys, candidates=summaries, n=3))[2:]
    assert ("manager.window.pin" in third.updates) == (third.updates["manager"] == "window")


def test_options_python_compares_equal_stay_apart():
    space = Space({"flag": OneOf(1, True, default=1)})
    measured = space.complete({"flag": 1})
    patches = propose(Thompson(), request(space, ["flag"], candidates=[summary(content_id(measured), measured)]))
    assert [(type(p.updates["flag"]), p.updates["flag"]) for p in patches] == [(bool, True)]


def test_held_values_python_compares_equal_stay_apart():
    space = Space({"option": OneOf("p", "q", default="p"), "flag": OneOf(1, True, default=1)})
    parent = Candidate(space.complete({"flag": True}), None, "baseline", "")
    at_one, at_true = space.complete({"option": "q", "flag": 1}), space.complete({"option": "q", "flag": True})

    def options(measured):
        listed = [summary(content_id(measured), measured)]
        reqs = (request(space, ["option"], candidates=listed, parent=parent, iteration=i) for i in range(20))
        return {propose(Thompson(), req)[0].updates["option"] for req in reqs}

    assert options(at_one) == {"p", "q"}  # measured under flag=1, not under the parent's flag=True: still fresh
    assert options(at_true) == {"p"}


def test_nothing_to_propose_when_no_block_key_is_active():
    space = Space({"manager": choice(window=component(Window, pin=OneOf(0, 1)), summary=component(Summary))})
    parent = Candidate(space.complete({"manager": "summary"}), None, "baseline", "")
    assert propose(Thompson(), request(space, ["manager.window.pin"], parent=parent)) == []


def test_many_combinations_are_sampled_down():
    space = Space({f"k{i}": OneOf(0, 1, default=0) for i in range(10)})  # 1,024 combinations
    thompson = Thompson()
    thompson.MAX_COMBINATIONS = 8
    (patch,) = propose(thompson, request(space, list(space._params)))
    space.validate(space.complete(patch.updates))


@pytest.mark.parametrize("seed, random_weights", [(None, True), (3, False)])
def test_config_is_its_constructor_arguments(seed, random_weights):
    assert Thompson(seed=seed, random_weights=random_weights).config() == {
        "seed": seed,
        "random_weights": random_weights,
    }


@pytest.mark.parametrize("kwargs", [{"seed": "1"}, {"seed": True}, {"random_weights": 1}])
def test_bad_arguments_raise(kwargs):
    with pytest.raises(TypeError):
        Thompson(**kwargs)
