import copy

import pytest

from strands_evals.optimization.record.events import EventLog
from strands_evals.optimization.record.resume import FitReplay, load_state
from strands_evals.optimization.record.run_directory import RunDirectory
from strands_evals.types.optimization import Patch, Usage

SPACE = [
    {"key": "cm", "kind": "choice", "options": ["window", "summary"]},
    {"key": "cm.window.window_size", "kind": "int", "min_value": 4, "max_value": 40},
    # A file-backed text: run.json keeps its baseline as sha256 + length, never the text.
    {
        "key": "rules",
        "kind": "text",
        "default": {"sha256": "a" * 64, "length": 53},
        "source": {"path": "/project/agent_files/rules.md", "pointer": None, "as_path": True},
    },
]
CONFIG = {
    "cases": ["a", "b"],
    "space": SPACE,
    "objective": "Maximize('Helpfulness')",
    "constraints": [],
    "search": {"blocks": [{"name": "structure"}, {"name": "numbers"}], "redraws": 3},
    "selection": {"name": "top_ranked", "config": {}},
    "score_model": {"name": "BayesBenchModel", "config": {}},
    "budget": {"max_rollouts": 500},
    "max_workers": 4,
}


def _usage(tokens):
    return Usage(
        tokens=tokens,
        input_tokens=tokens,
        output_tokens=0,
        cache_read_tokens=None,
        cache_write_tokens=None,
        model_calls=1,
        seconds=1.0,
        models=[],
    )


def _patch(x, tokens=None):
    return Patch(updates={"x": x}, rationale="r", usage=None if tokens is None else _usage(tokens))


def _observation(candidate_id, patch, *, source_id=None):
    return {
        "patch": patch,
        "candidate_id": candidate_id,
        "values": {"x": 1},
        "source_id": source_id,
        "objective": 0.5,
        "term_values": None,
        "outcome": "feasible",
        "reason": "",
        "violations": None,
    }


def _trial(candidate_id, case_name, *, rollouts=1, previous=0):
    attempt = {"rollouts": 1}
    return {
        "candidate_id": candidate_id,
        "case_name": case_name,
        "run_index": 0,
        "rollouts": rollouts,
        "previous_attempts": [attempt] * previous,
    }


def _iteration_end(log, iteration, parent, blocks, *, exhausted=(), round=None, role="inner"):
    record = {
        "iteration": iteration,
        "parent_id": parent,
        "round": round,
        "role": role,
        "blocks": blocks,
        "n_candidates": 1,
        "proposals": [],
        "rollouts_spent": 0,
    }
    log.append("iteration_end", iteration=iteration, record=record, exhausted=list(exhausted), best_after=[])


def _propose(log, candidate_id, parent, origin, *, rescore=False, **fields):
    log.append(
        "propose",
        candidate_id=candidate_id,
        number=0,
        parent_id=parent,
        origin=origin,
        values={"x": 1},
        rationale="",
        inherited={},
        defaulted=[],
        rescore=rescore,
        **fields,
    )


@pytest.fixture
def run(tmp_path):
    """A run killed in iteration 3, with two nested rounds open and a fit that raised."""
    run = RunDirectory.create(tmp_path, CONFIG)
    log = run.events
    # Setup: the baseline.
    _propose(log, "base", None, "baseline", iteration=0)
    log.append("trial", record=_trial("base", "a"))
    log.append("trial", record=_trial("base", "b", previous=1))
    log.append("fit", candidates=["base"], is_fitted=False, error=None, flagged=[], diagnostics={})
    # Iteration 1: the outer Nested's opener, observed at its round's end.
    outer = _patch("outer", tokens=5)
    log.append("select", iteration=1, parent_id="base", strategy="top_ranked")
    _propose(log, "c1", "base", "structure", round=[0], role="opener", patch=outer)
    log.append("trial", record=_trial("c1", "a"))
    log.append("fit", candidates=["base", "c1"], is_fitted=False, error="ValueError: boom", flagged=[], diagnostics={})
    log.append("evidence", block="numbers", evidence=_evidence("c1"))
    _iteration_end(log, 1, "base", ["structure"], round=(0,), role="opener")
    # Iteration 2: the inner Nested's opener repeats a measured candidate (a duplicate), and is still open.
    inner = _patch("inner")
    log.append("select", iteration=2, parent_id="c1", strategy="top_ranked")
    log.append("duplicate", block="numbers", redraw=0, candidate_id="base", patch=inner, round=[0, 0])
    _propose(log, "base", "base", "numbers", rescore=True)
    log.append("observe", block="numbers", candidate_id=None, observation=_observation(None, _patch(9, tokens=7)))
    log.append("proposer_usage", block="numbers", redraw=0, usage=_usage(3))  # a call that returned no patch
    log.append("fit", candidates=["base", "c1"], is_fitted=True, error=None, flagged=[], diagnostics={})
    _iteration_end(log, 2, "c1", ["numbers"], exhausted=["numbers"], round=(0, 0), role="opener")
    # Iteration 3, killed: its search events are dropped, its trials stay.
    log.append("select", iteration=3, parent_id="c1", strategy="top_ranked")
    _propose(log, "c3", "c1", "numbers")
    log.append("observe", block="structure", candidate_id="c1", observation=_observation("c1", outer, source_id="c1"))
    log.append("trial", record=_trial("c3", "a"))
    log.append("proposer_usage", block="numbers", redraw=0, usage=_usage(100))
    return run


def _evidence(candidate_id):
    return {
        "candidate_id": candidate_id,
        "values": {"x": 1},
        "origin": "structure",
        "objective": 0.4,
        "term_values": None,
        "violations": None,
        "outcome": "feasible",
    }


def test_load_state_rebuilds_the_search_up_to_the_last_closed_iteration(run):
    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    assert state.iteration == 2
    assert state.candidates == ["base", "c1"]
    assert state.parents == {"base": None, "c1": "base"}
    assert state.history == ["base", "c1"]
    assert [r.iteration for r in state.iterations] == [1, 2]
    assert state.exhausted == {"numbers"}
    assert state.fits == [
        FitReplay(["base"], False, None),
        FitReplay(["base", "c1"], False, "ValueError: boom"),
        FitReplay(["base", "c1"], True, None),
    ]
    assert [type(f).__name__ for f in state.feedback["numbers"]] == ["Evidence", "Observation"]
    assert "structure" not in state.feedback  # the outer opener's observe was in the killed iteration
    assert state.search_cost.tokens == 10  # the closed iterations' observe (7) and proposer_usage (3) events
    assert state.stop_reason is None
    assert state.finalists is None
    assert state.is_test_done is False


def test_load_state_counts_every_trial_in_spent(run):
    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    # base a + base b (and its earlier attempt) + c1 a + the killed iteration's c3 a.
    assert state.spent == 5


def test_resume_parses_the_log_once_until_the_next_append(run, monkeypatch):
    parses = []
    parse = EventLog._parse
    monkeypatch.setattr(EventLog, "_parse", lambda self: parses.append(1) or parse(self))

    reopened = RunDirectory.open(run.root)
    load_state(reopened, config=CONFIG)
    assert len(parses) == 1

    reopened.events.append("resume", spent=5, budget={}, code={})
    assert reopened.events.read()[-1]["event"] == "resume"  # an append makes the next read parse again
    assert len(parses) == 2


def test_two_nested_rounds_stay_open_outermost_first(run):
    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    assert [(r.round, r.opener_id, r.block) for r in state.open_rounds] == [
        ((0,), "c1", "structure"),
        ((0, 0), "base", "numbers"),
    ]
    assert state.open_rounds[0].patch == _patch("outer", tokens=5)


def test_an_openers_observe_closes_its_round(run):
    _iteration_end(run.events, 3, "c1", ["numbers"])

    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    assert [r.round for r in state.open_rounds] == [(0, 0)]
    assert state.candidates == ["base", "c1", "c3"]
    assert state.feedback["structure"][0].source_id == "c1"


def test_an_unscored_opener_has_no_opener_id(tmp_path):
    run = RunDirectory.create(tmp_path, CONFIG)
    _propose(run.events, "c1", "base", "structure", round=[0], role="opener", patch=_patch("outer"))
    run.events.append("unscored", candidate_id="c1", pending=2, remaining=1, reserve=0)
    _iteration_end(run.events, 1, "base", ["structure"])

    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    assert [(r.round, r.opener_id) for r in state.open_rounds] == [((0,), None)]


def test_a_second_resume_does_not_replay_the_first_kills_iteration(run):
    log = run.events
    log.append("resume", spent=5, budget={}, code={})
    log.append("select", iteration=3, parent_id="base", strategy="top_ranked")
    _iteration_end(log, 3, "base", ["structure"])

    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    assert state.history == ["base", "c1", "base"]
    assert state.candidates == ["base", "c1"]
    assert state.iteration == 3


def test_a_stop_keeps_its_partial_iteration_and_the_end_phase_progress(run):
    log = run.events
    log.append("stop", reason="budget", message="")
    log.append("finalists", candidate_ids=["c3", "base"])
    log.append("picked", report=_validation("c3"))

    state = load_state(RunDirectory.open(run.root), config=CONFIG)

    assert state.stop_reason == "budget"
    assert state.candidates == ["base", "c1", "c3"]
    assert state.history[-1] == "c1"
    assert state.finalists == ["c3", "base"]
    assert state.validation.champion_id == "c3"
    assert state.is_test_done is False

    log.append("test", report=None, reason="no test cases", front={})
    state = load_state(RunDirectory.open(run.root), config=CONFIG)
    assert state.is_test_done is True
    assert state.test is None
    assert state.test_reason == "no test cases"
    assert state.front_test == {}


def _validation(champion):
    return {
        "champion_id": champion,
        "front": [],
        "objectives": {champion: 0.9},
        "estimates": {},
        "infeasible": {},
        "measured": {},
        "problem": None,
        "reason": "highest estimate",
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda c: c.update(objective="Minimize('tokens')"),
            r"^objective changed: \"Maximize\('Helpfulness'\)\" -> \"Minimize",
        ),
        (
            lambda c: c["space"][1].update(max_value=50),
            r"^space\['cm\.window\.window_size'\]\.max_value changed: 40 -> 50; start a new run",
        ),
        (lambda c: c["space"].pop(1), r"^space\['cm\.window\.window_size'\] changed: \{.* -> None"),
        (
            lambda c: c["space"][2]["default"].update(sha256="b" * 64),
            r"^space\['rules'\]\.default\.sha256 changed: 'a{64}' -> 'b{64}'; start a new run",
        ),
        (
            lambda c: c["selection"].update(name="random_top_k"),
            r"^selection\.name changed: 'top_ranked' -> 'random_top_k'",
        ),
        (lambda c: c["search"]["blocks"].reverse(), r"^search\.blocks\[0\]\.name changed: 'structure' -> 'numbers'"),
        (lambda c: c.update(cases=["a", "b", "c"]), r"^cases changed: \['a', 'b'\] -> \['a', 'b', 'c'\]"),
    ],
)
def test_a_changed_field_fails_resume_naming_it(run, change, message):
    config = copy.deepcopy(CONFIG)
    change(config)

    with pytest.raises(ValueError, match=message):
        load_state(RunDirectory.open(run.root), config=config)


def test_budget_and_max_workers_may_change(run):
    config = {**CONFIG, "budget": {"max_rollouts": 900}, "max_workers": 16}

    assert load_state(RunDirectory.open(run.root), config=config).iteration == 2


def test_tuples_in_the_given_config_compare_equal_to_the_saved_lists(run):
    config = {**CONFIG, "cases": ("a", "b")}

    assert load_state(RunDirectory.open(run.root), config=config).iteration == 2
