import json
import random
import sys

import numpy as np
import pytest

from strands_evals.optimization import (
    AtMost,
    Criteria,
    Int,
    Maximize,
    MetricRegistry,
    OneOf,
    TextValue,
    choice,
    component,
)
from strands_evals.optimization.declaration.space import Space
from strands_evals.optimization.score_model.bayesbench import BayesBenchMetric, BayesBenchModel
from strands_evals.optimization.score_model.design import design
from strands_evals.optimization.score_model.effects import effects, effects_json
from strands_evals.optimization.score_model.protocols import FitData
from strands_evals.types.optimization import CandidateScores

# ---- a world with known effects -------------------------------------------------------------------------------------
SPACE = Space(
    {"option": OneOf("a", "b", "c", default="a"), "level": Int(1, 10, default=5), "text": TextValue(default="default")}
)
OPTION = {"a": 0.0, "b": 0.08, "c": -0.06}


def _truth(values, text_effects):
    return 0.5 + OPTION[values["option"]] + 0.005 * values["level"] + text_effects.get(values["text"], 0.0)


def _world(seed, *, n=40, cases=20, runs=3, noise=0.15, text_spread=0.0, options=True):
    """Candidates at random configurations measured on the same cases, and each one's true mean over them."""
    rng = random.Random(seed)
    case_effects = {f"case{k}": rng.gauss(0, 0.1) for k in range(cases)}
    text_effects: dict[str, float] = {}
    candidates, observations, truths = {}, {}, {}
    for i in range(n):
        if i == 0:
            values = {"option": "a", "level": 5, "text": "default"}
        else:
            values = {
                "option": rng.choice("abc") if options else "a",
                "level": rng.randint(1, 10),
                "text": f"rewrite {i}",
            }
            text_effects[values["text"]] = rng.gauss(0, text_spread) if text_spread else 0.0
        cid = f"c{i}"
        candidates[cid] = values
        observations[cid] = {
            case: [_truth(values, text_effects) + offset + rng.gauss(0, noise) for _ in range(runs)]
            for case, offset in case_effects.items()
        }
        truths[cid] = _truth(values, text_effects) + sum(case_effects.values()) / cases
    return candidates, observations, truths, list(case_effects)


def _scores(by_metric, *, ineligible=()):
    """`CandidateScores` per candidate from goal metric -> candidate id -> case -> runs; eligible unless named."""
    ids = dict.fromkeys(cid for by_candidate in by_metric.values() for cid in by_candidate)
    out = {}
    for cid in ids:
        runs: dict = {}
        for metric, by_candidate in by_metric.items():
            for case, values in by_candidate.get(cid, {}).items():
                runs.setdefault(case, {})[metric] = list(values)
        out[cid] = CandidateScores(
            candidate_id=cid,
            objective_by_case={},
            metrics_by_case={},
            unavailable={},
            objective_by_run={},
            metrics_by_run=runs,
            objective=None,
            is_eligible=cid not in ineligible,
            failures=0,
        )
    return out


def _data(candidates, observations, cases, *, space=SPACE, text_blocks=None, ranges=None, ineligible=()):
    return FitData(
        space=space,
        text_blocks={"text": ["text"]} if text_blocks is None else text_blocks,
        candidates=candidates,
        cases=cases,
        scores=_scores({"score": observations}, ineligible=ineligible),
        ranges=ranges or {"score": (None, None)},
    )


def _fit(candidates, observations, cases, **kwargs):
    return BayesBenchModel().fit(_data(candidates, observations, cases, **kwargs))


def _measured(observations, cid):
    return float(np.mean([v for runs in observations[cid].values() for v in runs]))


def test_estimate_intervals_cover_the_truth():
    covered = total = 0
    for seed in range(12):
        candidates, observations, truths, cases = _world(seed, text_spread=0.03)
        fit = _fit(candidates, observations, cases)
        for cid, values in candidates.items():
            low, high = fit.metrics["score"].at(values, cid).ci90
            covered += low <= truths[cid] <= high
            total += 1
    assert 0.8 <= covered / total <= 0.98  # 90% intervals


def test_a_candidate_measured_lucky_is_shrunk_toward_what_the_others_say():
    # No rewrite does anything: the best measured rewrite is only the luckiest.
    closer = 0
    for seed in range(12):
        candidates, observations, truths, cases = _world(seed, options=False)
        fit = _fit(candidates, observations, cases)
        luckiest = max(list(candidates)[1:], key=lambda cid: _measured(observations, cid))
        estimate = fit.metrics["score"].at(candidates[luckiest], luckiest).mean
        closer += abs(estimate - truths[luckiest]) < abs(_measured(observations, luckiest) - truths[luckiest])
    assert closer >= 10


def test_a_deterministic_metric_is_estimated_as_measured():
    candidates, observations, _, cases = _world(3, runs=2, noise=0.0, text_spread=0.03)
    fit = _fit(candidates, observations, cases)
    for cid, values in candidates.items():
        assert fit.metrics["score"].at(values, cid).mean == pytest.approx(_measured(observations, cid), abs=1e-3)


def test_effects_recover_an_option_effect():
    candidates, observations, _, cases = _world(5, n=60)
    fit = _fit(candidates, observations, cases)
    rows = {row["level"]: row for row in effects(fit) if row["key"] == "option"}
    for level in ("b", "c"):
        low, high = rows[level]["ci90"]
        assert low <= OPTION[level] <= high
    assert rows["b"]["p_positive"] > 0.95 and rows["c"]["p_positive"] < 0.05
    assert rows["b"]["reference"] == "a" and rows["b"]["metric"] == "score"


def test_a_configuration_not_fitted_is_predicted_with_its_own_effect_unknown():
    candidates, observations, _, cases = _world(2, text_spread=0.05)
    model = _fit(candidates, observations, cases).metrics["score"]
    assert model.candidate_sd > 0
    assert model.at(candidates["c5"]).sd > model.at(candidates["c5"], "c5").sd
    assert model.at(candidates["c0"], "c0").sd < model.at(candidates["c0"]).sd


def test_a_number_is_predicted_along_its_trend_within_its_bin():
    rng = random.Random(7)
    space = Space({"level": Int(1, 100, default=5)})
    cases = [f"case{k}" for k in range(6)]
    levels = [5, *rng.sample(range(1, 101), 30)]
    candidates = {f"c{i}": {"level": level} for i, level in enumerate(levels)}
    observations = {f"c{i}": {case: [0.2 + 0.005 * level] for case in cases} for i, level in enumerate(levels)}
    fit = _fit(candidates, observations, cases, space=space, text_blocks={})
    for level in (3, 50, 97):  # values never measured
        assert fit.metrics["score"].at({"level": level}).mean == pytest.approx(0.2 + 0.005 * level, abs=0.02)


# ---- the fit's contract ---------------------------------------------------------------------------------------------
def test_fewer_than_two_eligible_candidates_give_no_fit():
    candidates, observations, _, cases = _world(1, n=3)
    # c1 and c2 are not eligible: not in the observations.
    assert _fit(candidates, {"c0": observations["c0"]}, cases) is None


def test_only_eligible_candidates_are_fitted():
    candidates, observations, _, cases = _world(1, n=6)
    del observations["c3"]  # no scores at all
    fit = BayesBenchModel().fit(_data(candidates, observations, cases, ineligible={"c4"}))
    assert fit.candidates == frozenset(candidates) - {"c3", "c4"}
    assert fit.metrics["score"].posterior.n == 4 * len(cases)  # c4's observations are left out too


def test_each_goal_metric_is_fitted_from_the_runs_with_a_value():
    candidates, observations, _, cases = _world(1, n=5)
    tokens = {cid: {case: [100.0 + i, None] for case in cases} for i, cid in enumerate(candidates)}
    data = FitData(
        space=SPACE,
        text_blocks={"text": ["text"]},
        candidates=candidates,
        cases=cases,
        scores=_scores({"score": observations, "tokens": tokens}),
        ranges={"score": (None, None), "tokens": (0.0, None)},
    )
    fit = BayesBenchModel().fit(data)
    assert set(fit.metrics) == {"score", "tokens"}
    assert fit.metrics["tokens"].unit == pytest.approx(200.0)  # twice the baseline's 100, the None run dropped


def test_a_goal_metric_no_eligible_candidate_has_raises():
    candidates, observations, _, cases = _world(1, n=4)
    data = FitData(
        space=SPACE,
        text_blocks={"text": ["text"]},
        candidates=candidates,
        cases=cases,
        scores=_scores({"score": observations}),
        ranges={"score": (None, None), "tokens": (0.0, None)},
    )
    with pytest.raises(ValueError, match="no eligible candidate has a value of tokens"):
        BayesBenchModel().fit(data)


def test_fewer_than_two_eligible_candidates_give_no_fit_whatever_their_scores():
    candidates, observations, _, cases = _world(1, n=4)
    assert BayesBenchModel().fit(_data(candidates, observations, cases, ineligible={"c1", "c2", "c3"})) is None


def test_the_fit_is_deterministic():
    candidates, observations, _, cases = _world(4, n=12, text_spread=0.03)
    first, second = _fit(candidates, observations, cases), _fit(candidates, observations, cases)
    for cid, values in candidates.items():
        assert first.estimates(values, cid) == second.estimates(values, cid)
    assert first.diagnostics == second.diagnostics
    assert effects(first) == effects(second)


def _chain_data(candidates, observations, cases, n):
    ids = list(candidates)[:n]
    return _data({c: candidates[c] for c in ids}, {c: observations[c] for c in ids}, cases)


def test_the_held_out_checks_carry_forward_without_changing_the_previous_fit():
    candidates, observations, _, cases = _world(6, n=18)
    model = BayesBenchModel()
    first = model.fit(_chain_data(candidates, observations, cases, 6))
    second = model.fit(_chain_data(candidates, observations, cases, 12), previous=first)
    assert first.diagnostics["score"]["n_holdout"] == 0
    assert second.checks["score"].n_holdout == 6  # the candidates new since the first fit
    third = model.fit(_chain_data(candidates, observations, cases, 18), previous=second)
    third_again = model.fit(_chain_data(candidates, observations, cases, 18), previous=second)
    assert second.checks["score"].n_holdout == 6  # refitting from `second` twice never changes its checks
    assert third.checks["score"].n_holdout == third_again.checks["score"].n_holdout == 12
    assert third.diagnostics == third_again.diagnostics


def test_the_held_out_check_flags_a_model_whose_predictions_keep_missing():
    # Later candidates do systematically better than any option explains, so the checks fill a window and flag.
    rng = random.Random(4)
    space = Space({"mode": OneOf("a", "b", default="a"), "k": Int(1, 10, default=3)})
    candidates = {"c0": {"mode": "a", "k": 3}}
    candidates.update({f"c{i}": {"mode": rng.choice("ab"), "k": rng.randint(1, 10)} for i in range(1, 28)})
    cases = ["x", "y", "z"]
    observations = {}
    for i, (cid, v) in enumerate(candidates.items()):
        drift = 0.012 * i + rng.gauss(0, 0.08)
        observations[cid] = {c: [0.3 + 0.05 * (v["mode"] == "b") + drift + rng.gauss(0, 0.02)] for c in cases}
    model, fit = BayesBenchModel(), None
    for n in (2, 4, 8, 12, 16, 20):
        ids = list(candidates)[:n]
        data = _data(
            {c: candidates[c] for c in ids},
            {c: observations[c] for c in ids},
            cases,
            space=space,
            text_blocks={},
            ranges={"score": (0.0, 1.0)},
        )
        fit = model.fit(data, previous=fit)
    assert fit.diagnostics["score"]["n_holdout"] == 18
    assert fit.flagged == ["score"]
    out = effects_json(fit, CRITERIA)
    assert out["reason"] == "flagged: score" and out["effects"] == []


def test_the_held_out_check_of_an_unbounded_metric_ignores_a_change_of_scale():
    # Only the baseline's runs differ, which moves this fit's unit (twice the baseline's mean): the new candidates'
    # z-scores, against the earlier fit's prediction, must not move with it.
    candidates, observations, _, cases = _world(3, n=12)
    model = BayesBenchModel()
    first = model.fit(_chain_data(candidates, observations, cases, 6))
    shifted = {**observations, "c0": {case: [*runs, 2 * runs[0]] for case, runs in observations["c0"].items()}}
    same_scale = model.fit(_data(candidates, observations, cases), previous=first)
    new_scale = model.fit(_data(candidates, shifted, cases), previous=first)
    assert new_scale.metrics["score"].unit != pytest.approx(same_scale.metrics["score"].unit)
    for key in ("n_holdout", "z_mean", "z_sq_sum"):
        assert new_scale.diagnostics["score"][key] == pytest.approx(same_scale.diagnostics["score"][key])


def test_an_unbounded_metric_is_scaled_by_twice_the_baselines_mean():
    candidates, observations, _, cases = _world(1, n=5)
    fit = _fit(candidates, observations, cases)
    baseline = np.mean([np.mean(runs) for runs in observations["c0"].values()])
    assert fit.metrics["score"].offset == 0.0
    assert fit.metrics["score"].unit == pytest.approx(2 * baseline)
    bounded = _fit(candidates, observations, cases, ranges={"score": (-1.0, 3.0)})
    assert (bounded.metrics["score"].offset, bounded.metrics["score"].unit) == (-1.0, 4.0)
    assert bounded.diagnostics["score"]["scale"] == {"unit": 4.0, "offset": -1.0}


def test_a_ratio_against_a_reference_whose_mean_is_zero_is_none():
    candidates, observations, _, cases = _world(1, n=4)
    model = _fit(candidates, observations, cases).metrics["score"]

    class Posterior:  # the reference's mean is exactly 0
        def predictive(self, rows):
            return np.array([1.0, 0.0]), np.array([0.01, 0.01])

        def covariance(self, a, b):
            return 0.0

    zero = BayesBenchMetric(model.design, Posterior(), 0.0, 1.0, model.noise, model.candidate_sd)
    assert zero.ratio(candidates["c1"], candidates["c0"], candidate="c1", reference_candidate="c0") is None
    ratio = model.ratio(candidates["c1"], candidates["c0"], candidate="c1", reference_candidate="c0")
    at = model.at(candidates["c1"], "c1").mean / model.at(candidates["c0"], "c0").mean
    assert ratio.mean == pytest.approx(at)


def test_sampled_effects_center_on_the_posterior():
    candidates, observations, _, cases = _world(8, n=15)
    model = _fit(candidates, observations, cases).metrics["score"]
    draws = model.sample_effects(4000, np.random.default_rng(0))
    assert draws.shape == (4000, model.design.p)
    under = model.value_under(draws, candidates["c3"])
    assert np.mean(under) == pytest.approx(model.at(candidates["c3"]).mean, abs=0.01)


# ---- the design -----------------------------------------------------------------------------------------------------
class Window:
    def __init__(self, size: int = 10):
        self.size = size


class Summary:
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio


def test_the_design_lists_factors_inactive_levels_and_omitted_keys():
    space = Space(
        {
            "cm": choice(window=component(Window, size=Int(2, 40)), off=None),
            "fixed": component(Summary),
            "k": Int(1, 9, default=3),
            "prompt": TextValue(default="Be helpful."),
        }
    )
    candidates = {
        "c0": {"cm": "window", "cm.window.size": 10, "fixed": "Summary", "k": 3, "prompt": "Be helpful."},
        "c1": {"cm": "off", "fixed": "Summary", "k": 3, "prompt": "Be brief."},
        "c2": {"cm": "window", "cm.window.size": 30, "fixed": "Summary", "k": 3, "prompt": "Be helpful."},
    }
    observations = {cid: {"x": [0.5], "y": [0.6]} for cid in candidates}
    fit = _fit(
        candidates, observations, ["x", "y"], space=space, text_blocks={"text": ["prompt"]}, ranges={"score": (0, 1)}
    )
    described = design(fit)
    factors = {f["key"]: f for f in described["factors"]}
    assert factors["cm"] == {"key": "cm", "kind": "choice", "levels": ["window", "off"], "reference": "window"}
    assert factors["cm.window.size"]["kind"] == "range"
    assert factors["cm.window.size"]["levels"][-1] == "inactive"
    assert factors["rewritten:text"] == {
        "key": "rewritten:text",
        "kind": "text",
        "levels": ["default", "rewritten"],
        "reference": "default",
    }
    assert "fixed" not in factors  # a bare component can never change
    assert described["omitted"] == [{"key": "k", "why": "one value"}]
    assert {row["level"] for row in effects(fit)} == {"off", "rewritten", *factors["cm.window.size"]["levels"][1:-1]}


def test_another_models_fit_has_no_design_or_effects():
    class Other:
        metrics, candidates, flagged, diagnostics = {}, frozenset(), [], {}

        def estimates(self, values, candidate=None):
            return {}

    assert design(Other()) == {"factors": [], "omitted": []}
    assert effects(Other()) == []


# ---- effects.json ---------------------------------------------------------------------------------------------------
CRITERIA = Criteria(
    Maximize("score"), [AtMost("score", 2.0)], MetricRegistry({"score": (None, None)}), agent_failure_score=None
)


def test_effects_json_holds_effects_design_goals_and_metrics():
    candidates, observations, _, cases = _world(5, n=20)
    fit = _fit(candidates, observations, cases)
    out = effects_json(fit, CRITERIA)
    json.dumps(out)  # serializable
    assert out["reason"] is None
    assert out["effects"] == effects(fit)
    assert out["factors"] == design(fit)["factors"]
    assert out["goals"] == [
        {"role": "objective", "index": 0, "metric": "score", "better": "higher"},
        {"role": "constraint", "index": 0, "metric": "score", "better": "lower"},
    ]
    assert out["metrics"]["score"]["observations"] == 20 * len(cases)
    assert set(out["metrics"]["score"]) == {
        "scale",
        "noise",
        "observations",
        "diagnostics",
        "prior_scales",
        "prior_scale_weights",
    }


def test_effects_json_says_why_it_has_no_effects():
    assert effects_json(None, CRITERIA)["reason"] == "no fit: fewer than 2 eligible candidates"
    raised = effects_json(None, CRITERIA, error="LinAlgError: singular")
    assert raised["reason"] == "the model raised: LinAlgError: singular"
    assert raised["effects"] == raised["factors"] == raised["omitted"] == []

    class Other:
        metrics, candidates, flagged, diagnostics = {}, frozenset(), [], {}

    assert effects_json(Other(), CRITERIA)["reason"] == "effects need a BayesBenchModel fit"


def test_a_flagged_fit_gives_no_effects_but_keeps_its_diagnostics():
    candidates, observations, _, cases = _world(5, n=8)
    fit = _fit(candidates, observations, cases)
    fit.diagnostics["score"]["flagged"] = True
    out = effects_json(fit, CRITERIA)
    assert out["reason"] == "flagged: score"
    assert out["effects"] == out["factors"] == []
    assert out["metrics"]["score"]["diagnostics"]["flagged"] is True


def test_without_the_extra_the_model_cant_be_built(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ImportError, match=r"pip install \"strands-agents-evals\[optimization\]\""):
        BayesBenchModel()


def test_config_is_empty():
    assert BayesBenchModel().config() == {}
    assert BayesBenchModel.name == "bayesbench"
