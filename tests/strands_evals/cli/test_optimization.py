"""Offline coverage for the bounded instruction optimization state machine."""

import json
from unittest.mock import Mock

import pytest

from strands_evals.cli._optimization import InstructionOptimizer, digest, generate_proposal


def make_optimizer(tmp_path, evaluator=None, proposer=None, **kwargs):
    path = tmp_path / "instructions.md"
    if not path.exists():
        path.write_text("original\n")
    defaults = dict(
        root=tmp_path,
        instructions_path=path,
        original=path.read_text(),
        cases=[{"input": str(i)} for i in range(8)],
        source_digest=digest("source"),
        options={},
        evaluate=evaluator or (lambda text, cases: {"overall_score": 0.5, "scores": [0.5] * len(cases)}),
        propose=proposer or (lambda text, feedback, attempt: f"candidate {attempt}\n"),
        budget=2,
    )
    defaults.update(kwargs)
    return InstructionOptimizer(**defaults)


def test_strict_gating_and_cached_resume(tmp_path):
    evaluate = Mock(side_effect=lambda text, cases: {"overall_score": 0.5, "scores": [0.5] * len(cases)})
    propose = Mock(side_effect=lambda text, feedback, attempt: f"candidate {attempt}\n")
    optimizer = make_optimizer(tmp_path, evaluate, propose)
    result = optimizer.run()
    assert result["accepted"] is False
    assert result["patch_path"] is None
    assert evaluate.call_count == 3
    assert propose.call_count == 2
    assert optimizer.run(result["run_id"]) == result
    assert evaluate.call_count == 3
    assert propose.call_count == 2
    assert (tmp_path / "instructions.md").read_text() == "original\n"


def test_holdout_gates_training_winner_and_only_evaluates_winner(tmp_path):
    optimizer = make_optimizer(tmp_path)
    train = len(optimizer.training_indices)

    def evaluate(text, cases):
        score = 0.3 if text == "original\n" else (0.9 if len(cases) == train else 0.2)
        return {"overall_score": score, "scores": [score] * len(cases)}

    optimizer.evaluate = Mock(side_effect=evaluate)
    result = optimizer.run()
    assert result["best_proposal"] == 0
    assert not result["accepted"]
    assert optimizer.evaluate.call_count == 5
    assert set(result["training_indices"]).isdisjoint(result["holdout_indices"])


def test_accepted_patch_preserves_source(tmp_path):
    def evaluate(text, cases):
        score = 0.2 if text == "original\n" else 0.8
        return {"overall_score": score, "scores": [score] * len(cases)}

    result = make_optimizer(tmp_path, evaluate).run()
    assert result["accepted"]
    assert "-original\n+candidate 0\n" in result["patch"]
    assert (tmp_path / "instructions.md").read_text() == "original\n"
    assert json.loads(open(result["manifest_path"]).read())["accepted"]


@pytest.mark.parametrize("change", [{"source_digest": "different"}, {"cases": [{}] * 9}, {"budget": 3}])
def test_resume_rejects_changed_inputs(tmp_path, change):
    result = make_optimizer(tmp_path).run()
    with pytest.raises(ValueError, match="changed"):
        make_optimizer(tmp_path, **change).run(result["run_id"])


def test_resume_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="directly under"):
        make_optimizer(tmp_path).run(str(tmp_path))


@pytest.mark.parametrize("kwargs", [{"budget": 0}, {"holdout_size": 1}, {"cases": [{}] * 5}])
def test_invalid_bounds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_optimizer(tmp_path, **kwargs)


def test_small_holdout_explicit_opt_in_and_plan(tmp_path):
    optimizer = make_optimizer(tmp_path, holdout_size=1, allow_small_holdout=True, cases=[{}, {}])
    assert optimizer.plan()["max_rollouts"] == 5
    assert optimizer.plan()["monetary_estimate"] is None


def test_failed_evaluation_is_persisted_and_not_retried(tmp_path):
    evaluate = Mock(side_effect=RuntimeError("runner failed"))
    optimizer = make_optimizer(tmp_path, evaluate)
    with pytest.raises(RuntimeError, match="runner failed"):
        optimizer.run()
    manifest = next(tmp_path.glob(".strands/runs/*/manifest.json"))
    state = json.loads(manifest.read_text())
    assert state["status"] == "failed"
    with pytest.raises(ValueError, match="cached evaluation failure"):
        optimizer.run(state["run_id"])
    assert evaluate.call_count == 1


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True])
def test_invalid_scores_cannot_win(tmp_path, score):
    with pytest.raises(ValueError, match="incomplete"):
        make_optimizer(tmp_path, lambda text, cases: {"overall_score": score, "scores": [score] * len(cases)}).run()


def test_failed_proposal_consumes_budget_slot(tmp_path):
    proposer = Mock(side_effect=RuntimeError("provider failed"))
    optimizer = make_optimizer(tmp_path, proposer=proposer)
    with pytest.raises(RuntimeError, match="provider failed"):
        optimizer.run()
    manifest = next(tmp_path.glob(".strands/runs/*/manifest.json"))
    state = json.loads(manifest.read_text())
    assert len(state["proposals"]) == 1
    with pytest.raises(ValueError, match="cached proposal"):
        optimizer.run(state["run_id"])
    assert proposer.call_count == 1


def test_model_generator_is_explicit_and_mockable(monkeypatch):
    agent = Mock(return_value=" replacement ")
    constructor = Mock(return_value=agent)
    monkeypatch.setattr("strands.Agent", constructor)
    assert generate_proposal("model-id", "original", [{"overall_score": 0.5}], 0) == "replacement"
    assert constructor.call_args.kwargs["model"] == "model-id"
    assert constructor.call_args.kwargs["callback_handler"] is None
    assert "training_feedback" in agent.call_args.args[0]


def test_interrupted_batch_reuses_completed_case_reports(tmp_path):
    seen = []
    interrupted = False

    def evaluate(text, cases):
        nonlocal interrupted
        if len(seen) == 1 and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        seen.append((text, cases))
        return {"overall_score": 0.5, "scores": [0.5]}

    optimizer = make_optimizer(tmp_path, evaluate, per_case=True)
    with pytest.raises(KeyboardInterrupt):
        optimizer.run()
    manifest = next(tmp_path.glob(".strands/runs/*/manifest.json"))
    state = json.loads(manifest.read_text())
    assert "baseline_training_case_" in next(iter(state["evaluations"]))
    optimizer.run(state["run_id"])
    assert len(seen) == 9  # three training cases each for baseline and two candidates
