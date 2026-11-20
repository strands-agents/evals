"""The optimize skill (`.agents/skills/optimize`) against this branch: `scripts/plan.py`'s numbers against the
`Optimizer`'s own, and the flags and log lines `SKILL.md` relies on against the CLI that prints them."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from strands_evals import Case
from strands_evals.cli import _optimize
from strands_evals.cli.commands import optimize as optimize_cmd
from strands_evals.cli.main import _build_parser
from strands_evals.evaluators import Evaluator
from strands_evals.experiment import Experiment
from strands_evals.optimization import Budget, FinalPick, Maximize, Minimize, OneOf, Optimizer, Pareto, tunable
from strands_evals.types.evaluation import EvaluationOutput

SKILL = Path(__file__).parents[3] / ".agents" / "skills" / "optimize"
SKILL_TEXT = (SKILL / "SKILL.md").read_text(encoding="utf-8")


def _load_plan() -> ModuleType:
    spec = importlib.util.spec_from_file_location("optimize_skill_plan", SKILL / "scripts" / "plan.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


plan = _load_plan()


class Score(Evaluator):
    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=1.0, test_pass=True, reason="ok")]


@tunable(temperature=OneOf(0.0, 1.0))
def make_agent(temperature):
    return temperature


def _experiment_file(tmp_path: Path, n_cases: int, n_evaluators: int = 1) -> Path:
    path = tmp_path / "experiment.json"
    cases = [{"name": f"case-{i}", "input": "q"} for i in range(n_cases)]
    path.write_text(json.dumps({"cases": cases, "evaluators": [{"evaluator_type": "Score"}] * n_evaluators}))
    return path


def _plan(tmp_path: Path, n_cases: int, *flags: str) -> dict:
    args = plan._parser().parse_args([str(_experiment_file(tmp_path, n_cases)), *flags])
    return plan.plan(args)


def _optimizer(n_cases: int, *, runs: int, finalists: int, pareto: bool, split=None, max_rollouts=10**6):
    experiment = Experiment(cases=[Case(name=f"case-{i}", input="q") for i in range(n_cases)], evaluators=[Score()])
    objective = Pareto(Maximize("Score"), Minimize("tokens", scale=1000)) if pareto else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # few cases, or a budget with no room to search
        return Optimizer(
            experiment,
            make_agent,
            objective=objective,
            runs_per_candidate=runs,
            final_pick=FinalPick(n_finalists=finalists),
            split=split,
            budget=Budget(max_rollouts=max_rollouts),
        )


@pytest.mark.parametrize(
    ("n_cases", "runs", "finalists", "pareto", "split"),
    [
        (6, 1, 3, False, None),
        (6, 1, 3, True, None),
        (7, 2, 1, False, "0.5,0.25,0.25"),
        (20, 1, 3, False, None),
        (20, 3, 4, True, "0.6,0.4,0"),
        (240, 1, 3, False, "0.5,0.25,0.25"),
    ],
)
def test_the_plan_deals_the_cases_and_reserves_the_budget_as_the_optimizer_does(
    tmp_path, n_cases, runs, finalists, pareto, split
):
    flags = ["--runs", str(runs), "--finalists", str(finalists), *(["--pareto"] if pareto else [])]
    result = _plan(tmp_path, n_cases, *flags, *(["--split", split] if split else []))
    optimizer = _optimizer(
        n_cases, runs=runs, finalists=finalists, pareto=pareto, split=_optimize.parse_split(split) if split else None
    )
    dealt = optimizer._case_split
    assert result["split"] == {"train": len(dealt.train), "validation": len(dealt.validation), "test": len(dealt.test)}
    minimum = result["minimum_max_rollouts"]
    with pytest.raises(ValueError, match=f"give max_rollouts >= {minimum}$"):
        _optimizer(
            n_cases, runs=runs, finalists=finalists, pareto=pareto, split=optimizer._split, max_rollouts=minimum - 1
        )
    _optimizer(n_cases, runs=runs, finalists=finalists, pareto=pareto, split=optimizer._split, max_rollouts=minimum)


def test_the_plan_for_the_skills_six_case_example(tmp_path):
    # The optimizer's default split deals 6 cases 2/2/2; 4 steps x 4 candidates on 2 train cases, 3 finalists.
    result = _plan(tmp_path, 6)
    assert result["split"] == {"train": 2, "validation": 2, "test": 2} and result["split_flag"] is None
    assert result["rollouts"] == 2 * (1 + 4 * 4) + 2 * (3 + 1) + 2 * 2 == 46
    assert result["judge_calls"] == 46 and result["baseline_rollouts"] == 2 and result["step_rollouts"] == 8
    assert "46 rollouts" in SKILL_TEXT and "2 train / 2 validation / 2 test" in SKILL_TEXT


def test_a_cap_that_fits_fewer_steps_is_flagged(tmp_path):
    # 3/2/1: the optimizer holds back 3 + 4 x (2 + 2) + 2 x (1 + 2) = 25, and each step costs 12.
    result = _plan(tmp_path, 6, "--counts", "3,2,1", "--max-rollouts", "61")
    assert result["minimum_max_rollouts"] == 25 and result["steps_that_fit"] == 3
    assert "a cap of 61 fits 3 of the 4 steps; all 4 need a cap of 73" in result["warnings"]
    assert any(w.startswith("2 validation and 1 test cases") for w in result["warnings"])


@pytest.mark.parametrize("counts", [(4, 1, 1), (3, 2, 2), (5, 3, 0), (9, 4, 2), (1, 1, 1)])
def test_case_counts_become_shares_the_optimizer_deals_back_to_the_same_counts(tmp_path, counts):
    result = _plan(tmp_path, sum(counts), "--counts", ",".join(map(str, counts)))
    assert tuple(result["split"].values()) == counts
    dealt = _optimizer(
        sum(counts), runs=1, finalists=3, pareto=False, split=_optimize.parse_split(result["split_flag"])
    )
    assert (len(dealt._case_split.train), len(dealt._case_split.validation), len(dealt._case_split.test)) == counts


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--counts", "4,1,2"], "sum to 7, but the experiment has 6 cases"),
        (["--counts", "0,3,3"], "train and validation need a share above 0"),
        (["--split", "0.5,0.5,0.5"], "sums to 1.5"),
    ],
)
def test_values_the_optimizer_would_refuse_exit_2(tmp_path, capsys, flags, message):
    with pytest.raises(SystemExit) as stopped:
        _plan(tmp_path, 6, *flags)
    assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_check_prints_the_installed_version(capsys):
    assert plan.main(["--check"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == {"optimizer"}


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    found = {option for action in parser._actions for option in action.option_strings}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                found |= _option_strings(child)
    return found


def test_every_flag_the_skill_writes_exists():
    known = _option_strings(_build_parser()) | _option_strings(plan._parser()) | {"--version", "--input-type"}
    written = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*[a-z]", SKILL_TEXT))
    assert written - known == set()


def test_the_progress_line_the_skill_relays_is_the_one_the_cli_prints(capsys):
    quoted = re.search(r"`(strands-evals optimize: iteration N \(<blocks>\): P proposals, R rollouts)`", SKILL_TEXT)
    assert quoted is not None
    pattern = re.escape(quoted.group(1)).replace("N", r"\d+").replace("<blocks>", ".+")
    pattern = pattern.replace("P", r"\d+").replace("R", r"\d+")
    record = SimpleNamespace(iteration=2, blocks=["reflection"], proposals=[object()] * 4, rollouts_spent=12)
    _optimize.print_iteration(record)
    assert re.fullmatch(pattern, capsys.readouterr().err.strip())


def test_a_proposer_error_exits_3_with_the_continue_line_the_skill_looks_for(capsys, tmp_path):
    assert optimize_cmd._proposer_stopped(RuntimeError("model failed"), tmp_path, debug=False) == 3
    err = capsys.readouterr().err
    assert f"strands-evals optimize: continue with `strands-evals optimize resume {tmp_path}`" in err
    assert "strands-evals optimize: continue with ... resume <RUN_DIR>" in SKILL_TEXT


def test_the_run_folder_line_the_skill_looks_for(capsys, tmp_path):
    result = SimpleNamespace(summary=lambda: "summary", to_dict=dict)
    optimize_cmd._print_result(result, argparse.Namespace(format="rich"), tmp_path)
    assert f"strands-evals optimize: run folder {tmp_path}" in capsys.readouterr().err
    assert "`strands-evals optimize: run folder <RUN_DIR>`" in SKILL_TEXT


@pytest.mark.skipif(sys.platform == "win32", reason="the skill's shell lines are POSIX")
def test_the_script_runs_as_the_skill_runs_it(tmp_path):
    experiment = _experiment_file(tmp_path, 6)
    done = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "plan.py"), str(experiment), "--steps", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["rollouts"] == 2 + 2 + 2  # 0 steps: only the current agent, once per split
