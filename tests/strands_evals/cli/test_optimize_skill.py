"""The optimize skill (`.agents/skills/optimize`) against this branch: `scripts/plan.py`'s numbers against what the
`Optimizer` checks and spends, and the flags, log lines, and shell lines `SKILL.md` relies on against the CLI."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from strands_evals import Case
from strands_evals.cli import _optimize
from strands_evals.cli.commands import optimize as optimize_cmd
from strands_evals.cli.main import _build_parser, main
from strands_evals.evaluators import Evaluator
from strands_evals.experiment import Experiment
from strands_evals.optimization import (
    Budget,
    FinalPick,
    Int,
    Maximize,
    Minimize,
    Optimizer,
    Pareto,
    Schedule,
    Search,
    tunable,
)
from strands_evals.types.evaluation import EvaluationOutput
from tests.strands_evals.cli.test_optimize import _run_args, experiment_path  # noqa: F401  (a fixture)

SKILL = Path(__file__).parents[3] / ".agents" / "skills" / "optimize"
PLAN_SCRIPT = SKILL / "scripts" / "plan.py"
SKILL_TEXT = (SKILL / "SKILL.md").read_text(encoding="utf-8")
PROGRESS = "strands-evals optimize: iteration N (<blocks>): P proposals, R rollouts"
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z-]*[a-z]")


def _load_plan() -> ModuleType:
    spec = importlib.util.spec_from_file_location("optimize_skill_plan", PLAN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


plan = _load_plan()


class Score(Evaluator):
    """The agent's `size` as a score, so every new version beats the current agent (size 0) and the test check
    measures two arms, as the estimate counts."""

    async def evaluate_async(self, evaluation_case):
        return [EvaluationOutput(score=int(evaluation_case.actual_output) / 10**6, test_pass=True, reason="size")]


@tunable(size=Int(0, 10**6))  # wide, so seeded proposals don't repeat and every one is measured
def make_agent(size=0):
    return size


async def _answer(agent_factory, case):
    return str(agent_factory())


class BaselineOnly(Schedule):  # as SKILL.md's hooks.py writes it for 0 steps
    name = "baseline_only"

    def next(self, ctx):
        return None


def _experiment_file(tmp_path: Path, n_cases: int) -> Path:
    path = tmp_path / "experiment.json"
    cases = [{"name": f"case-{i}", "input": "q"} for i in range(n_cases)]
    path.write_text(json.dumps({"cases": cases, "evaluators": [{"evaluator_type": "Score"}]}))
    return path


def _plan(tmp_path: Path, n_cases: int, *flags: str) -> dict:
    return plan.plan(plan._parser().parse_args([str(_experiment_file(tmp_path, n_cases)), *flags]))


def _optimizer(n_cases: int, *, pareto: bool = False, finalists: int = 3, split: str | None = None, **kwargs):
    experiment = Experiment(cases=[Case(name=f"case-{i}", input="q") for i in range(n_cases)], evaluators=[Score()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # few cases, a budget with no room to search, a step over max_workers
        return Optimizer(
            experiment,
            make_agent,
            rollout_fn=_answer,
            objective=Pareto(Maximize("Score"), Minimize("tokens", scale=1000)) if pareto else None,
            final_pick=FinalPick(n_finalists=finalists),
            split=_optimize.parse_split(split) if split else None,
            **kwargs,
        )


# plan.py against the Optimizer


@pytest.mark.parametrize(
    ("n_cases", "runs", "finalists", "pareto", "split"),
    [
        (6, 1, 3, False, None),
        (6, 1, 3, True, None),
        (7, 2, 1, False, "0.5,0.25,0.25"),
        (20, 3, 4, True, "0.6,0.4,0"),
        (240, 1, 3, False, "0.5,0.25,0.25"),
    ],
)
def test_the_minimum_cap_is_where_the_optimizer_starts_refusing(tmp_path, n_cases, runs, finalists, pareto, split):
    flags = ["--runs", str(runs), "--finalists", str(finalists), *(["--pareto"] if pareto else [])]
    minimum = _plan(tmp_path, n_cases, *flags, *(["--split", split] if split else []))["minimum_max_rollouts"]
    common = {"pareto": pareto, "finalists": finalists, "split": split, "runs_per_candidate": runs}
    with pytest.raises(ValueError, match=f"give max_rollouts >= {minimum}$"):
        _optimizer(n_cases, budget=Budget(max_rollouts=minimum - 1), **common)
    _optimizer(n_cases, budget=Budget(max_rollouts=minimum), **common)


@pytest.mark.parametrize(("steps", "candidates"), [(2, 2), (1, 1), (0, 4)])
async def test_the_estimate_is_what_a_run_spends_when_every_proposal_is_new(tmp_path, steps, candidates):
    estimate = _plan(tmp_path, 6, "--steps", str(steps), "--candidates", str(candidates))["rollouts"]
    search = Search([], schedule=BaselineOnly()) if steps == 0 else None
    budget = Budget(max_iterations=steps) if steps else Budget()
    optimizer = _optimizer(
        6, search=search, budget=budget, proposals_per_iteration=candidates, max_workers=64, base_dir=tmp_path
    )
    result = await optimizer.run_async()
    assert result.rollouts_spent == estimate
    # Each iteration line's R is that iteration's spend, so they sum to the search's: 2 train cases, 1 run.
    assert sum(record.rollouts_spent for record in result.iterations) == steps * candidates * 2


async def test_under_pareto_the_estimate_bounds_the_spend(tmp_path):
    result = _plan(tmp_path, 6, "--steps", "1", "--candidates", "2", "--pareto")
    assert result["finalists"] == 5  # raised to the Pareto minimum; at most 2 candidates can be finalists
    optimizer = _optimizer(
        6, pareto=True, budget=Budget(max_iterations=1), proposals_per_iteration=2, max_workers=64, base_dir=tmp_path
    )
    assert (await optimizer.run_async()).rollouts_spent <= result["rollouts"] == 2 * 3 + 2 * 3 + 2 * 3


def test_the_plan_for_the_skills_six_case_example(tmp_path):
    # The optimizer's default split deals 6 cases 2/2/2; 4 steps x 4 candidates on 2 train cases, 3 finalists.
    result = _plan(tmp_path, 6)
    assert result["split"] == {"train": 2, "validation": 2, "test": 2} and result["split_flag"] is None
    assert result["rollouts"] == result["judge_calls"] == 2 * (1 + 4 * 4) + 2 * (3 + 1) + 2 * 2 == 46
    assert "about 46 rollouts and 46 judge calls" in SKILL_TEXT and "2 train / 2 validation / 2 test" in SKILL_TEXT


@pytest.mark.filterwarnings("ignore:Split:UserWarning")  # few train or test cases, on purpose
@pytest.mark.parametrize("counts", [(4, 1, 1), (3, 2, 2), (5, 3, 0), (9, 4, 2), (1, 1, 1)])
def test_case_counts_become_shares_the_optimizer_deals_back_to_the_same_counts(tmp_path, counts):
    result = _plan(tmp_path, sum(counts), "--counts", ",".join(map(str, counts)))
    cases = [Case(name=f"case-{i}", input="q") for i in range(sum(counts))]
    dealt = _optimize.parse_split(result["split_flag"]).assign(cases, run_seed=0)
    assert tuple(result["split"].values()) == (len(dealt.train), len(dealt.validation), len(dealt.test)) == counts


# plan.py's warnings and refusals


def test_a_small_test_split_says_how_many_cases_to_add(tmp_path):
    warned = _plan(tmp_path, 6, "--counts", "3,2,1")["warnings"]
    assert "2 validation and 1 test cases: the scores can be luck; 4 more cases would give 2 of each" in warned


def test_a_test_share_that_deals_no_test_case_is_flagged(tmp_path):
    result = _plan(tmp_path, 6, "--split", "0.5,0.45,0.05")
    assert result["split"]["test"] == 0
    assert any(w.startswith("3 validation and 0 test cases") for w in result["warnings"])


def test_a_split_without_test_cases_isnt_flagged(tmp_path):
    assert _plan(tmp_path, 6, "--split", "0.5,0.5,0", "--workers", "64")["warnings"] == []


def test_a_cap_that_fits_fewer_steps_or_none_is_flagged(tmp_path):
    # 3/2/1: the optimizer holds back 3 + 4 x (2 + 2) + 2 x (1 + 2) = 25, and each step costs 12.
    result = _plan(tmp_path, 6, "--counts", "3,2,1", "--max-rollouts", "61")
    assert result["minimum_max_rollouts"] == 25 and result["steps_that_fit"] == 3
    assert "a cap of 61 fits 3 of the 4 steps; all 4 need a cap of 73" in result["warnings"]
    refused = _plan(tmp_path, 6, "--counts", "3,2,1", "--max-rollouts", "24")["warnings"]
    assert "the optimizer refuses a cap below 25 rollouts" in refused


def test_a_step_wider_than_the_trials_at_once_is_flagged(tmp_path):
    assert "each step's 8 trials run 4 at a time, so steps take longer" in _plan(tmp_path, 6)["warnings"]
    assert _plan(tmp_path, 6, "--workers", "8")["warnings"] == []


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
    assert json.loads(capsys.readouterr().out) == {"optimizer": importlib.metadata.version("strands-agents-evals")}


def test_an_install_without_the_optimizer_exits_3():
    # -I -S: no site-packages and no PYTHONPATH, so strands_evals can't be imported.
    done = subprocess.run(
        [sys.executable, "-I", "-S", str(PLAN_SCRIPT), "--check"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 3 and plan.REINSTALL in done.stderr
    assert plan.REINSTALL in SKILL_TEXT  # the command the skill gives the user is the one the script prints


def test_the_script_runs_with_its_own_interpreter(tmp_path):
    done = subprocess.run(
        [sys.executable, str(PLAN_SCRIPT), str(_experiment_file(tmp_path, 6)), "--steps", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["rollouts"] == 2 + 2 + 2  # 0 steps: only the current agent, once per split


# SKILL.md against the CLI


def _help_flags(*command: str, capsys: pytest.CaptureFixture[str]) -> set[str]:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([*command, "--help"])
    return set(FLAG.findall(capsys.readouterr().out))


def test_every_flag_the_skill_writes_belongs_to_the_command_it_goes_to(capsys):
    cli = _help_flags("optimize", "run", capsys=capsys) | _help_flags("optimize", "resume", capsys=capsys)
    plan_only = set(FLAG.findall(plan._parser().format_help())) - cli
    node = {"--version", "--input-type"}
    lines = SKILL_TEXT.splitlines()
    for number, line in enumerate(lines):
        for flag in FLAG.findall(line):
            if flag in plan_only:  # only on a plan.py line, or the line continuing one
                assert "plan.py" in line or "plan.py" in lines[number - 1], (number + 1, flag)
            else:
                assert flag in cli | node, (number + 1, flag)


def test_the_progress_line_the_skill_relays_is_the_one_the_cli_prints(capsys):
    assert f"`{PROGRESS}`" in SKILL_TEXT
    pattern = re.escape(PROGRESS).replace("N", r"\d+").replace("<blocks>", ".+")
    record = SimpleNamespace(iteration=2, blocks=["reflection"], proposals=[object()] * 4, rollouts_spent=12)
    _optimize.print_iteration(record)
    assert re.fullmatch(pattern.replace("P", r"\d+").replace("R", r"\d+"), capsys.readouterr().err.strip())


def test_the_cli_names_the_run_folder_before_the_first_iteration_and_prints_each_iterations_spend(
    experiment_path,  # noqa: F811
    tmp_path,
    capsys,
):
    assert main(_run_args(experiment_path, tmp_path)) == 0
    _, err = capsys.readouterr()
    (root,) = (tmp_path / "runs").iterdir()
    lines = err.splitlines()
    first_iteration = next(i for i, line in enumerate(lines) if line.startswith("strands-evals optimize: iteration"))
    assert f"strands-evals optimize: run folder {root}" in lines[:first_iteration]
    assert "`strands-evals optimize: run folder <RUN_DIR>`" in SKILL_TEXT
    printed = [int(m) for m in re.findall(r"proposals, (\d+) rollouts", err)]
    recorded = [record["rollouts_spent"] for record in json.loads((root / "result.json").read_text())["iterations"]]
    assert printed == recorded and len(printed) >= 2  # 2 steps; an iteration that saved nothing isn't a step


def test_a_proposer_error_exits_3_with_the_continue_line_the_skill_looks_for(capsys, tmp_path):
    assert optimize_cmd._proposer_stopped(RuntimeError("model failed"), tmp_path, debug=False) == 3
    assert (
        f"strands-evals optimize: continue with `strands-evals optimize resume {tmp_path}`" in capsys.readouterr().err
    )
    assert "strands-evals optimize: continue with ... resume <RUN_DIR>" in SKILL_TEXT


def test_the_skill_says_argument_not_knob():
    assert "knob" not in SKILL_TEXT.lower()


def _shell_block(command: str) -> str:
    """The code block that starts `command` in the background (SKILL.md's Start, or Resuming a run)."""
    blocks = re.findall(r"```bash\n(.*?)```", SKILL_TEXT, flags=re.S)
    (block,) = [b for b in blocks if command in b and "run.pid" in b]
    return block


@pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32", reason="the shell lines are bash")
@pytest.mark.parametrize("command", ["strands-evals optimize run", "strands-evals optimize resume"])
def test_the_start_and_resume_lines_record_the_optimizers_pid_and_its_exit(tmp_path, command):
    folder, bin_dir = tmp_path / "F", tmp_path / "bin"
    folder.mkdir()
    bin_dir.mkdir()
    fake = bin_dir / "strands-evals"  # stands in for the optimizer: records its own pid, then exits 3
    fake.write_text('#!/bin/sh\necho $$ > "$F/optimizer.pid"\nsleep 1\nexit 3\n')
    fake.chmod(0o755)
    (folder / "run.exit").write_text("130\n")  # a stale exit code from an earlier run
    block = _shell_block(command)
    for placeholder in re.findall(r"\[--[^]]*\]|<[^>]+>", block):
        block = block.replace(placeholder, str(tmp_path) if placeholder == "<project>" else "x")
    env = {**os.environ, "F": str(folder), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(["bash", "-c", block], env=env, check=True, timeout=10)
    # When the lines return, the stale code is gone and the pid is the optimizer's, while it is still running.
    assert not (folder / "run.exit").exists() and (folder / "run.pid").read_text().strip()
    deadline = time.monotonic() + 10
    while not (folder / "run.exit").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert (folder / "run.exit").read_text().strip() == "3"
    assert (folder / "run.pid").read_text().strip() == (folder / "optimizer.pid").read_text().strip()
