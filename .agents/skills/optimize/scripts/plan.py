"""Work out an optimize run's case split, rollout estimate, and minimum rollout cap with the installed optimizer.

Run it with the Python of the `strands-evals` command (SKILL.md, step 1):

    "$PY" plan.py --check
    "$PY" plan.py EXPERIMENT_FILE [--split T,V,X | --counts A,B,C] [--steps N] [--candidates N] [--runs N]
        [--finalists N] [--pareto] [--workers N] [--max-rollouts N]

Prints one JSON object on stdout. The split counts come from the optimizer's own `Split` and `--split` parser,
and the minimum cap from its `end_phase_reserve`, so the numbers are what `strands-evals optimize run` does with
the same flags. `--check` only checks that the installed strands-evals can run what the skill builds.

Exit codes: 2 for bad input, 3 when the installed strands-evals is missing or too old for the skill.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import inspect
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

EXIT_BAD_INPUT = 2
EXIT_RUNTIME_ERROR = 3
DISTRIBUTION = "strands-agents-evals"
REINSTALL = (
    "python -m pip install "
    "'strands-agents-evals[optimization] @ git+https://github.com/strands-agents/evals@workstream/optimizer-workflow'"
)
SKILL_STEPS = 4  # the skill's default number of steps; the optimizer itself has no iteration cap by default
MAX_EXTRA_CASES = 200  # how far `_cases_to_add` looks before giving up


def _fail(message: str, code: int) -> NoReturn:
    print(f"plan: error: {message}", file=sys.stderr)  # noqa: T201
    sys.exit(code)


# The installed optimizer, not this repository's copy: the skill runs whatever `strands-evals` is installed, so
# everything this script needs from it is looked up here, and a build that lacks any of it exits 3.
try:
    from strands_evals import Case
    from strands_evals.cli._optimize import is_agent_file, parse_split  # noqa: F401  (is_agent_file: agent files)
    from strands_evals.optimization import Budget, FinalPick, Optimizer, Split
    from strands_evals.optimization.end_phase import PARETO_MIN_FINALISTS
    from strands_evals.optimization.setup.budget import end_phase_reserve

    _parameters = inspect.signature(Optimizer).parameters
    DEFAULT_CANDIDATES = int(_parameters["proposals_per_iteration"].default)
    DEFAULT_RUNS = int(_parameters["runs_per_candidate"].default)
    DEFAULT_WORKERS = int(_parameters["max_workers"].default)
    RERUN_PASSES = int(_parameters["rerun_passes"].default)  # the skill never changes it
    DEFAULT_FINALISTS = FinalPick().n_finalists
    DEFAULT_CAP = Budget().max_rollouts
except (ImportError, KeyError, AttributeError, TypeError) as missing:
    _fail(f"the installed strands-evals can't run this skill ({missing!r}); reinstall it with: {REINSTALL}", 3)


@dataclasses.dataclass(frozen=True)
class Counts:
    """Cases per role, as the optimizer deals them."""

    train: int
    validation: int
    test: int


def _version() -> str:
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:  # imported from a source tree, not installed
        return "unknown"


def _read_experiment(path: Path) -> tuple[int, int]:
    """The experiment file's case and evaluator counts."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _fail(f"{path}: {error}", EXIT_BAD_INPUT)
    cases = data.get("cases") if isinstance(data, dict) else None
    evaluators = data.get("evaluators", []) if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        _fail(f"{path} needs a non-empty 'cases' list", EXIT_BAD_INPUT)
    if not isinstance(evaluators, list):
        _fail(f"{path}: 'evaluators', when given, must be a list", EXIT_BAD_INPUT)
    return len(cases), len(evaluators)


def _assign(split: Split, n_cases: int) -> Counts:
    """How `split` deals `n_cases` cases. Only the counts matter here, so placeholder names do."""
    cases = [Case(name=f"case-{i}", input="") for i in range(n_cases)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the optimizer's few-cases warnings; `_warnings` gives the skill's own
        try:
            assigned = split.assign(cases, run_seed=0)
        except ValueError as error:
            _fail(str(error), EXIT_BAD_INPUT)
    return Counts(len(assigned.train), len(assigned.validation), len(assigned.test))


def _split_flag(split_text: str | None, counts: Counts | None, n_cases: int) -> str | None:
    """The `--split` value to pass, or None for the optimizer's default split."""
    if split_text is not None:
        return split_text
    if counts is None:
        return None
    given = dataclasses.astuple(counts)
    if sum(given) != n_cases:
        _fail(f"--counts {list(given)} sum to {sum(given)}, but the experiment has {n_cases} cases", EXIT_BAD_INPUT)
    return ",".join(repr(count / n_cases) for count in given)


def _split(flag: str | None) -> Split:
    """The `Split` that `optimize run --split <flag>` builds, or the optimizer's default."""
    if flag is None:
        return Split()
    try:
        return parse_split(flag)
    except ValueError as error:
        _fail(str(error), EXIT_BAD_INPUT)


def _end_phase_arms(finalists: int, pareto: bool) -> tuple[int, int]:
    """Candidates measured on validation (the finalists and the current agent) and on test (under Pareto every
    front member, else the winner and the current agent; one arm when only the current agent is left)."""
    validated = finalists + 1
    return validated, validated if pareto else min(validated, 2)


def _rollouts(counts: Counts, steps: int, candidates: int, runs: int, finalists: int, pareto: bool) -> int:
    """First-pass rollouts: the current agent on train, each candidate on train, then the end phase. Finalists
    can't outnumber the candidates the steps propose; with 0 steps only the current agent runs, once per split.
    A run spends less when proposals repeat or the current agent wins (one arm on test), and more when failed
    trials are re-run."""
    validated, tested = _end_phase_arms(min(finalists, steps * candidates), pareto)
    return runs * (counts.train * (1 + steps * candidates) + counts.validation * validated + counts.test * tested)


def _minimum_cap(counts: Counts, runs: int, finalists: int, pareto: bool) -> int:
    """The smallest `--max-rollouts` the optimizer accepts: the baseline on train plus the end phase's reserves,
    re-run headroom included, for the configured finalists (as `Optimizer.__init__` checks it)."""
    validated, tested = _end_phase_arms(finalists, pareto)
    validation = end_phase_reserve(validated, counts.validation * runs, RERUN_PASSES)
    test = end_phase_reserve(tested, counts.test * runs, RERUN_PASSES)
    return counts.train * runs + validation + test


def _too_few(counts: Counts, wants_test: bool) -> bool:
    """Fewer than 2 validation cases, or fewer than 2 test cases when the split asks for test cases."""
    return counts.validation < 2 or (wants_test and counts.test < 2)


def _cases_to_add(split: Split, n_cases: int, wants_test: bool) -> int | None:
    """How many more cases give this split enough validation and test cases (`_too_few`), if any do."""
    for extra in range(1, MAX_EXTRA_CASES + 1):
        if not _too_few(_assign(split, n_cases + extra), wants_test):
            return extra
    return None


def _warnings(
    split: Split, n_cases: int, counts: Counts, *, cap: int, minimum: int, per_step: int, steps: int, workers: int
) -> list[str]:
    """What the skill tells the user before the run: too few validation or test cases, and a cap too small."""
    found = []
    wants_test = float(split.test or 0) > 0  # type: ignore[arg-type]  # shares here, never case lists
    if _too_few(counts, wants_test):
        extra = _cases_to_add(split, n_cases, wants_test)
        more = f"; {extra} more cases would give 2 of each" if extra is not None else ""
        found.append(f"{counts.validation} validation and {counts.test} test cases: the scores can be luck{more}")
    fitting = max(0, cap - minimum) // per_step
    if cap < minimum:
        found.append(f"the optimizer refuses a cap below {minimum} rollouts")
    elif steps > fitting:
        all_steps = minimum + steps * per_step
        found.append(f"a cap of {cap} fits {fitting} of the {steps} steps; all {steps} need a cap of {all_steps}")
    if steps > 0 and per_step > workers:
        found.append(f"each step's {per_step} trials run {workers} at a time, so steps take longer")
    return found


def plan(args: argparse.Namespace) -> dict[str, Any]:
    """The plan for one experiment and budget, as the JSON object `main` prints."""
    n_cases, n_evaluators = _read_experiment(args.experiment)
    flag = _split_flag(args.split, args.counts, n_cases)
    split = _split(flag)
    counts = _assign(split, n_cases)
    if args.counts is not None and counts != args.counts:
        _fail(f"--counts {args.counts} can't be written as shares: the optimizer would deal {counts}", EXIT_BAD_INPUT)
    runs = args.runs or DEFAULT_RUNS
    candidates = args.candidates or DEFAULT_CANDIDATES
    workers = args.workers or DEFAULT_WORKERS
    finalists = args.finalists or DEFAULT_FINALISTS
    if args.pareto:
        finalists = max(finalists, PARETO_MIN_FINALISTS)
    cap = args.max_rollouts or DEFAULT_CAP
    rollouts = _rollouts(counts, args.steps, candidates, runs, finalists, args.pareto)
    minimum = _minimum_cap(counts, runs, finalists, args.pareto)
    per_step = candidates * counts.train * runs
    return {
        "optimizer": _version(),
        "cases": n_cases,
        "split": dataclasses.asdict(counts),
        "split_flag": flag,
        "finalists": finalists,
        "rollouts": rollouts,
        "judge_calls": rollouts * n_evaluators,
        "max_rollouts": cap,
        "minimum_max_rollouts": minimum,
        "steps_that_fit": max(0, cap - minimum) // per_step,
        "warnings": _warnings(
            split, n_cases, counts, cap=cap, minimum=minimum, per_step=per_step, steps=args.steps, workers=workers
        ),
    }


def _count(least: int) -> Callable[[str], int]:
    """An argparse type for a whole number of at least `least`."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
        if value < least:
            raise argparse.ArgumentTypeError(f"expected a number of at least {least}, got {value}")
        return value

    return parse


def _counts(text: str) -> Counts:
    """An argparse type for `TRAIN,VALIDATION,TEST` case counts."""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected TRAIN,VALIDATION,TEST case counts, got {text!r}")
    train, validation, test = (_count(0)(part) for part in parts)
    return Counts(train, validation, test)


def _parser() -> argparse.ArgumentParser:
    """The command line in the module docstring."""
    parser = argparse.ArgumentParser(
        prog="plan.py", description="Work out an optimize run's split, rollout estimate, and minimum rollout cap."
    )
    parser.add_argument("experiment", nargs="?", type=Path, help="the experiment file")
    parser.add_argument("--check", action="store_true", help="only check the installed strands-evals")
    split = parser.add_mutually_exclusive_group()
    split.add_argument("--split", metavar="TRAIN,VALIDATION,TEST", help="shares, as `optimize run --split`")
    split.add_argument("--counts", type=_counts, metavar="TRAIN,VALIDATION,TEST", help="case counts")
    parser.add_argument("--steps", type=_count(0), default=SKILL_STEPS, help=f"default {SKILL_STEPS}")
    parser.add_argument("--candidates", type=_count(1), help="candidates per step")
    parser.add_argument("--runs", type=_count(1), help="runs of each case")
    parser.add_argument("--finalists", type=_count(1))
    parser.add_argument("--pareto", action="store_true", help="the best trade-offs objective")
    parser.add_argument("--workers", type=_count(1), help="trials at once")
    parser.add_argument("--max-rollouts", type=_count(1), help="the rollout cap")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print the plan, or with `--check` the installed version; returns the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps({"optimizer": _version()}))  # noqa: T201
        return 0
    if args.experiment is None:
        parser.error("give the experiment file, or --check")
    print(json.dumps(plan(args), indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
