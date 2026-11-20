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
import importlib.metadata
import inspect
import json
import sys
import warnings
from dataclasses import dataclass
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


def _fail(message: str, code: int) -> NoReturn:
    print(f"plan: error: {message}", file=sys.stderr)  # noqa: T201
    sys.exit(code)


try:  # the installed optimizer, not this repository's copy: the skill runs whatever `strands-evals` is installed
    from strands_evals import Case
    from strands_evals.cli._optimize import is_agent_file, parse_split  # is_agent_file: `--agent agent.py`
    from strands_evals.optimization import Budget, FinalPick, Optimizer, Split
    from strands_evals.optimization.end_phase import PARETO_MIN_FINALISTS
    from strands_evals.optimization.setup.budget import end_phase_reserve
except ImportError as missing:
    _fail(
        f"the installed strands-evals can't run this skill ({missing}); reinstall it with: {REINSTALL}",
        EXIT_RUNTIME_ERROR,
    )

_OPTIMIZER_DEFAULTS = inspect.signature(Optimizer).parameters
assert callable(is_agent_file)  # imported only to check the installed build takes an agent file


@dataclass(frozen=True)
class Counts:
    train: int
    validation: int
    test: int

    def as_dict(self) -> dict[str, int]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def _default(name: str) -> int:
    """The Optimizer's default for one of its integer arguments."""
    return int(_OPTIMIZER_DEFAULTS[name].default)


def _read_experiment(path: Path) -> tuple[int, int]:
    """The experiment file's case and evaluator counts."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _fail(f"{path}: {error}", EXIT_BAD_INPUT)
    cases = data.get("cases") if isinstance(data, dict) else None
    evaluators = data.get("evaluators", []) if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases or not isinstance(evaluators, list):
        _fail(f"{path} needs a non-empty 'cases' list and an 'evaluators' list", EXIT_BAD_INPUT)
    return len(cases), len(evaluators)


def _assign(split: Split, n_cases: int) -> Counts:
    """How `split` deals `n_cases` cases. Only the counts matter here, so placeholder names do."""
    cases = [Case(name=f"case-{i}", input="") for i in range(n_cases)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the optimizer's few-cases warnings; `_warnings` reports its own
        try:
            assigned = split.assign(cases, run_seed=0)
        except ValueError as error:
            _fail(str(error), EXIT_BAD_INPUT)
    return Counts(len(assigned.train), len(assigned.validation), len(assigned.test))


def _split_flag(args: argparse.Namespace, n_cases: int) -> str | None:
    """The `--split` value to pass, or None for the optimizer's default split."""
    if args.split is not None:
        return str(args.split)
    if args.counts is None:
        return None
    if sum(args.counts) != n_cases:
        _fail(
            f"--counts {args.counts} sum to {sum(args.counts)}, but the experiment has {n_cases} cases", EXIT_BAD_INPUT
        )
    return ",".join(repr(count / n_cases) for count in args.counts)


def _split(flag: str | None) -> Split:
    if flag is None:
        return Split()
    try:
        return parse_split(flag)
    except ValueError as error:
        _fail(str(error), EXIT_BAD_INPUT)


def _rollouts(counts: Counts, steps: int, candidates: int, runs: int, finalists: int, pareto: bool) -> int:
    """First-pass rollouts: the current agent on train, each candidate on train, the finalists and the current
    agent on validation, then the test check. With 0 steps only the current agent runs, once per split."""
    if steps == 0:
        return runs * (counts.train + counts.validation + counts.test)
    tested = finalists + 1 if pareto else 2  # under Pareto every front member, else the winner and the baseline
    return runs * (counts.train * (1 + steps * candidates) + counts.validation * (finalists + 1) + counts.test * tested)


def _minimum_cap(counts: Counts, runs: int, finalists: int, pareto: bool) -> int:
    """The smallest `--max-rollouts` the optimizer accepts: the baseline on train plus the end phase's reserves,
    re-run headroom included (as `Optimizer.__init__` checks it)."""
    passes = _default("rerun_passes")
    validation = end_phase_reserve(finalists + 1, counts.validation * runs, passes)
    test = end_phase_reserve(finalists + 1 if pareto else 2, counts.test * runs, passes)
    return counts.train * runs + validation + test


def _cases_to_add(split: Split, n_cases: int) -> int | None:
    """How many more cases give this split at least 2 validation and, when it has test cases, 2 test cases."""
    for extra in range(1, 200):
        counts = _assign(split, n_cases + extra)
        if counts.validation >= 2 and (counts.test == 0 or counts.test >= 2):
            return extra
    return None


def _warnings(
    split: Split, n_cases: int, counts: Counts, result: dict[str, Any], steps: int, workers: int
) -> list[str]:
    """What the skill should tell the user before the run: few validation or test cases, and a cap too small."""
    found = []
    if counts.validation < 2 or 0 < counts.test < 2:
        extra = _cases_to_add(split, n_cases)
        more = f"; {extra} more cases would give 2 of each" if extra is not None else ""
        found.append(f"{counts.validation} validation and {counts.test} test cases: the scores can be luck{more}")
    cap, minimum, per_step = result["max_rollouts"], result["minimum_max_rollouts"], result["step_rollouts"]
    if cap < minimum:
        found.append(f"the optimizer refuses a cap below {minimum} rollouts")
    elif steps > result["steps_that_fit"]:
        found.append(
            f"a cap of {cap} fits {result['steps_that_fit']} of the {steps} steps; "
            f"all {steps} need a cap of {minimum + steps * per_step}"
        )
    if steps > 0 and per_step > workers:
        found.append(f"each step's {per_step} trials run {workers} at a time, so steps take longer")
    return found


def _version() -> str:
    try:
        return importlib.metadata.version(DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:  # imported from a source tree, not installed
        return "unknown"


def plan(args: argparse.Namespace) -> dict[str, Any]:
    """The plan for one experiment and budget, as the JSON object `main` prints."""
    n_cases, n_evaluators = _read_experiment(args.experiment)
    flag = _split_flag(args, n_cases)
    split = _split(flag)
    counts = _assign(split, n_cases)
    if args.counts is not None and list(counts.as_dict().values()) != args.counts:
        _fail(
            f"--counts {args.counts} can't be written as shares: the optimizer would deal {counts.as_dict()}",
            EXIT_BAD_INPUT,
        )
    runs = args.runs or _default("runs_per_candidate")
    candidates = args.candidates or _default("proposals_per_iteration")
    workers = args.workers or _default("max_workers")
    finalists = args.finalists or FinalPick().n_finalists
    if args.pareto:
        finalists = max(finalists, PARETO_MIN_FINALISTS)
    cap = args.max_rollouts or Budget().max_rollouts
    rollouts = _rollouts(counts, args.steps, candidates, runs, finalists, args.pareto)
    minimum = _minimum_cap(counts, runs, finalists, args.pareto)
    step_rollouts = candidates * counts.train * runs
    result: dict[str, Any] = {
        "optimizer": _version(),
        "cases": n_cases,
        "split": counts.as_dict(),
        "split_flag": flag,
        "finalists": finalists,
        "rollouts": rollouts,
        "judge_calls": rollouts * n_evaluators,
        "baseline_rollouts": counts.train * runs,
        "step_rollouts": step_rollouts,
        "max_rollouts": cap,
        "minimum_max_rollouts": minimum,
        "steps_that_fit": max(0, cap - minimum) // step_rollouts,
    }
    result["warnings"] = _warnings(split, n_cases, counts, result, args.steps, workers)
    return result


def _count(least: int) -> Any:
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
        if value < least:
            raise argparse.ArgumentTypeError(f"expected a number of at least {least}, got {value}")
        return value

    return parse


def _counts(text: str) -> list[int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected TRAIN,VALIDATION,TEST case counts, got {text!r}")
    return [_count(0)(part) for part in parts]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plan.py", description=__doc__.split("\n\n")[0])
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
