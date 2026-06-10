"""`strands-evals validate` — schema-check a serialized experiment."""

from __future__ import annotations

import argparse
import json
import sys

from ...experiment import Experiment
from .._common import EXIT_OK, resolve_format
from .._entrypoint import resolve_custom_evaluators


def _run(args: argparse.Namespace) -> int:
    custom_evaluators = resolve_custom_evaluators(args.custom_evaluator or [])
    experiment: Experiment = Experiment.from_file(args.experiment_file, custom_evaluators=custom_evaluators)

    case_count = len(experiment.cases)
    evaluator_names = [evaluator.get_name() for evaluator in experiment.evaluators]

    fmt = resolve_format(args)
    if fmt == "json":
        payload = {
            "experiment_file": args.experiment_file,
            "case_count": case_count,
            "evaluator_count": len(evaluator_names),
            "evaluators": evaluator_names,
            "valid": True,
        }
        print(json.dumps(payload, indent=2))  # noqa: T201
    else:
        eval_list = ", ".join(evaluator_names) if evaluator_names else "(none)"
        print(  # noqa: T201
            f"valid: {case_count} case(s), {len(evaluator_names)} evaluator(s) [{eval_list}]",
            file=sys.stderr,
        )

    return EXIT_OK


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "validate",
        parents=[parent],
        help="schema-check a serialized experiment file",
        description=(
            "Load EXPERIMENT_FILE via Experiment.from_file and report case + evaluator "
            "counts. Exits non-zero on schema or I/O errors. Useful as a CI gate before "
            "`strands-evals run`."
        ),
    )
    parser.add_argument(
        "experiment_file",
        metavar="EXPERIMENT_FILE",
        help="path to a serialized Experiment JSON file",
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help="custom Evaluator subclass to register before from_file (repeatable)",
    )
    parser.set_defaults(func=_run)
    return parser
