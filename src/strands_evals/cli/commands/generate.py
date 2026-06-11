"""`strands-evals generate` — synthesize an Experiment via ExperimentGenerator."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from ...evaluators import InteractionsEvaluator, OutputEvaluator, TrajectoryEvaluator
from ...evaluators.evaluator import Evaluator
from ...experiment import Experiment
from ...generators import ExperimentGenerator
from .._common import EXIT_BAD_INPUT, EXIT_OK, emit_error
from .._entrypoint import resolve_custom_evaluators
from .._io import write_text_output

# `ExperimentGenerator.construct_evaluator_async` only supports these three.
# The CLI restricts `--evaluator` to the same set so users get a fast,
# CLI-flavored error instead of a deep ValueError from the generator listing
# raw class objects.
_GENERATOR_EVALUATORS: dict[str, type[Evaluator]] = {
    "OutputEvaluator": OutputEvaluator,
    "TrajectoryEvaluator": TrajectoryEvaluator,
    "InteractionsEvaluator": InteractionsEvaluator,
}


def _validate_args(args: argparse.Namespace) -> str | None:
    """Return an error string if argument combinations are invalid, else None.

    The argparse mutually-exclusive group already enforces "exactly one of
    --context/--experiment". This layer guards mode-specific flag misuse:

    - `--evaluator` and `--num-topics` only apply to context mode; the
      `from_experiment_async` library API derives evaluators from the source
      and doesn't take a topic plan, so silently dropping them would mislead
      users into thinking the flag took effect.
    - `--extra-information` and `--custom-evaluator` only apply to experiment
      mode; in context mode the equivalent guidance goes inline into
      `--context` and there's no source experiment to load custom evaluators
      for.
    """
    if args.experiment is not None:
        if args.evaluator is not None:
            return "--evaluator is not supported with --experiment (evaluators are inherited from the source)."
        if args.num_topics is not None:
            return "--num-topics is not supported with --experiment (topic planning runs only in --context mode)."
    else:
        if args.extra_information is not None:
            return "--extra-information requires --experiment."
        if args.custom_evaluator:
            return "--custom-evaluator requires --experiment."
    return None


def _run(args: argparse.Namespace) -> int:
    error = _validate_args(args)
    if error is not None:
        emit_error(error)
        return EXIT_BAD_INPUT

    generator: ExperimentGenerator[str, str] = ExperimentGenerator(input_type=str, output_type=str, model=args.model)

    if args.experiment is not None:
        custom_evaluators = resolve_custom_evaluators(args.custom_evaluator or [])
        source = Experiment.from_file(args.experiment, custom_evaluators=custom_evaluators)
        experiment = asyncio.run(
            generator.from_experiment_async(
                source_experiment=source,
                task_description=args.task_description or "",
                num_cases=args.num_cases,
                extra_information=args.extra_information,
            )
        )
    else:
        evaluator_cls = _GENERATOR_EVALUATORS[args.evaluator] if args.evaluator else None
        experiment = asyncio.run(
            generator.from_context_async(
                context=args.context,
                task_description=args.task_description or "",
                num_cases=args.num_cases,
                evaluator=evaluator_cls,
                num_topics=args.num_topics,
            )
        )

    if args.output is not None:
        # `experiment.to_file` enforces a `.json` extension and writes via
        # `to_dict()`; route through it so the on-disk shape matches what
        # `Experiment.from_file` round-trips.
        experiment.to_file(args.output)
    else:
        write_text_output(json.dumps(experiment.to_dict(), indent=2, ensure_ascii=False), None)

    print(  # noqa: T201
        f"generated {len(experiment.cases)} case(s), {len(experiment.evaluators)} evaluator(s)",
        file=sys.stderr,
    )
    return EXIT_OK


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "generate",
        parents=[parent],
        help="generate a starter Experiment via ExperimentGenerator",
        description=(
            "Synthesize an Experiment by wrapping ExperimentGenerator. Two "
            "modes: --context creates cases (and an optional rubric) from a "
            "free-form description via from_context_async; --experiment "
            "derives new cases from an existing Experiment file via "
            "from_experiment_async, inheriting the source's evaluators. "
            "--context/--experiment are mutually exclusive. With -o the "
            "experiment is written via Experiment.to_file (.json enforced); "
            "without -o the JSON document is emitted on stdout."
        ),
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--context",
        metavar="CONTEXT",
        default=None,
        help=(
            "context describing the task, tools, or domain (from_context_async). "
            'Use shell substitution for file contents, e.g. --context "$(cat tools.txt)".'
        ),
    )
    source.add_argument(
        "--experiment",
        metavar="EXPERIMENT_FILE",
        default=None,
        help=(
            "path to an existing Experiment JSON used as a reference "
            "(from_experiment_async). Cases are inspired by the source; "
            "evaluators are derived from the source's defaults."
        ),
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help="number of test cases to generate (default: 5)",
    )
    parser.add_argument(
        "--task-description",
        metavar="STR",
        default=None,
        help="short description of the task the agent will perform (default: empty)",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=None,
        help="--context only: expand into N topic-specific prompts for diverse coverage",
    )
    parser.add_argument(
        "--evaluator",
        metavar="NAME",
        choices=sorted(_GENERATOR_EVALUATORS),
        default=None,
        help=(
            "--context only: default evaluator to attach with a generated rubric. "
            f"Choices: {', '.join(sorted(_GENERATOR_EVALUATORS))}. "
            "Omit to produce an experiment with a placeholder Evaluator."
        ),
    )
    parser.add_argument(
        "--extra-information",
        metavar="TEXT",
        default=None,
        help="--experiment only: extra context for the new cases/rubric",
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help=(
            "--experiment only: custom Evaluator subclass to register before "
            "loading the source experiment file (repeatable)"
        ),
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        default=None,
        help="override the judge model used by the generator",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the generated experiment JSON to PATH (via Experiment.to_file)",
    )
    parser.set_defaults(func=_run)
    return parser
