"""`strands-evals run` — execute an experiment against an agent or task callable."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from ...case import Case
from ...display.display_console import CollapsibleTableReportDisplay
from ...evaluators.deterministic import Contains, Equals
from ...evaluators.evaluator import Evaluator
from ...evaluators.output_evaluator import OutputEvaluator
from ...experiment import Experiment
from ...local_file_task_result_store import LocalFileTaskResultStore
from ...types.detector import ConfidenceLevel, DiagnosisConfig, DiagnosisTrigger
from ...types.evaluation_report import EvaluationReport
from ...types.trace import EvaluationLevel
from .._agent_task import synthesize_task_function
from .._common import EXIT_BAD_INPUT, EXIT_FAILURES, EXIT_OK, emit_error, resolve_format
from .._entrypoint import (
    EntryPointError,
    resolve_agent,
    resolve_custom_evaluators,
    resolve_evaluator_spec,
    resolve_task,
)
from .._io import read_text_input, write_text_output

logger = logging.getLogger(__name__)


def _parse_trace_attribute(raw: str) -> tuple[str, str]:
    """Parse a `KEY=VALUE` pair from a `--trace-attributes` argument."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--trace-attributes value must be KEY=VALUE, got '{raw}'")
    key, value = raw.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"--trace-attributes KEY may not be empty (got '{raw}')")
    return key, value


def _parse_fail_on(raw: str) -> tuple[str, float | None]:
    """Parse `--fail-on={any|none|threshold:0.X}` into `(mode, threshold)`."""
    if raw in ("any", "none"):
        return raw, None
    if raw.startswith("threshold:"):
        try:
            threshold = float(raw.split(":", 1)[1])
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"invalid threshold in '{raw}'") from e
        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(f"threshold in '{raw}' must be between 0.0 and 1.0")
        return "threshold", threshold
    raise argparse.ArgumentTypeError(f"--fail-on must be 'any', 'none', or 'threshold:0.X' (got '{raw}')")


def _build_diagnosis_config(args: argparse.Namespace) -> DiagnosisConfig | None:
    if args.diagnose is None:
        return None
    return DiagnosisConfig(
        trigger=DiagnosisTrigger(args.diagnose),
        confidence_threshold=ConfidenceLevel(args.confidence),
    )


def _ad_hoc_flags_set(args: argparse.Namespace) -> list[str]:
    """Return the names of ad-hoc flags the user actually provided.

    Used by `_validate_args` to enforce mutual exclusion with EXPERIMENT_FILE
    and to format helpful error messages naming the conflicting flag.
    """
    set_flags: list[str] = []
    if args.input is not None:
        set_flags.append("--input")
    if args.input_file is not None:
        set_flags.append("--input-file")
    if args.name is not None:
        set_flags.append("--name")
    if args.expected_output is not None:
        set_flags.append("--expected-output")
    if args.rubric is not None:
        set_flags.append("--rubric")
    if args.evaluator:
        set_flags.append("--evaluator")
    return set_flags


def _read_input(args: argparse.Namespace) -> str:
    """Resolve the case input text from `--input` or `--input-file`.

    `--input-file -` reads stdin, mirroring `diagnose`/`report`. The two flags
    are mutually exclusive; argparse enforces that via a group.
    """
    if args.input is not None:
        return args.input
    return read_text_input(args.input_file)


def _build_ad_hoc_evaluators(args: argparse.Namespace) -> list[Evaluator]:
    """Resolve `--evaluator` specs and apply auto-wiring rules.

    Auto-wire convenience evaluators only when `--evaluator` is omitted, so an
    explicit list is never silently extended. `--expected-output` → `Contains`
    and `--rubric` → `OutputEvaluator` compose: passing both yields both checks.
    """
    evaluators: list[Evaluator] = [resolve_evaluator_spec(spec) for spec in (args.evaluator or [])]
    if not evaluators:
        if args.expected_output is not None:
            evaluators.append(Contains(value=args.expected_output))
        if args.rubric is not None:
            evaluators.append(OutputEvaluator(rubric=args.rubric))
    return evaluators


def _build_ad_hoc_experiment(
    args: argparse.Namespace,
    evaluators: list[Evaluator],
    diagnosis_config: DiagnosisConfig | None,
) -> Experiment:
    """Synthesize an in-memory Experiment from ad-hoc CLI flags.

    Shape:
      - one `Case` built from `--input`/`--input-file` + optional
        `--name`/`--expected-output`
      - evaluators are pre-resolved by `_build_ad_hoc_evaluators` so that
        `_validate_args` can introspect them before construction.
    """
    case = Case[str, str](
        name=args.name or "ad_hoc",
        input=_read_input(args),
        expected_output=args.expected_output,
    )
    return Experiment(cases=[case], evaluators=evaluators, diagnosis_config=diagnosis_config)


_TRACE_DEPENDENT_LEVELS = {
    EvaluationLevel.SESSION_LEVEL,
    EvaluationLevel.TRACE_LEVEL,
    EvaluationLevel.TOOL_LEVEL,
}


def _check_ad_hoc_evaluator_compat(args: argparse.Namespace, evaluators: list[Evaluator]) -> str | None:
    """Reject ad-hoc evaluator/flag combinations that would always misfire.

    Two checks:
      1. Trace/Session/Tool-level evaluators need a `Session` trajectory, which
         only `--agent` produces. With `--task` they raise inside `_run_evaluator`
         and surface as a cryptic per-case `score=0`. Fail fast instead.
      2. `Equals()` without `--expected-output` falls back to `case.expected_output`
         which is `None` in ad-hoc mode, so the comparison can never pass.
    """
    if args.task is not None:
        offenders = sorted({type(e).__name__ for e in evaluators if e.evaluation_level in _TRACE_DEPENDENT_LEVELS})
        if offenders:
            return (
                f"--evaluator {', '.join(offenders)} requires a Session trajectory; "
                f"use --agent instead of --task in ad-hoc mode."
            )

    if args.expected_output is None:
        equals_offenders = [e for e in evaluators if isinstance(e, Equals) and e.value is None]
        if equals_offenders:
            return (
                "--evaluator equals requires --expected-output in ad-hoc mode "
                "(otherwise the comparison is against None)."
            )

    return None


def _validate_args(args: argparse.Namespace) -> str | None:
    """Return an error string if argument combinations are invalid, else None.

    Three rules:
      1. EXPERIMENT_FILE and any ad-hoc flag are mutually exclusive — pick a
         mode.
      2. Without EXPERIMENT_FILE the user must supply `--input` or
         `--input-file`; otherwise there is nothing to evaluate.
      3. In ad-hoc mode at least one evaluator must be derivable: an explicit
         `--evaluator` or an `--expected-output` (which defaults to `Contains`).
    """
    ad_hoc = _ad_hoc_flags_set(args)
    if args.experiment_file is not None and ad_hoc:
        joined = ", ".join(ad_hoc)
        return (
            f"EXPERIMENT_FILE and ad-hoc flags ({joined}) are mutually exclusive; "
            f"either pass an experiment file or use --input + --evaluator/--expected-output."
        )
    if args.experiment_file is None:
        if args.input is None and args.input_file is None:
            return "EXPERIMENT_FILE or --input/--input-file is required."
        if not args.evaluator and args.expected_output is None and args.rubric is None:
            return (
                "ad-hoc mode requires --evaluator, --expected-output (auto-wires Contains), "
                "or --rubric (auto-wires OutputEvaluator)."
            )
    return None


def _decide_exit_code(report: EvaluationReport, fail_on: tuple[str, float | None]) -> int:
    """Map `--fail-on` to an exit code based on the flattened report."""
    mode, threshold = fail_on
    if mode == "none":
        return EXIT_OK

    if mode == "any":
        return EXIT_FAILURES if any(not p for p in report.test_passes) else EXIT_OK

    # mode == "threshold"
    if threshold is None:
        raise ValueError("--fail-on=threshold:X requires a numeric threshold")
    return EXIT_FAILURES if report.overall_score < threshold else EXIT_OK


def _print_summary(report: EvaluationReport) -> None:
    """Emit the stable per-evaluator summary line on stderr.

    `run_evaluations_async` returns a single flattened report whose `cases`
    rows are tagged with an `evaluator` key; regroup by that tag so the
    summary still reads `<evaluator>: P/T passed (avg)` per evaluator.
    """
    by_eval: dict[str, list[int]] = {}
    by_eval_scores: dict[str, list[float]] = {}
    for i, case in enumerate(report.cases):
        name = case.get("evaluator", "unknown")
        by_eval.setdefault(name, []).append(int(report.test_passes[i]))
        by_eval_scores.setdefault(name, []).append(report.scores[i])

    parts: list[str] = ["strands-evals run"]
    for name, passes in by_eval.items():
        passed = sum(passes)
        total = len(passes)
        scores = by_eval_scores[name]
        avg = sum(scores) / len(scores) if scores else 0.0
        parts.append(f"{name}: {passed}/{total} passed ({avg:.2f})")

    parts.append(f"overall: {report.overall_score:.2f}")
    print(" | ".join(parts), file=sys.stderr)  # noqa: T201


def _trace_attrs_dict(pairs: list[tuple[str, str]] | None) -> dict[str, Any]:
    if not pairs:
        return {}
    merged: dict[str, Any] = {}
    for key, value in pairs:
        merged[key] = value
    return merged


def _display_expanded(
    report: EvaluationReport,
    *,
    include_recommendations: bool,
    interactive: bool = False,
) -> None:
    """Render the flattened report as a Rich table.

    Static mode (default) pre-expands every row so values are visible without
    dropping into a prompt — `EvaluationReport.display()` would collapse the
    non-core columns to literal "...". Interactive mode (`--interactive`) seeds
    rows collapsed and hands control to `CollapsibleTableReportDisplay.run`'s
    expand/collapse loop, matching `EvaluationReport.run_display()`.
    """
    items: dict[str, dict[str, Any]] = {}
    for i in range(len(report.scores)):
        case = report.cases[i] if i < len(report.cases) else {}
        details: dict[str, Any] = {"name": case.get("name", f"Test {i + 1}")}
        if "evaluator" in case:
            details["evaluator"] = case["evaluator"]
        details["score"] = f"{report.scores[i]:.2f}"
        details["test_pass"] = report.test_passes[i] if i < len(report.test_passes) else False
        details["reason"] = report.reasons[i] if i < len(report.reasons) else ""
        details["input"] = EvaluationReport.format_input_for_display(case.get("input"))
        details["actual_output"] = str(case.get("actual_output"))
        details["expected_output"] = str(case.get("expected_output"))
        if include_recommendations:
            rec = report.recommendations[i] if i < len(report.recommendations) else None
            if rec is not None:
                details["recommendation"] = rec

        items[str(i)] = {
            "details": details,
            "detailed_results": report.detailed_results[i] if i < len(report.detailed_results) else [],
            "expanded": not interactive,
        }

    CollapsibleTableReportDisplay(items=items, overall_score=report.overall_score).run(static=not interactive)


def _run(args: argparse.Namespace) -> int:
    error = _validate_args(args)
    if error is not None:
        emit_error(error)
        return EXIT_BAD_INPUT

    diagnosis_config = _build_diagnosis_config(args)

    if args.experiment_file is not None:
        custom_evaluators = resolve_custom_evaluators(args.custom_evaluator or [])
        loaded = Experiment.from_file(args.experiment_file, custom_evaluators=custom_evaluators)
        # Experiment.from_file does not accept diagnosis_config; rebuild via the
        # public constructor so we don't poke private attrs. Cases are deep-copied
        # by the property accessor, which is fine for a one-shot CLI run.
        experiment = Experiment(
            cases=loaded.cases,
            evaluators=loaded.evaluators,
            diagnosis_config=diagnosis_config,
        )
    else:
        if args.custom_evaluator:
            logger.warning(
                "--custom-evaluator is ignored in ad-hoc mode; pass MODULE:CLASS directly to --evaluator instead"
            )
        try:
            evaluators = _build_ad_hoc_evaluators(args)
        except EntryPointError as e:
            emit_error(str(e))
            return EXIT_BAD_INPUT
        compat_error = _check_ad_hoc_evaluator_compat(args, evaluators)
        if compat_error is not None:
            emit_error(compat_error)
            return EXIT_BAD_INPUT
        experiment = _build_ad_hoc_experiment(args, evaluators, diagnosis_config)

    if args.agent is not None:
        entry = resolve_agent(args.agent)
        task_function = synthesize_task_function(
            entry,
            extra_trace_attributes=_trace_attrs_dict(args.trace_attributes),
        )
    else:
        entry = resolve_task(args.task)
        if args.trace_attributes:
            logger.warning(
                "trace_attributes=<%s> | --trace-attributes ignored when using --task; "
                "the user owns agent instantiation",
                args.trace_attributes,
            )
        task_function = entry.obj

    data_store = LocalFileTaskResultStore(args.data_store) if args.data_store else None

    report = asyncio.run(
        experiment.run_evaluations_async(
            task_function,
            max_workers=args.max_workers,
            evaluation_data_store=data_store,
        )
    )

    if args.output is not None:
        write_text_output(report.model_dump_json(indent=2), args.output)

    if args.display or args.interactive:
        _display_expanded(
            report,
            include_recommendations=args.diagnose is not None,
            interactive=args.interactive,
        )
    elif args.output is None:
        # `run` intentionally only emits JSON to stdout (or nothing, when -o is set).
        # Rich rendering is opt-in via `--display`: building the table eagerly walks
        # every case row and is wasteful on large experiments where users typically
        # pipe to `strands-evals report` or a file.
        fmt = resolve_format(args)
        if fmt == "json":
            sys.stdout.write(report.model_dump_json(indent=2))
            sys.stdout.write("\n")

    _print_summary(report)

    if args.exit_zero:
        return EXIT_OK
    return _decide_exit_code(report, args.fail_on)


def add_subparser(
    subparsers: argparse._SubParsersAction,
    parent: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "run",
        parents=[parent],
        help="run an experiment against an agent or task callable",
        description=(
            "Load an Experiment, resolve --agent or --task, execute "
            "run_evaluations_async, and write a flattened EvaluationReport. "
            "--agent synthesizes the standard task wrapper (telemetry → invoke "
            "→ Session mapping); --task is the escape hatch for custom shapes. "
            "Pass EXPERIMENT_FILE to run a serialized experiment, or omit it "
            "and supply --input + --evaluator/--expected-output for an ad-hoc "
            "single-case run without authoring an experiment file."
        ),
    )
    parser.add_argument(
        "experiment_file",
        metavar="EXPERIMENT_FILE",
        nargs="?",
        default=None,
        help=(
            "path to a serialized Experiment JSON file. Omit to run in ad-hoc "
            "single-case mode using --input + --evaluator/--expected-output."
        ),
    )

    ad_hoc = parser.add_argument_group("ad-hoc case (no experiment file)")
    ad_hoc_input = ad_hoc.add_mutually_exclusive_group()
    ad_hoc_input.add_argument(
        "--input",
        metavar="TEXT",
        default=None,
        help="case input text (ad-hoc mode); pair with --evaluator or --expected-output",
    )
    ad_hoc_input.add_argument(
        "--input-file",
        metavar="PATH",
        default=None,
        help="read case input from PATH (or '-' for stdin); mutually exclusive with --input",
    )
    ad_hoc.add_argument(
        "--name",
        metavar="STR",
        default=None,
        help="case name in ad-hoc mode (default: 'ad_hoc')",
    )
    ad_hoc.add_argument(
        "--expected-output",
        metavar="TEXT",
        default=None,
        help=(
            "expected output for the ad-hoc case. When --evaluator is omitted, "
            "this also defaults to Contains(value=TEXT) so the run is a complete "
            "one-liner."
        ),
    )
    ad_hoc.add_argument(
        "--rubric",
        metavar="TEXT",
        default=None,
        help=(
            "rubric for an LLM-as-judge OutputEvaluator. When --evaluator is "
            "omitted, this auto-wires OutputEvaluator(rubric=TEXT). Composes "
            "with --expected-output (both auto-evaluators are appended). For "
            "richer config (model, system_prompt) use an experiment file."
        ),
    )
    ad_hoc.add_argument(
        "--evaluator",
        metavar="SPEC",
        action="append",
        default=None,
        help=(
            "evaluator for the ad-hoc case (repeatable). SPEC is either a "
            "built-in shortname (helpfulness, equals, faithfulness, ...) or "
            "MODULE:CLASS for a custom Evaluator subclass. Built-in shortnames "
            "instantiate with no arguments; richer config belongs in an "
            "experiment file."
        ),
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--agent",
        metavar="MODULE:ATTR",
        help=(
            "entry point for a factory callable that returns a strands.Agent. "
            "Zero-arg ('def build_agent() -> Agent: ...') or one-arg "
            "('def build_agent(case: Case) -> Agent: ...'). The factory is "
            "invoked once per case so each case runs against a fresh Agent. "
            "Per-case freshness is enforced for strands.Agent only — other "
            "callables (custom agent classes, partials, instances with __call__) "
            "are accepted but cross-case isolation becomes the factory's "
            "responsibility. For non-standard task shapes use --task instead."
        ),
    )
    target.add_argument(
        "--task",
        metavar="MODULE:ATTR",
        help="entry point for a Callable[[Case], dict|str] (escape hatch)",
    )

    parser.add_argument(
        "--trace-attributes",
        metavar="KEY=VALUE",
        action="append",
        type=_parse_trace_attribute,
        default=None,
        help=(
            "extra OTel trace attributes (repeatable). Set as W3C Baggage on "
            "the per-case context and stamped on every span emitted by the "
            "agent. session.id and gen_ai.conversation.id are always set "
            "from the case; --trace-attributes is for additional keys. "
            "No-op when --task is used (the user owns agent instantiation)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="max parallel workers for run_evaluations_async (default: 1)",
    )
    parser.add_argument(
        "--data-store",
        metavar="DIR",
        default=None,
        help="LocalFileTaskResultStore directory; cached task outputs short-circuit reruns",
    )
    parser.add_argument(
        "--diagnose",
        choices=["on_failure", "always"],
        default=None,
        help="run diagnose_session on cases (on failure or always); requires Session trajectories",
    )
    parser.add_argument(
        "--confidence",
        choices=[level.value for level in ConfidenceLevel],
        default=ConfidenceLevel.MEDIUM.value,
        help="confidence threshold passed to DiagnosisConfig (default: medium)",
    )
    parser.add_argument(
        "--custom-evaluator",
        metavar="MODULE:CLASS",
        action="append",
        default=None,
        help="custom Evaluator subclass to register before from_file (repeatable)",
    )
    parser.add_argument(
        "--fail-on",
        type=_parse_fail_on,
        default=("any", None),
        help="exit-code rule: any | none | threshold:0.X (default: any)",
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="always exit 0 regardless of --fail-on result",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="write the flattened reports JSON to PATH instead of stdout",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "render the flattened report as a Rich table on stdout "
            "(includes input, expected_output, actual_output; recommendations "
            "when --diagnose is set)"
        ),
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help=(
            "render the flattened report as an interactive Rich table where "
            "rows can be expanded/collapsed (implies --display)"
        ),
    )

    parser.set_defaults(func=_run)
    return parser
