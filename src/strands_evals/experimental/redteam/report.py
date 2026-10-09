"""Red team report."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

from rich.console import Console

from ...types.evaluation import EvaluationOutput
from ...types.evaluation_report import EvaluationReport
from .strategies.base import RUN_RESULTS

_console = Console()


@dataclass
class AttackResult:
    """One attack case with scores from every evaluator that ran on it."""

    case_name: str
    risk_category: str
    strategy: str
    severity: str
    objective: str = ""
    turns_used: int | None = None
    backtracks: int | None = None
    conversation: list[dict] = field(default_factory=list)
    pruned_branches: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    passes: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def score(self) -> float:
        # max(): the strongest attack signal across evaluators is what matters for red teaming.
        return max(self.scores.values()) if self.scores else 0.0

    @property
    def passed(self) -> bool:
        return all(self.passes.values()) if self.passes else True

    @property
    def errored(self) -> bool:
        """True when evaluators reported on the case but none produced a judgment.

        Judgments are per evaluator: an evaluator that errored or had nothing to judge (`NOT_APPLICABLE`)
        contributes only its reason, never a score/pass, so it can neither hide nor fabricate a breach
        that another judge scored.
        """
        return bool(self.reasons) and not self.passes

    @property
    def state(self) -> str:
        """Structural verdict: `errored` cases are excluded from breach/defend accounting.

        An errored case ran into an infrastructure failure (target crash, or no judge could score it) or
        had nothing any evaluator could judge, so it carries no attack signal. Keep it separate rather
        than let a `passed=False` error masquerade as a breach.
        """
        if self.errored:
            return "errored"
        return "defended" if self.passed else "breached"

    @property
    def reason(self) -> str:
        return " | ".join(f"[{k}] {v}" for k, v in self.reasons.items() if v)


@dataclass
class GroupedSummary:
    """Aggregated summary for a group of attack results."""

    group_name: str
    count: int
    # Computed over scored (non-errored) attacks; None when every attack in the group errored.
    avg_score: float | None
    pass_rate: float | None
    errored: int = 0


class RedTeamReport(EvaluationReport):
    """Case-centric report for red team evaluation.

    Note:
        `trajectory` holds raw tool I/O — sanitize before sharing if target tools return sensitive data.
    """

    @classmethod
    def from_evaluation_report(cls, report: EvaluationReport, run_meta: dict[str, dict] | None = None) -> RedTeamReport:
        """Wrap a flattened `EvaluationReport` as a case-centric red team report.

        Args:
            report: Flattened report from the base experiment, one row per (case, evaluator).
            run_meta: Deprecated. Per-case strategy run metadata keyed by case name; merged into each case's
                metadata so the report sees it. Return `AttackRunResult.to_environment_state()` in the task's
                `environment_state` instead.
        """
        if run_meta is not None:
            warnings.warn(
                "`run_meta` is deprecated and will be removed in a future release. Return "
                "`AttackRunResult.to_environment_state()` in the task's `environment_state` instead; the run "
                "stats are then read from the case's `actual_environment_state`, not its `metadata`.",
                DeprecationWarning,
                stacklevel=2,
            )
        run_meta = run_meta or {}
        n = len(report.cases)
        if not (len(report.scores) == n and len(report.test_passes) == n and len(report.reasons) == n):
            raise ValueError("EvaluationReport: cases/scores/passes/reasons length mismatch")

        cases = []
        for case_data in report.cases:
            case_meta = case_data.get("metadata") or {}
            merged_metadata = {**case_meta, **run_meta.get(case_data.get("name", ""), {})}
            evaluator = case_data.get("evaluator", "evaluator")
            cases.append({**case_data, "evaluator": evaluator, "metadata": merged_metadata})

        detailed_results = [report.detailed_results[i] if i < len(report.detailed_results) else [] for i in range(n)]
        # Recompute rather than inherit the base mean, which counts error and failing NOT_APPLICABLE rows as 0.0
        # judgments; the red team view excludes them everywhere else, so overall_score must agree.
        judged = [report.scores[i] for i in range(n) if _is_judgment(report.reasons[i], detailed_results[i])]
        return cls(
            overall_score=sum(judged) / len(judged) if judged else 0.0,
            scores=list(report.scores),
            cases=cases,
            test_passes=list(report.test_passes),
            reasons=list(report.reasons),
            detailed_results=detailed_results,
        )

    def attack_results(self) -> list[AttackResult]:
        by_case: dict[str, AttackResult] = {}
        for i, case_data in enumerate(self.cases):
            name = case_data.get("name", f"case_{i}")
            evaluator = case_data.get("evaluator", "evaluator")
            metadata = case_data.get("metadata") or {}
            run_results = _run_results(case_data)
            result = by_case.setdefault(
                name,
                AttackResult(
                    case_name=name,
                    risk_category=metadata.get("risk_category", "unknown"),
                    strategy=metadata.get("strategy", "unknown"),
                    severity=metadata.get("severity", "unknown"),
                    objective=metadata.get("actor_goal", ""),
                    turns_used=run_results.get("turns_used"),
                    backtracks=run_results.get("backtracks"),
                    conversation=case_data.get("actual_output") or [],
                    pruned_branches=run_results.get("pruned_branches") or [],
                ),
            )
            result.reasons[evaluator] = self.reasons[i]
            # A crashed attack or a judge that could not score it surfaces only as a base-recorded error reason
            # with test_pass=False, and an evaluator with nothing to judge emits only NOT_APPLICABLE outputs;
            # keep the reason for the report but don't record either as a judgment.
            outputs = self.detailed_results[i] if i < len(self.detailed_results) else []
            if _is_judgment(self.reasons[i], outputs):
                result.scores[evaluator] = self.scores[i]
                result.passes[evaluator] = self.test_passes[i]
        return list(by_case.values())

    def _group_by(self, key: str) -> dict[str, list[AttackResult]]:
        groups: dict[str, list[AttackResult]] = {}
        for r in self.attack_results():
            groups.setdefault(getattr(r, key), []).append(r)
        return groups

    def _summarize(self, groups: dict[str, list[AttackResult]]) -> list[GroupedSummary]:
        summaries = []
        for name, items in groups.items():
            # Errored attacks carry no attack signal; leave them out of the score/pass-rate averages.
            scored = [r for r in items if not r.errored]
            summaries.append(
                GroupedSummary(
                    group_name=name,
                    count=len(items),
                    avg_score=sum(r.score for r in scored) / len(scored) if scored else None,
                    pass_rate=sum(1 for r in scored if r.passed) / len(scored) if scored else None,
                    errored=len(items) - len(scored),
                )
            )
        # All-errored groups (avg_score None) sort last.
        return sorted(summaries, key=lambda s: (s.avg_score is None, s.avg_score or 0.0))

    def by_risk_category(self) -> list[GroupedSummary]:
        return self._summarize(self._group_by("risk_category"))

    def by_strategy(self) -> list[GroupedSummary]:
        return self._summarize(self._group_by("strategy"))

    @property
    def failed_cases(self) -> list[AttackResult]:
        return sorted([r for r in self.attack_results() if r.state == "breached"], key=lambda r: r.score)

    def display(self, *, verbose: bool = False, **_kwargs) -> None:  # type: ignore[override]
        """Print the report: case x strategy matrix, then one row per attack worst-first.

        Args:
            verbose: Also print each attack's full conversation and blocked attempts.
        """
        results = self.attack_results()
        total = len(results)
        if total == 0:
            _console.print("Red Team Report\n===============\nNo results.")
            return

        breached = sorted(results, key=lambda r: r.score, reverse=True)
        n_breached = sum(1 for r in results if r.state == "breached")
        n_errored = sum(1 for r in results if r.errored)
        n_blocked = sum(len(r.pruned_branches) // 2 for r in results)
        # ASR excludes errored cases: an infrastructure failure is not a defense, so counting it in the
        # denominator would understate the true success rate against cases the target actually answered.
        scored = total - n_errored
        if n_breached:
            verdict = "FAIL"
        elif scored == 0:
            # Every attack errored (target down, expired credentials): no evidence either way, so not green.
            verdict = "ERROR"
        else:
            verdict = "PASS"
        strategies = sorted({r.strategy for r in results})
        # Strip the "__{strategy}" suffix so the matrix pivots on the original case;
        # fall back to full names if stripping would collapse two distinct cells.
        row_key = _base_case if _base_case_is_unique(results) else (lambda r: r.case_name)
        cases = sorted({row_key(r) for r in results})

        asr = 100 * n_breached / scored if scored else 0.0
        errored_note = f", {n_errored} errored excluded" if n_errored else ""

        _console.print("Red Team Report")
        _console.print("===============")
        _console.print(
            f"Result: {verdict} -- {n_breached} of {scored} attacks breached "
            f"({asr:.1f}%{errored_note}) | {len(cases)} cases x {len(strategies)} strategies"
        )

        self._print_matrix(results, cases, strategies, row_key)
        self._print_flat(breached)

        footer = f"\n{total} attacks · {n_breached} breached · {n_blocked} blocked"
        if n_errored:
            footer += f" · {n_errored} errored"
        _console.print(footer, end="")
        _console.print("" if verbose else "  [verbose for transcripts]")

        if verbose:
            self._print_transcripts(breached)

    def _print_matrix(
        self,
        results: list[AttackResult],
        cases: list[str],
        strategies: list[str],
        row_key: Callable[[AttackResult], str],
    ) -> None:
        """Print a case x strategy score matrix (``*`` marks a breached cell)."""
        by_cell = {(row_key(r), r.strategy): r for r in results}

        def case_worst(case_name: str) -> float:
            cells = [by_cell[(case_name, s)] for s in strategies if (case_name, s) in by_cell]
            return max((r.score for r in cells), default=0.0)

        def case_verdict(case_name: str) -> str:
            states = [by_cell[(case_name, s)].state for s in strategies if (case_name, s) in by_cell]
            if "breached" in states:
                return "BREACH"
            return "ERROR" if all(st == "errored" for st in states) else "ok"

        _console.print("\nAttack matrix (score, * = breached, ! = errored)")
        _console.print(f"  {'case':<24}" + "".join(f"{s:<14}" for s in strategies) + "worst")
        for case_name in sorted(cases, key=lambda c: -case_worst(c)):
            cells = ""
            for s in strategies:
                r = by_cell.get((case_name, s))
                if r is None:
                    cells += f"{'-':<14}"
                elif r.errored:
                    cells += f"{'err !':<14}"
                else:
                    mark = " *" if r.state == "breached" else ""
                    cells += f"{f'{r.score:.2f}{mark}':<14}"
            _console.print(f"  {case_name:<24}{cells}{case_worst(case_name):.2f} {case_verdict(case_name)}")

    def _print_flat(self, results: list[AttackResult]) -> None:
        """Print one row per attack (breached, defended, errored), worst-first."""
        _console.print("\nAll attacks (worst first)")
        _console.print(f"  {'case':<22}{'risk':<22}{'strategy':<14}{'turns':<7}{'blocked':<9}{'result':<8}score")
        for r in results:
            result_label = _result_label(r)
            turns = "" if r.turns_used is None else str(r.turns_used)
            blocked = len(r.pruned_branches) // 2
            # show the base case name; the strategy column already disambiguates the
            # cross-product, and the full "{case}__{strategy}" name overflows the column.
            _console.print(
                f"  {_base_case(r):<22}{r.risk_category:<22}{r.strategy:<14}"
                f"{turns:<7}{blocked:<9}{result_label:<8}{r.score:.2f}"
            )

    def _print_transcripts(self, results: list[AttackResult]) -> None:
        """Print full conversations and blocked attempts for every attack (verbose)."""
        for r in results:
            result_label = _result_label(r)
            _console.print(
                f"\n{_base_case(r)} / {r.strategy}  {result_label}  score={r.score:.2f} {_format_run_stats(r)}"
            )
            if r.objective:
                _console.print(f"  objective: {r.objective}")
            if r.reason:
                _console.print(f"  {r.reason}")
            if r.pruned_branches:
                _console.print(f"  blocked attempts ({len(r.pruned_branches) // 2}):")
                for turn in r.pruned_branches:
                    _console.print(f"    [{turn.get('role', '?')}] {turn.get('content', '')}")
            if r.conversation:
                _console.print("  conversation:")
                for turn in r.conversation:
                    _console.print(f"    [{turn.get('role', '?')}] {turn.get('content', '')}")


# Prefixes the base `Experiment` uses when it isolates a failure into a result row (see experiment.py).
# A judge that could not score a case reaches the red team report only through one of these reasons.
_ERROR_REASON_PREFIXES = ("An error occurred:", "Evaluator error:")


def _is_error_reason(reason: str) -> bool:
    """Return True if `reason` is a base-recorded error string rather than a real judgment."""
    return reason.startswith(_ERROR_REASON_PREFIXES)


def _all_not_applicable(outputs: list[EvaluationOutput]) -> bool:
    """Return True if every output declined to judge, so the row's score/pass is a placeholder.

    Unlike `EvaluationReport.is_applicable`, a failing NOT_APPLICABLE row is dropped too: in a red team
    report a failure reads as a breach, and "absent data" is not evidence the attack succeeded.
    """
    return bool(outputs) and all(o.not_applicable for o in outputs)


def _is_judgment(reason: str, outputs: list[EvaluationOutput]) -> bool:
    """Return True if an evaluator row carries a real verdict (not an error, not all NOT_APPLICABLE)."""
    return not _is_error_reason(reason) and not _all_not_applicable(outputs)


def _result_label(result: AttackResult) -> str:
    """Render the per-attack verdict label used across the flat and transcript views."""
    return {"errored": "ERROR", "breached": "BREACH", "defended": "ok"}[result.state]


def _base_case(result: AttackResult) -> str:
    """Return the case name with the cross-product `__{strategy}` suffix removed."""
    suffix = f"__{result.strategy}"
    if result.case_name.endswith(suffix):
        return result.case_name[: -len(suffix)]
    return result.case_name


def _base_case_is_unique(results: list[AttackResult]) -> bool:
    """Return True if `(base_case, strategy)` keys stay 1:1 after stripping the suffix."""
    keys = [(_base_case(r), r.strategy) for r in results]
    return len(set(keys)) == len(keys)


def _run_results(case_data: dict) -> dict:
    """Return the row's `RUN_RESULTS` environment state, falling back to `metadata` for older reports."""
    for state in case_data.get("actual_environment_state") or []:
        if state.get("name") == RUN_RESULTS:
            run_results = state.get("state")
            # Custom tasks own `environment_state`; tolerate a non-dict state rather than crash the report.
            return run_results if isinstance(run_results, dict) else {}
    return case_data.get("metadata") or {}


def _format_run_stats(result: AttackResult) -> str:
    """Render the strategy's per-run stats when present."""
    parts = []
    if result.turns_used is not None:
        parts.append(f"turns={result.turns_used}")
    if result.backtracks is not None:
        parts.append(f"backtracks={result.backtracks}")
    if result.pruned_branches:
        parts.append(f"blocked={len(result.pruned_branches) // 2}")
    return ", ".join(parts)
