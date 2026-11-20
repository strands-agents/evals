"""Write a finished optimization run's report to RUN_DIR/report.md and print its path."""

import argparse
import json
import sys
from pathlib import Path

from strands_evals.optimization import OptimizationResult

FILES = (
    ("best.diff", "the champion's text changes, as a diff on your real files"),
    ("best.values.json", "every typed value of the champion"),
    ("effects.json", "what each option did to each metric, from the search fit"),
    ("result.json", "the whole result, reports included"),
)


def _test_check(result: OptimizationResult) -> str:
    """One sentence on whether the champion's gain on the unseen test cases looks real."""
    if result.test is None:
        return f"No test check: {result.test_reason or 'no test cases were left to check the champion on'}."
    comparison = result.test.objective
    if comparison is None:
        return "No test case was measured for both the champion and the baseline."
    gain = f"{comparison.mean_diff:+.4g} on {comparison.n_cases} test case{'s' if comparison.n_cases != 1 else ''}"
    if comparison.ci_low is None or comparison.ci_high is None:
        return f"{gain}. One test case is too few for a confidence range."
    span = f"{gain}, 90% range {comparison.ci_low:+.4g} to {comparison.ci_high:+.4g}"
    if comparison.ci_low > 0:
        return f"{span}. The whole range is above 0, so the gain is likely real."
    if comparison.ci_high < 0:
        return f"{span}. The whole range is below 0, so the change likely makes the agent worse."
    return f"{span}. The range includes 0, so the gain may be luck; more test cases make the range narrower."


def render(result: OptimizationResult, run_dir: Path) -> str:
    """The report as Markdown: verdict, test check, changes, the optimizer's summary, and the run's files."""
    best = result.best
    lines = [f"# Optimization report: {result.run_id}", "", f"Run folder: `{run_dir}`", ""]
    if not result.improved:
        lines += ["No candidate beat your current agent, so there is nothing to change.", ""]
    else:
        lines += [f"Ships candidate {best.number} (`{best.id}`), proposed by {' + '.join(best.origin)}.", ""]
        lines += ["## Test check", "", _test_check(result), ""]
        lines += ["## Changes", "", "| Value | Before | After |", "| --- | --- | --- |"]
        for key, change in best.changed_values().items():
            old, new = (json.dumps(change[side], ensure_ascii=False)[:120] for side in ("old", "new"))
            lines.append(f"| `{key}` | `{old}` | `{new}` |")
        lines.append("")
        if best.diff():
            lines += ["```diff", best.diff().rstrip("\n"), "```", ""]
        lines += ["Nothing in your project changed. Apply the diff and the values yourself, then evaluate again.", ""]
    lines += ["## Summary", "", "```text", result.summary().rstrip("\n"), "```", "", "## Files", ""]
    lines += [f"- `{name}`: {about}" for name, about in FILES if (run_dir / name).exists()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="the run folder `strands-evals optimize run` printed")
    run_dir = parser.parse_args().run_dir.resolve()
    if not (run_dir / "result.json").is_file():
        sys.stderr.write(f"{run_dir} has no result.json: the run hasn't finished\n")
        return 2
    report = run_dir / "report.md"
    report.write_text(render(OptimizationResult.from_file(run_dir / "result.json"), run_dir), encoding="utf-8")
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
