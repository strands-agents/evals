"""Write a synthetic optimizer run directory for the UI tests.

Nothing writes `events.jsonl` yet — `EventLog.append` and every `RunDirectory` method on
`workstream/optimizer-workflow` still raise `NotImplementedError`. So the UI's reader is coded
against the schema documented in `optimization/record/events.py`, and this is the fixture that
stands in for a real run until the record layer lands. Regenerate with:

    python tests/ui/fixtures/make_optimizer_run.py

Everything here is deterministic: fixed timestamps, fixed scores, no RNG. The story is a run that
improves in fits and starts, because a monotonically improving run would not exercise the
best-so-far line, and one with no gaps would not exercise an objective of None.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).parent / "optimizer-run"

START = datetime(2026, 11, 10, 9, 15, 0, tzinfo=timezone.utc)
RUN_ID = "support-tuning-20261110-091500-7af3e1"

TRAIN = [f"train_case_{i:02d}" for i in range(1, 9)]
VALIDATION = [f"val_case_{i:02d}" for i in range(1, 5)]
TEST = [f"test_case_{i:02d}" for i in range(1, 4)]
RUNS = 1

OBJECTIVE = "WeightedSum(Maximize(Helpfulness), Minimize(tokens, scale=1000))"
CONSTRAINTS = ["tokens <= 4000"]
MAX_ROLLOUTS = 240

# (candidate, parent, iteration, origin, outcome, objective, tokens, changed_keys, rationale)
# objective is None wherever the candidate was never scored: an invalid patch, a duplicate of a
# candidate already proposed, or one the budget could not afford. Those must not read as zero.
PLAN = [
    ("c0", None, 0, "baseline", "feasible", 0.512, 3180, [], "The starting candidate, measured once."),
    (
        "c1",
        "c0",
        1,
        "instructions",
        "feasible",
        0.548,
        3240,
        ["system_prompt"],
        "Name the refund window explicitly; the baseline left it implied.",
    ),
    (
        "c2",
        "c0",
        1,
        "instructions",
        "feasible",
        0.503,
        3090,
        ["system_prompt"],
        "Ask a clarifying question first. Worth testing even though it costs a turn.",
    ),
    (
        "c3",
        "c1",
        2,
        "instructions",
        "feasible",
        0.611,
        3310,
        ["system_prompt", "temperature"],
        "Keep c1's refund wording and lower the temperature for consistency.",
    ),
    (
        "c4",
        "c1",
        2,
        "tools",
        "unscored",
        None,
        None,
        ["tool_descriptions"],
        "Rewrite the lookup tool description. Left unscored: the budget could not afford it this iteration.",
    ),
    (
        "c5",
        "c3",
        3,
        "tools",
        "feasible",
        0.589,
        3520,
        ["tool_descriptions"],
        "Spell out the order-id format in the tool description.",
    ),
    (
        "c6",
        "c3",
        3,
        "instructions",
        "feasible",
        0.640,
        3285,
        ["system_prompt"],
        "Ask for the order id up front when it is missing.",
    ),
    (
        "c7",
        "c6",
        4,
        "tools",
        "invalid",
        None,
        None,
        [],
        "Proposed a tool that is not in the registry, so the patch never became a candidate.",
    ),
    # Deliberately the highest objective in the run, and infeasible. A best-so-far line that
    # counts it is wrong: a candidate that breaks a constraint cannot ship, so it can never be
    # the best. Scoring it below the real best would let that mistake pass unnoticed.
    (
        "c8",
        "c6",
        4,
        "instructions",
        "infeasible",
        0.690,
        4210,
        ["system_prompt", "max_tokens"],
        "Add worked examples. Scores best of any candidate but breaks the token constraint.",
    ),
    (
        "c9",
        "c6",
        5,
        "instructions",
        "duplicate",
        None,
        None,
        ["system_prompt"],
        "Re-proposed c6's wording; the search had already seen this exact candidate.",
    ),
    (
        "c10",
        "c6",
        5,
        "instructions",
        "feasible",
        0.671,
        3350,
        ["system_prompt", "temperature"],
        "Combine c6's up-front id request with c3's lower temperature.",
    ),
]

# A couple of trials fail, so the status breakdown is not a single bar.
FAILED_TRIALS = {("c2", "train_case_05"): "agent_failure", ("c5", "train_case_03"): "evaluator_invalid"}


def transcript(*parts: str) -> dict[str, str]:
    """The first 16 hex of a sha256, as `RunDirectory` names transcript files.

    Not `hash()`: that is salted per process, which would make this fixture differ run to run.
    """
    return {"hash": hashlib.sha256("/".join(parts).encode()).hexdigest()[:16], "format": "session"}


def case_scores(objective: float, cases: list[str]) -> dict[str, float]:
    """Per-case values that average to roughly the candidate's objective, spread deterministically."""
    spread = [-0.18, 0.11, -0.06, 0.21, -0.13, 0.04, 0.16, -0.09, 0.07, -0.02, 0.13, -0.15]
    out = {}
    for i, case in enumerate(cases):
        value = objective + spread[i % len(spread)]
        out[case] = round(min(1.0, max(0.0, value)), 3)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clock = [START]

    def stamp(seconds: int = 7) -> str:
        clock[0] = clock[0] + timedelta(seconds=seconds)
        return clock[0].isoformat()

    events: list[dict] = []

    def emit(event: str, iteration: int, **fields) -> None:
        events.append({"event": event, "iteration": iteration, "time": stamp(), **fields})

    config = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "name": "support-tuning",
        "workdir": "/home/liz/support-agent",
        "objective": OBJECTIVE,
        "constraints": CONSTRAINTS,
        "is_pareto": False,
        "goal_metrics": ["Helpfulness", "tokens"],
        "budget": {"max_rollouts": MAX_ROLLOUTS, "max_iterations": 6},
        "split": {"train": TRAIN, "validation": VALIDATION, "test": TEST, "seed": 20261110},
        "runs": RUNS,
        "blocks": ["instructions", "tools"],
        "final_pick": {"n_finalists": 3, "rule": {"kind": "top_ranked"}},
        "score_model": {"kind": "bayesbench", "settings": {}},
        "agent_failure_score": "worst",
        "feasibility_threshold": 0.9,
    }
    (OUT / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    emit("config", 0, **config)

    by_iteration: dict[int, list[dict]] = {}
    for cid, parent, iteration, origin, outcome, objective, tokens, changed, rationale in PLAN:
        by_iteration.setdefault(iteration, []).append(
            {
                "candidate_id": cid,
                "parent": parent,
                "origin": origin,
                "outcome": outcome,
                "objective": objective,
                "tokens": tokens,
                "changed_keys": changed,
                "rationale": rationale,
            }
        )

    number = 0
    numbers: dict[str, int] = {}
    spent = 0
    best = None

    for iteration in sorted(by_iteration):
        proposals = by_iteration[iteration]
        rollouts_this_iteration = 0

        for p in proposals:
            cid, outcome = p["candidate_id"], p["outcome"]

            if outcome == "invalid":
                # No candidate exists, so there is no propose event — only the invalid one.
                emit(
                    "invalid",
                    iteration,
                    block=p["origin"],
                    redraw=0,
                    updates={"tool_name": "refund_lookup_v2"},
                    reason=p["rationale"],
                )
                continue

            numbers[cid] = number
            number += 1
            emit(
                "propose",
                iteration,
                candidate_id=cid,
                number=numbers[cid],
                parent_id=p["parent"] or cid,
                origin=p["origin"],
                values={
                    "system_prompt": f"<prompt for {cid}>",
                    "temperature": 0.2 if "temperature" in p["changed_keys"] else 0.7,
                },
                changed_keys=p["changed_keys"],
                rationale=p["rationale"],
                inherited=[],
                defaulted=[],
                round=None,
                role="inner" if iteration else "opener",
            )

            if outcome == "duplicate":
                emit("duplicate", iteration, block=p["origin"], redraw=0, candidate_id="c6")
                continue

            if outcome == "unscored":
                emit(
                    "unscored",
                    iteration,
                    candidate_id=cid,
                    pending=len(TRAIN),
                    remaining=MAX_ROLLOUTS - spent,
                    reserve=48,
                )
                continue

            per_case = case_scores(p["objective"], TRAIN)
            statuses: dict[str, int] = {}
            for case in TRAIN:
                status = FAILED_TRIALS.get((cid, case), "completed")
                statuses[status] = statuses.get(status, 0) + 1
                record = {
                    "candidate_id": cid,
                    "case_name": case,
                    "run_index": 0,
                    "split": "train",
                    "status": status,
                    "error_type": "ToolError" if status == "agent_failure" else None,
                    "error": "the refund tool timed out" if status == "agent_failure" else None,
                    "attempts": 1,
                    "previous_attempts": [],
                    "output": f"<answer for {case} under {cid}>",
                    "transcript": transcript(cid, case, "train"),
                    "interactions": None,
                    "environment_state": None,
                    "evaluations": {"Helpfulness": {"score": per_case[case], "reason": "judged"}},
                    "details": {},
                    "evaluator_errors": {},
                    "usage": {
                        "tokens": p["tokens"],
                        "input_tokens": int(p["tokens"] * 0.7),
                        "output_tokens": int(p["tokens"] * 0.3),
                        "cache_read_tokens": None,
                        "cache_write_tokens": None,
                        "model_calls": 3,
                        "seconds": 4.2,
                        "models": ["us.anthropic.claude-sonnet-4-5"],
                    },
                    "judge_usage": {},
                    "rollouts": 1,
                }
                emit("trial", iteration, record=record)
                spent += 1
                rollouts_this_iteration += 1

            emit(
                "score",
                iteration,
                candidate_id=cid,
                split="train",
                case_names=TRAIN,
                runs=RUNS,
                rollouts=len(TRAIN),
                objective_by_run={case: [per_case[case]] for case in TRAIN},
                statuses=statuses,
                unavailable={},
            )
            emit(
                "fit", iteration, candidates=[cid], is_fitted=True, error=None, flagged=[], diagnostics={"r_hat": 1.01}
            )
            emit(
                "label",
                iteration,
                candidate_id=cid,
                label={
                    "outcome": outcome,
                    "reason": "token constraint exceeded" if outcome == "infeasible" else "met",
                    "unmet_constraints": ["tokens <= 4000"] if outcome == "infeasible" else [],
                    "estimates": {},
                    "p_met": {"tokens <= 4000": 0.12 if outcome == "infeasible" else 0.98},
                    "is_from_fit": False,
                    "measured": {
                        "Helpfulness": {"mean": p["objective"], "n": len(TRAIN)},
                        "tokens": {"mean": p["tokens"], "n": len(TRAIN)},
                    },
                },
            )

            if outcome == "feasible" and (best is None or p["objective"] > best["objective"]):
                best = {"candidate_id": cid, "number": numbers[cid], "objective": p["objective"]}

        if iteration:
            emit("select", iteration, parent_id=proposals[0]["parent"] or "c0", strategy="top_ranked")

        record = {
            "iteration": iteration,
            "parent_id": proposals[0]["parent"] or "c0",
            "round": None,
            "role": "inner" if iteration else "opener",
            "blocks": ["instructions", "tools"],
            "proposals": [
                {
                    "candidate_id": None if p["outcome"] == "invalid" else p["candidate_id"],
                    "number": None if p["outcome"] == "invalid" else numbers.get(p["candidate_id"]),
                    "parent_id": p["parent"] or "c0",
                    "origin": p["origin"],
                    "changed_keys": p["changed_keys"],
                    "rationale": p["rationale"],
                    "outcome": p["outcome"],
                    "objective": p["objective"],
                }
                for p in proposals
            ],
            "rollouts_spent": rollouts_this_iteration,
        }
        emit("iteration_end", iteration, record=record, exhausted=[], best_after=best)

    finalists = ["c10", "c6", "c3"]
    emit("finalists", 5, candidate_ids=finalists)

    validation_objectives = {"c10": 0.658, "c6": 0.622, "c3": 0.604}
    for cid in finalists:
        for case in VALIDATION:
            emit(
                "trial",
                5,
                record={
                    "candidate_id": cid,
                    "case_name": case,
                    "run_index": 0,
                    "split": "validation",
                    "status": "completed",
                    "error_type": None,
                    "error": None,
                    "attempts": 1,
                    "previous_attempts": [],
                    "output": f"<validation answer {case} / {cid}>",
                    "transcript": transcript(cid, case, "validation"),
                    "interactions": None,
                    "environment_state": None,
                    "evaluations": {"Helpfulness": {"score": validation_objectives[cid], "reason": "judged"}},
                    "details": {},
                    "evaluator_errors": {},
                    "usage": {
                        "tokens": 3300,
                        "input_tokens": 2310,
                        "output_tokens": 990,
                        "cache_read_tokens": None,
                        "cache_write_tokens": None,
                        "model_calls": 3,
                        "seconds": 4.1,
                        "models": ["us.anthropic.claude-sonnet-4-5"],
                    },
                    "judge_usage": {},
                    "rollouts": 1,
                },
            )
            spent += 1

    emit(
        "picked",
        5,
        report={
            "champion_id": "c10",
            "front": ["c10"],
            "objectives": validation_objectives,
            "estimates": {},
            "infeasible": {"c8": "tokens <= 4000"},
            "measured": {
                cid: {"Helpfulness": {"mean": validation_objectives[cid], "n": len(VALIDATION)}} for cid in finalists
            },
            "problem": None,
            "reason": "c10 ranked first on validation and met every constraint.",
        },
    )

    for cid in ("c10", "c0"):
        for case in TEST:
            emit(
                "trial",
                5,
                record={
                    "candidate_id": cid,
                    "case_name": case,
                    "run_index": 0,
                    "split": "test",
                    "status": "completed",
                    "error_type": None,
                    "error": None,
                    "attempts": 1,
                    "previous_attempts": [],
                    "output": f"<test answer {case} / {cid}>",
                    "transcript": transcript(cid, case, "test"),
                    "interactions": None,
                    "environment_state": None,
                    "evaluations": {"Helpfulness": {"score": 0.664 if cid == "c10" else 0.521, "reason": "judged"}},
                    "details": {},
                    "evaluator_errors": {},
                    "usage": {
                        "tokens": 3340 if cid == "c10" else 3190,
                        "input_tokens": 2300,
                        "output_tokens": 1040,
                        "cache_read_tokens": None,
                        "cache_write_tokens": None,
                        "model_calls": 3,
                        "seconds": 4.3,
                        "models": ["us.anthropic.claude-sonnet-4-5"],
                    },
                    "judge_usage": {},
                    "rollouts": 1,
                },
            )
            spent += 1

    emit(
        "test",
        5,
        report={
            "champion_id": "c10",
            "baseline_id": "c0",
            "objective": {"mean_diff": 0.143, "ci_low": 0.061, "ci_high": 0.225, "n_cases": len(TEST)},
            "difference": {
                "Helpfulness": {"mean_diff": 0.143, "ci_low": 0.061, "ci_high": 0.225, "n_cases": len(TEST)}
            },
            "ratio": {"Helpfulness": {"mean_diff": 1.274, "ci_low": 1.09, "ci_high": 1.46, "n_cases": len(TEST)}},
            "missing": {},
            "means": {
                "c10": {"Helpfulness": {"mean": 0.664, "n": len(TEST)}, "tokens": {"mean": 3340, "n": len(TEST)}},
                "c0": {"Helpfulness": {"mean": 0.521, "n": len(TEST)}, "tokens": {"mean": 3190, "n": len(TEST)}},
            },
        },
        reason=None,
        front={},
    )

    emit("stop", 5, reason="iterations", message="reached max_iterations=6")

    lines = "".join(json.dumps(e) + "\n" for e in events)
    (OUT / "events.jsonl").write_text(lines)

    print(f"wrote {OUT}/events.jsonl: {len(events)} events, {len(lines)} bytes")
    print(f"      {OUT}/run.json")
    print(f"      rollouts spent {spent} of {MAX_ROLLOUTS}, champion c10")


if __name__ == "__main__":
    main()
