"""Write a synthetic optimizer run directory for the UI tests.

Nothing writes `events.jsonl` yet: the record layer in `optimization/record` still raises
`NotImplementedError`. The page's reader is coded against the schema documented in
`optimization/record/events.py`, and this fixture stands in for a real run. Regenerate with:

    python tests/ui/fixtures/make_optimizer_run.py

Deterministic: fixed timestamps, fixed scores, no RNG. The run improves in fits and starts, so that
the best-so-far line and an objective of None are both exercised.
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
    # The highest objective in the run, and infeasible: a best-so-far line that counts it is wrong,
    # and a lower score here would let that mistake pass.
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

# The baseline's whole configuration, and what each candidate changes about its parent's. A
# candidate's `values` is whole, not a patch, so `compose_values` applies these down the lineage.
BASELINE_VALUES = {
    "system_prompt": """You are the support assistant for Northwind Outfitters.

Answer the customer's question about their order. Be polite and brief.""",
    "temperature": 0.7,
    "tool_descriptions": {
        "order_lookup": "Look up an order.",
        "refund_start": "Start a refund.",
    },
    "max_tokens": 1024,
}

UPDATES: dict[str, dict] = {
    "c1": {
        "system_prompt": """You are the support assistant for Northwind Outfitters.

Answer the customer's question about their order. Be polite and brief.

Refunds are available within 30 days of delivery. Say the window explicitly
when a refund comes up, rather than leaving the customer to ask.""",
    },
    "c2": {
        "system_prompt": """You are the support assistant for Northwind Outfitters.

Before answering, ask one clarifying question about what the customer needs.
Then answer it. Be polite and brief.""",
    },
    "c3": {
        "system_prompt": """You are the support assistant for Northwind Outfitters.

Answer the customer's question about their order. Be polite and brief.

Refunds are available within 30 days of delivery. Say the window explicitly
when a refund comes up, rather than leaving the customer to ask.

Give the same answer to the same question every time. Do not improvise policy.""",
        "temperature": 0.4,
    },
    "c4": {
        "tool_descriptions": {
            "order_lookup": "Look up an order by its id and return status, items and delivery date.",
            "refund_start": "Start a refund for an order that is within the refund window.",
        },
    },
    "c5": {
        "tool_descriptions": {
            "order_lookup": "Look up an order by its id, which looks like NW-000000.",
            "refund_start": "Start a refund for an order id of the form NW-000000.",
        },
    },
    "c6": {
        "system_prompt": """You are the support assistant for Northwind Outfitters.

Answer the customer's question about their order. Be polite and brief.

If the message does not contain an order id, ask for it before doing anything
else. Do not guess which order they mean.

Refunds are available within 30 days of delivery. Say the window explicitly
when a refund comes up, rather than leaving the customer to ask.

Give the same answer to the same question every time. Do not improvise policy.""",
    },
    "c8": {
        "system_prompt": """You are the support assistant for Northwind Outfitters.

Answer the customer's question about their order. Be polite and brief.

If the message does not contain an order id, ask for it before doing anything
else. Do not guess which order they mean.

Refunds are available within 30 days of delivery. Say the window explicitly
when a refund comes up, rather than leaving the customer to ask.

Worked examples.

  Customer: where is my stuff
  You: Happy to check - what is the order id? It looks like NW-000000.

  Customer: NW-481902 arrived torn, I want my money back
  You: Sorry about that. NW-481902 was delivered on 2 November, so it is inside
  the 30-day refund window. I can start the refund now - shall I?

  Customer: do you ship to Alaska
  You: We do, and standard delivery there takes 5-7 business days.

Give the same answer to the same question every time. Do not improvise policy.""",
        "max_tokens": 2048,
    },
    # A re-proposal of c6's wording, byte for byte. That is what makes it a duplicate.
    "c9": {"system_prompt": None},
    "c10": {
        "system_prompt": """You are the support assistant for Northwind Outfitters.

Answer the customer's question about their order. Be polite and brief.

If the message does not contain an order id, ask for it before doing anything
else. Do not guess which order they mean. The id looks like NW-000000.

Refunds are available within 30 days of delivery. Say the window explicitly
when a refund comes up, rather than leaving the customer to ask.

Give the same answer to the same question every time. Do not improvise policy.""",
        "temperature": 0.2,
    },
}
UPDATES["c9"]["system_prompt"] = UPDATES["c6"]["system_prompt"]


# The conversations by hash, as `transcripts/<hash>.json` holds them. `transcript()` fills it.
DOCUMENTS: dict[str, dict | list] = {}

# One trial's transcript is a list of Strands messages, not a recorded session. The page must tell
# the two apart from the document, not from the `format` the pointer claims.
AS_MESSAGE_LIST = ("c0", "train_case_02", "train")

# One pointer names a file `main` does not write: a run copied without a transcript. That is not
# the same as a trial that recorded none, which the agent-failure trial is.
LEFT_UNWRITTEN = ("c6", "val_case_02", "validation")


# What each case asks, the short answer, and what a better-instructed agent adds to it.
CONVERSATIONS = [
    ("Where is my order NW-104233?", "Order NW-104233 was delivered on 2 October.", "It was left at the front desk."),
    (
        "Can I still return the rain jacket I bought?",
        "Yes, you can return it.",
        "Returns are open for 30 days from delivery.",
    ),
    ("I was charged twice for order NW-104871.", "I see two charges on NW-104871.", "I have refunded the duplicate."),
    ("My tent arrived with a torn seam.", "I am sorry about the tent.", "A replacement ships today, free of charge."),
    ("How do I change the delivery address?", "I can change it before the order ships.", "Which order is it for?"),
    (
        "Cancel order NW-105002, please.",
        "Order NW-105002 is cancelled.",
        "The refund reaches your card in 3 to 5 days.",
    ),
    ("Do you ship to Canada?", "Yes, we ship to Canada.", "Delivery takes 6 to 9 working days."),
    (
        "The discount code TRAIL10 is not working.",
        "TRAIL10 expired last week.",
        "I have applied the same 10% to your basket.",
    ),
    (
        "When will the boots in NW-105344 be back in stock?",
        "They are due back on 18 November.",
        "I can email you then.",
    ),
    (
        "I never got a confirmation email.",
        "I have sent the confirmation again.",
        "It went to the address on the order.",
    ),
    ("Can I exchange a size M for a size L?", "Yes, you can exchange it.", "I have emailed a prepaid return label."),
    ("My refund for NW-103990 has not arrived.", "The refund was issued on 4 November.", "Banks can take 5 days."),
    ("Is the stove covered by a warranty?", "Yes, for two years.", "Yours runs until March 2028."),
    (
        "I want to speak to a person.",
        "I am passing you to a colleague.",
        "They will have this conversation in front of them.",
    ),
    ("What is your price-match policy?", "We match any authorised retailer.", "Send a link within 14 days of buying."),
]
# The baseline and the candidate that asks a clarifying question first answer in one short sentence.
CURT = {"c0", "c2"}


def conversation(case: str) -> tuple[str, str, str]:
    return CONVERSATIONS[(TRAIN + VALIDATION + TEST).index(case)]


def answer(cid: str, case: str) -> str:
    _, short, more = conversation(case)
    return short if cid in CURT else f"{short} {more}"


def order_of(case: str) -> str:
    """The order a case names, or a stand-in when it names none."""
    named = [word.strip("?.,") for word in conversation(case)[0].split() if word.startswith("NW-")]
    return named[0] if named else "NW-104233"


def transcript(cid: str, case: str, split: str, output: str | None = None) -> dict[str, str] | None:
    """The `{hash, format}` pointer for one trial, registering the document it points at.

    The hash is the first 16 hex of a sha256, as `RunDirectory` names transcript files.
    """
    key = (cid, case, split)
    digest = hashlib.sha256("/".join(key).encode()).hexdigest()
    as_list = key == AS_MESSAGE_LIST
    DOCUMENTS[digest[:16]] = message_list(cid, case, output) if as_list else session(digest, cid, case, output)
    return {"hash": digest[:16], "format": "list" if as_list else "session"}


def session(digest: str, cid: str, case: str, output: str | None) -> dict:
    """A recorded session as `Session` serializes one, with a span of each kind the page reads differently."""
    sid = digest[:32]
    asked = conversation(case)[0]
    spans = [
        {
            "span_type": "invoke_agent",
            "span_info": {"span_id": digest[32:48], "session_id": sid, "trace_id": sid},
            "system_prompt": compose_values().get(cid, BASELINE_VALUES)["system_prompt"],
            "user_prompt": asked,
            "agent_response": output,
        },
        {
            "span_type": "inference",
            "span_info": {"span_id": digest[48:64], "session_id": sid, "trace_id": sid},
            "messages": [
                {"role": "user", "content": [{"content_type": "text", "text": asked}]},
                {
                    "role": "assistant",
                    "content": [{"content_type": "text", "text": output or "I could not complete that."}],
                },
            ],
        },
        {
            "span_type": "execute_tool",
            "span_info": {"span_id": digest[16:32], "session_id": sid, "trace_id": sid},
            "tool_call": {"name": "lookup_order", "arguments": {"order_id": order_of(case)}},
            "tool_result": (
                {"content": '{"output": "delivered 2026-10-02"}', "error": None}
                if output
                else {"content": None, "error": "the refund tool timed out"}
            ),
        },
    ]
    return {"session_id": sid, "traces": [{"session_id": sid, "spans": spans}]}


def message_list(cid: str, case: str, output: str | None) -> list[dict]:
    """The other shape: Strands messages, with reasoning, a tool call and its result, and per-message usage."""
    return [
        {"role": "user", "content": [{"text": conversation(case)[0]}]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "No order id in the message, so ask for it."}}},
                {"toolUse": {"toolUseId": "tu-1", "name": "lookup_order", "input": {"order_id": order_of(case)}}},
            ],
            "metadata": {"usage": {"inputTokens": 2345, "outputTokens": 1005, "cacheReadInputTokens": 1800}},
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tu-1",
                        "status": "success",
                        # Tool output arrives as a JSON string far more often than as a structure.
                        "content": [{"text": '{"output": "delivered 2026-10-02"}'}],
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": output or "I could not complete that."}]},
    ]


def compose_values() -> dict[str, dict]:
    """Each candidate's whole typed configuration: its parent's, with its own updates applied.

    Parents always appear earlier in `PLAN`, so one pass in order is enough.
    """
    out: dict[str, dict] = {}
    for cid, parent, _iteration, _origin, outcome, *_rest in PLAN:
        if outcome == "invalid":
            continue  # the patch never became a candidate, so it has no configuration
        out[cid] = {**(out[parent] if parent else BASELINE_VALUES), **UPDATES.get(cid, {})}
    return out


def case_scores(objective: float, cases: list[str]) -> dict[str, float]:
    """Per-case values that average to roughly the candidate's objective, spread deterministically."""
    spread = [-0.18, 0.11, -0.06, 0.21, -0.13, 0.04, 0.16, -0.09, 0.07, -0.02, 0.13, -0.15]
    out = {}
    for i, case in enumerate(cases):
        value = objective + spread[i % len(spread)]
        out[case] = round(min(1.0, max(0.0, value)), 3)
    return out


def judged(score: float) -> dict:
    """One evaluator's `EvaluationOutput`, as a trial record serializes it."""
    passed = score >= 0.5
    reason = "The reply answers what was asked." if passed else "The reply leaves the customer to ask again."
    return {"score": score, "test_pass": passed, "reason": reason, "label": None, "metadata": None}


def text_default(body: str) -> dict:
    """How a text key's default is recorded: a digest and a length, never the prompt itself."""
    return {"sha256": hashlib.sha256(body.encode()).hexdigest(), "length": len(body)}


# The declared space, one `ParamInfo` per flat key (`optimization/declaration/space.py`), each kind
# once. The last two are a choice the search never touched and a key that exists under one option only.
SPACE = [
    {
        "key": "system_prompt",
        "name": "system_prompt",
        "kind": "text",
        "default": text_default(BASELINE_VALUES["system_prompt"]),
        "min_value": None,
        "max_value": None,
        "log": False,
        "options": None,
        "size_cap": 2000,
        "edit_instructions": None,
        "active_when": None,
        "source": None,
    },
    {
        "key": "temperature",
        "name": "temperature",
        "kind": "float",
        "default": 0.7,
        "min_value": 0.0,
        "max_value": 1.0,
        "log": False,
        "options": None,
        "size_cap": None,
        "edit_instructions": None,
        "active_when": None,
        "source": None,
    },
    {
        "key": "max_tokens",
        "name": "max_tokens",
        "kind": "int",
        "default": 1024,
        "min_value": 256,
        "max_value": 2048,
        "log": False,
        "step": 256,
        "options": None,
        "size_cap": None,
        "edit_instructions": None,
        "active_when": None,
        "source": None,
    },
    {
        "key": "tool_descriptions",
        "name": "tool_descriptions",
        "kind": "text",
        "default": text_default("Look up an order. / Start a refund."),
        "min_value": None,
        "max_value": None,
        "log": False,
        "options": None,
        "size_cap": None,
        "edit_instructions": None,
        "active_when": None,
        "source": None,
    },
    {
        "key": "escalation.policy",
        "name": "escalation",
        "kind": "choice",
        "default": "ask",
        "min_value": None,
        "max_value": None,
        "log": False,
        "options": ["ask", "auto"],
        "size_cap": None,
        "edit_instructions": None,
        "active_when": None,
        "source": None,
    },
    {
        "key": "escalation.auto.threshold",
        "name": "escalation",
        "kind": "float",
        "default": 0.5,
        "min_value": 0.1,
        "max_value": 0.9,
        "log": False,
        "options": None,
        "size_cap": None,
        "edit_instructions": None,
        "active_when": ["escalation.policy", "auto"],
        "source": None,
    },
]

# `effects.json`, as `RunDirectory.save_effects` writes it. It holds one of everything the plot
# branches on: a settled effect and two unsettled, a goal that wants less, a key left out of the
# model, and a goal whose fit is flagged.
EFFECTS = {
    "effects": [
        {
            "metric": "Helpfulness",
            "key": "system_prompt",
            "level": "rewritten",
            "reference": "default",
            "mean": 0.12,
            "ci90": [0.04, 0.2],
            "p_positive": 0.96,
        },
        {
            "metric": "Helpfulness",
            "key": "temperature",
            "level": "(0.15, 0.45]",
            "reference": "(0.45, 0.75]",
            "mean": 0.03,
            "ci90": [-0.05, 0.11],
            "p_positive": 0.73,
        },
        {
            "metric": "Helpfulness",
            "key": "max_tokens",
            "level": "(1024, 2048]",
            "reference": "(256, 1024]",
            "mean": -0.01,
            "ci90": [-0.09, 0.07],
            "p_positive": 0.41,
        },
        {
            "metric": "tokens",
            "key": "system_prompt",
            "level": "rewritten",
            "reference": "default",
            "mean": 180.0,
            "ci90": [60.0, 300.0],
            "p_positive": 0.99,
        },
        {
            "metric": "tokens",
            "key": "temperature",
            "level": "(0.15, 0.45]",
            "reference": "(0.45, 0.75]",
            "mean": -20.0,
            "ci90": [-140.0, 100.0],
            "p_positive": 0.37,
        },
        {
            "metric": "tokens",
            "key": "max_tokens",
            "level": "(1024, 2048]",
            "reference": "(256, 1024]",
            "mean": 45.0,
            "ci90": [-60.0, 150.0],
            "p_positive": 0.76,
        },
    ],
    "factors": [
        {"key": "system_prompt", "kind": "text", "levels": ["default", "rewritten"], "reference": "default"},
        {
            "key": "temperature",
            "kind": "range",
            "levels": ["(0.45, 0.75]", "(0.15, 0.45]"],
            "reference": "(0.45, 0.75]",
        },
        {
            "key": "max_tokens",
            "kind": "range",
            "levels": ["(256, 1024]", "(1024, 2048]"],
            "reference": "(256, 1024]",
        },
    ],
    "omitted": [
        {"key": "tool_descriptions", "why": "two values, one candidate each"},
        {"key": "escalation.policy", "why": "one value"},
    ],
    "goals": [
        {"role": "objective", "index": 0, "metric": "Helpfulness", "better": "higher"},
        {"role": "constraint", "index": 0, "metric": "tokens", "better": "lower"},
    ],
    "metrics": {
        "Helpfulness": {
            "scale": {"unit": 1.0, "offset": 0.0},
            "noise": 0.004,
            "observations": 48,
            "diagnostics": {"n_holdout": 8, "z_mean": 0.11, "z_sq_sum": 1.3, "flagged": False, "messages": []},
        },
        "tokens": {
            "scale": {"unit": 1000.0, "offset": 0.0},
            "noise": 0.02,
            "observations": 48,
            "diagnostics": {
                "n_holdout": 8,
                "z_mean": 1.4,
                "z_sq_sum": 9.6,
                "flagged": True,
                "messages": ["held-out residuals run high: the model under-states the spread of token counts"],
            },
        },
    },
    "reason": None,
}


def main(out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
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
        "space": SPACE,
        "final_pick": {"n_finalists": 3, "rule": {"kind": "top_ranked"}},
        "score_model": {"kind": "bayesbench", "settings": {}},
        "agent_failure_score": "worst",
        "feasibility_threshold": 0.9,
    }
    (out / "run.json").write_text(json.dumps(config, indent=2) + "\n")
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

    values = compose_values()
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
                # No candidate exists, so there is an invalid event and no propose event.
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
                values=values[cid],
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
                failed, unjudged = status == "agent_failure", status == "evaluator_invalid"
                output = None if failed else answer(cid, case)
                record = {
                    "candidate_id": cid,
                    "case_name": case,
                    "run_index": 0,
                    "split": "train",
                    "status": status,
                    "error_type": "ToolError" if failed else None,
                    "error": "the refund tool timed out" if failed else None,
                    "attempts": 2 if failed else 1,
                    # What went wrong the first time, on the one trial that was retried.
                    "previous_attempts": [{"status": "agent_failure", "error_type": "ToolError"}] if failed else [],
                    "output": output,
                    # The agent never finished, so there is no pointer at all.
                    "transcript": None if failed else transcript(cid, case, "train", output),
                    "interactions": None,
                    "environment_state": None,
                    # The evaluator raised: the trial completed and nothing was judged on it.
                    "evaluations": {} if unjudged else {"Helpfulness": judged(per_case[case])},
                    "details": {},
                    "evaluator_errors": (
                        {"Helpfulness": "the judge returned no score for this answer"} if unjudged else {}
                    ),
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
                    "output": answer(cid, case),
                    "transcript": transcript(cid, case, "validation", answer(cid, case)),
                    "interactions": None,
                    "environment_state": None,
                    "evaluations": {"Helpfulness": judged(validation_objectives[cid])},
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
                    "output": answer(cid, case),
                    "transcript": transcript(cid, case, "test", answer(cid, case)),
                    "interactions": None,
                    "environment_state": None,
                    "evaluations": {"Helpfulness": judged(0.664 if cid == "c10" else 0.521)},
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
    (out / "events.jsonl").write_text(lines)

    # Rewritten from scratch, so a hash no trial points at any more does not linger.
    held = out / "transcripts"
    for stale in sorted(held.glob("*.json")) if held.is_dir() else []:
        stale.unlink()
    held.mkdir(exist_ok=True)
    unwritten = hashlib.sha256("/".join(LEFT_UNWRITTEN).encode()).hexdigest()[:16]
    for digest, document in sorted(DOCUMENTS.items()):
        if digest == unwritten:
            continue
        (held / f"{digest}.json").write_text(json.dumps(document, indent=1) + "\n")

    # `effects.json`, fitted after the fact, which is why a run can have a record and no model.
    (out / "effects.json").write_text(json.dumps(EFFECTS, indent=2) + "\n")

    print(f"wrote {out}/events.jsonl: {len(events)} events, {len(lines)} bytes")
    print(f"      {out}/run.json")
    print(f"      {out}/transcripts/: {len(DOCUMENTS) - 1} files, 1 pointer left unwritten on purpose")
    print(f"      {out}/effects.json: {len(EFFECTS['effects'])} estimates over {len(EFFECTS['goals'])} goals")
    print(f"      rollouts spent {spent} of {MAX_ROLLOUTS}, champion c10")


if __name__ == "__main__":
    main()
