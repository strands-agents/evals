"""Fixture study records for stress-testing the study explorer, written by the real optimization framework: every
fixture is a real ``Study`` (blocks, methods, policies, projection, record) measuring real Strands agents whose model
is scripted (no network). The model plans calls of a real ``lookup`` tool (some fail and are retried), then replies
with the trial's outcome as JSON, which scripted evaluators read; outcomes, latency and token usage depend on the
configuration, with seeded noise. Text blocks are rewritten by a scripted reflection that adds rules (most help, some
hurt; now and then it fails), with the real method's origin shape. The fixtures: ``nested-deep`` (3 objectives, 4
constraints, a 6-way planner choice whose plan_execute option holds a 3-way memory choice and one option that cannot be
built, five custom blocks, Nested inside Nested, the memory level skipped in rounds whose planner has no memory);
``joint-200-failures`` (200 Joint candidates over numbers and a Choice: agent failures, single-trial grader and endpoint
failures that leave candidates partial, an invalid region, and, from a custom ``Outage`` schedule, a credentials outage over
candidates 60-75 (failed candidates) and a grader outage for hard cases over candidates 120-123 (partial candidates missing a case));
``alternate-8num-3text`` (8 numbers with log and step domains, 3 texts, Alternate with steps=3, two
constraints some candidates break); ``independent-running`` (Independent over 2 objectives, a OneOf and a Switch, Nested
around Alternate, cut in the middle of a round so it reads as still running: open scopes, no finished event);
``weighted-sum`` (WeightedSum with scales, Joint); ``lexicographic`` (Lexicographic with a tolerance, Nested over a
4-way choice); ``stopped`` (a custom schedule raises mid-round, so the record ends with ``stopped``).

    .venv/bin/python explorer/fixtures/generate.py studies    # (re)writes studies/fixture-<name>/ for each
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from botocore.exceptions import NoCredentialsError, ReadTimeoutError
from strands import Agent, ToolContext, tool
from strands.models.model import Model
from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.types.evaluation import EvaluationOutput

from strands_evals.optimization import (
    Alternate,
    AtLeast,
    AtMost,
    Block,
    Choice,
    Draw,
    Float,
    Independent,
    Int,
    Joint,
    Lexicographic,
    Maximize,
    Method,
    Minimize,
    Nested,
    OneOf,
    Pareto,
    Search,
    SkillLibrary,
    Study,
    Switch,
    TextValue,
    TPE,
    WeightedSum,
    component,
    tool_descriptions,
    tunable,
)

# ---- the run being generated -----------------------------------------------------------------------------------------
# The scripted pieces draw from generators seeded by the fixture, the candidate being measured and the trial, so the same
# records come out every time, whatever order the concurrent trials run in.
RUN: dict[str, Any] = {"seed": 0, "study": None, "outage": False, "grader_down": False}
_INSTANCES: Counter = Counter()  # models made per (candidate, case): the k-th is the k-th repetition
FLAKY: set[str] = set()  # tool calls scripted to fail, by toolUseId


def _trial_number() -> int:
    study = RUN["study"]
    return len(study.candidates) if study is not None else -1


def rng_for(*key: Any) -> random.Random:
    return random.Random("|".join(map(str, (RUN["seed"], _trial_number(), *key))))


# ---- the scripted agent ----------------------------------------------------------------------------------------------
@dataclass
class Profile:
    """What one configuration does on average; the model draws each trial's outcome around it."""

    metrics: dict[str, str]  # evaluator metric -> what it reads: quality | tool_errors | format | latency
    quality: float  # chance of solving a case of medium difficulty (with ``credit``, the mean credit)
    steps: float = 2.0  # lookups per trial
    retries: int = 0  # retries of a failed lookup
    tool_error: float = 0.03  # chance a lookup fails
    timeout_s: float = 20.0  # a failed lookup waits this long, in part
    context: float = 3000.0  # input tokens of the first model call, besides the system prompt
    growth: float = 800.0  # input tokens every further call adds
    output: float = 200.0  # output tokens per call
    cached: float = 0.0  # share of the input read from the prompt cache
    latency: float = 2.0  # seconds per model call
    format: float = 0.9  # the mean format score
    credit: bool = False  # partial credit (multiples of 0.05) rather than pass or fail
    fail: float = 0.0  # chance the agent fails mid-trial
    failure: str | tuple[str, ...] = "max_tokens"  # how (one, or one at random): the reply is cut at max_tokens, or
    # the message of an error the model raises
    max_tokens: float = 4096.0
    infra: float = 0.0  # chance the model endpoint times out (the trial measures nothing)
    model_id: str = "sim-haiku"


DIFFICULTY = {"easy": 0.15, "medium": 0.0, "hard": -0.2}
ASPECTS = ("known issues", "recent changes", "configuration", "workarounds", "related tickets", "release notes")
ENDPOINT = (
    "https://bedrock-runtime.us-west-2.amazonaws.com/model/"
    "us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse-stream"
)


SNIPPETS = (
    "Applies to version 4.2 and later; earlier versions need the manual steps in the appendix.",
    "Two related reports were merged into this article in July.",
    "The figures in section 3 supersede the ones in the summary table.",
    "Known to recur after a region migration; see the linked runbook.",
    "Confirmed by support engineering; the workaround is safe to share with customers.",
)


@tool(context=True)
def lookup(query: str, tool_context: ToolContext) -> str:
    """Search the knowledge base; returns the best matching article."""
    if tool_context.tool_use["toolUseId"] in FLAKY:
        FLAKY.discard(tool_context.tool_use["toolUseId"])
        raise TimeoutError("the knowledge base did not answer before the tool timeout")
    n = zlib.crc32(query.encode()) % 9000 + 1000
    return f"KB-{n}: {query}. {SNIPPETS[n % len(SNIPPETS)]} Last reviewed 2026-08-{n % 28 + 1:02d}."


def _first_text(messages: list) -> str:
    for m in messages:
        for block in m.get("content", []):
            if "text" in block:
                return block["text"]
    return ""


class Scripted(Model):
    """A model scripted from its configuration's ``Profile``. Its first call plans the trial (the lookups, which of
    them fail and are retried, whether the agent fails, and the outcome); then it makes the lookups and replies with
    the outcome as JSON. Each call reports usage from the context it was given: the system prompt and every lookup
    cost tokens."""

    def __init__(self, profile: Profile) -> None:
        self.profile, self.calls = profile, 0
        self.config = {"model_id": profile.model_id}

    def update_config(self, **kw: Any) -> None:
        self.config.update(kw)

    def get_config(self) -> dict:
        return self.config

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        yield {"output": None}

    def _plan(self, request: str) -> None:
        p = self.profile
        if RUN["outage"]:
            raise NoCredentialsError()
        _INSTANCES[(_trial_number(), request)] += 1
        rng = self.rng = rng_for(request, _INSTANCES[(_trial_number(), request)])
        if rng.random() < p.infra:
            raise ReadTimeoutError(endpoint_url=ENDPOINT)
        m = re.match(r"\[(\w+)\]\s*([^:]+):", request)
        self.level, self.topic = (m.group(1), m.group(2)) if m else ("medium", request[:40])
        steps = max(0, round(rng.gauss(p.steps, 0.6)))
        self.tries: list[tuple[str, str, bool]] = []
        lost = 0
        for s in range(steps):
            query = f"{self.topic} {ASPECTS[s % len(ASPECTS)]}"
            fails = rng.random() < p.tool_error
            self.tries.append((f"tooluse_{rng.getrandbits(40):010x}", query, fails))
            r = 0
            while fails and r < p.retries:
                r += 1
                fails = rng.random() < 0.45
                self.tries.append((f"tooluse_{rng.getrandbits(40):010x}", f"{query} (retry {r})", fails))
            lost += fails
        FLAKY.update(ident for ident, _, fails in self.tries if fails)
        q = min(0.98, max(0.02, p.quality + DIFFICULTY.get(self.level, 0.0) - (0.3 * lost / steps if steps else 0.0)))
        quality = (round(min(1.0, max(0.0, rng.gauss(q, 0.13))) * 20) / 20) if p.credit else float(rng.random() < q)
        latency = sum(rng.lognormvariate(math.log(p.latency), 0.25) for _ in range(len(self.tries) + 1))
        latency += sum(p.timeout_s * 0.4 if fails else rng.uniform(0.2, 0.7) for *_, fails in self.tries)
        kinds = {
            "quality": quality,
            "tool_errors": round(lost / steps, 4) if steps else 0.0,
            "format": round(min(1.0, max(0.0, rng.gauss(p.format, 0.05))), 3),
            "latency": round(latency, 2),
        }
        self.outcome = {metric: kinds[kind] for metric, kind in p.metrics.items()}
        self.solved = quality >= 0.5
        self.fail_at = rng.randrange(len(self.tries) + 1) if rng.random() < p.fail else None
        self.failure = p.failure if isinstance(p.failure, str) else rng.choice(p.failure)

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        if self.calls == 0:
            self._plan(_first_text(messages))
        i, self.calls = self.calls, self.calls + 1
        p, rng = self.profile, self.rng
        given = (p.context + len(system_prompt or "") / 4 + i * p.growth) * rng.uniform(0.93, 1.07)
        out = max(20.0, rng.gauss(p.output, 0.2 * p.output))
        if i == self.fail_at and self.failure != "max_tokens":
            raise RuntimeError(self.failure)
        yield {"messageStart": {"role": "assistant"}}
        if i == self.fail_at:
            text, stop, out = (
                f"Working through the {self.topic} step by step. First, from the",
                "max_tokens",
                p.max_tokens,
            )
            yield {"contentBlockDelta": {"delta": {"text": text}}}
            yield {"contentBlockStop": {}}
        elif i < len(self.tries):
            ident, query, _ = self.tries[i]
            yield {"contentBlockDelta": {"delta": {"text": f"Looking up {query}."}}}
            yield {"contentBlockStop": {}}
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": ident, "name": "lookup"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"query": query})}}}}
            yield {"contentBlockStop": {}}
            stop = "tool_use"
        else:
            answer = ANSWERS.get(self.topic, "See the cited article.")
            text = (
                f"Answer: {answer}"
                if self.solved
                else f"I could not confirm this from the sources. Best guess: {answer.split(';')[0].split(':')[0]}, "
                f"but the figures disagree."
            )
            text += f"\n\nOUTCOME {json.dumps(self.outcome)}"
            yield {"contentBlockDelta": {"delta": {"text": text}}}
            yield {"contentBlockStop": {}}
            stop = "end_turn"
        yield {"messageStop": {"stopReason": stop}}
        read = given * p.cached
        write = given * 0.1 if p.cached and i == 0 else 0.0
        usage = {
            "inputTokens": int(given - read - write),
            "outputTokens": int(out),
            "cacheReadInputTokens": int(read),
            "cacheWriteInputTokens": int(write),
        }
        usage["totalTokens"] = sum(usage.values())
        yield {
            "metadata": {
                "usage": usage,
                "metrics": {"latencyMs": int(rng.lognormvariate(math.log(p.latency), 0.25) * 1000)},
            }
        }


def agent(profile: Profile, system_prompt: str) -> Agent:
    return Agent(model=Scripted(profile), tools=[lookup], system_prompt=system_prompt, callback_handler=None)


class Reads(Evaluator):
    """Scores one field of the outcome the model reported. The grader times out with chance ``crash``, and is
    down for hard cases while ``RUN["grader_down"]``."""

    REASONS: dict[str, Callable[[float], str]] = {
        "quality": lambda v: (
            "The reply resolves the issue the way the reference does."
            if v >= 0.5
            else "The reply does not resolve the issue."
        ),
        "credit": lambda v: f"{v:.0%} of the reference facts are in the answer.",
        "tool_errors": lambda v: f"{v:.0%} of lookups still failed after retries.",
        "format": lambda v: f"{v:.0%} of the format checks passed.",
        "latency": lambda v: f"End-to-end latency {v:.1f}s.",
    }

    def __init__(self, metric: str, kind: str, credit: bool = False, crash: float = 0.0) -> None:
        super().__init__(name=metric)
        self.metric, self.kind, self.crash = metric, "credit" if kind == "quality" and credit else kind, crash
        self.score_range = (0, None) if kind == "latency" else (0, 1)  # seconds are unbounded

    def evaluate(self, case):
        if RUN["grader_down"] and str(case.input).startswith("[hard]"):
            raise ConnectionError("the grader for hard cases returned 503 Service Unavailable")
        if self.crash and rng_for(case.name, self.metric, "grader").random() < self.crash:
            raise TimeoutError("the grader model did not answer within 60s")
        value = float(json.loads(str(case.actual_output).rsplit("OUTCOME ", 1)[1])[self.metric])
        return [
            EvaluationOutput(
                score=value,
                test_pass=self.kind not in ("quality", "credit") or value >= 0.5,
                reason=self.REASONS[self.kind](value),
            )
        ]

    async def evaluate_async(self, case):
        return self.evaluate(case)


def experiment(
    cases: list[tuple[str, ...]], metrics: dict[str, str], *, credit: bool = False, crash: float = 0.0
) -> Experiment:
    return Experiment(
        cases=[
            Case(name=name, input=f"[{level}] {title}: {text}", expected_output=answer)
            for name, level, title, text, answer in cases
        ],
        evaluators=[Reads(m, k, credit, crash) for m, k in metrics.items()],
    )


# (name, difficulty, title, request, reference answer)
SUPPORT_CASES = [
    (
        "sso-loop",
        "hard",
        "SSO login loop",
        "After enabling SSO, users are sent back to the login page in a loop.",
        "Set the IdP's ACS URL to the regional endpoint and clear the stale session cookie.",
    ),
    (
        "refund-duplicate",
        "easy",
        "duplicate charge",
        "A customer was charged twice for one order and wants a refund.",
        "Void the second authorization; it drops off the statement in 3-5 days.",
    ),
    (
        "vpn-drops",
        "medium",
        "VPN disconnects",
        "The VPN client disconnects every ten minutes on macOS.",
        "Disable the client's power-nap reconnect and raise the keepalive to 60s.",
    ),
    (
        "export-timeout",
        "medium",
        "report export timeout",
        "Exporting a report with more than 50k rows times out.",
        "Use the asynchronous export; the link arrives by email when the file is ready.",
    ),
    (
        "billing-currency",
        "hard",
        "invoice currency",
        "Invoices show USD although the account is set to EUR.",
        "The billing entity, not the account, sets the currency: move the subscription to the EU entity.",
    ),
]
QA_CASES = [
    (
        "capital-flows",
        "medium",
        "capital flow policies",
        "Which two policies changed capital flows in 2024, according to the filings?",
        "The reserve requirement change and the new withholding tax.",
    ),
    (
        "drug-interaction",
        "hard",
        "compound interaction",
        "Does the candidate report an interaction between the two compounds at high dose?",
        "Yes: a QT prolongation signal above 40 mg, in 3 of 212 patients.",
    ),
    (
        "release-date",
        "easy",
        "version 3 release",
        "When was version 3 of the library released, and what did it remove?",
        "March 2023; it removed the legacy sync client.",
    ),
    (
        "budget-delta",
        "medium",
        "research budget change",
        "By how much did the research budget change between the two reports?",
        "It rose by 12%, from 41.0M to 45.9M.",
    ),
]
MATH_CASES = [
    (
        "train-meeting",
        "medium",
        "trains meeting",
        "Two trains leave 300 km apart at 80 and 70 km/h: when do they meet?",
        "After 2 hours.",
    ),
    (
        "compound-interest",
        "easy",
        "compound interest",
        "What is 1000 at 5% compounded yearly after 3 years?",
        "1157.63",
    ),
    (
        "probability-dice",
        "hard",
        "three dice",
        "What is the chance that three dice sum to exactly 10?",
        "27/216 = 0.125",
    ),
    ("geometry-chord", "hard", "chord radius", "A chord of length 8 is 3 from the center: what is the radius?", "5"),
    (
        "work-rate",
        "medium",
        "two pipes",
        "One pipe fills a tank in 6 hours, another in 4: how long together?",
        "2.4 hours",
    ),
]
ANSWERS = {title: answer for cases in (SUPPORT_CASES, QA_CASES, MATH_CASES) for _, _, title, _, answer in cases}

# ---- text: rules a reflection might add, and what each does ---------------------------------------------------------
ADDITIVE = ("quality", "format", "steps")  # the other effects multiply
SYSTEM_RULES = [
    ("WHEN the request is ambiguous, DO ask one clarifying question before acting.", {"quality": 0.05}),
    ("WHEN you find the answer, DO state it first, then the steps that support it.", {"format": 0.05, "quality": 0.02}),
    ("DO end every reply with the OUTCOME line, exactly as specified, and nothing after it.", {"format": 0.07}),
    (
        "DO explain your reasoning in full before giving the answer.",
        {"quality": 0.01, "output": 1.9, "format": -0.06, "latency": 1.15},
    ),
    ("WHEN an earlier step already established a fact, DO NOT look it up again.", {"steps": -0.7}),
]
TOOL_RULES = [
    ("WHEN a lookup fails, DO retry once with a narrower query before giving up.", {"tool_error": 0.6}),
    ("DO NOT call the same tool twice with the same arguments.", {"steps": -0.5, "growth": 0.9}),
    ("WHEN the knowledge base returns an article, DO cite its KB id in the answer.", {"quality": 0.03}),
    ("DO call lookup before every reply, even when you already know the answer.", {"steps": 1.5, "quality": 0.01}),
]
FORMAT_RULES = [
    ("Reply in at most five sentences.", {"output": 0.7, "format": 0.03}),
    ("Put the final answer on its own line, starting with 'Answer:'.", {"format": 0.06, "quality": 0.02}),
    ("Quote the exact error message or figure the request mentions.", {"quality": 0.03}),
    ("Include a short summary of every source you read.", {"output": 1.6, "format": -0.04, "growth": 1.1}),
]
RULES = {
    "system_prompt": SYSTEM_RULES,
    "tool_instructions": TOOL_RULES,
    "tool_guide": TOOL_RULES,
    "answer_format": FORMAT_RULES,
}
PROMPT = "You are a support agent. Resolve the customer's issue using the knowledge base."


def apply_rules(p: Profile, texts: dict[str, str]) -> None:
    for key, text in texts.items():
        for rule, effects in RULES[key]:
            if rule in text:
                for name, v in effects.items():
                    setattr(p, name, getattr(p, name) + v if name in ADDITIVE else getattr(p, name) * v)


class _Rewrite(Draw):
    def __init__(self, values: dict[str, str], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_text(self, name: str, default: str) -> str:
        return self.values.get(name, default)


class ScriptedReflection(Method):
    """In place of contrastive reflection (which runs a model with a shell): adds to one of the held candidate's texts a
    rule a reflection might learn from its trials that fell short. Most rules help, some hurt; now and then the
    reflection fails and the text is kept. The origin has the real method's shape (status, usage, seconds)."""

    kinds = ("text",)

    def __init__(self, seed: int = 0, fail: float = 0.05) -> None:
        self.seed, self.fail, self.rng = seed, fail, random.Random(seed)

    def copy(self, k: int) -> "ScriptedReflection":
        return ScriptedReflection(self.seed + k, self.fail)

    def describe(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, "seed": self.seed}

    def begin(self, block, base, history, structure, study, objective):
        name = type(self).__name__
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _Rewrite({}, {"method": name, "from_candidate": None})
        held = {k: base.params[k] for k in keys}
        rewards = study.projection.rewards(base, study.combine, objective)
        if not rewards:
            return _Rewrite(held, {"method": name, "from_candidate": base.number, "status": "nothing to learn from"})
        rng = self.rng
        usage = {
            "input_tokens": float(rng.randint(60_000, 140_000)),
            "output_tokens": float(rng.randint(1_500, 5_000)),
            "cache_read_tokens": float(rng.choice([0, rng.randint(10_000, 40_000)])),
            "cache_write_tokens": float(rng.randint(0, 6_000)),
        }
        origin: dict[str, Any] = {"method": name, "from_candidate": base.number, "trials": len(rewards)}
        if rng.random() < self.fail:
            usage = {k: float(round(v * 0.4)) for k, v in usage.items()}
            usage["tokens"] = sum(usage.values())
            return _Rewrite(
                held,
                {
                    **origin,
                    "status": "failed",
                    "usage": usage,
                    "error": "ModelThrottledException: Too many tokens per minute, please wait before trying again.",
                },
            )
        usage["tokens"] = sum(usage.values())
        origin.update(usage=usage, seconds=round(rng.uniform(40, 170), 1))
        lost = sum(1 for _, reward, _ in rewards if reward < 0.999)
        rules = [(k, rule) for k in keys for rule, _ in RULES.get(k, ()) if rule not in held[k]]
        if not rules or lost == 0:
            return _Rewrite(held, {**origin, "status": "unchanged"})
        key, rule = rng.choice(rules)
        section = "" if "## Learned Behaviors" in held[key] else "\n\n## Learned Behaviors\n"
        return _Rewrite({**held, key: f"{held[key]}{section}\n- {rule}"}, {**origin, "status": "rewritten"})


# ---- a scripted structural method ------------------------------------------------------------------------------------
class _Pick(Draw):
    def __init__(self, option: str, key: str, near: dict[str, Any], rng: random.Random, origin: dict) -> None:
        self.option, self.key, self.near, self.rng, self.origin = option, key, near, rng, origin

    def suggest_categorical(self, name, choices):
        choices = list(choices)
        if name == self.key:
            return self.option
        if self.near.get(name) in choices and self.rng.random() < 0.6:
            return self.near[name]
        return self.rng.choice(choices)

    def suggest_int(self, name, low, high, *, step=1, log=False):
        top = low + step * ((high - low) // step)
        if name in self.near:
            return min(top, max(low, self.near[name] + step * self.rng.choice((-2, -1, 0, 1, 1, 2))))
        return low + step * self.rng.randint(0, (high - low) // step)

    def suggest_float(self, name, low, high, *, step=None, log=False):
        if log:
            lo, hi = math.log(low), math.log(high)
            x = (
                math.log(self.near[name]) + self.rng.gauss(0, 0.1 * (hi - lo))
                if name in self.near
                else self.rng.uniform(lo, hi)
            )
            return min(high, max(low, math.exp(x)))
        x = (
            self.near[name] + self.rng.gauss(0, 0.1 * (high - low))
            if name in self.near
            else self.rng.uniform(low, high)
        )
        return min(high, max(low, x))


class Sweep(Method):
    """Tries each option of ``key`` in ``order`` (one per round), then mostly the option whose round did best on the
    first objective, sometimes another that has not failed. The chosen option's settings are drawn near the latest
    values tried (at random the first time)."""

    kinds = ("one_of", "switch", "choice", "int", "float")

    def __init__(self, key: str, order: list[str], seed: int = 0, stay: float = 0.7) -> None:
        self.key, self.order, self.seed, self.stay = key, list(order), seed, stay
        self.rng, self.drawn, self.failed, self.scores = random.Random(seed), 0, set(), {}

    def copy(self, k: int) -> "Sweep":
        return Sweep(self.key, self.order, self.seed + k, self.stay)

    def describe(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, "key": self.key, "order": self.order, "seed": self.seed, "stay": self.stay}

    def begin(self, block, base, history, structure, study, objective):
        n, self.drawn = self.drawn, self.drawn + 1
        best = max(self.scores, key=self.scores.get) if self.scores else None
        if n < len(self.order):
            option, why = self.order[n], "sweep"
        else:
            others = [o for o in study.params[self.key].choices if o not in self.failed and o != best]
            if best is not None and (self.rng.random() < self.stay or not others):
                option, why = best, "best"
            else:
                option, why = self.rng.choice(others), "explore"
        near: dict[str, Any] = {}
        for t in history:
            near.update(t.params)
        near.update(base.params if base is not None else {})
        return _Pick(
            option,
            self.key,
            near,
            self.rng,
            {"method": type(self).__name__, "option": option, "why": why, "best": best},
        )

    def observe(self, block, candidate, draw, study, objective):
        if not isinstance(draw, _Pick):
            return
        if candidate is None or not candidate.eligible:
            self.failed.add(draw.option)
        else:
            score = study.projection.oriented(candidate)[0]
            self.scores[draw.option] = max(score, self.scores.get(draw.option, score))


# ---- the fixtures' variables, as objects --------------------------------------------------------------------------
KINDS = {"structure": (OneOf, Switch), "numbers": (Float, Int, Choice), "text": (TextValue,), "skills": (SkillLibrary,)}


def _declared(factory) -> list[tuple[Any, str]]:
    """Each variable the factory declares: the object, and its key."""
    return [(obj, keys[0]) for obj, keys, direct in factory.references() if direct]


def _unique(objects: list) -> list:
    return list({id(o): o for o in objects}.values())


def by_type(factory) -> dict[str, list]:
    """The factory's variables by kind, as objects (these fixtures' blocks hold every variable of one kind)."""
    out: dict[str, list] = {kind: [] for kind in KINDS}
    for obj, _ in _declared(factory):
        out[next(kind for kind, types in KINDS.items() if isinstance(obj, types))].append(obj)
    return {kind: _unique(objs) for kind, objs in out.items()}


def by_keys(factory, **claims: list[str]) -> dict[str, list]:
    """The factory's variables under each block's keys, as objects; the most specific key wins."""
    out: dict[str, list] = {name: [] for name in claims}
    for obj, key in _declared(factory):
        matches = [(p, name) for name, ps in claims.items() for p in ps if key == p or key.startswith(p + ".")]
        if matches:
            out[max(matches, key=lambda m: len(m[0]))[1]].append(obj)
    return {name: _unique(objs) for name, objs in out.items()}


# ---- custom schedules ------------------------------------------------------------------------------------------------
class Outage(Joint):
    """Joint, with the model endpoint's credentials gone while candidates ``outage[0]`` to ``outage[1] - 1`` are measured,
    and the grader of hard cases down for the candidates in ``grader``."""

    def __init__(self, outage: tuple[int, int], grader: tuple[int, int]) -> None:
        self.outage, self.grader = outage, grader

    def describe(self, study) -> dict[str, Any]:
        return {"kind": type(self).__name__, "outage": list(self.outage), "grader": list(self.grader)}

    def run(self, scope) -> None:
        k = 0
        try:
            while scope.budget > 0:
                n = len(scope.study.candidates)
                RUN["outage"] = self.outage[0] <= n < self.outage[1]
                RUN["grader_down"] = self.grader[0] <= n < self.grader[1]
                if scope.candidate(self.blocks(k, scope.blocks), step=k) is None:
                    return
                k += 1
        finally:
            RUN.update(outage=False, grader_down=False)


class StopAfter(Joint):
    """Joint until the study has ``candidates`` candidates; then it raises, as a lost machine would stop a study."""

    def __init__(self, candidates: int) -> None:
        self.candidates = candidates

    def describe(self, study) -> dict[str, Any]:
        return {"kind": type(self).__name__, "candidates": self.candidates}

    def run(self, scope) -> None:
        k = 0
        while scope.budget > 0:
            if len(scope.study.candidates) >= self.candidates:
                raise RuntimeError("the spot instance running the study was reclaimed")
            if scope.candidate(self.blocks(k, scope.blocks), step=k) is None:
                return
            k += 1


# ---- nested-deep: a support agent ------------------------------------------------------------------------------------
class NoMemory:
    def adjust(self, fx: dict) -> None:
        pass


class BufferMemory:
    def __init__(self, size: int) -> None:
        self.size = size

    def adjust(self, fx: dict) -> None:
        fx["growth"] *= min(1.0, self.size / 40)
        fx["quality"] -= 0.12 * max(0, 12 - self.size) / 12


class SummaryMemory:
    def __init__(self, max_items: int, every: int) -> None:
        self.max_items, self.every = max_items, every

    def adjust(self, fx: dict) -> None:
        fx["growth"] *= 0.35
        fx["quality"] += 0.09 - 0.008 * abs(self.max_items - 10)
        fx["latency"] *= 1 + 0.25 / self.every
        fx["output"] = 200 * (1 + 0.5 / self.every)


class ReAct:
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps

    def profile(self) -> dict:
        return dict(
            quality=0.50 + 0.035 * min(self.max_steps, 7) - 0.01 * max(0, self.max_steps - 7),
            steps=0.55 * self.max_steps,
            context=4200,
            growth=2200,
            latency=3.6,
        )


class PlanExecute:
    def __init__(self, max_plan_steps: int, memory: Any) -> None:
        self.max_plan_steps, self.memory = max_plan_steps, memory

    def profile(self) -> dict:
        fx = dict(
            quality=0.64 + 0.03 * min(self.max_plan_steps, 6),
            steps=0.8 * self.max_plan_steps + 1,
            context=5200,
            growth=2400,
            latency=2.6,
            output=200.0,
        )
        self.memory.adjust(fx)
        return fx


class Reflexion:
    def __init__(self, reflections: int) -> None:
        self.reflections = reflections

    def profile(self) -> dict:
        return dict(
            quality=0.56 + 0.05 * self.reflections,
            steps=2 + 1.6 * self.reflections,
            context=4800,
            growth=2300,
            latency=4.2,
            output=380,
            format=0.87,
        )


class Router:
    def __init__(self, experts: int) -> None:
        self.experts = experts

    def profile(self) -> dict:
        return dict(
            quality=0.30 + 0.03 * self.experts, steps=1.2, context=3000 + 700 * self.experts, growth=900, latency=2.4
        )


class SingleShot:
    def profile(self) -> dict:
        return dict(quality=0.16, steps=0.3, context=3600, growth=0, latency=3.0, format=0.92)


class TreeSearch:
    def __init__(self, beam: int) -> None:
        raise ValueError("tree_search needs token log-probabilities, which this model endpoint does not return")


class Decoding:
    def __init__(self, temperature: float, top_p: float) -> None:
        self.temperature, self.top_p = temperature, top_p

    def apply(self, p: Profile) -> None:
        p.quality -= 0.3 * (self.temperature - 0.35) ** 2 + 0.6 * max(0.0, 0.8 - self.top_p) ** 2
        p.format += 0.05 - 0.12 * self.temperature
        p.steps += 0.5 * self.temperature


class ToolSettings:
    def __init__(self, timeout_s: int, retries: int) -> None:
        self.timeout_s, self.retries = timeout_s, retries

    def apply(self, p: Profile) -> None:
        p.timeout_s, p.retries = self.timeout_s, self.retries
        p.tool_error = 0.03 + 0.4 * math.exp(-self.timeout_s / 12)


SUPPORT_METRICS = {
    "resolved": "quality",
    "tool_error_rate": "tool_errors",
    "format_score": "format",
    "latency_s": "latency",
}


@tunable(
    planner=OneOf(
        component(ReAct, max_steps=Int(2, 12, default=6)),
        component(
            PlanExecute,
            max_plan_steps=Int(2, 10, default=4),
            memory=OneOf(
                component(NoMemory),
                component(BufferMemory, size=Int(4, 64, default=16)),
                component(SummaryMemory, max_items=Int(2, 20, default=8), every=Int(1, 5, default=2)),
                labels=["none", "buffer", "summary"],
            ),
        ),
        component(Reflexion, reflections=Int(1, 4, default=2)),
        component(Router, experts=Int(2, 6, default=3)),
        component(SingleShot),
        component(TreeSearch, beam=Int(2, 8, default=4)),
        labels=["react", "plan_execute", "reflexion", "router", "single_shot", "tree_search"],
    ),
    decoding=component(Decoding, temperature=Float(0.0, 1.5, default=0.7), top_p=Float(0.5, 1.0, default=0.95)),
    tools=component(ToolSettings, timeout_s=Int(5, 120, default=30), retries=Int(0, 5, default=1)),
    system_prompt=TextValue(default=PROMPT),
    tool_instructions=TextValue(default="Use lookup to search the knowledge base."),
)
def support_agent(planner, decoding, tools, system_prompt, tool_instructions):
    p = Profile(metrics=SUPPORT_METRICS, **planner.profile())
    decoding.apply(p)
    tools.apply(p)
    apply_rules(p, {"system_prompt": system_prompt, "tool_instructions": tool_instructions})
    return agent(p, f"{system_prompt}\n\n{tool_instructions}")


def nested_deep(record: Path) -> tuple[Study, int]:
    order = ["plan_execute", "reflexion", "tree_search", "router", "single_shot", "react"]
    v = by_keys(
        support_agent,
        architecture=["planner"],
        memory=["planner.plan_execute.memory"],
        decoding=["decoding"],
        prompts=["system_prompt", "tool_instructions"],
        tools=["tools"],
    )
    architecture = Block(Sweep("planner", order, seed=2), *v["architecture"], name="architecture")
    memory = Block(TPE(seed=2), *v["memory"], name="memory")
    decoding = Block(TPE(seed=3), *v["decoding"], name="decoding")
    prompts = Block(ScriptedReflection(seed=4), *v["prompts"], name="prompts")
    tools = Block(TPE(seed=5), *v["tools"], name="tools")
    study = Study(
        repetitions=2,
        agent=support_agent,
        experiment=experiment(SUPPORT_CASES, SUPPORT_METRICS),
        objectives=Pareto(Maximize("resolved"), Minimize("tokens"), Minimize("latency_s")),
        constraints=[
            AtMost("tool_error_rate", 0.05),
            AtMost("tokens", 60000),
            AtLeast("format_score", 0.8),
            AtMost("latency_s", 30),
        ],
        search=Search(
            blocks=[architecture, memory, decoding, prompts, tools],
            schedule=Nested(
                outer=architecture, inner=Nested(outer=memory, inner=Joint(), inner_iterations=2), inner_iterations=8
            ),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 60


# ---- joint-200-failures: retrieval QA over numbers and one Choice ----------------------------------------------------
@tunable(
    temperature=Float(0.0, 1.2, default=0.6),
    top_k=Int(1, 20, default=4),
    chunk_size=Int(200, 4000, default=800, log=True),
    max_tokens=Int(256, 4096, default=1024),
    model=Choice(["small", "medium", "large"], default="medium"),
)
def qa_agent(temperature, top_k, chunk_size, max_tokens, model):
    chars = top_k * chunk_size
    if chars > 20000:
        raise ValueError(f"top_k * chunk_size = {chars} characters of passages do not fit the 20000-character context")
    base, latency = {"small": (0.52, 1.4), "medium": (0.66, 2.2), "large": (0.76, 3.4)}[model]
    coverage = 1 - math.exp(-chars / 3500)
    quality = (
        base * (0.55 + 0.45 * coverage)
        - 0.10 * (chars / 20000) ** 2
        - 0.35 * (temperature - 0.25) ** 2
        - (0.08 if max_tokens < 512 else 0.0)
    )
    fail, failure = 0.015, ("max_tokens", "the model returned an empty message")
    if max_tokens < (900 if model == "small" else 400):
        fail, failure = 0.4, "max_tokens"
    elif temperature > 0.95:
        fail, failure = 0.05 + 0.6 * (temperature - 0.95), "the model's tool call arguments were not valid JSON"
    p = Profile(
        metrics={"accuracy": "quality"},
        quality=quality,
        credit=True,
        steps=0.8 + top_k / 8,
        context=900 + chars / 4,
        growth=chars / 8,
        output=min(max_tokens, 260.0),
        latency=latency,
        cached=0.35 if model != "small" else 0.0,
        fail=fail,
        failure=failure,
        max_tokens=max_tokens,
        infra=0.012,
        model_id=f"sim-{model}",
    )
    return agent(p, "Answer the question from the retrieved passages. Cite the passage ids you used.")


def joint_200_failures(record: Path) -> tuple[Study, int]:
    study = Study(
        repetitions=2,
        agent=qa_agent,
        experiment=experiment(QA_CASES, {"accuracy": "quality"}, credit=True, crash=0.01),
        objectives=Maximize("accuracy"),
        search=Search(
            blocks=[Block(TPE(seed=7), *by_type(qa_agent)["numbers"])],
            schedule=Outage(outage=(60, 76), grader=(120, 124)),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 200


# ---- alternate-8num-3text: a research agent with many numbers and three texts ----------------------------------------
@tunable(
    temperature=Float(0.0, 1.5, default=0.7),
    top_p=Float(0.5, 1.0, default=0.95),
    context_budget=Float(2000.0, 128000.0, default=16000.0, log=True),
    min_relevance=Float(1e-4, 0.5, default=0.01, log=True),
    max_turns=Int(2, 24, default=8),
    chunk_chars=Int(256, 4096, default=1024, step=256),
    top_k=Int(2, 32, default=8, step=2),
    retries=Int(0, 4, default=1),
    system_prompt=TextValue(default="You are a research assistant. Answer from the sources you find."),
    tool_guide=TextValue(default="Use lookup to find sources."),
    answer_format=TextValue(default="Answer clearly."),
)
def research_agent(
    temperature,
    top_p,
    context_budget,
    min_relevance,
    max_turns,
    chunk_chars,
    top_k,
    retries,
    system_prompt,
    tool_guide,
    answer_format,
):
    retrieved = min(context_budget, top_k * chunk_chars / 4)
    coverage = (1 - math.exp(-retrieved / 6000)) * (1 - min_relevance)
    quality = (
        0.30
        + 0.30 * coverage
        + 0.10 * min(1.0, max_turns / 10)
        - 0.3 * (temperature - 0.3) ** 2
        - 0.2 * (top_p - 0.9) ** 2
        - 0.04 * (math.log10(min_relevance) + 1.7) ** 2
        + 0.03 * min(retries, 2)
        - 0.08 * max(0.0, (retrieved - 30000) / 100000)
    )
    p = Profile(
        metrics={"score": "quality", "latency_s": "latency"},
        quality=quality,
        credit=True,
        steps=min(max_turns - 1, 1.5 + top_k / 8),
        retries=retries,
        tool_error=0.08,
        timeout_s=10,
        context=1200 + retrieved * 0.5,
        growth=retrieved * 0.3,
        latency=1.2 + retrieved / 30000,
    )
    apply_rules(p, {"system_prompt": system_prompt, "tool_guide": tool_guide, "answer_format": answer_format})
    return agent(p, f"{system_prompt}\n\n{tool_guide}\n\n{answer_format}")


def alternate_8num_3text(record: Path) -> tuple[Study, int]:
    t = by_type(research_agent)
    numbers, text = Block(TPE(seed=11), *t["numbers"]), Block(ScriptedReflection(seed=12), *t["text"])
    study = Study(
        repetitions=2,
        agent=research_agent,
        experiment=experiment(QA_CASES, {"score": "quality", "latency_s": "latency"}, credit=True),
        objectives=Maximize("score"),
        constraints=[AtMost("tokens", 40000), AtMost("latency_s", 10)],
        search=Search(blocks=[numbers, text], schedule=Alternate(numbers, text, iterations=3)),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 40


# ---- independent-running: retrieval with a choice of retriever and an optional reranker ------------------------------
class BM25:
    def __init__(self, k1: float, b: float) -> None:
        self.quality = 0.48 - 0.1 * (k1 - 1.3) ** 2 - 0.1 * (b - 0.7) ** 2
        self.context = 2500


class Dense:
    def __init__(self, top_k: int, encoder: str) -> None:
        self.quality = 0.52 + 0.012 * min(top_k, 12) + (0.06 if encoder == "large-embed" else 0.0)
        self.context = 600 + 420 * top_k + (300 if encoder == "large-embed" else 0)


class Hybrid:
    def __init__(self, alpha: float, top_k: int) -> None:
        self.quality = 0.56 + 0.012 * min(top_k, 12) - 0.25 * (alpha - 0.6) ** 2
        self.context = 900 + 450 * top_k


@tunable(
    retriever=OneOf(
        component(BM25, k1=Float(0.5, 2.0, default=1.2), b=Float(0.0, 1.0, default=0.75)),
        component(
            Dense, top_k=Int(2, 20, default=8), encoder=Choice(["small-embed", "large-embed"], default="small-embed")
        ),
        component(Hybrid, alpha=Float(0.0, 1.0, default=0.5), top_k=Int(2, 20, default=8)),
        labels=["bm25", "dense", "hybrid"],
    ),
    reranker=Switch(
        {"off": {}, "on": {"depth": Int(5, 50, default=20), "threshold": Float(0.0, 1.0, default=0.3)}}, default="off"
    ),
    temperature=Float(0.0, 1.2, default=0.5),
    max_turns=Int(1, 10, default=4),
    system_prompt=TextValue(default="Answer the question from the retrieved passages."),
)
def retrieval_agent(retriever, reranker, temperature, max_turns, system_prompt):
    option, settings = reranker
    quality, context = retriever.quality, retriever.context
    if option == "on":
        quality += 0.08 - 0.15 * (settings["threshold"] - 0.35) ** 2
        context = context * (0.6 + 0.4 * settings["threshold"]) + 30 * settings["depth"]
    quality += 0.04 * min(max_turns, 4) / 4 - 0.25 * (temperature - 0.3) ** 2
    p = Profile(
        metrics={"answer_f1": "quality"},
        quality=quality,
        credit=True,
        steps=0.6 * min(max_turns, 5),
        context=context + 800,
        growth=context * 0.4,
        latency=1.8,
    )
    apply_rules(p, {"system_prompt": system_prompt})
    return agent(p, system_prompt)


def independent_running(record: Path) -> tuple[Study, int]:
    t = by_type(retrieval_agent)
    structure, numbers = Block(TPE(seed=21), *t["structure"]), Block(TPE(seed=22), *t["numbers"])
    text = Block(ScriptedReflection(seed=23), *t["text"])
    study = Study(
        repetitions=2,
        agent=retrieval_agent,
        experiment=experiment(QA_CASES, {"answer_f1": "quality"}, credit=True),
        objectives=Independent(Maximize("answer_f1"), Minimize("tokens")),
        search=Search(
            blocks=[structure, numbers, text],
            schedule=Nested(outer=structure, inner=Alternate(numbers, text, iterations=2), inner_iterations=6),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 25


# ---- weighted-sum: a summarizer --------------------------------------------------------------------------------------
@tunable(
    temperature=Float(0.0, 1.2, default=0.7),
    max_turns=Int(1, 12, default=6),
    context_budget=Int(2000, 32000, default=8000, step=1000),
    cache_prompt=Choice([False, True], default=False),
    system_prompt=TextValue(default="Summarize what the documents say about the question."),
)
def summary_agent(temperature, max_turns, context_budget, cache_prompt, system_prompt):
    quality = (
        0.40
        + 0.25 * (1 - math.exp(-context_budget / 9000))
        + 0.05 * min(max_turns, 6) / 6
        - 0.3 * (temperature - 0.4) ** 2
    )
    p = Profile(
        metrics={"quality": "quality"},
        quality=quality,
        credit=True,
        steps=0.4 * max_turns,
        context=600 + context_budget * 0.45,
        growth=context_budget * 0.08,
        latency=1.6 if cache_prompt else 2.0,
        cached=0.55 if cache_prompt else 0.0,
    )
    apply_rules(p, {"system_prompt": system_prompt})
    return agent(p, system_prompt)


def weighted_sum(record: Path) -> tuple[Study, int]:
    t = by_type(summary_agent)
    study = Study(
        repetitions=2,
        agent=summary_agent,
        experiment=experiment(QA_CASES, {"quality": "quality"}, credit=True),
        objectives=WeightedSum(Maximize("quality", scale=0.05), Minimize("tokens", scale=5000)),
        search=Search(
            blocks=[Block(TPE(seed=31), *t["numbers"]), Block(ScriptedReflection(seed=32), *t["text"])],
            schedule=Joint(),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 30


# ---- lexicographic and stopped: a math solver with a choice of strategy ----------------------------------------------
class Direct:
    fx = dict(quality=0.45, steps=0.0, context=900, growth=0, output=150)


class ChainOfThought:
    def __init__(self, steps: int) -> None:
        self.fx = dict(quality=0.52 + 0.04 * min(steps, 5), steps=0.0, context=900, growth=0, output=180 + 120 * steps)


class ToolFirst:
    def __init__(self, max_calls: int) -> None:
        self.fx = dict(
            quality=0.58 + 0.02 * min(max_calls, 5), steps=0.5 * max_calls, context=1100, growth=500, output=160
        )


class SelfVerify:
    def __init__(self, passes: int) -> None:
        self.fx = dict(
            quality=0.62 + 0.05 * passes, steps=float(passes), context=1300, growth=900, output=260 + 60 * passes
        )


@tunable(
    strategy=OneOf(
        component(Direct),
        component(ChainOfThought, steps=Int(1, 8, default=3)),
        component(ToolFirst, max_calls=Int(1, 10, default=4)),
        component(SelfVerify, passes=Int(1, 3, default=1)),
        labels=["direct", "chain_of_thought", "tool_first", "self_verify"],
    ),
    temperature=Float(0.0, 1.2, default=0.7),
    system_prompt=TextValue(default="Solve the problem. Give the final answer as a number."),
)
def math_agent(strategy, temperature, system_prompt):
    p = Profile(metrics={"accuracy": "quality"}, credit=True, latency=1.5, **strategy.fx)
    p.quality -= 0.35 * (temperature - 0.2) ** 2
    apply_rules(p, {"system_prompt": system_prompt})
    return agent(p, system_prompt)


def lexicographic(record: Path) -> tuple[Study, int]:
    t = by_type(math_agent)
    structure = Block(TPE(seed=41), *t["structure"])
    study = Study(
        repetitions=1,
        agent=math_agent,
        experiment=experiment(MATH_CASES, {"accuracy": "quality"}, credit=True),
        objectives=Lexicographic(Maximize("accuracy", tolerance=0.03), Minimize("tokens")),
        search=Search(
            blocks=[structure, Block(TPE(seed=42), *t["numbers"]), Block(ScriptedReflection(seed=43), *t["text"])],
            schedule=Nested(outer=structure, inner=Joint(), inner_iterations=5),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 25


def stopped(record: Path) -> tuple[Study, int]:
    t = by_type(math_agent)
    structure = Block(TPE(seed=51), *t["structure"])
    study = Study(
        repetitions=1,
        agent=math_agent,
        experiment=experiment(MATH_CASES, {"accuracy": "quality"}, credit=True),
        objectives=Maximize("accuracy"),
        constraints=[AtMost("tokens", 2500)],
        search=Search(
            blocks=[structure, Block(TPE(seed=52), *t["numbers"]), Block(ScriptedReflection(seed=53), *t["text"])],
            schedule=Nested(outer=structure, inner=StopAfter(8), inner_iterations=4),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 20


# ---- language: a system prompt, a tool's description and a skill library, rewritten and curated ---------------------
RULES["tools.lookup"] = TOOL_RULES
SKILLS = {  # skills a curator might write, and what each does when the agent has it
    "refund-policy": ("Use when a customer asks for a refund or a credit.", {"quality": 0.06}),
    "escalation": ("Use when the issue needs an engineer: collect logs, versions and steps first.", {"quality": 0.04}),
    "kb-search": ("Use when searching the knowledge base: narrow queries, cite the KB id.", {"steps": -0.4}),
    "verbose-apology": ("Use at the start of every reply: apologize at length.", {"format": -0.08, "output": 1.4}),
}


class _Library(Draw):
    def __init__(self, values: dict[str, str | None], origin: dict[str, Any]) -> None:
        self.values, self.origin = values, origin

    def suggest_skills(self, name: str, default: str | None) -> str | None:
        return self.values.get(name, default)


class ScriptedCuration(Method):
    """In place of skill-library curation (a model with a shell): each call creates one skill a curator might learn
    from the held candidate's trials, or retires one that hurt, or leaves the library as it is. Each version is stored
    by content under the study's artifacts, as the real method stores them."""

    kinds = ("skills",)

    def __init__(self, seed: int = 0) -> None:
        self.seed, self.rng = seed, random.Random(seed)

    def describe(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, "seed": self.seed}

    def begin(self, block, base, history, structure, study, objective):
        from strands_evals.optimization.space.skills import store

        name, rng = type(self).__name__, self.rng
        keys = [k for k in block.keys if base is not None and k in base.params]
        if base is None or not keys:
            return _Library({}, {"method": name, "from_candidate": None})
        key = keys[0]
        held = base.params[key]
        have = sorted(os.listdir(held)) if held else []
        usage = {
            "input_tokens": float(rng.randint(40_000, 90_000)),
            "output_tokens": float(rng.randint(1_000, 4_000)),
            "cache_read_tokens": 0.0,
            "cache_write_tokens": 0.0,
        }
        usage["tokens"] = sum(usage.values())
        origin: dict[str, Any] = {
            "method": name,
            "from_candidate": base.number,
            "usage": usage,
            "seconds": round(rng.uniform(30, 120), 1),
        }
        missing = [s for s in SKILLS if s not in have]
        harmful = [s for s in have if SKILLS[s][1].get("format", 0) < 0]
        roll = rng.random()
        if harmful and roll < 0.5:
            decisions, members = {"retire": [harmful[0]]}, [s for s in have if s != harmful[0]]
        elif missing and roll < 0.85:
            new = rng.choice(missing)
            decisions, members = {"create": [new]}, [*have, new]
        else:
            return _Library({key: held}, {**origin, "status": "unchanged", key: {"decisions": {}}})
        with tempfile.TemporaryDirectory() as tmp:
            for s in members:
                (Path(tmp) / s).mkdir()
                (Path(tmp) / s / "SKILL.md").write_text(f"---\nname: {s}\ndescription: {SKILLS[s][0]}\n---\n# {s}\n")
            value = store({s: str(Path(tmp) / s) for s in members}, study.artifacts / "skills")
        return _Library(
            {key: value}, {**origin, "status": "rewritten", "changed": [key], key: {"decisions": decisions}}
        )


@tunable(
    system_prompt=TextValue(default=PROMPT, about="the system prompt"),
    tools=tool_descriptions(lookup),
    skills=SkillLibrary(),
    temperature=Float(0.0, 1.2, default=0.7),
)
def skilled_agent(system_prompt, tools, skills, temperature):
    p = Profile(metrics=SUPPORT_METRICS, quality=0.5, tool_error=0.03 + 0.04 * temperature)
    apply_rules(p, {"system_prompt": system_prompt, "tools.lookup": tools[0].tool_spec["description"]})
    for s in sorted(os.listdir(skills)) if skills else []:
        for k, v in SKILLS[s][1].items():
            setattr(p, k, getattr(p, k) + v if k in ADDITIVE else getattr(p, k) * v)
    return Agent(model=Scripted(p), tools=tools, system_prompt=system_prompt, callback_handler=None)


def language(record: Path) -> tuple[Study, int]:
    t = by_type(skilled_agent)
    study = Study(
        repetitions=2,
        agent=skilled_agent,
        experiment=experiment(SUPPORT_CASES, SUPPORT_METRICS),
        objectives=Pareto(Maximize("resolved"), Minimize("tokens")),
        constraints=[AtLeast("format_score", 0.8)],
        search=Search(
            blocks=[
                Block(ScriptedReflection(seed=7), *t["text"]),
                Block(ScriptedCuration(seed=8), *t["skills"]),
                Block(TPE(seed=9), *t["numbers"]),
            ],
            schedule=Joint(),
        ),
        max_concurrency=4,
        record=record,
        log=None,
    )
    return study, 25


FIXTURES: dict[str, Callable[[Path], tuple[Study, int]]] = {
    "nested-deep": nested_deep,
    "joint-200-failures": joint_200_failures,
    "alternate-8num-3text": alternate_8num_3text,
    "independent-running": independent_running,
    "weighted-sum": weighted_sum,
    "lexicographic": lexicographic,
    "stopped": stopped,
    "language": language,
}


# ---- running ---------------------------------------------------------------------------------------------------------
def run(name: str, record: Path) -> None:
    RUN.update(seed=zlib.crc32(name.encode()), study=None, outage=False, grader_down=False)
    _INSTANCES.clear()
    FLAKY.clear()
    study, candidates = FIXTURES[name](record)
    RUN["study"] = study
    try:
        study.optimize(iterations=candidates - 1)  # the baseline, then the rest
    except RuntimeError as exc:
        if name != "stopped":
            raise
        print(f"   (stopped as intended: {exc})")
    finally:
        RUN["study"] = None


def events(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "study.jsonl").read_text().splitlines()]


def cut_mid_round(complete: Path, out: Path, round_index: int = 2, keep: int = 4) -> None:
    """Copy the record of a complete run up to the ``keep``-th candidate of its ``round_index``-th top-level round, as if
    the study were still running: the open scopes have no scope_end, and there is no finished event."""
    lines = (complete / "study.jsonl").read_text().splitlines()
    log = [json.loads(line) for line in lines]
    rounds = [e["id"] for e in log if e["type"] == "scope" and e["parent"] == 1]
    in_round = [i for i, e in enumerate(log) if e["type"] == "candidate" and rounds[round_index] in e["path"]]
    last = in_round[keep - 1]
    out.mkdir(parents=True)
    (out / "study.jsonl").write_text("\n".join(lines[: last + 1]) + "\n")
    (out / "transcripts").mkdir()
    for e in log[: last + 1]:
        for a in e.get("trials", []) if e["type"] == "candidate" else []:
            if a["transcript"]:
                name = f"{a['transcript']['hash']}.json"
                shutil.copy(complete / "transcripts" / name, out / "transcripts" / name)


def generate(name: str, root: Path) -> Path:
    out = root / f"fixture-{name}"
    if out.exists():
        shutil.rmtree(out)
    if name == "independent-running":
        with tempfile.TemporaryDirectory(prefix="fixture-complete-") as tmp:
            run(name, Path(tmp) / "record")
            cut_mid_round(Path(tmp) / "record", out)
    else:
        run(name, out)
    return out


# ---- checking and summarizing a record -------------------------------------------------------------------------------
def check(directory: Path) -> list[str]:
    """Check the record's structure (raising on a violation) and summarize it in a few lines."""
    log = events(directory)
    types = [e["type"] for e in log]
    assert types[0] == "study" and types[1] == "run", types[:2]
    scopes: dict[int, dict] = {}
    stack: list[int] = []
    candidates, told, repeats = [], [], []
    for i, e in enumerate(log[2:], start=2):
        kind = e["type"]
        if kind == "scope":
            assert e["parent"] == (stack[-1] if stack else None), f"scope {e['id']} opens outside the innermost scope"
            assert e["id"] not in scopes
            scopes[e["id"]] = {**e, "depth": len(stack) + 1, "candidates": [], "told": [], "ended": None}
            stack.append(e["id"])
        elif kind == "scope_end":
            assert stack and stack[-1] == e["id"], f"scope {e['id']} closes before scope {stack[-1:]}"
            scopes[stack.pop()]["ended"] = e
        elif kind == "candidate":
            assert "path" in e and "role" in e, f"candidate {e.get('number')} lacks path/role"
            assert e["number"] == len(candidates), f"candidate {e['number']} out of order"
            if e["role"] == "baseline":
                assert e["path"] == [] and e["scope"] is None
            else:
                assert e["path"] == stack and e["scope"] == stack[-1], (e["number"], e["path"], stack)
                for s in e["path"]:
                    scopes[s]["candidates"].append(e)
            if e["role"] == "outer":
                s = scopes[e["scope"]]
                assert not s["candidates"][:-1] and s["round"] is not None, (
                    f"outer candidate {e['number']} does not open"
                )
            candidates.append(e)
        elif kind == "repeat":
            assert e["candidate"] < len(candidates), f"a repeat of candidate {e['candidate']}, not measured yet"
            assert e["path"] == stack and e["scope"] == stack[-1], (e["candidate"], e["path"], stack)
            if e["role"] == "outer":  # the round opens on an earlier candidate
                s = scopes[e["scope"]]
                assert not s["candidates"] and s["round"] is not None, f"a repeat opens scope {e['scope']} late"
            repeats.append(e)
        elif kind == "told":
            assert stack and e["scope"] == stack[-1], f"told for scope {e['scope']} outside it"
            scopes[e["scope"]]["told"].append(e)
            told.append(e)
        elif kind in ("finished", "stopped"):
            assert i == len(log) - 1, f"{kind} is not the last event"
        else:
            raise AssertionError(f"unknown event {kind}")
    for s in scopes.values():
        if s["ended"] is not None:
            assert s["ended"]["ran"] == len(s["candidates"]), (s["id"], s["ended"]["ran"], len(s["candidates"]))
            assert len(s["candidates"]) <= s["limit"]
            if s["round"] is not None:
                assert len(s["told"]) == 1, f"round {s['id']} told {len(s['told'])} times"
    depth = Counter(s["depth"] for s in scopes.values())
    nodes = Counter((s["depth"], s["node"] or "<root>", s["kind"]) for s in scopes.values())
    status = Counter(t["status"] for t in candidates)
    trials = Counter(a["status"] for t in candidates for a in t["trials"])
    feasible = Counter(str(t["feasible"]) for t in candidates)
    lines = [
        f"candidates {len(candidates)}  roles {dict(Counter(t['role'] for t in candidates))}  statuses {dict(status)}"
        f"  repeats {len(repeats)}",
        f"trials {dict(trials)}  feasible {dict(feasible)}",
        f"scopes by depth {dict(sorted(depth.items()))}: "
        + ", ".join(f"d{d} {node} {kind} x{n}" for (d, node, kind), n in sorted(nodes.items())),
        f"open scopes {[s['id'] for s in scopes.values() if s['ended'] is None]}  told {len(told)} "
        f"(null {sum(e['candidate'] is None for e in told)})  last event {types[-1]}"
        + (f"  best {log[-1]['best']}" if types[-1] == "finished" else ""),
    ]
    head = log[0]
    if "architecture" in head["blocks"]:  # nested-deep: the memory level only under plan_execute
        rows = []
        for s in scopes.values():
            if s["depth"] == 2:
                opener = s["candidates"][0]
                children = [c for c in scopes.values() if c["parent"] == s["id"]]
                planner = opener["params"]["planner"]
                assert bool(children) == (planner == "plan_execute" and opener["eligible"]), (s["id"], planner)
                mem = [c["candidates"][0]["params"].get("planner.plan_execute.memory") for c in children]
                rows.append(f"r{s['round']} {planner}:{len(s['candidates'])}" + (f" memory{mem}" if mem else ""))
        lines.append("architecture rounds: " + "  ".join(rows))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="the directory to write fixture-<name>/ records in")
    parser.add_argument("--only", nargs="*", choices=list(FIXTURES), help="default: every fixture")
    args = parser.parse_args()
    logging.getLogger("strands").setLevel(logging.CRITICAL)  # the agents' scripted failures are in the records
    total = time.monotonic()
    for name in args.only or list(FIXTURES):
        start = time.monotonic()
        out = generate(name, args.out)
        print(f"== {name} -> {out} ({time.monotonic() - start:.1f}s)")
        for line in check(out):
            print(f"   {line}")
    print(f"done in {time.monotonic() - total:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
