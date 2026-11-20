"""The optimizer end to end on a real Strands agent: declaration -> run -> reviewable diff (#450).

The agent is a real `Agent` whose model is scripted (no network): it answers a case right with a probability set
by its temperature, its conversation manager, and whether its system prompt asks for the exact answer, and reports
token usage per call by manager. The judge is a real agent on a scripted model too, so its spans and usage are
kept apart from the agent's. Every run gives the same numbers.

One scenario per feature; each later PR adds its own.
"""

import json
import random
import shutil
from collections.abc import AsyncIterator

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from strands.models import Model

from strands_evals import Case, Experiment
from strands_evals.evaluators import Evaluator
from strands_evals.optimization import (
    AtMost,
    Block,
    Budget,
    FileText,
    Float,
    Int,
    Joint,
    Maximize,
    OptimizationResult,
    Optimizer,
    Patch,
    Proposer,
    Search,
    Selection,
    choice,
    component,
    tunable,
)
from strands_evals.types import EvaluationOutput

EXACT = "Give the exact answer."
JUDGE_TOKENS = 40


class World(Model):
    """The agent's scripted model."""

    def __init__(self, temperature: float, manager: object) -> None:
        self.temperature, self.manager = temperature, manager
        self.config = {"model_id": "scripted-world"}

    def update_config(self, **config):
        self.config.update(config)

    def get_config(self):
        return self.config

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterator[dict]:
        question = messages[-1]["content"][0]["text"]
        p = 0.95 - 0.8 * abs(self.temperature - 0.25)
        if isinstance(self.manager, SummarizingConversationManager):
            p, tokens = p + 0.03, int(300 + 100 * self.manager.summary_ratio)
        else:  # a sliding window, or none: always over the 450-token limit
            tokens = 500 + 5 * getattr(self.manager, "window_size", 40)
        if EXACT not in (system_prompt or ""):
            p *= 0.4
        right = random.Random(f"{question}:{self.temperature:.4f}:{tokens}:{system_prompt}").random() < p
        reply = question.replace("question", "answer") if right else "I'm not sure."
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": reply}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": tokens - 20, "outputTokens": 20, "totalTokens": tokens}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


class Verdict(Model):
    """The judge's scripted model: replies with nothing that matters; only its usage is read."""

    config = {"model_id": "scripted-judge"}

    def update_config(self, **config):
        pass

    def get_config(self):
        return self.config

    async def structured_output(self, *args, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs) -> AsyncIterator[dict]:
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": "ok"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        usage = {"inputTokens": JUDGE_TOKENS - 5, "outputTokens": 5, "totalTokens": JUDGE_TOKENS}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}


class Correct(Evaluator):
    """A mocked LLM judge: a real agent call (its usage is the judge's), then an exact-match verdict."""

    async def evaluate_async(self, data):
        await Agent(model=Verdict(), callback_handler=None).invoke_async("judge this")
        hit = str(data.actual_output).strip() == data.expected_output
        return [EvaluationOutput(score=float(hit), test_pass=hit, reason="right" if hit else "wrong answer")]


@tunable(
    manager=choice(
        window=component(SlidingWindowConversationManager, window_size=Int(4, 40)),
        summary=component(SummarizingConversationManager, summary_ratio=Float(0.1, 0.8)),
    ),
    temperature=Float(0.0, 1.0, default=0.9),
)
def manager_agent(manager=None, temperature=0.9):
    """A space without text: the default search is TPE only."""
    return Agent(
        model=World(temperature, manager), conversation_manager=manager, system_prompt=EXACT, callback_handler=None
    )


def _experiment(n=20):
    cases = [Case(name=f"q{i:02d}", input=f"question {i}", expected_output=f"answer {i}") for i in range(n)]
    return Experiment(cases=cases, evaluators=[Correct()])


def _run(tmp_path, factory=manager_agent, **kwargs):
    kwargs.setdefault("budget", Budget(max_iterations=6))
    optimizer = Optimizer(
        _experiment(),
        factory,
        objective=Maximize("Correct"),
        constraints=[AtMost("tokens", 450)],
        proposals_per_iteration=2,
        max_workers=40,
        base_dir=tmp_path / "runs",
        **kwargs,
    )
    return optimizer, optimizer.run()


def _events(root):
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


# component swap + a constraint, on the default search


def test_a_component_swap_under_a_token_limit_ships_with_its_values_and_files(tmp_path):
    optimizer, result = _run(tmp_path)
    # Every sliding window is over 450 tokens, so only the summarizing manager can ship.
    assert result.improved and result.best.values["manager"] == "summary"
    assert result.validation.infeasible[result.baseline_id].startswith("tokens <= 450")
    assert all(
        "tokens <= 450" in why
        for c, why in result.validation.infeasible.items()
        if result.candidate(c).values["manager"] == "window"
    )
    assert result.test.objective.mean_diff > 0 and result.test.difference["tokens"].mean_diff < 0

    root = result.run_dir
    changed = json.loads((root / "best.values.json").read_text())
    assert changed["manager"] == {"old": "window", "new": "summary"}
    assert changed["manager.window.window_size"] == {"old": 40}  # active on one side only
    assert (root / "best.diff").read_text() == ""  # nothing textual is tunable here
    assert json.loads((root / "effects.json").read_text())["reason"] is None
    assert OptimizationResult.from_file(root / "result.json") == result
    assert {b.name for b, _ in optimizer._blocks} == {"structure", "numbers"}


def test_the_judges_usage_is_kept_apart_from_the_agents(tmp_path):
    _, result = _run(tmp_path, budget=Budget(max_iterations=2))
    trials = [e["record"] for e in _events(result.run_dir) if e["event"] == "trial"]
    assert result.judge_cost.tokens == JUDGE_TOKENS * len(trials)
    # Each trial: the agent's one model call on its own model, the judge's on the judge's, never mixed.
    assert all(t["usage"]["models"] == ["scripted-world"] and t["usage"]["model_calls"] == 1 for t in trials)
    assert all(t["judge_usage"]["Correct"]["models"] == ["scripted-judge"] for t in trials)
    assert {split: usage.models for split, usage in result.agent_cost.items()} == {
        "train": ["scripted-world"],
        "validation": ["scripted-world"],
        "test": ["scripted-world"],
    }


# text, through a file, to a reviewable diff


class AddInstruction(Proposer):
    """A scripted text proposer: appends the missing instruction to the parent's text, once."""

    kinds = frozenset({"text"})

    def __init__(self):
        self.done = False

    async def propose(self, request):
        if self.done:
            return []
        self.done = True
        key = request.keys[0]
        return [Patch(updates={key: request.parent.values[key] + EXACT + "\n"}, rationale="ask for the exact answer")]

    def observe(self, observation):
        pass


def test_a_weakened_prompt_file_is_improved_into_a_diff_on_the_real_file(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "prompt.md").write_text("You answer questions.\n")
    monkeypatch.chdir(project)  # diff paths are shown relative to the working directory

    @tunable(prompt=FileText("prompt.md"), temperature=Float(0.0, 1.0, default=0.3))
    def prompt_agent(prompt, temperature=0.3):
        return Agent(
            model=World(temperature, SummarizingConversationManager()), system_prompt=prompt, callback_handler=None
        )

    optimizer, result = _run(tmp_path, prompt_agent, search=Search([Block(AddInstruction(), "prompt")]))
    assert result.improved and EXACT in result.best.values["prompt"]
    diff = (result.run_dir / "best.diff").read_text()
    assert "--- a/prompt.md" in diff and "+++ b/prompt.md" in diff and f"+{EXACT}" in diff
    assert (project / "prompt.md").read_text() == "You answer questions.\n"  # your file is never written
    assert result.validation.objectives[result.best.id] > result.validation.objectives[result.baseline_id]


# resume


def test_a_run_killed_mid_step_resumes_to_the_same_result(tmp_path):
    optimizer, original = _run(tmp_path)
    root = original.run_dir
    lines = (root / "events.jsonl").read_text().splitlines()
    names = [json.loads(line)["event"] for line in lines]
    second_end = [i for i, n in enumerate(names) if n == "iteration_end"][1]
    cut = next(i for i, n in enumerate(names) if i > second_end and n == "score") + 1
    killed = tmp_path / "killed" / root.name
    shutil.copytree(root, killed)
    (killed / "events.jsonl").write_text("\n".join(lines[:cut]) + "\n")

    resumed = Optimizer.resume(
        killed,
        experiment=_experiment(),
        factory=manager_agent,
        objective=Maximize("Correct"),
        constraints=[AtMost("tokens", 450)],
    ).run()
    assert resumed.best.id == original.best.id and resumed.candidates == original.candidates
    assert resumed.rollouts_spent == original.rollouts_spent  # every trial saved before the kill reused
    assert resumed.validation.objectives == original.validation.objectives
    assert resumed.test.objective == original.test.objective


# a custom selection strategy


class FromBaseline(Selection):
    """Always proposes from the baseline (the candidate with no parent)."""

    name = "from_baseline"

    def choose(self, ctx):
        return next(c for c in ctx.ranking if ctx.parents[c] is None)


def test_a_custom_selection_picks_every_parent(tmp_path):
    _, result = _run(tmp_path, selection=FromBaseline(), search=Search(schedule=Joint()))
    assert {r.parent_id for r in result.iterations} == {result.baseline_id}
    assert json.loads((result.run_dir / "run.json").read_text())["selection"]["name"] == "from_baseline"
