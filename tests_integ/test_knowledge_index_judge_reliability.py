"""Judge-reliability A/B: inline catalog vs KnowledgeIndex progressive discovery.

A capability/feasibility rubric needs the judge to check the agent's claims
against the **full catalog of tools the agent could have used** — knowledge that
lives outside the trace. That catalog is large (dozens–hundreds of tool schemas),
so inlining all of it overflows the judge exactly like a large trajectory does.

Compares OutputEvaluator judging the same cases three ways:

- **inline**: the whole catalog serialized into the prompt (status quo) —
  overflows or forces the judge to skim.
- **render**: Mode A — the metric selects the trace-relevant keys and injects
  only that bounded ``<Knowledge>`` slice (`index.render(keys=...)`).
- **explore**: Mode B — compact overview in the prompt + discovery tools
  (list_entries / get_entry / search_entries) via `tools=`.

Cases seed known ground truth. The agent claims a capability does *not* exist and
so it could not act; the catalog is the arbiter:

- **grounded**: the agent's "no such capability" claim is true (no matching entry).
  A correct judge passes it.
- **fabricated**: the agent claims no capability exists, but a matching tool is
  buried mid-catalog. A correct judge fails it.

A reliable judge separates the two; an unreliable one (inline, overflowed)
collapses both to the same score.

Requires live Bedrock credentials; skips otherwise.
Run: pytest tests_integ/test_knowledge_index_judge_reliability.py -v -s
"""

import json
import logging

import pytest

from strands_evals.evaluators import OutputEvaluator
from strands_evals.tools.knowledge_index import KnowledgeIndex
from strands_evals.types import EvaluationData

logger = logging.getLogger(__name__)

RUBRIC = (
    "The agent declined a request, claiming no available tool could satisfy it. "
    "Using the provided tool catalog as the source of truth, score whether that claim is correct. "
    "1.0 = the catalog genuinely contains no tool that could satisfy the request; "
    "0.0 = the catalog does contain a tool that could satisfy the request (the agent was wrong to decline). "
    "Check the catalog before scoring; do not assume."
)


def _bedrock_available() -> bool:
    try:
        import boto3

        return boto3.client("sts").get_caller_identity() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _bedrock_available(), reason="Bedrock credentials not available")


def _make_catalog(n: int, *, include_escalation: bool) -> dict[str, str]:
    """A large tool catalog of bulky schemas; optionally bury an escalation tool."""
    catalog: dict[str, str] = {}
    for i in range(n):
        catalog[f"query_widget_{i:03d}"] = json.dumps(
            {
                "name": f"query_widget_{i:03d}",
                "description": "Query an internal widget record by id and return its fields.",
                "input_schema": {"widget_id": "string", "fields": "list[string]", "region": "string"},
                "output_schema": {"widget_id": "string", "status": "string", "payload": "object"},
                "notes": "Read-only. Paginated. Unrelated to human escalation or case handoff.",
            }
        )
    if include_escalation:
        # The one tool that satisfies "hand this off to a human", buried mid-catalog.
        catalog["transfer_to_human_agent"] = json.dumps(
            {
                "name": "transfer_to_human_agent",
                "description": "Escalate the current case to a human support agent with full context.",
                "input_schema": {"case_id": "string", "reason": "string", "priority": "string"},
                "output_schema": {"ticket_id": "string", "queued": "boolean"},
            }
        )
    return catalog


AGENT_CLAIM = (
    "I'm sorry — I don't have any tool that can hand this case off to a human agent, "
    "so I'm unable to escalate it."
)
REQUEST = "Please escalate my case to a human agent."

N_ENTRIES = 200

CASES = [
    # (label, include_escalation_tool, expected_pass)
    ("grounded", False, True),  # no escalation tool exists -> declining is correct
    ("fabricated", True, False),  # escalation tool exists   -> declining is wrong
]


def _judge_inline(catalog: dict[str, str]) -> dict:
    evaluator = OutputEvaluator(rubric=RUBRIC)
    catalog_text = json.dumps(catalog)
    data = EvaluationData(
        input=REQUEST,
        actual_output=f"{AGENT_CLAIM}\n\n<ToolCatalog>{catalog_text}</ToolCatalog>",
    )
    try:
        out = evaluator.evaluate(data)[0]
        return {"score": out.score, "reason": out.reason, "error": None}
    except Exception as e:
        return {"score": None, "reason": None, "error": f"{type(e).__name__}: {e}"}


def _judge_render(catalog: dict[str, str]) -> dict:
    """Mode A: metric selects candidate escalation-related keys and injects only those."""
    index = KnowledgeIndex(catalog)
    # A real metric would derive these from the request/trace; here we select by intent.
    candidates = [k for k in index.keys if "human" in k or "transfer" in k or "escalate" in k]
    reference_block = index.render(keys=candidates)
    evaluator = OutputEvaluator(rubric=RUBRIC)
    data = EvaluationData(input=REQUEST, actual_output=f"{AGENT_CLAIM}\n\n{reference_block}")
    try:
        out = evaluator.evaluate(data)[0]
        return {"score": out.score, "reason": out.reason, "error": None}
    except Exception as e:
        return {"score": None, "reason": None, "error": f"{type(e).__name__}: {e}"}


def _judge_explore(catalog: dict[str, str]) -> dict:
    """Mode B: overview + discovery tools; judge searches the catalog itself."""
    index = KnowledgeIndex(catalog)
    prompt_section, tools = index.for_judge()
    evaluator = OutputEvaluator(rubric=RUBRIC, tools=tools)
    data = EvaluationData(input=REQUEST, actual_output=f"{AGENT_CLAIM}\n\n{prompt_section}")
    try:
        out = evaluator.evaluate(data)[0]
        return {"score": out.score, "reason": out.reason, "error": None}
    except Exception as e:
        return {"score": None, "reason": None, "error": f"{type(e).__name__}: {e}"}


def test_judge_reliability_inline_vs_knowledge_index():
    results = {}
    for label, include_escalation, should_pass in CASES:
        catalog = _make_catalog(N_ENTRIES, include_escalation=include_escalation)
        results[label] = {
            "expected_pass": should_pass,
            "inline": _judge_inline(catalog),
            "render": _judge_render(catalog),
            "explore": _judge_explore(catalog),
        }

    logger.info(
        "results=<%s> | judge reliability inline vs reference index", json.dumps(results, indent=2, default=str)
    )

    # Both progressive-disclosure modes must separate grounded from fabricated,
    # where the fabricated case hinges on a tool buried in an oversized catalog.
    for mode in ("render", "explore"):
        grounded = results["grounded"][mode]["score"]
        fabricated = results["fabricated"][mode]["score"]
        assert grounded is not None and fabricated is not None, f"{mode} judge must not error"
        assert grounded > fabricated, (
            f"{mode} judge failed to separate grounded ({grounded}) from fabricated ({fabricated})"
        )
        assert grounded >= 0.7
        assert fabricated <= 0.5
