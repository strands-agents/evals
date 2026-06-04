"""Shared fixtures for CLI tests.

Builds minimal Session / Experiment / EvaluationReport JSON files in a temp
directory by serializing in-memory instances. Keeps fixtures schema-aligned
without committing static JSON to the tree.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from strands_evals.case import Case
from strands_evals.evaluators.output_evaluator import OutputEvaluator
from strands_evals.experiment import Experiment
from strands_evals.types.evaluation_report import EvaluationReport
from strands_evals.types.trace import (
    AgentInvocationSpan,
    InferenceSpan,
    Session,
    SpanInfo,
    TextContent,
    ToolConfig,
    Trace,
    UserMessage,
)


def _span_info(span_id: str, session_id: str = "sess_1") -> SpanInfo:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SpanInfo(
        session_id=session_id,
        span_id=span_id,
        trace_id="trace_1",
        start_time=now,
        end_time=now,
    )


def _build_session() -> Session:
    spans = [
        AgentInvocationSpan(
            span_info=_span_info("span_1"),
            user_prompt="Hello",
            agent_response="Hi there",
            available_tools=[ToolConfig(name="search")],
        ),
        InferenceSpan(
            span_info=_span_info("span_2"),
            messages=[UserMessage(content=[TextContent(text="Hello")])],
        ),
    ]
    return Session(
        session_id="sess_1",
        traces=[Trace(trace_id="trace_1", session_id="sess_1", spans=spans)],
    )


def _build_experiment() -> Experiment:
    return Experiment(
        cases=[Case[str, str](name="c1", input="What is 2+2?", expected_output="4")],
        evaluators=[OutputEvaluator(rubric="The output should be correct.")],
    )


def _build_report() -> EvaluationReport:
    return EvaluationReport(
        evaluator_name="OutputEvaluator",
        overall_score=0.75,
        scores=[1.0, 0.5],
        cases=[
            {"name": "c1", "input": "q1", "actual_output": "a1"},
            {"name": "c2", "input": "q2", "actual_output": "a2"},
        ],
        test_passes=[True, False],
        reasons=["good", "wrong answer"],
        detailed_results=[[], []],
        diagnoses=[None, None],
        recommendations=[None, "use a different prompt"],
    )


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    """Build session.json, experiment.json, reports.json in a temp directory."""
    session_path = tmp_path / "session.json"
    session_path.write_text(_build_session().model_dump_json(indent=2), encoding="utf-8")

    experiment_path = tmp_path / "experiment.json"
    _build_experiment().to_file(str(experiment_path))

    reports_path = tmp_path / "reports.json"
    reports_path.write_text(_build_report().model_dump_json(indent=2), encoding="utf-8")

    return tmp_path


@pytest.fixture
def session_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "session.json"


@pytest.fixture
def experiment_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "experiment.json"


@pytest.fixture
def reports_file(fixtures_dir: Path) -> Path:
    return fixtures_dir / "reports.json"
