"""Tests for ``strands-evals diagnose``."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

from strands_evals.cli.main import main
from strands_evals.types.detector import (
    ConfidenceLevel,
    DiagnosisResult,
    FailureItem,
    FailureOutput,
    RCAItem,
    RCAOutput,
)


def _empty_diagnosis() -> DiagnosisResult:
    return DiagnosisResult(session_id="sess_1", failures=[], root_causes=[])


def _populated_diagnosis() -> DiagnosisResult:
    failure = FailureItem(
        span_id="span_1",
        category=["hallucination"],
        confidence=[0.9],
        evidence=["agent invented a tool"],
    )
    rca = RCAItem(
        failure_span_id="span_1",
        location="span_1",
        causality="PRIMARY_FAILURE",
        propagation_impact=["TASK_TERMINATION"],
        failure_detection_timing="IMMEDIATELY_AT_OCCURRENCE",
        completion_status="COMPLETE_FAILURE",
        root_cause_explanation="Tool hallucinated",
        fix_type="SYSTEM_PROMPT_FIX",
        fix_recommendation="Tighten the system prompt",
    )
    return DiagnosisResult(session_id="sess_1", failures=[failure], root_causes=[rca])


def test_diagnose_default_calls_diagnose_session(session_file: Path, capsys):
    with patch(
        "strands_evals.cli.commands.diagnose.diagnose_session",
        return_value=_populated_diagnosis(),
    ) as mock_diag:
        exit_code = main(["--json", "diagnose", str(session_file)])

    captured = capsys.readouterr()
    assert exit_code == 0
    mock_diag.assert_called_once()
    _, kwargs = mock_diag.call_args
    assert kwargs == {"model": None, "confidence_threshold": ConfidenceLevel.LOW}

    payload = json.loads(captured.out)
    assert payload["session_id"] == "sess_1"
    assert len(payload["failures"]) == 1
    assert len(payload["root_causes"]) == 1


def test_diagnose_detect_only(session_file: Path, capsys):
    failure = FailureItem(
        span_id="span_1",
        category=["bad"],
        confidence=[0.7],
        evidence=["e"],
    )
    output = FailureOutput(session_id="sess_1", failures=[failure])

    with patch(
        "strands_evals.cli.commands.diagnose.detect_failures",
        return_value=output,
    ) as mock_detect:
        exit_code = main(
            [
                "--json",
                "diagnose",
                str(session_file),
                "--detect-only",
                "--confidence",
                "medium",
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    mock_detect.assert_called_once()
    _, kwargs = mock_detect.call_args
    assert kwargs["confidence_threshold"] == ConfidenceLevel.MEDIUM
    assert kwargs["model"] is None

    payload = json.loads(captured.out)
    assert payload["session_id"] == "sess_1"
    assert "failures" in payload


def test_diagnose_rca_only(session_file: Path, capsys):
    rca = RCAItem(
        failure_span_id="span_1",
        location="span_1",
        causality="PRIMARY_FAILURE",
        propagation_impact=[],
        failure_detection_timing="ONLY_AT_TASK_END",
        completion_status="PARTIAL_SUCCESS",
        root_cause_explanation="x",
        fix_type="OTHERS",
        fix_recommendation="y",
    )
    output = RCAOutput(root_causes=[rca])

    with patch(
        "strands_evals.cli.commands.diagnose.analyze_root_cause",
        return_value=output,
    ) as mock_rca:
        exit_code = main(["--json", "diagnose", str(session_file), "--rca-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    mock_rca.assert_called_once()
    payload = json.loads(captured.out)
    assert len(payload["root_causes"]) == 1


def test_diagnose_model_override(session_file: Path):
    with patch(
        "strands_evals.cli.commands.diagnose.diagnose_session",
        return_value=_empty_diagnosis(),
    ) as mock_diag:
        exit_code = main(["--json", "diagnose", str(session_file), "--model", "anthropic.claude-sonnet"])

    assert exit_code == 0
    _, kwargs = mock_diag.call_args
    assert kwargs["model"] == "anthropic.claude-sonnet"


def test_diagnose_output_file(session_file: Path, tmp_path: Path):
    with patch(
        "strands_evals.cli.commands.diagnose.diagnose_session",
        return_value=_populated_diagnosis(),
    ):
        out_path = tmp_path / "diag.json"
        exit_code = main(["diagnose", str(session_file), "-o", str(out_path)])

    assert exit_code == 0
    payload = json.loads(out_path.read_text())
    assert payload["session_id"] == "sess_1"


def test_diagnose_stdin(session_file: Path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(session_file.read_text()))
    with patch(
        "strands_evals.cli.commands.diagnose.diagnose_session",
        return_value=_empty_diagnosis(),
    ) as mock_diag:
        exit_code = main(["--json", "diagnose", "-"])

    captured = capsys.readouterr()
    assert exit_code == 0
    mock_diag.assert_called_once()
    payload = json.loads(captured.out)
    assert payload == {"session_id": "sess_1", "failures": [], "root_causes": []}


def test_diagnose_detect_and_rca_mutually_exclusive(session_file: Path, capsys):
    """argparse should reject --detect-only and --rca-only together."""
    import pytest

    with pytest.raises(SystemExit):
        main(["diagnose", str(session_file), "--detect-only", "--rca-only"])
