"""Tests for ``strands-evals fetch``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strands_evals.cli.main import main
from strands_evals.providers.exceptions import SessionNotFoundError
from strands_evals.types.evaluation import TaskOutput
from strands_evals.types.trace import Session


def _task_output(session: Session) -> TaskOutput:
    return {"output": "hi", "trajectory": session}


# --- cloudwatch ----------------------------------------------------------------


def test_fetch_cloudwatch_writes_session_json(session_file: Path, tmp_path: Path):
    session = Session.model_validate_json(session_file.read_text())
    out_path = tmp_path / "out.json"

    provider = MagicMock()
    provider.get_evaluation_data.return_value = _task_output(session)

    with patch(
        "strands_evals.providers.cloudwatch_provider.CloudWatchProvider",
        return_value=provider,
    ) as cw_cls:
        exit_code = main(
            [
                "fetch",
                "cloudwatch",
                "--session-id",
                "sess_1",
                "--log-group",
                "/aws/foo",
                "--region",
                "us-west-2",
                "--lookback-days",
                "7",
                "-o",
                str(out_path),
            ]
        )

    assert exit_code == 0
    cw_cls.assert_called_once_with(
        region="us-west-2",
        log_group="/aws/foo",
        agent_name=None,
        lookback_days=7,
    )
    provider.get_evaluation_data.assert_called_once_with("sess_1")

    payload = json.loads(out_path.read_text())
    assert payload["session_id"] == "sess_1"


def test_fetch_cloudwatch_to_stdout(session_file: Path, capsys):
    session = Session.model_validate_json(session_file.read_text())

    provider = MagicMock()
    provider.get_evaluation_data.return_value = _task_output(session)

    with patch(
        "strands_evals.providers.cloudwatch_provider.CloudWatchProvider",
        return_value=provider,
    ):
        exit_code = main(
            [
                "fetch",
                "cloudwatch",
                "--session-id",
                "sess_1",
                "--log-group",
                "/aws/foo",
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["session_id"] == "sess_1"


def test_fetch_cloudwatch_requires_one_target(capsys):
    """Both --log-group and --agent-name omitted (or both set) → exit 2."""
    exit_code = main(["fetch", "cloudwatch", "--session-id", "sess_1"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "exactly one of --log-group or --agent-name" in captured.err


def test_fetch_cloudwatch_rejects_both_targets(capsys):
    exit_code = main(
        [
            "fetch",
            "cloudwatch",
            "--session-id",
            "sess_1",
            "--log-group",
            "/aws/foo",
            "--agent-name",
            "agent-x",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "exactly one of --log-group or --agent-name" in captured.err


def test_fetch_cloudwatch_provider_error_exits_3(session_file: Path):
    provider = MagicMock()
    provider.get_evaluation_data.side_effect = SessionNotFoundError("nope")

    with patch(
        "strands_evals.providers.cloudwatch_provider.CloudWatchProvider",
        return_value=provider,
    ):
        exit_code = main(
            [
                "fetch",
                "cloudwatch",
                "--session-id",
                "missing",
                "--log-group",
                "/aws/foo",
            ]
        )

    assert exit_code == 3


# --- langfuse -----------------------------------------------------------------


def test_fetch_langfuse_writes_session_json(session_file: Path, tmp_path: Path):
    session = Session.model_validate_json(session_file.read_text())
    out_path = tmp_path / "out.json"

    provider = MagicMock()
    provider.get_evaluation_data.return_value = _task_output(session)

    with patch(
        "strands_evals.providers.langfuse_provider.LangfuseProvider",
        return_value=provider,
    ) as lf_cls:
        exit_code = main(
            [
                "fetch",
                "langfuse",
                "--session-id",
                "sess_1",
                "--host",
                "https://lf.example.com",
                "-o",
                str(out_path),
            ]
        )

    assert exit_code == 0
    lf_cls.assert_called_once_with(host="https://lf.example.com", timeout=120)
    provider.get_evaluation_data.assert_called_once_with("sess_1")
    assert json.loads(out_path.read_text())["session_id"] == "sess_1"


def test_fetch_langfuse_missing_extra_exits_2(monkeypatch, capsys):
    """Setting the provider module to None makes `from ... import` raise ImportError."""
    monkeypatch.setitem(sys.modules, "strands_evals.providers.langfuse_provider", None)

    exit_code = main(["fetch", "langfuse", "--session-id", "sess_1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "langfuse extra not installed" in captured.err


# --- opensearch ---------------------------------------------------------------


def test_fetch_opensearch_writes_session_json(session_file: Path, tmp_path: Path):
    session = Session.model_validate_json(session_file.read_text())
    out_path = tmp_path / "out.json"

    provider = MagicMock()
    provider.get_evaluation_data.return_value = _task_output(session)

    with patch(
        "strands_evals.providers.opensearch_provider.OpenSearchProvider",
        return_value=provider,
    ) as os_cls:
        exit_code = main(
            [
                "fetch",
                "opensearch",
                "--session-id",
                "sess_1",
                "--host",
                "https://os.example.com",
                "--index",
                "my-spans-*",
                "--username",
                "u",
                "--password",
                "p",
                "--no-verify-certs",
                "-o",
                str(out_path),
            ]
        )

    assert exit_code == 0
    os_cls.assert_called_once_with(
        host="https://os.example.com",
        index="my-spans-*",
        auth=("u", "p"),
        verify_certs=False,
    )
    assert json.loads(out_path.read_text())["session_id"] == "sess_1"


def test_fetch_opensearch_default_no_auth(session_file: Path, capsys):
    session = Session.model_validate_json(session_file.read_text())
    provider = MagicMock()
    provider.get_evaluation_data.return_value = _task_output(session)

    with patch(
        "strands_evals.providers.opensearch_provider.OpenSearchProvider",
        return_value=provider,
    ) as os_cls:
        exit_code = main(["fetch", "opensearch", "--session-id", "sess_1"])

    assert exit_code == 0
    os_cls.assert_called_once_with(
        host="https://localhost:9200",
        index="otel-v1-apm-span-*",
        auth=None,
        verify_certs=True,
    )


def test_fetch_opensearch_missing_extra_exits_2(capsys):
    """OpenSearch's heavy dep is imported lazily inside __init__, so the
    constructor — not the module import — raises ImportError when the extra
    is absent. Reproduce that by patching the provider class to raise on
    instantiation; assert exit 2 + install hint.
    """
    with patch(
        "strands_evals.providers.opensearch_provider.OpenSearchProvider",
        side_effect=ImportError("opensearch_genai_observability_sdk_py is required"),
    ):
        exit_code = main(["fetch", "opensearch", "--session-id", "sess_1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "opensearch extra not installed" in captured.err


def test_fetch_opensearch_partial_credentials_rejected(capsys):
    exit_code = main(
        [
            "fetch",
            "opensearch",
            "--session-id",
            "sess_1",
            "--username",
            "u",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "must be provided together" in captured.err


# --- subcommand routing -------------------------------------------------------


def test_fetch_requires_provider():
    with pytest.raises(SystemExit):
        main(["fetch"])
