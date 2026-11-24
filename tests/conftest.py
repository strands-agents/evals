"""Fixtures that apply to every unit test."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_usage_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn usage reporting off, so a test that runs the CLI never sends a real event.

    `tests/strands_evals/cli/test_usage.py` turns it back on for the tests that are about
    reporting, and stubs the send.
    """
    monkeypatch.setenv("STRANDS_EVALS_TELEMETRY", "0")
