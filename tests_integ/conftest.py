"""Shared fixtures for trace provider integration tests.

Each provider test module defines its own `provider` and `session_id` fixtures.
This conftest provides common fixtures that build on those.
"""

import json
import logging
import os
from datetime import datetime, timezone

import boto3
import pytest

from strands_evals.types.trace import AgentInvocationSpan, Session, SpanInfo, Trace

logger = logging.getLogger(__name__)


def _load_api_keys_from_secrets_manager():
    """Load API keys as environment variables from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager")
    if "STRANDS_TEST_API_KEYS_SECRET_NAME" in os.environ:
        try:
            secret_name = os.environ["STRANDS_TEST_API_KEYS_SECRET_NAME"]
            response = client.get_secret_value(SecretId=secret_name)

            if "SecretString" in response:
                secret = json.loads(response["SecretString"])
                for key, value in secret.items():
                    os.environ[f"{key.upper()}_API_KEY"] = str(value)

        except Exception as e:
            logger.warning("Error retrieving secret: %s", e)


def pytest_sessionstart(session):
    """Load API keys from Secrets Manager at session start."""
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("AWS_REGION", "us-east-1")
    _load_api_keys_from_secrets_manager()


@pytest.fixture(scope="module")
def evaluation_data(provider, session_id):
    """Fetch evaluation data for the test session."""
    return provider.get_evaluation_data(session_id)


@pytest.fixture
def make_single_turn_session():
    """Build a one-turn Session so trace-level judges can run without a live agent."""

    def _make(user_prompt: str, agent_response: str, session_id: str = "integ-session") -> Session:
        now = datetime.now(timezone.utc)
        span = AgentInvocationSpan(
            span_info=SpanInfo(session_id=session_id, start_time=now, end_time=now),
            user_prompt=user_prompt,
            agent_response=agent_response,
            available_tools=[],
        )
        trace = Trace(spans=[span], trace_id=f"{session_id}-trace", session_id=session_id)
        return Session(traces=[trace], session_id=session_id)

    return _make
