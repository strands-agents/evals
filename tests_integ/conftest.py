"""Shared fixtures for trace provider integration tests.

Each provider test module defines its own `provider` and `session_id` fixtures.
This conftest provides common fixtures that build on those.
"""

import pytest


@pytest.fixture(scope="module")
def evaluation_data(provider, session_id):
    """Fetch evaluation data for the test session."""
    return provider.get_evaluation_data(session_id)
