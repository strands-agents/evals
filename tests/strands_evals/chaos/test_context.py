"""Unit tests for the chaos _context module."""

from strands_evals.chaos import ChaosScenario
from strands_evals.chaos._context import _current_scenario


class TestContextVar:
    """Tests for the _current_scenario ContextVar."""

    def test_set_and_get(self):
        scenario = ChaosScenario(name="test_scenario")
        token = _current_scenario.set(scenario)
        try:
            assert _current_scenario.get() is scenario
            assert _current_scenario.get().name == "test_scenario"
        finally:
            _current_scenario.reset(token)

    def test_nested_set_and_reset(self):
        s1 = ChaosScenario(name="outer")
        s2 = ChaosScenario(name="inner")

        token1 = _current_scenario.set(s1)
        try:
            assert _current_scenario.get().name == "outer"
            token2 = _current_scenario.set(s2)
            try:
                assert _current_scenario.get().name == "inner"
            finally:
                _current_scenario.reset(token2)
            assert _current_scenario.get().name == "outer"
        finally:
            _current_scenario.reset(token1)
