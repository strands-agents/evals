"""Unit tests for the chaos _context module."""

from strands_evals.chaos import ChaosCase
from strands_evals.chaos._context import _current_chaos_case


class TestContextVar:
    """Tests for the _current_chaos_case ContextVar."""

    def test_default_is_none(self):
        assert _current_chaos_case.get() is None

    def test_set_and_get(self):
        case = ChaosCase(name="test_case", input="hello")
        token = _current_chaos_case.set(case)
        try:
            assert _current_chaos_case.get() is case
            assert _current_chaos_case.get().name == "test_case"
        finally:
            _current_chaos_case.reset(token)

    def test_nested_set_and_reset(self):
        c1 = ChaosCase(name="outer", input="hello")
        c2 = ChaosCase(name="inner", input="world")

        token1 = _current_chaos_case.set(c1)
        try:
            assert _current_chaos_case.get().name == "outer"
            token2 = _current_chaos_case.set(c2)
            try:
                assert _current_chaos_case.get().name == "inner"
            finally:
                _current_chaos_case.reset(token2)
            assert _current_chaos_case.get().name == "outer"
        finally:
            _current_chaos_case.reset(token1)

    def test_reset_restores_none(self):
        case = ChaosCase(name="test", input="hello")
        token = _current_chaos_case.set(case)
        _current_chaos_case.reset(token)
        assert _current_chaos_case.get() is None
