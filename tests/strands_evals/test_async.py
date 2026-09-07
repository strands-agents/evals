"""Tests for `strands_evals._async`."""

import asyncio

import pytest

from strands_evals._async import bounded_gather, bounded_gather_fail_fast

# --- bounded_gather (return_exceptions=True) ---


async def test_bounded_gather_returns_results_in_order():
    async def _v(x: int) -> int:
        await asyncio.sleep(0)
        return x

    results = await bounded_gather((_v(i) for i in range(5)), max_workers=2)
    assert results == [0, 1, 2, 3, 4]


async def test_bounded_gather_isolates_failures():
    async def _ok(x: int) -> int:
        return x

    async def _boom() -> int:
        raise RuntimeError("boom")

    results = await bounded_gather([_ok(1), _boom(), _ok(3)], max_workers=2)
    assert results[0] == 1
    assert isinstance(results[1], RuntimeError)
    assert results[2] == 3


async def test_bounded_gather_invalid_max_workers():
    with pytest.raises(ValueError):
        await bounded_gather([], 0)


# --- bounded_gather_fail_fast ---


async def test_bounded_gather_fail_fast_returns_results_in_order():
    async def _v(x: int) -> int:
        await asyncio.sleep(0)
        return x

    results = await bounded_gather_fail_fast((_v(i) for i in range(5)), max_workers=2)
    assert results == [0, 1, 2, 3, 4]


async def test_bounded_gather_fail_fast_empty():
    assert await bounded_gather_fail_fast([], max_workers=3) == []


async def test_bounded_gather_fail_fast_invalid_max_workers():
    with pytest.raises(ValueError):
        await bounded_gather_fail_fast([], 0)


async def test_bounded_gather_fail_fast_cancels_siblings_on_first_error():
    """The whole point of fail-fast: a sibling task that's still mid-flight
    when another raises must be cancelled rather than allowed to complete.
    """
    sibling_completed = False
    sibling_started = asyncio.Event()

    async def _slow_sibling() -> str:
        nonlocal sibling_completed
        sibling_started.set()
        try:
            await asyncio.sleep(10)  # would never finish in a real test
        except asyncio.CancelledError:
            raise
        sibling_completed = True
        return "should-not-happen"

    async def _fail_after_sibling_starts() -> str:
        await sibling_started.wait()
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await bounded_gather_fail_fast(
            [_slow_sibling(), _fail_after_sibling_starts()],
            max_workers=2,
        )

    # Give the cancelled sibling a chance to settle.
    await asyncio.sleep(0)
    assert sibling_completed is False, "fail-fast must cancel the still-running sibling"


async def test_bounded_gather_fail_fast_first_exception_wins():
    """When two tasks fail near-simultaneously, the one observed first is the
    one that surfaces. The contract is `wait(FIRST_EXCEPTION)`, not 'collect
    every exception'."""

    async def _fail(label: str, delay: float) -> str:
        await asyncio.sleep(delay)
        raise RuntimeError(label)

    with pytest.raises(RuntimeError) as exc_info:
        await bounded_gather_fail_fast(
            [_fail("first", 0.0), _fail("second", 0.05)],
            max_workers=2,
        )

    assert str(exc_info.value) == "first"
