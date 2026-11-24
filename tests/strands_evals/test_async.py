import asyncio
import contextvars
import threading
import time

import pytest

from strands_evals import _async
from strands_evals._async import run_async

_CALLER = contextvars.ContextVar("caller", default=None)


def test_the_function_runs_in_a_copy_of_the_callers_context_and_its_result_is_returned():
    async def main():
        return _CALLER.get()

    token = _CALLER.set("set by the caller")
    try:
        assert run_async(main) == "set by the caller"
    finally:
        _CALLER.reset(token)


async def test_it_runs_inside_a_running_loop_and_from_a_worker_thread():
    async def main():
        return "done"

    assert run_async(main) == "done"
    assert await asyncio.to_thread(run_async, main) == "done"


def test_an_error_the_function_raises_reaches_the_caller():
    async def main():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_async(main)


def test_an_interrupt_cancels_the_function_and_propagates_once_the_cancellation_finished(sigint):
    cleaned_up = []

    async def main():
        await asyncio.sleep(0.1)  # the caller is waiting
        sigint.send()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.2)  # cleanup that takes a while
            cleaned_up.append(True)
            raise

    start = time.monotonic()
    with pytest.raises(sigint.error) as raised:
        run_async(main)
    assert time.monotonic() - start < 5 and cleaned_up == [True]
    assert raised.value.__cause__ is None


def test_an_interrupt_while_the_worker_thread_starts_still_cancels_the_running_function(sigint, monkeypatch):
    """The interrupt lands in `submit` after the worker began running the function (a starved main thread)."""
    started, cleaned_up = threading.Event(), []

    class InterruptedWhileStarting(_async.ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            super().submit(*args, **kwargs)
            assert started.wait(5)
            raise sigint.error

    async def main():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cleaned_up.append(True)
            raise

    monkeypatch.setattr(_async, "ThreadPoolExecutor", InterruptedWhileStarting)
    start = time.monotonic()
    with pytest.raises(sigint.error):
        run_async(main)
    assert time.monotonic() - start < 5 and cleaned_up == [True]


def test_an_interrupt_before_the_work_is_queued_propagates_at_once(sigint, monkeypatch):
    ran, raised = [], []

    class InterruptedBeforeQueuing(_async.ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            raise sigint.error

    async def main():
        ran.append(True)

    def call():
        try:
            run_async(main)
        except sigint.error:
            raised.append(True)

    monkeypatch.setattr(_async, "ThreadPoolExecutor", InterruptedBeforeQueuing)
    caller = threading.Thread(target=call, daemon=True)  # a daemon: if `run_async` hangs, the session still exits
    caller.start()
    caller.join(5)
    assert not caller.is_alive(), "run_async waited for work that was never queued"
    assert raised == [True] and ran == []


def test_an_error_raised_instead_of_the_cancellation_is_the_interrupts_cause(sigint):
    async def main():
        await asyncio.sleep(0.1)  # the caller is waiting
        sigint.send()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise RuntimeError("cleanup failed") from None

    with pytest.raises(sigint.error) as raised:
        run_async(main)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_second_interrupt_cancels_again_and_stops_waiting(sigint):
    release, cancelled, second_handled = threading.Event(), [], threading.Event()

    async def main():
        await asyncio.sleep(0.1)  # the caller is waiting
        sigint.send()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append("first")  # caught: the function goes on
        sigint.send()  # a second Ctrl-C while the caller waits for the cancellation
        try:
            await asyncio.to_thread(release.wait, 10)  # `asyncio.run` waits for this thread before it returns
        except asyncio.CancelledError:
            cancelled.append("second")
            second_handled.set()
            raise

    threads = set(threading.enumerate())
    start = time.monotonic()
    with pytest.raises(sigint.error):
        run_async(main)
    elapsed = time.monotonic() - start
    # The second cancellation is delivered in the background, after `run_async` raised: wait for it while the
    # function is still blocked. Released first, the function was seen to finish before the cancellation ran.
    assert second_handled.wait(5)
    release.set()
    # Joining the leftover threads can't be relied on (3.14 on Windows can enumerate a thread `join` rejects as
    # never started), hence the `is_alive` check.
    for thread in set(threading.enumerate()) - threads:
        if thread.is_alive():
            thread.join(5)
    assert elapsed < 5 and cancelled == ["first", "second"]
