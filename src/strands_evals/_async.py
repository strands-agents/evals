"""Async execution utilities for detectors.

Provides a helper to run async functions from sync contexts, used by
detectors that call Model.stream() directly.
"""

import asyncio
import contextvars
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import suppress
from typing import Any, TypeVar

T = TypeVar("T")

_POLL_SECONDS = 0.1


def run_async(async_func: Callable[[], Awaitable[T]]) -> T:
    """Run an async function in a separate thread to avoid event loop conflicts.

    This utility handles the common pattern of running async code from sync contexts
    by using ThreadPoolExecutor to isolate the async execution.

    An exception raised in the calling thread while it waits, such as the `KeyboardInterrupt` of a Ctrl-C,
    cancels the task, waits for the cancellation to finish, then propagates; one that arrives before the function
    started keeps it from starting and propagates at once. If the function raised another
    error instead of the cancellation, that error is the propagated exception's `__cause__`. A second exception
    during that wait cancels the task again and propagates at once, leaving the thread to finish in the
    background (the interpreter still waits for it at exit).

    Args:
        async_func: A callable that returns an awaitable.

    Returns:
        The result of the async function.
    """
    lock = threading.Lock()
    running: tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]] | None = None
    interrupted = False

    async def execute_async() -> T:
        nonlocal running
        task = asyncio.current_task()
        assert task is not None
        with lock:
            if interrupted:
                raise asyncio.CancelledError
            running = asyncio.get_running_loop(), task
        return await async_func()

    outcome: Future[T] = Future()

    def execute() -> None:
        try:
            outcome.set_result(asyncio.run(execute_async()))
        except BaseException as error:
            outcome.set_exception(error)

    def cancel() -> bool:
        """Cancel the task, or keep it from starting; returns whether it had started."""
        nonlocal interrupted
        with lock:
            interrupted = True
            if running is None:
                return False
            loop, task = running
            with suppress(RuntimeError):  # the loop is closed: the function already returned
                loop.call_soon_threadsafe(task.cancel)
            return True

    executor = ThreadPoolExecutor(max_workers=1)
    context = contextvars.copy_context()
    try:
        # Inside the `try`: an interrupt can land while `submit` starts the worker thread, which may already be
        # running the function by then.
        executor.submit(context.run, execute)
        _wait(outcome)
    except BaseException as interrupt:
        try:
            if not cancel():
                raise  # nothing started; if the work was queued, it sees `interrupted` and stops at once
            _wait(outcome)
        except BaseException:
            cancel()  # again, in case the function caught the first cancellation
            raise
        error = outcome.exception()
        if error is not None and not isinstance(error, asyncio.CancelledError):
            raise interrupt from error
        raise
    finally:
        executor.shutdown(wait=outcome.done())
    return outcome.result()


def _wait(future: Future[Any]) -> None:
    """Wait for `future`, waking every `_POLL_SECONDS` to run a pending signal handler: an untimed wait isn't
    interrupted by a Ctrl-C on Windows before Python 3.14, nor by a signal delivered to another thread."""
    while not future.done():
        wait([future], timeout=_POLL_SECONDS)
