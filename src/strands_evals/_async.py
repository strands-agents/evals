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
    cancels the task, waits for the cancellation to finish, then propagates. If the function raised another
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

    def execute() -> T:
        return asyncio.run(execute_async())

    def cancel() -> None:
        nonlocal interrupted
        with lock:
            interrupted = True
            if running is not None:
                loop, task = running
                with suppress(RuntimeError):  # the loop is closed: the function already returned
                    loop.call_soon_threadsafe(task.cancel)

    executor = ThreadPoolExecutor(max_workers=1)
    context = contextvars.copy_context()
    future = executor.submit(context.run, execute)
    try:
        _wait(future)
    except BaseException as interrupt:
        try:
            cancel()
            _wait(future)
        except BaseException:
            cancel()  # again, in case the function caught the first cancellation
            raise
        error = future.exception()
        if error is not None and not isinstance(error, asyncio.CancelledError):
            raise interrupt from error
        raise
    finally:
        executor.shutdown(wait=future.done())
    return future.result()


def _wait(future: Future[Any]) -> None:
    """Wait for `future`, waking every `_POLL_SECONDS` to run a pending signal handler: an untimed wait isn't
    interrupted by a Ctrl-C on Windows before Python 3.14, nor by a signal delivered to another thread."""
    while not future.done():
        wait([future], timeout=_POLL_SECONDS)
