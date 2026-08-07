"""Async execution utilities.

Provides helpers to run async functions from sync contexts and to fan out
awaitables under a concurrency cap.
"""

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


def run_async(async_func: Callable[[], Awaitable[T]]) -> T:
    """Run an async function in a separate thread to avoid event loop conflicts.

    This utility handles the common pattern of running async code from sync contexts
    by using ThreadPoolExecutor to isolate the async execution.

    Args:
        async_func: A callable that returns an awaitable.

    Returns:
        The result of the async function.
    """

    async def execute_async() -> T:
        return await async_func()

    def execute() -> T:
        return asyncio.run(execute_async())

    with ThreadPoolExecutor() as executor:
        context = contextvars.copy_context()
        future = executor.submit(context.run, execute)
        return future.result()


async def bounded_gather(
    coros: Iterable[Awaitable[T]],
    max_workers: int,
    *,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    """Run awaitables concurrently with at most `max_workers` running at once.

    Wraps each coroutine in a `Semaphore`-gated task and `asyncio.gather`s them.
    `return_exceptions` defaults to `True` so a single failing task does not
    cancel its siblings — callers filter the result.

    Args:
        coros: An iterable of awaitables to run.
        max_workers: Maximum concurrent tasks. Must be `>= 1`.
        return_exceptions: Forwarded to `asyncio.gather`.

    Returns:
        The list of results in the order the awaitables were supplied. When
        `return_exceptions=True`, failed tasks appear as `BaseException`
        instances rather than raising.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    semaphore = asyncio.Semaphore(max_workers)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_bounded(c) for c in coros), return_exceptions=return_exceptions)


async def bounded_gather_fail_fast(
    coros: Iterable[Awaitable[T]],
    max_workers: int,
) -> list[T]:
    """Run awaitables concurrently with at most `max_workers` running at once,
    cancelling siblings on the first exception.

    Wraps each coroutine in a `Semaphore`-gated task and waits with
    `FIRST_EXCEPTION`. As soon as one task raises, every still-running
    sibling is cancelled and drained, then the original exception is
    re-raised. Use when sibling work has nontrivial cost (e.g. LLM calls)
    and one failure means the rest aren't worth completing. Prefer
    `bounded_gather` when you want sibling isolation (one bad item must not
    kill the rest of the batch).

    Args:
        coros: Iterable of awaitables to run.
        max_workers: Maximum concurrent tasks. Must be `>= 1`.

    Returns:
        Results in the order the awaitables were supplied.

    Raises:
        Re-raises the first exception observed; pending tasks are cancelled
        and drained before the raise propagates.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    semaphore = asyncio.Semaphore(max_workers)

    async def _bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    tasks: list[asyncio.Task[T]] = [asyncio.create_task(_bounded(c)) for c in coros]
    if not tasks:
        return []

    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except BaseException:
        # External cancellation: roll back our spawned tasks before propagating.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    # A task raised: cancel still-running siblings, drain, then re-raise.
    pending = [t for t in tasks if not t.done()]
    if pending:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for t in tasks:
            if t.cancelled():
                continue
            exc = t.exception()
            if exc is not None:
                raise exc

    return [t.result() for t in tasks]
