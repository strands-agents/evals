"""Rotation: one optimizer per fold of `Split.folds`, run at once under one `max_workers` cap, their test reports
pooled, so every case is tested once.

Sits above `optimizer.py`; imported only by the package `__init__`.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from .._async import run_async as _run_in_thread
from ..types.optimization import PooledTest
from .candidate import Candidate
from .end_phase import pool_tests
from .optimizer import Optimizer, _gather_or_cancel
from .result import OptimizationResult

logger = logging.getLogger(__name__)


class RotationResult(BaseModel):
    """What `rotate` returns; kept in memory only (each fold writes its own run folder and `result.json`).

    Attributes:
        folds: Each fold's result, in the order the optimizers were given.
        test: The folds' test reports pooled over every case; None when they can't be pooled (`test_reason`).
        test_reason: Why `test` is None: the first fold without a test report and that fold's `test_reason`, or
            why `pool_tests` refused the reports.
    """

    model_config = ConfigDict(frozen=True)

    folds: list[OptimizationResult]
    test: PooledTest | None
    test_reason: str | None


def rotate(optimizers: Sequence[Optimizer]) -> RotationResult:
    """Run every fold's optimizer and pool their test reports; the sync form of `rotate_async`.

    Raises:
        ValueError: As `rotate_async`.
    """
    return _run_in_thread(lambda: rotate_async(optimizers))


async def rotate_async(optimizers: Sequence[Optimizer]) -> RotationResult:
    """Run every fold's optimizer at once, sharing one `max_workers` cap, then pool their test reports.

    Give one optimizer per fold of `Split.folds`, as in
    `rotate([Optimizer(experiment, create_agent, split=fold) for fold in split.folds(experiment.cases)])`,
    all with the same `max_workers`, and each with its own `Search`: a proposer keeps what it was told, so one
    shared by two folds would be told one fold's test cases as the other's train cases. Each fold spends its
    own `budget`, and `max_workers` caps the trials of every fold together, so while every fold is scoring,
    each runs about `max_workers / k` trials at once. Optimizers from `Optimizer.resume` may be given too; a
    finished one only rewrites its files.

    When one fold raises, the others are cancelled, every fold's run folder is logged, and the error
    propagates. Trials a cancelled fold had already started run on in `Experiment`'s workers until they
    finish, as when any `Optimizer.run_async` is cancelled. The cancelled folds can be resumed with
    `Optimizer.resume` and passed to `rotate` again; a fold whose setup failed can't be resumed and needs a
    new `Optimizer` on its fold.

    Args:
        optimizers: One per fold; their test cases are disjoint and together are every case.

    Returns:
        Every fold's result, and their test reports pooled (`pool_tests`), with each fold's train and
        validation cases as its train size. When the reports can't be pooled (a fold without a test report,
        or `pool_tests` refusing them), `test` is None and `test_reason` says why.

    Raises:
        ValueError: Before any fold runs: fewer than 2 optimizers; different `max_workers`; folds over
            different cases, or test cases that overlap or leave a case untested; agents with different
            defaults, so different baselines; a proposer in two folds.
        Exception: Whatever a fold raises (for example `RuntimeError` when its setup fails), once the other
            folds are cancelled.
    """
    _check_folds(optimizers)
    limiter = asyncio.Semaphore(optimizers[0]._max_workers)
    for optimizer in optimizers:
        optimizer._limiter = limiter
    try:
        results = await _gather_or_cancel([_run_fold(i, o) for i, o in enumerate(optimizers, 1)])
    except BaseException:
        run_dirs = [None if o._run is None else str(o._run.root) for o in optimizers]
        logger.warning("run_dirs=<%s> | the rotation stopped before pooling", run_dirs)
        raise
    finally:
        for optimizer in optimizers:
            optimizer._limiter = None
    reports = []
    for i, result in enumerate(results, 1):
        if result.test is None:
            return RotationResult(
                folds=results, test=None, test_reason=f"fold {i} has no test report: {result.test_reason}"
            )
        reports.append(result.test)
    sizes = [len(o._case_split.train) + len(o._case_split.validation) for o in optimizers]
    try:
        pooled = pool_tests(reports, train_sizes=sizes)
    except ValueError as error:
        return RotationResult(folds=results, test=None, test_reason=str(error))
    return RotationResult(folds=results, test=pooled, test_reason=None)


async def _run_fold(index: int, optimizer: Optimizer) -> OptimizationResult:
    """`optimizer.run_async()`, logging which fold raised."""
    try:
        return await optimizer.run_async()
    except Exception:
        run_dir = None if optimizer._run is None else optimizer._run.root
        logger.error("fold=<%d>, run_dir=<%s> | fold raised | cancelling the other folds", index, run_dir)
        raise


def _check_folds(optimizers: Sequence[Optimizer]) -> None:
    """Raise `ValueError` unless the optimizers are folds that can run together and pool (see `rotate_async`)."""
    if len(optimizers) < 2:
        raise ValueError(f"rotate needs one optimizer per fold, 2 or more; got {len(optimizers)}")
    workers = sorted({o._max_workers for o in optimizers})
    if len(workers) > 1:
        raise ValueError(
            f"rotate: the folds share one max_workers cap, so give every optimizer the same; got {workers}"
        )
    splits = [o._case_split for o in optimizers]
    cases = [{*split.train, *split.validation, *split.test} for split in splits]
    if any(c != cases[0] for c in cases[1:]):
        raise ValueError("rotate: the optimizers split different cases; build every fold from one experiment")
    tested = Counter(case for split in splits for case in split.test)
    twice = sorted(case for case, count in tested.items() if count > 1)
    never = sorted(cases[0] - set(tested))
    if twice or never:
        raise ValueError(
            f"rotate: every case must be tested by exactly one fold (Split.folds deals them so); tested by more "
            f"than one: {twice}, by none: {never}"
        )
    if len({Candidate.baseline(o._space).id for o in optimizers}) > 1:
        raise ValueError(
            "rotate: the folds' agents have different defaults, so each fold would test against a different "
            "baseline; build every fold from one agent"
        )
    owner: dict[int, int] = {}
    for i, optimizer in enumerate(optimizers, 1):
        for block, _ in optimizer._blocks:
            first = owner.setdefault(id(block.proposer), i)
            if first != i:
                raise ValueError(
                    f"rotate: folds {first} and {i} share one {type(block.proposer).__name__}; a proposer keeps "
                    "what it was told, so give each fold its own Search"
                )
