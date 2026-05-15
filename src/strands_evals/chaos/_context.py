"""Internal context variable for tracking the active chaos case.

The ChaosPlugin reads from this ContextVar at hook time.
The ChaosExperiment sets and resets it around each case's task invocation.

Using a ContextVar ensures correct behavior under:
- Sequential execution (trivially correct)
- Async execution (each asyncio.Task inherits the var from its parent)
- Threaded execution (each thread gets its own copy)
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .case import ChaosCase

_current_chaos_case: ContextVar["ChaosCase | None"] = ContextVar(
    "chaos_current_case",
    default=None,
)
