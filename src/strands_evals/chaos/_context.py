"""Internal context variable for tracking the active chaos scenario.

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
    from .scenario import ChaosScenario

_current_scenario: ContextVar["ChaosScenario | None"] = ContextVar(
    "chaos_current_scenario",
    default=None,
)
