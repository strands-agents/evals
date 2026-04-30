from strands_evals.types.simulation import SimulatorResult

from .actor_simulator import ActorSimulator
from .tool_simulator import ToolSimulator

# Alias for backward compatibility
UserSimulator = ActorSimulator

__all__ = [
    "ActorSimulator",
    "UserSimulator",
    "ToolSimulator",
    "SimulatorResult",
]
