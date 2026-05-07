from strands_evals.types.simulation import ActorOutputBase

from .actor_simulator import ActorSimulator
from .prompt_templates.actor_system_prompt import DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE
from .tool_simulator import ToolSimulator

# Alias for backward compatibility
UserSimulator = ActorSimulator

__all__ = [
    "ActorOutputBase",
    "ActorSimulator",
    "UserSimulator",
    "ToolSimulator",
    "DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE",
]
