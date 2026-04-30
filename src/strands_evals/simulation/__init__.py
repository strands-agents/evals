from strands_evals.types.simulation import ActorStructuredResponse

from .actor_simulator import ActorSimulator
from .prompt_templates.actor_system_prompt import (
    DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE,
    STRUCTURED_USER_SIMULATOR_PROMPT_TEMPLATE,
)
from .tool_simulator import ToolSimulator

# Alias for backward compatibility
UserSimulator = ActorSimulator

__all__ = [
    "ActorSimulator",
    "UserSimulator",
    "ToolSimulator",
    "ActorStructuredResponse",
    "DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE",
    "STRUCTURED_USER_SIMULATOR_PROMPT_TEMPLATE",
]
