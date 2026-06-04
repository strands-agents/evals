from .base import AttackStrategy
from .crescendo import CrescendoStrategy
from .prompt_strategy import PromptStrategy
from .prompt_strategy.gradual_escalation import get_template as _gradual_escalation_template

# Strategies registered here are shared across cases. Implementers must keep
# `__init__` for static config and clear runtime state in `reset()`.
BUILTIN_STRATEGIES: dict[str, AttackStrategy] = {
    "gradual_escalation": PromptStrategy("gradual_escalation", _gradual_escalation_template().SYSTEM_PROMPT_TEMPLATE),
}

DEFAULT_STRATEGY = "gradual_escalation"


def resolve_strategy(strategy: AttackStrategy | str) -> AttackStrategy:
    """Resolve a strategy name or instance to an AttackStrategy."""
    if isinstance(strategy, str):
        if strategy not in BUILTIN_STRATEGIES:
            raise ValueError(f"Unknown strategy: '{strategy}'. Available: {list(BUILTIN_STRATEGIES)}")
        return BUILTIN_STRATEGIES[strategy]
    return strategy


__all__ = [
    "BUILTIN_STRATEGIES",
    "DEFAULT_STRATEGY",
    "AttackStrategy",
    "CrescendoStrategy",
    "PromptStrategy",
    "resolve_strategy",
]
