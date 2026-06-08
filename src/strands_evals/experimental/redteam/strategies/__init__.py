from .base import AttackRunResult, AttackStrategy
from .crescendo import CrescendoStrategy
from .prompt_strategy import PromptStrategy
from .prompt_strategy.gradual_escalation import get_template as _gradual_escalation_template
from .target_session import AgentTargetSession, CallableTargetSession, TargetCheckpoint, TargetSession

# Ready-made strategy instances users can pass to RedTeamExperiment(attack_strategies=[...]).
# Strategy instances are shared across cases, so each must keep `__init__` for static
# config and clear any runtime state in `reset()`.
BUILTIN_STRATEGIES: dict[str, AttackStrategy] = {
    "gradual_escalation": PromptStrategy("gradual_escalation", _gradual_escalation_template().SYSTEM_PROMPT_TEMPLATE),
}


__all__ = [
    "BUILTIN_STRATEGIES",
    "AgentTargetSession",
    "AttackRunResult",
    "AttackStrategy",
    "CallableTargetSession",
    "CrescendoStrategy",
    "PromptStrategy",
    "TargetCheckpoint",
    "TargetSession",
]
