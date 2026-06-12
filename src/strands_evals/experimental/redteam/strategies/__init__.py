from .bad_likert_judge import BadLikertJudgeStrategy
from .base import AttackRunResult, AttackStrategy
from .crescendo import CrescendoStrategy
from .prompt_strategy import PromptStrategy
from .prompt_strategy.gradual_escalation import get_template as _gradual_escalation_template
from .target_session import StrandsAgentSession, TargetCheckpoint, TargetSession, ToolUseEntry

# Ready-made strategy instances users can pass to RedTeamExperiment(attack_strategies=[...]).
# Strategy instances are shared across cases, so each must keep `__init__` for static
# config and clear any runtime state in `reset()`.
BUILTIN_STRATEGIES: dict[str, AttackStrategy] = {
    "gradual_escalation": PromptStrategy("gradual_escalation", _gradual_escalation_template().SYSTEM_PROMPT_TEMPLATE),
}


__all__ = [
    "BUILTIN_STRATEGIES",
    "AttackRunResult",
    "AttackStrategy",
    "BadLikertJudgeStrategy",
    "CrescendoStrategy",
    "PromptStrategy",
    "StrandsAgentSession",
    "TargetCheckpoint",
    "TargetSession",
    "ToolUseEntry",
]
