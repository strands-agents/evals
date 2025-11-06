"""
Goal completion assessment prompt template for actor simulation.

This module contains the prompt template used to evaluate whether a conversation
has successfully achieved the actor's initial goals using a 3-point assessment scale.
"""

from textwrap import dedent

GOAL_COMPLETION_PROMPT = dedent(
    """Please evaluate the following conversation against its intended goals using this
3-point assessment scale:

1 = Does not meet the goal at all
2 = Partially meets the goal with significant gaps
3 = Fully meets the goal

Initial Goal:
{initial_goal}

Conversation to evaluate:
{conversation}

Please provide:
- A score (1-3)
- Brief one line justification"""
)
