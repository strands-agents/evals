"""
Prompt template for actor profile generation.

This module contains the prompt template used to generate realistic actor profiles
from scenario information for conversation simulation.
"""

from textwrap import dedent

ACTOR_PROFILE_PROMPT_TEMPLATE = dedent("""Generate exactly 1 realistic actor profile for the following task:

Actor's Initial Query: {initial_query}
Tasks Description: {task_description}

Generate a complete actor profile with the following structure with:
1. Traits: Key traits (as key-value pairs)
2. Context: Background context (as a paragraph in 2-3 sentences)
3. Actor Goal: What the actor ultimately wants to achieve in this interaction - should be
   specific, actionable, and written from the actor's perspective

IMPORTANT: Return JSON in the following format! IT MUST HAVE THE EXACT STRUCTURE YOU SEE HERE WITH EXACTLY THESE KEYS.

{example}

Be specific and realistic.""")
