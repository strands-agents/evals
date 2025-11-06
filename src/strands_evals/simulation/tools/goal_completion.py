import logging

from strands import Agent, tool
from typing_extensions import Any

from strands_evals.simulation.prompt_templates.goal_completion import GOAL_COMPLETION_PROMPT

logger = logging.getLogger(__name__)


@tool
def get_conversation_goal_completion(initial_goal: str, conversation: list[dict[str, str]]) -> str:
    """
    Evaluate conversation goal completion using a 3-point assessment scale.

    Analyzes the conversation against the actor's initial goal and provides a score
    with justification.

    Args:
        initial_goal: The actor's original goal or objective.
        conversation: List of conversation turns, each with 'role' and 'content' keys.

    Returns:
        Assessment string with score (1-3) and brief justification.

    Raises:
        ValueError: If the conversation format is invalid.
    """
    # Format conversation for the prompt
    conversation_text = _format_conversation_for_assessment(conversation)

    # Create the assessment prompt
    prompt = GOAL_COMPLETION_PROMPT.format(initial_goal=initial_goal, conversation=conversation_text)

    goal_completion_agent = Agent(callback_handler=None)
    response = goal_completion_agent(prompt)
    logger.info("Successfully completed goal completion assessment")
    return str(response)


def _format_conversation_for_assessment(conversation: list[dict[str, Any]]) -> str:
    """
    Format conversation history for goal completion assessment.

    Args:
        conversation: List of conversation turns with 'role' and 'content' keys.
            Content can be either a string or a list of content blocks.

    Returns:
        Formatted conversation string with each turn on a separate line.

    Raises:
        ValueError: If conversation format is invalid.
    """
    try:
        formatted_turns = []

        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                raise ValueError(f"Conversation turn {i} must be a dictionary")

            role = turn.get("role", "").strip()
            content_raw = turn.get("content", "")

            # Handle both string format and list of content blocks
            if isinstance(content_raw, str):
                content = content_raw.strip()
            elif isinstance(content_raw, list):
                content_parts = []
                for block in content_raw:
                    if isinstance(block, dict) and "text" in block:
                        content_parts.append(block["text"])
                content = " ".join(content_parts).strip()
            else:
                logger.warning(f"Skipping conversation turn {i} with invalid content type: {type(content_raw)}")
                continue

            if not role or not content:
                logger.warning(f"Skipping conversation turn {i} with missing role or content")
                continue

            formatted_turn = f"{role.upper()}: {content}"
            formatted_turns.append(formatted_turn)

        if not formatted_turns:
            raise ValueError("No valid conversation turns found")

        return "\n\n".join(formatted_turns)

    except ValueError:
        raise
    except Exception as e:
        raise ValueError("Error formatting conversation") from e
