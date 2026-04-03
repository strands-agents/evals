IMPROVEMENT_SYSTEM_PROMPT = """You are an expert at analyzing AI agent failures and suggesting system prompt improvements.

Given an agent's current system prompt, the user's request, the agent's output, and evaluation failures, suggest a modified system prompt that addresses the failures while preserving the agent's core capabilities.

Focus on:
- Adding specific instructions that address the evaluation failure reasons
- Preserving existing useful instructions from the current system prompt
- Being concise and actionable
- Not changing the fundamental purpose of the agent

Return the complete improved system prompt that should replace the current one."""


def compose_improvement_prompt(
    user_prompt: str,
    actual_output: str,
    failure_reasons: list[str],
    current_system_prompt: str | None,
    expected_output: str | None = None,
) -> str:
    parts = []

    if current_system_prompt:
        parts.append(f"<CurrentSystemPrompt>{current_system_prompt}</CurrentSystemPrompt>")
    else:
        parts.append("<CurrentSystemPrompt>No system prompt set</CurrentSystemPrompt>")

    parts.append(f"<UserRequest>{user_prompt}</UserRequest>")
    parts.append(f"<AgentOutput>{actual_output}</AgentOutput>")

    if expected_output is not None:
        parts.append(f"<ExpectedOutput>{expected_output}</ExpectedOutput>")

    reasons_text = "\n".join(f"- {reason}" for reason in failure_reasons)
    parts.append(f"<EvaluationFailures>\n{reasons_text}\n</EvaluationFailures>")

    return "\n\n".join(parts)
