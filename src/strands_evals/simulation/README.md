# Actor Simulator

A framework for simulating realistic multi-turn conversations with AI-powered actors for agent evaluation.

## Overview

ActorSimulator creates realistic actor personas that interact with agents in multi-turn conversations. It automatically generates actor profiles from test cases, maintains conversation context, and produces contextually appropriate responses aligned with the actor's goals and traits.

## Quick Start

```python
from strands import Agent
from strands_evals import ActorSimulator, Case

# Create agent under test
agent = Agent(system_prompt="You are a helpful travel assistant.", callback_handler=None)

# Create test case
case = Case(
    input="I want to plan a trip to Tokyo with hotel and activities",
    metadata={"task_description": "Complete travel package arranged"}
)

# Create user simulator with max_turns
user_sim = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=5)

# Run conversation
user_message = case.input
while user_sim.has_next():
    agent_response = agent(user_message)
    user_result = user_sim.act(str(agent_response))
    user_message = str(user_result.structured_output.message)
```

## How It Works

1. **Profile Generation**: Creates a realistic actor profile with traits, context, and goals from the test case
2. **Conversation Initialization**: Sets up conversation with a greeting and the actor's initial query
3. **Contextual Responses**: Generates responses that maintain consistency with the actor's profile and goals
4. **Goal Tracking**: Built-in tool allows actors to assess progress toward their goals

## API Reference

### ActorSimulator

Main class for simulating actor behavior in conversations.

#### Factory Method (Recommended)

```python
ActorSimulator.from_case_for_user_simulator(
    case: Case,
    system_prompt_template: str | None = None,
    tools: list | None = None,
    model: str | None = None,
    max_turns: int = 10
) -> ActorSimulator
```

Creates an ActorSimulator configured as a user simulator from a test case. Automatically generates a realistic actor profile from `case.input` and optionally `case.metadata["task_description"]`.

**Parameters:**
- `case`: Test case with input (initial query) and optional task_description in metadata
- `system_prompt_template`: Custom system prompt template (uses default if None)
- `tools`: Additional tools for the actor (defaults to goal completion tool only)
- `model`: Model identifier (uses Strands default if None)
- `max_turns`: Maximum number of conversation turns (default: 10)

**Example:**
```python
case = Case(
    input="I need help booking a flight to Paris",
    metadata={"task_description": "Book round-trip flight under $800"}
)

user_sim = ActorSimulator.from_case_for_user_simulator(
    case=case,
    max_turns=5
)
```

#### Direct Initialization

```python
ActorSimulator(
    actor_profile: ActorProfile,
    initial_query: str,
    system_prompt_template: str,
    tools: list | None = None,
    model: str | None = None,
    max_turns: int = 10
)
```

Initialize with an existing actor profile. Use this when you have a pre-defined profile instead of generating one from a test case.

**Parameters:**
- `actor_profile`: ActorProfile object with traits, context, and actor_goal
- `initial_query`: The actor's first query or message
- `system_prompt_template`: Template string for actor behavior (formatted with profile)
- `tools`: Additional tools for the actor
- `model`: Model identifier
- `max_turns`: Maximum number of conversation turns (default: 10)

#### Methods

**`act(agent_message: str) -> AgentResult`**

Generate the actor's next message in response to the agent's message.

**Parameters:**
- `agent_message`: The agent's response to react to

**Returns:**
- `AgentResult` containing the actor's structured response with reasoning and message

**Example:**
```python
agent_response = agent("I can help you book that flight")
user_result = user_sim.act(str(agent_response))
user_message = str(user_result.structured_output.message)
```

**`has_next() -> bool`**

Check if the conversation should continue. Returns False if the stop token (`<stop/>`) is present in the last message or if the maximum number of turns has been reached.

**Returns:**
- `True` if the conversation should continue, `False` otherwise

**Example:**
```python
while user_sim.has_next():
    agent_response = agent(user_message)
    user_result = user_sim.act(str(agent_response))
    user_message = str(user_result.structured_output.message)
```

### Data Models

**ActorProfile:**
```python
class ActorProfile(BaseModel):
    traits: dict[str, Any]  # Actor characteristics and personality
    context: str  # Background information and situation
    actor_goal: str  # What the actor wants to achieve
```

**ActorResponse:**
```python
class ActorResponse(BaseModel):
    reasoning: str  # Actor's internal reasoning process
    message: str  # The actual message to send
```

## Usage Examples

### Complete Multi-Turn Conversation Example

```python
from strands import Agent
from strands_evals import ActorSimulator, Case

# Create agent under test
agent = Agent(system_prompt="You are a helpful travel assistant.", callback_handler=None)

# Create test case
case = Case(
    input="I want to plan a trip to Tokyo with hotel and activities",
    metadata={"task_description": "Complete travel package arranged"}
)

# Create user simulator
user_sim = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=5)

# Run conversation
conversation = []
user_message = case.input

while user_sim.has_next():
    # Agent responds
    agent_response = agent(user_message)
    agent_message = str(agent_response)
    conversation.append({"role": "assistant", "content": agent_message})
    
    # User responds
    user_result = user_sim.act(agent_message)
    user_message = str(user_result.structured_output.message)
    conversation.append({"role": "user", "content": user_message})

print(f"Conversation completed in {len(conversation) // 2} turns")
```

### Custom Actor Profile

```python
from strands_evals.types.simulation import ActorProfile
from strands_evals.simulation.prompt_templates.actor_system_prompt import (
    DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE
)

# Create custom actor profile
actor_profile = ActorProfile(
    traits={
        "personality": "analytical and detail-oriented",
        "communication_style": "direct and concise",
        "technical_level": "expert"
    },
    context="Experienced business traveler with elite status",
    actor_goal="Book business class flight with specific seat preferences"
)

# Initialize with custom profile
user_sim = ActorSimulator(
    actor_profile=actor_profile,
    initial_query="I need to book a business class flight to London",
    system_prompt_template=DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE,
    max_turns=15
)
```

## Tools

### Built-in Goal Completion Tool

ActorSimulator automatically includes a goal completion assessment tool that actors can use to evaluate their progress:

```python
from strands_evals.simulation.tools.goal_completion import (
    get_conversation_goal_completion
)

# The actor can call this tool during conversation to assess progress
assessment = get_conversation_goal_completion(
    initial_goal="Book a flight to Tokyo",
    conversation=[
        {"role": "user", "content": "I need a flight to Tokyo"},
        {"role": "assistant", "content": "I can help with that..."}
    ]
)
# Returns assessment with score and reasoning
```

### Adding Custom Tools

Extend actor capabilities with custom tools:

```python
from strands import tool

@tool
def check_booking_status(booking_id: str) -> str:
    """Check the status of a booking."""
    return f"Booking {booking_id} is confirmed"

# Add custom tools to the simulator
user_sim = ActorSimulator.from_case_for_user_simulator(
    case=case,
    tools=[check_booking_status]
)
```

## Advanced Configuration

### Custom System Prompt Templates

Customize actor behavior with a custom system prompt template. The template receives the actor profile as a format parameter:

```python
custom_prompt_template = """
You are simulating a user with the following profile:
{actor_profile}

Behavior guidelines:
- Be persistent but professional
- Express concerns clearly
- Stay focused on your goal

Respond naturally based on your profile and the conversation context.
"""

user_sim = ActorSimulator.from_case_for_user_simulator(
    case=case,
    system_prompt_template=custom_prompt_template
)
```

### Conversation Initialization

ActorSimulator automatically initializes conversations with a random greeting from a predefined set:

```python
# Built-in greetings:
# - "hi! how can I help you today?"
# - "hello! what can I assist you with?"
# - "hi there! how may I help you?"
# - "good day! what can I do for you?"
# - "hello! what would you like to know?"

# The conversation starts with:
# 1. Random greeting (as user message)
# 2. Actor's initial query (as assistant message)
```

### Model Selection

Specify a custom model for the actor simulator:

```python
user_sim = ActorSimulator.from_case_for_user_simulator(
    case=case,
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    max_turns=10
)
```

## Best Practices

1. **Include Task Description**: Add `task_description` in case metadata for better goal generation
2. **Set max_turns**: Configure `max_turns` during initialization to prevent infinite conversations
3. **Use has_next()**: Always use `has_next()` in your conversation loop to respect turn limits and stop tokens
4. **Track Conversation**: Append messages to a conversation list for evaluation and debugging
5. **Access Structured Output**: Use `result.structured_output.message` to get the actor's message and `result.structured_output.reasoning` to see internal reasoning