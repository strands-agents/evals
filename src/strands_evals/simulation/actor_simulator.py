import logging
import random

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.types.content import Message
from typing_extensions import cast

from strands_evals.case import Case
from strands_evals.simulation.profiles.actor_profile import DEFAULT_USER_PROFILE_SCHEMA
from strands_evals.simulation.prompt_templates.actor_profile_extraction import ACTOR_PROFILE_PROMPT_TEMPLATE
from strands_evals.simulation.prompt_templates.actor_system_prompt import DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE
from strands_evals.simulation.tools.goal_completion import get_conversation_goal_completion
from strands_evals.types.simulation import ActorProfile, ActorResponse

logger = logging.getLogger(__name__)


class ActorSimulator:
    """
    Simulates an actor in multi-turn conversations for agent evaluation.

    ActorSimulator wraps a Strands Agent configured to behave as a specific actor
    (typically a user) in conversation scenarios. It maintains conversation history,
    generates contextually appropriate responses, and can assess goal completion.

    Attributes:
        agent: The underlying Strands Agent configured with actor behavior.
        actor_profile: The actor's profile containing traits, context, and goal.
        initial_query: The actor's first query in the conversation.
        conversation_history: List of conversation messages in Strands format.
        model_id: Model identifier for the underlying agent.
    """

    INITIAL_GREETINGS = [
        "hi! how can I help you today?",
        "hello! what can I assist you with?",
        "hi there! how may I help you?",
        "good day! what can I do for you?",
        "hello! what would you like to know?",
    ]

    @classmethod
    def from_case_for_user_simulator(
        cls,
        case: Case,
        system_prompt_template: str | None = None,
        tools: list | None = None,
        model: str | None = None,
        max_turns: int = 10,
    ) -> "ActorSimulator":
        """
        Create an ActorSimulator configured as a user simulator from a test case.

        Generates a realistic user profile and goal from case.input and optionally
        case.metadata["task_description"], then configures the simulator with
        user-specific defaults. If you already have a profile, use __init__() directly.

        Args:
            case: Test case containing input (initial query) and optional metadata with "task_description".
            system_prompt_template: Custom system prompt template. Uses DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE if None.
            tools: Additional tools available to the user. Defaults to goal completion tool only.
            model: Model identifier for the underlying agent. Uses Strands default if None.
            max_turns: Maximum number of conversation turns before stopping (default: 10).

        Returns:
            ActorSimulator configured for user simulation.

        Example:
            ```python
            from strands_evals import Case, ActorSimulator
            from strands import Agent

            # Create test case
            case = Case(
                input="I need to book a flight to Paris",
                metadata={"task_description": "Flight booking confirmed"}
            )

            # Create user simulator
            user_sim = ActorSimulator.from_case_for_user_simulator(
                case=case,
                max_turns=5
            )

            # Create target agent to evaluate
            agent = Agent(system_prompt="You are a travel assistant.")

            # Run conversation
            user_message = case.input
            while user_sim.has_next():
                agent_response = agent(user_message)
                user_result = user_sim.act(str(agent_response))
                user_message = str(user_result.structured_output.message)
            ```
        """
        actor_profile = cls._generate_profile_from_case(case)

        if system_prompt_template is None:
            system_prompt_template = DEFAULT_USER_SIMULATOR_PROMPT_TEMPLATE

        return cls(
            actor_profile=actor_profile,
            initial_query=case.input,
            system_prompt_template=system_prompt_template,
            tools=tools,
            model=model,
            max_turns=max_turns,
        )

    @staticmethod
    def _generate_profile_from_case(case: Case) -> ActorProfile:
        """
        Generate user profile from case.

        Private helper for from_case_for_user_simulator factory method.
        Uses case.input and optionally case.metadata["task_description"] if present.

        Args:
            case: Test case with input and optional task_description in metadata.

        Returns:
            ActorProfile with generated traits, context, and goal.
        """
        initial_query = case.input
        task_description = case.metadata.get("task_description", "") if case.metadata else ""

        profile_prompt = ACTOR_PROFILE_PROMPT_TEMPLATE.format(
            initial_query=initial_query,
            task_description=task_description,
            example=DEFAULT_USER_PROFILE_SCHEMA,
        )
        profile_agent = Agent(callback_handler=None)
        result = profile_agent(profile_prompt, structured_output_model=ActorProfile)
        return result.structured_output

    def __init__(
        self,
        actor_profile: ActorProfile,
        initial_query: str,
        system_prompt_template: str,
        tools: list | None = None,
        model: str | None = None,
        max_turns: int = 10,
    ):
        """
        Initialize an ActorSimulator with profile and goal.

        Use this constructor when you have a pre-defined ActorProfile. For automatic
        profile generation from test cases, use from_case_for_user_simulator() instead.

        Args:
            actor_profile: ActorProfile object containing traits, context, and actor_goal.
            initial_query: The actor's first query or message.
            system_prompt_template: Template string for system prompt. Must include {actor_profile} placeholder.
            tools: Additional tools available to the actor. Defaults to goal completion tool only.
            model: Model identifier for the underlying agent. Uses Strands default if None.
            max_turns: Maximum number of conversation turns before stopping (default: 10).

        Example:
            ```python
            from strands_evals.simulation import ActorSimulator
            from strands_evals.types.simulation import ActorProfile

            # Define custom actor profile
            profile = ActorProfile(
                traits={
                    "expertise_level": "expert",
                    "communication_style": "technical"
                },
                context="A software engineer debugging a production issue.",
                actor_goal="Identify and resolve the memory leak."
            )

            # Create simulator with custom profile
            simulator = ActorSimulator(
                actor_profile=profile,
                initial_query="Our service is experiencing high memory usage.",
                system_prompt_template="You are simulating: {actor_profile}",
                max_turns=15
            )
            ```
        """
        self.actor_profile = actor_profile
        self.initial_query = initial_query
        self.conversation_history: list[Message] = []
        self.model_id = model
        self._turn_count = 0
        self._last_message = ""
        self._max_turns = max_turns

        system_prompt = system_prompt_template.format(actor_profile=actor_profile.model_dump())

        # Combine tools
        all_tools = [get_conversation_goal_completion]
        if tools:
            all_tools.extend(tools)

        self._initialize_conversation()

        # Create agent
        self.agent = Agent(
            system_prompt=system_prompt,
            messages=self.conversation_history,
            tools=all_tools,
            model=self.model_id,
            callback_handler=None,
        )

    def _initialize_conversation(self):
        """
        Initialize the conversation history with a greeting and initial query.

        Sets up the conversation with a random greeting from the assistant followed
        by the actor's initial query. This establishes the conversation context.

        Note: This is a private method called during initialization.
        """
        selected_greeting = random.choice(self.INITIAL_GREETINGS)
        greeting_message = {"role": "user", "content": [{"text": selected_greeting}]}
        self.conversation_history.append(greeting_message)

        initial_query_message = {"role": "assistant", "content": [{"text": self.initial_query.strip()}]}
        self.conversation_history.append(initial_query_message)

    def act(self, agent_message: str) -> AgentResult:
        """
        Generate the next actor message in the conversation.

        Processes the agent's message and generates a contextually appropriate
        response from the actor's perspective, maintaining consistency with the actor's
        profile and goal. The response includes reasoning about the actor's thought
        process and the actual message to send.

        Args:
            agent_message: The agent's response to react to (required).

        Returns:
            AgentResult containing the actor's structured response with:
                - structured_output.reasoning: Actor's internal reasoning
                - structured_output.message: Actor's response message

        Example:
            ```python
            # Agent responds to user
            agent_response = agent("I need help booking a flight")

            # User simulator generates next message
            user_result = user_sim.act(str(agent_response))

            # Access the response
            print(user_result.structured_output.reasoning)  # Why the actor responded this way
            print(user_result.structured_output.message)    # The actual message

            # Continue conversation
            next_message = str(user_result.structured_output.message)
            ```
        """
        response = self.agent(agent_message.strip(), structured_output_model=ActorResponse)
        self._turn_count += 1
        self._last_message = str(cast(ActorResponse, response.structured_output).message)
        return response

    def has_next(self) -> bool:
        """
        Check if the conversation should continue.

        Returns False if the stop token (<stop/>) is present in the last message or if
        the maximum number of turns has been reached. Use this in a loop to control
        multi-turn conversations.

        Returns:
            True if the conversation should continue, False otherwise.

        Example:
            ```python
            user_message = case.input

            # Continue conversation until completion
            while user_sim.has_next():
                agent_response = agent(user_message)
                user_result = user_sim.act(str(agent_response))
                user_message = str(user_result.structured_output.message)

            # Conversation ended either by:
            # - Actor including <stop/> token in message
            # - Reaching max_turns limit
            ```
        """
        if self._turn_count >= self._max_turns:
            return False
        return "<stop/>" not in self._last_message
